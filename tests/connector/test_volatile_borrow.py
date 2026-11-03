"""Pinned ownership control checks using CPU bytes and a mocked H2D consumer."""
import hashlib
import os
import threading
import unittest
from unittest.mock import Mock

from test_failure_policy import connector as make_connector, import_connector
from kvsnap.connector.volatile import VolatileOwner
from kvsnap.models import IdentityMismatch
from kvsnap.store.crc32c import crc32c


class CpuOwnedAllocation:
    def __init__(self, data):
        self.data, self.size, self.closed = data, len(data), False

    def read(self):
        if self.closed:
            raise RuntimeError("CPU test allocation was freed")
        return self.data

    def close(self):
        self.closed = True


class VolatileBorrow(unittest.TestCase):
    def setUp(self):
        self.module = import_connector()
        self.allocation = CpuOwnedAllocation(b"cpu-fixture-kv")
        key = hashlib.sha256(self.allocation.data).hexdigest()
        owner = VolatileOwner.__new__(VolatileOwner)
        owner._pid, owner._stop, owner._lock = os.getpid(), threading.Event(), threading.RLock()
        owner.descriptor = {"owner_id": "owner"}
        owner._entries = {key: {"handle": "handle", "allocation": self.allocation}}
        segment = {"blob_id": key, "length": self.allocation.size, "offset": 0,
                   "first_block": 0, "nblocks": 1, "crc32c": crc32c(self.allocation.data)}
        transfer = self.module.Transfer("request", [7], {"snapshot_id": "source", "durability": "volatile",
                                             "segments": [segment]}, lease_id="load-lease",
                                  volatile_locations={key: {"owner": owner.descriptor, "handle": "handle"}})
        worker = make_connector(self.module, async_load=False)
        worker.volatile = owner
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[transfer])
        self.owner, self.worker, self.transfer = owner, worker, transfer

    def test_borrow_holds_gc_lock_until_h2d_finishes(self):
        def h2d(ids, sources, **kwargs):
            self.assertEqual(ids, [7])
            self.assertEqual(sources, [(self.allocation, 0)])
            attempts = []

            def gc_try_lock():
                held = self.owner._lock.acquire(blocking=False)
                attempts.append(held)
                if held:
                    self.owner._lock.release()

            thread = threading.Thread(target=gc_try_lock)
            thread.start()
            thread.join(timeout=2)
            self.assertEqual(attempts, [False])

        self.worker.dataplane.load_scatter.side_effect = h2d
        self.worker.start_load_kv(None)
        self.assertEqual(self.worker.worker_meta.loaded_leases, ["load-lease"])
        self.assertFalse(self.allocation.closed)
        self.worker.dataplane.staging.assert_not_called()   # borrowed in place, no staging copy

    def test_unknown_h2d_fence_retains_load_lease_and_never_marks_source_corrupt(self):
        self.worker.dataplane.load_scatter.side_effect = RuntimeError("simulated unknown H2D completion")
        with self.assertRaisesRegex(RuntimeError, "H2D completion could not be proven"):
            self.worker.start_load_kv(None)
        self.assertEqual(self.worker.worker_meta.loaded_leases, [])
        self.assertEqual(self.worker.worker_meta.invalid_snapshots, [])
        self.assertEqual(self.worker.load_errors, set())
        self.assertFalse(self.allocation.closed)

    def test_incompatible_destination_layout_cannot_invalidate_healthy_source(self):
        self.worker._check_layout.side_effect = IdentityMismatch("destination uses another physical KV layout")
        self.worker.start_load_kv(None)
        self.assertEqual(self.worker.worker_meta.invalid_snapshots, [])
        self.worker.dataplane.load_scatter.assert_not_called()
        self.assertFalse(self.allocation.closed)


if __name__ == "__main__":
    unittest.main()
