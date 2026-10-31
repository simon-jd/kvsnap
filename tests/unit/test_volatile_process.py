"""Real worker processes and AF_UNIX IPC with explicitly CPU fixture allocations.

These tests prove ownership protocol only: no CUDA allocation or DMA occurs.
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvsnap.models import KVIdentity, SamplingConfig, SnapshotManifest
from kvsnap.store import CorruptSnapshotError, GenerationConflict, SnapshotStore, StoreError


def manifest(snapshot_id="snapshot", session_id="session", generation=0, durability="volatile"):
    identity = KVIdentity("cpu-volatile-fixture", "revision", "tokenizer", "bf16", 2,
                          "cpu-test-layout", {"layers": 1, "kv_heads": 1, "head_dim": 1},
                          {"max_model_len": 16})
    return SnapshotManifest(snapshot_id, session_id, generation, identity,
                            (1, 2, 3, 4), 3, (4,), 77, SamplingConfig(), 3, 0,
                            durability=durability).to_dict()


class CPUFixtureAllocation:
    """Ordinary Python bytes; specifically not pinned and not CUDA accessible."""
    def __init__(self, data):
        self.data, self.size, self.closed = data, len(data), False

    def read(self):
        if self.closed:
            raise RuntimeError("test allocation was released too early")
        return self.data

    def close(self):
        self.closed = True


def _owner_main(root, capacity, connection):
    from kvsnap.connector.volatile import VolatileOwner
    owner = None
    try:
        owner = VolatileOwner(root, capacity, reconcile_interval=60)
        connection.send(("ready", os.getpid()))
        while True:
            command, arguments = connection.recv()
            if command == "stop":
                break
            try:
                if command == "prepare":
                    blocks = arguments.pop("blocks")
                    result = owner.prepare_allocations([CPUFixtureAllocation(value) for value in blocks], **arguments)
                elif command == "stats":
                    result = owner.stats()
                elif command == "reconcile":
                    result = owner.reconcile()
                else:
                    raise ValueError("unknown test operation")
                connection.send(("ok", result))
            except Exception as exc:
                connection.send(("error", type(exc).__name__, str(exc)))
    except BaseException as exc:
        connection.send(("fatal", type(exc).__name__, str(exc)))
    finally:
        if owner is not None:
            owner.close()
        connection.close()


class OwnerProcess:
    def __init__(self, root, capacity=10):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_owner_main, args=(str(root), capacity, child))
        self.process.start()
        child.close()
        if not self.connection.poll(15):
            self.close()
            raise AssertionError("volatile owner failed to start")
        ready = self.connection.recv()
        if ready != ("ready", self.process.pid):
            self.close()
            raise AssertionError(ready)

    def request(self, command, **arguments):
        self.connection.send((command, arguments))
        if not self.connection.poll(15):
            raise AssertionError("volatile worker command timed out")
        response = self.connection.recv()
        if response[0] != "ok":
            raise AssertionError(response)
        return response[1]

    def prepare(self, blocks=(b"prefix", b"tail"), session_id="session", generation=0):
        return self.request("prepare", blocks=list(blocks), session_id=session_id, generation=generation)

    def kill(self):
        if self.process.is_alive():
            self.process.kill()
        self.process.join(10)
        if self.process.exitcode != -9:
            raise AssertionError(f"expected SIGKILL, got {self.process.exitcode}")

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(("stop", {}))
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(5)
        self.connection.close()


class VolatileProcessAcceptance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kvsnap-vtest-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = SnapshotStore(self.root)
        self.addCleanup(self.store.close)

    def owner(self, capacity=10):
        worker = OwnerProcess(self.root, capacity)
        self.addCleanup(worker.close)
        self.assertNotEqual(worker.process.pid, os.getpid())
        return worker

    def reopen(self):
        self.store.close()
        self.store = SnapshotStore(self.root)
        self.addCleanup(self.store.close)

    def test_scheduler_restart_keeps_live_worker_then_sigkill_invalidates_only_volatile(self):
        worker = self.owner()
        prepared = worker.prepare()
        self.store.commit_prepared(manifest(), prepared, expected_generation=0)
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])
        self.assertEqual(list((self.root / "blobs").glob("*.blob")), [])
        self.assertEqual(worker.request("stats")["resident_bytes"], 10)
        self.store.commit(manifest("disk", "disk", durability="durable"), [b"disk-full", b"disk-tail"])
        self.reopen()
        self.assertIn("snapshot", self.store.recovery_report["valid"])
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])
        worker.kill()
        self.reopen()
        self.assertIn("snapshot", self.store.recovery_report["invalid"])
        with self.assertRaises(CorruptSnapshotError):
            self.store.load("snapshot")
        self.assertEqual(self.store.load("disk"), [b"disk-full", b"disk-tail"])

    def test_commit_confirms_ownership_without_reading_blocks_back(self):
        worker = self.owner()
        prepared = worker.prepare()
        with patch("kvsnap.connector.volatile.VolatileClient.read", side_effect=AssertionError("payload read at commit")):
            self.store.commit_prepared(manifest(), prepared, expected_generation=0)
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])

    def test_fork_and_lease_hold_one_copy_until_last_root_is_released(self):
        worker = self.owner()
        self.store.commit_prepared(manifest(), worker.prepare(), expected_generation=0)
        self.store.fork("snapshot", "fork", "branch", expected_parent_generation=0)
        self.assertEqual(worker.request("reconcile")["resident_bytes"], 10)
        self.assertEqual(self.store.load("fork"), [b"prefix", b"tail"])
        self.assertEqual({row["refcount"] for row in self.store.export()["blobs"]}, {2})
        # Admission failure must keep existing snapshots and discard temporary allocations.
        worker.connection.send(("prepare", dict(blocks=[b"other!", b"more"], session_id="full", generation=0)))
        self.assertTrue(worker.connection.poll(15))
        error = worker.connection.recv()
        self.assertEqual(error[:2], ("error", "CapacityError"), error)
        self.assertEqual(worker.request("stats")["resident_bytes"], 10)
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])
        lease = self.store.acquire_lease("snapshot")
        self.store.delete_snapshot("fork")
        self.store.delete_snapshot("snapshot")
        self.store.collect_garbage()
        self.assertEqual(worker.request("reconcile")["resident_bytes"], 10)
        self.store.renew_lease(lease)
        self.store.release_lease(lease)
        self.store.collect_garbage()
        self.assertEqual(worker.request("reconcile")["resident_bytes"], 0)

    def test_stale_generation_and_forged_descriptor_never_publish(self):
        worker = self.owner()
        prepared = worker.prepare()
        forged = copy.deepcopy(prepared)
        first = next(iter(forged["locations"].values()))
        first["handle"] = "0" * 32
        with self.assertRaises(StoreError):
            self.store.commit_prepared(manifest(), forged, expected_generation=0)
        self.store.claim_generation("session", 0, 1)
        with self.assertRaises(GenerationConflict):
            self.store.commit_prepared(manifest(), prepared, expected_generation=0)
        self.assertEqual(self.store.list_manifests(include_invalid=True), [])
        # The rejected preparation remains a temporary root only until expiry.
        path = self.root / "prepared" / (prepared["preparation_id"] + ".json")
        expired = json.loads(path.read_text())
        expired["expires_at"] = 0
        path.write_text(json.dumps(expired))
        self.store.collect_garbage()
        self.assertEqual(worker.request("reconcile")["resident_bytes"], 0)

    def test_same_blob_in_two_owners_does_not_break_surviving_snapshot(self):
        first, second = self.owner(), self.owner()
        self.store.commit_prepared(manifest("first", "first"), first.prepare(session_id="first"))
        self.store.commit_prepared(manifest("second", "second"), second.prepare(session_id="second"))
        first.request("reconcile")
        second.request("reconcile")
        self.assertEqual(self.store.load("first"), [b"prefix", b"tail"])
        second.kill()
        self.reopen()
        self.assertIn("first", self.store.recovery_report["valid"])
        self.assertEqual(self.store.load("first"), [b"prefix", b"tail"])


if __name__ == "__main__":
    unittest.main()
