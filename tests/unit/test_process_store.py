"""Process-boundary tests of the store, on CPU bytes; CUDA is not involved."""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest

from kvsnap.models import KVIdentity, SamplingConfig, SnapshotManifest
from kvsnap.store import BlobWriter, GenerationConflict, SnapshotStore, StoreError


def manifest(snapshot_id="checkpoint", session_id="session", generation=0, durability="durable"):
    identity = KVIdentity("process-fixture", "r1", "t1", "bf16", 2,
                          "cpu-fixture", {"layers": 1, "kv_heads": 1, "head_dim": 1},
                          {"max_model_len": 16})
    return SnapshotManifest(snapshot_id, session_id, generation, identity,
                            (1, 2, 3, 4), 3, (4,), 17, SamplingConfig(), 4, 0,
                            durability=durability).to_dict()


def _compete(root, ready, start, output):
    try:
        with SnapshotStore(root, recover_on_open=False) as store:
            ready.put(os.getpid())
            if not start.wait(10):
                raise RuntimeError("parent did not start generation race")
            try:
                store.claim_generation("race", 0, 1)
                output.put((os.getpid(), "won"))
            except GenerationConflict:
                output.put((os.getpid(), "fenced"))
    except BaseException as exc:
        output.put((os.getpid(), "error", repr(exc)))


def _prepare_and_wait(root, connection):
    prepared = BlobWriter(root).prepare_blocks([b"prefix", b"tail"], "durable")
    connection.send((os.getpid(), prepared))
    connection.recv()  # Parent deliberately kills this process after publication.


def _inherited_store(store, connection):
    try:
        store.stats()
    except StoreError as exc:
        connection.send(str(exc))
    else:
        connection.send("incorrectly accepted inherited connection")
    finally:
        connection.close()


class ProcessStoreAcceptance(unittest.TestCase):
    def test_generation_cas_has_exactly_one_winner_across_processes(self):
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as root, SnapshotStore(root) as store:
            ready, output, start = context.Queue(), context.Queue(), context.Event()
            processes = [context.Process(target=_compete, args=(root, ready, start, output)) for _ in range(4)]
            try:
                for process in processes:
                    process.start()
                pids = {ready.get(timeout=15) for _ in processes}
                self.assertEqual(len(pids), len(processes))
                self.assertNotIn(os.getpid(), pids)
                start.set()
                results = [output.get(timeout=15) for _ in processes]
                self.assertEqual(sorted(result[1] for result in results), ["fenced"] * 3 + ["won"], results)
                for process in processes:
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(store.get_generation("race"), 1)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.kill()
                    process.join(5)
                ready.close()
                output.close()

    def test_killed_worker_preparation_is_protected_then_stale_completion_is_fenced(self):
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as root, SnapshotStore(root) as store:
            parent, child = context.Pipe()
            process = context.Process(target=_prepare_and_wait, args=(root, child))
            process.start()
            child.close()
            try:
                self.assertTrue(parent.poll(15), "worker did not finish durable preparation")
                pid, prepared = parent.recv()
                self.assertEqual(pid, process.pid)
                process.kill()
                process.join(10)
                self.assertEqual(process.exitcode, -9)
                store.collect_garbage()
                self.assertEqual(len(list((Path(root) / "blobs").glob("*.blob"))), 2)
                store.claim_generation("session", 0, 1)
                with self.assertRaises(GenerationConflict):
                    store.commit_prepared(manifest(), prepared, expected_generation=0)
                self.assertEqual(store.list_manifests(include_invalid=True), [])
                self.assertEqual(store.get_generation("session"), 1)
                published = store.commit_prepared(manifest("current", generation=1), prepared,
                                                  expected_generation=1)
                self.assertEqual(published["generation"], 1)
                self.assertEqual(store.load("current"), [b"prefix", b"tail"])
            finally:
                parent.close()
                if process.is_alive():
                    process.kill()
                process.join(5)

    def test_inherited_sqlite_connection_is_rejected_in_real_fork(self):
        context = mp.get_context("fork")
        with tempfile.TemporaryDirectory() as root, SnapshotStore(root) as store:
            store.commit(manifest(), [b"prefix", b"tail"])
            parent, child = context.Pipe()
            process = context.Process(target=_inherited_store, args=(store, child))
            process.start()
            child.close()
            try:
                self.assertTrue(parent.poll(10), "inherited store access blocked instead of being rejected")
                self.assertIn("cannot be inherited across fork", parent.recv())
                process.join(10)
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(store.load("checkpoint"), [b"prefix", b"tail"])
            finally:
                parent.close()
                if process.is_alive():
                    process.kill()
                process.join(5)


if __name__ == "__main__":
    unittest.main()
