"""CPU storage contract tests. They test neither CUDA copies nor power loss."""
import errno
import fcntl
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvsnap.models import KVIdentity, SamplingConfig, SnapshotManifest
from kvsnap.store import (BlobWriter, CapacityError, CorruptSnapshotError,
                       GenerationConflict, SnapshotStore, StoreError, crc32c)
from kvsnap.store.store import HEADER, _write_preparation


def manifest(snapshot_id="snapshot", session_id="session", durability="durable", generation=0):
    identity = KVIdentity("fixture", "rev", "tokenizer-rev", "bf16", 2, "fixture-BHTD",
                          {"layers": 1, "kv_heads": 1, "head_dim": 1},
                          {"max_model_len": 32, "rope_theta": 10000})
    return SnapshotManifest(snapshot_id, session_id, generation, identity,
                            (10, 11, 12, 13), 3, (13,), 42, SamplingConfig(),
                            8, 0, durability=durability).to_dict()


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(self.temp.name)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def reopen(self):
        self.store.close()
        self.store = SnapshotStore(self.temp.name)

    def test_castagnoli_reference_vector_and_incremental(self):
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(crc32c(b"456789", crc32c(b"123")), 0xE3069283)
        self.assertEqual(crc32c(b""), 0)

    def test_three_levels_have_explicit_restart_contract(self):
        for level in ("volatile", "process", "durable"):
            result = self.store.commit(manifest(level, level, level), [b"prefix", b"tail"])
            self.assertEqual(result["state"], "HostValid" if level == "volatile" else "Durable")
            self.assertEqual(self.store.load(level), [b"prefix", b"tail"])
        self.reopen()
        self.assertEqual(set(self.store.recovery_report["valid"]), {"process", "durable"})
        with self.assertRaises(CorruptSnapshotError):
            self.store.load("volatile")
        for level in ("process", "durable"):
            self.assertEqual(self.store.load(level), [b"prefix", b"tail"])

    def test_persistent_read_never_hides_bad_disk_behind_host_cache(self):
        self.store.commit(manifest("host", "a", "volatile"), [b"prefix", b"tail"])
        disk = self.store.commit(manifest("disk", "b"), [b"prefix", b"tail"])
        path = self.store.blob_dir / (disk["segments"][0]["blob_id"] + ".blob")
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaises(CorruptSnapshotError):
            self.store.load("disk")
        self.assertEqual(self.store.get_manifest("disk", include_invalid=True)["state"], "Invalid")
        self.assertEqual(self.store.load("host"), [b"prefix", b"tail"])

    def test_recovery_detects_corruption_and_missing_blob_before_any_load(self):
        body = self.store.commit(manifest(), [b"prefix", b"tail"])
        path = self.store.blob_dir / (body["segments"][1]["blob_id"] + ".blob")
        path.unlink()
        self.reopen()
        self.assertEqual(self.store.recovery_report["invalid"], ["snapshot"])
        with self.assertRaises(CorruptSnapshotError):
            self.store.load("snapshot")
        self.assertTrue(all(row["refcount"] == 0 for row in self.store.export()["blobs"]))

    def test_identity_and_token_mismatch_do_not_destroy_valid_snapshot(self):
        original = manifest()
        self.store.commit(original, [b"prefix", b"tail"])
        changed = dict(original["identity"], kv_dtype="fp8")
        with self.assertRaises(StoreError):
            self.store.load("snapshot", identity=changed)
        with self.assertRaises(StoreError):
            self.store.load("snapshot", token_prefix=[10, 99])
        self.assertEqual(self.store.load("snapshot", token_prefix=[10, 11, 12, 13, 14]), [b"prefix", b"tail"])

    def test_rejects_bad_boundary_budget_and_sampling_before_writing(self):
        bad = manifest()
        bad["pending_tokens"] = [12, 13]
        with self.assertRaises(ValueError):
            self.store.commit(bad, [b"prefix", b"tail"])
        bad = manifest()
        bad["sampling"]["frequency_penalty"] = 1
        with self.assertRaises(ValueError):
            self.store.commit(bad, [b"prefix", b"tail"])
        with self.assertRaises(ValueError):
            self.store.commit(manifest(), [b"prefix"])
        self.assertEqual(self.store.list_manifests(include_invalid=True), [])

    def test_gc_fork_reference_reconstruction_and_private_tail(self):
        first = self.store.commit(manifest(), [b"prefix", b"tail"])
        self.store.fork("snapshot", "fork", "branch")
        self.assertEqual({r["refcount"] for r in self.store.export()["blobs"]}, {2})
        branch = manifest("branch-next", "branch")
        branch["parent_id"] = "fork"
        second = self.store.commit(branch, [b"prefix", b"new-tail"])
        self.assertEqual(first["segments"][0]["blob_id"], second["segments"][0]["blob_id"])
        self.assertNotEqual(first["segments"][1]["blob_id"], second["segments"][1]["blob_id"])
        self.store.db.execute("UPDATE blobs SET refcount=99")
        self.reopen()
        self.assertEqual(sorted(row["refcount"] for row in self.store.export()["blobs"]), [1, 2, 3])
        with self.assertRaises(StoreError):
            self.store.delete_snapshot("snapshot")
        self.store.delete_snapshot("branch-next")
        self.store.delete_snapshot("fork")
        self.store.collect_garbage()
        self.assertEqual(self.store.stats()["unique_blobs"], 2)
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])

    def test_invalidation_and_deletion_keep_counts_equal_to_a_rebuild(self):
        def counts():
            return {row["blob_id"]: row["refcount"] for row in self.store.export()["blobs"]}

        self.store.commit(manifest("a", "a"), [b"shared", b"a-tail"])
        self.store.fork("a", "a-fork", "fork")
        child = manifest("a-child", "a", generation=0)
        child["parent_id"] = "a"
        self.store.commit(child, [b"shared", b"child-tail"])
        self.store.commit(manifest("b", "b", "volatile"), [b"shared", b"b-tail"])
        self.store.commit(manifest("c", "c"), [b"c-head", b"c-tail"])
        self.store.invalidate_snapshot("a", "test")
        self.store.invalidate_snapshot("b", "test")
        self.store.invalidate_snapshot("b", "again")      # already invalid: no change
        self.store.delete_snapshot("c")
        incremental = counts()
        with self.store._lock, self.store._transaction():
            self.store._rebuild_references()
        self.assertEqual(incremental, counts())
        shared = self.store.commit(manifest("d", "d"), [b"shared", b"d"])["segments"][0]["blob_id"]
        self.assertEqual(counts()[shared], 3)              # a-fork, a-child, d

    def test_a_host_memory_invalidation_skips_the_fsync(self):
        self.store.commit(manifest("v", "v", "volatile"), [b"x", b"y"])
        self.store.commit(manifest("d", "d"), [b"x", b"z"])
        levels = []
        original = self.store._transaction

        def recording(durability="durable"):
            levels.append(durability)
            return original(durability)

        with patch.object(self.store, "_transaction", recording):
            self.store.invalidate_snapshot("v", "test")
            self.store.invalidate_snapshot("d", "test")
        self.assertEqual(levels, ["volatile", "durable"])

    def test_repeated_block_counts_one_snapshot_reference(self):
        self.store.commit(manifest(), [b"same", b"same"])
        self.assertEqual(self.store.stats()["unique_blobs"], 1)
        self.assertEqual(self.store.export()["blobs"][0]["refcount"], 1)

    def test_active_loading_lease_protects_deleted_manifest(self):
        body = self.store.commit(manifest(), [b"prefix", b"tail"])
        lease = self.store.acquire_lease("snapshot")
        self.store.delete_snapshot("snapshot")
        self.store.collect_garbage()
        self.assertEqual(len(list(self.store.blob_dir.glob("*.blob"))), 2)
        self.store.renew_lease(lease)
        self.store.release_lease(lease)
        self.store.collect_garbage()
        self.assertEqual(len(list(self.store.blob_dir.glob("*.blob"))), 0)

    def test_expired_lease_does_not_hold_capacity(self):
        self.store.commit(manifest(), [b"prefix", b"tail"])
        lease = self.store.acquire_lease("snapshot")
        self.store.delete_snapshot("snapshot")
        self.store.db.execute("UPDATE leases SET expires_at=0 WHERE lease_id=?", (lease,))
        with self.assertRaises(StoreError):
            self.store.renew_lease(lease)
        self.store.collect_garbage()
        self.assertEqual(self.store.stats()["disk_bytes"], 0)

    def test_capacity_admission_never_evicts_committed_snapshot(self):
        self.store.host_capacity_bytes = 10
        self.store.commit(manifest("one", "one", "volatile"), [b"prefix", b"tail"])
        with self.assertRaises(CapacityError):
            self.store.commit(manifest("two", "two", "volatile"), [b"new", b"bytes"])
        self.assertEqual(self.store.load("one"), [b"prefix", b"tail"])
        self.store.disk_capacity_bytes = 2 * HEADER.size + 10
        self.store.commit(manifest("spill", "spill", "durable"), [b"prefix", b"tail"])
        with self.assertRaises(CapacityError):
            self.store.commit(manifest("full", "full"), [b"another", b"tail"])

    def test_fencing_cas_between_store_instances(self):
        self.store.claim_generation("session", 0, 1)
        with SnapshotStore(self.temp.name, recover_on_open=False) as second:
            self.assertEqual(second.get_generation("session"), 1)
            second.claim_generation("session", 1, 2)
            with self.assertRaises(GenerationConflict):
                self.store.claim_generation("session", 1, 2)
            with self.assertRaises(GenerationConflict):
                self.store.commit(manifest(generation=1), [b"prefix", b"tail"])
            self.store.commit(manifest(generation=2), [b"prefix", b"tail"])

    def test_worker_preparation_survives_gc_and_scheduler_publishes(self):
        worker = BlobWriter(self.temp.name)
        prepared = worker.prepare_blocks([b"prefix", b"tail"])
        self.assertEqual(self.store.list_manifests(), [])
        self.store.collect_garbage()
        self.assertEqual(len(list(self.store.blob_dir.glob("*.blob"))), 2)
        body = self.store.commit_prepared(manifest(), prepared, expected_generation=0)
        self.assertEqual(body["state"], "Durable")
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])
        self.assertEqual(list(self.store.prepared_dir.glob("*.json")), [])

    def test_prepared_metadata_cannot_be_forged(self):
        prepared = BlobWriter(self.temp.name).prepare_blocks([b"prefix", b"tail"])
        prepared["segments"][0]["length"] += 1
        with self.assertRaises(StoreError):
            self.store.commit_prepared(manifest(), prepared)

    def test_enospc_invalidates_intent_and_recovery_needs_no_manual_fix(self):
        # This fixture injects Python's syscall. Native syscall failures have
        # separate independent C++ tests and must not bypass this injection.
        from kvsnap.store.native_io import get_io_backend
        self.store.io = get_io_backend("python")
        with patch("kvsnap.store.store.os.pwrite", side_effect=OSError(errno.ENOSPC, "injected full device")):
            with self.assertRaises(OSError):
                self.store.commit(manifest(), [b"prefix", b"tail"])
        self.reopen()
        self.assertEqual(self.store.recovery_report["invalid"], ["snapshot"])
        self.store.collect_garbage()
        self.assertEqual(self.store.stats()["disk_bytes"], 0)

    def test_a_host_memory_claim_fences_and_survives_a_process_restart(self):
        # No fsync, but committed to the WAL: only an OS crash can lose it.
        self.store.claim_generation("session", 0, 1, durable=False)
        self.reopen()
        self.assertEqual(self.store.get_generation("session"), 1)
        with self.assertRaises(GenerationConflict):
            self.store.claim_generation("session", 0, 2, durable=False)

    def test_a_reader_never_sees_a_durable_commit_in_progress(self):
        # Opened first: opening takes the writer lock the commit holds.
        reader = SnapshotStore(self.temp.name)
        self.addCleanup(reader.close)
        seen = []
        def peek(name):
            if name == "after_d2h":   # between the Writing insert and the publish
                try:
                    seen.append(reader.get_manifest("snapshot")["state"])
                except KeyError:
                    seen.append("missing")
        self.store.failpoint = peek
        self.store.commit(manifest(), [b"prefix", b"tail"])
        self.assertEqual(seen, ["missing"])
        self.assertEqual(reader.get_manifest("snapshot")["state"], "Durable")

    def test_callback_after_commit_does_not_retract_committed_manifest(self):
        def fail(name):
            if name == "after_manifest_commit":
                raise RuntimeError("late failure")
        self.store.failpoint = fail
        with self.assertRaises(RuntimeError):
            self.store.commit(manifest(), [b"prefix", b"tail"])
        self.reopen()
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])

    def test_recovery_rejects_malformed_manifest_and_rebuilds_other_snapshots(self):
        self.store.commit(manifest("bad", "bad"), [b"prefix", b"tail"])
        self.store.commit(manifest("good", "good"), [b"prefix", b"tail"])
        self.store.db.execute("UPDATE snapshots SET manifest_json='{' WHERE snapshot_id='bad'")
        self.reopen()
        self.assertEqual(self.store.recovery_report["invalid"], ["bad"])
        self.assertEqual(self.store.load("good"), [b"prefix", b"tail"])
        self.assertEqual({row["refcount"] for row in self.store.export()["blobs"]}, {1})

    def test_child_cannot_rewrite_parent_prefix(self):
        self.store.commit(manifest(), [b"prefix", b"tail"])
        child = manifest("child", "child")
        child["parent_id"] = "snapshot"
        child["token_ids"][0] = 999
        with self.assertRaises(StoreError):
            self.store.commit(child, [b"new-prefix", b"tail"])

    def test_worker_validates_all_prepared_bytes_before_return(self):
        worker = BlobWriter(self.temp.name)
        prepared = worker.prepare_blocks([b"prefix", b"tail"])
        self.assertEqual(worker.read_prepared_blocks(prepared["segments"]), [b"prefix", b"tail"])
        path = self.store.blob_dir / (prepared["segments"][1]["blob_id"] + ".blob")
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(CorruptSnapshotError):
            worker.read_prepared_blocks(prepared["segments"])

    def test_second_process_cannot_invalidate_live_volatile_snapshot_by_loading(self):
        self.store.commit(manifest(durability="volatile"), [b"prefix", b"tail"])
        with SnapshotStore(self.temp.name, recover_on_open=False) as other:
            with self.assertRaises(StoreError):
                other.load("snapshot")
        self.assertEqual(self.store.load("snapshot"), [b"prefix", b"tail"])


class HostTierAndWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(self.temp.name)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_host_tier_evicts_least_recently_used_snapshots_but_never_a_leased_one(self):
        clock = itertools.count(1_000_000)
        with patch("kvsnap.store.store.time.time", side_effect=lambda: float(next(clock))):
            for name in "abc":
                self.store.commit(manifest(name, name, "volatile"), [name.encode() * 8, name.upper().encode() * 8])
            self.store.release_lease(self.store.acquire_lease("a"))          # a is now the most recent
            self.assertEqual(self.store.volatile_bytes(), 48)
            self.assertEqual(self.store.make_room_volatile(16, 48), ["b"])
            self.assertEqual(self.store.volatile_bytes(), 32)
            lease = self.store.acquire_lease("c")
            self.assertEqual(self.store.make_room_volatile(48, 48), ["a"])   # c is leased: kept, room not made
            self.assertEqual(self.store.volatile_bytes(), 16)
            self.store.release_lease(lease)
        with self.assertRaisesRegex(CorruptSnapshotError, "evicted"):
            self.store.get_manifest("b")
        self.assertEqual(self.store.load("c"), [b"c" * 8, b"C" * 8])

    def test_save_failures_are_recorded_for_the_gateway_and_survive_reopen(self):
        self.store.record_save_failure("s", "PinnedCapacityError: pinned capacity exhausted")
        self.store.close()
        self.store = SnapshotStore(self.temp.name)
        self.assertEqual(self.store.save_failure("s"), "PinnedCapacityError: pinned capacity exhausted")
        self.assertIsNone(self.store.save_failure("other"))

    def test_blob_writer_fsyncs_without_holding_the_blob_lock(self):
        writer = BlobWriter(self.temp.name)
        lock_path = Path(self.temp.name) / ".blob.lock"
        observed = []
        real_fdatasync = os.fdatasync

        def fdatasync(fd):
            # A restore takes LOCK_SH on the blob lock; it must not wait for this flush.
            with lock_path.open("a+b") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    observed.append(True)
                except BlockingIOError:
                    observed.append(False)
            real_fdatasync(fd)

        with patch("kvsnap.store.store.os.fdatasync", side_effect=fdatasync):
            record = writer.prepare_blocks([b"block-0", b"block-1"], "durable")
        self.assertTrue(observed)
        self.assertTrue(all(observed))
        result = self.store.commit_prepared(manifest(), record)
        self.assertEqual(result["state"], "Durable")

    def test_gc_keeps_a_live_writers_temporary_files(self):
        live, stray = "a" * 64, "b" * 64
        _write_preparation(self.store.prepared_dir, dict(preparation_id="c" * 32, durability="durable",
                                                         expires_at=9e18, segments=[dict(blob_id=live)]))
        (self.store.blob_dir / f"{live}.{'1' * 32}.tmp").write_bytes(b"partial")
        (self.store.blob_dir / f"{stray}.{'2' * 32}.tmp").write_bytes(b"orphan")
        self.store.collect_garbage()
        self.assertEqual(sorted(p.name.split(".")[0] for p in self.store.blob_dir.iterdir()), [live])


if __name__ == "__main__":
    unittest.main()
