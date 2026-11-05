"""TP>1 snapshots: one shard per rank under one name, and loud misses when a shard or the TP size differs.

CPU only: real SQLite store, real blob files and real volatile owners with
fixture allocations; no vLLM, no CUDA.
"""
import json
import tempfile
import unittest

from kvsnap.connector.protocol import matched_tokens
from kvsnap.models import (ContractError, IdentityMismatch, KVIdentity, SamplingConfig, SnapshotManifest,
                           rank_segments)
from kvsnap.store import BlobWriter, CorruptSnapshotError, SnapshotStore, StoreError
from kvsnap.connector.volatile import VolatileOwner

from test_volatile import FixturePool


def identity(tp=1):
    """At TP=2 each rank holds half of the KV heads, so its block layout differs too."""
    sharded = tp is not None and tp > 1
    shape = {"layers": 1, "kv_heads": 4 if sharded else 8, "head_dim": 1}
    if tp is not None:
        shape["tensor_parallel_size"] = tp
    return KVIdentity("fixture", "rev", "tokenizer-rev", "bf16", 2, "layout-4-heads" if sharded else "layout-8-heads",
                      shape, {"max_model_len": 32})


def manifest(tp=2, snapshot_id="snapshot", durability="durable", segments=(), state="Writing"):
    return SnapshotManifest(snapshot_id, "session", 0, identity(tp), (10, 11, 12, 13), 3, (13,), 42,
                            SamplingConfig(), 8, 0, durability=durability, state=state,
                            segments=tuple(segments)).to_dict()


def segment(first_block, rank=0, blob="b"):
    return {"blob_id": f"{blob}{rank}{first_block}", "offset": 0, "length": 4, "first_block": first_block,
            "nblocks": 1, "crc32c": 0, "rank": rank}


def committed(segments, tp=2):
    """Validate a committed manifest's segment layout (3 KV tokens of block size 2: 2 blocks per rank)."""
    body = manifest(tp)
    body.update(state="Durable", segments=list(segments))
    return SnapshotManifest.from_dict(body)


RANK0 = [b"r0-prefix", b"r0-tail"]
RANK1 = [b"r1-prefix", b"r1-tail"]


class ShardLayoutTests(unittest.TestCase):
    def test_tp1_identity_and_segments_keep_their_original_form(self):
        self.assertEqual(identity(1), identity(None))
        self.assertNotIn("tensor_parallel_size", identity(1).to_dict()["model_shape"])
        body = committed([segment(0), segment(1)], tp=1).to_dict()
        self.assertTrue(all("rank" not in s for s in body["segments"]))

    def test_tp_size_is_part_of_the_kv_identity(self):
        self.assertEqual(identity(2).tensor_parallel_size, 2)
        with self.assertRaises(IdentityMismatch) as caught:
            identity(1).require_match(identity(2))
        self.assertIn("model_shape", str(caught.exception))
        for bad in (0, -1, "2", 1.0):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                KVIdentity("f", "r", "t", "bf16", 2, "l", {"tensor_parallel_size": bad}, {"max_model_len": 8})

    def test_each_rank_numbers_its_own_blocks_and_ranks_come_in_order(self):
        body = committed([segment(0), segment(1), segment(0, 1), segment(1, 1)]).to_dict()
        self.assertEqual([s["blob_id"] for s in rank_segments(body, 0)], ["b00", "b01"])
        self.assertEqual([s["blob_id"] for s in rank_segments(body, 1)], ["b10", "b11"])
        self.assertEqual([s.get("rank", 0) for s in body["segments"]], [0, 0, 1, 1])

    def test_a_committed_snapshot_needs_every_ranks_whole_shard(self):
        cases = {
            "rank 1 missing": [segment(0), segment(1)],
            "rank 0 missing": [segment(0, 1), segment(1, 1)],
            "rank 1 short": [segment(0), segment(1), segment(0, 1)],
            "ranks out of order": [segment(0, 1), segment(1, 1), segment(0), segment(1)],
            "rank beyond TP size": [segment(0), segment(1), segment(0, 1), segment(1, 1), segment(0, 2)],
            "rank 1 not numbered from 0": [segment(0), segment(1), segment(1, 1), segment(2, 1)],
        }
        for name, segments in cases.items():
            with self.subTest(name), self.assertRaises(ContractError):
                committed(segments)

    def test_a_resume_at_another_tp_size_or_of_a_partial_snapshot_misses(self):
        good = committed([segment(0), segment(1), segment(0, 1), segment(1, 1)]).to_dict()
        prompt = [10, 11, 12, 13, 14]
        self.assertEqual(matched_tokens(good, identity(2).to_dict(), prompt, 0), 3)
        with self.assertRaises(IdentityMismatch):
            matched_tokens(good, identity(1).to_dict(), prompt, 0)
        partial = dict(good, segments=good["segments"][:2])
        with self.assertRaises(ContractError):
            matched_tokens(partial, identity(2).to_dict(), prompt, 0)


class ShardStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(self.temp.name)
        self.writer = BlobWriter(self.temp.name)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def reopen(self):
        self.store.close()
        self.store = SnapshotStore(self.temp.name)

    def commit_both(self):
        shards = [self.writer.prepare_blocks(RANK0), self.writer.prepare_blocks(RANK1)]
        return self.store.commit_prepared(manifest(), shards, expected_generation=0)

    def test_both_ranks_shards_are_published_under_one_name_and_survive_a_restart(self):
        body = self.commit_both()
        self.assertEqual(body["state"], "Durable")
        self.assertEqual(self.writer.read_prepared_blocks(rank_segments(body, 0)), RANK0)
        self.assertEqual(self.writer.read_prepared_blocks(rank_segments(body, 1)), RANK1)
        self.assertEqual(self.store.load("snapshot"), RANK0 + RANK1)
        self.assertEqual(list(self.store.prepared_dir.glob("*.json")), [])
        self.reopen()
        self.assertEqual(self.store.recovery_report["valid"], ["snapshot"])
        self.assertEqual(self.store.get_manifest("snapshot")["segments"], body["segments"])

    def test_a_commit_without_every_ranks_preparation_publishes_nothing(self):
        rank0 = self.writer.prepare_blocks(RANK0)
        for prepared in (rank0, [rank0]):
            with self.subTest(prepared=type(prepared).__name__), self.assertRaises(StoreError):
                self.store.commit_prepared(manifest(), prepared)
        short = self.writer.prepare_blocks(RANK1[:1])
        with self.assertRaises(ValueError):
            self.store.commit_prepared(manifest(), [rank0, short])
        with self.assertRaises(StoreError):
            self.store.commit(manifest(), RANK0)       # one rank's blocks cannot stand for a TP snapshot
        with self.assertRaises(KeyError):
            self.store.get_manifest("snapshot")

    def test_a_lost_shard_blob_invalidates_the_snapshot_at_recovery(self):
        body = self.commit_both()
        (self.store.blob_dir / (rank_segments(body, 1)[0]["blob_id"] + ".blob")).unlink()
        self.reopen()
        self.assertEqual(self.store.recovery_report["invalid"], ["snapshot"])
        with self.assertRaises(CorruptSnapshotError):
            self.store.get_manifest("snapshot")

    def test_a_manifest_that_lost_a_shard_is_never_loadable(self):
        body = self.commit_both()
        body["segments"] = rank_segments(body, 0)
        self.store.db.execute("UPDATE snapshots SET manifest_json=? WHERE snapshot_id=?",
                              (json.dumps(body, sort_keys=True, separators=(",", ":")), "snapshot"))
        with self.assertRaises(ContractError):
            matched_tokens(self.store.get_manifest("snapshot"), identity(2).to_dict(), [10, 11, 12, 13, 14], 0)
        with self.assertRaises(CorruptSnapshotError):
            self.store.load("snapshot")
        self.assertEqual(self.store.get_manifest("snapshot", include_invalid=True)["state"], "Invalid")


class ShardVolatileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kvsnap-tp-")
        self.store = SnapshotStore(self.temp.name)
        self.owners = [VolatileOwner(self.temp.name, 64, reconcile_interval=60) for _ in range(2)]
        self.pool = FixturePool(128)

    def tearDown(self):
        for owner in self.owners:
            owner.close()
        self.store.close()
        self.temp.cleanup()

    def prepare(self, rank, values):
        return self.owners[rank].prepare_allocations((self.pool.reserve(v) for v in values),
                                                     session_id="session", generation=0)

    def test_each_ranks_host_memory_shard_stays_with_its_own_worker(self):
        body = self.store.commit_prepared(manifest(durability="volatile"),
                                          [self.prepare(0, RANK0), self.prepare(1, RANK1)])
        self.assertEqual(body["state"], "HostValid")
        locations = self.store.volatile_locations(body)
        for rank in (0, 1):
            owners = {locations[s["blob_id"]]["owner"]["owner_id"] for s in rank_segments(body, rank)}
            self.assertEqual(owners, {self.owners[rank].owner_id})
        self.assertEqual(self.store.load("snapshot"), RANK0 + RANK1)

    def test_a_host_memory_commit_missing_a_rank_publishes_nothing(self):
        with self.assertRaises(StoreError):
            self.store.commit_prepared(manifest(durability="volatile"), [self.prepare(0, RANK0)])
        with self.assertRaises(KeyError):
            self.store.get_manifest("snapshot")


if __name__ == "__main__":
    unittest.main()
