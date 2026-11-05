"""The connector at TP=2: per-rank shards, all-or-nothing commits, shared leases, loud misses.

vLLM and CUDA are stubbed as in test_failure_policy.
"""
from contextlib import contextmanager
import ctypes
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_failure_policy import connector as make_connector, import_connector, meta
from kvsnap.connector.identity import runtime_identity
from kvsnap.models import KVIdentity, rank_segments
from kvsnap.store import BlobWriter, SnapshotStore


def connector(module, rank=0):
    c = make_connector(module)
    c.tp_size, c.rank = 2, rank
    return c


def config(tp):
    return SimpleNamespace(
        model_config=SimpleNamespace(model="model", revision="revision", tokenizer_revision="tokenizer-revision",
            dtype="torch.bfloat16", max_model_len=4096,
            hf_text_config=SimpleNamespace(to_dict=lambda: {"rope_theta": 10000}),
            get_num_layers=lambda _: 36, get_num_kv_heads=lambda _: 8 // tp, get_head_size=lambda: 128),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp))


def two_rank_manifest(ranks=(0, 1)):
    segments = [{"blob_id": f"r{rank}b{block}", "offset": 0, "length": 4, "first_block": block, "nblocks": 1,
                 "crc32c": 0, **({"rank": rank} if rank else {})} for rank in ranks for block in range(2)]
    return {"snapshot_id": "snapshot", "num_kv_tokens": 2, "durability": "durable", "segments": segments,
            "session_id": "session", "generation": 1}


class TwoRankSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = import_connector()

    def saving(self):
        c = connector(self.module)
        c.saving = {"request": self.module.Transfer("request", [1, 2], {"snapshot_id": "snapshot", "generation": 1})}
        return c

    def test_worker_metadata_of_both_ranks_merges_per_request(self):
        merged = self.module.KvsnapWorkerMetadata(shards={"request": {0: "p0"}}, loaded_leases=["lease"]).aggregate(
            self.module.KvsnapWorkerMetadata(shards={"request": {1: "p1"}, "other": {1: "q1"}},
                                             loaded_leases=["lease"]))
        self.assertEqual(merged.shards, {"request": {0: "p0", 1: "p1"}, "other": {1: "q1"}})
        self.assertEqual(merged.loaded_leases, ["lease", "lease"])
        with self.assertRaises(RuntimeError):
            merged.aggregate(self.module.KvsnapWorkerMetadata(shards={"request": {1: "again"}}))

    def test_a_snapshot_is_committed_only_once_every_rank_has_prepared_its_shard(self):
        c = self.saving()
        c._event = Mock()
        p0, p1 = {"preparation_id": "p0"}, {"preparation_id": "p1"}
        c.update_connector_output(meta(self.module, shards={"request": {1: p1}}))
        c.store.commit_prepared.assert_not_called()
        c.update_connector_output(meta(self.module, shards={"request": {0: p0}}))
        c.store.commit_prepared.assert_called_once()
        self.assertEqual(c.store.commit_prepared.call_args.args[1], [p0, p1])
        self.assertEqual((c.saving, c.shards), ({}, {}))
        c.update_connector_output(meta(self.module))     # a failed commit job would raise here
        self.assertEqual(c.committing, set())
        event, = [e for e in c._event.call_args_list if e.args[0] == "manifest_committed"]
        self.assertEqual(event.kwargs["preparation_id"], ["p0", "p1"])

    def test_one_ranks_failed_save_fails_the_snapshot_and_its_other_shard_is_dropped(self):
        c = self.saving()
        c.update_connector_output(meta(self.module, shards={"request": {0: "p0"}}))
        c.update_connector_output(meta(self.module, errors={"request": "PinnedCapacityError: full"}))
        c.store.record_save_failure.assert_called_once()
        self.assertEqual(c.shards, {})
        c.update_connector_output(meta(self.module, shards={"request": {1: "p1"}}))   # arrives late
        c.store.commit_prepared.assert_not_called()
        self.assertEqual(c.shards, {})

    def test_a_load_lease_outlives_every_rank_still_reading_its_shard(self):
        c = connector(self.module)
        c.update_connector_output(meta(self.module, loaded_leases=["lease"]))
        c.store.release_lease.assert_not_called()
        c.update_connector_output(meta(self.module, loaded_leases=["lease"]))
        c.store.release_lease.assert_called_once_with("lease")
        self.assertEqual(c.lease_reports, {})

    def test_a_snapshot_saved_at_tp2_misses_at_tp1_and_the_reverse(self):
        for saved, engine in ((2, 1), (1, 2)):
            with self.subTest(saved=saved, engine=engine):
                c = make_connector(self.module)
                c._vllm_config = config(engine)
                del c._check_engine_identity
                c._event = Mock()
                claimed = runtime_identity(config(saved), "layout").to_dict()
                request = SimpleNamespace(request_id="request", prompt_token_ids=[1, 2, 3],
                                          kv_transfer_params={"kvsnap": {"resume_snapshot_id": "snapshot",
                                                                         "generation": 2, "identity": claimed}})
                self.assertEqual(c.get_num_new_matched_tokens(request, 0), (0, False))
                c.store.get_manifest.assert_not_called()
                event, = [e for e in c._event.call_args_list if e.args[0] == "restore_miss"]
                self.assertIn("model_shape", event.kwargs["reason"])

    def test_host_memory_room_is_made_for_every_ranks_shard(self):
        c = connector(self.module)
        c._check_engine_identity = Mock()
        c.store.make_room_volatile.return_value = []
        op = {"op": "save", "snapshot_id": "snapshot", "identity": {"kv_layout": '[{"bytes": 100}]'}}
        request = SimpleNamespace(request_id="request", kv_transfer_params={"kvsnap": op}, num_output_placeholders=0,
                                  all_token_ids=[1, 2], num_computed_tokens=2, output_token_ids=[],
                                  status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"))
        manifest = {"snapshot_id": "snapshot", "num_kv_tokens": 2, "durability": "volatile", "session_id": "s",
                    "generation": 1, "identity": op["identity"], "token_ids": [1, 2], "pending_tokens": []}
        with patch.object(self.module, "finish_manifest", return_value=manifest):
            c.request_finished(request, [5, 6])
        c.store.make_room_volatile.assert_called_once_with(2 * 100 * 2, c.host_tier_bytes * 2)


class TwoRankWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = import_connector()

    def load(self, worker, manifest):
        transfer = self.module.Transfer("request", [11, 12], manifest, lease_id="lease")
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[transfer])
        worker.start_load_kv(None)
        return worker.get_finished(set())

    def test_each_rank_reads_and_copies_only_its_own_shard(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                worker = connector(self.module, rank)
                self.assertEqual(self.load(worker, two_rank_manifest()), (set(), {"request"}))
                segments = worker.writer.read_prepared_into.call_args.args[0]
                self.assertEqual([s["blob_id"] for s in segments], [f"r{rank}b0", f"r{rank}b1"])
                self.assertEqual(worker.dataplane.load_scatter.call_args.args[0], [11, 12])
                self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_a_rank_whose_shard_is_missing_fails_the_load_before_any_copy(self):
        worker = connector(self.module, rank=1)
        self.assertEqual(self.load(worker, two_rank_manifest(ranks=(0,))), (set(), {"request"}))
        worker.writer.read_prepared_into.assert_not_called()
        worker.dataplane.load_scatter.assert_not_called()
        self.assertEqual(worker.get_block_ids_with_load_errors(), {11, 12})   # vLLM recomputes them
        result = worker.build_connector_worker_meta()
        self.assertEqual((result.failed_loads, result.loaded_leases), (["request"], ["lease"]))

    def test_a_rank_borrows_its_own_host_memory_shard_in_place(self):
        worker = connector(self.module, rank=1)
        worker.volatile = Mock(descriptor={"owner_id": "rank1"})
        worker.volatile.borrow.return_value.__enter__ = Mock(return_value=["a10", "a11"])
        worker.volatile.borrow.return_value.__exit__ = Mock(return_value=False)
        manifest = dict(two_rank_manifest(), durability="volatile")
        locations = {s["blob_id"]: {"owner": {"owner_id": f"rank{s.get('rank', 0)}"}, "handle": "h"}
                     for s in manifest["segments"]}
        transfer = self.module.Transfer("request", [11, 12], manifest, lease_id="lease", volatile_locations=locations)
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[transfer])
        worker.start_load_kv(None)
        self.assertEqual(worker.get_finished(set()), (set(), {"request"}))
        # Rank 0's blocks live in rank 0's worker; rank 1's own are borrowed, with no staging copy.
        self.assertEqual([s["blob_id"] for s in worker.volatile.borrow.call_args.args[0]], ["r1b0", "r1b1"])
        worker.dataplane.staging.assert_not_called()
        self.assertEqual(worker.dataplane.load_scatter.call_args.args[1], [("a10", 0), ("a11", 0)])

    def test_a_rank_reports_its_save_as_its_own_shard(self):
        worker = connector(self.module, rank=1)
        worker.dataplane.pool.reserve.return_value = Mock(read_range=lambda offset, size: b"k" * size)
        worker.writer.prepare_blocks.return_value = {"preparation_id": "p", "segments": []}
        transfer = self.module.Transfer("request", [11], {"snapshot_id": "snapshot", "num_kv_tokens": 1,
                                        "durability": "durable", "segments": [], "session_id": "s", "generation": 1})
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(saves=[transfer])
        worker.wait_for_save()
        self.assertEqual(worker.get_finished({"request"}), ({"request"}, set()))
        result = worker.build_connector_worker_meta()
        self.assertEqual((result.prepared, result.shards), ({}, {"request": {1: {"preparation_id": "p",
                                                                                 "segments": []}}}))


class Pinned:
    """A CPU stand-in for a pinned allocation: real memory at a real address."""
    def __init__(self, size):
        self.size, self.memory = size, ctypes.create_string_buffer(size)
        self.pointer = ctypes.addressof(self.memory)

    def read_range(self, offset, size):
        return ctypes.string_at(self.pointer + offset, size)

    def close(self):
        pass


class TwoRankRoundTripTests(unittest.TestCase):
    """Save, commit, match and restore across a scheduler and two rank workers on one real store."""
    BLOCK = 4

    @classmethod
    def setUpClass(cls):
        cls.module = import_connector()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.store.close)
        self.identity = KVIdentity("model", "rev", "tok", "bf16", 1, "layout",
                                   {"layers": 1, "kv_heads": 4, "head_size": 1, "tensor_parallel_size": 2},
                                   {"max_model_len": 64}).to_dict()
        self.scheduler = connector(self.module)
        self.scheduler.store = self.store
        self.scheduler._event = Mock()
        self.workers = [connector(self.module, rank) for rank in (0, 1)]
        for worker in self.workers:
            worker.writer = BlobWriter(self.temp.name, io_backend="python")
            worker.dataplane.block_bytes = self.BLOCK
            worker.dataplane.pool.reserve.side_effect = Pinned
            worker.dataplane.save_gather.side_effect = self.gather(worker.rank)

            @contextmanager
            def staging(nblocks):
                yield Pinned(self.BLOCK * nblocks)
            worker.dataplane.staging.side_effect = staging

    def gather(self, rank):
        def copy(block_ids, targets, num_tokens):
            for block, (allocation, offset) in zip(block_ids, targets):
                ctypes.memmove(allocation.pointer + offset, bytes([rank, block, 7, 7]), self.BLOCK)
        return copy

    def step(self, meta_for_workers, finished=()):
        metas = []
        for worker in self.workers:
            worker.bind_connector_metadata(meta_for_workers)
            worker.wait_for_save()
            worker.start_load_kv(None)
            worker.get_finished(set(finished))
            metas.append(worker.build_connector_worker_meta())
        self.scheduler.update_connector_output(SimpleNamespace(kv_connector_worker_meta=metas[0].aggregate(metas[1])))
        self.scheduler.update_connector_output(SimpleNamespace(kv_connector_worker_meta=None))  # surfaces commit errors
        return metas

    def request(self, request_id, op, tokens):
        return SimpleNamespace(request_id=request_id, kv_transfer_params={"kvsnap": op}, prompt_token_ids=tokens,
                               all_token_ids=tokens, num_computed_tokens=2, output_token_ids=[tokens[-1]],
                               num_output_placeholders=0, status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"))

    def test_each_rank_restores_the_shard_it_saved(self):
        save = {"op": "save", "snapshot_id": "snap", "session_id": "session", "generation": 1,
                "identity": self.identity, "seed": 0, "sampling": {"temperature": 0.0},
                "durability": "durable", "output_seq_start": 0, "outputs": [], "remaining_max_tokens": 4}
        self.assertEqual(self.scheduler.request_finished(self.request("turn1", save, [10, 11, 12]), [5, 6])[0], True)
        self.step(self.scheduler.build_connector_meta(SimpleNamespace(total_num_scheduled_tokens=0,
                                                                      num_scheduled_tokens={})), {"turn1"})
        manifest = self.store.get_manifest("snap")
        self.assertEqual(manifest["state"], "Durable")
        for rank in (0, 1):
            blocks = self.workers[rank].writer.read_prepared_blocks(rank_segments(manifest, rank))
            self.assertEqual(blocks, [bytes([rank, 5, 7, 7]), bytes([rank, 6, 7, 7])])

        resume = self.request("turn2", {"resume_snapshot_id": "snap", "identity": self.identity, "generation": 2},
                              [10, 11, 12, 13])
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(resume, 0), (2, True))
        self.scheduler.update_state_after_alloc(resume, SimpleNamespace(get_block_ids=lambda: ([8, 9, 10],)), 2)
        lease = self.scheduler.loads[0].lease_id
        loaded = {}
        for worker in self.workers:
            worker.dataplane.load_scatter.side_effect = (
                lambda targets, sources, progress=None, rank=worker.rank: loaded.setdefault(
                    rank, [a.read_range(offset, self.BLOCK) for a, offset in sources]) and 1)
        metas = self.step(self.scheduler.build_connector_meta(SimpleNamespace(total_num_scheduled_tokens=0,
                                                                              num_scheduled_tokens={})))
        self.assertEqual(loaded, {rank: [bytes([rank, 5, 7, 7]), bytes([rank, 6, 7, 7])] for rank in (0, 1)})
        self.assertEqual([m.loaded_leases for m in metas], [[lease], [lease]])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM leases").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
