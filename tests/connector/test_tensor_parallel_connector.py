"""The connector at TP=2: per-rank shards, all-or-nothing commits, shared leases, loud misses.

vLLM and CUDA are stubbed as in test_failure_policy.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_failure_policy import connector as make_connector, import_connector, meta
from kvsnap.connector.identity import runtime_identity


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
        c.update_connector_output(meta(self.module, shards={"request": {1: "p1"}}))
        c.store.commit_prepared.assert_not_called()
        c.update_connector_output(meta(self.module, shards={"request": {0: "p0"}}))
        c.store.commit_prepared.assert_called_once()
        self.assertEqual(c.store.commit_prepared.call_args.args[1], ["p0", "p1"])
        self.assertEqual((c.saving, c.shards), ({}, {}))

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


if __name__ == "__main__":
    unittest.main()
