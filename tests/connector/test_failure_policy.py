"""Connector control flow and failure policy; vLLM and CUDA are stubbed.

Failures before a copy is issued must be misses (recompute) or recorded save
failures; only a copy whose completion cannot be proven may stop the engine.
"""
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from kvsnap.connector.dataplane import PinnedCapacityError
from kvsnap.connector.identity import runtime_identity
from kvsnap.connector.telemetry import EventRecorder
from kvsnap.store import GenerationConflict as StoreGenerationConflict
from kvsnap.store.store import CorruptSnapshotError, StoreError


def import_connector():
    fake_vllm = ModuleType("vllm")
    fake_vllm.__version__ = "0.31.0"
    base_name = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    fake_base = ModuleType(base_name)
    for name in ("KVConnectorMetadata", "KVConnectorWorkerMetadata"):
        setattr(fake_base, name, type(name, (), {}))

    class KVConnectorBase_V1:
        def bind_connector_metadata(self, connector_metadata):
            self._connector_metadata = connector_metadata

        def _get_connector_metadata(self):
            return self._connector_metadata
    fake_base.KVConnectorBase_V1 = KVConnectorBase_V1
    fake_base.KVConnectorRole = SimpleNamespace(SCHEDULER="scheduler", WORKER="worker")
    name = "_kvsnap_stubbed_connector"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / "kvsnap/connector/vllm.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"vllm": fake_vllm, base_name: fake_base, name: module}):
        spec.loader.exec_module(module)
    return module


class InlineExecutor:
    """Runs each job at submit time, so tests see its outcome deterministically."""
    def __init__(self, module):
        self.module = module

    def submit(self, fn, *args):
        return self.module._completed(fn, *args)

    def shutdown(self, wait=True):
        pass


def connector(module, *, async_load=True):
    c = module.KvsnapConnector.__new__(module.KvsnapConnector)
    c.telemetry = EventRecorder(directory="")
    c.async_load, c.verify_host_checksums, c.block_size = async_load, False, 1
    c.host_tier_bytes = 1 << 20
    c._init_state()
    c.executor = c.loader = c.committer = InlineExecutor(module)
    c._check_layout = Mock()
    c._check_engine_identity = Mock()
    c.dataplane = MagicMock()
    c.dataplane.block_bytes = 4
    c.writer, c.volatile, c.store = Mock(), Mock(), Mock()
    return c


def meta(module, **fields):
    return SimpleNamespace(kv_connector_worker_meta=module.KvsnapWorkerMetadata(**fields))


class WorkerSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = import_connector()

    def save(self, worker, durability="durable"):
        transfer = self.module.Transfer("request", [11, 12], {
            "snapshot_id": "snapshot", "num_kv_tokens": 2, "durability": durability, "segments": [],
            "session_id": "session", "generation": 1})
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(saves=[transfer])
        worker.wait_for_save()
        return transfer

    def test_d2h_failure_stops_the_engine_and_keeps_gpu_blocks_and_pinned_memory(self):
        worker = connector(self.module)
        allocation = Mock()
        worker.dataplane.pool.reserve.return_value = allocation
        worker.dataplane.save_gather.side_effect = RuntimeError("CUDA error: unspecified launch failure")
        self.save(worker)
        with self.assertRaises(self.module.DMAUnknown):
            worker.get_finished({"request"})
        self.assertEqual(worker.finished, set())
        allocation.close.assert_not_called()

    def test_full_pinned_pool_fails_the_save_before_any_copy_and_frees_the_blocks(self):
        worker = connector(self.module)
        worker.dataplane.pool.reserve.side_effect = PinnedCapacityError("pinned capacity exhausted")
        self.save(worker)
        self.assertEqual(worker.get_finished({"request"}), ({"request"}, set()))
        worker.volatile.reconcile.assert_called_once()          # one retry after the sweep
        worker.dataplane.save_gather.assert_not_called()
        self.assertIn("PinnedCapacityError", worker.build_connector_worker_meta().errors["request"])

    def test_gpu_blocks_are_freed_at_the_d2h_fence_while_hashing_continues(self):
        worker = connector(self.module)
        worker.executor = self.module.ThreadPoolExecutor(max_workers=1)
        self.addCleanup(worker.executor.shutdown)
        staged = Mock(read_range=lambda offset, size: b"k" * size)
        worker.dataplane.pool.reserve.return_value = staged
        release = threading.Event()
        worker.writer.prepare_blocks.side_effect = lambda blocks, durability: (
            release.wait(5), {"preparation_id": "p", "segments": []})[1]
        fenced = threading.Event()
        worker.dataplane.save_gather.side_effect = lambda *args: fenced.set()
        self.save(worker)
        self.assertTrue(fenced.wait(5))
        _, done = worker.save_jobs["request"]
        self.assertTrue(done.wait(5))
        self.assertEqual(worker.get_finished({"request"}), ({"request"}, set()))
        self.assertEqual(worker.build_connector_worker_meta().prepared, {})
        release.set()
        worker.save_jobs["request"][0].result(timeout=5)
        self.assertEqual(worker.get_finished(set()), (set(), set()))
        self.assertEqual(worker.build_connector_worker_meta().prepared, {"request": {"preparation_id": "p", "segments": []}})
        staged.close.assert_called_once()

    def test_save_copies_only_after_this_steps_kernels(self):
        worker = connector(self.module)
        order = Mock()
        worker.dataplane.producer_event.return_value = order.ready
        worker.dataplane.save_gather.side_effect = lambda *args, **kwargs: order.copy()
        worker.writer.prepare_blocks.return_value = {"preparation_id": "p", "segments": []}
        self.save(worker)
        self.assertEqual(order.mock_calls[:2], [call.ready.synchronize(), call.copy()])

    def test_known_finished_keeps_only_requests_with_saves(self):
        worker = connector(self.module)
        worker.get_finished({"never-saved", "other"})
        self.assertEqual(worker.known_finished, set())


class WorkerLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = import_connector()

    def load(self, worker):
        transfer = self.module.Transfer("request", [11], {"snapshot_id": "snapshot", "num_kv_tokens": 1,
                                        "durability": "durable", "segments": [{}]}, lease_id="lease")
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[transfer])
        worker.start_load_kv(None)
        return transfer

    def test_async_load_reports_the_request_once_its_copies_finish(self):
        worker = connector(self.module)
        self.load(worker)
        self.assertEqual(worker.get_finished(set()), (set(), {"request"}))
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
        self.assertEqual(worker.build_connector_worker_meta().loaded_leases, ["lease"])

    def in_flight(self, worker, launched):
        """Start a load whose copy is still running; ``launched`` is [(event done?, bytes left after it)]."""
        jobs = []
        worker.loader = SimpleNamespace(submit=lambda fn, *args: jobs.append(args) or Future())
        self.load(worker)
        progress = jobs[0][2]
        events = []
        progress.append((None, 300 << 20))
        for done, left in launched:
            events.append(Mock(query=Mock(return_value=done)))
            progress.append((events[-1], left))
        return worker.load_jobs["request"][0], events

    def test_a_load_is_reported_once_its_copy_has_little_left_and_the_next_forward_waits(self):
        worker = connector(self.module)
        mib = 1 << 20
        future, events = self.in_flight(worker, [(True, 200 * mib), (True, 100 * mib), (False, 0)])
        self.assertEqual(worker.get_finished(set()), (set(), {"request"}))
        self.assertEqual(worker.forward_waits, [events[-1]])
        self.assertEqual(worker.worker_meta.loaded_leases, [])        # the lease outlives the copy
        worker.bind_connector_metadata(self.module.KvsnapMetadata())
        worker.dataplane.wait_on_engine_stream.assert_called_once_with(events[-1])
        self.assertEqual(worker.forward_waits, [])
        self.assertEqual(worker.get_finished(set()), (set(), set()))  # not reported twice
        future.set_result(3)
        self.assertEqual(worker.get_finished(set()), (set(), set()))
        self.assertEqual(worker.worker_meta.loaded_leases, ["lease"])
        self.assertEqual(worker.load_jobs, {})

    def test_a_load_with_much_left_or_launches_still_queuing_is_not_reported(self):
        mib = 1 << 20
        for launched in ([(True, 200 * mib), (False, 100 * mib), (False, 0)],     # 200 MiB left
                         [(True, 100 * mib), (False, 50 * mib)]):                  # not every launch queued
            with self.subTest(launched=launched):
                worker = connector(self.module)
                future, _ = self.in_flight(worker, launched)
                self.assertEqual(worker.get_finished(set()), (set(), set()))
                self.assertEqual(worker.forward_waits, [])
                future.set_result(3)
                self.assertEqual(worker.get_finished(set()), (set(), {"request"}))

    def test_a_load_that_fails_after_it_was_reported_stops_the_engine(self):
        worker = connector(self.module)
        future, _ = self.in_flight(worker, [(True, 0)])
        self.assertEqual(worker.get_finished(set()), (set(), {"request"}))
        future.set_exception(RuntimeError("pinned release failed"))
        with self.assertRaises(self.module.DMAUnknown):
            worker.get_finished(set())

    def test_h2d_failure_stops_the_engine_and_keeps_the_lease(self):
        for async_load in (True, False):
            with self.subTest(async_load=async_load):
                worker = connector(self.module, async_load=async_load)
                worker.dataplane.load_scatter.side_effect = RuntimeError("CUDA error")
                with self.assertRaises(self.module.DMAUnknown):
                    self.load(worker)
                    worker.get_finished(set())
                self.assertEqual(worker.get_block_ids_with_load_errors(), set())
                self.assertEqual(worker.worker_meta.loaded_leases, [])

    def test_corrupt_block_before_any_copy_is_recomputed(self):
        worker = connector(self.module)
        worker.writer.read_prepared_into.side_effect = CorruptSnapshotError("CRC32C mismatch")
        self.load(worker)
        self.assertEqual(worker.get_finished(set()), (set(), {"request"}))
        self.assertEqual(worker.get_block_ids_with_load_errors(), {11})
        produced = worker.build_connector_worker_meta()
        self.assertEqual(produced.loaded_leases, ["lease"])
        self.assertEqual(produced.invalid_snapshots, ["snapshot"])
        self.assertEqual(produced.failed_loads, ["request"])
        worker.dataplane.load_scatter.assert_not_called()

    def test_an_async_load_waits_for_the_work_queued_before_the_forward_not_the_forward(self):
        for async_load in (True, False):
            with self.subTest(async_load=async_load):
                worker = connector(self.module, async_load=async_load)
                order = Mock()
                worker.dataplane.producer_event.side_effect = lambda: order.event(len(order.mock_calls))
                worker.dataplane.load_scatter.side_effect = lambda *args, **kwargs: order.copy()
                transfer = self.module.Transfer("request", [11], {"snapshot_id": "snapshot", "num_kv_tokens": 1,
                                                "durability": "durable", "segments": [{}]}, lease_id="lease")
                worker.bind_connector_metadata(self.module.KvsnapMetadata(loads=[transfer]))
                order.forward()        # an async load starts after the forward; a sync one before it
                worker.start_load_kv(None)
                recorded = [c for c in order.mock_calls if c[0] == "event"]
                # The async load's event was recorded before the forward (no calls yet), the sync one after.
                self.assertEqual(recorded, [call.event(0 if async_load else 1)])
                self.assertEqual(order.mock_calls[-2:][1], call.copy())
                worker.bind_connector_metadata(self.module.KvsnapMetadata())
                self.assertEqual(worker.dataplane.producer_event.call_count, 1)   # no loads, no event

    def test_load_copies_only_after_the_engine_streams_queued_work(self):
        worker = connector(self.module)
        order = Mock()
        worker.dataplane.producer_event.return_value = order.ready
        worker.dataplane.load_scatter.side_effect = lambda *args, **kwargs: order.copy()
        self.load(worker)
        self.assertEqual(order.mock_calls, [call.ready.synchronize(), call.copy()])


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = import_connector()

    def resume_request(self, identity=None):
        return SimpleNamespace(request_id="request", prompt_token_ids=[1, 2, 3],
                               kv_transfer_params={"kvsnap": {"resume_snapshot_id": "snapshot", "generation": 2,
                                                           "identity": identity or {"kv_layout": "[]"}}})

    def test_identity_mismatch_is_a_miss_checked_before_lookup(self):
        c = connector(self.module)
        c._vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(model="model", revision="actual-revision",
                tokenizer_revision="tokenizer-revision", dtype="torch.bfloat16",
                max_model_len=4096, hf_text_config=SimpleNamespace(to_dict=lambda: {"rope_theta": 10000}),
                get_num_layers=lambda _: 28, get_num_kv_heads=lambda _: 8, get_head_size=lambda: 128),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=16), parallel_config=None)
        del c._check_engine_identity
        claimed = runtime_identity(c._vllm_config, "layout").to_dict()
        claimed["model_revision"] = "previous-revision"
        self.assertEqual(c.get_num_new_matched_tokens(self.resume_request(claimed), 0), (0, False))
        c.store.get_manifest.assert_not_called()

    def test_evicted_snapshot_is_a_miss(self):
        c = connector(self.module)
        c.store.get_manifest.side_effect = CorruptSnapshotError("snapshot is Invalid: evicted")
        self.assertEqual(c.get_num_new_matched_tokens(self.resume_request(), 0), (0, False))

    def test_match_is_loaded_asynchronously_under_a_lease(self):
        c = connector(self.module)
        manifest = {"snapshot_id": "snapshot", "num_kv_tokens": 2, "durability": "durable",
                    "session_id": "session", "generation": 1, "segments": [{"length": 4}] * 2}
        c.store.get_manifest.return_value = manifest
        c.store.acquire_lease.return_value = "lease"
        with patch.object(self.module, "matched_tokens", return_value=2):
            self.assertEqual(c.get_num_new_matched_tokens(self.resume_request(), 0), (2, True))
        c.update_state_after_alloc(self.resume_request(), SimpleNamespace(get_block_ids=lambda: ([5, 6, 7],)), 2)
        [transfer] = c.loads
        self.assertEqual((transfer.block_ids, transfer.lease_id, transfer.error), ([5, 6], "lease", None))
        self.assertEqual(c.matched, {})
        self.assertEqual(c.parked, {"request"})

    def test_snapshot_lost_between_match_and_allocation_becomes_a_failed_load(self):
        c = connector(self.module)
        c.store.get_manifest.return_value = {"snapshot_id": "snapshot", "num_kv_tokens": 1, "durability": "volatile",
                                             "session_id": "session", "generation": 1, "segments": [{"length": 4}]}
        c.store.acquire_lease.return_value = "lease"
        c.store.volatile_locations.side_effect = CorruptSnapshotError("no worker-owned allocation")
        with patch.object(self.module, "matched_tokens", return_value=1):
            c.get_num_new_matched_tokens(self.resume_request(), 0)
        c.update_state_after_alloc(self.resume_request(), SimpleNamespace(get_block_ids=lambda: ([5],)), 1)
        [transfer] = c.loads
        self.assertIsNone(transfer.lease_id)
        self.assertIn("no worker-owned allocation", transfer.error)
        c.store.release_lease.assert_called_once_with("lease")

    def test_a_request_whose_load_failed_misses_when_it_is_rescheduled(self):
        c = connector(self.module)
        c.update_connector_output(meta(self.module, failed_loads=["request"]))
        self.assertEqual(c.get_num_new_matched_tokens(self.resume_request(), 0), (0, False))
        c.store.get_manifest.assert_not_called()

    def test_save_and_commit_failures_are_recorded_for_the_gateway(self):
        c = connector(self.module)
        transfer = self.module.Transfer("a", [1], {"snapshot_id": "sa", "generation": 1})
        other = self.module.Transfer("b", [2], {"snapshot_id": "sb", "generation": 1})
        c.saving = {"a": transfer, "b": other}
        c.store.commit_prepared.side_effect = StoreError("preparation differs from intent or expired")
        c.update_connector_output(meta(self.module, errors={"a": "OSError: disk full"},
                                       prepared={"b": {"preparation_id": "p"}}))
        c.store.record_save_failure.assert_has_calls([call("sa", "OSError: disk full"),
                                                      call("sb", "commit failed: preparation differs from intent or expired")],
                                                     any_order=True)
        self.assertEqual(c.saving, {})

    def test_commit_runs_off_the_scheduler_thread_and_keeps_the_engine_stepping(self):
        from concurrent.futures import ThreadPoolExecutor
        c = connector(self.module)
        c.committer = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(c.committer.shutdown)
        started, release = threading.Event(), threading.Event()
        def slow_commit(*args, **kwargs):
            started.set()
            release.wait(5)
        c.store.commit_prepared.side_effect = slow_commit
        c.saving = {"request": self.module.Transfer("request", [11], {"snapshot_id": "s", "generation": 1})}
        c.update_connector_output(meta(self.module, prepared={"request": {"preparation_id": "p"}}))
        self.assertTrue(started.wait(5))
        self.assertTrue(c.has_pending_push_work())     # vLLM keeps stepping while it commits
        release.set()
        for future in list(c.committing):
            future.result(5)
        c.update_connector_output(meta(self.module))
        self.assertFalse(c.has_pending_push_work())

    def test_unexpected_commit_failure_stops_the_engine_at_the_next_step(self):
        c = connector(self.module)
        c.saving = {"request": self.module.Transfer("request", [11], {"snapshot_id": "s", "generation": 1})}
        c.store.commit_prepared.side_effect = RuntimeError("fixture store bug")
        c.update_connector_output(meta(self.module, prepared={"request": {"preparation_id": "p"}}))
        with self.assertRaisesRegex(RuntimeError, "fixture store bug"):
            c.update_connector_output(meta(self.module))

    def test_stale_save_completion_is_discarded_without_a_failure(self):
        c = connector(self.module)
        c.saving = {"request": self.module.Transfer("request", [11], {"snapshot_id": "s", "generation": 1})}
        c.store.commit_prepared.side_effect = StoreGenerationConflict("stale snapshot generation")
        c.update_connector_output(meta(self.module, prepared={"request": {"preparation_id": "p"}}))
        self.assertEqual(c.saving, {})
        c.store.record_save_failure.assert_not_called()

    def test_host_memory_save_evicts_least_recently_used_snapshots_first(self):
        c = connector(self.module)
        c.block_size = 4
        c.store.make_room_volatile.return_value = ["old"]
        request = SimpleNamespace(request_id="r", all_token_ids=[1, 2, 3, 4, 5], output_token_ids=[5],
                                  num_computed_tokens=4, num_output_placeholders=0,
                                  status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"),
                                  kv_transfer_params={"kvsnap": {"op": "save", "snapshot_id": "s", "identity": {}}})
        manifest = {"snapshot_id": "s", "num_kv_tokens": 4, "durability": "volatile", "token_ids": [1, 2, 3, 4, 5],
                    "pending_tokens": [5], "session_id": "x", "generation": 1,
                    "identity": {"kv_layout": '[{"bytes": 1000}, {"bytes": 24}]'}}
        with patch.object(self.module, "finish_manifest", return_value=manifest):
            self.assertTrue(c.request_finished(request, [9])[0])
        c.store.make_room_volatile.assert_called_once_with(1024, 1 << 20)


if __name__ == "__main__":
    unittest.main()
