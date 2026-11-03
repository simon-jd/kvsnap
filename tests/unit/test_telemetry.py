"""Event schema and control-boundary tests, against a mock CUDA data plane."""
from concurrent.futures import Future
import errno
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from kvsnap.connector.telemetry import EventRecorder
from kvsnap.store import GenerationConflict


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_telemetry_connector_fixture', ROOT / 'tests/connector/test_failure_policy.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class RecorderTests(unittest.TestCase):
    def test_disabled_has_no_clock_lock_json_or_filesystem_work(self):
        recorder = EventRecorder(directory='')
        with patch('kvsnap.connector.telemetry.time.time_ns', side_effect=AssertionError('clock')), \
                patch('kvsnap.connector.telemetry.json.dumps', side_effect=AssertionError('json')), \
                patch('kvsnap.connector.telemetry.os.open', side_effect=AssertionError('filesystem')):
            self.assertFalse(recorder.emit('event'))
        self.assertIsNone(recorder._lock)

    def test_real_subprocesses_write_distinct_metadata_only_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            script = "from kvsnap.connector.telemetry import EventRecorder; r=EventRecorder('runtime',role='worker'); r.emit('cpu_schema_fixture', request_id='request', bytes=4); r.close()"
            env = dict(os.environ, KVSNAP_EVENT_DIR=directory)
            processes = [subprocess.Popen([sys.executable, '-c', script], cwd=ROOT, env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
            for process in processes:
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stderr.decode())
            paths = list(Path(directory).glob('*.jsonl'))
            self.assertEqual(len(paths), 2)
            records = [json.loads(path.read_text()) for path in paths]
            self.assertEqual(len({record['pid'] for record in records}), 2)
            for record, path in zip(records, paths):
                self.assertEqual(record['schema_version'], 1)
                self.assertEqual(record['evidence_tier'], 'runtime')
                self.assertEqual(record['runtime_id'], 'runtime')
                self.assertEqual(record['time_ns'], record['timestamp_ns'])
                self.assertGreater(record['monotonic_ns'], 0)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertNotIn('token_ids', record)
                self.assertNotIn('prompt', record)

    def test_write_failure_is_sticky_and_explicit_without_raising_into_dma(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EventRecorder('runtime', directory=directory)
            self.assertTrue(recorder.emit('started'))
            diagnostics = []
            def fail_write(fd, data):
                if fd == 2:
                    diagnostics.append(bytes(data))
                    return len(data)
                raise OSError(errno.ENOSPC, 'fixture full disk')
            with patch('kvsnap.connector.telemetry.os.write', side_effect=fail_write):
                self.assertFalse(recorder.emit('completed'))
                self.assertFalse(recorder.emit('later'))
            self.assertEqual(recorder.error['errno'], errno.ENOSPC)
            self.assertEqual(len(diagnostics), 1)
            self.assertIn(b'KVSNAP_TELEMETRY_ERROR', diagnostics[0])
            self.assertEqual([json.loads(line)['event'] for line in recorder.path.read_text().splitlines()], ['started'])
            recorder.close()


class ConnectorEventBoundaries(unittest.TestCase):
    """Events mark real completion boundaries and never steer DMA or lease control."""

    @classmethod
    def setUpClass(cls):
        cls.module = fixture.import_connector()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        worker = fixture.connector(self.module)
        worker.telemetry = EventRecorder('runtime', role='worker', directory=self.temp.name)
        worker.dataplane.load_scatter.return_value = 1   # copy calls issued
        self.worker = worker
        self.transfer = self.module.Transfer('request', [11, 12],
            {'snapshot_id': 'snapshot', 'num_kv_tokens': 3, 'durability': 'durable',
             'generation': 0, 'session_id': 'session', 'segments': [{}, {}]}, skip_blocks=1, lease_id='lease')
        self.prepared = {'preparation_id': 'preparation', 'segments': [{'length': 4}, {'length': 4}]}

    def tearDown(self):
        self.worker.telemetry.close()
        self.temp.cleanup()

    def records(self):
        path = self.worker.telemetry.path
        return [] if path is None else [json.loads(line) for line in path.read_text().splitlines()]

    def names(self):
        return [record['event'] for record in self.records()]

    def test_h2d_completed_is_after_fenced_load_and_uses_actual_transfer_range(self):
        worker, transfer = self.worker, self.transfer
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[transfer])
        def fenced_load(ids, sources):
            self.assertEqual(self.names(), ['h2d_started'])
            self.assertEqual(ids, [12])
            return 1
        worker.dataplane.load_scatter.side_effect = fenced_load
        worker.start_load_kv(None)
        # Only the blocks after the local prefix are read, into one staging buffer.
        worker.dataplane.staging.assert_called_once_with(1)
        self.assertEqual(worker.writer.read_prepared_into.call_args.kwargs['first_block'], 1)
        self.assertEqual(self.names(), ['h2d_started', 'h2d_completed'])
        start, end = self.records()
        self.assertEqual(start['operation_id'], end['operation_id'])
        self.assertEqual((end['block_ids'], end['skip_blocks'], end['bytes'], end['copies']), ([12], 1, 4, 1))
        self.assertEqual(worker.get_finished(set()), (set(), {'request'}))
        self.assertEqual(worker.worker_meta.loaded_leases, ['lease'])

    def test_a_load_records_when_its_copies_could_start_and_when_it_was_reported(self):
        worker = self.worker
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[self.transfer])
        worker.start_load_kv(None)
        self.assertEqual(self.names(), ['h2d_started', 'h2d_completed'])
        started, completed = self.records()
        self.assertLessEqual(started['monotonic_ns'], completed['ready_ns'])
        self.assertLessEqual(completed['ready_ns'], completed['monotonic_ns'])
        worker.get_finished(set())
        self.assertEqual(self.names()[-1], 'load_reported')
        self.assertEqual(self.records()[-1]['operation_id'], started['operation_id'])

    def test_the_scheduler_records_when_a_parked_load_is_scheduled_once(self):
        scheduler = self.worker
        scheduler.telemetry = self.worker.telemetry
        scheduler.parked = {'request'}
        scheduler.build_connector_meta(SimpleNamespace(num_scheduled_tokens={'other': 1},
                                                       total_num_scheduled_tokens=1))
        self.assertEqual(self.names(), [])
        for _ in range(2):
            scheduler.build_connector_meta(SimpleNamespace(num_scheduled_tokens={'request': 64, 'other': 1},
                                                           total_num_scheduled_tokens=65))
        self.assertEqual(self.names(), ['load_scheduled'])
        self.assertEqual(self.records()[0]['scheduled_tokens'], 64)
        self.assertEqual(scheduler.parked, set())

    def test_h2d_failure_never_emits_completed_or_releases_lease(self):
        worker = self.worker
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[self.transfer])
        worker.dataplane.load_scatter.side_effect = RuntimeError('fixture CUDA wait failed')
        worker.start_load_kv(None)
        with self.assertRaisesRegex(RuntimeError, 'completion could not be proven'):
            worker.get_finished(set())
        self.assertEqual(self.names(), ['h2d_started'])
        self.assertEqual(worker.worker_meta.loaded_leases, [])

    def test_d2h_preparation_and_manifest_events_follow_separate_completion_boundaries(self):
        worker, transfer = self.worker, self.transfer
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(saves=[transfer])
        worker.dataplane.pool.reserve.return_value = Mock(read_range=lambda offset, size: b'k' * size)
        def fenced_save(*args):
            self.assertEqual(self.names(), [])          # the inline job runs before the hand-off event
        worker.dataplane.save_gather.side_effect = fenced_save
        worker.writer.prepare_blocks.return_value = self.prepared
        worker.wait_for_save()
        self.assertEqual(self.names(), ['d2h_completed', 'preparation_completed', 'save_handed_off'])
        ready, receiving = worker.get_finished({'request'})
        self.assertEqual((ready, receiving), ({'request'}, set()))
        self.assertEqual(self.names()[-1], 'save_free_authorized')
        worker.saving = {'request': transfer}
        def commit(*args, **kwargs):
            self.assertNotIn('manifest_committed', self.names())
        worker.store.commit_prepared.side_effect = commit
        worker.update_connector_output(SimpleNamespace(kv_connector_worker_meta=worker.build_connector_worker_meta()))
        self.assertEqual(self.names()[-1], 'manifest_committed')

    def test_volatile_preparation_failure_after_fenced_d2h_is_reported_not_prepared(self):
        worker, transfer = self.worker, self.transfer
        transfer.manifest['durability'] = 'volatile'
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(saves=[transfer])
        worker.dataplane.pool.reserve.side_effect = lambda size: Mock()
        def failed_prepare(source, **kwargs):
            self.assertEqual(len(source), 2)
            self.assertEqual(self.names()[-1], 'd2h_completed')
            raise OSError('fixture intent failed')
        worker.volatile.prepare_allocations.side_effect = failed_prepare
        worker.wait_for_save()
        ready, _ = worker.get_finished({'request'})
        # The copy is fenced, so the GPU blocks may be freed; the save is reported failed.
        self.assertEqual(ready, {'request'})
        self.assertIn('fixture intent failed', worker.worker_meta.errors['request'])
        self.assertNotIn('request', worker.worker_meta.prepared)
        self.assertNotIn('preparation_completed', self.names())

    def test_telemetry_failure_after_h2d_does_not_change_dma_or_lease_control(self):
        worker = self.worker
        worker._get_connector_metadata = lambda: self.module.KvsnapMetadata(loads=[self.transfer])
        original_write = os.write
        def fail_completed(fd, data):
            if b'h2d_completed' in bytes(data):
                raise OSError(errno.ENOSPC, 'fixture telemetry full')
            if fd == 2:
                return len(data)
            return original_write(fd, data)
        with patch('kvsnap.connector.telemetry.os.write', side_effect=fail_completed):
            worker.start_load_kv(None)
            self.assertEqual(worker.get_finished(set()), (set(), {'request'}))
        self.assertEqual(self.names(), ['h2d_started'])
        self.assertEqual(worker.telemetry.error['errno'], errno.ENOSPC)
        self.assertEqual(worker.worker_meta.loaded_leases, ['lease'])
        self.assertEqual(worker.load_errors, set())

    def test_stale_generation_never_emits_manifest_commit(self):
        worker = self.worker
        worker.saving = {'request': self.transfer}
        worker.store.commit_prepared.side_effect = GenerationConflict('stale')
        worker.update_connector_output(SimpleNamespace(kv_connector_worker_meta=
            self.module.KvsnapWorkerMetadata(prepared={'request': self.prepared})))
        self.assertEqual(self.names(), [])