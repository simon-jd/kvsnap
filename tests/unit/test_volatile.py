"""CPU allocation and protocol tests of the host tier; no pinned CUDA memory is involved."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from kvsnap.connector.volatile import VolatileClient, VolatileOwner, _send_json, _receive_json
from kvsnap.store import CapacityError, CorruptSnapshotError, SnapshotStore, StoreError
from test_store import manifest


class FixturePool:
    def __init__(self, capacity):
        self.capacity, self.used, self.peak = capacity, 0, 0

    def reserve(self, value):
        if self.used + len(value) > self.capacity:
            raise CapacityError('CPU fixture pool full')
        self.used += len(value)
        self.peak = max(self.peak, self.used)
        return FixtureAllocation(self, value)


class FixtureAllocation:
    def __init__(self, pool, value):
        self.pool, self.size = pool, len(value)
        self.memory = ctypes.create_string_buffer(value, self.size)
        self.closed = False

    def read(self):
        if self.closed:
            raise RuntimeError('closed CPU fixture allocation')
        return self.memory.raw

    def close(self):
        if not self.closed:
            self.pool.used -= self.size
            self.closed = True
            self.memory = None


class LongSocketPathContracts(unittest.TestCase):
    def test_long_store_uses_real_authenticated_ipc_and_cleans_original_socket(self):
        with tempfile.TemporaryDirectory(prefix='kvsnap-long-socket-') as directory:
            root=Path(directory)/('long-store-'*7)/('nested-evidence-'*6)
            pool=FixturePool(8)
            with SnapshotStore(root) as store:
                owner=VolatileOwner(root,8,reconcile_interval=60)
                try:
                    self.assertGreater(len(os.fsencode(owner.path)),107)
                    self.assertTrue(stat.S_ISSOCK(owner.path.stat().st_mode))
                    prepared=owner.prepare_allocations([pool.reserve(b'AAAA'),pool.reserve(b'BBBB')],session_id='session',generation=0)
                    store.commit_prepared(manifest(durability='volatile'),prepared)
                    self.assertEqual(store.load('snapshot'),[b'AAAA',b'BBBB'])
                    code="import json,sys; from kvsnap.connector.volatile import VolatileClient; print(json.dumps(VolatileClient(sys.argv[1],json.loads(sys.argv[2])).request('stats')[0]))"
                    child=subprocess.run([sys.executable,'-B','-c',code,str(root),json.dumps(owner.descriptor)],
                                         capture_output=True,text=True,timeout=10,check=True)
                    self.assertEqual(json.loads(child.stdout)['resident_bytes'],8)
                    with self.assertRaises(CorruptSnapshotError):
                        VolatileClient(root,dict(owner.descriptor,token='0'*64)).request('stats')
                finally:
                    owner.close()
                self.assertFalse(owner.path.exists())
                self.assertEqual(pool.used,0)


class VolatileProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='kvsnap-v-')
        self.store = SnapshotStore(self.temp.name)
        self.owner = VolatileOwner(self.temp.name, 12, reconcile_interval=60)
        self.pool = FixturePool(12)

    def tearDown(self):
        self.owner.close()
        self.store.close()
        self.temp.cleanup()

    def prepare(self, values, **kwargs):
        return self.owner.prepare_allocations(
            (self.pool.reserve(value) for value in values), session_id='session', generation=0, **kwargs)

    def test_streaming_dedup_needs_only_one_block_headroom(self):
        first = self.prepare([b'AAAA', b'BBBB'])
        self.store.commit_prepared(manifest(durability='volatile'), first)
        second = self.prepare([b'AAAA', b'BBBB'])
        again = manifest(snapshot_id='again', durability='volatile')
        self.store.commit_prepared(again, second)
        self.assertEqual(self.pool.used, 8)
        self.assertEqual(self.pool.peak, 12)
        self.assertEqual(self.owner.stats()['blob_count'], 2)
        self.assertEqual(self.store.stats()['worker_host_bytes'], 8)
        self.assertEqual(self.store.load('again'), [b'AAAA', b'BBBB'])
        self.assertEqual(list(self.store.blob_dir.iterdir()), [])

    def test_resave_checks_committed_blocks_in_one_round_trip_per_owner(self):
        self.store.commit_prepared(manifest(durability='volatile'), self.prepare([b'AAAA', b'BBBB']))
        second = self.prepare([b'AAAA', b'BBBB'])
        calls, request = [], VolatileClient.request
        def counted(client, op, **args):
            calls.append((op, len(args.get('blocks', ()))))
            return request(client, op, **args)
        with patch.object(VolatileClient, 'request', counted):
            self.store.commit_prepared(manifest(snapshot_id='again', durability='volatile'), second)
        # The new locations, then every block already committed, at once.
        self.assertEqual(calls, [('check', 2), ('check', 2)])
        self.assertEqual(self.store.load('again'), [b'AAAA', b'BBBB'])

    def test_stale_committed_location_is_replaced_block_by_block(self):
        self.store.commit_prepared(manifest(durability='volatile'), self.prepare([b'AAAA', b'BBBB']))
        key = hashlib.sha256(b'AAAA').hexdigest()
        (stale,) = self.store.db.execute('SELECT location_json FROM volatile_locations WHERE blob_id=?', (key,)).fetchone()
        self.store.db.execute('UPDATE volatile_locations SET location_json=? WHERE blob_id=?',
                              (json.dumps(dict(json.loads(stale), handle='0' * 32)), key))
        self.store.commit_prepared(manifest(snapshot_id='again', durability='volatile'), self.prepare([b'AAAA', b'BBBB']))
        (location,) = self.store.db.execute('SELECT location_json FROM volatile_locations WHERE blob_id=?', (key,)).fetchone()
        self.assertEqual(json.loads(location)['handle'], json.loads(stale)['handle'])
        self.assertEqual(self.store.load('again'), [b'AAAA', b'BBBB'])

    def test_durable_copy_does_not_keep_unreferenced_volatile_allocation_alive(self):
        record = self.prepare([b'AAAA', b'BBBB'])
        self.store.commit_prepared(manifest(durability='volatile'), record)
        self.store.commit(manifest(snapshot_id='disk', session_id='disk-session'), [b'AAAA', b'BBBB'])
        self.store.delete_snapshot('snapshot')
        self.store.collect_garbage()
        self.owner.reconcile()
        self.assertEqual(self.pool.used, 0)
        self.assertEqual(self.store.stats()['worker_host_bytes'], 0)
        self.assertEqual(self.store.load('disk'), [b'AAAA', b'BBBB'])

    def test_inherited_owner_cannot_unlink_original_workers_socket(self):
        with patch('kvsnap.connector.volatile.os.getpid', return_value=self.owner._pid + 1):
            with self.assertRaises(StoreError):
                self.owner.close()
        self.assertTrue(self.owner.path.exists())
        self.assertEqual(VolatileClient(self.temp.name, self.owner.descriptor).request('stats')[0]['blob_count'], 0)

    def test_failed_intent_closes_every_new_allocation(self):
        with patch('kvsnap.connector.volatile._write_preparation', side_effect=OSError('no metadata space')):
            with self.assertRaises(OSError):
                self.prepare([b'AAAA', b'BBBB'])
        self.assertEqual(self.pool.used, 0)
        self.assertEqual(self.owner.stats()['blob_count'], 0)

    def test_expired_unpublished_intent_is_reclaimed_without_release_notification(self):
        self.prepare([b'AAAA', b'BBBB'], ttl_seconds=0.01)
        time.sleep(0.02)
        self.owner.reconcile()
        self.assertEqual(self.pool.used, 0)

    def test_server_survives_bad_request_then_authenticates_and_bounds_reads(self):
        record = self.prepare([b'AAAA', b'BBBB'])
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.connect(str(self.owner.path))
            _send_json(stream, [])
            self.assertFalse(_receive_json(stream)['ok'])
        client = VolatileClient(self.temp.name, self.owner.descriptor)
        self.assertEqual(client.request('stats')[0]['resident_bytes'], 8)
        wrong = dict(self.owner.descriptor, token='0' * 64)
        with self.assertRaises(CorruptSnapshotError):
            VolatileClient(self.temp.name, wrong).request('stats')
        key = hashlib.sha256(b'AAAA').hexdigest()
        location = record['locations'][key]
        for offset, length in [(-1, 1), (0, 9 * 1024**2), (3, 2), (False, 1)]:
            with self.subTest(offset=offset, length=length), self.assertRaises(CorruptSnapshotError):
                client.request('read', blob_id=key, handle=location['handle'], offset=offset, length=length)
        self.assertEqual(client.read(key, location), b'AAAA')
        with self.assertRaises(CorruptSnapshotError):
            VolatileClient(self.temp.name, dict(self.owner.descriptor, socket='../other.sock'))

    def test_corrupt_owner_payload_is_invalid_before_any_block_return(self):
        record = self.prepare([b'AAAA', b'BBBB'])
        self.store.commit_prepared(manifest(durability='volatile'), record)
        key = hashlib.sha256(b'BBBB').hexdigest()
        self.owner._entries[key]['allocation'].memory[0] = b'X'
        with self.assertRaises(CorruptSnapshotError):
            self.store.load('snapshot')
        self.assertEqual(self.store.get_manifest('snapshot', include_invalid=True)['state'], 'Invalid')
        self.owner.reconcile()
        self.assertEqual(self.pool.used, 0)
