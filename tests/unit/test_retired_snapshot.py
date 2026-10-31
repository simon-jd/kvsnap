"""CPU persistent-owner/lease contracts; no GPU fork claim."""
import tempfile
import unittest
from kvsnap.gateway.engine import fixture_identity
from kvsnap.models import SamplingConfig,SnapshotManifest
from kvsnap.store import SnapshotStore,StoreError


class RetiredSnapshotContracts(unittest.TestCase):
    def test_retiring_parent_or_sibling_keeps_child_bytes_and_dag(self):
        with tempfile.TemporaryDirectory() as directory, SnapshotStore(directory) as store:
            body=SnapshotManifest('parent','session',0,fixture_identity(2),(1,2,3,4),3,(4,),7,
                                  SamplingConfig(),32,0,durability='durable').to_dict()
            store.commit(body,[b'prefix',b'partial-tail'])
            store.fork('parent','left','left-session')
            store.fork('parent','right','right-session')
            store.retire_snapshot('parent')
            store.collect_garbage()
            self.assertEqual(store.load('left'),[b'prefix',b'partial-tail'])
            self.assertEqual(store.load('right'),[b'prefix',b'partial-tail'])
            self.assertEqual(store.get_manifest('parent',include_invalid=True)['state'],'Invalid')
            with self.assertRaises(StoreError):store.load('parent')
            store.retire_snapshot('left')
            store.collect_garbage()
            self.assertEqual(store.load('right'),[b'prefix',b'partial-tail'])
            self.assertEqual({b['refcount'] for b in store.export()['blobs']},{1})
            lease=store.acquire_lease('right')
            store.retire_snapshot('right')
            self.assertEqual(store.collect_garbage(),[])
            self.assertEqual(len(list(store.blob_dir.glob('*.blob'))),2)
            store.release_lease(lease)
            self.assertEqual(len(store.collect_garbage()),2)
            self.assertEqual(store.export()['leases'],[])
