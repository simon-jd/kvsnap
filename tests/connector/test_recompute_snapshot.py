"""A failed external load may be recomputed without reviving corrupt KV.

These tests exercise real store publication with CPU payloads after the worker
has reported an invalid source. They do not execute an inference engine.
"""
import copy
import tempfile
import unittest

from kvsnap.connector.protocol import finish_manifest
from kvsnap.gateway.engine import fixture_identity
from kvsnap.models import SamplingConfig
from kvsnap.store import CorruptSnapshotError, SnapshotStore, StoreError


class RecomputedSnapshot(unittest.TestCase):
    def test_complete_recompute_can_commit_beneath_invalid_kv_parent(self):
        operation = dict(snapshot_id="source", session_id="session", generation=0,
                         identity=fixture_identity(4).to_dict(), seed=3,
                         sampling=SamplingConfig().to_dict(), durability="process",
                         outputs=[], output_seq_start=0, remaining_max_tokens=3)
        with tempfile.TemporaryDirectory() as directory, SnapshotStore(directory) as store:
            source = finish_manifest(operation, [1, 2, 3], 2, [3])
            store.commit(source, [b"original-source-kv"])
            store.claim_generation("session", 0, 1)
            # Mirrors update_connector_output(invalid_snapshots=["source"])
            # after start_load_kv reported these blocks for full recompute.
            store.invalidate_snapshot("source", "source bytes failed CRC before H2D")
            operation.update(snapshot_id="recomputed", generation=1, parent_id="source",
                             outputs=[{"seq": 1, "token_id": 3}], output_seq_start=1,
                             remaining_max_tokens=2)
            child = finish_manifest(operation, [1, 2, 3, 4], 3, [4])
            prepared = store.prepare_blocks([b"fresh-recomputed-kv"], "process")
            committed = store.commit_prepared(child, prepared, expected_generation=1)
            self.assertEqual(committed["parent_id"], "source")
            self.assertEqual(store.load("recomputed"), [b"fresh-recomputed-kv"])
            with self.assertRaises(CorruptSnapshotError):
                store.load("source")
            changed_prefix = copy.deepcopy(child)
            changed_prefix["snapshot_id"] = "changed-prefix"
            changed_prefix["token_ids"][0] = 99
            with self.assertRaises(StoreError):
                store.commit(changed_prefix, [b"different-kv"])


if __name__ == "__main__":
    unittest.main()
