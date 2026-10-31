import unittest
from kvsnap.connector.protocol import finish_manifest, matched_tokens
from kvsnap.gateway.engine import fixture_identity
from kvsnap.models import ContractError, SamplingConfig
from kvsnap.store import SnapshotStore
import tempfile


class ConnectorContractTests(unittest.TestCase):
    def instruction(self):
        return dict(snapshot_id="snapshot", session_id="session", generation=1,
                    identity=fixture_identity(4).to_dict(), seed=7,
                    sampling=SamplingConfig().to_dict(), durability="durable",
                    outputs=[{"seq": 1, "token_id": 8}], output_seq_start=1,
                    remaining_max_tokens=5)

    def test_settled_boundary_keeps_sampled_pending_token(self):
        saved = finish_manifest(self.instruction(), [1, 2, 8, 9, 10], 4, [9, 10])
        self.assertEqual(saved["pending_tokens"], [10])
        self.assertEqual(saved["remaining_max_tokens"], 3)
        self.assertEqual(saved["output_seq_committed"], 3)
        self.assertEqual(saved["outputs"][-1], {"seq": 3, "token_id": 10})

    def test_n_plus_one_can_load_n_including_partial_tail(self):
        saved = finish_manifest(self.instruction(), [1, 2, 8, 9], 3, [9])
        with tempfile.TemporaryDirectory() as directory, SnapshotStore(directory) as store:
            saved = store.commit(saved, [b"one-tail-block"])
            self.assertEqual(matched_tokens(saved, saved["identity"], [1, 2, 8, 9], 0), 3)
            self.assertEqual(matched_tokens(saved, saved["identity"], [1, 2, 8, 9], 3), 0)
            for bad in ([1, 2, 7, 9], [1, 2, 8]):
                with self.assertRaises(ContractError):
                    matched_tokens(saved, saved["identity"], bad, 0)
            changed = dict(saved["identity"], kv_dtype="other")
            with self.assertRaises(ContractError):
                matched_tokens(saved, changed, [1, 2, 8, 9], 0)

    def test_unsettled_or_inconsistent_metadata_rejected(self):
        for tokens, computed, output in (([1], 2, []), ([1], -1, []), ([1, 2], 1, [3])):
            with self.assertRaises(ContractError):
                finish_manifest(self.instruction(), tokens, computed, output)
        bad = self.instruction(); bad["output_seq_start"] = 5
        with self.assertRaises(ContractError):
            finish_manifest(bad, [1, 2], 1, [2])


if __name__ == "__main__":
    unittest.main()
