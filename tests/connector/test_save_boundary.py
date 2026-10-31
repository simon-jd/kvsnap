"""Scheduler-side save bookkeeping: the settled token boundary and per-step metadata."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_failure_policy import connector as make_connector, import_connector
from kvsnap.connector.protocol import finish_manifest
from kvsnap.gateway.engine import fixture_identity
from kvsnap.models import SamplingConfig


class ConnectorLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = import_connector()

    def test_request_finished_captures_pending_sample_with_terminal_stop_flag(self):
        connector = make_connector(self.module)
        connector.block_size = 4
        instruction = dict(op="save", snapshot_id="s", session_id="session", generation=1,
                           identity=fixture_identity(4).to_dict(), seed=9,
                           sampling=SamplingConfig().to_dict(), durability="process",
                           outputs=[{"seq": 1, "token_id": 8}], output_seq_start=1,
                           remaining_max_tokens=4)
        request = SimpleNamespace(request_id="request", kv_transfer_params={"kvsnap": instruction},
                                  all_token_ids=[1, 2, 8, 9, 10], output_token_ids=[9, 10],
                                  num_computed_tokens=4, num_output_placeholders=0,
                                  status=SimpleNamespace(name="FINISHED_STOPPED"))
        delay_free, result = connector.request_finished(request, [7, 8])
        self.assertTrue(delay_free)
        self.assertEqual(result["kvsnap_snapshot_id"], "s")
        transfer = connector.saves[0]
        self.assertEqual(transfer.block_ids, [7])
        self.assertEqual(transfer.manifest["pending_tokens"], [10])
        self.assertTrue(transfer.manifest["stopped"])
        self.assertEqual(transfer.manifest["remaining_max_tokens"], 2)
        self.assertEqual(transfer.manifest["outputs"], [
            {"seq": 1, "token_id": 8}, {"seq": 2, "token_id": 9}, {"seq": 3, "token_id": 10}])

    def test_current_step_metadata_is_drained_without_mutating_previous_step(self):
        connector = make_connector(self.module)
        transfer = self.module.Transfer("request", [3], {"snapshot_id": "s"})
        connector.saves, connector.loads = [transfer], []
        meta = connector.build_connector_meta(SimpleNamespace())
        self.assertEqual(meta.saves, [transfer])
        next_meta = connector.build_connector_meta(SimpleNamespace())
        self.assertEqual(next_meta.saves, [])
        self.assertEqual(meta.saves, [transfer])


if __name__ == "__main__":
    unittest.main()
