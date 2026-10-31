"""Runtime construction stubs, with real CPU disk admission at the endpoint."""
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_failure_policy import import_connector
from kvsnap.connector.engine import create_async_llm
from kvsnap.store import CapacityError


class ProductionCapacity(unittest.TestCase):
    def test_user_budget_reaches_scheduler_and_real_worker_disk_admission(self):
        modules = {}
        for name, attribute, value in (
            ("vllm.engine.arg_utils", "AsyncEngineArgs", SimpleNamespace),
            ("vllm.config", "KVTransferConfig", SimpleNamespace),
            ("vllm.v1.engine.async_llm", "AsyncLLM", SimpleNamespace(
                from_engine_args=Mock(side_effect=lambda args: SimpleNamespace(args=args)))),
        ):
            module = ModuleType(name)
            setattr(module, attribute, value)
            modules[name] = module
        connector = import_connector()

        def base_init(instance, config, role, cache):
            instance._kv_transfer_config = config.kv_transfer_config
            instance._vllm_config = config

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {"KVSNAP_IO_BACKEND": "python", "KVSNAP_EVENT_DIR": ""}), \
             patch.object(connector.KVConnectorBase_V1, "__init__", base_init):
            runtime = create_async_llm(directory, pinned_pool_bytes=512, disk_capacity_bytes=128)
            config = SimpleNamespace(kv_transfer_config=runtime.args.kv_transfer_config,
                                     parallel_config=SimpleNamespace(tensor_parallel_size=1),
                                     scheduler_config=SimpleNamespace(async_scheduling=False),
                                     cache_config=SimpleNamespace(block_size=16))
            cache = SimpleNamespace(kv_cache_groups=[object()])
            scheduler = connector.KvsnapConnector(config, connector.KVConnectorRole.SCHEDULER, cache)
            worker = connector.KvsnapConnector(config, connector.KVConnectorRole.WORKER, cache)
            try:
                self.assertEqual(scheduler.store.host_capacity_bytes, 512)
                self.assertEqual(scheduler.store.disk_capacity_bytes, 128)
                self.assertEqual(worker.capacity, 512)
                self.assertEqual(worker.writer.disk_capacity_bytes, 128)
                with self.assertRaises(CapacityError):
                    worker.writer.prepare_blocks([b"x" * 129], "process")
                self.assertEqual(list((Path(directory) / "blobs").iterdir()), [])
                self.assertEqual(list((Path(directory) / "prepared").glob("*.json")), [])
            finally:
                worker.shutdown()
                scheduler.shutdown()


if __name__ == "__main__":
    unittest.main()
