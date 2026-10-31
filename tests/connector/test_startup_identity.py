"""Worker registration -> fresh runtime identity handshake; no actual engine."""
from dataclasses import replace
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_failure_policy import connector as make_connector, import_connector
from kvsnap.connector.runtime import identity_path, publish_runtime_identity, wait_runtime_identity
from kvsnap.gateway.engine import fixture_identity
from kvsnap.models import IdentityMismatch


class StartupIdentity(unittest.IsolatedAsyncioTestCase):
    async def test_worker_registration_publishes_actual_layout_before_gateway_uses_identity(self):
        module = import_connector()
        with tempfile.TemporaryDirectory() as directory:
            worker = make_connector(module)
            worker.root, worker.runtime_id = directory, "a" * 32
            worker.block_size, worker.capacity, worker._vllm_config = 4, 4096, object()
            layers = [{"name": "layer.0", "shape": [3, 4, 8], "stride": [8, 24, 1],
                       "dtype": "torch.bfloat16", "bytes": 192}]
            with patch.object(module, "TorchBlockDataPlane", return_value=SimpleNamespace(layers=layers, block_bytes=4)), \
                 patch.object(module, "VolatileOwner", return_value=Mock()), \
                 patch.object(module, "runtime_identity", side_effect=lambda config, layout: replace(fixture_identity(4), kv_layout=layout)):
                worker.register_kv_caches({"layer.0": object()})
            resolved = await wait_runtime_identity(directory, "a" * 32, timeout=0.2)
            self.assertEqual(json.loads(resolved.kv_layout), layers)
            with self.assertRaises(IdentityMismatch):
                await wait_runtime_identity(directory, "a" * 32, timeout=0.2,
                                            expected=replace(resolved, model_revision="old-model-revision"))

    async def test_different_runtime_nonce_cannot_reuse_previous_worker_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            old = publish_runtime_identity(directory, "a" * 32, fixture_identity())
            identity_path(directory, "b" * 32).write_text(old.read_text())
            with self.assertRaisesRegex(ValueError, "does not match this engine"):
                await wait_runtime_identity(directory, "b" * 32, timeout=0.2)

    async def test_dead_publisher_is_rejected_even_with_valid_identity_json(self):
        with tempfile.TemporaryDirectory() as directory:
            publish_runtime_identity(directory, "a" * 32, fixture_identity())
            with patch("kvsnap.connector.runtime.os.kill", side_effect=ProcessLookupError):
                with self.assertRaisesRegex(RuntimeError, "no longer alive"):
                    await wait_runtime_identity(directory, "a" * 32, timeout=0.2)

    async def test_missing_fresh_runtime_times_out_instead_of_selecting_an_older_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            publish_runtime_identity(directory, "a" * 32, fixture_identity())
            with self.assertRaisesRegex(TimeoutError, "did not publish"):
                await wait_runtime_identity(directory, "b" * 32, timeout=0.001)


if __name__ == "__main__":
    unittest.main()
