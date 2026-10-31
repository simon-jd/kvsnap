from types import SimpleNamespace
import unittest

from kvsnap.connector.identity import runtime_identity
from kvsnap.models import IdentityMismatch


def config():
    model = SimpleNamespace(model="Qwen/Qwen3-1.7B", revision="resolved-sha",
                            tokenizer_revision="resolved-tokenizer-sha", dtype="torch.bfloat16",
                            max_model_len=32768, hf_text_config=SimpleNamespace(
                                to_dict=lambda: {"rope_theta": 10000, "hidden_size": 2048}),
                            get_num_layers=lambda parallel: 28,
                            get_num_kv_heads=lambda parallel: 8,
                            get_head_size=lambda: 128)
    return SimpleNamespace(model_config=model, cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
                           parallel_config=SimpleNamespace(tensor_parallel_size=1))


class RuntimeIdentityTests(unittest.TestCase):
    def test_client_cannot_relabel_model_revision_dtype_or_rope(self):
        original = runtime_identity(config(), "actual-layout")
        changes = [
            lambda c: setattr(c.model_config, "model", "different-weights"),
            lambda c: setattr(c.model_config, "revision", "new-sha"),
            lambda c: setattr(c.model_config, "tokenizer_revision", "new-tokenizer"),
            lambda c: setattr(c.model_config, "max_model_len", 16384),
            lambda c: setattr(c.cache_config, "cache_dtype", "fp8"),
            lambda c: setattr(c.model_config.hf_text_config, "to_dict", lambda: {"rope_theta": 1000000, "hidden_size": 2048}),
        ]
        for change in changes:
            with self.subTest(change=change):
                changed = config(); change(changed)
                with self.assertRaises(IdentityMismatch):
                    original.require_match(runtime_identity(changed, "actual-layout"))

    def test_missing_revision_fails_closed(self):
        changed = config(); changed.model_config.revision = None
        with self.assertRaises(ValueError):
            runtime_identity(changed, "actual-layout")
