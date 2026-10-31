import dataclasses
import math
import unittest

from kvsnap.gateway.engine import fixture_identity
from kvsnap.models import (ContractError, IdentityMismatch, KVIdentity, OutputToken,
                        SamplingConfig, SnapshotManifest)


class ModelTests(unittest.TestCase):
    def manifest(self, **overrides):
        kwargs = dict(snapshot_id="snapshot", session_id="session", generation=2,
                      identity=fixture_identity(), token_ids=(1, 2, 3), num_kv_tokens=2,
                      pending_tokens=(3,), seed=42, sampling=SamplingConfig(),
                      remaining_max_tokens=3, output_seq_committed=1, outputs=(OutputToken(1, 3),))
        kwargs.update(overrides)
        return SnapshotManifest(**kwargs)

    def test_round_trip(self):
        value = self.manifest()
        self.assertEqual(SnapshotManifest.from_dict(value.to_dict()), value)

    def test_pending_is_exact_suffix(self):
        for changes in ({"pending_tokens": ()}, {"num_kv_tokens": 4}, {"pending_tokens": (2, 3)}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.manifest(**changes)

    def test_reject_token_types_and_bad_budget(self):
        for changes in ({"token_ids": (True, 2, 3)}, {"remaining_max_tokens": -1},
                        {"seed": 2**63}, {"generation": 1.0}, {"output_seq_committed": 2}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.manifest(**changes)

    def test_strict_sampling_contract(self):
        for config in ({"presence_penalty": 0.1}, {"frequency_penalty": -0.1},
                       {"temperature": math.nan}, {"temperature": True}, {"top_p": 0},
                       {"top_k": 0}, {"min_p": 1.1}, {"logit_bias": {}}, {"n": 2}):
            with self.subTest(config=config), self.assertRaises(ContractError):
                SamplingConfig.from_dict(config)
        self.assertEqual(SamplingConfig.from_dict({"temperature": 0.8, "top_k": 40, "top_p": 0.9}).top_k, 40)

    def test_identity_covers_every_key(self):
        original = fixture_identity()
        for key, value in (("model_revision", "other"), ("tokenizer_revision", "other"),
                           ("kv_dtype", "fp8"), ("block_size", 32), ("kv_layout", "other"),
                           ("vllm_version", "another"), ("model_shape", {"layers": 2}),
                           ("position_encoding", {"max_model_len": 999, "type": "fixture"})):
            with self.subTest(key=key), self.assertRaises(IdentityMismatch):
                original.require_match(dataclasses.replace(original, **{key: value}))

    def test_identity_copies_caller_config(self):
        config = {"max_model_len": 10, "rope_scaling": {"factor": 2}}
        identity = dataclasses.replace(fixture_identity(), position_encoding=config)
        config["rope_scaling"]["factor"] = 4
        self.assertEqual(identity.position_encoding["rope_scaling"]["factor"], 2)

    def test_no_path_traversal(self):
        for value in ("../bad", "x/y", "x\\y", ""):
            with self.subTest(value=value), self.assertRaises(ContractError):
                self.manifest(snapshot_id=value)

    def test_unknown_manifest_schema(self):
        with self.assertRaises(ContractError):
            self.manifest(schema_version=2)
        with self.assertRaises(ContractError):
            SnapshotManifest.from_dict({**self.manifest().to_dict(), "unrecognized": True})

    def test_reject_committed_manifest_without_blocks(self):
        with self.assertRaises(ContractError):
            self.manifest(state="HostValid")

    def test_output_log_is_contiguous(self):
        with self.assertRaises(ContractError):
            self.manifest(outputs=(OutputToken(2, 3),))


if __name__ == "__main__":
    unittest.main()
