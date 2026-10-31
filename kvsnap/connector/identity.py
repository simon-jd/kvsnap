"""Derive identity from resolved engine state, never from a client's assertion."""
import hashlib
from kvsnap.models import KVIdentity, canonical_json


def runtime_identity(config, kv_layout):
    model = config.model_config
    cache = config.cache_config
    hf = model.hf_text_config
    if not model.revision or not model.tokenizer_revision:
        raise ValueError("resolved model and tokenizer revisions are required")
    dtype = cache.cache_dtype
    if dtype == "auto":
        dtype = str(model.dtype).removeprefix("torch.")
    return KVIdentity(
        model_id=model.model, model_revision=model.revision,
        tokenizer_revision=model.tokenizer_revision, kv_dtype=dtype,
        block_size=cache.block_size, kv_layout=kv_layout,
        model_shape={"layers": model.get_num_layers(config.parallel_config),
                     "kv_heads": model.get_num_kv_heads(config.parallel_config),
                     "head_size": model.get_head_size()},
        position_encoding={"max_model_len": model.max_model_len,
                           "hf_config_sha256": hashlib.sha256(canonical_json(hf.to_dict()).encode()).hexdigest()},
        vllm_version="0.31.0",
    )
