"""Dependency-free scheduler contract used by the vLLM adapter."""
from __future__ import annotations

from kvsnap.models import ContractError, KVIdentity, SnapshotManifest


def matched_tokens(manifest, identity, prompt_tokens, num_computed_tokens, *, external_limit=None):
    snapshot = SnapshotManifest.from_dict(manifest)
    KVIdentity.from_dict(identity).require_match(snapshot.identity)
    if type(num_computed_tokens) is not int or num_computed_tokens < 0:
        raise ContractError("invalid local computed token count")
    if list(prompt_tokens[:len(snapshot.token_ids)]) != list(snapshot.token_ids):
        raise ContractError("resume must preserve the complete saved token prefix")
    if snapshot.num_kv_tokens >= len(prompt_tokens):
        raise ContractError("resume requires an uncomputed pending or appended token")
    limit = snapshot.num_kv_tokens if external_limit is None else external_limit
    if (type(limit) is not int or not 0 <= limit <= snapshot.num_kv_tokens
            or (limit != snapshot.num_kv_tokens and limit % snapshot.identity.block_size)):
        raise ContractError("external limit must be exact KV or a complete-block prefix")
    return max(0, limit - num_computed_tokens)


def finish_manifest(instruction, all_token_ids, num_computed_tokens, output_token_ids):
    """Capture only settled scheduler state, never async output placeholders."""
    tokens = list(all_token_ids)
    if not 0 <= num_computed_tokens <= len(tokens):
        raise ContractError("unsettled scheduler token boundary")
    previous = list(instruction.get("outputs", []))
    output_start = instruction["output_seq_start"]
    if len(previous) != output_start:
        raise ContractError("output history and committed sequence disagree")
    produced = list(output_token_ids)
    if produced and tokens[-len(produced):] != produced:
        raise ContractError("scheduler output tokens are not the token suffix")
    outputs = previous + [{"seq": output_start + i + 1, "token_id": token}
                          for i, token in enumerate(produced)]
    budget = instruction["remaining_max_tokens"] - len(produced)
    if budget < 0:
        raise ContractError("scheduler exceeded saved token budget")
    data = {
        "schema_version": 1, "snapshot_id": instruction["snapshot_id"],
        "session_id": instruction["session_id"], "generation": instruction["generation"],
        "identity": instruction["identity"], "token_ids": tokens,
        "num_kv_tokens": num_computed_tokens, "pending_tokens": tokens[num_computed_tokens:],
        "seed": instruction["seed"], "sampling": instruction["sampling"],
        "remaining_max_tokens": budget, "output_seq_committed": len(outputs),
        "outputs": outputs, "parent_id": instruction.get("parent_id"),
        "durability": instruction["durability"], "state": "Writing", "segments": [],
    }
    return SnapshotManifest.from_dict(data).to_dict()
