"""Versioned, JSON-serializable contracts shared by gateway and connector.

No vLLM import is required to inspect a snapshot. All token boundaries are in
token IDs; decoded text is deliberately absent from this contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping


class ContractError(ValueError):
    pass


class IdentityMismatch(ContractError):
    pass


class Durability(str, Enum):
    VOLATILE = "volatile"
    PROCESS = "process"
    DURABLE = "durable"


class SnapshotState(str, Enum):
    WRITING = "Writing"
    HOST_VALID = "HostValid"
    DURABLE = "Durable"
    INVALID = "Invalid"


class SessionState(str, Enum):
    ACTIVE = "Active"
    PAUSING = "Pausing"
    PAUSED = "Paused"
    RESUMING = "Resuming"
    CLOSED = "Closed"
    FAILED = "Failed"


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def tokens(values: Any, name: str = "token_ids") -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ContractError(f"{name} must be a list of token IDs")
    values = tuple(values)
    # Two C-level passes rather than an integer() call per token: a manifest
    # carries every token of the session and is validated at each turn end.
    if values and (not set(map(type, values)) <= {int} or min(values) < 0):
        raise ContractError(f"{name} must be an integer >= 0")
    return values


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ContractError(f"{name} must be a nonempty string <= 256 characters")
    if any(c in value for c in ("/", "\\", "\x00")):
        raise ContractError(f"{name} must not contain path separators")
    return value


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("value must be finite JSON") from exc


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        for key in ("temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ContractError(f"{key} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ContractError("temperature >= 0, 0 < top_p <= 1 and 0 <= min_p <= 1 required")
        if type(self.top_k) is not int or (self.top_k != -1 and self.top_k < 1):
            raise ContractError("top_k must be -1 or a positive integer")
        if self.presence_penalty != 0 or self.frequency_penalty != 0:
            raise ContractError("presence/frequency penalties cannot preserve resumed output semantics")
        if type(self.ignore_eos) is not bool:
            raise ContractError("ignore_eos must be boolean")
        object.__setattr__(self, "stop_token_ids", tokens(self.stop_token_ids, "stop_token_ids"))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["stop_token_ids"] = list(self.stop_token_ids)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SamplingConfig":
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"unsupported sampling parameters: {exc}") from exc


@dataclass(frozen=True)
class KVIdentity:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    kv_dtype: str
    block_size: int
    kv_layout: str
    model_shape: dict[str, Any]
    position_encoding: dict[str, Any]
    vllm_version: str = "0.31.0"

    def __post_init__(self) -> None:
        for key in ("model_id", "model_revision", "tokenizer_revision", "kv_dtype", "kv_layout", "vllm_version"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ContractError(f"identity {key} is required")
        integer(self.block_size, "block_size", 1)
        for key in ("model_shape", "position_encoding"):
            value = getattr(self, key)
            if not isinstance(value, dict) or not value:
                raise ContractError(f"identity {key} must be a nonempty object")
            object.__setattr__(self, key, json.loads(canonical_json(value)))
        integer(self.position_encoding.get("max_model_len"), "max_model_len", 1)
        # Each TP rank holds its own shard of the KV heads, so the TP size is
        # part of what the KV means. It is absent at TP=1: identities and
        # snapshots written before TP support stay valid and compare equal.
        tp = self.model_shape.get("tensor_parallel_size", 1)
        integer(tp, "tensor_parallel_size", 1)
        if tp == 1:
            self.model_shape.pop("tensor_parallel_size", None)

    @property
    def tensor_parallel_size(self) -> int:
        return self.model_shape.get("tensor_parallel_size", 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KVIdentity":
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ContractError(f"invalid identity: {exc}") from exc

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def require_match(self, other: "KVIdentity") -> None:
        if self.fingerprint != other.fingerprint:
            changed = [k for k, v in self.to_dict().items() if other.to_dict().get(k) != v]
            raise IdentityMismatch("incompatible KV identity: " + ", ".join(changed))


@dataclass(frozen=True)
class OutputToken:
    seq: int
    token_id: int
    # The sampled token's log-probability as the engine reported it. Streamed
    # to the client only; manifests record the token, not the float.
    logprob: float | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        integer(self.seq, "seq", 1)
        integer(self.token_id, "token_id")

    def to_dict(self) -> dict[str, int]:
        return {"seq": self.seq, "token_id": self.token_id}


@dataclass(frozen=True)
class Segment:
    blob_id: str
    offset: int
    length: int
    first_block: int
    nblocks: int
    crc32c: int
    # The TP rank whose KV shard this is; block numbers count within the rank.
    rank: int = 0

    def __post_init__(self) -> None:
        identifier(self.blob_id, "blob_id")
        integer(self.offset, "offset")
        integer(self.length, "length", 1)
        integer(self.first_block, "first_block")
        integer(self.nblocks, "nblocks", 1)
        integer(self.crc32c, "crc32c")
        integer(self.rank, "rank")
        if self.crc32c > 0xFFFFFFFF:
            raise ContractError("crc32c must fit uint32")

    def to_dict(self) -> dict[str, Any]:
        # Rank 0 is implicit, so a TP=1 manifest keeps its original form.
        return {k: v for k, v in vars(self).items() if k != "rank" or v}


def rank_segments(manifest: Mapping[str, Any], rank: int) -> list[dict[str, Any]]:
    """The segments of one TP rank's shard, in block order."""
    return [s for s in manifest["segments"] if s.get("rank", 0) == rank]


def tensor_parallel_size(identity: Mapping[str, Any]) -> int:
    """The TP size an identity dictionary was resolved for (1 when absent)."""
    return identity.get("model_shape", {}).get("tensor_parallel_size", 1)


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    session_id: str
    generation: int
    identity: KVIdentity
    token_ids: tuple[int, ...]
    num_kv_tokens: int
    pending_tokens: tuple[int, ...]
    seed: int
    sampling: SamplingConfig
    remaining_max_tokens: int
    output_seq_committed: int
    durability: Durability = Durability.VOLATILE
    state: SnapshotState = SnapshotState.WRITING
    parent_id: str | None = None
    outputs: tuple[OutputToken, ...] = ()
    segments: tuple[Segment, ...] = ()
    schema_version: int = 1
    stopped: bool = False

    def __post_init__(self) -> None:
        identifier(self.snapshot_id, "snapshot_id")
        identifier(self.session_id, "session_id")
        if self.parent_id is not None:
            identifier(self.parent_id, "parent_id")
            if self.parent_id == self.snapshot_id:
                raise ContractError("snapshot cannot be its own parent")
        for key in ("generation", "num_kv_tokens", "remaining_max_tokens", "output_seq_committed"):
            integer(getattr(self, key), key)
        integer(self.seed, "seed")
        if self.seed >= 2**63:
            raise ContractError("seed must fit a signed 64-bit integer")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ContractError("unsupported manifest schema_version")
        if type(self.stopped) is not bool:
            raise ContractError("stopped must be boolean")
        if not isinstance(self.identity, KVIdentity) or not isinstance(self.sampling, SamplingConfig):
            raise ContractError("identity and sampling must use their contract models")
        object.__setattr__(self, "durability", Durability(self.durability))
        object.__setattr__(self, "state", SnapshotState(self.state))
        object.__setattr__(self, "token_ids", tokens(self.token_ids))
        object.__setattr__(self, "pending_tokens", tokens(self.pending_tokens, "pending_tokens"))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "segments", tuple(self.segments))
        if self.num_kv_tokens > len(self.token_ids) or self.pending_tokens != self.token_ids[self.num_kv_tokens:]:
            raise ContractError("pending_tokens must equal token_ids[num_kv_tokens:]")
        if len(self.token_ids) > self.identity.position_encoding["max_model_len"]:
            raise ContractError("snapshot token sequence exceeds model context")
        if len(self.outputs) != self.output_seq_committed:
            raise ContractError("outputs must retain all committed tokens for ack replay")
        for expected_seq, output in enumerate(self.outputs, 1):
            if not isinstance(output, OutputToken) or output.seq != expected_seq:
                raise ContractError("outputs must have contiguous sequence numbers starting at 1")
        # One run of segments per TP rank, in rank order; each rank's blocks
        # are numbered from 0 and must be contiguous.
        tp = self.identity.tensor_parallel_size
        covered, rank, expected_block = {}, 0, 0
        for segment in self.segments:
            if not isinstance(segment, Segment):
                raise ContractError("segments must cover contiguous non-overlapping blocks")
            if segment.rank >= tp:
                raise ContractError(f"segment for TP rank {segment.rank} in a TP={tp} snapshot")
            if segment.rank != rank:
                if segment.rank < rank:
                    raise ContractError("segments must be grouped by TP rank, in rank order")
                covered[rank], rank, expected_block = expected_block, segment.rank, 0
            if segment.first_block != expected_block:
                raise ContractError("segments must cover contiguous non-overlapping blocks")
            expected_block += segment.nblocks
        covered[rank] = expected_block
        blocks = (self.num_kv_tokens + self.identity.block_size - 1) // self.identity.block_size
        if self.state in (SnapshotState.HOST_VALID, SnapshotState.DURABLE):
            incomplete = [r for r in range(tp) if covered.get(r, 0) != blocks]
            if incomplete:
                raise ContractError("committed segments must cover exactly the KV token blocks"
                                    + (f" on every TP rank; incomplete shards: {incomplete}" if tp > 1 else ""))
        if self.state == SnapshotState.HOST_VALID and self.durability != Durability.VOLATILE:
            raise ContractError("persistent snapshots cannot commit at HostValid")

    def to_dict(self) -> dict[str, Any]:
        # Field by field: asdict() deep-copied every token, output and segment
        # before most of that copy was replaced here.
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result["identity"] = self.identity.to_dict()
        result["durability"] = self.durability.value
        result["state"] = self.state.value
        result["token_ids"] = list(self.token_ids)
        result["pending_tokens"] = list(self.pending_tokens)
        result["outputs"] = [o.to_dict() for o in self.outputs]
        result["segments"] = [s.to_dict() for s in self.segments]
        result["sampling"] = self.sampling.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SnapshotManifest":
        data = dict(value)
        try:
            data["identity"] = KVIdentity.from_dict(data["identity"])
            data["sampling"] = SamplingConfig.from_dict(data["sampling"])
            data["outputs"] = tuple(OutputToken(**o) for o in data.get("outputs", ()))
            data["segments"] = tuple(Segment(**s) for s in data.get("segments", ()))
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ContractError(f"invalid snapshot manifest: {exc}") from exc

    def validate_resume(self, identity: KVIdentity, prompt: tuple[int, ...]) -> int:
        if self.state not in (SnapshotState.HOST_VALID, SnapshotState.DURABLE):
            raise ContractError(f"snapshot in {self.state.value} is not loadable")
        self.identity.require_match(identity)
        if tuple(prompt[:len(self.token_ids)]) != self.token_ids:
            raise ContractError("resume prompt does not preserve the exact snapshot token prefix")
        if len(prompt) > identity.position_encoding["max_model_len"]:
            raise ContractError("resume prompt exceeds model context")
        # vLLM needs one token to compute logits. Do not silently load fewer KV
        # tokens than the manifest declares: that would weaken the contract.
        if self.num_kv_tokens > len(prompt) - 1:
            raise ContractError("resume requires at least one uncomputed token or appended token")
        return self.num_kv_tokens
