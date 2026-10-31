"""Engine boundary, plus an explicitly synthetic CPU correctness fixture."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, AsyncIterator, Protocol

from kvsnap.models import Durability, KVIdentity, OutputToken, SamplingConfig


@dataclass(frozen=True)
class EngineRequest:
    request_id: str
    session_id: str
    snapshot_id: str
    generation: int
    identity: KVIdentity
    token_ids: tuple[int, ...]
    seed: int
    sampling: SamplingConfig
    max_tokens: int
    external_num_tokens: int = 0
    durability: Durability = Durability.VOLATILE
    output_seq_start: int = 0
    remaining_max_tokens: int = 0
    outputs: tuple[OutputToken, ...] = ()
    parent_id: str | None = None


@dataclass(frozen=True)
class EngineToken:
    token_id: int
    # Computed prefix length after sampling this token, excluding that token.
    num_kv_tokens: int
    generation: int
    logprob: float | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class EngineSnapshot:
    blocks: tuple[bytes, ...]
    num_kv_tokens: int
    committed_manifest: dict[str, Any] | None = None


class EngineAdapter(Protocol):
    """Events yielded by generate must represent a settled token boundary.

    abort returns only once no more output can become committed. snapshot waits
    for all D2H copies; release may only run after a successful store commit.
    restore validates and prepares loading. An engine that allocates GPU blocks
    during generate must finish H2D before it can emit any output. Adapters must
    reject unsafe modes and retain load buffers/leases through the H2D event.
    """

    def generate(self, request: EngineRequest) -> AsyncIterator[EngineToken]: ...
    async def abort(self, request_id: str) -> None: ...
    async def snapshot(self, request_id: str, num_kv_tokens: int) -> EngineSnapshot: ...
    async def restore(self, request: EngineRequest, blocks: list[bytes]) -> None: ...
    async def release(self, request_id: str) -> None: ...


class DeterministicEngine:
    """CPU fixture only: deterministic token IDs and synthetic KV bytes.

    It does not exercise vLLM, CUDA copies or performance.
    Tokens depend on the whole prefix, explicit seed and absolute position, so
    an uninterrupted run is an independent oracle for split/resumed requests.
    """

    evidence_tier = "cpu_fixture"
    snapshot_committed_at_request_end = False

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.requests: dict[str, EngineRequest] = {}
        self.histories: dict[str, list[int]] = {}
        self.aborted: set[str] = set()
        self.events: list[dict[str, Any]] = []

    def _event(self, event: str, request_id: str, **details: Any) -> None:
        request = self.requests[request_id]
        self.events.append({"event": event, "request_id": request_id,
                            "generation": request.generation,
                            "evidence_tier": self.evidence_tier, **details})

    @staticmethod
    def _block(ids: list[int]) -> bytes:
        return json.dumps(ids, separators=(",", ":")).encode()

    async def generate(self, request: EngineRequest) -> AsyncIterator[EngineToken]:
        self.requests[request.request_id] = request
        history = self.histories.setdefault(request.request_id, list(request.token_ids))
        for _ in range(request.max_tokens):
            await asyncio.sleep(self.delay)
            if request.request_id in self.aborted:
                break
            payload = json.dumps([request.seed, history, request.sampling.to_dict()], sort_keys=True).encode()
            token_id = 1 + int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % 32000
            num_kv_tokens = len(history)
            history.append(token_id)
            yield EngineToken(token_id, num_kv_tokens, request.generation, -1.0)
            if token_id in request.sampling.stop_token_ids:
                break

    async def abort(self, request_id: str) -> None:
        self.aborted.add(request_id)

    async def snapshot(self, request_id: str, num_kv_tokens: int) -> EngineSnapshot:
        request = self.requests[request_id]
        history = self.histories[request_id]
        if not 0 <= num_kv_tokens <= len(history):
            raise ValueError("invalid computed token boundary")
        self._event("D2H_started", request_id)
        blocks = tuple(self._block(history[i:min(i + request.identity.block_size, num_kv_tokens)])
                       for i in range(0, num_kv_tokens, request.identity.block_size))
        await asyncio.sleep(0)
        self._event("D2H_complete", request_id)
        self._event("connector_save_finished", request_id)
        return EngineSnapshot(blocks, num_kv_tokens)

    async def restore(self, request: EngineRequest, blocks: list[bytes]) -> None:
        self.requests[request.request_id] = request
        self._event("H2D_started", request.request_id)
        expected = [self._block(list(request.token_ids[i:min(i + request.identity.block_size,
                                                           request.external_num_tokens)]))
                    for i in range(0, request.external_num_tokens, request.identity.block_size)]
        if list(blocks) != expected:
            raise ValueError("synthetic KV bytes differ from token prefix")
        self.histories[request.request_id] = list(request.token_ids)
        await asyncio.sleep(0)
        self._event("H2D_complete", request.request_id)
        self._event("pinned_released", request.request_id)

    async def release(self, request_id: str) -> None:
        self._event("GPU_released", request_id)
        self.histories.pop(request_id, None)


def fixture_identity(block_size: int = 16, max_model_len: int = 32768) -> KVIdentity:
    return KVIdentity(model_id="kvsnap-cpu-fixture", model_revision="fixture-v1",
                      tokenizer_revision="token-id-only-v1", kv_dtype="synthetic-json",
                      block_size=block_size, kv_layout="synthetic-token-blocks",
                      model_shape={"layers": 1, "kv_heads": 1, "head_size": 1},
                      position_encoding={"max_model_len": max_model_len, "type": "fixture"},
                      vllm_version="not-used-cpu-fixture")
