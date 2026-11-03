"""AsyncLLM bridge for the session gateway. Snapshots are taken at the end of a turn only."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

from kvsnap.gateway.engine import EngineSnapshot, EngineToken
from kvsnap.models import ContractError, Durability
from kvsnap.connector.telemetry import EventRecorder
from kvsnap.store.store import StoreError


class SaveFailed(StoreError):
    """The connector reported that a requested snapshot will never be committed."""


class VLLMEngine:
    snapshot_committed_at_request_end = True
    # The connector's worker reads and CRC-checks the blobs while loading them.
    restore_needs_payload = False

    def __init__(self, engine, store, *, snapshot_timeout=60.0,
                 seeded_sampling_verified=False, midturn_pause_verified=False,
                 require_batch_invariant=True):
        import vllm
        if vllm.__version__ != "0.31.0":
            raise RuntimeError("vLLM 0.31.0 is required")
        # Batch-invariant kernels make a restored turn bit-identical to a full
        # recompute. Benchmarks may turn the requirement off to measure the
        # default (faster, non-invariant) kernels.
        if require_batch_invariant and os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise RuntimeError("set VLLM_BATCH_INVARIANT=1 before starting the engine")
        self.engine, self.store = engine, store
        self.snapshot_timeout = snapshot_timeout
        self.seeded_sampling_verified = seeded_sampling_verified
        self.midturn_pause_verified = midturn_pause_verified
        self.telemetry = EventRecorder(getattr(engine, "kvsnap_runtime_id", None), role="adapter")
        self.requests = {}
        self.resume_sources = {}
        self.running = set()

    async def generate(self, request):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        if request.sampling.temperature and not self.seeded_sampling_verified:
            raise ContractError("only greedy decoding is supported: restored turns are verified for temperature 0")
        instruction = {
            "op": "save", "snapshot_id": request.snapshot_id,
            "session_id": request.session_id, "generation": request.generation,
            "identity": request.identity.to_dict(), "seed": request.seed,
            "sampling": request.sampling.to_dict(), "durability": request.durability.value,
            "output_seq_start": request.output_seq_start,
            "outputs": [o.to_dict() for o in request.outputs],
            "remaining_max_tokens": request.remaining_max_tokens,
            "parent_id": request.parent_id,
            "resume_snapshot_id": self.resume_sources.pop(request.request_id, None),
        }
        params = SamplingParams(**request.sampling.to_dict(), seed=request.seed,
                                max_tokens=request.max_tokens, detokenize=False,
                                output_kind=RequestOutputKind.DELTA, logprobs=1,
                                extra_args={"kv_transfer_params": {"kvsnap": instruction}})
        self.requests[request.request_id] = request
        self.running.add(request.request_id)
        self.telemetry.emit("adapter_generate", request_id=request.request_id,
                            session_id=request.session_id, snapshot_id=request.snapshot_id,
                            generation=request.generation, external_num_tokens=request.external_num_tokens,
                            prompt_tokens=len(request.token_ids), max_tokens=request.max_tokens)
        consumed = 0
        try:
            async for result in self.engine.generate(
                {"prompt_token_ids": list(request.token_ids)}, params, request.request_id
            ):
                if len(result.outputs) != 1:
                    raise ContractError("kvsnap supports one output sequence")
                if not consumed:
                    self.telemetry.emit("adapter_first_output", request_id=request.request_id)
                output = result.outputs[0]
                for index, token_id in enumerate(output.token_ids):
                    boundary = len(request.token_ids) + consumed
                    logprob = None
                    if output.logprobs and token_id in output.logprobs[index]:
                        logprob = output.logprobs[index][token_id].logprob
                    consumed += 1
                    reason = output.finish_reason if index == len(output.token_ids) - 1 else None
                    yield EngineToken(token_id, boundary, request.generation, logprob, reason)
        finally:
            self.running.discard(request.request_id)

    async def abort(self, request_id):
        if request_id not in self.running:
            return
        if not self.midturn_pause_verified:
            raise ContractError("pausing in the middle of a turn is not supported")
        await self.engine.abort(request_id)

    async def snapshot(self, request_id, num_kv_tokens):
        request = self.requests[request_id]
        deadline = time.monotonic() + self.snapshot_timeout
        while True:
            try:
                manifest = await asyncio.to_thread(self.store.get_manifest, request.snapshot_id)
                break
            except KeyError:
                reason = await asyncio.to_thread(self.store.save_failure, request.snapshot_id)
                if reason is not None:
                    raise SaveFailed(f"snapshot {request.snapshot_id} was not saved: {reason}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("connector did not commit the requested snapshot")
                await asyncio.sleep(0.01)
        if manifest["num_kv_tokens"] != num_kv_tokens:
            raise ContractError("gateway and settled scheduler KV boundaries differ")
        self.telemetry.emit("adapter_snapshot_observed", request_id=request_id,
                            snapshot_id=request.snapshot_id, generation=request.generation,
                            num_kv_tokens=num_kv_tokens, pending_token_count=len(manifest["pending_tokens"]))
        return EngineSnapshot((), num_kv_tokens, manifest)

    async def restore(self, request, blocks):
        if not request.parent_id:
            raise ContractError("restore requires a snapshot source")
        # The gateway has checked the snapshot against the prompt, and the
        # scheduler looks it up again when it admits the request, missing and
        # recomputing if it is gone by then. A third lookup here cost a
        # manifest read and parse on the admission path.
        if request.external_num_tokens:
            self.resume_sources[request.request_id] = request.parent_id

    async def release(self, request_id):
        # vLLM frees source blocks following connector get_finished. There is no
        # public API for gateway-side block-pool mutation.
        self.telemetry.emit("adapter_release_bookkeeping", request_id=request_id,
                            proves_engine_free=False)
        self.requests.pop(request_id, None)


def create_async_llm(store_path, *, model="Qwen/Qwen3-1.7B", revision="70d244cc", tokenizer_revision=None,
                     max_model_len=32768, block_size=16, kv_cache_memory_bytes=8 * 1024**3,
                     pinned_pool_bytes=4 * 1024**3, disk_capacity_bytes=200 * 1024**3,
                     max_num_batched_tokens=16384, cudagraph_mode=None, connector=True,
                     async_load=True, host_tier_bytes=None, enable_prefix_caching=False, engine_args=None):
    """Build the in-process engine the gateway drives.

    ``connector=False`` builds the identical engine without the kvsnap connector,
    the baseline for "recompute the whole prompt". ``cudagraph_mode=None``
    keeps vLLM's default CUDA graphs. Set ``VLLM_BATCH_INVARIANT=1`` before
    calling this for bit-exact comparisons against recompute. ``engine_args``
    adds or overrides vLLM engine arguments, for example vLLM's own CPU
    offloading (``kv_offloading_size``) on an engine built with
    ``connector=False``.
    """
    os.environ.setdefault("KVSNAP_IO_BACKEND", "native")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.config import KVTransferConfig
    runtime_id = uuid.uuid4().hex
    store_path = str(Path(store_path).resolve())
    transfer = KVTransferConfig(
        kv_connector="KvsnapConnector", kv_role="kv_both", kv_connector_module_path="kvsnap.connector.vllm",
        kv_connector_extra_config={"store_path": store_path, "runtime_id": runtime_id,
                                   "pinned_pool_bytes": pinned_pool_bytes,
                                   "disk_capacity_bytes": disk_capacity_bytes,
                                   "async_load": async_load,
                                   **({"host_tier_bytes": host_tier_bytes} if host_tier_bytes else {})},
        # A block that fails to load is recomputed rather than failing the request.
        kv_load_failure_policy="recompute")
    settings = dict(model=model, revision=revision, tokenizer_revision=tokenizer_revision or revision,
                dtype="bfloat16", tensor_parallel_size=1,
                max_model_len=max_model_len, block_size=block_size,
                kv_cache_memory_bytes=kv_cache_memory_bytes, async_scheduling=False,
                # vLLM checks this before honouring the explicit KV size;
                # leave room for desktop processes on the same GPU.
                gpu_memory_utilization=0.80,
                enable_prefix_caching=enable_prefix_caching,
                max_num_batched_tokens=max_num_batched_tokens,
                compilation_config={"cudagraph_mode": cudagraph_mode} if cudagraph_mode else {},
                kv_transfer_config=transfer if connector else None)
    settings.update(engine_args or {})
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**settings))
    engine.kvsnap_runtime_id = runtime_id
    return engine
