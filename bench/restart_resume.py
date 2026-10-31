"""Resume sessions after the engine restarts: the durable tier vs recompute.

    .venv-gpu/bin/python -m bench.restart_resume --output results/restart-qwen3-8b.json

1. Engine A: for each length and rep, a session prefills L tokens and
   ``pause`` saves it durably (blobs fdatasynced, manifest committed in
   SQLite WAL with synchronous=FULL).
2. Engine A shuts down; its EngineCore process, GPU memory and host-memory
   tier are gone. Engine B starts on the same store directory.
3. ``gateway.recover`` re-registers each session from its manifest and turn 2
   appends 64 tokens. Half the sessions are first evicted from the page cache.

One process is one run; ``--combine`` merges separate runs into one file.

TTFT is client-side. The recompute reference is the same prompt sent to
engine B directly, once the resumed turn's own durable save has finished. An offload tier that lives in host memory (vLLM's own CPU
offloading, or this project's volatile tier) starts empty after step 2.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time
import uuid

from bench.environment import environment, gpu_state, model_label
from bench.restore_ttft import evict_from_page_cache

SUFFIX, OUTPUT = 64, 16


async def engine(args, root):
    from kvsnap.connector.engine import VLLMEngine, create_async_llm
    from kvsnap.connector.runtime import wait_runtime_identity
    from kvsnap.gateway import SessionGateway
    from kvsnap.store import SnapshotStore
    runtime = create_async_llm(root, model=args.model, revision=args.revision,
                               max_model_len=max(args.lengths) + SUFFIX + OUTPUT + 16,
                               kv_cache_memory_bytes=args.kv_cache_gib * 1024**3,
                               max_num_batched_tokens=args.max_num_batched_tokens)
    store = SnapshotStore(root)
    identity = await wait_runtime_identity(root, runtime.kvsnap_runtime_id, timeout=600)
    return runtime, store, identity, SessionGateway(VLLMEngine(runtime, store, require_batch_invariant=False), store)


async def run(args) -> dict:
    measured = environment(args.store_dir)
    root = Path(tempfile.mkdtemp(prefix="kvsnap-restart-", dir=args.store_dir))
    from vllm import SamplingParams
    from kvsnap.models import SamplingConfig
    rng = random.Random(args.seed)
    saved = []
    try:
        runtime, store, identity, gateway = await engine(args, root)
        for length in args.lengths:
            for rep in range(args.reps + 1):              # rep 0 is a discarded warm-up
                session = await gateway.create(identity, sampling=SamplingConfig(ignore_eos=True),
                                               max_tokens=1 + OUTPUT, durability="durable")
                prompt = [rng.randrange(1000, 100_000) for _ in range(length)]
                async for _ in await gateway.start_turn(session.session_id, prompt,
                                                        expected_generation=session.generation, max_tokens=1):
                    pass
                manifest = await gateway.pause(session.session_id, expected_generation=session.generation)
                saved.append((length, rep, manifest.snapshot_id))
        runtime.shutdown()
        store.close()
        stopped = time.perf_counter()

        runtime, store, identity, gateway = await engine(args, root)
        restart_s = time.perf_counter() - stopped
        measured["gpu_at_measurement"] = gpu_state()
        rows = []
        for length, rep, snapshot_id in saved:
            cold = rep % 2 == 1
            session = gateway.recover(snapshot_id)
            manifest = store.get_manifest(snapshot_id)
            if cold:
                evict_from_page_cache(root, manifest)
            suffix = [rng.randrange(1000, 100_000) for _ in range(SUFFIX)]
            start, first, tokens = time.perf_counter(), None, []
            handle = await gateway.resume(session.session_id, expected_generation=session.generation,
                                          ack_seq=len(session.outputs), append_tokens=suffix, max_tokens=OUTPUT)
            async for token in handle:
                first = first or time.perf_counter()
                tokens.append(token.token_id)
            # Turn 2 saves durably when it ends. Wait for that, as restore_ttft
            # does, so the recompute does not share the GPU and the disk with it.
            await gateway.pause(session.session_id, expected_generation=session.generation)
            params = SamplingParams(max_tokens=OUTPUT, temperature=0.0, ignore_eos=True, detokenize=False)
            began, reference_first, reference = time.perf_counter(), None, None
            async for out in runtime.generate({"prompt_token_ids": list(manifest["token_ids"]) + suffix},
                                              params, uuid.uuid4().hex):
                reference_first = reference_first or time.perf_counter()
                reference = list(out.outputs[0].token_ids)
            if rep:
                rows.append({"context_tokens": length, "cold": cold, "ttft_ms": (first - start) * 1e3,
                             "recompute_ttft_ms": (reference_first - began) * 1e3, "match": tokens == reference})
                print(json.dumps(rows[-1]), flush=True)
        runtime.shutdown()
        store.close()
        return {"model": model_label(args.model), "revision": args.revision, "lengths": args.lengths,
                "kv_cache_gib": args.kv_cache_gib, "max_num_batched_tokens": args.max_num_batched_tokens,
                "seed": args.seed, "engine_restart_s": restart_s, "rows": rows, "summary": summarise(rows),
                "environment": measured}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def summarise(rows) -> list[dict]:
    summary = []
    for length in sorted({r["context_tokens"] for r in rows}):
        for cold in (False, True):
            picked = [r for r in rows if r["context_tokens"] == length and r["cold"] == cold]
            if not picked:
                continue
            ttft = [r["ttft_ms"] for r in picked]
            recompute = [r["recompute_ttft_ms"] for r in picked]
            summary.append({"context_tokens": length, "cold": cold, "n": len(picked),
                            "ttft_ms_median": statistics.median(ttft), "ttft_ms_min": min(ttft),
                            "ttft_ms_max": max(ttft), "recompute_ttft_ms_median": statistics.median(recompute),
                            "recompute_ttft_ms_min": min(recompute), "recompute_ttft_ms_max": max(recompute),
                            "tokens_match": sum(r["match"] for r in picked)})
    return summary


def combine(runs: list[dict]) -> dict:
    """Pool the sessions of separate runs of one configuration."""
    keep = {k: v for k, v in runs[0].items()
            if k not in ("seed", "engine_restart_s", "rows", "summary", "environment")}
    for other in runs[1:]:
        if {k: v for k, v in other.items() if k in keep} != keep:
            raise ValueError("runs differ in configuration")
    rows = [{**row, "run": index} for index, run in enumerate(runs) for row in run["rows"]]
    return {**keep, "runs": len(runs), "seeds": [run["seed"] for run in runs],
            "engine_restart_s_by_run": [run["engine_restart_s"] for run in runs],
            "rows": rows, "summary": summarise(rows), "environment_by_run": [run["environment"] for run in runs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=int, default=6)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combine", type=Path, nargs="+", metavar="RUN_JSON",
                        help="merge these per-run outputs into --output instead of running")
    args = parser.parse_args()
    if args.revision.lower() == "none":
        args.revision = None
    if args.combine:
        result = combine([json.loads(path.read_text()) for path in args.combine])
    else:
        result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
