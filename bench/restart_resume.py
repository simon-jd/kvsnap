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

``--restart-tensor-parallel-size`` starts engine B at another TP size than
engine A. Every snapshot must then miss (the TP size is part of the KV
identity) and turn 2 is recomputed; connector telemetry is on in that mode,
and the output counts engine B's ``restore_miss`` events and loads.

TTFT is client-side. The recompute reference is the same prompt sent to
engine B directly, once the resumed turn's own durable save has finished. An offload tier that lives in host memory (vLLM's own CPU
offloading, or this project's volatile tier) starts empty after step 2.

``--engine lmcache-disk`` runs the same sequence without kvsnap: prefix
caching on and LMCache with its CPU backend and its local-disk backend in
the store directory (needs the LMCache environment). Engine A prefills each
session and generates one token; once the disk directory stops growing,
engine A shuts down, and engine B starts on the same directory. Turn 2 sends
the whole prompt and records ``num_cached_tokens``, the prompt tokens vLLM
did not compute.
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
from bench.resume_baselines import offloader_args

SUFFIX, OUTPUT = 64, 16


async def engine(args, root, tensor_parallel_size=None):
    from kvsnap.connector.engine import VLLMEngine, create_async_llm
    from kvsnap.connector.runtime import wait_runtime_identity
    from kvsnap.gateway import SessionGateway
    from kvsnap.store import SnapshotStore
    tp = tensor_parallel_size or args.tensor_parallel_size
    runtime = create_async_llm(root, model=args.model, revision=args.revision,
                               max_model_len=max(args.lengths) + SUFFIX + OUTPUT + 16,
                               # The same KV capacity in tokens at every TP size.
                               kv_cache_memory_bytes=args.kv_cache_gib * 1024**3 * args.tensor_parallel_size // tp,
                               max_num_batched_tokens=args.max_num_batched_tokens,
                               tensor_parallel_size=tp)
    store = SnapshotStore(root)
    identity = await wait_runtime_identity(root, runtime.kvsnap_runtime_id, timeout=600)
    return runtime, store, identity, SessionGateway(VLLMEngine(runtime, store, require_batch_invariant=False), store)


async def run(args) -> dict:
    measured = environment(args.store_dir)
    root = Path(tempfile.mkdtemp(prefix="kvsnap-restart-", dir=args.store_dir))
    cross = args.restart_tensor_parallel_size not in (None, args.tensor_parallel_size)
    if cross:
        (root / "events").mkdir()
        os.environ["KVSNAP_EVENT_DIR"] = str(root / "events")
    from vllm import SamplingParams
    from kvsnap.connector.engine import SaveFailed
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

        restarted = time.monotonic_ns()
        runtime, store, identity, gateway = await engine(args, root, args.restart_tensor_parallel_size)
        restart_s = time.perf_counter() - stopped
        measured["gpu_at_measurement"] = gpu_state()
        rows, saves_refused = [], 0
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
            # Across TP sizes that save must fail: the session's identity is
            # engine A's.
            try:
                await gateway.pause(session.session_id, expected_generation=session.generation)
            except SaveFailed:
                if not cross:
                    raise
                saves_refused += 1
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
        extra = {}
        if cross:
            events = [json.loads(line) for path in (root / "events").glob("kvsnap-events-*.jsonl")
                      for line in path.read_text().splitlines()]
            after = [e for e in events if e["monotonic_ns"] >= restarted]
            misses = [e for e in after if e["event"] == "restore_miss"]
            extra = {"restart_tensor_parallel_size": args.restart_tensor_parallel_size,
                     "restore_misses": len(misses), "miss_reasons": sorted({e["reason"] for e in misses}),
                     "loads_after_restart": sum(e["event"] == "h2d_completed" for e in after),
                     "turn2_saves_refused": saves_refused}
        return {"model": model_label(args.model), "revision": args.revision, "lengths": args.lengths, **extra,
                "kv_cache_gib": args.kv_cache_gib, "max_num_batched_tokens": args.max_num_batched_tokens,
                **({"tensor_parallel_size": args.tensor_parallel_size} if args.tensor_parallel_size > 1 else {}),
                "seed": args.seed, "engine_restart_s": restart_s, "rows": rows, "summary": summarise(rows),
                "environment": measured}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def disk_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


async def run_lmcache_disk(args) -> dict:
    measured = environment(args.store_dir)
    root = Path(tempfile.mkdtemp(prefix="kvsnap-restart-lmcache-", dir=args.store_dir))
    disk = root / "lmcache-disk"
    disk.mkdir()
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from kvsnap.connector.engine import create_async_llm
    engine_args = offloader_args("lmcache", args.offload_gib)
    os.environ.update(LMCACHE_LOCAL_DISK=f"file://{disk}/", LMCACHE_MAX_LOCAL_DISK_SIZE=str(args.disk_gib))
    rng = random.Random(args.seed)

    def build():
        return create_async_llm(root, connector=False, enable_prefix_caching=True, engine_args=engine_args,
                                model=args.model, revision=args.revision,
                                max_model_len=max(args.lengths) + 1 + SUFFIX + OUTPUT + 16,
                                kv_cache_memory_bytes=args.kv_cache_gib * 1024**3,
                                max_num_batched_tokens=args.max_num_batched_tokens)

    async def direct(runtime, prompt, max_tokens):
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, detokenize=False,
                                output_kind=RequestOutputKind.DELTA)
        start, first, produced, cached = time.perf_counter(), None, [], None
        async for out in runtime.generate({"prompt_token_ids": list(prompt)}, params, uuid.uuid4().hex):
            first = first or time.perf_counter()
            produced.extend(out.outputs[0].token_ids)
            cached = out.num_cached_tokens if out.num_cached_tokens is not None else cached
        return (first - start) * 1e3, produced, cached

    try:
        runtime = build()
        saved = []
        for length in args.lengths:
            for rep in range(args.reps + 1):              # rep 0 is a discarded warm-up
                prompt = [rng.randrange(1000, 100_000) for _ in range(length)]
                _, produced, _ = await direct(runtime, prompt, 1)
                saved.append((length, rep, prompt + produced))
        # LMCache writes the disk backend asynchronously: wait until it settles.
        written, settled = -1, time.perf_counter()
        while time.perf_counter() - settled < 10:
            await asyncio.sleep(1)
            now = disk_bytes(disk)
            if now != written:
                written, settled = now, time.perf_counter()
        runtime.shutdown()
        stopped = time.perf_counter()
        runtime = build()
        restart_s = time.perf_counter() - stopped
        measured["gpu_at_measurement"] = gpu_state()
        rows = []
        for length, rep, tokens in saved:
            suffix = [rng.randrange(1000, 100_000) for _ in range(SUFFIX)]
            ttft, _, cached = await direct(runtime, tokens + suffix, OUTPUT)
            if rep:
                rows.append({"context_tokens": length, "ttft_ms": ttft, "prompt_tokens": len(tokens) + SUFFIX,
                             "num_cached_tokens": cached})
                print(json.dumps(rows[-1]), flush=True)
        runtime.shutdown()
        return {"model": model_label(args.model), "revision": args.revision, "engine": "lmcache-disk",
                "lengths": args.lengths, "kv_cache_gib": args.kv_cache_gib, "offload_gib": args.offload_gib,
                "disk_gib": args.disk_gib, "max_num_batched_tokens": args.max_num_batched_tokens,
                "seed": args.seed, "disk_bytes_before_restart": written, "engine_restart_s": restart_s,
                "rows": rows, "summary": summarise_lmcache(rows), "environment": measured}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def summarise_lmcache(rows) -> list[dict]:
    summary = []
    for length in sorted({r["context_tokens"] for r in rows}):
        picked = [r for r in rows if r["context_tokens"] == length]
        ttft = [r["ttft_ms"] for r in picked]
        cached = [r["num_cached_tokens"] for r in picked]
        summary.append({"context_tokens": length, "n": len(picked), "ttft_ms_median": statistics.median(ttft),
                        "ttft_ms_min": min(ttft), "ttft_ms_max": max(ttft),
                        "num_cached_tokens_max": max(cached), "prompt_tokens": picked[0]["prompt_tokens"]})
    return summary


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
            if k not in ("seed", "engine_restart_s", "disk_bytes_before_restart", "rows", "summary", "environment")}
    for other in runs[1:]:
        if {k: v for k, v in other.items() if k in keep} != keep:
            raise ValueError("runs differ in configuration")
    rows = [{**row, "run": index} for index, run in enumerate(runs) for row in run["rows"]]
    lmcache = keep.get("engine") == "lmcache-disk"
    extra = {"disk_bytes_before_restart_by_run": [run["disk_bytes_before_restart"] for run in runs]} if lmcache else {}
    return {**keep, "runs": len(runs), "seeds": [run["seed"] for run in runs],
            "engine_restart_s_by_run": [run["engine_restart_s"] for run in runs], **extra,
            "rows": rows, "summary": (summarise_lmcache if lmcache else summarise)(rows),
            "environment_by_run": [run["environment"] for run in runs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("kvsnap", "lmcache-disk"), default="kvsnap")
    parser.add_argument("--offload-gib", type=int, default=8, help="lmcache-disk: its CPU backend")
    parser.add_argument("--disk-gib", type=int, default=40, help="lmcache-disk: its local-disk backend")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=int, default=6, help="per TP rank")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="kvsnap engine only")
    parser.add_argument("--restart-tensor-parallel-size", type=int, default=None,
                        help="engine B's TP size, if not engine A's: every resume must miss")
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
        result = asyncio.run(run_lmcache_disk(args) if args.engine == "lmcache-disk" else run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
