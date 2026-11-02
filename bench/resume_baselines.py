"""Resumed-turn TTFT for each way of serving a session's earlier turns.

    .venv-gpu/bin/python -m bench.resume_baselines --engine prefix-cache --regime resident \\
        --output results/raw/baselines/prefix-cache-resident-1.json

Every engine uses the settings of the other benchmarks: synchronous
scheduling (kvsnap needs it), vLLM's default kernels and CUDA graphs.

  recompute       no connector, prefix caching off: each turn prefills its whole prompt
  prefix-cache    no connector, vLLM's prefix caching on
  native-offload  prefix caching on plus vLLM's own CPU offloading
                  (``--kv-offloading-size``, the OffloadingConnector), ``--offload-gib``
  lmcache         prefix caching on plus LMCache's in-process connector with a CPU
                  backend of ``--offload-gib`` (needs an environment with lmcache)
  kvsnap          the kvsnap connector through the session gateway, host-memory
                  snapshots, prefix caching off unless ``--prefix-caching``

Regimes:

  resident  one session at a time: turn 2 follows turn 1 directly, so its KV is
            still in GPU memory. This is prefix caching's best case and the lower
            bound for any KV offloader. ``--reps`` sessions per entry of
            ``--lengths``.
  evicted   ``--sessions`` sessions take turn 1 in order, then later turns in the
            same order, ``--rounds`` times. With sessions x length well above the
            GPU's KV capacity, each session's blocks have left GPU memory by the
            time it comes back.

Turn 1 prefills ``--length`` random tokens and generates 1; every later turn
appends 64 tokens and generates 16. TTFT is client-side, from submitting a
turn to its first token. For the engines driven directly, ``num_cached_tokens``
records how many prompt tokens vLLM did not recompute (GPU prefix-cache hits
plus tokens loaded by a connector). One discarded session warms each engine
first. One process is one run; ``--combine`` merges runs.
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

SUFFIX, OUTPUT = 64, 16
ENGINES = ("recompute", "prefix-cache", "native-offload", "lmcache", "kvsnap")
OFFLOADERS = ("native-offload", "lmcache")


def build(args, root: Path):
    from kvsnap.connector.engine import create_async_llm
    rounds = 1 if args.regime == "resident" else args.rounds
    common = dict(model=args.model, revision=args.revision,
                  max_model_len=max(args.lengths) + 1 + rounds * (SUFFIX + OUTPUT) + 16,
                  kv_cache_memory_bytes=args.kv_cache_gib * 1024**3,
                  max_num_batched_tokens=args.max_num_batched_tokens)
    if args.engine == "kvsnap":
        return create_async_llm(root, pinned_pool_bytes=args.pinned_gib * 1024**3,
                                enable_prefix_caching=args.prefix_caching, **common)
    extra = offloader_args(args.engine, args.offload_gib) if args.engine in OFFLOADERS else {}
    return create_async_llm(root, connector=False, enable_prefix_caching=args.engine != "recompute",
                            engine_args=extra, **common)


def offloader_args(engine: str, gib: int) -> dict:
    """vLLM engine arguments for vLLM's own CPU offloading or LMCache, with ``gib`` GiB of CPU memory."""
    if engine == "native-offload":
        return {"kv_offloading_size": gib, "kv_offloading_backend": "native"}
    from vllm.config import KVTransferConfig
    os.environ.update(LMCACHE_CHUNK_SIZE="256", LMCACHE_LOCAL_CPU="True", LMCACHE_MAX_LOCAL_CPU_SIZE=str(gib))
    return {"kv_transfer_config": KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")}


async def run(args) -> dict:
    measured = environment(args.store_dir)
    root = Path(tempfile.mkdtemp(prefix="kvsnap-baseline-", dir=args.store_dir))
    (root / "events").mkdir()
    os.environ["KVSNAP_EVENT_DIR"] = str(root / "events")
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    runtime = build(args, root)
    rng = random.Random(args.seed)
    store = gateway = None
    try:
        if args.engine == "kvsnap":
            from kvsnap.connector.engine import VLLMEngine
            from kvsnap.connector.runtime import wait_runtime_identity
            from kvsnap.gateway import SessionGateway
            from kvsnap.models import SamplingConfig
            from kvsnap.store import SnapshotStore
            store = SnapshotStore(root, host_capacity_bytes=args.pinned_gib * 1024**3)
            identity = await wait_runtime_identity(root, runtime.kvsnap_runtime_id, timeout=600)
            gateway = SessionGateway(VLLMEngine(runtime, store, require_batch_invariant=False), store)
            sampling = SamplingConfig(ignore_eos=True)
        rounds = 1 if args.regime == "resident" else args.rounds

        def tokens(count):
            return [rng.randrange(1000, 100_000) for _ in range(count)]

        async def direct(prompt, max_tokens):
            params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, detokenize=False,
                                    output_kind=RequestOutputKind.DELTA)
            start, first, produced, cached = time.perf_counter(), None, [], None
            async for out in runtime.generate({"prompt_token_ids": list(prompt)}, params, uuid.uuid4().hex):
                first = first or time.perf_counter()
                produced.extend(out.outputs[0].token_ids)
                cached = out.num_cached_tokens if out.num_cached_tokens is not None else cached
            return (first - start) * 1e3, produced, cached

        async def open_session(length):
            prompt = tokens(length)
            if gateway is None:
                _, produced, _ = await direct(prompt, 1)
                return {"tokens": prompt + produced}
            session = await gateway.create(identity, sampling=sampling, max_tokens=1 + rounds * OUTPUT,
                                           durability="volatile")
            async for _ in await gateway.start_turn(session.session_id, prompt,
                                                    expected_generation=session.generation, max_tokens=1):
                pass
            await gateway.pause(session.session_id, expected_generation=session.generation)
            return session

        async def next_turn(session) -> dict:
            suffix = tokens(SUFFIX)
            if gateway is None:
                prompt = session["tokens"] + suffix
                ttft, produced, cached = await direct(prompt, OUTPUT)
                session["tokens"] = prompt + produced
                return {"ttft_ms": ttft, "prompt_tokens": len(prompt), "num_cached_tokens": cached}
            prompt_tokens = len(session.token_ids) + SUFFIX
            start, first = time.perf_counter(), None
            handle = await gateway.resume(session.session_id, expected_generation=session.generation,
                                          ack_seq=len(session.outputs), append_tokens=suffix, max_tokens=OUTPUT)
            async for _ in handle:
                first = first or time.perf_counter()
            await gateway.pause(session.session_id, expected_generation=session.generation)
            return {"ttft_ms": (first - start) * 1e3, "prompt_tokens": prompt_tokens}

        async def close(session):
            if gateway is not None:
                from bench.restore_ttft import drop_session
                await gateway.close(session.session_id, expected_generation=session.generation)
                drop_session(store, session.session_id)

        samples = []
        for length in args.lengths:
            warm = await open_session(length)
            await next_turn(warm)
            await close(warm)
            measured.setdefault("gpu_at_measurement", gpu_state())
            if args.regime == "resident":
                for rep in range(args.reps):
                    session = await open_session(length)
                    samples.append({"length": length, "session": rep, "round": 0, **await next_turn(session)})
                    await close(session)
                continue
            sessions = [await open_session(length) for _ in range(args.sessions)]
            for round_index in range(rounds):
                for index, session in enumerate(sessions):
                    samples.append({"length": length, "session": index, "round": round_index,
                                    **await next_turn(session)})
            for session in sessions:
                await close(session)
        for sample in samples:
            print(json.dumps(sample), flush=True)
        return {**config(args), "seed": args.seed, "samples": samples, "summary": summarise(samples),
                "environment": measured}
    finally:
        runtime.shutdown()
        if store is not None:
            store.close()
        shutil.rmtree(root, ignore_errors=True)


def config(args) -> dict:
    return {"model": model_label(args.model), "revision": args.revision, "engine": args.engine,
            "regime": args.regime, "lengths": args.lengths,
            "sessions": args.sessions if args.regime == "evicted" else 1,
            "rounds": args.rounds if args.regime == "evicted" else 1,
            "kv_cache_gib": args.kv_cache_gib, "offload_gib": args.offload_gib, "pinned_gib": args.pinned_gib,
            "prefix_caching": args.engine not in ("recompute", "kvsnap") or args.prefix_caching,
            "max_num_batched_tokens": args.max_num_batched_tokens, "suffix_tokens": SUFFIX,
            "output_tokens": OUTPUT}


def summarise(samples, runs=None) -> list[dict]:
    """One row per context length; with ``runs``, also each run's median."""
    rows = []
    for length in sorted({s["length"] for s in samples}):
        picked = [s for s in samples if s["length"] == length]
        ttft = [s["ttft_ms"] for s in picked]
        row = {"context_tokens": length, "n": len(ttft), "ttft_ms_median": statistics.median(ttft),
               "ttft_ms_min": min(ttft), "ttft_ms_max": max(ttft)}
        if runs is not None:
            row["ttft_ms_median_by_run"] = [statistics.median(s["ttft_ms"] for s in picked if s["run"] == run)
                                            for run in range(runs)]
        cached = [s["num_cached_tokens"] for s in picked if s.get("num_cached_tokens") is not None]
        if cached:
            row["num_cached_tokens_median"] = statistics.median(cached)
            row["prompt_tokens_median"] = statistics.median(s["prompt_tokens"] for s in picked)
        rows.append(row)
    return rows


def combine(runs: list[dict]) -> dict:
    keep = {k: v for k, v in runs[0].items() if k not in ("seed", "samples", "summary", "environment")}
    for other in runs[1:]:
        if {k: v for k, v in other.items() if k in keep} != keep:
            raise ValueError("runs differ in configuration")
    samples = [{**sample, "run": index} for index, run in enumerate(runs) for sample in run["samples"]]
    return {**keep, "runs": len(runs), "seeds": [run["seed"] for run in runs], "samples": samples,
            "summary": summarise(samples, len(runs)),
            "environment_by_run": [run["environment"] for run in runs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--regime", choices=("resident", "evicted"))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192])
    parser.add_argument("--reps", type=int, default=5, help="resident: sessions measured")
    parser.add_argument("--sessions", type=int, default=4, help="evicted: sessions in the rotation")
    parser.add_argument("--rounds", type=int, default=3, help="evicted: turns after turn 1, per session")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=int, default=2)
    parser.add_argument("--offload-gib", type=int, default=8, help="CPU capacity of native-offload and lmcache")
    parser.add_argument("--pinned-gib", type=int, default=8, help="kvsnap: host tier is 3/4 of it")
    parser.add_argument("--prefix-caching", action="store_true", help="kvsnap: also turn vLLM's prefix caching on")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8448)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combine", type=Path, nargs="+", metavar="RUN_JSON",
                        help="merge these per-run outputs into --output instead of running")
    args = parser.parse_args()
    if not args.combine and (args.engine is None or args.regime is None):
        parser.error("--engine and --regime are required unless --combine is given")
    if args.combine:
        result = combine([json.loads(path.read_text()) for path in args.combine])
    else:
        result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
