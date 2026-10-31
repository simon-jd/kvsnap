"""What a resumed turn costs the requests already running.

    .venv-gpu/bin/python -m bench.restore_under_load --output results/under-load-qwen3-1.7b.json

``--streams`` decode streams run greedy requests of ``--decode-tokens``
tokens back to back (128-token prompts, no EOS). While they run,
``--sessions`` sessions of ``--length`` tokens resume one every
``--interval`` seconds, round robin, until ``--resumes`` turns have been
measured; each turn appends 64 tokens and generates 16. Each arm first runs
``--warmup`` resumes that are not counted. A turn is served one of three ways:

  recompute  the whole prompt goes straight to the engine (no gateway, no
             snapshot), and vLLM prefills it alongside the decodes; prefix
             caching is off, as in every engine here;
  sync       the session restores from its host-memory snapshot, loaded
             inside start_load_kv, so every running request waits for the copy;
  async      the loader thread copies it while the batch keeps running.

Restored sessions pause after every turn, which saves the turn into the host
tier again. ``sync`` and ``async`` need differently configured engines, so
the benchmark builds one engine per load mode and runs recompute on both.

Each stream timestamps every output it receives. A gap between two outputs of
one request counts as "during" if it overlaps a measured resume's window
(submit to first token) and "outside" otherwise; gaps that overlap a warm-up
window are dropped. p99 is reported only where there are at least 100
values; otherwise the maximum stands alone.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time
import uuid

from bench.environment import environment, gpu_state, model_label
from bench.restore_ttft import drop_session

SUFFIX, OUTPUT, DECODE_PROMPT = 64, 16, 128
MIN_FOR_P99 = 100


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def describe(values) -> dict:
    """n, p50, p99 (only when n >= 100) and max."""
    return {"n": len(values), "p50": percentile(values, 0.5),
            "p99": percentile(values, 0.99) if len(values) >= MIN_FOR_P99 else None,
            "max": max(values) if values else None}


async def run_engine(args, async_load: bool) -> dict:
    mode = "async" if async_load else "sync"
    root = Path(tempfile.mkdtemp(prefix="kvsnap-load-", dir=args.store_dir))
    os.environ["KVSNAP_EVENT_DIR"] = str(root / "events")
    if args.trace_dir:
        os.environ["KVSNAP_GIL_PROBE"] = "1"
    (root / "events").mkdir()
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from kvsnap.connector.engine import VLLMEngine, create_async_llm
    from kvsnap.connector.runtime import wait_runtime_identity
    from kvsnap.gateway import SessionGateway, freeze_startup_heap
    from kvsnap.models import SamplingConfig
    from kvsnap.store import SnapshotStore

    rounds = math.ceil((args.resumes + args.warmup) / args.sessions)
    runtime = create_async_llm(root, model=args.model, revision=args.revision,
                               max_model_len=args.length + 1 + rounds * (SUFFIX + OUTPUT) + 16,
                               kv_cache_memory_bytes=args.kv_cache_gib * 1024**3,
                               pinned_pool_bytes=args.pinned_gib * 1024**3,
                               max_num_batched_tokens=args.max_num_batched_tokens, async_load=async_load)
    store = SnapshotStore(root, host_capacity_bytes=args.pinned_gib * 1024**3)
    rng = random.Random(args.seed)
    try:
        identity = await wait_runtime_identity(root, runtime.kvsnap_runtime_id, timeout=600)
        gateway = SessionGateway(VLLMEngine(runtime, store, require_batch_invariant=False), store)
        freeze_startup_heap()  # as the gateway server does once its engine is up
        sampling = SamplingConfig(ignore_eos=True)

        def params(max_tokens):
            return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, detokenize=False,
                                  output_kind=RequestOutputKind.DELTA)

        async def generate(prompt, max_tokens):
            """Send a prompt straight to the engine; return (first-token time, tokens)."""
            first, tokens = None, []
            async for out in runtime.generate({"prompt_token_ids": list(prompt)}, params(max_tokens),
                                              uuid.uuid4().hex):
                first = first or time.perf_counter_ns()
                tokens.extend(out.outputs[0].token_ids)
            return first, tokens

        async def prepare_snapshot():
            session = await gateway.create(identity, sampling=sampling, max_tokens=1 + rounds * OUTPUT,
                                           durability="volatile")
            prompt = [rng.randrange(1000, 100_000) for _ in range(args.length)]
            async for _ in await gateway.start_turn(session.session_id, prompt,
                                                    expected_generation=session.generation, max_tokens=1):
                pass
            await gateway.pause(session.session_id, expected_generation=session.generation)
            return session

        async def prepare_plain():
            prompt = [rng.randrange(1000, 100_000) for _ in range(args.length)]
            _, tokens = await generate(prompt, 1)
            return {"tokens": prompt + tokens}

        async def resume_snapshot(session, suffix):
            start, first = time.perf_counter_ns(), None
            handle = await gateway.resume(session.session_id, expected_generation=session.generation,
                                          ack_seq=len(session.outputs), append_tokens=suffix, max_tokens=OUTPUT)
            async for _ in handle:
                first = first or time.perf_counter_ns()
            ended = time.perf_counter_ns()
            await gateway.pause(session.session_id, expected_generation=session.generation)
            return start, first, ended, time.perf_counter_ns()

        async def resume_plain(state, suffix):
            prompt = state["tokens"] + suffix
            start = time.perf_counter_ns()
            first, tokens = await generate(prompt, OUTPUT)
            state["tokens"] = prompt + tokens
            ended = time.perf_counter_ns()
            return start, first, ended, ended

        async def decode(requests, stop):
            # Back-to-back requests, so a stream outlasts any number of resumes
            # without holding more KV than one request's worth.
            while not stop.is_set():
                prompt = [rng.randrange(1000, 100_000) for _ in range(DECODE_PROMPT)]
                outputs = []
                requests.append(outputs)
                async for out in runtime.generate({"prompt_token_ids": prompt}, params(args.decode_tokens),
                                                  f"decode-{uuid.uuid4().hex[:12]}"):
                    outputs.append(time.perf_counter_ns())
                    if stop.is_set():
                        break

        loop_lag = []

        async def lag_probe():
            # Streams timestamp outputs on this event loop, so a blocked loop
            # shows up as a gap in every stream at once.
            while True:
                before = time.perf_counter_ns()
                await asyncio.sleep(0.001)
                late = time.perf_counter_ns() - before - 1_000_000
                if late > 5_000_000:
                    loop_lag.append({"t0": before, "late_ms": late / 1e6})

        gc_pauses, gc_started = [], []

        def gc_probe(phase, info):
            # A collection holds the GIL and so blocks the event loop too.
            if phase == "start":
                gc_started[:] = [time.perf_counter_ns()]
            elif gc_started:
                took = time.perf_counter_ns() - gc_started[0]
                if took > 5_000_000:
                    gc_pauses.append({"t0": gc_started[0], "ms": took / 1e6, "generation": info["generation"]})

        prober = asyncio.create_task(lag_probe()) if args.trace_dir else None
        if args.trace_dir:
            gc.callbacks.append(gc_probe)
        results = {}
        for arm in ("recompute", mode):
            if arm == "recompute":
                sessions = [await prepare_plain() for _ in range(args.sessions)]
                resume = resume_plain
            else:
                sessions = [await prepare_snapshot() for _ in range(args.sessions)]
                resume = resume_snapshot
            turns, windows, warmups = [], [], []
            loop_lag.clear()
            gc_pauses.clear()
            stop = asyncio.Event()
            streams = [[] for _ in range(args.streams)]
            decoders = [asyncio.create_task(decode(streams[i], stop)) for i in range(args.streams)]
            await asyncio.sleep(args.interval)            # decodes reach steady state
            results.setdefault("gpu_at_measurement", gpu_state())
            for index in range(args.warmup + args.resumes):
                suffix = [rng.randrange(1000, 100_000) for _ in range(SUFFIX)]
                start, first, ended, paused = await resume(sessions[index % args.sessions], suffix)
                (warmups if index < args.warmup else windows).append((start, first))
                if index >= args.warmup:
                    turns.append({"submitted": start, "first_token": first, "turn_ended": ended, "paused": paused})
                await asyncio.sleep(args.interval)
            stop.set()
            await asyncio.gather(*decoders, return_exceptions=True)
            during, outside = [], []
            for requests in streams:
                for outputs in requests:
                    for t0, t1 in zip(outputs, outputs[1:]):
                        if any(t0 < b and t1 > a for a, b in warmups):
                            continue
                        (during if any(t0 < b and t1 > a for a, b in windows) else outside).append((t1 - t0) / 1e6)
            ttft = [(b - a) / 1e6 for a, b in windows]
            results[arm] = {
                "ttft_ms": ttft, "ttft_ms_median": statistics.median(ttft), "ttft_ms_max": max(ttft),
                "ttft_ms_p99": percentile(ttft, 0.99) if len(ttft) >= MIN_FOR_P99 else None,
                "decode_gap_ms": {"during_resume": describe(during), "outside": describe(outside)}}
            print(arm, json.dumps(results[arm]["decode_gap_ms"]), "ttft median",
                  round(results[arm]["ttft_ms_median"], 1), "max", round(max(ttft), 1), flush=True)
            if args.trace_dir:
                # perf_counter_ns and the connector's monotonic_ns share CLOCK_MONOTONIC.
                slow = [{"stream": i, "t0": t0, "t1": t1, "gap_ms": (t1 - t0) / 1e6}
                        for i, requests in enumerate(streams) for outputs in requests
                        for t0, t1 in zip(outputs, outputs[1:]) if t1 - t0 > 20_000_000]
                args.trace_dir.mkdir(parents=True, exist_ok=True)
                (args.trace_dir / f"{arm}-{mode}.json").write_text(
                    json.dumps({"turns": turns, "slow_gaps": slow, "loop_lag": list(loop_lag),
                                "gc_pauses": list(gc_pauses)}))
            if arm != "recompute":
                for session in sessions:
                    await gateway.close(session.session_id, expected_generation=session.generation)
                    drop_session(store, session.session_id)
        if prober:
            prober.cancel()
            gc.callbacks.remove(gc_probe)
        return results
    finally:
        runtime.shutdown()
        store.close()
        if args.trace_dir:
            shutil.copytree(root / "events", args.trace_dir / f"events-{mode}", dirs_exist_ok=True)
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=1000, help="tokens per decode request")
    parser.add_argument("--resumes", type=int, default=54, help="measured resumes per arm")
    parser.add_argument("--warmup", type=int, default=1, help="uncounted resumes per arm")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=int, default=6)
    parser.add_argument("--pinned-gib", type=int, default=14, help="host tier is 3/4 of it; every session must fit")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engines", choices=("both", "async"), default="both",
                        help="async skips the engine that loads inside start_load_kv")
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="keep connector events and every decode gap over 20 ms, with timestamps")
    args = parser.parse_args()
    result = {"model": model_label(args.model), "revision": args.revision, "length": args.length,
              "sessions": args.sessions, "streams": args.streams, "resumes": args.resumes,
              "warmup": args.warmup, "interval_s": args.interval, "decode_tokens": args.decode_tokens,
              "max_num_batched_tokens": args.max_num_batched_tokens, "environment": environment(args.store_dir)}
    for async_load in (False, True) if args.engines == "both" else (True,):
        result["async" if async_load else "sync"] = asyncio.run(run_engine(args, async_load))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
