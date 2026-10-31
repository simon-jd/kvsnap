"""What a turn-2 restore costs the requests already running.

    .venv-gpu/bin/python -m bench.restore_under_load --output results/under-load-qwen3-1.7b.json

Before measuring, ``--sessions`` sessions each prefill ``--length`` tokens and
pause into the host-memory tier. Then ``--streams`` decode requests start
(128-token prompts, greedy, no EOS) and, while they run, the sessions resume
one every ``--interval`` seconds (64 appended tokens, 16 generated). Each
resume is served one of three ways:

  recompute  the snapshot is invalidated first, so the connector misses and
             vLLM prefills the whole prompt alongside the decodes;
  sync       the connector loads the snapshot inside start_load_kv, so every
             running request waits for the copy;
  async      the loader thread copies it while the batch keeps running.

``sync`` and ``async`` need differently configured engines, so the benchmark
builds one engine per load mode and runs recompute on both. Each decode
stream timestamps every output it receives. A gap between two outputs counts
as "during" if it overlaps a resume's window (submit to first token). The
report gives decode gap percentiles during and outside those windows, and the
resumed turns' TTFT.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
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
from bench.restore_ttft import drop_session

SUFFIX, OUTPUT, DECODE_PROMPT = 64, 16, 128


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


async def run_engine(args, async_load: bool) -> dict:
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

    runtime = create_async_llm(root, model=args.model, revision=args.revision,
                               max_model_len=args.length + SUFFIX + OUTPUT + 16,
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

        async def prepare():
            session = await gateway.create(identity, sampling=sampling, max_tokens=1 + OUTPUT,
                                           durability="volatile")
            prompt = [rng.randrange(1000, 100_000) for _ in range(args.length)]
            async for _ in await gateway.start_turn(session.session_id, prompt,
                                                    expected_generation=session.generation, max_tokens=1):
                pass
            await gateway.pause(session.session_id, expected_generation=session.generation)
            return session

        async def decode(stream_id, outputs):
            params = SamplingParams(max_tokens=args.decode_tokens, temperature=0.0, ignore_eos=True,
                                    detokenize=False, output_kind=RequestOutputKind.DELTA)
            prompt = [rng.randrange(1000, 100_000) for _ in range(DECODE_PROMPT)]
            async for out in runtime.generate({"prompt_token_ids": prompt}, params, f"decode-{stream_id}-{uuid.uuid4().hex[:8]}"):
                outputs.append((time.perf_counter_ns(), len(out.outputs[0].token_ids)))

        async def resume(session, recompute):
            if recompute:
                store.invalidate_snapshot(session.snapshot_id, "benchmark: recompute arm")
            suffix = [rng.randrange(1000, 100_000) for _ in range(SUFFIX)]
            start, first = time.perf_counter_ns(), None
            handle = await gateway.resume(session.session_id, expected_generation=session.generation,
                                          ack_seq=len(session.outputs), append_tokens=suffix, max_tokens=OUTPUT)
            async for _ in handle:
                first = first or time.perf_counter_ns()
            ended = time.perf_counter_ns()
            await gateway.pause(session.session_id, expected_generation=session.generation)
            turns.append({"submitted": start, "first_token": first, "turn_ended": ended,
                          "paused": time.perf_counter_ns()})
            return start, first

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
        for arm in ("recompute", "async" if async_load else "sync"):
            sessions = [await prepare() for _ in range(args.sessions)]
            turns = []
            loop_lag.clear()
            gc_pauses.clear()
            streams = [[] for _ in range(args.streams)]
            decoders = [asyncio.create_task(decode(i, streams[i])) for i in range(args.streams)]
            await asyncio.sleep(args.interval)            # decodes reach steady state
            results.setdefault("gpu_at_measurement", gpu_state())
            windows = []
            for session in sessions:
                windows.append(await resume(session, arm == "recompute"))
                await asyncio.sleep(args.interval)
            for task in decoders:
                task.cancel()
            await asyncio.gather(*decoders, return_exceptions=True)
            during, outside = [], []
            for outputs in streams:
                for (t0, _), (t1, n) in zip(outputs, outputs[1:]):
                    gap = (t1 - t0) / 1e6
                    (during if any(t0 < b and t1 > a for a, b in windows) else outside).append(gap)
            ttft = [(b - a) / 1e6 for a, b in windows]
            results[arm] = {
                "ttft_ms": ttft, "ttft_ms_median": statistics.median(ttft),
                "decode_gap_ms": {name: {"n": len(gaps), "p50": percentile(gaps, 0.5), "p99": percentile(gaps, 0.99),
                                         "max": max(gaps) if gaps else None}
                                  for name, gaps in (("during_resume", during), ("outside", outside))}}
            print(arm, json.dumps(results[arm]["decode_gap_ms"]), "ttft", [round(t) for t in ttft], flush=True)
            if args.trace_dir:
                # perf_counter_ns and the connector's monotonic_ns share CLOCK_MONOTONIC.
                slow = [{"stream": i, "t0": t0, "t1": t1, "gap_ms": (t1 - t0) / 1e6}
                        for i, outputs in enumerate(streams)
                        for (t0, _), (t1, _) in zip(outputs, outputs[1:]) if t1 - t0 > 20_000_000]
                args.trace_dir.mkdir(parents=True, exist_ok=True)
                (args.trace_dir / f"{arm}-{'async' if async_load else 'sync'}.json").write_text(
                    json.dumps({"turns": turns, "slow_gaps": slow, "loop_lag": list(loop_lag),
                                "gc_pauses": list(gc_pauses)}))
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
            shutil.copytree(root / "events", args.trace_dir / f"events-{'async' if async_load else 'sync'}",
                            dirs_exist_ok=True)
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=4000)
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
              "sessions": args.sessions, "streams": args.streams,
              "max_num_batched_tokens": args.max_num_batched_tokens, "environment": environment(args.store_dir)}
    for async_load in (False, True) if args.engines == "both" else (True,):
        result["async" if async_load else "sync"] = asyncio.run(run_engine(args, async_load))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
