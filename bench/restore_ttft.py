"""Turn-2 time to first token: restore a saved session's KV vs recompute the prompt.

    .venv-gpu/bin/python -m bench.restore_ttft --lengths 1024 2048 4096 8192 16384 --reps 5

One engine, one GPU. Each measurement runs a fresh session:

1. turn 1 prefills L prompt tokens and generates 1 token;
2. ``pause`` saves the turn's KV (D2H, then host memory or disk) and vLLM frees
   the GPU blocks (prefix caching is off, so nothing survives on the GPU);
3. turn 2 appends 64 tokens and generates 16, either by
   * ``recompute``: the same full prompt sent straight to the engine, or
   * ``restore-host``: KV restored from a host-memory (pinned) snapshot,
   * ``restore-disk``: KV restored from an on-disk snapshot still in page cache,
   * ``restore-cold``: the same after evicting the snapshot files from page cache.

TTFT is client-side: from submitting turn 2 to its first token. The restore
arms include the gateway's manifest checks, SQLite lease, reading and
CRC-checking every block, the H2D copy and the 64-token suffix prefill. Each
length gets one discarded warm-up sample of every arm first.

The 16 greedy tokens and their log-probabilities are compared with the
recompute's. With ``--batch-invariant`` they must be bit-identical; with
vLLM's default kernels the prefill of a 64-token suffix and of the whole
prompt take different kernel paths, so small logprob differences are
expected. ``--perturb`` is the negative control: it flips the sign of one
head's key vector for one token in each disk snapshot (fixing the checksums
so the load accepts it) and should make the logprobs differ.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time
import uuid

SUFFIX, OUTPUT = 64, 16
ARMS = ("recompute", "restore-host", "restore-disk", "restore-cold")


def evict_from_page_cache(root: Path, manifest: dict) -> None:
    """Drop a durable snapshot's blob pages so the next read goes to the device."""
    for segment in manifest["segments"]:
        path = root / "blobs" / f"{segment['blob_id']}.blob"
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def drop_session(store, session_id: str) -> None:
    """Delete a finished session's snapshots (children first) and reclaim their blobs."""
    manifests = [m for m in store.list_manifests(include_invalid=True) if m["session_id"] == session_id]
    for manifest in sorted(manifests, key=lambda m: m["generation"], reverse=True):
        store.delete_snapshot(manifest["snapshot_id"])
    store.collect_garbage()


def perturb_snapshot(store, root: Path, snapshot_id: str) -> None:
    """Negative control: flip the sign of 64 bf16 values in the first block, keeping checksums valid."""
    from kvsnap.store.crc32c import crc32c
    from kvsnap.store.store import HEADER
    manifest = store.get_manifest(snapshot_id)
    segment = manifest["segments"][0]
    path = root / "blobs" / f"{segment['blob_id']}.blob"
    raw = bytearray(path.read_bytes())
    for i in range(64):
        raw[HEADER.size + 2 * i + 1] ^= 0x80
    payload = bytes(raw[HEADER.size:])
    magic, version, length, _, reserved = HEADER.unpack(bytes(raw[:HEADER.size]))
    checksum = crc32c(payload)
    path.write_bytes(HEADER.pack(magic, version, length, checksum, reserved) + payload)
    segment["crc32c"] = checksum
    store.db.execute("UPDATE snapshots SET manifest_json=? WHERE snapshot_id=?",
                     (json.dumps(manifest, sort_keys=True, separators=(",", ":")), snapshot_id))
    store.db.execute("UPDATE blobs SET crc32c=? WHERE blob_id=?", (checksum, segment["blob_id"]))


def h2d_breakdown(event_dir: Path, since_ns: int, first_token_ns: int) -> dict | None:
    """Split one restore's TTFT into phases, from connector telemetry (CLOCK_MONOTONIC).

    admit: submit -> scheduler allocates blocks and admits the external hit;
    read: worker reads + CRC-checks the snapshot into pinned memory;
    h2d: strided copies into the GPU blocks; after: suffix prefill to first token.
    """
    events = []
    for path in event_dir.glob("kvsnap-events-*.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["monotonic_ns"] >= since_ns and record["event"] in {"load_allocated", "h2d_started", "h2d_completed"}:
                events.append(record)
    by = {e["event"]: e for e in sorted(events, key=lambda e: e["monotonic_ns"])}
    if not {"load_allocated", "h2d_started", "h2d_completed"} <= set(by):
        return None
    allocated, started, completed = (by[k]["monotonic_ns"] for k in ("load_allocated", "h2d_started", "h2d_completed"))
    return {"admit_ms": (allocated - since_ns) / 1e6, "read_ms": (started - allocated) / 1e6,
            "h2d_ms": (completed - started) / 1e6, "after_ms": (first_token_ns - completed) / 1e6,
            "bytes": by["h2d_completed"]["bytes"], "copies": by["h2d_completed"].get("copies")}


async def run(args) -> dict:
    root = Path(tempfile.mkdtemp(prefix="kvsnap-bench-", dir=args.store_dir))
    event_dir = root / "events"
    event_dir.mkdir()
    os.environ["KVSNAP_EVENT_DIR"] = str(event_dir)
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if args.batch_invariant else "0"
    from vllm import SamplingParams
    from kvsnap.connector.engine import VLLMEngine, create_async_llm
    from kvsnap.connector.runtime import wait_runtime_identity
    from kvsnap.gateway import SessionGateway
    from kvsnap.models import SamplingConfig
    from kvsnap.store import SnapshotStore

    max_len = max(args.lengths) + SUFFIX + OUTPUT + 16
    revision = None if args.revision.lower() == "none" else args.revision
    runtime = create_async_llm(root, model=args.model, revision=revision, max_model_len=max_len,
                               kv_cache_memory_bytes=args.kv_cache_gib * 1024**3,
                               pinned_pool_bytes=args.pinned_gib * 1024**3,
                               max_num_batched_tokens=args.max_num_batched_tokens,
                               cudagraph_mode=args.cudagraph_mode, async_load=args.async_load)
    store = SnapshotStore(root, host_capacity_bytes=args.pinned_gib * 1024**3)
    try:
        identity = await wait_runtime_identity(root, runtime.kvsnap_runtime_id, timeout=600)
        engine = VLLMEngine(runtime, store, require_batch_invariant=args.batch_invariant)
        gateway = SessionGateway(engine, store)
        sampling = SamplingConfig(ignore_eos=True)
        rng = random.Random(args.seed)

        async def recompute(prompt):
            params = SamplingParams(max_tokens=OUTPUT, temperature=0.0, ignore_eos=True, detokenize=False,
                                    logprobs=1)
            start, first, out = time.perf_counter(), None, None
            async for out in runtime.generate({"prompt_token_ids": list(prompt)}, params, uuid.uuid4().hex):
                first = first or time.perf_counter()
            tokens = list(out.outputs[0].token_ids)
            logprobs = [step[token].logprob for step, token in zip(out.outputs[0].logprobs, tokens)]
            return (first - start) * 1e3, tokens, logprobs

        async def restore(length, durability, cold):
            prompt = [rng.randrange(1000, 100_000) for _ in range(length)]
            suffix = [rng.randrange(1000, 100_000) for _ in range(SUFFIX)]
            session = await gateway.create(identity, sampling=sampling, max_tokens=1 + OUTPUT,
                                           durability=durability)
            turn1 = time.monotonic_ns()
            async for _ in await gateway.start_turn(session.session_id, prompt,
                                                    expected_generation=session.generation, max_tokens=1):
                pass
            saved = time.perf_counter()
            manifest = await gateway.pause(session.session_id, expected_generation=session.generation)
            save_ms = (time.perf_counter() - saved) * 1e3
            # Recompute while the engine is idle: the turn-1 save has committed
            # and nothing else is queued.
            reference_ttft, reference, reference_logprobs = await recompute(tuple(manifest.token_ids) + tuple(suffix))
            if args.perturb and durability != "volatile":
                perturb_snapshot(store, root, manifest.snapshot_id)
            if cold:
                evict_from_page_cache(root, store.get_manifest(manifest.snapshot_id))
            marker = time.monotonic_ns()
            start, first, tokens, logprobs = time.perf_counter(), None, [], []
            handle = await gateway.resume(session.session_id, expected_generation=session.generation,
                                          ack_seq=len(session.outputs), append_tokens=suffix, max_tokens=OUTPUT)
            first_ns = None
            async for token in handle:
                if first is None:
                    first, first_ns = time.perf_counter(), time.monotonic_ns()
                tokens.append(token.token_id)
                logprobs.append(token.logprob)
            ttft = (first - start) * 1e3
            breakdown = h2d_breakdown(event_dir, marker, first_ns)
            # Turn 2 also saves at its end; wait for that so it cannot overlap the next sample.
            await gateway.pause(session.session_id, expected_generation=session.generation)
            await gateway.close(session.session_id, expected_generation=session.generation)
            drop_session(store, session.session_id)
            return {"ttft_ms": ttft, "save_ms": save_ms, "match": tokens == reference,
                    "logprobs_equal": logprobs == reference_logprobs,
                    "max_logprob_delta": max(abs(a - b) for a, b in zip(logprobs, reference_logprobs)),
                    "recompute_ttft_ms": reference_ttft, **(breakdown or {})}

        samples = defaultdict(list)
        for length in args.lengths:
            # One discarded sample of each arm warms CUDA graphs, the allocator
            # and the page cache for this length.
            for durability in ("volatile", "durable"):
                await restore(length, durability, False)
            for rep in range(args.reps):
                for arm, durability, cold in (("restore-host", "volatile", False),
                                              ("restore-disk", "durable", False),
                                              ("restore-cold", "durable", True)):
                    result = await restore(length, durability, cold)
                    samples[(length, arm)].append(result)
                    samples[(length, "recompute")].append({"ttft_ms": result["recompute_ttft_ms"]})
                    print(f"L={length} rep={rep} {arm}: ttft {result['ttft_ms']:.1f} ms "
                          f"(recompute {result['recompute_ttft_ms']:.1f} ms) save {result['save_ms']:.0f} ms "
                          f"admit {result.get('admit_ms', float('nan')):.1f} read {result.get('read_ms', float('nan')):.1f} "
                          f"h2d {result.get('h2d_ms', float('nan')):.1f} after {result.get('after_ms', float('nan')):.1f} "
                          f"copies {result.get('copies')} match={result['match']} "
                          f"logprobs_equal={result['logprobs_equal']} max_delta={result['max_logprob_delta']:.2e}",
                          flush=True)
        return summarise(samples, args)
    finally:
        runtime.shutdown()
        store.close()
        if not args.keep_store:
            shutil.rmtree(root, ignore_errors=True)


def summarise(samples, args) -> dict:
    rows = []
    for (length, arm), values in sorted(samples.items()):
        ttft = [v["ttft_ms"] for v in values]
        row = {"context_tokens": length, "arm": arm, "n": len(values),
               "ttft_ms_median": statistics.median(ttft), "ttft_ms_min": min(ttft), "ttft_ms_max": max(ttft),
               "ttft_ms": ttft}
        for key in ("save_ms", "admit_ms", "read_ms", "h2d_ms", "after_ms", "bytes", "copies"):
            present = [v[key] for v in values if v.get(key) is not None]
            if present:
                row[f"{key}_median"] = statistics.median(present)
        if arm != "recompute":
            row["tokens_match"] = sum(v["match"] for v in values)
            row["logprobs_equal"] = sum(v["logprobs_equal"] for v in values)
            row["max_logprob_delta"] = max(v["max_logprob_delta"] for v in values)
        rows.append(row)
    return {"model": args.model, "revision": args.revision, "batch_invariant": args.batch_invariant,
            "cudagraph_mode": args.cudagraph_mode, "async_load": args.async_load, "perturb": args.perturb,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "suffix_tokens": SUFFIX, "output_tokens": OUTPUT, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                        help="'none' when --model is a local snapshot directory")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=int, default=6)
    parser.add_argument("--pinned-gib", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--cudagraph-mode", default=None, help="default: vLLM's own CUDA graph mode")
    parser.add_argument("--batch-invariant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perturb", action="store_true", help="negative control for the logprob comparison")
    parser.add_argument("--store-dir", default=None, help="parent directory for the snapshot store (an SSD)")
    parser.add_argument("--keep-store", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
