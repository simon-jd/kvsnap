"""H2D bandwidth of each restore copy pattern, on KV caches shaped like Qwen3-8B.

    python -m bench.h2d_patterns --out results/h2d-patterns.json

36 layers, 64 KiB per (layer, block), 512 blocks (8k tokens, 1.2 GB). Each
pattern runs through the connector's own data plane and is checked by reading
the blocks back. Block ids are either one consecutive run (a fresh engine) or
shuffled (a long-running engine's free list). Every pattern is timed as the
median of 5 copies in each of ``--rounds`` rounds; the file reports the median
round and every round.

    peak            one contiguous 1.2 GB pinned -> device copy
    2d              staging buffer, one cudaMemcpy2DAsync per (layer, run)
    per_block       one pinned allocation per block (host tier), one copy per (layer, block)

``2d`` and ``per_block`` are the data plane's earlier load paths and live here
now; ``scatter`` and ``per_block_scatter`` call the connector's own
``load_scatter``. Every pattern is checked by reading the blocks back with
the connector's ``save_gather``.
    scatter         staging buffer, contiguous H2D in 64 MiB chunks + index_copy_ per layer
    per_block_scatter  per-block allocations, one 2.25 MiB copy per block + index_copy_
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch

from bench.environment import environment, gpu_state
from kvsnap.connector.dataplane import TorchBlockDataPlane

LAYERS, BLOCKS, CACHE_BLOCKS, BLOCK_SIZE = 36, 512, 1024, 16


def copy_runs(plane, allocation, block_ids, *, to_host):
    """Copy ``block_ids`` between the KV cache and a packed pinned buffer.

    The buffer holds block i at ``i * block_bytes``, its layers back to back.
    For each layer, every run of consecutive block ids is one
    cudaMemcpy2DAsync (host pitch = packed block, device pitch = the layer's
    block stride), so an N-block transfer costs layers x runs copies.
    """
    runs, start = [], 0
    for i in range(1, len(block_ids) + 1):
        if i == len(block_ids) or block_ids[i] != block_ids[i - 1] + 1:
            runs.append((start, i - start))
            start = i
    layer_offset = 0
    for layer, tensor in zip(plane.layers, plane.caches.values()):
        # Block b of a layer starts at base + b * pitch.
        base, pitch = tensor.data_ptr(), tensor.stride(0) * tensor.element_size()
        for first, count in runs:
            allocation.copy_2d(base + block_ids[first] * pitch, pitch, first * plane.block_bytes + layer_offset,
                               plane.block_bytes, layer["bytes"], count, to_host=to_host)
        layer_offset += layer["bytes"]


def load_2d(plane, block_ids, staged):
    """H2D a filled staging buffer into ``block_ids`` and wait for the copies."""
    plane._validate_ids(block_ids)
    copy_runs(plane, staged, block_ids, to_host=False)
    staged.wait()


def load_per_block(plane, block_ids, allocations):
    """H2D one allocation per block; the copy stream is serial, so enqueue all, then fence each."""
    plane._validate_ids(block_ids)
    for block_id, allocation in zip(block_ids, allocations):
        copy_runs(plane, allocation, [block_id], to_host=False)
    for allocation in allocations:
        allocation.wait()


def measure(fn, repeat=5):
    fn()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, help="independent rounds of every pattern; medians reported")
    args = parser.parse_args()
    measured = environment()
    device = torch.device("cuda:0")
    # [blocks, heads, tokens, dim] per layer, 32 x 16 x 64 bf16 = 64 KiB per block.
    caches = {f"layer.{i:02d}": torch.zeros((CACHE_BLOCKS, 32, BLOCK_SIZE, 64), dtype=torch.bfloat16, device=device)
              for i in range(LAYERS)}
    nbytes = BLOCKS * LAYERS * 64 * 1024
    plane = TorchBlockDataPlane(caches, BLOCK_SIZE, 3 * nbytes + (256 << 20))
    rng = random.Random(0)
    orders = {"consecutive": list(range(100, 100 + BLOCKS)),
              "shuffled": rng.sample(range(1, CACHE_BLOCKS), BLOCKS)}

    with plane.staging(BLOCKS) as staged:
        source = torch.randint(-2**15, 2**15, (nbytes // 2,), dtype=torch.int16, device=device)
        torch.cuda.synchronize()
        host = torch.empty(nbytes // 2, dtype=torch.int16, pin_memory=True)
        host.copy_(source.cpu())
        import ctypes
        ctypes.memmove(staged.pointer, host.data_ptr(), nbytes)
        singles = [plane.pool.reserve(plane.block_bytes) for _ in range(BLOCKS)]
        for i, allocation in enumerate(singles):
            ctypes.memmove(allocation.pointer, staged.pointer + i * plane.block_bytes, plane.block_bytes)

        def check(ids):
            size = plane.block_bytes
            torch.cuda.synchronize()
            with plane.staging(len(ids)) as back:
                plane.save_gather(ids, [(back, i * size) for i in range(len(ids))], len(ids) * BLOCK_SIZE)
                for i in (0, len(ids) // 2, len(ids) - 1):
                    if back.read_range(i * size, size) != ctypes.string_at(staged.pointer + i * size, size):
                        raise AssertionError(f"block {i} differs after load")
            for tensor in caches.values():
                tensor.zero_()

        dense = torch.empty(nbytes, dtype=torch.uint8, device=device)
        pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        measured["gpu_at_measurement"] = gpu_state()
        by_round = {}
        for _ in range(args.rounds):
            by_round.setdefault("peak", []).append(nbytes / measure(lambda: dense.copy_(pinned, non_blocking=True)) / 1e9)
            for name, ids in orders.items():
                patterns = {
                    "2d": lambda: load_2d(plane, ids, staged),
                    "per_block": lambda: load_per_block(plane, ids, singles),
                    "scatter": lambda: plane.load_scatter(ids, [(staged, i * plane.block_bytes) for i in range(BLOCKS)]),
                    "per_block_scatter": lambda: plane.load_scatter(ids, [(a, 0) for a in singles]),
                }
                for pattern, fn in patterns.items():
                    seconds = measure(fn)
                    fn()
                    check(ids)
                    by_round.setdefault(f"{pattern}/{name}", []).append(nbytes / seconds / 1e9)
                    print(f"{pattern:>18} {name:<12} {nbytes / seconds / 1e9:6.2f} GB/s  {seconds * 1e3:6.1f} ms",
                          flush=True)
        result = {"bytes": nbytes, "blocks": BLOCKS, "layers": LAYERS, "layer_block_bytes": 64 * 1024,
                  "rounds": args.rounds, **{key: statistics.median(values) for key, values in by_round.items()},
                  "gb_per_s_by_round": {key: [round(v, 2) for v in values] for key, values in by_round.items()}}
        for allocation in singles:
            allocation.close()
    result["environment"] = measured
    args.out.write_text(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in result.items()}, indent=2)
                        + "\n")
    print(f"peak {result['peak']:.2f} GB/s -> {args.out}")


if __name__ == "__main__":
    main()
