"""H2D bandwidth of each restore copy pattern, on KV caches shaped like Qwen3-8B.

    python -m bench.h2d_patterns --out results/h2d-patterns.json

36 layers, 64 KiB per (layer, block), 512 blocks (8k tokens, 1.2 GB). Each
pattern runs through the connector's own data plane and is checked by reading
the blocks back. Block ids are either one consecutive run (a fresh engine) or
shuffled (a long-running engine's free list).

    peak            one contiguous 1.2 GB pinned -> device copy
    2d              staging buffer, one cudaMemcpy2DAsync per (layer, run)
    per_block       one pinned allocation per block (host tier), one copy per (layer, block)
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

from kvsnap.connector.dataplane import TorchBlockDataPlane

LAYERS, BLOCKS, CACHE_BLOCKS, BLOCK_SIZE = 36, 512, 1024, 16


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
    args = parser.parse_args()
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
            back = plane.save(ids, len(ids) * BLOCK_SIZE)
            for i in (0, len(ids) // 2, len(ids) - 1):
                expected = ctypes.string_at(staged.pointer + i * plane.block_bytes, plane.block_bytes)
                if back[i] != expected:
                    raise AssertionError(f"block {i} differs after load")
            for tensor in caches.values():
                tensor.zero_()

        dense = torch.empty(nbytes, dtype=torch.uint8, device=device)
        pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        result = {"bytes": nbytes, "blocks": BLOCKS, "layers": LAYERS, "layer_block_bytes": 64 * 1024,
                  "peak": nbytes / measure(lambda: dense.copy_(pinned, non_blocking=True)) / 1e9}
        for name, ids in orders.items():
            patterns = {
                "2d": lambda: plane.load_staged(ids, staged),
                "per_block": lambda: plane.load_pinned(ids, singles),
                "scatter": lambda: plane.load_scatter(ids, [(staged, i * plane.block_bytes) for i in range(BLOCKS)]),
                "per_block_scatter": lambda: plane.load_scatter(ids, [(a, 0) for a in singles]),
            }
            for pattern, fn in patterns.items():
                seconds = measure(fn)
                fn()
                check(ids)
                result[f"{pattern}/{name}"] = nbytes / seconds / 1e9
                print(f"{pattern:>18} {name:<12} {nbytes / seconds / 1e9:6.2f} GB/s  {seconds * 1e3:6.1f} ms", flush=True)
        for allocation in singles:
            allocation.close()
    args.out.write_text(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in result.items()}, indent=2))
    print(f"peak {result['peak']:.2f} GB/s -> {args.out}")


if __name__ == "__main__":
    main()
