"""The restore's H2D copy as one kernel that reads pinned host memory directly.

A packed block is its layers' spans back to back. The kernel walks
(block, layer) jobs and copies each span from its host address straight into
that layer's KV cache row, so the bytes cross PCIe once and land in place:
there is no device staging buffer, no separate scatter, and one launch per
restore instead of a copy per block and a scatter per layer and chunk.
Pinned memory from ``cudaHostAlloc`` is mapped into the device's address space
under unified addressing, so the kernel takes host pointers as they are.

The launch uses ``PROGRAMS`` programs, a small share of the GPU's SMs, so the
copy runs beside the decoding batch instead of taking the whole device.
"""
from __future__ import annotations

import triton
import triton.language as tl

PROGRAMS, WORDS, WARPS = 12, 1024, 4


@triton.jit
def _copy_host_blocks(sources, blocks, layer_offsets, layer_bases, layer_pitches, layer_words,
                      n_blocks, n_layers, WORDS: tl.constexpr):
    """Job j copies layer j % n_layers of packed block j // n_layers, in 8-byte words."""
    program = tl.program_id(0)
    offsets = tl.arange(0, WORDS)
    job = program
    while job < n_blocks * n_layers:
        block = job // n_layers
        layer = job % n_layers
        source = (tl.load(sources + block) + tl.load(layer_offsets + layer)).to(tl.pointer_type(tl.int64))
        target = (tl.load(layer_bases + layer) + tl.load(blocks + block) * tl.load(layer_pitches + layer)
                  ).to(tl.pointer_type(tl.int64))
        words = tl.load(layer_words + layer)
        for start in range(0, words, WORDS):
            index = start + offsets
            mask = index < words
            tl.store(target + index, tl.load(source + index, mask=mask), mask=mask)
        job += tl.num_programs(0)


def copy_host_blocks(sources, blocks, layer_offsets, layer_bases, layer_pitches, layer_words):
    """Launch on the current stream. ``sources``: host address of each packed block; ``blocks``: its
    target block id; the ``layer_*`` tables: per layer, its offset in a packed block, the cache's
    base address and block pitch in bytes, and its span in words. All int64 CUDA tensors."""
    n_blocks, n_layers = sources.numel(), layer_offsets.numel()
    _copy_host_blocks[(PROGRAMS,)](sources, blocks, layer_offsets, layer_bases, layer_pitches, layer_words,
                                   n_blocks, n_layers, WORDS=WORDS, num_warps=WARPS)
