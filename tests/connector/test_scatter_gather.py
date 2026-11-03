"""The production copy paths, ``save_gather`` and ``load_scatter``, on real CPU tensors.

The geometry is what vLLM 0.31.0 registers for Qwen3-8B: 36 layers named
``model.layers.N.self_attn.attn``, each a [blocks, 8 heads, 16 tokens, 256]
bf16 view of [blocks, 16 tokens, 8 heads, 256] storage (K and V side by side),
so one (layer, block) span is 64 KiB and a packed block is 2.25 MiB. The layers
are carved out of one buffer, so every layer but the first has a storage
offset. Only CUDA is replaced: the pool's copies are memmoves between host
buffers and the staging tensor, the restore kernel is a memmove per (block,
layer) job driven by the tables the data plane builds for it, and streams and
events do nothing. The gather, chunking, coalescing of adjacent sources, the
restore's address tables and bounds checks, and tail zeroing are the
connector's own code. Without torch these tests are skipped.
"""
from contextlib import contextmanager, nullcontext
import ctypes as C
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:          # the CPU environment has no torch
    torch = None

from kvsnap.connector.dataplane import TorchBlockDataPlane

LAYERS, HEADS, BLOCK_SIZE, WIDTH = 36, 8, 16, 256


class HostAllocation:
    """A pinned allocation stand-in: plain host memory, copies are memmoves."""
    def __init__(self, size):
        self.size, self.waits, self.closed = size, 0, False
        self.buffer = C.create_string_buffer(size)
        self.pointer = C.addressof(self.buffer)

    def _open(self):
        if self.closed:
            raise RuntimeError("pinned allocation closed")

    def copy(self, device_ptr, size, offset=0, *, to_host, producer_stream=0):
        if min(size, offset) < 0 or offset + size > self.size:
            raise ValueError("copy outside the allocation")
        host = self.pointer + offset
        C.memmove(host if to_host else device_ptr, device_ptr if to_host else host, size)

    def wait(self):
        self.waits += 1

    def zero(self, offset, size):
        if offset < 0 or offset + size > self.size:
            raise ValueError("zero outside the allocation")
        C.memset(self.pointer + offset, 0, size)

    def bytes(self, offset, size):
        return C.string_at(self.pointer + offset, size)


class HostPool:
    @contextmanager
    def borrow_copy_stream(self):
        yield 1

    @contextmanager
    def operation(self):
        yield


def emulated_kernel(sources, blocks, layer_offsets, layer_bases, layer_pitches, layer_words):
    """copy_kernel's semantics: each (block, layer) span from its host address to its cache row."""
    layers = list(zip(*(table.tolist() for table in (layer_offsets, layer_bases, layer_pitches, layer_words))))
    for source, block in zip(sources.tolist(), blocks.tolist()):
        for offset, base, pitch, words in layers:
            C.memmove(base + block * pitch, source + offset, words * 8)


restore_kernel = patch("kvsnap.connector.dataplane.copy_host_blocks", new=emulated_kernel)


class Pinnable:
    """torch.tensor's result, whose pin_memory() needs no CUDA here."""
    def __init__(self, tensor):
        self.tensor = tensor

    def pin_memory(self):
        return self.tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


class Event:
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


class CpuTorch:
    """torch, with the few CUDA calls of the copy paths made CPU no-ops."""
    cuda = SimpleNamespace(stream=lambda stream: nullcontext(), Event=Event,
                           ExternalStream=lambda handle, device=None: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    def tensor(self, *args, **kwargs):
        return Pinnable(torch.tensor(*args, **kwargs))


class CudaLabelled:
    """A CPU tensor that reports a CUDA device, for registration's device check."""
    device = SimpleNamespace(type="cuda", index=0)

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        return self.tensor[index]


def register(caches, block_size=BLOCK_SIZE):
    """Register as the connector does, then hand the copy paths the CPU tensors."""
    with patch.dict(sys.modules, {"torch": CpuTorch()}), \
         patch("kvsnap.connector.dataplane.CudaPool", return_value=HostPool()):
        plane = TorchBlockDataPlane({name: CudaLabelled(t) for name, t in caches.items()}, block_size, 1 << 30)
    plane.caches = dict(sorted(caches.items()))
    return plane


def qwen3_8b_caches(blocks):
    """vLLM's per-layer KV views, filled with distinct random bits."""
    generator = torch.Generator().manual_seed(0)
    backing = torch.randint(-2**15, 2**15, (LAYERS, blocks, BLOCK_SIZE, HEADS, WIDTH),
                            dtype=torch.int16, generator=generator).view(torch.bfloat16)
    return {f"model.layers.{i}.self_attn.attn": backing[i].permute(0, 2, 1, 3) for i in range(LAYERS)}


def raw(tensor):
    """A tensor's bytes (no numpy: the CPU torch wheel does not bring it)."""
    tensor = tensor.contiguous()
    return C.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


def block_span(view, block):
    """A layer's bytes for one block, in storage order (tokens, heads, width)."""
    return raw(view[block].permute(1, 0, 2))


def bits(view):
    """A layer in storage order, as int16: random bf16 bits include NaNs, which never compare equal."""
    return view.permute(0, 2, 1, 3).view(torch.int16)


def snapshot(caches):
    return {name: bits(view).clone() for name, view in caches.items()}


def targets_for(kind, count, size):
    """Per-block allocations (the host tier) or one staging buffer (disk)."""
    if kind == "per-block":
        return [(HostAllocation(size), 0) for _ in range(count)]
    staged = HostAllocation(size * count)
    return [(staged, i * size) for i in range(count)]


@unittest.skipIf(torch is None, "needs torch (CPU tensors only)")
@restore_kernel
class Qwen3EightBGeometry(unittest.TestCase):
    def setUp(self):
        self.caches = qwen3_8b_caches(12)
        self.plane = register(self.caches)
        self.names = sorted(self.caches)

    def test_registration_matches_the_layout_vllm_reports(self):
        layer = self.plane.layers[0]
        self.assertEqual((layer["shape"], layer["stride"], layer["bytes"]), ([8, 16, 256], [256, 2048, 1], 65536))
        self.assertEqual(self.plane.block_bytes, 36 * 65536)
        self.assertEqual([l["name"] for l in self.plane.layers], self.names)

    def test_save_gather_packs_layers_in_name_order_and_zeroes_the_tail(self):
        block_ids, tokens = [7, 2, 9, 3, 4], 4 * BLOCK_SIZE + 5
        size = self.plane.block_bytes
        before = snapshot(self.caches)
        for kind, copies in (("per-block", 5), ("staging", 3)):
            with self.subTest(kind):
                targets = targets_for(kind, len(block_ids), size)
                # Two blocks per chunk: three gathers, and the staging buffer's
                # adjacent blocks go out in one copy per chunk.
                self.assertEqual(self.plane.save_gather(block_ids, targets, tokens, chunk_bytes=2 * size), copies)
                for index, block in enumerate(block_ids):
                    allocation, offset = targets[index]
                    for position, name in enumerate(self.names):
                        expected = bytearray(block_span(self.caches[name], block))
                        if index == len(block_ids) - 1:
                            expected[5 * HEADS * WIDTH * 2:] = bytes((BLOCK_SIZE - 5) * HEADS * WIDTH * 2)
                        got = allocation.bytes(offset + position * 65536, 65536)
                        self.assertEqual(got, bytes(expected), f"block {block}, layer {name}")
                self.assertTrue(all(allocation.waits for allocation, _ in targets))
                for name in self.names:          # the KV cache is only read
                    self.assertTrue(torch.equal(bits(self.caches[name]), before[name]))

    def test_load_scatter_writes_only_the_target_blocks(self):
        block_ids = [10, 0, 5, 6, 1]
        size = self.plane.block_bytes
        for kind in ("per-block", "staging"):
            with self.subTest(kind):
                sources = targets_for(kind, len(block_ids), size)
                payload = torch.randint(-2**15, 2**15, (len(block_ids), size // 2), dtype=torch.int16,
                                        generator=torch.Generator().manual_seed(len(kind)))
                for (allocation, offset), row in zip(sources, payload):
                    C.memmove(allocation.pointer + offset, row.data_ptr(), size)
                before = snapshot(self.caches)
                self.assertEqual(self.plane.load_scatter(block_ids, sources), 1)     # one launch
                for name_index, name in enumerate(self.names):
                    after = bits(self.caches[name])
                    for block in range(12):
                        if block in block_ids:
                            row = raw(payload[block_ids.index(block)])
                            expected = row[name_index * 65536:(name_index + 1) * 65536]
                            self.assertEqual(block_span(self.caches[name], block), expected, f"{name} {block}")
                        else:
                            self.assertTrue(torch.equal(after[block], before[name][block]), f"{name} {block}")

    def test_load_scatter_checks_every_source_before_the_launch(self):
        size = self.plane.block_bytes
        before = snapshot(self.caches)
        short = HostAllocation(size - 8)
        closed = HostAllocation(size)
        closed.closed = True
        cases = {"outside its allocation": [(HostAllocation(size), 0), (short, 0)],
                 "closed": [(HostAllocation(size), 0), (closed, 0)],
                 "negative offset": [(HostAllocation(2 * size), -8), (HostAllocation(size), 0)]}
        for reason, sources in cases.items():
            with self.subTest(reason), patch("kvsnap.connector.dataplane.copy_host_blocks") as launch:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.plane.load_scatter([1, 2], sources)
                launch.assert_not_called()
        for name in self.names:
            self.assertTrue(torch.equal(bits(self.caches[name]), before[name]))

    def test_a_failed_launch_drains_the_copy_stream_before_the_sources_can_be_released(self):
        self.plane._load_tables()
        self.plane._copy_stream = Mock()
        sources = targets_for("staging", 2, self.plane.block_bytes)
        with patch("kvsnap.connector.dataplane.copy_host_blocks", side_effect=RuntimeError("launch failed")):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                self.plane.load_scatter([1, 2], sources)
        self.plane._copy_stream.synchronize.assert_called_once_with()

    def test_a_saved_snapshot_restores_into_other_blocks(self):
        # The default 64 MiB chunk holds all five blocks: one gather; one launch restores them.
        saved, tokens = [3, 4, 5, 8, 11], 5 * BLOCK_SIZE
        restored = [0, 1, 2, 6, 7]
        size = self.plane.block_bytes
        targets = targets_for("staging", len(saved), size)
        expected = {name: [block_span(self.caches[name], b) for b in saved] for name in self.names}
        self.assertEqual(self.plane.save_gather(saved, targets, tokens), 1)
        self.assertEqual(self.plane.load_scatter(restored, targets), 1)
        for name in self.names:
            self.assertEqual([block_span(self.caches[name], b) for b in restored], expected[name])


@unittest.skipIf(torch is None, "needs torch (CPU tensors only)")
@restore_kernel
class OtherLayouts(unittest.TestCase):
    """Dense [B,H,N,C] and [B,N,H,C] blocks, alone or interleaved with another layer's."""

    def test_tail_zeroing_and_restore_follow_the_actual_strides(self):
        shape = (5, 3, 4, 8)            # blocks, heads, tokens, width; 192 bytes a block
        for strides in ((96, 32, 8, 1), (96, 8, 24, 1), (192, 32, 8, 1), (192, 8, 24, 1)):
            with self.subTest(strides=strides):
                storage = torch.arange(5 * strides[0], dtype=torch.int16).view(torch.bfloat16)
                cache = storage.as_strided(shape, strides)
                plane = register({"layer.0": cache}, block_size=4)
                before = storage.clone()
                targets = targets_for("per-block", 2, plane.block_bytes)
                plane.save_gather([1, 3], targets, 6)
                for index, block in enumerate((1, 3)):
                    span = before[block * strides[0]:block * strides[0] + 96].view(torch.int16).clone()
                    if index == 1:      # tokens 2 and 3 of the last block are past the end
                        for head in range(3):
                            for token in (2, 3):
                                start = head * strides[1] + token * strides[2]
                                span[start:start + 8] = 0
                    self.assertEqual(targets[index][0].bytes(0, 192), raw(span))
                self.assertTrue(torch.equal(storage.view(torch.int16), before.view(torch.int16)))
                # Blocks 0 and 4 take the payloads; everything else, including
                # an interleaved neighbour layer's half, keeps its bytes.
                plane.load_scatter([0, 4], targets)
                expected = before.view(torch.int16).clone()
                for block, (allocation, _) in zip((0, 4), targets):
                    payload = torch.frombuffer(bytearray(allocation.bytes(0, 192)), dtype=torch.int16)
                    expected[block * strides[0]:block * strides[0] + 96] = payload
                self.assertTrue(torch.equal(storage.view(torch.int16), expected))


if __name__ == "__main__":
    unittest.main()
