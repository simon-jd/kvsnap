"""Upstream v0.31 logical [B,H,N,C] layout regressions, using CPU descriptors.

The descriptor carries real shape/stride arithmetic but advertises a CUDA
device to isolate registration validation. No allocation or CUDA is executed.
"""
from types import SimpleNamespace
import ctypes as C
import sys
import unittest
from unittest.mock import Mock, patch

from kvsnap.connector.dataplane import TorchBlockDataPlane


class TensorDescriptor:
    def __init__(self, shape, strides, *, device=None, pointer=0):
        self.shape, self._strides = tuple(shape), tuple(strides)
        self.device = device or SimpleNamespace(type="cuda", index=0)
        self.ndim = len(shape)
        self.dtype = "torch.bfloat16"
        self.pointer = pointer

    def stride(self, axis=None):
        return self._strides if axis is None else self._strides[axis]

    def numel(self):
        product = 1
        for size in self.shape:
            product *= size
        return product

    def element_size(self):
        return 2

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self._strides)):
            if size > 1 and stride != expected:
                return False
            expected *= size
        return True

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("descriptor only implements block indexing")
        return TensorDescriptor(self.shape[1:], self._strides[1:], device=self.device,
                                pointer=self.pointer + index * self._strides[0] * self.element_size())


class CpuAllocation:
    """Memory-copy oracle only; this intentionally has no CUDA/event semantics."""
    def __init__(self, size):
        self.size, self.closed = size, False
        self.copies = []
        self.buffer = C.create_string_buffer(size)
        self.pointer = C.addressof(self.buffer)

    def copy(self, device_ptr, size, offset=0, *, to_host, producer_stream=0):
        if min(size, offset) < 0 or offset + size > self.size:
            raise ValueError("copy exceeds CPU test allocation")
        source, destination = (device_ptr, self.pointer + offset) if to_host else (self.pointer + offset, device_ptr)
        C.memmove(destination, source, size)

    def copy_2d(self, device_ptr, device_pitch, host_offset, host_pitch, width, height, *,
                to_host, producer_stream=0):
        if height and host_offset + (height - 1) * host_pitch + width > self.size:
            raise ValueError("2D copy exceeds CPU test allocation")
        self.copies.append((height, to_host))
        for row in range(height):
            host, device = self.pointer + host_offset + row * host_pitch, device_ptr + row * device_pitch
            C.memmove(host if to_host else device, device if to_host else host, width)

    def wait(self):
        pass

    def zero(self, offset, size):
        if offset < 0 or offset + size > self.size:
            raise ValueError("zero exceeds CPU test allocation")
        C.memset(self.pointer + offset, 0, size)

    def read(self):
        return C.string_at(self.pointer, self.size)

    def read_range(self, offset, size):
        return C.string_at(self.pointer + offset, size)

    def write(self, data):
        if len(data) != self.size:
            raise ValueError("wrong payload size")
        C.memmove(self.pointer, data, self.size)

    def close(self):
        self.closed = True


class RegisteredLayout(unittest.TestCase):
    def register(self, strides):
        cache = TensorDescriptor((5, 3, 4, 8), strides)
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
             patch("kvsnap.connector.dataplane.CudaPool", return_value=Mock()):
            return TorchBlockDataPlane({"layer.0": cache}, 4, 4096)

    def test_default_lbn_hc_view_registers_without_being_logically_contiguous(self):
        # Physical [B,N,H,C], logical [B,H,N,C]. Upstream FlashAttention
        # expresses no layout preference, so the resolver selects LBNHC.
        plane = self.register((96, 8, 24, 1))
        self.assertEqual(plane.block_bytes, 192)

    def test_block_outer_layer_interleaving_does_not_require_global_contiguity(self):
        # Physical [B,L,H,N,C] for two layers: a layer's block stride is twice
        # its dense page span, while each individual HNC page is contiguous.
        plane = self.register((192, 32, 8, 1))
        self.assertEqual(plane.block_bytes, 192)

    def test_overlapping_inner_axes_are_rejected_before_pool_allocation(self):
        # H and N would address overlapping elements; copying a contiguous
        # byte span cannot represent this as an immutable logical KV block.
        with self.assertRaises(ValueError):
            self.register((96, 8, 8, 1))

    def test_overlapping_block_stride_is_rejected(self):
        with self.assertRaises(ValueError):
            self.register((48, 32, 8, 1))

    def test_tail_zero_and_restore_follow_actual_strides_without_touching_neighbor_blocks(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(cuda_stream=17)))
        for strides in ((96, 32, 8, 1), (96, 8, 24, 1), (192, 32, 8, 1), (192, 8, 24, 1)):
            with self.subTest(strides=strides):
                size = 5 * strides[0] * 2
                buffer = C.create_string_buffer(size)
                C.memset(C.addressof(buffer), 0xA7, size)
                descriptor = TensorDescriptor((5, 3, 4, 8), strides, pointer=C.addressof(buffer))
                allocated = []

                def reserve(length):
                    item = CpuAllocation(length)
                    allocated.append(item)
                    return item

                pool = SimpleNamespace(reserve=reserve)
                with patch.dict(sys.modules, {"torch": fake_torch}), \
                     patch("kvsnap.connector.dataplane.CudaPool", return_value=pool):
                    plane = TorchBlockDataPlane({"layer.0": descriptor}, 4, 4096)
                for block in (1, 3):
                    for head in range(3):
                        for token in range(4):
                            for column in range(8):
                                position = block * strides[0] + head * strides[1] + token * strides[2] + column
                                C.c_uint16.from_address(descriptor.pointer + position * 2).value = 1 + block * 1000 + head * 100 + token * 10 + column
                source_before = C.string_at(descriptor.pointer, size)
                packed = plane.save([1, 3], 6)
                self.assertEqual(C.string_at(descriptor.pointer, size), source_before)
                self.assertTrue(all(item.closed for item in allocated))
                for index, source_block in enumerate((1, 3)):
                    expected = bytearray(source_before[source_block * strides[0] * 2: source_block * strides[0] * 2 + 192])
                    if index == 1:
                        for head in range(3):
                            for token in (2, 3):
                                position = (head * strides[1] + token * strides[2]) * 2
                                expected[position:position + 16] = bytes(16)
                    self.assertEqual(packed[index], bytes(expected))
                # Restore into blocks 0 and 4; the untouched pages and any
                # interleaved-other-layer space must retain their old bytes.
                plane.load([0, 4], packed)
                expected_memory = bytearray(source_before)
                for target, payload in zip((0, 4), packed):
                    start = target * strides[0] * 2
                    expected_memory[start:start + 192] = payload
                self.assertEqual(C.string_at(descriptor.pointer, size), bytes(expected_memory))
                self.assertTrue(all(item.closed for item in allocated))


    def test_consecutive_blocks_move_in_one_strided_copy_per_layer(self):
        # Blocks 1..3 are consecutive, so save and restore each issue a single
        # 2D copy of height 3 instead of three per-block copies.
        strides = (96, 8, 24, 1)
        size = 6 * strides[0] * 2
        buffer = C.create_string_buffer(size)
        for i in range(size // 2):
            C.c_uint16.from_address(C.addressof(buffer) + 2 * i).value = i % 65521
        descriptor = TensorDescriptor((6, 3, 4, 8), strides, pointer=C.addressof(buffer))
        allocated = []

        def reserve(length):
            allocated.append(CpuAllocation(length))
            return allocated[-1]
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda d: SimpleNamespace(cuda_stream=0)))
        with patch.dict(sys.modules, {"torch": fake_torch}), \
             patch("kvsnap.connector.dataplane.CudaPool", return_value=SimpleNamespace(reserve=reserve)):
            plane = TorchBlockDataPlane({"layer.0": descriptor}, 4, 4096)
        before = C.string_at(descriptor.pointer, size)
        packed = plane.save([1, 2, 3], 12)
        self.assertEqual(allocated[-1].copies, [(3, True)])
        self.assertEqual(b"".join(packed), before[1 * 192:4 * 192])
        C.memset(descriptor.pointer, 0, size)
        plane.load([2, 3, 4], packed)
        self.assertEqual(allocated[-1].copies, [(3, False)])
        self.assertEqual(C.string_at(descriptor.pointer + 2 * 192, 3 * 192), before[192:4 * 192])
        self.assertEqual(C.string_at(descriptor.pointer, 2 * 192), bytes(2 * 192))


if __name__ == "__main__":
    unittest.main()
