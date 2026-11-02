"""Upstream v0.31 logical [B,H,N,C] layout regressions, using CPU descriptors.

The descriptor carries real shape/stride arithmetic but advertises a CUDA
device to isolate registration validation. No allocation or CUDA is executed.
The copy paths run on real CPU tensors in test_scatter_gather.py.
"""
from types import SimpleNamespace
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


if __name__ == "__main__":
    unittest.main()
