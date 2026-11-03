"""The restore kernel on a real GPU: pinned host blocks land bit for bit in their KV cache rows.

The caches are laid out as vLLM 0.31.0 registers them (a [blocks, heads,
tokens, width] view of [blocks, tokens, heads, width] storage, layers carved
from one buffer), and the pinned memory comes from the connector's own pool.
Skipped without CUDA, Triton or the native pool library.
"""
import random
import unittest

try:
    import torch
    import triton  # noqa: F401
    CUDA = torch.cuda.is_available()
except ImportError:
    torch, CUDA = None, False

LAYERS, BLOCKS, BLOCK_SIZE, HEADS, WIDTH = 6, 96, 16, 8, 256


@unittest.skipUnless(CUDA, "needs a CUDA device and Triton")
class RestoreKernel(unittest.TestCase):
    def setUp(self):
        from kvsnap.connector.dataplane import TorchBlockDataPlane, resolve_cuda_library
        if not resolve_cuda_library().is_file():
            self.skipTest("native CUDA pool library not built")
        generator = torch.Generator(device="cuda").manual_seed(0)
        backing = torch.randint(-2**15, 2**15, (LAYERS, BLOCKS, BLOCK_SIZE, HEADS, WIDTH), dtype=torch.int16,
                                device="cuda", generator=generator).view(torch.bfloat16)
        self.backing = backing
        self.caches = {f"model.layers.{i}.self_attn.attn": backing[i].permute(0, 2, 1, 3) for i in range(LAYERS)}
        self.plane = TorchBlockDataPlane(self.caches, BLOCK_SIZE, 64 << 20)
        self.plane.warm_up()

    def tearDown(self):
        self.plane.close()

    def bits(self):
        return self.backing.view(torch.int16).clone()

    def test_a_saved_snapshot_restores_bit_for_bit_into_shuffled_blocks(self):
        rng = random.Random(1)
        saved = rng.sample(range(1, BLOCKS), 20)
        restored = rng.sample([b for b in range(1, BLOCKS) if b not in saved], 20)
        size = self.plane.block_bytes
        before = self.bits()
        for kind in ("staging", "per-block"):
            with self.subTest(kind):
                owned = ([self.plane.pool.reserve(size * len(saved))] if kind == "staging"
                         else [self.plane.pool.reserve(size) for _ in saved])
                targets = ([(owned[0], i * size) for i in range(len(saved))] if kind == "staging"
                           else [(allocation, 0) for allocation in owned])
                self.plane.save_gather(saved, targets, len(saved) * BLOCK_SIZE)
                self.assertEqual(self.plane.load_scatter(restored, targets), 1)
                torch.cuda.synchronize()
                after = self.bits()
                expected = before.clone()
                expected[:, restored] = before[:, saved]
                self.assertTrue(torch.equal(after, expected))
                self.backing.view(torch.int16).copy_(before)
                for allocation in owned:
                    allocation.close()


if __name__ == "__main__":
    unittest.main()
