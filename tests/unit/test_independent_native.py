"""Compile the real C++ control path against fault-injectable test CUDA ABI.

Passing proves error-handling control flow in a CPU shim, never real CUDA safety.
"""
import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeErrorPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("g++"):
            raise unittest.SkipTest("g++ unavailable: C++ error-path shim cannot be compiled")
        cls.temporary = tempfile.TemporaryDirectory(prefix="kvsnap-native-test-")
        library = Path(cls.temporary.name) / "libfakecuda.so"
        # -Bsymbolic binds the pool to this fake even if a real CUDA runtime is
        # already loaded in the process (torch with CUDA, in the GPU environment).
        result = subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-Wall", "-Wextra",
            "-Werror", "-I", str(ROOT / "tests/native"), str(ROOT / "csrc/dataplane/cuda_pool.cpp"),
            str(ROOT / "tests/native/fake_cuda.cpp"), "-o", str(library)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        cls.lib = C.CDLL(str(library))
        declarations = {
            "kvsnap_pool_create": ([C.c_size_t, C.c_int], C.c_void_p),
            "kvsnap_pool_destroy": ([C.c_void_p], None),
            "kvsnap_pool_reserve": ([C.c_void_p, C.c_size_t, C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)], C.c_int),
            "kvsnap_pool_copy": ([C.c_void_p, C.c_uint64, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int, C.c_size_t], C.c_int),
            "kvsnap_pool_copy_2d": ([C.c_void_p, C.c_uint64, C.c_size_t, C.c_size_t, C.c_void_p, C.c_size_t,
                                  C.c_size_t, C.c_size_t, C.c_int, C.c_size_t], C.c_int),
            "kvsnap_pool_release": ([C.c_void_p, C.c_uint64], C.c_int),
            "kvsnap_pool_wait": ([C.c_void_p, C.c_uint64], C.c_int),
            "kvsnap_pool_ready": ([C.c_void_p, C.c_uint64], C.c_int),
            "kvsnap_test_pending": ([], C.c_int),
            "kvsnap_test_fail_next_record": ([], None),
            "kvsnap_test_fail_next_stream_sync": ([], None),
        }
        for name, (args, result_type) in declarations.items():
            function = getattr(cls.lib, name)
            function.argtypes, function.restype = args, result_type

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_record_failure_never_releases_inflight_pinned_bytes(self):
        pool = self.lib.kvsnap_pool_create(16, 0)
        self.assertTrue(pool)
        try:
            allocation, pointer = C.c_uint64(), C.c_void_p()
            self.assertEqual(self.lib.kvsnap_pool_reserve(pool, 16, C.byref(allocation), C.byref(pointer)), 0)
            destination = C.create_string_buffer(16)
            self.lib.kvsnap_test_fail_next_record()
            self.assertEqual(self.lib.kvsnap_pool_copy(pool, allocation.value, 0, destination, 16, 0, 0), -1)
            released = self.lib.kvsnap_pool_release(pool, allocation.value)
            # Either retain/quarantine the allocation, or drain all queued work
            # before successful release. A stale/unrecorded event is insufficient.
            if released == 0:
                self.assertEqual(self.lib.kvsnap_test_pending(), 0,
                                 "allocation released while DMA remains queued after event-record failure")
        finally:
            self.lib.kvsnap_pool_destroy(pool)

    def test_strided_copy_moves_one_layer_of_consecutive_blocks_and_checks_bounds(self):
        # Pinned layout: 3 packed blocks of 8 bytes, layer 1 = bytes 4..7 of each.
        # Device layer tensor: block pitch 6 bytes, layer row = first 4 bytes.
        pool = self.lib.kvsnap_pool_create(24, 0)
        try:
            allocation, pointer = C.c_uint64(), C.c_void_p()
            self.assertEqual(self.lib.kvsnap_pool_reserve(pool, 24, C.byref(allocation), C.byref(pointer)), 0)
            C.memmove(pointer.value, bytes(range(24)), 24)
            device = C.create_string_buffer(b"\xee" * 18, 18)
            self.assertEqual(self.lib.kvsnap_pool_copy_2d(pool, allocation.value, 4, 8, device, 6, 4, 3, 0, 0), 0)
            self.assertEqual(self.lib.kvsnap_pool_wait(pool, allocation.value), 0)
            self.assertEqual(device.raw, bytes([4, 5, 6, 7, 0xee, 0xee, 12, 13, 14, 15, 0xee, 0xee,
                                                20, 21, 22, 23, 0xee, 0xee]))
            # The last row would end at 4 + 2*8 + 5 = 25 > 24 bytes.
            self.assertEqual(self.lib.kvsnap_pool_copy_2d(pool, allocation.value, 4, 8, device, 6, 5, 3, 0, 0), -1)
            # Pitches narrower than the row would overlap rows.
            self.assertEqual(self.lib.kvsnap_pool_copy_2d(pool, allocation.value, 0, 2, device, 6, 4, 3, 0, 0), -1)
            self.assertEqual(self.lib.kvsnap_pool_release(pool, allocation.value), 0)
        finally:
            self.lib.kvsnap_pool_destroy(pool)

    def test_normal_async_copy_drains_before_release_and_checks_capacity(self):
        pool = self.lib.kvsnap_pool_create(16, 0)
        try:
            allocation, pointer = C.c_uint64(), C.c_void_p()
            self.assertEqual(self.lib.kvsnap_pool_reserve(pool, 16, C.byref(allocation), C.byref(pointer)), 0)
            extra, extra_ptr = C.c_uint64(), C.c_void_p()
            self.assertEqual(self.lib.kvsnap_pool_reserve(pool, 1, C.byref(extra), C.byref(extra_ptr)), -1)
            destination = C.create_string_buffer(16)
            self.assertEqual(self.lib.kvsnap_pool_copy(pool, allocation.value, 0, destination, 17, 0, 0), -1)
            self.assertEqual(self.lib.kvsnap_pool_copy(pool, allocation.value, 0, destination, 16, 0, 0), 0)
            self.assertEqual(self.lib.kvsnap_pool_ready(pool, allocation.value), 0)
            self.assertEqual(self.lib.kvsnap_pool_release(pool, allocation.value), 0)
            self.assertEqual(self.lib.kvsnap_test_pending(), 0)
            self.assertEqual(self.lib.kvsnap_pool_reserve(pool, 16, C.byref(extra), C.byref(extra_ptr)), 0)
        finally:
            self.lib.kvsnap_pool_destroy(pool)

    def test_failed_stream_drain_keeps_allocation_unavailable_for_reuse(self):
        pool = self.lib.kvsnap_pool_create(16, 0)
        try:
            allocation, pointer = C.c_uint64(), C.c_void_p()
            self.assertEqual(self.lib.kvsnap_pool_reserve(pool, 16, C.byref(allocation), C.byref(pointer)), 0)
            destination = C.create_string_buffer(16)
            self.lib.kvsnap_test_fail_next_record()
            self.assertEqual(self.lib.kvsnap_pool_copy(pool, allocation.value, 0, destination, 16, 0, 0), -1)
            self.lib.kvsnap_test_fail_next_stream_sync()
            self.assertEqual(self.lib.kvsnap_pool_release(pool, allocation.value), -1)
            self.assertGreater(self.lib.kvsnap_test_pending(), 0)
            extra, extra_ptr = C.c_uint64(), C.c_void_p()
            self.assertEqual(self.lib.kvsnap_pool_reserve(pool, 1, C.byref(extra), C.byref(extra_ptr)), -1)
            self.assertEqual(self.lib.kvsnap_pool_release(pool, allocation.value), 0)
            self.assertEqual(self.lib.kvsnap_test_pending(), 0)
        finally:
            self.lib.kvsnap_pool_destroy(pool)


if __name__ == "__main__":
    unittest.main()
