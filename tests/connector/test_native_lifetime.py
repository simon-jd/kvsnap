"""Deterministic close races with a blocked ABI call, without real CUDA/UAF.

The fake ABI records when native pool/allocation lifetime ends. A blocked copy
models the GIL-released interval before or during submission in ctypes/pybind.
"""
import ctypes as C
import gc
import threading
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

from kvsnap.connector.dataplane import CudaPool, PinnedCapacityError


class Function:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class BlockingCudaABI:
    def __init__(self):
        self.entered, self.continue_copy = threading.Event(), threading.Event()
        self.destroyed, self.released = False, False
        self.fail_close = False
        self.kvsnap_pool_abi_version = Function(lambda: 2)
        self.kvsnap_cuda_error = Function(lambda: b"CPU ABI mock error")
        self.kvsnap_pool_create = Function(lambda capacity, device: 101)
        self.kvsnap_pool_destroy = Function(self.destroy)
        self.kvsnap_pool_close_checked = Function(self.close_checked)
        self.kvsnap_pool_reserve = Function(self.reserve)
        self.kvsnap_pool_copy = Function(self.copy)
        self.kvsnap_pool_copy_2d = Function(self.copy)
        self.kvsnap_pool_wait = Function(lambda pool, allocation: 0)
        self.kvsnap_pool_ready = Function(lambda pool, allocation: 1)
        self.kvsnap_pool_release = Function(self.release)

    def destroy(self, pool):
        self.destroyed = True

    def close_checked(self, pool):
        if self.fail_close:
            return -1
        self.destroyed = True
        return 0

    def reserve(self, pool, size, ident, pointer):
        C.cast(ident, C.POINTER(C.c_uint64))[0] = 7
        C.cast(pointer, C.POINTER(C.c_void_p))[0] = 128
        return 0

    def release(self, pool, allocation):
        self.released = True
        return 0

    def copy(self, *args):
        self.entered.set()
        if not self.continue_copy.wait(3):
            raise RuntimeError("test copy gate timed out")
        if self.destroyed or self.released:
            raise AssertionError("native lifetime ended during an active copy call")
        return 0


class NativeLifetime(unittest.TestCase):
    def test_failed_wait_and_release_survive_caller_dropping_allocation(self):
        abi = BlockingCudaABI()
        with patch("kvsnap.connector.dataplane.C.CDLL", return_value=abi):
            pool = CudaPool(4096)
        allocation = pool.reserve(128)
        tensor = type("WeakTensor", (), {})()
        allocation._tensor_references = [tensor]
        allocation_ref, tensor_ref = weakref.ref(allocation), weakref.ref(tensor)
        abi.kvsnap_pool_wait = Function(lambda pool, allocation: -1)
        abi.kvsnap_pool_release = Function(lambda pool, allocation: -1)
        with self.assertRaises(RuntimeError):
            allocation.wait()
        with self.assertRaises(RuntimeError):
            allocation.close()
        del allocation, tensor
        gc.collect()
        try:
            self.assertIsNotNone(allocation_ref(), "failed release lost Python allocation owner")
            self.assertIsNotNone(tensor_ref(), "unknown DMA completion lost tensor storage reference")
            abi.kvsnap_pool_release = Function(abi.release)
            allocation_ref().close()
            gc.collect()
            self.assertIsNone(tensor_ref(), "successful release did not drop tensor reference")
            self.assertIsNone(allocation_ref(), "successful release kept an unreachable allocation")
        finally:
            pool.close()

    def test_failed_checked_close_retains_unreachable_allocation_until_retry(self):
        abi = BlockingCudaABI()
        with patch("kvsnap.connector.dataplane.C.CDLL", return_value=abi):
            pool = CudaPool(4096)
        allocation = pool.reserve(128)
        tensor = type("WeakTensor", (), {})()
        allocation._tensor_references = [tensor]
        allocation_ref, tensor_ref = weakref.ref(allocation), weakref.ref(tensor)
        del allocation, tensor
        abi.fail_close = True
        try:
            with self.assertRaises(RuntimeError):
                pool.close()
            gc.collect()
            self.assertIsNotNone(allocation_ref(), "failed close lost allocation owner")
            self.assertIsNotNone(tensor_ref(), "failed close lost unresolved tensor storage")
        finally:
            abi.fail_close = False
            pool.close()
        gc.collect()
        self.assertIsNone(allocation_ref())
        self.assertIsNone(tensor_ref())

    def run_race(self, *, strided, close_pool):
        abi = BlockingCudaABI()
        with patch("kvsnap.connector.dataplane.C.CDLL", return_value=abi):
            pool = CudaPool(4096)
        allocation = pool.reserve(128)
        errors, closed = [], threading.Event()

        def copy():
            try:
                if strided:
                    allocation.copy_2d(256, 64, 0, 64, 64, 2, to_host=True)
                else:
                    allocation.copy(256, 128, to_host=True)
            except BaseException as exc:
                errors.append(exc)

        def close():
            try:
                (pool.close if close_pool else allocation.close)()
            except BaseException as exc:
                errors.append(exc)
            finally:
                closed.set()

        # The mock ABI never dereferences these dummy pointers or runs a GPU
        # op; it only holds the call open to probe the close/copy window.
        copier = threading.Thread(target=copy)
        closer = threading.Thread(target=close)
        copier.start()
        self.assertTrue(abi.entered.wait(2))
        closer.start()
        try:
            self.assertFalse(closed.wait(0.05), "close returned while native copy still borrowed its resource")
        finally:
            abi.continue_copy.set()
            copier.join(2)
            closer.join(2)
            pool.close()
        self.assertFalse(copier.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])

    def test_pool_close_waits_for_inflight_copy(self):
        self.run_race(strided=False, close_pool=True)

    def test_allocation_close_waits_for_inflight_strided_copy(self):
        self.run_race(strided=True, close_pool=False)

    def test_failed_pool_close_preserves_handle_and_tensor_references_for_retry(self):
        abi = BlockingCudaABI()
        with patch("kvsnap.connector.dataplane.C.CDLL", return_value=abi):
            pool = CudaPool(4096)
        allocation = pool.reserve(128)
        tensor = object()
        allocation._tensor_references = [tensor]
        abi.fail_close = True
        with self.assertRaises(RuntimeError):
            pool.close()
        self.assertEqual(pool.handle, 101)
        self.assertFalse(abi.destroyed)
        self.assertEqual(allocation._tensor_references, [tensor])
        abi.fail_close = False
        pool.close()
        self.assertIsNone(pool.handle)

    def test_failed_event_wait_keeps_tensor_storage_alive(self):
        abi = BlockingCudaABI()
        with patch("kvsnap.connector.dataplane.C.CDLL", return_value=abi):
            pool = CudaPool(4096)
        allocation = pool.reserve(128)
        tensor = object()
        allocation._tensor_references = [tensor]
        abi.kvsnap_pool_wait = Function(lambda pool, allocation: -1)
        with self.assertRaises(RuntimeError):
            allocation.wait()
        self.assertEqual(allocation._tensor_references, [tensor])
        abi.kvsnap_pool_wait = Function(lambda pool, allocation: 0)
        allocation.wait()
        self.assertEqual(allocation._tensor_references, [])
        allocation.close()
        pool.close()

    def test_incompatible_native_abi_is_rejected_before_creating_a_pool(self):
        abi = BlockingCudaABI()
        abi.kvsnap_pool_abi_version = Function(lambda: 1)
        create = Mock(return_value=101)
        abi.kvsnap_pool_create = Function(create)
        with patch("kvsnap.connector.dataplane.C.CDLL", return_value=abi):
            with self.assertRaisesRegex(RuntimeError, "unsupported.*ABI"):
                CudaPool(4096)
        create.assert_not_called()


class CheckedNativeClose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("g++"):
            raise unittest.SkipTest("g++ is required for the compiled fake-CUDA control test")
        cls.temporary = tempfile.TemporaryDirectory(prefix="kvsnap-native-close-")
        root = Path(__file__).resolve().parents[2]
        library = Path(cls.temporary.name) / "close-check.so"
        subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-pthread",
                        "-I", str(root / "tests/native"), str(root / "csrc/dataplane/cuda_pool.cpp"),
                        str(root / "tests/native/fake_cuda.cpp"), "-o", str(library)],
                       check=True, capture_output=True, text=True, timeout=60)
        cls.lib = C.CDLL(str(library))
        cls.lib.kvsnap_test_fail_next_stream_sync.argtypes = []
        cls.lib.kvsnap_test_fail_next_stream_sync.restype = None
        cls.lib.kvsnap_test_pending.argtypes = []
        cls.lib.kvsnap_test_pending.restype = C.c_int
        cls.lib.kvsnap_test_last_copy_stream.argtypes = []
        cls.lib.kvsnap_test_last_copy_stream.restype = C.c_size_t
        cls.library = library

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_compiled_failed_close_preserves_pending_allocation_until_retry_drains(self):
        pool = CudaPool(16, library=self.library)
        source = C.create_string_buffer(b"0123456789abcdef", 16)
        allocation = pool.reserve(16)
        allocation.copy(C.addressof(source), 16, to_host=True)
        self.assertEqual(self.lib.kvsnap_test_pending(), 1)
        handle = pool.handle
        self.lib.kvsnap_test_fail_next_stream_sync()
        with self.assertRaises(RuntimeError):
            pool.close()
        self.assertEqual(pool.handle, handle)
        self.assertEqual(self.lib.kvsnap_test_pending(), 1)
        self.assertFalse(allocation.ready())
        with self.assertRaises(PinnedCapacityError):     # the pending allocation still fills the pool
            pool.reserve(1)
        pool.close()
        self.assertEqual(self.lib.kvsnap_test_pending(), 0)
        self.assertIsNone(pool.handle)

    def test_compiled_borrowed_stream_matches_copy_stream_and_pool_lifetime(self):
        first = CudaPool(16, library=self.library)
        second = CudaPool(16, library=self.library)
        source = C.create_string_buffer(b"0123456789abcdef", 16)
        try:
            with first.borrow_copy_stream() as first_stream, second.borrow_copy_stream() as second_stream:
                self.assertNotEqual(first_stream, second_stream)
                with first.borrow_copy_stream() as same_stream:
                    self.assertEqual(first_stream, same_stream)
                for pool, stream in ((first, first_stream), (second, second_stream)):
                    allocation = pool.reserve(16)
                    allocation.copy(C.addressof(source), 16, to_host=True)
                    self.assertEqual(self.lib.kvsnap_test_last_copy_stream(), stream)
                    allocation.wait()
                    self.assertEqual(allocation.read(), b"0123456789abcdef")
                    allocation.close()
        finally:
            first.close()
            second.close()
        with self.assertRaisesRegex(RuntimeError, "pool closed"):
            with first.borrow_copy_stream():
                self.fail("closed pool yielded a diagnostic stream")


if __name__ == "__main__":
    unittest.main()
