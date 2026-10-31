"""Real C++ I/O with injected syscalls; no CUDA or power-loss claims."""
from __future__ import annotations

import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from kvsnap.store import CorruptSnapshotError, SnapshotStore
from kvsnap.store.native_io import NativeIO, get_io_backend
from kvsnap.models import KVIdentity, SamplingConfig, SnapshotManifest

ROOT = Path(__file__).resolve().parents[2]


class NativeIOAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("g++"):
            raise unittest.SkipTest("g++ is required for native I/O fault injection")
        cls.temporary = tempfile.TemporaryDirectory(prefix="kvsnap-native-io-")
        library = Path(cls.temporary.name) / "libkvsnap_io_faults.so"
        cls.library = library
        result = subprocess.run([
            "g++", "-std=c++17", "-shared", "-fPIC", "-pthread", "-Wall", "-Wextra", "-Werror",
            str(ROOT / "csrc/dataplane/io.cpp"), str(ROOT / "tests/native/io_faults.cpp"),
            "-Wl,--wrap=pwrite", "-Wl,--wrap=pread", "-Wl,--wrap=fdatasync", "-o", str(library),
        ], capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cls.lib = C.CDLL(str(library))
        declarations = {
            "kvsnap_pwrite_all": ([C.c_int, C.c_void_p, C.c_size_t, C.c_int64, C.c_int], C.c_int),
            "kvsnap_pread_all": ([C.c_int, C.c_void_p, C.c_size_t, C.c_int64], C.c_int),
            "kvsnap_crc32c": ([C.c_void_p, C.c_size_t], C.c_uint32),
            "kvsnap_crc32c_extend": ([C.c_void_p, C.c_size_t, C.c_uint32], C.c_uint32),
            "kvsnap_test_io_reset": ([], None),
            "kvsnap_test_io_fault": ([C.c_int, C.c_int, C.c_size_t, C.c_int, C.c_int], None),
            "kvsnap_test_io_calls": ([C.c_int], C.c_int),
            "kvsnap_test_io_hold": ([C.c_int, C.c_int], None),
        }
        for name, (arguments, result_type) in declarations.items():
            function = getattr(cls.lib, name)
            function.argtypes, function.restype = arguments, result_type

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.lib.kvsnap_test_io_reset()
        self.file = tempfile.TemporaryFile()
        self.addCleanup(self.file.close)

    def test_short_writes_and_eintr_preserve_bytes_and_offsets(self):
        self.lib.kvsnap_test_io_fault(0, 2, 3, 0, 0)
        payload = b"independent-native-io"
        self.assertEqual(self.lib.kvsnap_pwrite_all(self.file.fileno(), payload, len(payload), 5, 0), 0)
        self.assertEqual(os.pread(self.file.fileno(), 100, 0), b"\0" * 5 + payload)
        self.assertEqual(self.lib.kvsnap_test_io_calls(0), 2 + (len(payload) + 2) // 3)
        self.assertEqual(self.lib.kvsnap_test_io_calls(2), 0)

    def test_zero_write_is_error_and_does_not_spin_or_sync(self):
        self.lib.kvsnap_test_io_fault(0, 0, 0, 0, 1)
        self.assertEqual(self.lib.kvsnap_pwrite_all(self.file.fileno(), b"abc", 3, 0, 1), errno.EIO)
        self.assertEqual(self.lib.kvsnap_test_io_calls(0), 1)
        self.assertEqual(self.lib.kvsnap_test_io_calls(2), 0)

    def test_enospc_is_returned_without_false_durable_success(self):
        self.lib.kvsnap_test_io_fault(0, 1, 0, errno.ENOSPC, 0)
        self.assertEqual(self.lib.kvsnap_pwrite_all(self.file.fileno(), b"abc", 3, 0, 1), errno.ENOSPC)
        self.assertEqual(self.lib.kvsnap_test_io_calls(0), 2)
        self.assertEqual(self.lib.kvsnap_test_io_calls(2), 0)
        self.assertEqual(os.fstat(self.file.fileno()).st_size, 0)

    def test_sync_retries_eintr_and_propagates_eio(self):
        self.lib.kvsnap_test_io_fault(2, 2, 0, errno.EIO, 0)
        self.assertEqual(self.lib.kvsnap_pwrite_all(self.file.fileno(), b"abc", 3, 0, 1), errno.EIO)
        self.assertEqual(self.lib.kvsnap_test_io_calls(2), 3)
        self.assertEqual(os.pread(self.file.fileno(), 3, 0), b"abc")

    def test_durable_empty_write_still_syncs_and_integer_overflow_is_rejected(self):
        self.assertEqual(self.lib.kvsnap_pwrite_all(self.file.fileno(), b"", 0, 0, 1), 0)
        self.assertEqual(self.lib.kvsnap_test_io_calls(0), 0)
        self.assertEqual(self.lib.kvsnap_test_io_calls(2), 1)
        self.assertEqual(self.lib.kvsnap_pwrite_all(self.file.fileno(), b"x", 1, -1, 0), errno.EINVAL)
        self.assertEqual(self.lib.kvsnap_pwrite_all(self.file.fileno(), b"xx", 2, 2**63 - 2, 0), errno.EINVAL)
        self.assertEqual(self.lib.kvsnap_test_io_calls(0), 0)

    def test_castagnoli_known_vectors_independent_of_python_implementation(self):
        for value, checksum in [(b"", 0), (b"123456789", 0xE3069283),
                                (bytes(32), 0x8A9136AA), (bytes([255]) * 32, 0x62A8AB43)]:
            with self.subTest(value=value):
                self.assertEqual(self.lib.kvsnap_crc32c(value, len(value)), checksum)

    def test_incremental_crc_matches_independent_bitwise_oracle_for_unaligned_lengths(self):
        def reference(data, crc=0):
            value = crc ^ 0xFFFFFFFF
            for byte in data:
                value ^= byte
                for _ in range(8):
                    value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
            return value ^ 0xFFFFFFFF
        data = bytes(range(256)) * 4
        for length in (0, 1, 7, 8, 9, 15, 31, 127, 1024):
            value = data[:length]
            for split in (0, length // 2, length):
                with self.subTest(length=length, split=split):
                    first = self.lib.kvsnap_crc32c_extend(value[:split], split, 0)
                    self.assertEqual(self.lib.kvsnap_crc32c_extend(value[split:], length - split, first), reference(value))

    def test_short_reads_eintr_and_eof_are_not_exposed_as_complete(self):
        payload = b"read-exact-native-bytes"
        os.pwrite(self.file.fileno(), payload, 0)
        result = C.create_string_buffer(len(payload))
        self.lib.kvsnap_test_io_fault(1, 2, 3, 0, 0)
        self.assertEqual(self.lib.kvsnap_pread_all(self.file.fileno(), result, len(payload), 0), 0)
        self.assertEqual(result.raw, payload)
        self.assertEqual(self.lib.kvsnap_test_io_calls(1), 2 + (len(payload) + 2) // 3)
        self.lib.kvsnap_test_io_reset()
        self.assertEqual(self.lib.kvsnap_pread_all(self.file.fileno(), result, len(payload), 1), errno.EIO)
        self.lib.kvsnap_test_io_fault(1, 0, 0, errno.EIO, 0)
        self.assertEqual(self.lib.kvsnap_pread_all(self.file.fileno(), result, len(payload), 0), errno.EIO)

    def test_bounded_backend_chunks_payload_and_surfaces_enospc(self):
        backend = NativeIO(self.library, queue_bytes=7)
        self.addCleanup(backend.close)
        payload = bytes(range(256))
        backend.pwrite_all(self.file.fileno(), payload, 3)
        self.assertEqual(backend.pread_all(self.file.fileno(), len(payload), 3), payload)
        self.assertEqual(self.lib.kvsnap_test_io_calls(0), (len(payload) + 6) // 7)
        self.lib.kvsnap_test_io_fault(0, 0, 0, errno.ENOSPC, 0)
        with self.assertRaises(OSError) as raised:
            backend.pwrite_all(self.file.fileno(), b"failed", 0)
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(os.pread(self.file.fileno(), len(payload), 3), payload)

    def test_close_waits_for_two_real_native_jobs_then_rejects_new_io(self):
        backend = NativeIO(self.library, workers=2, queue_bytes=128)
        self.addCleanup(backend.close)
        self.lib.kvsnap_test_io_hold(0, 1)
        closed = threading.Event()
        executor = ThreadPoolExecutor(max_workers=3)
        try:
            first = executor.submit(backend.pwrite_all, self.file.fileno(), b"a" * 64, 0)
            second = executor.submit(backend.pwrite_all, self.file.fileno(), b"b" * 64, 64)
            deadline = time.monotonic() + 5
            while self.lib.kvsnap_test_io_calls(0) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(self.lib.kvsnap_test_io_calls(0), 2, "native jobs should execute concurrently")
            def shutdown():
                backend.close()
                closed.set()
            closing = executor.submit(shutdown)
            self.assertFalse(closed.wait(0.05), "pool closed while syscall buffers were borrowed")
            self.lib.kvsnap_test_io_hold(0, 0)
            first.result(timeout=5)
            second.result(timeout=5)
            closing.result(timeout=5)
            self.assertEqual(os.pread(self.file.fileno(), 128, 0), b"a" * 64 + b"b" * 64)
            with self.assertRaises(RuntimeError):
                backend.pread_all(self.file.fileno(), 1, 0)
        finally:
            self.lib.kvsnap_test_io_hold(0, 0)
            executor.shutdown(wait=True)

    def test_python_and_native_stores_exchange_exact_blob_format_and_reject_native_corruption(self):
        identity = KVIdentity("native-io", "r", "t", "bytes", 2, "cpu-fixture",
                              {"layers": 1}, {"max_model_len": 16})
        def manifest(name):
            return SnapshotManifest(name, name, 0, identity, (1, 2, 3, 4), 3, (4,), 1,
                                    SamplingConfig(), 3, 0, durability="durable").to_dict()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"KVSNAP_IO_LIBRARY": str(self.library)}):
            for write_mode, read_mode in (("python", "native"), ("native", "python")):
                payloads = [write_mode.encode() + b"-first", write_mode.encode() + b"-second"]
                with SnapshotStore(root, io_backend=write_mode) as store:
                    self.assertEqual(store.io.name, write_mode)
                    body = store.commit(manifest(write_mode), payloads)
                # Parse independently rather than obtaining expected headers from the writer.
                import struct
                blob = Path(root) / "blobs" / (body["segments"][0]["blob_id"] + ".blob")
                raw = blob.read_bytes()
                magic, version, size, checksum, reserved = struct.unpack("<8sIQI8s", raw[:32])
                self.assertEqual((magic, version, size, reserved), (b"KVSBLOB1", 1, len(payloads[0]), bytes(8)))
                self.assertEqual(raw[32:], payloads[0])
                self.assertEqual(hashlib.sha256(raw[32:]).hexdigest(), blob.stem)
                self.assertEqual(checksum, self.lib.kvsnap_crc32c(payloads[0], len(payloads[0])))
                with SnapshotStore(root, io_backend=read_mode) as store:
                    self.assertEqual(store.io.name, read_mode)
                    self.assertEqual(store.load(write_mode), payloads)
            raw = bytearray(blob.read_bytes())
            raw[-1] ^= 1
            blob.write_bytes(raw)
            with SnapshotStore(root, io_backend="native") as recovered:
                self.assertEqual(set(recovered.recovery_report["invalid"]), {"native"})
                self.assertEqual(recovered.load("python"), [b"python-first", b"python-second"])
                with self.assertRaises(CorruptSnapshotError):
                    recovered.load("native")

    def test_explicit_unavailable_native_library_never_silently_falls_back(self):
        with patch.dict(os.environ, {"KVSNAP_IO_LIBRARY": str(Path(self.temporary.name) / "missing.so")}):
            for mode in ("native", "auto"):
                with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                    get_io_backend(mode)
            self.assertEqual(get_io_backend("python").name, "python")

    def test_native_enospc_through_store_cannot_publish_target_or_evict_survivor(self):
        identity = KVIdentity("native-io", "r", "t", "bytes", 2, "cpu-fixture",
                              {"layers": 1}, {"max_model_len": 16})
        def manifest(name):
            return SnapshotManifest(name, name, 0, identity, (1, 2, 3, 4), 3, (4,), 1,
                                    SamplingConfig(), 3, 0, durability="durable").to_dict()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"KVSNAP_IO_LIBRARY": str(self.library)}):
            with SnapshotStore(root, io_backend="native") as store:
                store.commit(manifest("survivor"), [b"survivor-full", b"survivor-tail"])
                self.lib.kvsnap_test_io_fault(0, 0, 0, errno.ENOSPC, 0)
                with self.assertRaises(OSError) as raised:
                    store.commit(manifest("target"), [b"target-full", b"target-tail"])
                self.assertEqual(raised.exception.errno, errno.ENOSPC)
                self.assertEqual(store.get_manifest("target", include_invalid=True)["state"], "Invalid")
                self.lib.kvsnap_test_io_reset()
                self.assertEqual(store.load("survivor"), [b"survivor-full", b"survivor-tail"])
            with SnapshotStore(root, io_backend="python") as recovered:
                self.assertEqual(recovered.recovery_report["valid"], ["survivor"])
                self.assertEqual(recovered.recovery_report["invalid"], ["target"])
                recovered.collect_garbage()
                self.assertEqual(recovered.load("survivor"), [b"survivor-full", b"survivor-tail"])


if __name__ == "__main__":
    unittest.main()
