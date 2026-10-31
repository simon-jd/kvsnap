"""Native C++ blob I/O with an explicit, portable Python backend.

auto uses a present compatible library; an explicitly selected native backend
fails if unavailable. Native corruption/ABI errors never silently fall back.
"""
from __future__ import annotations
import atexit
import os
from pathlib import Path
import threading


class PythonIO:
    name = "python"

    def pwrite_all(self, fd, data, offset):
        view = memoryview(data)
        while view:
            size = os.pwrite(fd, view, offset)
            if size <= 0:
                raise OSError("pwrite made no progress")
            offset += size
            view = view[size:]

    def pread_all(self, fd, size, offset):
        result = bytearray()
        while len(result) < size:
            part = os.pread(fd, size - len(result), offset + len(result))
            if not part:
                raise OSError("pread reached unexpected EOF")
            result.extend(part)
        return bytes(result)

    def pread_into(self, fd, address, size, offset):
        """Read exactly ``size`` bytes at ``offset`` into raw memory at ``address``."""
        import ctypes
        view = memoryview((ctypes.c_char * size).from_address(address)).cast("B")
        done = 0
        while done < size:
            count = os.preadv(fd, [view[done:]], offset + done)
            if count <= 0:
                raise OSError("pread reached unexpected EOF")
            done += count

    def crc32c_at(self, address, size, crc=0):
        import ctypes
        from .crc32c import crc32c
        return crc32c(ctypes.string_at(address, size), crc)


class NativeIO:
    name = "native"

    def __init__(self, path, *, workers=2, queue_bytes=64 * 1024**2):
        # Importing the real Python backend must not initialize a native FFI.
        global C
        import ctypes as C
        self.lib = C.CDLL(str(path))
        api = {
            "kvsnap_io_abi_version": ([], C.c_int),
            "kvsnap_crc32c_extend": ([C.c_void_p, C.c_size_t, C.c_uint32], C.c_uint32),
            "kvsnap_io_pool_create": ([C.c_size_t, C.c_size_t], C.c_void_p),
            "kvsnap_io_pool_destroy": ([C.c_void_p], None),
            "kvsnap_io_pool_transfer": ([C.c_void_p, C.c_int, C.c_void_p, C.c_size_t, C.c_int64, C.c_int, C.c_int], C.c_int),
        }
        for name, (args, result) in api.items():
            method = getattr(self.lib, name)
            method.argtypes, method.restype = args, result
        if self.lib.kvsnap_io_abi_version() != 1:
            raise RuntimeError("unsupported kvsnap native I/O ABI")
        if type(workers) is not int or not 1 <= workers <= 64 or type(queue_bytes) is not int or queue_bytes <= 0:
            raise ValueError("invalid native I/O capacity")
        self.handle = self.lib.kvsnap_io_pool_create(workers, queue_bytes)
        if not self.handle:
            raise RuntimeError("native I/O pool creation failed")
        self.queue_bytes = queue_bytes
        self.pid = os.getpid()
        self._lifecycle = threading.Condition()
        self._active = 0
        self._closing = False
        atexit.register(self.close)

    def _open(self):
        if self.pid != os.getpid():
            raise RuntimeError("native I/O pools must be reopened after fork")
        if not self.handle or self._closing:
            raise RuntimeError("native I/O pool is closed")

    @staticmethod
    def _offset(size, offset):
        if type(size) is not int or type(offset) is not int or min(size, offset) < 0 or size + offset > 2**63 - 1:
            raise ValueError("invalid I/O range")

    def _transfer(self, fd, data, size, offset, write):
        self._offset(size, offset)
        if self.pid != os.getpid():
            raise RuntimeError("native I/O pools must be reopened after fork")
        with self._lifecycle:
            self._open()
            self._active += 1
        try:
            code = self.lib.kvsnap_io_pool_transfer(self.handle, fd, data, size, offset, write, 0)
        finally:
            with self._lifecycle:
                self._active -= 1
                self._lifecycle.notify_all()
        if code:
            raise OSError(code, os.strerror(code))

    def pwrite_all(self, fd, data, offset):
        data = bytes(data)
        self._offset(len(data), offset)
        # Submit bounded chunks even if a nonstandard model uses large blocks.
        for index in range(0, len(data), self.queue_bytes):
            part = data[index:index + self.queue_bytes]
            self._transfer(fd, C.c_char_p(part), len(part), offset + index, 1)

    def pread_all(self, fd, size, offset):
        self._offset(size, offset)
        result = C.create_string_buffer(size)
        for index in range(0, size, self.queue_bytes):
            count = min(self.queue_bytes, size - index)
            self._transfer(fd, C.byref(result, index), count, offset + index, 0)
        return result.raw

    def crc32c(self, data, crc=0):
        self._open()
        data = bytes(data)
        return int(self.lib.kvsnap_crc32c_extend(C.c_char_p(data), len(data), crc))

    def pread_into(self, fd, address, size, offset):
        """Read straight into caller memory (e.g. pinned host memory), no Python copy."""
        self._offset(size, offset)
        for index in range(0, size, self.queue_bytes):
            count = min(self.queue_bytes, size - index)
            self._transfer(fd, C.c_void_p(address + index), count, offset + index, 0)

    def crc32c_at(self, address, size, crc=0):
        self._open()
        return int(self.lib.kvsnap_crc32c_extend(C.c_void_p(address), size, crc))

    def close(self):
        if self.pid != os.getpid():
            return
        with self._lifecycle:
            if not self.handle:
                return
            self._closing = True
            self._lifecycle.wait_for(lambda: self._active == 0)
            self.lib.kvsnap_io_pool_destroy(self.handle)
            self.handle = None


_backends = {}
_cache_lock = threading.Lock()


def _after_fork():
    global _backends, _cache_lock
    # Inherited pools have no live worker threads in the child. Do not acquire
    # their locks or destroy them; reopen a new process-owned backend lazily.
    _backends = {}
    _cache_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def get_io_backend(mode=None):
    mode = mode or os.environ.get("KVSNAP_IO_BACKEND", "auto")
    if mode not in ("auto", "native", "python"):
        raise ValueError("KVSNAP_IO_BACKEND must be auto, native or python")
    if mode == "python":
        return PythonIO()
    override = os.environ.get("KVSNAP_IO_LIBRARY")
    candidates = [Path(override)] if override else [
        *sorted((Path(__file__).resolve().parent.parent / "_native").glob("_io_native*.so")),
        Path(__file__).resolve().parent.parent / "_native/libkvsnap_io.so",
        Path(__file__).resolve().parents[2] / "build/dataplane/libkvsnap_io.so"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        if mode == "native" or override:
            raise RuntimeError("native I/O library unavailable; install with pip install . or build csrc/dataplane with CMake")
        return PythonIO()
    key = (os.getpid(), str(path.resolve()))
    with _cache_lock:
        if key not in _backends:
            _backends[key] = NativeIO(path)
        return _backends[key]
