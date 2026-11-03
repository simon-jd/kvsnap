"""ctypes binding for the bounded C++ CUDA pool; no CPU emulation fallback."""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
from importlib.machinery import EXTENSION_SUFFIXES
import os
from pathlib import Path
import threading


def resolve_cuda_library(library=None):
    """Resolve the C ABI library without importing a module or initializing CUDA."""
    override = library or os.environ.get("KVSNAP_CUDA_LIBRARY")
    if override:
        return Path(override)
    directory = Path(__file__).resolve().parent.parent / "_native"
    for suffix in EXTENSION_SUFFIXES:
        candidate = directory / ("_cuda_native" + suffix)
        if candidate.is_file():
            return candidate
    # This library exposes a Python-independent C ABI. A build for another
    # interpreter can serve as a fallback, but the current build takes priority.
    packaged = sorted(directory.glob("_cuda_native*.so"))
    return packaged[0] if packaged else Path(__file__).resolve().parents[2] / "build/dataplane/libkvsnap_cuda.so"


def copy_host_blocks(*args):
    """The restore kernel's launch, imported on first use: Triton loads only with CUDA tensors."""
    from kvsnap.connector.copy_kernel import copy_host_blocks as launch
    launch(*args)


class PinnedCapacityError(MemoryError):
    """The pinned pool cannot fit a reservation. Nothing was copied."""


class CudaPool:
    def __init__(self, capacity_bytes: int, device: int = 0, library=None):
        self._pid = os.getpid()
        self._lock = threading.RLock()
        # The native pool retains raw memory; retain Python tensor owners too
        # if an error prevents the caller from proving completion/releasing it.
        self._allocations = {}
        path = resolve_cuda_library(library)
        self.lib = C.CDLL(str(path))
        api = {
            "kvsnap_pool_abi_version": ([], C.c_int),
            "kvsnap_cuda_error": ([], C.c_char_p),
            "kvsnap_pool_create": ([C.c_size_t, C.c_int], C.c_void_p),
            "kvsnap_pool_destroy": ([C.c_void_p], None),
            "kvsnap_pool_close_checked": ([C.c_void_p], C.c_int),
            "kvsnap_pool_reserve": ([C.c_void_p, C.c_size_t, C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)], C.c_int),
            "kvsnap_pool_copy": ([C.c_void_p, C.c_uint64, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int, C.c_size_t], C.c_int),
            "kvsnap_pool_copy_2d": ([C.c_void_p, C.c_uint64, C.c_size_t, C.c_size_t, C.c_void_p, C.c_size_t,
                                  C.c_size_t, C.c_size_t, C.c_int, C.c_size_t], C.c_int),
            "kvsnap_pool_wait": ([C.c_void_p, C.c_uint64], C.c_int),
            "kvsnap_pool_ready": ([C.c_void_p, C.c_uint64], C.c_int),
            "kvsnap_pool_release": ([C.c_void_p, C.c_uint64], C.c_int),
        }
        for name, (args, result) in api.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.abi_version = self.lib.kvsnap_pool_abi_version()
        if self.abi_version != 2:
            raise RuntimeError("unsupported kvsnap CUDA pool ABI; rebuild both CUDA and torch extensions")
        self.handle = self.lib.kvsnap_pool_create(capacity_bytes, device)
        if not self.handle:
            raise RuntimeError(self.lib.kvsnap_cuda_error().decode())

    def check(self, result):
        if result < 0:
            raise RuntimeError(self.lib.kvsnap_cuda_error().decode())
        return result

    def reserve(self, size):
        with self.operation():
            ident, ptr = C.c_uint64(), C.c_void_p()
            if self.lib.kvsnap_pool_reserve(self.handle, size, C.byref(ident), C.byref(ptr)) < 0:
                message = self.lib.kvsnap_cuda_error().decode()
                if "pinned capacity exhausted" in message:
                    raise PinnedCapacityError(message)
                raise RuntimeError(message)
            allocation = PinnedAllocation(self, ident.value, ptr.value, size)
            self._allocations[ident.value] = allocation
            return allocation

    @contextmanager
    def operation(self):
        if self._pid != os.getpid():
            raise RuntimeError("CUDA pools must not be inherited across fork")
        with self._lock:
            if not self.handle:
                raise RuntimeError("pool closed")
            yield

    def close(self):
        if self._pid != os.getpid():
            raise RuntimeError("CUDA pools must not be inherited across fork")
        with self._lock:
            if self.handle:
                self.check(self.lib.kvsnap_pool_close_checked(self.handle))
                self.handle = None
                for allocation in self._allocations.values():
                    allocation.closed = True
                    if hasattr(allocation, "_tensor_references"):
                        allocation._tensor_references.clear()
                self._allocations.clear()

    @contextmanager
    def borrow_copy_stream(self):
        """Borrow the native copy stream for diagnostic event measurements.

        Keep this context open through event completion; it excludes concurrent
        pool close. It gives neither stream ownership nor permission to release
        pinned allocations before their normal completion checks. Older ABI 2
        libraries still support copies, but need rebuilding for this optional
        observation API.
        """
        with self.operation():
            try:
                getter = self.lib.kvsnap_pool_copy_stream
            except AttributeError as exc:
                raise RuntimeError("native copy-stream timing API unavailable; rebuild CUDA and torch extensions") from exc
            getter.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
            getter.restype = C.c_int
            handle = C.c_size_t()
            self.check(getter(self.handle, C.byref(handle)))
            if not handle.value:
                raise RuntimeError("native copy stream handle is invalid")
            yield handle.value


class PinnedAllocation:
    def __init__(self, pool, ident, pointer, size):
        self.pool, self.ident, self.pointer, self.size = pool, ident, pointer, size
        self.closed = False

    def _open(self):
        if self.closed or not self.pool.handle:
            raise RuntimeError("pinned allocation closed")

    @contextmanager
    def operation(self):
        with self.pool.operation():
            self._open()
            yield

    def copy(self, device_ptr, size, offset=0, *, to_host, producer_stream=0):
        with self.operation():
            if min(size, offset, device_ptr, producer_stream) < 0:
                raise ValueError("negative copy argument")
            self.pool.check(self.pool.lib.kvsnap_pool_copy(
                self.pool.handle, self.ident, offset, device_ptr, size, to_host, producer_stream))

    def copy_2d(self, device_ptr, device_pitch, host_offset, host_pitch, width, height, *,
                to_host, producer_stream=0):
        """``height`` rows of ``width`` bytes between strided device and host memory."""
        with self.operation():
            if min(device_ptr, device_pitch, host_offset, host_pitch, width, height, producer_stream) < 0:
                raise ValueError("negative copy argument")
            self.pool.check(self.pool.lib.kvsnap_pool_copy_2d(
                self.pool.handle, self.ident, host_offset, host_pitch, device_ptr, device_pitch,
                width, height, to_host, producer_stream))

    def wait(self):
        with self.operation():
            self.pool.check(self.pool.lib.kvsnap_pool_wait(self.pool.handle, self.ident))
            if hasattr(self, "_tensor_references"):
                self._tensor_references.clear()

    def ready(self):
        with self.operation():
            return bool(self.pool.check(self.pool.lib.kvsnap_pool_ready(self.pool.handle, self.ident)))

    def read(self):
        with self.operation():
            self.wait()
            return C.string_at(self.pointer, self.size)

    def read_range(self, offset, size):
        with self.operation():
            self.wait()
            if type(offset) is not int or type(size) is not int or offset < 0 or size < 0 or offset + size > self.size:
                raise ValueError("pinned read outside allocation")
            return C.string_at(self.pointer + offset, size)

    def zero(self, offset, size):
        with self.operation():
            self.wait()
            if offset < 0 or size < 0 or offset + size > self.size:
                raise ValueError("pinned zero outside allocation")
            C.memset(self.pointer + offset, 0, size)

    def write(self, data):
        with self.operation():
            self.wait()
            if len(data) != self.size:
                raise ValueError("pinned allocation length mismatch")
            C.memmove(self.pointer, data, self.size)

    def close(self):
        if self.pool._pid != os.getpid():
            raise RuntimeError("CUDA pools must not be inherited across fork")
        with self.pool._lock:
            if not self.closed and self.pool.handle:
                self.pool.check(self.pool.lib.kvsnap_pool_release(self.pool.handle, self.ident))
            self.closed = True
            self.pool._allocations.pop(self.ident, None)
            if hasattr(self, "_tensor_references"):
                self._tensor_references.clear()


class TorchBlockDataPlane:
    """Packed physical blocks: layer-name order, dense [H,N,C] stride permutations.

    Actual layout is recorded and compared. Unsupported raw cache views fail
    closed, never guessed from a model name. Allocation stays live until the
    CUDA event completes on all paths, including exceptions.
    """
    def __init__(self, caches, block_size, capacity_bytes):
        import torch
        if not caches:
            raise ValueError("empty KV cache registration")
        self.torch = torch
        self.caches = dict(sorted(caches.items()))
        self.block_size = block_size
        self.layers = []
        device = next(iter(self.caches.values())).device
        for name, tensor in self.caches.items():
            if tensor.device != device or tensor.device.type != "cuda":
                raise ValueError("one CUDA device required")
            if tensor.ndim != 4 or tensor.shape[2] != block_size or tensor.stride(3) != 1:
                raise ValueError(f"unsupported KV layout for {name}: need dense [B,H,N,C] blocks with contiguous C")
            # vLLM's default LBNHC backing storage exposes a noncontiguous
            # BHNC view. H/N permutations are safe when each block is dense;
            # copy its physical span and retain exact strides in the identity.
            dense = 1
            for stride, size in sorted((tensor.stride(i), tensor.shape[i]) for i in (1, 2, 3) if tensor.shape[i] > 1):
                if stride != dense:
                    raise ValueError(f"unsupported KV layout for {name}: block has holes or overlapping axes")
                dense *= size
            if tensor.stride(0) < dense:
                raise ValueError(f"unsupported KV layout for {name}: overlapping blocks")
            self.layers.append({"name": name, "shape": list(tensor.shape[1:]),
                                "dtype": str(tensor.dtype), "stride": list(tensor.stride()[1:]),
                                "bytes": tensor[0].numel() * tensor.element_size()})
        self.block_bytes = sum(l["bytes"] for l in self.layers)
        self.pool = CudaPool(capacity_bytes, device.index)

    def _validate_ids(self, ids):
        for tensor in self.caches.values():
            if any(type(i) is not int or i < 0 or i >= tensor.shape[0] for i in ids):
                raise ValueError("GPU block id out of range")

    def producer_event(self):
        """A CUDA event recorded now on the engine's current stream."""
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(next(iter(self.caches.values())).device))
        return event

    def _zero_tail(self, allocation, base, valid):
        """Zero token slots past ``valid`` in one packed block, so snapshots never carry stale KV."""
        offset = base
        for layer in self.layers:
            h, n, c = layer["shape"]
            sh, sn, _ = layer["stride"]
            element_bytes = layer["bytes"] // h // n // c
            for head in range(h):
                for token in range(valid, n):
                    allocation.zero(offset + (head * sh + token * sn) * element_bytes, c * element_bytes)
            offset += layer["bytes"]

    @contextmanager
    def staging(self, nblocks):
        """A packed pinned buffer for ``nblocks`` blocks, released (after its copies fence) on exit."""
        allocation = self.pool.reserve(self.block_bytes * nblocks)
        try:
            yield allocation
        finally:
            allocation.close()

    def _scatter_targets(self):
        """Each layer's KV cache as int64 rows: row b is block b's dense span.

        Built on first use: the H2D scatter path needs real torch tensors, and
        the layout tests register descriptors without storage.
        """
        if getattr(self, "_raw_layers", None) is None:
            torch, raw, ranges, offset = self.torch, [], [], 0
            for layer, tensor in zip(self.layers, self.caches.values()):
                size = tensor.element_size()
                pitch, start = tensor.stride(0) * size, tensor.storage_offset() * size
                if pitch % 8 or start % 8 or layer["bytes"] % 8:
                    raise ValueError(f"KV layer {layer['name']} is not 8-byte aligned for the scatter path")
                words = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                    tensor.untyped_storage()).view(torch.int64)
                raw.append(words.as_strided((tensor.shape[0], layer["bytes"] // 8), (pitch // 8, 1), start // 8))
                ranges.append((offset // 8, (offset + layer["bytes"]) // 8))
                offset += layer["bytes"]
            device = next(iter(self.caches.values())).device
            with self.pool.borrow_copy_stream() as handle:
                self._copy_stream = torch.cuda.ExternalStream(handle, device=device)
            self._raw_layers, self._layer_words = raw, ranges
        return self._raw_layers, self._layer_words

    def _load_tables(self):
        """Per layer, as int64 device tensors: its offset in a packed block, its KV cache's
        base address and block pitch in bytes, and its span in 8-byte words."""
        if getattr(self, "_tables", None) is None:
            raw, ranges = self._scatter_targets()
            device = next(iter(self.caches.values())).device
            columns = ([start * 8 for start, _ in ranges], [rows.data_ptr() for rows in raw],
                       [rows.stride(0) * 8 for rows in raw], [stop - start for start, stop in ranges])
            self._tables = tuple(self.torch.tensor(column, dtype=self.torch.int64, device=device)
                                 for column in columns)
        return self._tables

    def warm_up(self):
        """Build the load tables and compile the restore kernel with an empty launch, at registration
        rather than on the first restore."""
        torch = self.torch
        tables = self._load_tables()
        device = next(iter(self.caches.values())).device
        with torch.cuda.stream(self._copy_stream):
            empty = torch.zeros(1, dtype=torch.int64, device=device)
            copy_host_blocks(empty[:0], empty[:0], *tables)
        self._copy_stream.synchronize()

    def wait_on_engine_stream(self, event):
        """Make the engine's current stream wait, on the GPU, for ``event``."""
        self.torch.cuda.current_stream(next(iter(self.caches.values())).device).wait_event(event)

    def load_scatter(self, block_ids, sources, *, progress=None, chunk_bytes=64 << 20):
        """Copy packed host blocks into ``block_ids`` with the restore kernel, and wait for it.

        ``sources[i]`` is ``(allocation, offset)`` holding block i packed (its
        layers back to back), in this pool's pinned memory. The kernel reads
        each (layer, block) span from host memory and writes it into its
        place in the KV cache (:mod:`kvsnap.connector.copy_kernel`): the
        bytes cross PCIe once, with no device staging, on the pool's copy
        stream. It is launched once per ``chunk_bytes`` of blocks; a
        ``progress`` list receives ``(None, total bytes)`` and then, as each
        launch is queued, ``(its completion event, bytes left once it
        completes)``, so another thread can tell how far the copy has got.
        Returns the number of launches.
        """
        torch = self.torch
        self._validate_ids(block_ids)
        if len(sources) != len(block_ids):
            raise ValueError("one source per block is required")
        if not block_ids:
            return 0
        tables = self._load_tables()
        device = next(iter(self.caches.values())).device
        n, per_launch = len(block_ids), max(1, chunk_bytes // self.block_bytes)
        launches = 0
        with self.pool.operation():
            # The pool lock keeps every source allocation open until the launch is queued.
            addresses = []
            for allocation, offset in sources:
                allocation._open()
                if offset < 0 or offset + self.block_bytes > allocation.size:
                    raise ValueError("load source outside its allocation")
                addresses.append(allocation.pointer + offset)
            with torch.cuda.stream(self._copy_stream):
                table = torch.tensor(addresses + list(block_ids), dtype=torch.int64).pin_memory().to(
                    device, non_blocking=True)
                if progress is not None:
                    progress.append((None, n * self.block_bytes))
                try:
                    for first in range(0, n, per_launch):
                        last = min(n, first + per_launch)
                        copy_host_blocks(table[first:last], table[n + first:n + last], *tables)
                        launches += 1
                        done = torch.cuda.Event()
                        done.record()
                        if progress is not None:
                            progress.append((done, (n - last) * self.block_bytes))
                except BaseException:
                    # A launch that failed part way may still be queued: nothing
                    # may release the sources before the copy stream drains.
                    self._copy_stream.synchronize()
                    raise
        done.synchronize()
        return launches

    def save_gather(self, block_ids, targets, num_kv_tokens, *, chunk_bytes=64 << 20):
        """Gather blocks on the GPU into device staging, then D2H them in large copies.

        The layout :meth:`load_scatter` reads: ``targets[i]`` is ``(allocation,
        offset)`` that receives block i packed. The caller must have fenced the
        producer stream (the copies run on the pool's copy stream). Token slots
        past ``num_kv_tokens`` in the last block are zeroed on the host, so a
        snapshot never carries stale KV.
        """
        torch = self.torch
        self._validate_ids(block_ids)
        expected = (num_kv_tokens + self.block_size - 1) // self.block_size
        if len(block_ids) != expected or len(targets) != len(block_ids):
            raise ValueError("block count does not cover computed tokens")
        if not block_ids:
            return 0
        raw, _ = self._scatter_targets()
        size = self.block_bytes
        per_chunk = max(1, min(len(block_ids), chunk_bytes // size))
        device = next(iter(self.caches.values())).device
        copies = 0
        with torch.cuda.stream(self._copy_stream):
            ids = torch.tensor(block_ids, dtype=torch.int64).pin_memory().to(device, non_blocking=True)
            stage = torch.empty((per_chunk, size // 8), dtype=torch.int64, device=device)
            for first in range(0, len(block_ids), per_chunk):
                count = min(per_chunk, len(block_ids) - first)
                # One cat per chunk, not a copy per layer: this runs on the save
                # thread, and each op it issues holds the GIL the engine needs.
                chunk = ids[first:first + count]
                torch.cat([source.index_select(0, chunk) for source in raw], dim=1, out=stage[:count])
                i = first
                while i < first + count:
                    allocation, offset = targets[i]
                    j = i + 1
                    while (j < first + count and targets[j][0] is allocation
                           and targets[j][1] == offset + (j - i) * size):
                        j += 1
                    allocation.copy(stage.data_ptr() + (i - first) * size, (j - i) * size, offset, to_host=True)
                    copies += 1
                    i = j
            done = torch.cuda.Event()
            done.record()
        done.synchronize()
        for allocation in {id(a): a for a, _ in targets}.values():
            allocation.wait()
        valid = num_kv_tokens % self.block_size
        if valid:
            allocation, offset = targets[-1]
            self._zero_tail(allocation, offset, valid)
        return copies

    def close(self):
        self.pool.close()
