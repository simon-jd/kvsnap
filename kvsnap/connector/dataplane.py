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


def _runs(block_ids):
    """Yield (start index, length) of maximal runs of consecutive block ids."""
    start = 0
    for i in range(1, len(block_ids) + 1):
        if i == len(block_ids) or block_ids[i] != block_ids[i - 1] + 1:
            yield start, i - start
            start = i


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
        # Block b of a layer starts at base + b * pitch; computing addresses
        # directly avoids building a tensor view for every copy.
        self._addresses = [(tensor.data_ptr(), tensor.stride(0) * tensor.element_size())
                           for tensor in self.caches.values()]
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

    def _producer_stream(self):
        return self.torch.cuda.current_stream(next(iter(self.caches.values())).device).cuda_stream

    def _copy_runs(self, allocation, block_ids, *, to_host, producer_stream=0):
        """Copy ``block_ids`` between the KV cache and a packed pinned buffer.

        The pinned buffer holds block i at ``i * block_bytes``, with its layers
        back to back. For each layer, every run of consecutive block ids is one
        cudaMemcpy2DAsync (host pitch = packed block, device pitch = the layer's
        block stride), so an N-block transfer costs layers x runs copies instead
        of layers x N.
        """
        runs = list(_runs(block_ids))
        layer_offset = 0
        for layer, (base, device_pitch) in zip(self.layers, self._addresses):
            for start, count in runs:
                allocation.copy_2d(base + block_ids[start] * device_pitch, device_pitch,
                                   start * self.block_bytes + layer_offset, self.block_bytes,
                                   layer["bytes"], count, to_host=to_host, producer_stream=producer_stream)
            layer_offset += layer["bytes"]
        return len(runs) * len(self.layers)

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

    def iter_save_pinned(self, block_ids, num_kv_tokens, *, batch=64):
        """Yield one owned pinned allocation per block (volatile snapshots keep them).

        Copies are issued ``batch`` blocks at a time and fenced together, so the
        stream never idles on a per-block wait, while at most ``batch`` blocks
        of pinned headroom are reserved before the consumer adopts them.
        """
        self._validate_ids(block_ids)
        expected = (num_kv_tokens + self.block_size - 1) // self.block_size
        if len(block_ids) != expected:
            raise ValueError("block count does not cover computed tokens")
        stream = self._producer_stream()
        valid = num_kv_tokens % self.block_size
        for first in range(0, len(block_ids), batch):
            group = block_ids[first:first + batch]
            allocations = []
            try:
                for block_id in group:
                    allocations.append(self.pool.reserve(self.block_bytes))
                    self._copy_runs(allocations[-1], [block_id], to_host=True, producer_stream=stream)
                for allocation in allocations:
                    allocation.wait()
                if valid and first + len(group) == len(block_ids):
                    self._zero_tail(allocations[-1], 0, valid)
            except BaseException:
                for allocation in allocations:
                    allocation.close()
                raise
            for index, allocation in enumerate(allocations):
                try:
                    yield allocation
                except BaseException:
                    for rest in allocations[index + 1:]:
                        rest.close()
                    raise

    def save_pinned(self, block_ids, num_kv_tokens):
        """Return owned, immutable pinned blocks after the final D2H event."""
        allocations = []
        try:
            allocations.extend(self.iter_save_pinned(block_ids, num_kv_tokens))
            return allocations
        except BaseException:
            for allocation in allocations:
                allocation.close()
            raise

    def save_staged(self, block_ids, num_kv_tokens):
        """D2H all blocks into one packed pinned buffer with batched copies and one wait.

        Returns the fenced allocation; the caller reads it and must close it.
        """
        self._validate_ids(block_ids)
        expected = (num_kv_tokens + self.block_size - 1) // self.block_size
        if len(block_ids) != expected or not block_ids:
            raise ValueError("block count does not cover computed tokens")
        allocation = self.pool.reserve(self.block_bytes * len(block_ids))
        try:
            self._copy_runs(allocation, block_ids, to_host=True, producer_stream=self._producer_stream())
            allocation.wait()
            valid = num_kv_tokens % self.block_size
            if valid:
                self._zero_tail(allocation, (len(block_ids) - 1) * self.block_bytes, valid)
            return allocation
        except BaseException:
            allocation.close()
            raise

    def save(self, block_ids, num_kv_tokens):
        """D2H all blocks; return one bytes object per block."""
        if not block_ids and not num_kv_tokens:
            return []
        allocation = self.save_staged(block_ids, num_kv_tokens)
        try:
            return [allocation.read_range(i * self.block_bytes, self.block_bytes) for i in range(len(block_ids))]
        finally:
            allocation.close()

    def load_pinned(self, block_ids, allocations):
        """Load an owner's borrowed per-block allocations without another pinned copy."""
        self._validate_ids(block_ids)
        if len(block_ids) != len(allocations) or any(a.size != self.block_bytes for a in allocations):
            raise ValueError("KV pinned payload length mismatch")
        # The pool's copy stream is serial: enqueue everything, then fence each.
        for block_id, allocation in zip(block_ids, allocations):
            self._copy_runs(allocation, [block_id], to_host=False)
        for allocation in allocations:
            allocation.wait()

    @contextmanager
    def staging(self, nblocks):
        """A packed pinned buffer for ``nblocks`` blocks, released (after its copies fence) on exit."""
        allocation = self.pool.reserve(self.block_bytes * nblocks)
        try:
            yield allocation
        finally:
            allocation.close()

    def load_staged(self, block_ids, allocation):
        """H2D a filled staging buffer into ``block_ids`` and wait for the copies."""
        self._validate_ids(block_ids)
        if allocation.size < self.block_bytes * len(block_ids):
            raise ValueError("staging buffer smaller than the blocks to load")
        copies = self._copy_runs(allocation, block_ids, to_host=False)
        # No dependency on layer hooks skipped by full CUDA graph replay.
        allocation.wait()
        return copies

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

    def load_scatter(self, block_ids, sources, *, chunk_bytes=64 << 20):
        """H2D packed blocks contiguously into device staging, then scatter them on the GPU.

        ``sources[i]`` is ``(allocation, offset)`` holding block i packed (its
        layers back to back). The PCIe transfer is a few large copies, one per
        chunk for a staging buffer or one per block for per-block host
        allocations, instead of one 64 KiB copy per (layer, block run). Each
        chunk is then written into the KV cache with one ``index_copy_`` per
        layer on the same stream, which costs device bandwidth only. Returns
        the number of H2D copies issued.
        """
        torch = self.torch
        self._validate_ids(block_ids)
        if len(sources) != len(block_ids):
            raise ValueError("one source per block is required")
        if not block_ids:
            return 0
        raw, ranges = self._scatter_targets()
        size = self.block_bytes
        per_chunk = max(1, min(len(block_ids), chunk_bytes // size))
        device = next(iter(self.caches.values())).device
        copies = 0
        with torch.cuda.stream(self._copy_stream):
            ids = torch.tensor(block_ids, dtype=torch.int64).pin_memory().to(device, non_blocking=True)
            stage = torch.empty((per_chunk, size // 8), dtype=torch.int64, device=device)
            for first in range(0, len(block_ids), per_chunk):
                count = min(per_chunk, len(block_ids) - first)
                i = first
                while i < first + count:
                    allocation, offset = sources[i]
                    j = i + 1
                    while (j < first + count and sources[j][0] is allocation
                           and sources[j][1] == offset + (j - i) * size):
                        j += 1
                    allocation.copy(stage.data_ptr() + (i - first) * size, (j - i) * size, offset, to_host=False)
                    copies += 1
                    i = j
                # Stream order: the scatter reads this chunk after its copies,
                # and the next chunk's copies overwrite stage only afterwards.
                for target, (start, stop) in zip(raw, ranges):
                    target.index_copy_(0, ids[first:first + count], stage[:count, start:stop])
            done = torch.cuda.Event()
            done.record()
        done.synchronize()
        for allocation in {id(a): a for a, _ in sources}.values():
            allocation.wait()
        return copies

    def save_gather(self, block_ids, targets, num_kv_tokens, *, chunk_bytes=64 << 20):
        """Gather blocks on the GPU into device staging, then D2H them in large copies.

        The reverse of :meth:`load_scatter`: ``targets[i]`` is ``(allocation,
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

    def load(self, block_ids, blocks):
        if len(block_ids) != len(blocks) or any(len(b) != self.block_bytes for b in blocks):
            raise ValueError("KV payload length mismatch")
        if not blocks:
            return
        with self.staging(len(blocks)) as allocation:
            for index, block in enumerate(blocks):
                C.memmove(allocation.pointer + index * self.block_bytes, block, self.block_bytes)
            self.load_staged(block_ids, allocation)

    def close(self):
        self.pool.close()
