"""Worker-owned immutable pinned buffers, published only by SnapshotStore.

The production owner accepts C++ PinnedAllocation objects. CPU tests may inject
an explicitly labelled fake allocator; this module never allocates a bytes pool.
Only descriptors, checksums and temporary preparation intents are written to disk.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import struct
import threading
import time
import uuid

from kvsnap.store.crc32c import crc32c
from kvsnap.store.native_io import get_io_backend
from kvsnap.store.store import CapacityError, CorruptSnapshotError, StoreError, _write_preparation

MAX_MESSAGE = 64 * 1024
MAX_CHUNK = 8 * 1024 * 1024


@contextmanager
def _socket_address(path):
    """Keep Linux sun_path short without moving sockets out of the private store.

    bind/connect resolve the fd-relative alias while the directory fd is open.
    The original path remains authoritative for descriptors, permissions and
    unlink; no process-wide cwd change, global socket directory or created symlink.
    """
    path=Path(path)
    if len(os.fsencode(path))<=107:
        yield str(path)
        return
    fd=os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info=os.fstat(fd)
        if info.st_uid!=os.getuid() or info.st_mode & 0o077:
            raise StoreError('volatile socket directory must be private and owned by current uid')
        address=f'/proc/self/fd/{fd}/{path.name}'
        if len(os.fsencode(address))>107:
            raise StoreError('volatile socket basename is too long')
        yield address
    finally:
        os.close(fd)


def _hex(value, size):
    return isinstance(value, str) and len(value) == size and all(c in '0123456789abcdef' for c in value)


def _receive(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = sock.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError('volatile owner disconnected')
        chunks.extend(chunk)
    return bytes(chunks)


def _send_json(sock, value):
    data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError('volatile protocol message too large')
    sock.sendall(struct.pack('!I', len(data)) + data)


def _receive_json(sock):
    count, = struct.unpack('!I', _receive(sock, 4))
    if count > MAX_MESSAGE:
        raise ValueError('volatile protocol message too large')
    return json.loads(_receive(sock, count))


class VolatileClient:
    def __init__(self, root, owner, *, timeout=5.0):
        self.root, self.owner, self.timeout = Path(root), dict(owner), timeout
        ident = self.owner.get('owner_id')
        if not _hex(ident, 32) or not _hex(self.owner.get('token'), 64):
            raise CorruptSnapshotError('invalid volatile owner identity')
        if self.owner.get('socket') != ident[:12] + '.sock':
            raise CorruptSnapshotError('invalid volatile owner socket')
        self.path = self.root / '.volatile' / self.owner['socket']

    def request(self, op, **args):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
                stream.settimeout(self.timeout)
                with _socket_address(self.path) as address:
                    stream.connect(address)
                _, uid, _ = struct.unpack('3i', stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != os.getuid():
                    raise CorruptSnapshotError('volatile owner peer uid mismatch')
                _send_json(stream, dict(op=op, token=self.owner['token'], **args))
                result = _receive_json(stream)
                if not result.get('ok'):
                    raise CorruptSnapshotError('volatile owner: ' + result.get('error', 'invalid response'))
                count = result.get('payload_bytes', 0)
                if type(count) is not int or not 0 <= count <= MAX_CHUNK:
                    raise CorruptSnapshotError('invalid volatile payload framing')
                return result, _receive(stream, count)
        except (OSError, ValueError, ConnectionError) as exc:
            raise CorruptSnapshotError(f'volatile owner unavailable: {exc}') from exc

    def check(self, locations):
        """Confirm the owner still holds every ``{blob_id: location}`` at its length, without moving bytes."""
        items = [dict(blob_id=key, handle=location.get('handle'), length=location.get('length'))
                 for key, location in locations.items()]
        for first in range(0, len(items), 256):
            self.request('check', blocks=items[first:first + 256])

    def read(self, blob_id, location):
        length = location.get('length')
        if type(length) is not int or length <= 0 or not _hex(location.get('handle'), 32):
            raise CorruptSnapshotError('invalid volatile allocation descriptor')
        result = bytearray()
        for offset in range(0, length, MAX_CHUNK):
            count = min(MAX_CHUNK, length - offset)
            _, chunk = self.request('read', blob_id=blob_id, handle=location['handle'], offset=offset, length=count)
            if len(chunk) != count:
                raise CorruptSnapshotError('volatile allocation length mismatch')
            result.extend(chunk)
        data = bytes(result)
        if hashlib.sha256(data).hexdigest() != blob_id:
            raise CorruptSnapshotError('volatile blob checksum mismatch')
        return data


def _digest(allocation):
    """(SHA-256 content id, CRC32C) of one block, hashed in place when it is pinned memory."""
    pointer = getattr(allocation, 'pointer', None)
    if pointer is None:
        data = allocation.read()
        if len(data) != allocation.size:
            raise ValueError('allocation length differs from size')
        return hashlib.sha256(data).hexdigest(), crc32c(data)
    allocation.wait()
    view = memoryview((ctypes.c_char * allocation.size).from_address(pointer)).cast('B')
    return hashlib.sha256(view).hexdigest(), get_io_backend().crc32c_at(pointer, allocation.size)


class VolatileOwner:
    """Own allocations until no committed snapshot, load lease or intent needs them.

    Reconciliation reads authoritative SQLite roots and unexpired preparation
    intents; dropped release RPCs or scheduler crashes therefore cannot leak
    pinned allocations indefinitely. The server never trusts pointers or paths
    supplied by callers. It accepts only same-UID peers and a random capability.
    """
    def __init__(self, root, capacity_bytes, *, reconcile_interval=1.0):
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError('volatile capacity must be nonnegative')
        if not math.isfinite(reconcile_interval) or reconcile_interval <= 0:
            raise ValueError('reconcile interval must be finite and positive')
        self.root, self.capacity_bytes = Path(root), capacity_bytes
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._entries = {}
        self._stop = threading.Event()
        directory = self.root / '.volatile'
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
            raise StoreError('volatile socket directory must be private and owned by current uid')
        self.owner_id = uuid.uuid4().hex
        self.descriptor = dict(owner_id=self.owner_id, token=uuid.uuid4().hex + uuid.uuid4().hex,
                               socket=self.owner_id[:12] + '.sock', pid=self._pid)
        self.path = directory / self.descriptor['socket']
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound=False
        try:
            with _socket_address(self.path) as address:
                self.server.bind(address)
            bound=True
            os.chmod(self.path, 0o600)
            self.server.listen(16)
            self.server.settimeout(0.2)
        except BaseException:
            self.server.close()
            if bound:
                self.path.unlink(missing_ok=True)
            raise
        self._thread = threading.Thread(target=self._serve, name='kvsnap-volatile-ipc', daemon=True)
        self._sweeper = threading.Thread(target=self._sweep, args=(reconcile_interval,), name='kvsnap-volatile-gc', daemon=True)
        self._thread.start()
        self._sweeper.start()

    def _check_open(self):
        if self._stop.is_set() or os.getpid() != self._pid:
            raise StoreError('volatile owner closed or inherited across fork')

    def stats(self):
        with self._lock:
            return dict(resident_bytes=sum(e['allocation'].size for e in self._entries.values()),
                        blob_count=len(self._entries), capacity_bytes=self.capacity_bytes)

    def prepare_allocations(self, allocations, *, session_id, generation, ttl_seconds=300):
        """Consume allocations on every path, including errors and duplicate blobs."""
        # A lazy producer lets the data plane deduplicate each completed block
        # before reserving the next one, avoiding a full-snapshot staging copy.
        pending = allocations if isinstance(allocations, (list, tuple)) else []
        def incoming():
            for allocation in allocations:
                if pending is not allocations:
                    pending.append(allocation)
                yield allocation
        source = incoming()
        adopted = set()
        added = []
        try:
            self._check_open()
            if not isinstance(session_id, str) or not session_id or type(generation) is not int or generation < 0:
                raise ValueError('invalid volatile preparation generation')
            if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
                raise ValueError('preparation TTL must be finite and positive')
            # Hash a materialised batch up front, outside the locks and in
            # parallel; a lazy producer is hashed block by block below.
            digests = None
            if pending is allocations:
                with ThreadPoolExecutor(max_workers=8, thread_name_prefix='kvsnap-hash') as pool:
                    digests = list(pool.map(_digest, allocations))
            # Same ordering as store writers, without taking the SQLite lock.
            with (self.root / '.blob.lock').open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with self._lock:
                    return self._prepare_locked(source, adopted, added, session_id, generation, ttl_seconds,
                                                digests)
        except BaseException:
            with self._lock:
                for key in added:
                    entry = self._entries.pop(key, None)
                    if entry is not None:
                        entry['allocation'].close()
            raise
        finally:
            source.close()
            if hasattr(allocations, 'close'):
                allocations.close()
            for allocation in pending:
                if id(allocation) not in adopted:
                    allocation.close()

    def _prepare_locked(self, allocations, adopted, added, session_id, generation, ttl_seconds, digests=None):
        self._check_open()
        segments, locations = [], {}
        used = self.stats()['resident_bytes']
        for index, allocation in enumerate(allocations):
            if type(allocation.size) is not int or allocation.size <= 0:
                raise ValueError('empty volatile block')
            blob_id, checksum = digests[index] if digests is not None else _digest(allocation)
            if blob_id not in self._entries:
                if used + allocation.size > self.capacity_bytes:
                    raise CapacityError('pinned capacity exhausted; request process/durable spill explicitly')
                self._entries[blob_id] = dict(allocation=allocation, handle=uuid.uuid4().hex)
                added.append(blob_id)
                adopted.add(id(allocation))
                used += allocation.size
            entry = self._entries[blob_id]
            locations[blob_id] = dict(owner=self.descriptor, handle=entry['handle'], length=allocation.size)
            segments.append(dict(blob_id=blob_id, offset=0, length=allocation.size,
                                 first_block=index, nblocks=1, crc32c=checksum))
            if id(allocation) not in adopted:
                allocation.close()
        record = dict(preparation_id=uuid.uuid4().hex, durability='volatile',
                      session_id=session_id, generation=generation,
                      expires_at=time.time() + ttl_seconds, segments=segments, locations=locations)
        _write_preparation(self.root / 'prepared', record)
        return record

    @contextmanager
    def borrow(self, segments, locations, *, verify_checksums=False):
        """Fence GC through local H2D completion; check every handle before any copy.

        These blocks were written by this worker's own D2H into its own pinned
        memory and never left the process, so by default only ownership and
        size are checked. ``verify_checksums`` also CRC-checks each block in
        place (about 12 GB/s), without copying it out of pinned memory.
        """
        with self._lock:
            self._check_open()
            allocations = []
            for segment in segments:
                key = segment['blob_id']
                location = locations[key]
                entry = self._entries.get(key)
                if location['owner'] != self.descriptor or entry is None or entry['handle'] != location['handle']:
                    raise CorruptSnapshotError('volatile handle no longer owned by this worker')
                allocation = entry['allocation']
                if segment['offset'] != 0 or segment['length'] != allocation.size:
                    raise CorruptSnapshotError('volatile block validation failed')
                if verify_checksums:
                    allocation.wait()
                    if get_io_backend().crc32c_at(allocation.pointer, allocation.size) != segment['crc32c']:
                        raise CorruptSnapshotError('volatile block checksum mismatch')
                allocations.append(allocation)
            yield allocations

    def reconcile(self):
        """Rebuild lifetime roots from metadata; no best-effort release dependency."""
        self._check_open()
        with (self.root / '.writer.lock').open('a+b') as writer, (self.root / '.blob.lock').open('a+b') as blobs:
            fcntl.flock(writer, fcntl.LOCK_EX)
            fcntl.flock(blobs, fcntl.LOCK_EX)
            keep = set()
            database = self.root / 'manifests.sqlite3'
            if database.exists():
                with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
                    # Startup may race the scheduler's first schema creation.
                    if db.execute("SELECT 1 FROM sqlite_master WHERE name='volatile_locations'").fetchone():
                        rows = db.execute('''SELECT location_json FROM volatile_locations WHERE blob_id IN
                          (SELECT sb.blob_id FROM snapshot_blobs sb JOIN snapshots s ON sb.snapshot_id=s.snapshot_id
                           WHERE s.durability='volatile' AND s.state='HostValid'
                           UNION SELECT lb.blob_id FROM lease_blobs lb
                           JOIN leases l ON lb.lease_id=l.lease_id WHERE l.expires_at>?)''', (time.time(),))
                        for row in rows:
                            location = json.loads(row[0])
                            if location['owner'] == self.descriptor:
                                keep.add(location['handle'])
            for path in (self.root / 'prepared').glob('*.json'):
                record = json.loads(path.read_text())
                if record['expires_at'] > time.time():
                    for location in record.get('locations', {}).values():
                        if location['owner'] == self.descriptor:
                            keep.add(location['handle'])
            with self._lock:
                for key, entry in list(self._entries.items()):
                    if entry['handle'] not in keep:
                        entry['allocation'].close()
                        del self._entries[key]
            return self.stats()

    def _sweep(self, interval):
        while not self._stop.wait(interval):
            try:
                self.reconcile()
            except (OSError, ValueError, sqlite3.Error, StoreError):
                # Metadata unreadable: retain allocations, never reclaim by guess.
                continue

    def _serve(self):
        while not self._stop.is_set():
            try:
                stream, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with stream:
                stream.settimeout(5.0)
                try:
                    _, uid, _ = struct.unpack('3i', stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    request = _receive_json(stream)
                    if uid != os.getuid() or not hmac.compare_digest(str(request.get('token', '')), self.descriptor['token']):
                        raise ValueError('volatile peer authentication failed')
                    with self._lock:
                        self._check_open()
                        if request.get('op') == 'stats':
                            _send_json(stream, dict(ok=True, **self.stats()))
                            continue
                        if request.get('op') == 'check':
                            for block in request.get('blocks', []):
                                entry = self._entries.get(block.get('blob_id'))
                                if (entry is None or entry['handle'] != block.get('handle')
                                        or entry['allocation'].size != block.get('length')):
                                    raise ValueError('volatile allocation missing or stale handle')
                            _send_json(stream, dict(ok=True))
                            continue
                        if request.get('op') != 'read':
                            raise ValueError('unsupported volatile operation')
                        key = request.get('blob_id')
                        entry = self._entries.get(key)
                        if entry is None or entry['handle'] != request.get('handle'):
                            raise ValueError('volatile allocation missing or stale handle')
                        offset, count = request.get('offset'), request.get('length')
                        if type(offset) is not int or type(count) is not int or offset < 0 or not 0 < count <= MAX_CHUNK:
                            raise ValueError('invalid volatile read bounds')
                        allocation = entry['allocation']
                        if offset + count > allocation.size:
                            raise ValueError('volatile read exceeds allocation')
                        # Production allocations expose a bounded read. Fakes may
                        # use the simpler read() protocol in CPU-only tests.
                        if hasattr(allocation, 'read_range'):
                            data = allocation.read_range(offset, count)
                        else:
                            data = allocation.read()[offset:offset + count]
                        _send_json(stream, dict(ok=True, payload_bytes=len(data)))
                        stream.sendall(data)
                except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
                    try:
                        _send_json(stream, dict(ok=False, error=str(exc)))
                    except OSError:
                        pass

    def close(self):
        if os.getpid() != self._pid:
            raise StoreError('volatile owner cannot be closed by an inheriting process')
        if self._stop.is_set():
            return
        self._stop.set()
        self.server.close()
        self._thread.join(timeout=6)
        self._sweeper.join(timeout=6)
        with self._lock:
            for entry in self._entries.values():
                entry['allocation'].close()
            self._entries.clear()
        self.path.unlink(missing_ok=True)
