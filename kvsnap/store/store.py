"""Single-writer SQLite WAL metadata and immutable checksummed block files.

The Python host pool is a CPU reference implementation, not pinned CUDA memory.
The connector must retain its pinned allocation until its own CUDA event ends.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Iterable

from .crc32c import crc32c


HEADER = struct.Struct("<8sIQI8s")
MAGIC = b"KVSBLOB1"
COMMITTED = ("HostValid", "Durable")
FAILPOINTS = ("after_d2h", "during_blob_write", "before_fdatasync",
              "after_rename", "before_manifest_commit", "after_manifest_commit",
              "before_gpu_release", "during_h2d")


class StoreError(RuntimeError):
    pass


class CapacityError(StoreError):
    pass


class CorruptSnapshotError(StoreError):
    pass


class GenerationConflict(StoreError):
    pass


class _ProcessLock:
    """Serialize threads and processes, with safe nesting by the same owner."""

    def __init__(self, stream):
        self.stream = stream
        self.thread_lock = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self.thread_lock.acquire()
        try:
            if self.depth == 0:
                fcntl.flock(self.stream, fcntl.LOCK_EX)
            self.depth += 1
        except BaseException:
            self.thread_lock.release()
            raise
        return self

    def __exit__(self, *_):
        self.depth -= 1
        if self.depth == 0 and not self.stream.closed:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.thread_lock.release()


def _dict(value):
    return copy.deepcopy(value.to_dict() if hasattr(value, "to_dict") else dict(value))


def _shallow(value):
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mkdir_with_parent_sync(path):
    """Persist newly created directory entries all the way to an existing root."""
    path = Path(path)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in missing:
        SnapshotStore._sync_directory(created.parent)


class SnapshotStore:
    """Open a storage directory. Concurrent threads and processes serialize writes.

    Storage requires a local filesystem with working advisory locks and SQLite WAL.
    A process must open its own store after fork, never inherit this connection.
    ``failpoint`` runs synchronously with a named boundary and may kill the process.
    """

    def __init__(self, root, *, host_capacity_bytes=24 * 1024**3,
                 disk_capacity_bytes=200 * 1024**3, failpoint: Callable | None = None,
                 recover_on_open=True, io_backend=None):
        from .native_io import get_io_backend
        self.io = get_io_backend(io_backend)
        if host_capacity_bytes < 0 or disk_capacity_bytes < 0:
            raise ValueError("capacities must be nonnegative")
        self.root = Path(root)
        _mkdir_with_parent_sync(self.root)
        self.blob_dir = self.root / "blobs"
        _mkdir_with_parent_sync(self.blob_dir)
        self.prepared_dir = self.root / "prepared"
        _mkdir_with_parent_sync(self.prepared_dir)
        self.host_capacity_bytes = host_capacity_bytes
        self.disk_capacity_bytes = disk_capacity_bytes
        self.failpoint = failpoint
        self._host: dict[str, bytes] = {}
        self._pid = os.getpid()
        self._lock_file = (self.root / ".writer.lock").open("a+b")
        self._lock = _ProcessLock(self._lock_file)
        self._blob_file = (self.root / ".blob.lock").open("a+b")
        self._blob_lock = _ProcessLock(self._blob_file)
        self._lock.__enter__()
        try:
            self.db = sqlite3.connect(self.root / "manifests.sqlite3", isolation_level=None,
                                      check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                INSERT OR IGNORE INTO metadata VALUES('schema_version','1');
                CREATE TABLE IF NOT EXISTS snapshots(
                  snapshot_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,generation INTEGER NOT NULL,
                  state TEXT NOT NULL,durability TEXT NOT NULL,manifest_json TEXT NOT NULL,error TEXT);
                CREATE TABLE IF NOT EXISTS blobs(
                  blob_id TEXT PRIMARY KEY,length INTEGER NOT NULL,crc32c INTEGER NOT NULL,
                  refcount INTEGER NOT NULL CHECK(refcount>=0));
                CREATE TABLE IF NOT EXISTS snapshot_blobs(
                  snapshot_id TEXT NOT NULL,blob_id TEXT NOT NULL,
                  PRIMARY KEY(snapshot_id,blob_id));
                CREATE TABLE IF NOT EXISTS leases(
                  lease_id TEXT PRIMARY KEY,snapshot_id TEXT NOT NULL,expires_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS lease_blobs(
                  lease_id TEXT NOT NULL,blob_id TEXT NOT NULL,PRIMARY KEY(lease_id,blob_id));
                CREATE TABLE IF NOT EXISTS volatile_locations(
                  blob_id TEXT PRIMARY KEY,location_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY,generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS closed_sessions(session_id TEXT PRIMARY KEY,generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS failed_saves(snapshot_id TEXT PRIMARY KEY,reason TEXT NOT NULL,at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshot_access(snapshot_id TEXT PRIMARY KEY,at REAL NOT NULL);
            """)
            version = self.db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]
            if version != "1":
                raise StoreError(f"unsupported storage schema version: {version}")
            self._sync_directory(self.root)
            self.recovery_report = self.recover() if recover_on_open else None
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self._lock_file.close()
            self._blob_file.close()
            raise
        finally:
            self._lock.__exit__(None, None, None)

    def _check_owner(self):
        if os.getpid() != self._pid:
            raise StoreError("store connection cannot be inherited across fork")
        if self._lock_file.closed:
            raise StoreError("store is closed")

    def _hit(self, name):
        if self.failpoint is not None:
            self.failpoint(name)

    @staticmethod
    def _sync_directory(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _reader(self):
        # A read needs only this process's connection lock. In WAL mode each
        # statement sees one committed state, so a reader never has to wait
        # for another process's commit, fsync included, as it did when every
        # read took the cross-process writer lock.
        with self._lock.thread_lock:
            self._check_owner()
            yield

    @contextmanager
    def _transaction(self, durability="durable"):
        self._check_owner()
        self.db.execute("PRAGMA synchronous=" + ("FULL" if durability == "durable" else "NORMAL"))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def claim_generation(self, session_id, expected_generation, new_generation, *, durable=True):
        """Atomic fencing CAS; generation zero is the initial absent-session value.

        ``durable=False`` skips the fsync. That suits a turn whose snapshot
        lives in host memory: an OS crash loses the claim together with every
        snapshot it fences, while a process crash loses neither.
        """
        from kvsnap.models import identifier

        identifier(session_id, "session_id")
        if (type(expected_generation) is not int or expected_generation < 0 or
                type(new_generation) is not int or new_generation < expected_generation):
            raise ValueError("generation must be a monotonic integer")
        with self._lock, self._transaction("durable" if durable else "volatile"):
            if self.db.execute("SELECT 1 FROM closed_sessions WHERE session_id=?", (session_id,)).fetchone():
                raise GenerationConflict("session is permanently closed")
            row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            current = row[0] if row else 0
            if current != expected_generation:
                raise GenerationConflict(f"expected generation {expected_generation}, current {current}")
            self.db.execute("INSERT INTO sessions VALUES(?,?) ON CONFLICT(session_id) DO UPDATE SET generation=excluded.generation",
                            (session_id, new_generation))
        return new_generation

    def close_session(self, session_id, expected_generation):
        """Atomically fence all old writes and persist the terminal session state."""
        from kvsnap.models import identifier, integer

        identifier(session_id, "session_id")
        integer(expected_generation, "expected_generation")
        with self._lock, self._transaction():
            row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None or row[0] != expected_generation:
                raise GenerationConflict("stale generation while closing session")
            if self.db.execute("SELECT 1 FROM closed_sessions WHERE session_id=?", (session_id,)).fetchone():
                raise GenerationConflict("session is permanently closed")
            generation = expected_generation + 1
            self.db.execute("UPDATE sessions SET generation=? WHERE session_id=?", (generation, session_id))
            self.db.execute("INSERT INTO closed_sessions VALUES(?,?)", (session_id, generation))
            return generation

    def is_session_closed(self, session_id):
        with self._reader():
            return self.db.execute("SELECT 1 FROM closed_sessions WHERE session_id=?", (session_id,)).fetchone() is not None

    def get_generation(self, session_id):
        with self._reader():
            row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            return None if row is None else row[0]

    def _validate_manifest(self, manifest):
        from kvsnap.models import SnapshotManifest

        # The storage boundary enforces the same semantic contract as the API.
        # A caller cannot bypass it by constructing an untyped dictionary.
        validated = SnapshotManifest.from_dict(manifest).to_dict()
        manifest.clear()
        manifest.update(validated)
        required = ("snapshot_id", "session_id", "generation", "identity", "token_ids", "num_kv_tokens",
                    "pending_tokens", "seed", "sampling", "remaining_max_tokens", "output_seq_committed")
        missing = [key for key in required if key not in manifest]
        if missing:
            raise ValueError(f"manifest missing fields: {missing}")
        if not isinstance(manifest["snapshot_id"], str) or not manifest["snapshot_id"]:
            raise ValueError("snapshot_id must be a nonempty string")
        if not isinstance(manifest["session_id"], str) or not manifest["session_id"]:
            raise ValueError("session_id must be a nonempty string")
        if manifest.get("schema_version", 1) != 1:
            raise ValueError("unsupported manifest schema")
        if manifest.get("durability", "volatile") not in ("volatile", "process", "durable"):
            raise ValueError("unsupported durability")
        if type(manifest["generation"]) is not int or manifest["generation"] < 0:
            raise ValueError("invalid generation")
        tokens, count = manifest["token_ids"], manifest["num_kv_tokens"]
        if not isinstance(tokens, list):
            raise ValueError("token_ids must be nonnegative integer IDs")
        if type(count) is not int or not 0 <= count <= len(tokens):
            raise ValueError("invalid KV token boundary")
        if manifest["pending_tokens"] != tokens[count:]:
            raise ValueError("pending_tokens must exactly describe the non-KV token suffix")
        if type(manifest["remaining_max_tokens"]) is not int or manifest["remaining_max_tokens"] < 0:
            raise ValueError("negative remaining token budget")
        block_size = manifest["identity"].get("block_size", 0)
        if type(block_size) is not int or block_size < 1:
            raise ValueError("identity requires positive block_size")
        _json(manifest)  # Reject non-JSON values and NaNs before any mutation.
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("parent_id", None)
        manifest.setdefault("durability", "volatile")
        return (count + block_size - 1) // block_size

    def _path(self, blob_id):
        if len(blob_id) != 64 or any(c not in "0123456789abcdef" for c in blob_id):
            raise CorruptSnapshotError("invalid blob identifier")
        return self.blob_dir / (blob_id + ".blob")

    def _read_blob(self, blob_id, *, disk_only=False):
        if not disk_only and blob_id in self._host:
            data = self._host[blob_id]
            if hashlib.sha256(data).hexdigest() != blob_id:
                raise CorruptSnapshotError("host blob checksum mismatch")
            return data
        path = self._path(blob_id)
        try:
            with path.open("rb") as stream:
                raw = self.io.pread_all(stream.fileno(), HEADER.size, 0)
                if len(raw) != HEADER.size:
                    raise CorruptSnapshotError("truncated blob header")
                magic, version, length, checksum, reserved = HEADER.unpack(raw)
                if magic != MAGIC or version != 1 or reserved != b"\0" * 8:
                    raise CorruptSnapshotError("invalid blob header")
                if length != os.fstat(stream.fileno()).st_size - HEADER.size:
                    raise CorruptSnapshotError("blob length mismatch")
                data = self.io.pread_all(stream.fileno(), length, HEADER.size)
                if crc32c(data) != checksum or hashlib.sha256(data).hexdigest() != blob_id:
                    raise CorruptSnapshotError("blob checksum mismatch")
                return data
        except OSError as exc:
            raise CorruptSnapshotError(f"unreadable blob {blob_id}: {exc}") from exc

    def _blob_header(self, blob_id, segment):
        """(length, crc32c) from a blob's header, checked against its manifest segment."""
        try:
            with self._path(blob_id).open("rb") as stream:
                raw = stream.read(HEADER.size)
                size = os.fstat(stream.fileno()).st_size
        except OSError as exc:
            raise CorruptSnapshotError(f"prepared blob {blob_id} is missing: {exc}") from exc
        if len(raw) != HEADER.size:
            raise CorruptSnapshotError("truncated blob header")
        magic, version, length, checksum, reserved = HEADER.unpack(raw)
        if (magic != MAGIC or version != 1 or reserved != b"\0" * 8 or length != size - HEADER.size
                or segment["offset"] != 0 or segment["length"] != length or segment["crc32c"] != checksum):
            raise CorruptSnapshotError("prepared blob header differs from its manifest segment")
        return length, checksum

    def _write_blob(self, blob_id, data, checksum, durability, *, sync_directory=True):
        final = self._path(blob_id)
        if final.exists():
            # Same SHA-256 name: the header check catches a truncated or
            # foreign file without reading and rehashing the payload.
            self._blob_header(blob_id, dict(offset=0, length=len(data), crc32c=checksum))
            # An earlier process-level commit is promoted before a durable reference.
            if durability == "durable":
                fd = os.open(final, os.O_RDONLY)
                try:
                    os.fdatasync(fd)
                finally:
                    os.close(fd)
                if sync_directory:
                    self._sync_directory(self.blob_dir)
            return
        temporary = self.blob_dir / (blob_id + "." + uuid.uuid4().hex + ".tmp")
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            header = HEADER.pack(MAGIC, 1, len(data), checksum, b"\0" * 8)
            self._pwrite_all(fd, header, 0)
            middle = max(1, len(data) // 2)
            self._pwrite_all(fd, data[:middle], HEADER.size)
            self._hit("during_blob_write")
            self._pwrite_all(fd, data[middle:], HEADER.size + middle)
            self._hit("before_fdatasync")
            if durability == "durable":
                os.fdatasync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, final)
        self._hit("after_rename")
        if durability == "durable" and sync_directory:
            self._sync_directory(self.blob_dir)

    def _pwrite_all(self, fd, data, offset):
        self.io.pwrite_all(fd, data, offset)

    def _admit(self, payloads, durability):
        if durability == "volatile":
            extra = sum(len(data) for key, data in payloads.items() if key not in self._host)
            if sum(map(len, self._host.values())) + extra > self.host_capacity_bytes:
                raise CapacityError("host capacity exhausted; request process/durable spill explicitly")
        else:
            used = sum(p.stat().st_size for p in self.blob_dir.iterdir() if p.is_file())
            extra = sum(HEADER.size + len(data) for key, data in payloads.items() if not self._path(key).exists())
            if used + extra > self.disk_capacity_bytes:
                raise CapacityError("disk capacity exhausted; committed snapshots are never silently evicted")

    def commit(self, manifest, blocks: Iterable[bytes]):
        """Commit complete KV blocks, deduplicated by immutable byte content.

        ``blocks`` includes the partial tail block. The caller owns the logical token
        boundary; payload geometry is validated by the data plane/identity contract.
        """
        body = _dict(manifest)
        expected = self._validate_manifest(body)
        payload_list = [bytes(block) for block in blocks]
        if len(payload_list) != expected or any(not block for block in payload_list):
            raise ValueError(f"expected {expected} nonempty blocks, got {len(payload_list)}")
        payloads, segments = {}, []
        for index, data in enumerate(payload_list):
            key = hashlib.sha256(data).hexdigest()
            payloads[key] = data
            segments.append(dict(blob_id=key, offset=0, length=len(data), first_block=index,
                                 nblocks=1, crc32c=crc32c(data)))
        body["segments"] = segments
        entries = {key: (len(data), crc32c(data)) for key, data in payloads.items()}
        with self._lock, self._blob_lock:
            self._check_owner()
            self._admit(payloads, body["durability"])

            def write():
                if body["durability"] == "volatile":
                    self._host.update(payloads)
                else:
                    for key, data in payloads.items():
                        self._write_blob(key, data, entries[key][1], body["durability"])
            return self._publish(body, entries, write)

    def _publish(self, body, entries, write=None):
        """Insert the snapshot as Writing, run ``write``, then publish it in one transaction.

        ``entries`` maps blob id to (length, crc32c). The caller holds the store
        and blob locks. Prepared snapshots pass no ``write``: their blobs were
        already written and fsynced by the worker before it reported them.
        """
        body["state"] = "Writing"
        level, snapshot_id = body["durability"], body["snapshot_id"]
        with self._transaction(level):
            if self.db.execute("SELECT 1 FROM closed_sessions WHERE session_id=?", (body["session_id"],)).fetchone():
                raise GenerationConflict("cannot commit to a closed session")
            if self.db.execute("SELECT 1 FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone():
                raise StoreError("snapshot IDs are immutable and cannot be reused")
            row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (body["session_id"],)).fetchone()
            if row is not None and row[0] != body["generation"]:
                raise GenerationConflict("stale snapshot generation")
            if row is None:
                self.db.execute("INSERT INTO sessions VALUES(?,?)", (body["session_id"], body["generation"]))
            parent = body.get("parent_id")
            if parent:
                # Validated when it was committed: only identity and prefix matter here.
                ancestor = self._get_manifest(parent, include_invalid=True)
                if (ancestor["identity"] != body["identity"] or
                        body["token_ids"][:len(ancestor["token_ids"])] != ancestor["token_ids"]):
                    raise StoreError("child snapshot must preserve parent identity and token prefix")
            self.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,NULL)",
                            (snapshot_id, body["session_id"], body["generation"], "Writing", level, _json(body)))
        try:
            self._hit("after_d2h")
            if write is not None:
                write()
            body["state"] = "HostValid" if level == "volatile" else "Durable"
            with self._transaction(level):
                # Generation may have advanced only through reentrant application callbacks.
                row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (body["session_id"],)).fetchone()
                if row[0] != body["generation"]:
                    raise GenerationConflict("generation advanced during snapshot commit")
                for key, (length, checksum) in entries.items():
                    self.db.execute("INSERT INTO blobs VALUES(?,?,?,1) ON CONFLICT(blob_id) DO UPDATE SET refcount=refcount+1",
                                    (key, length, checksum))
                    self.db.execute("INSERT INTO snapshot_blobs VALUES(?,?)", (snapshot_id, key))
                serialized = _json(body)
                self.db.execute("UPDATE snapshots SET state=?,manifest_json=? WHERE snapshot_id=?",
                                (body["state"], serialized, snapshot_id))
                self.db.execute("INSERT OR REPLACE INTO snapshot_access VALUES(?,?)", (snapshot_id, time.time()))
                self._hit("before_manifest_commit")
            self._hit("after_manifest_commit")
            self._hit("before_gpu_release")
            return json.loads(serialized)
        except BaseException as exc:
            # Once COMMIT succeeds, a later failure must never retract the promise.
            state = self.db.execute("SELECT state FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()[0]
            if state == "Writing":
                self._invalidate(snapshot_id, str(exc))
            raise

    def _get_manifest(self, snapshot_id, include_invalid=False):
        row = self.db.execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        body = json.loads(row["manifest_json"])
        if any(body.get(key) != row[key] for key in ("snapshot_id", "session_id", "generation", "state", "durability")):
            raise CorruptSnapshotError("manifest body disagrees with SQLite index columns")
        if not include_invalid and row["state"] not in COMMITTED:
            if row["state"] == "Writing":
                # Readers do not take the writer lock, so they can land between
                # a durable commit's two transactions: not published yet.
                raise KeyError(snapshot_id)
            raise CorruptSnapshotError(f"snapshot {snapshot_id} is {row['state']}: {row['error']}")
        return body

    def get_manifest(self, snapshot_id, *, include_invalid=False):
        with self._reader():
            return self._get_manifest(snapshot_id, include_invalid)

    def list_manifests(self, *, include_invalid=False):
        with self._lock:
            self._check_owner()
            rows = self.db.execute("SELECT snapshot_id FROM snapshots ORDER BY rowid").fetchall()
            result = []
            for row in rows:
                body = self._get_manifest(row[0], True)
                if include_invalid or body["state"] in COMMITTED:
                    result.append(body)
            return result

    def _validate_payloads(self, body):
        expected = self._validate_manifest(body)
        segments = body.get("segments", [])
        if len(segments) != expected:
            raise CorruptSnapshotError("manifest does not cover every KV block")
        result = []
        for index, segment in enumerate(segments):
            if segment["first_block"] != index or segment["nblocks"] != 1:
                raise CorruptSnapshotError("invalid or overlapping block mapping")
            data = (self._read_volatile_blob(segment["blob_id"]) if body["durability"] == "volatile"
                    else self._read_blob(segment["blob_id"], disk_only=True))
            offset, length = segment["offset"], segment["length"]
            if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or offset + length > len(data):
                raise CorruptSnapshotError("segment outside blob bounds")
            value = data[offset:offset + length]
            if crc32c(value) != segment["crc32c"]:
                raise CorruptSnapshotError("manifest segment checksum mismatch")
            result.append(value)
        return result

    def load(self, snapshot_id, *, identity=None, token_prefix=None):
        """Validate all bytes before exposing any block; mismatch never partially loads."""
        with self._lock:
            self._check_owner()
            body = self._get_manifest(snapshot_id)
            if body["durability"] == "volatile" and any(
                    segment["blob_id"] not in self._host and not self.db.execute(
                        "SELECT 1 FROM volatile_locations WHERE blob_id=?", (segment["blob_id"],)).fetchone()
                    for segment in body["segments"]):
                # The deliberately process-local CPU fixture has no remote
                # ownership record. A different connection cannot judge whether
                # that fixture's owning process is still alive.
                raise StoreError("volatile CPU reference bytes are not resident in this store process")
            if identity is not None and _dict(identity) != body["identity"]:
                raise StoreError("snapshot identity mismatch")
            if token_prefix is not None and list(token_prefix)[:len(body["token_ids"])] != body["token_ids"]:
                raise StoreError("snapshot token prefix mismatch")
            with self.loading_lease(snapshot_id):
                try:
                    result = self._validate_payloads(body)
                except (CorruptSnapshotError, ValueError, KeyError, TypeError) as exc:
                    self._invalidate(snapshot_id, str(exc))
                    raise CorruptSnapshotError(str(exc)) from exc
                self._hit("during_h2d")  # CPU boundary; actual CUDA injection belongs to data plane.
                return result

    read_blocks = load

    def prepare_blocks(self, blocks, durability="durable", *, ttl_seconds=300):
        """Prepare worker payloads without committing a snapshot.

        Cross-process workers use :class:`BlobWriter`; volatile preparations are
        valid only in this store process and use ordinary CPU reference buffers.
        """
        with self._lock:
            self._check_owner()
            if durability != "volatile":
                return BlobWriter(self.root, disk_capacity_bytes=self.disk_capacity_bytes,
                                  failpoint=self.failpoint, io_backend=self.io.name).prepare_blocks(
                                      blocks, durability, ttl_seconds=ttl_seconds)
            if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
                raise ValueError("preparation TTL must be finite and positive")
            payloads, segments = _prepare_payloads(blocks)
            self._admit(payloads, durability)
            self._host.update(payloads)
            record = dict(preparation_id=uuid.uuid4().hex, durability=durability,
                          expires_at=time.time() + ttl_seconds, segments=segments)
            _write_preparation(self.prepared_dir, record)
            return record

    def commit_prepared(self, manifest, prepared, expected_generation=None):
        """Scheduler-only atomic publication of previously prepared worker bytes."""
        # Shallow copies suffice: validation rebuilds every nested value, and
        # nothing below mutates the caller's lists. This runs on the scheduler
        # thread, and a deep copy of a manifest holding every session token
        # cost milliseconds per turn.
        body = _shallow(manifest)
        expected_blocks = self._validate_manifest(body)
        record = _shallow(prepared)
        if record.get("durability") == "volatile" and "locations" in record:
            return self._commit_volatile_prepared(body, record, expected_generation)
        preparation_id = record.get("preparation_id", "")
        if len(preparation_id) != 32 or any(c not in "0123456789abcdef" for c in preparation_id):
            raise ValueError("invalid preparation ID")
        with self._lock, self._blob_lock:
            self._check_owner()
            # Use the on-disk intent as authority, not untrusted worker metadata.
            path = self.prepared_dir / (preparation_id + ".json")
            saved = json.loads(path.read_text())
            if saved != record or record["expires_at"] <= time.time():
                raise StoreError("preparation differs from intent or expired")
            if record["durability"] != body["durability"]:
                raise StoreError("preparation durability mismatch")
            body["segments"] = record["segments"]
            if len(body["segments"]) != expected_blocks:
                raise ValueError("prepared blocks do not cover the KV prefix")
            if expected_generation is not None and body["generation"] != expected_generation:
                raise GenerationConflict("manifest generation differs from expected generation")
            # The worker wrote and fdatasynced every blob before reporting the
            # preparation, so this runs on the scheduler thread without reading
            # payloads: check each blob's header against the intent, then publish.
            entries = {}
            for index, segment in enumerate(body["segments"]):
                if segment["first_block"] != index or segment["nblocks"] != 1:
                    raise CorruptSnapshotError("invalid or overlapping block mapping")
                entries[segment["blob_id"]] = self._blob_header(segment["blob_id"], segment)
            result = self._publish(body, entries)
            path.unlink(missing_ok=True)
            self._sync_directory(self.prepared_dir)
            return result

    def volatile_locations(self, manifest):
        """Opaque worker handles; no process-local pointer is exposed."""
        keys = sorted({segment["blob_id"] for segment in manifest["segments"]})
        with self._reader():
            result = self._committed_volatile_locations(keys)
        if len(result) != len(keys):
            raise CorruptSnapshotError("volatile snapshot has no worker-owned allocation")
        return result

    def _committed_volatile_locations(self, keys):
        # One statement for any number of keys: a snapshot has one row per block.
        rows = self.db.execute("SELECT blob_id,location_json FROM volatile_locations "
                               "WHERE blob_id IN (SELECT value FROM json_each(?))", (json.dumps(keys),))
        return {row[0]: json.loads(row[1]) for row in rows}

    def _read_volatile_blob(self, key):
        from kvsnap.connector.volatile import VolatileClient
        row = self.db.execute("SELECT location_json FROM volatile_locations WHERE blob_id=?", (key,)).fetchone()
        if row is not None:
            location = json.loads(row[0])
            return VolatileClient(self.root, location["owner"]).read(key, location)
        if key in self._host:
            return self._read_blob(key)
        raise CorruptSnapshotError("volatile owner or resident allocation is unavailable")

    def _live_volatile_locations(self, keys):
        """Committed locations of ``keys`` whose owner still holds the allocation.

        A turn re-saves every block of the turns before it, so most keys are
        already committed. They are checked in one round trip per owner, not
        one per block: per-block checks held the scheduler thread for tens of
        milliseconds at every turn end. Only an owner that fails the batch is
        asked about each of its blocks.
        """
        from kvsnap.connector.volatile import VolatileClient
        owners = {}
        for key, location in self._committed_volatile_locations(keys).items():
            owners.setdefault(location["owner"]["owner_id"], (location["owner"], {}))[1][key] = location
        live = {}
        for owner, held in owners.values():
            client = VolatileClient(self.root, owner)
            try:
                client.check(held)
                live.update(held)
                continue
            except CorruptSnapshotError:
                pass
            for key, location in held.items():
                try:
                    client.check({key: location})
                    live[key] = location
                except CorruptSnapshotError:
                    pass
        return live

    def _commit_volatile_prepared(self, body, record, expected_generation):
        from kvsnap.connector.volatile import VolatileClient
        ident = record.get("preparation_id", "")
        if len(ident) != 32 or any(c not in "0123456789abcdef" for c in ident):
            raise ValueError("invalid preparation ID")
        with self._lock, self._blob_lock:
            self._check_owner()
            path = self.prepared_dir / (ident + ".json")
            if json.loads(path.read_text()) != record or record["expires_at"] <= time.time():
                raise StoreError("volatile preparation differs from intent or expired")
            if (body["durability"] != "volatile" or body["session_id"] != record["session_id"] or
                    body["generation"] != record["generation"]):
                raise GenerationConflict("volatile preparation identity or generation mismatch")
            if expected_generation is not None and body["generation"] != expected_generation:
                raise GenerationConflict("manifest generation differs from expected generation")
            body["segments"], body["state"] = record["segments"], "HostValid"
            expected = self._validate_manifest(body)
            if len(body["segments"]) != expected:
                raise CorruptSnapshotError("volatile preparation does not cover KV blocks")
            keys = {segment["blob_id"] for segment in body["segments"]}
            if keys != set(record["locations"]):
                raise CorruptSnapshotError("volatile location set differs from manifest")
            # The worker hashed each block in place right after its D2H fence
            # and wrote the digests into the intent. Here the owner only has to
            # confirm, in one round trip, that it still holds every handle at
            # the recorded length; no payload crosses the socket.
            for segment in body["segments"]:
                location = record["locations"][segment["blob_id"]]
                if segment["offset"] != 0 or segment["length"] != location["length"]:
                    raise CorruptSnapshotError("volatile segment length differs from its allocation")
            owners = {}
            for key, location in record["locations"].items():
                owners.setdefault(location["owner"]["owner_id"], (location["owner"], {}))[1][key] = location
            for owner, locations in owners.values():
                VolatileClient(self.root, owner).check(locations)
            with self._transaction("volatile"):
                if self.db.execute("SELECT 1 FROM closed_sessions WHERE session_id=?", (body["session_id"],)).fetchone():
                    raise GenerationConflict("cannot commit to a closed session")
                row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (body["session_id"],)).fetchone()
                if row and row[0] != body["generation"]:
                    raise GenerationConflict("stale snapshot generation")
                if self.db.execute("SELECT 1 FROM snapshots WHERE snapshot_id=?", (body["snapshot_id"],)).fetchone():
                    raise StoreError("snapshot IDs are immutable and cannot be reused")
                if body.get("parent_id"):
                    # Validated when it was committed: only identity and prefix matter here.
                    parent = self._get_manifest(body["parent_id"], include_invalid=True)
                    if (parent["identity"] != body["identity"] or
                            body["token_ids"][:len(parent["token_ids"])] != parent["token_ids"]):
                        raise StoreError("child snapshot must preserve parent identity and token prefix")
                self.db.execute("INSERT OR IGNORE INTO sessions VALUES(?,?)", (body["session_id"], body["generation"]))
                self.db.execute("INSERT OR REPLACE INTO snapshot_access VALUES(?,?)", (body["snapshot_id"], time.time()))
                serialized = _json(body)
                self.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,NULL)",
                                (body["snapshot_id"], body["session_id"], body["generation"],
                                 body["state"], body["durability"], serialized))
                # Keep a live canonical allocation when another worker
                # prepares the same bytes; never move existing roots to an
                # unnecessary duplicate owner that might die independently.
                locations = dict(record["locations"])
                locations.update(self._live_volatile_locations(list(locations)))
                segments = {segment["blob_id"]: segment for segment in body["segments"]}
                self.db.executemany("INSERT INTO volatile_locations VALUES(?,?) ON CONFLICT(blob_id) DO UPDATE SET location_json=excluded.location_json",
                                    [(key, _json(location)) for key, location in locations.items()])
                self.db.executemany("INSERT INTO blobs VALUES(?,?,?,1) ON CONFLICT(blob_id) DO UPDATE SET refcount=refcount+1",
                                    [(key, segments[key]["length"], segments[key]["crc32c"]) for key in locations])
                self.db.executemany("INSERT INTO snapshot_blobs VALUES(?,?)",
                                    [(body["snapshot_id"], key) for key in locations])
                self._hit("before_manifest_commit")
            # No directory fsync: if a crash brings the intent back, it is an
            # expired temporary root that collection reclaims, and the host
            # memory it described is gone with the process anyway.
            path.unlink(missing_ok=True)
            self._hit("after_manifest_commit")
            self._hit("before_gpu_release")
            return json.loads(serialized)

    def acquire_lease(self, snapshot_id, *, ttl_seconds=300):
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("lease TTL must be finite and positive")
        # A lease is a GC root for an in-flight load and dies with the process
        # that holds it, so its commit does not need an fsync (it runs on the
        # scheduler's critical path for every restore).
        with self._lock, self._transaction("volatile"):
            self._get_manifest(snapshot_id)
            lease_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO leases VALUES(?,?,?)", (lease_id, snapshot_id, time.time() + ttl_seconds))
            self.db.execute("INSERT INTO lease_blobs SELECT ?,blob_id FROM snapshot_blobs WHERE snapshot_id=?",
                            (lease_id, snapshot_id))
            self.db.execute("INSERT OR REPLACE INTO snapshot_access VALUES(?,?)", (snapshot_id, time.time()))
            return lease_id

    def renew_lease(self, lease_id, *, ttl_seconds=300):
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("lease TTL must be finite and positive")
        with self._lock, self._transaction():
            result = self.db.execute("UPDATE leases SET expires_at=? WHERE lease_id=? AND expires_at>?",
                                     (time.time() + ttl_seconds, lease_id, time.time()))
            if result.rowcount != 1:
                raise StoreError("lease missing or expired")

    def release_lease(self, lease_id):
        with self._lock, self._transaction("volatile"):
            self.db.execute("DELETE FROM lease_blobs WHERE lease_id=?", (lease_id,))
            self.db.execute("DELETE FROM leases WHERE lease_id=?", (lease_id,))

    @contextmanager
    def loading_lease(self, snapshot_id, *, ttl_seconds=300):
        lease_id = self.acquire_lease(snapshot_id, ttl_seconds=ttl_seconds)
        try:
            yield lease_id
        finally:
            self.release_lease(lease_id)

    def record_save_failure(self, snapshot_id, reason):
        """Tell the gateway a requested snapshot will never be committed."""
        with self._lock, self._transaction("volatile"):
            self.db.execute("INSERT OR REPLACE INTO failed_saves VALUES(?,?,?)", (snapshot_id, reason, time.time()))

    def save_failure(self, snapshot_id):
        with self._reader():
            row = self.db.execute("SELECT reason FROM failed_saves WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            return None if row is None else row[0]

    def volatile_bytes(self):
        """Bytes of distinct blobs referenced by committed host-memory snapshots."""
        with self._reader():
            return self.db.execute(
                "SELECT COALESCE(SUM(length),0) FROM blobs WHERE blob_id IN (SELECT sb.blob_id FROM snapshot_blobs sb "
                "JOIN snapshots s ON s.snapshot_id=sb.snapshot_id WHERE s.durability='volatile' AND s.state='HostValid')"
            ).fetchone()[0]

    def make_room_volatile(self, needed_bytes, capacity_bytes):
        """Invalidate least recently used host-memory snapshots until ``needed_bytes`` fit.

        The host tier is a cache: an evicted session's next turn misses and is
        recomputed. A snapshot under a load lease is never evicted. Recency is
        the last commit or restore. Returns the evicted snapshot ids; if only
        leased snapshots remain, the caller's save may still not fit.
        """
        evicted = []
        # Usually nothing has to go, and that needs no writer lock: this runs
        # on the scheduler thread at every turn end.
        if self.volatile_bytes() + needed_bytes <= capacity_bytes:
            return evicted
        with self._lock:
            self._check_owner()
            while self.volatile_bytes() + needed_bytes > capacity_bytes:
                row = self.db.execute(
                    "SELECT s.snapshot_id FROM snapshots s LEFT JOIN snapshot_access a ON a.snapshot_id=s.snapshot_id "
                    "WHERE s.durability='volatile' AND s.state='HostValid' AND s.snapshot_id NOT IN "
                    "(SELECT snapshot_id FROM leases WHERE expires_at>?) ORDER BY COALESCE(a.at,0) LIMIT 1",
                    (time.time(),)).fetchone()
                if row is None:
                    break
                self._invalidate(row[0], "evicted from the host tier (least recently used)")
                evicted.append(row[0])
        return evicted

    def invalidate_snapshot(self, snapshot_id, reason):
        with self._lock:
            self._check_owner()
            # A load lease can outlive an explicitly deleted manifest.
            if not self.db.execute("SELECT 1 FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone():
                return
            self._invalidate(snapshot_id, reason)

    def _invalidate(self, snapshot_id, reason):
        with self._transaction():
            body = self._get_manifest(snapshot_id, True)
            body["state"] = "Invalid"
            self.db.execute("UPDATE snapshots SET state='Invalid',manifest_json=?,error=? WHERE snapshot_id=?",
                            (_json(body), reason, snapshot_id))
            self.db.execute("DELETE FROM snapshot_blobs WHERE snapshot_id=?", (snapshot_id,))
            self._rebuild_references()

    def _rebuild_references(self):
        self.db.execute("DELETE FROM snapshot_blobs")
        for row in self.db.execute("SELECT snapshot_id,manifest_json FROM snapshots WHERE state IN ('HostValid','Durable')").fetchall():
            for segment in json.loads(row["manifest_json"]).get("segments", []):
                key = segment["blob_id"]
                self.db.execute("INSERT OR IGNORE INTO blobs VALUES(?,?,?,0)",
                                (key, segment["length"], segment["crc32c"]))
                self.db.execute("INSERT OR IGNORE INTO snapshot_blobs VALUES(?,?)", (row["snapshot_id"], key))
        self.db.execute("UPDATE blobs SET refcount=(SELECT count(*) FROM snapshot_blobs WHERE snapshot_blobs.blob_id=blobs.blob_id)")

    def recover(self):
        """Reject interrupted/volatile/corrupt snapshots and rebuild reachable counts."""
        with self._lock, self._blob_lock:
            self._check_owner()
            report = {"valid": [], "invalid": [], "temporary_files": []}
            invalid_updates = []
            for row in self.db.execute("SELECT * FROM snapshots").fetchall():
                reason = None
                if row["state"] == "Invalid":
                    report["invalid"].append(row["snapshot_id"])
                    continue
                try:
                    body = json.loads(row["manifest_json"])
                    if not isinstance(body, dict):
                        raise ValueError("manifest JSON is not an object")
                    if any(body.get(key) != row[key] for key in ("snapshot_id", "session_id", "generation", "state", "durability")):
                        raise CorruptSnapshotError("manifest body disagrees with SQLite index columns")
                    if row["state"] not in COMMITTED:
                        reason = "interrupted uncommitted snapshot"
                    else:
                        self._validate_payloads(body)
                except (CorruptSnapshotError, ValueError, KeyError, TypeError) as exc:
                    reason = str(exc)
                    # Preserve diagnosability even when the JSON itself is damaged.
                    try:
                        body = json.loads(row["manifest_json"])
                        if not isinstance(body, dict):
                            body = {}
                    except ValueError:
                        body = {}
                    body.update({key: row[key] for key in ("snapshot_id", "session_id", "generation", "durability")})
                if reason:
                    body["state"] = "Invalid"
                    invalid_updates.append((_json(body), reason, row["snapshot_id"]))
                    report["invalid"].append(row["snapshot_id"])
                else:
                    report["valid"].append(row["snapshot_id"])
            with self._transaction():
                self.db.executemany("UPDATE snapshots SET state='Invalid',manifest_json=?,error=? WHERE snapshot_id=?",
                                    invalid_updates)
                self._rebuild_references()
                self._expire_leases()
            for path in self.blob_dir.glob("*.tmp"):
                if path.name.split(".")[0] in self._intent_blobs():
                    continue                  # a live writer's file (BlobWriter writes unlocked)
                report["temporary_files"].append(path.name)
                path.unlink()
            if report["temporary_files"]:
                self._sync_directory(self.blob_dir)
            return report

    def _intent_blobs(self):
        """Blob ids listed by unexpired preparation intents."""
        keys = set()
        for path in self.prepared_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if record.get("expires_at", 0) > time.time():
                keys.update(segment["blob_id"] for segment in record.get("segments", []))
        return keys

    def fork(self, source_id, new_id, session_id, generation=0, *, expected_parent_generation=None):
        """Share immutable persistent segments; GPU targets are separately allocated.

        This operation does not establish GPU block sharing or device CoW.
        """
        with self._lock:
            self._check_owner()
            source = self._get_manifest(source_id)
            if expected_parent_generation is not None:
                row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (source["session_id"],)).fetchone()
                if row is None or row[0] != expected_parent_generation:
                    raise GenerationConflict("stale source session generation")
            # Validate parent bytes before making another promise about them.
            self.load(source_id)
            body = copy.deepcopy(source)
            body.update(snapshot_id=new_id, session_id=session_id, generation=generation, parent_id=source_id)
            self._validate_manifest(body)
            with self._transaction(body["durability"]):
                if self.db.execute("SELECT 1 FROM closed_sessions WHERE session_id=?", (session_id,)).fetchone():
                    raise GenerationConflict("fork target session is closed")
                if self.db.execute("SELECT 1 FROM snapshots WHERE snapshot_id=?", (new_id,)).fetchone():
                    raise StoreError("snapshot IDs cannot be reused")
                row = self.db.execute("SELECT generation FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                if row and row[0] != generation:
                    raise GenerationConflict("stale fork generation")
                self.db.execute("INSERT OR IGNORE INTO sessions VALUES(?,?)", (session_id, generation))
                self.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,NULL)",
                                (new_id, session_id, generation, body["state"], body["durability"], _json(body)))
                for key in {s["blob_id"] for s in body["segments"]}:
                    self.db.execute("INSERT INTO snapshot_blobs VALUES(?,?)", (new_id, key))
                    self.db.execute("UPDATE blobs SET refcount=refcount+1 WHERE blob_id=?", (key,))
            return body

    def delete_snapshot(self, snapshot_id):
        with self._lock, self._transaction():
            self._get_manifest(snapshot_id, True)
            # Parent metadata is retained while children exist, preserving a valid DAG.
            for row in self.db.execute("SELECT manifest_json FROM snapshots").fetchall():
                if json.loads(row[0]).get("parent_id") == snapshot_id:
                    raise StoreError("delete child snapshots before their parent")
            self.db.execute("DELETE FROM snapshot_blobs WHERE snapshot_id=?", (snapshot_id,))
            self.db.execute("DELETE FROM snapshots WHERE snapshot_id=?", (snapshot_id,))
            self._rebuild_references()

    def retire_snapshot(self, snapshot_id):
        """Logically delete one owner while retaining immutable DAG metadata.

        Forks keep their own segment references; active load leases remain GC
        roots. The tombstone cannot be resumed/forked, and physical metadata
        deletion still requires children to be removed first.
        """
        with self._lock:
            self._check_owner()
            self._get_manifest(snapshot_id)
            self._invalidate(snapshot_id, "snapshot owner explicitly retired; DAG metadata retained")

    def _expire_leases(self):
        self.db.execute("DELETE FROM lease_blobs WHERE lease_id IN (SELECT lease_id FROM leases WHERE expires_at<=?)", (time.time(),))
        self.db.execute("DELETE FROM leases WHERE expires_at<=?", (time.time(),))

    def collect_garbage(self):
        with self._lock, self._blob_lock:
            self._check_owner()
            with self._transaction():
                self._expire_leases()
                self._rebuild_references()
                reachable = {row[0] for row in self.db.execute("SELECT blob_id FROM snapshot_blobs UNION SELECT blob_id FROM lease_blobs")}
                volatile_reachable = {row[0] for row in self.db.execute(
                    "SELECT sb.blob_id FROM snapshot_blobs sb JOIN snapshots s ON s.snapshot_id=sb.snapshot_id "
                    "WHERE s.durability='volatile' UNION SELECT blob_id FROM lease_blobs")}
                for path in self.prepared_dir.glob("*.json"):
                    record = json.loads(path.read_text())
                    if record["expires_at"] > time.time():
                        reachable.update(segment["blob_id"] for segment in record["segments"])
                        if record["durability"] == "volatile":
                            volatile_reachable.update(segment["blob_id"] for segment in record["segments"])
                    else:
                        path.unlink()
                self.db.execute("DELETE FROM blobs WHERE refcount=0 AND blob_id NOT IN (SELECT blob_id FROM lease_blobs)")
                for row in self.db.execute("SELECT blob_id FROM volatile_locations").fetchall():
                    if row[0] not in volatile_reachable:
                        self.db.execute("DELETE FROM volatile_locations WHERE blob_id=?", (row[0],))
            removed = []
            for path in self.blob_dir.iterdir():
                # A temporary file named for a blob of a live intent is a write
                # in progress (BlobWriter writes without holding the blob lock).
                if path.suffix in (".tmp", ".blob") and path.name.split(".")[0] not in reachable:
                    path.unlink()
                    removed.append(path.name)
            for key in list(self._host):
                if key not in reachable:
                    del self._host[key]
            if removed:
                self._sync_directory(self.blob_dir)
            return removed

    def export(self, path=None):
        with self._lock:
            self._check_owner()
            data = dict(schema_version=1, snapshots=self.list_manifests(include_invalid=True),
                        blobs=[dict(row) for row in self.db.execute("SELECT * FROM blobs ORDER BY blob_id")],
                        leases=[dict(row) for row in self.db.execute("SELECT * FROM leases ORDER BY lease_id")],
                        lease_blobs=[dict(row) for row in self.db.execute("SELECT * FROM lease_blobs ORDER BY lease_id,blob_id")],
                        sessions=[dict(row) for row in self.db.execute("SELECT * FROM sessions ORDER BY session_id")])
            if path is not None:
                Path(path).write_text(_json(data) + "\n")
            return data

    def stats(self):
        with self._lock:
            self._check_owner()
            remote = {row[0]: row[1] for row in self.db.execute(
                "SELECT v.blob_id,b.length FROM volatile_locations v JOIN blobs b ON b.blob_id=v.blob_id "
                "WHERE EXISTS(SELECT 1 FROM snapshot_blobs sb JOIN snapshots s ON sb.snapshot_id=s.snapshot_id "
                "WHERE sb.blob_id=v.blob_id AND s.durability='volatile')")}
            remote_bytes = sum(length for key, length in remote.items() if key not in self._host)
            return dict(host_bytes=sum(map(len, self._host.values())) + remote_bytes,
                        worker_host_bytes=remote_bytes,
                        disk_bytes=sum(p.stat().st_size for p in self.blob_dir.iterdir() if p.is_file()),
                        committed_snapshots=len(self.list_manifests()),
                        unique_blobs=self.db.execute("SELECT count(*) FROM blobs WHERE refcount>0").fetchone()[0])

    def close(self):
        if self._lock_file.closed:
            return
        with self._lock:
            self._check_owner()
            self.db.close()
            self._host.clear()
            self._blob_file.close()
            self._lock_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _prepare_payloads(blocks):
    blocks = [bytes(block) for block in blocks]
    if any(not data for data in blocks):
        raise ValueError("KV blocks must not be empty")
    # SHA-256 releases the GIL on large buffers; hash blocks in parallel.
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="kvsnap-hash") as pool:
        digests = list(pool.map(lambda data: (hashlib.sha256(data).hexdigest(), crc32c(data)), blocks))
    payloads, segments = {}, []
    for index, (data, (key, checksum)) in enumerate(zip(blocks, digests)):
        payloads[key] = data
        segments.append(dict(blob_id=key, offset=0, length=len(data), first_block=index,
                             nblocks=1, crc32c=checksum))
    return payloads, segments


def _write_preparation(directory, record):
    temporary = directory / (record["preparation_id"] + ".tmp")
    with temporary.open("x") as stream:
        stream.write(_json(record))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / (record["preparation_id"] + ".json"))
    SnapshotStore._sync_directory(directory)


class BlobWriter:
    """Worker filesystem data plane: no SQLite connection or manifest authority.

    Serializes with store GC through ``.blob.lock``. Preparation reservations
    protect files until publication or expiry. They cannot promise pinned memory.
    """

    _path = SnapshotStore._path
    _read_blob = SnapshotStore._read_blob
    _write_blob = SnapshotStore._write_blob
    _pwrite_all = SnapshotStore._pwrite_all
    _sync_directory = staticmethod(SnapshotStore._sync_directory)
    _hit = SnapshotStore._hit
    _admit = SnapshotStore._admit
    _blob_header = SnapshotStore._blob_header

    def __init__(self, root, *, disk_capacity_bytes=200 * 1024**3, failpoint=None, io_backend=None):
        from .native_io import get_io_backend
        self.io = get_io_backend(io_backend)
        self.root = Path(root)
        self.blob_dir = self.root / "blobs"
        self.prepared_dir = self.root / "prepared"
        _mkdir_with_parent_sync(self.blob_dir)
        _mkdir_with_parent_sync(self.prepared_dir)
        self.disk_capacity_bytes = disk_capacity_bytes
        self.failpoint = failpoint
        self._host = {}

    def prepare_blocks(self, blocks, durability="durable", *, ttl_seconds=300):
        if durability not in ("process", "durable"):
            raise ValueError("worker filesystem preparation requires process or durable")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("preparation TTL must be finite and positive")
        payloads, segments = _prepare_payloads(blocks)
        record = dict(preparation_id=uuid.uuid4().hex, durability=durability,
                      expires_at=time.time() + ttl_seconds, segments=segments)
        # The intent is a GC root for every blob it lists, temporary files
        # included, so only its publication needs the blob lock. Writes and
        # fdatasyncs then run unlocked and never queue restores behind them.
        with (self.root / ".blob.lock").open("a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            self._admit(payloads, durability)
            _write_preparation(self.prepared_dir, record)
        checksums = {segment["blob_id"]: segment["crc32c"] for segment in segments}
        # Each blob is written and fdatasynced independently; doing them in
        # parallel overlaps the device flushes. The directory entries are then
        # made durable with a single directory fsync.
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="kvsnap-write") as pool:
            list(pool.map(lambda item: self._write_blob(item[0], item[1], checksums[item[0]], durability,
                                                        sync_directory=False), payloads.items()))
        if durability == "durable":
            self._sync_directory(self.blob_dir)
        return record

    def read_prepared_into(self, segments, address, block_bytes, *, first_block=0, workers=8):
        """Read blocks straight into packed pinned memory and CRC-check them there.

        Block ``first_block + i`` lands at ``address + i * block_bytes``. Every
        block is checked against its manifest CRC32C before the caller starts
        any H2D copy. This is the restore hot path: one pread per block into its
        final pinned slot, no intermediate bytes objects and no SHA-256 (the
        content address is verified when a blob is written, not on every read).
        Blocks are read by ``workers`` threads so a cold read keeps several
        requests in flight on the device; pread and CRC release the GIL.
        """
        for index, segment in enumerate(segments, start=first_block):
            if segment["first_block"] != index or segment["nblocks"] != 1:
                raise CorruptSnapshotError("noncontiguous prepared block mapping")
            offset, length = segment["offset"], segment["length"]
            if type(offset) is not int or type(length) is not int or offset < 0 or length != block_bytes:
                raise CorruptSnapshotError("prepared segment does not match the KV block size")

        def read(item):
            index, segment = item
            target = address + index * block_bytes
            offset, length = segment["offset"], segment["length"]
            try:
                with self._path(segment["blob_id"]).open("rb") as stream:
                    fd = stream.fileno()
                    magic, version, size, _, reserved = HEADER.unpack(os.pread(fd, HEADER.size, 0))
                    if magic != MAGIC or version != 1 or reserved != b"\0" * 8 or offset + length > size:
                        raise CorruptSnapshotError("invalid blob header or segment bounds")
                    self.io.pread_into(fd, target, length, HEADER.size + offset)
            except (OSError, struct.error) as exc:
                raise CorruptSnapshotError(f"unreadable blob {segment['blob_id']}: {exc}") from exc
            if self.io.crc32c_at(target, length) != segment["crc32c"]:
                raise CorruptSnapshotError("prepared segment checksum mismatch")

        with (self.root / ".blob.lock").open("a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kvsnap-read") as pool:
                list(pool.map(read, enumerate(segments)))

    def read_prepared_blocks(self, segments):
        """Validate every immutable block before returning any bytes to H2D."""
        result = []
        with (self.root / ".blob.lock").open("a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            for index, segment in enumerate(segments):
                if segment["first_block"] != index or segment["nblocks"] != 1:
                    raise CorruptSnapshotError("noncontiguous prepared block mapping")
                data = self._read_blob(segment["blob_id"], disk_only=True)
                offset, length = segment["offset"], segment["length"]
                if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or offset + length > len(data):
                    raise CorruptSnapshotError("prepared segment outside blob bounds")
                block = data[offset:offset + length]
                if crc32c(block) != segment["crc32c"]:
                    raise CorruptSnapshotError("prepared segment checksum mismatch")
                result.append(block)
        return result
