"""Single-writer SQLite WAL metadata and immutable checksummed block files.

The Python host pool is a CPU reference implementation, not pinned CUDA memory.
The connector must retain its pinned allocation until its own CUDA event ends.
"""

from __future__ import annotations

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

from .blobs import BlobFiles
from .common import (COMMITTED, DISK_USAGE, FAILPOINTS, HEADER, MAGIC, CapacityError, CorruptSnapshotError,
                     GenerationConflict, StoreError, _dict, _json, _shallow)
from .crc32c import crc32c
from kvsnap.models import rank_segments, tensor_parallel_size
from .maintenance import Maintenance
from .volatile_tier import VolatileTier


class _ProcessLock:
    """Serialize threads and processes, with safe nesting by the same owner."""

    def __init__(self, stream):
        self.stream = stream
        self.thread_lock = threading.RLock()
        self.depth = 0
        self.owner = None

    def __enter__(self):
        self.thread_lock.acquire()
        try:
            if self.depth == 0:
                fcntl.flock(self.stream, fcntl.LOCK_EX)
                self.owner = threading.get_ident()
            self.depth += 1
        except BaseException:
            self.thread_lock.release()
            raise
        return self

    def __exit__(self, *_):
        self.depth -= 1
        if self.depth == 0:
            self.owner = None
            if not self.stream.closed:
                fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.thread_lock.release()

    def held_here(self):
        return self.owner == threading.get_ident()


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


class SnapshotStore(VolatileTier, Maintenance, BlobFiles):
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
        self._read_local = threading.local()
        self._host_bytes_bound = None   # see make_room_volatile
        self._read_connections: list[sqlite3.Connection] = []
        self._read_connections_lock = threading.Lock()
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
                CREATE INDEX IF NOT EXISTS snapshot_blobs_by_blob ON snapshot_blobs(blob_id);
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

    @contextmanager
    def _reader(self):
        """A connection for one read, without waiting for any writer.

        In WAL mode each statement sees one committed state, so a reader needs
        no writer lock. Each thread reads through its own connection: sharing
        the writer's connection made the scheduler's lookups and the host-tier
        size check wait for whatever the commit thread was publishing. A
        thread that holds the writer lock reads through the writer's
        connection, so it sees its own transaction.
        """
        self._check_owner()
        if self._lock.held_here():
            yield self.db
            return
        db = getattr(self._read_local, "db", None)
        if db is None:
            db = sqlite3.connect(self.root / "manifests.sqlite3", isolation_level=None, check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            with self._read_connections_lock:
                self._read_connections.append(db)
            self._read_local.db = db
        yield db

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
        with self._reader() as db:
            return db.execute("SELECT 1 FROM closed_sessions WHERE session_id=?", (session_id,)).fetchone() is not None

    def get_generation(self, session_id):
        with self._reader() as db:
            row = db.execute("SELECT generation FROM sessions WHERE session_id=?", (session_id,)).fetchone()
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

    def commit(self, manifest, blocks: Iterable[bytes]):
        """Commit complete KV blocks, deduplicated by immutable byte content.

        ``blocks`` includes the partial tail block. The caller owns the logical token
        boundary; payload geometry is validated by the data plane/identity contract.
        """
        body = _dict(manifest)
        expected = self._validate_manifest(body)
        if tensor_parallel_size(body["identity"]) != 1:
            raise StoreError("commit takes one rank's blocks: a TP>1 snapshot is committed from one "
                             "preparation per rank (commit_prepared)")
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
            if level == "volatile":
                self._count_host_commit(entries.values())
            self._hit("after_manifest_commit")
            self._hit("before_gpu_release")
            return json.loads(serialized)
        except BaseException as exc:
            # Once COMMIT succeeds, a later failure must never retract the promise.
            state = self.db.execute("SELECT state FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()[0]
            if state == "Writing":
                self._invalidate(snapshot_id, str(exc))
            raise

    def _get_manifest(self, snapshot_id, include_invalid=False, db=None):
        row = (db or self.db).execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
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
        with self._reader() as db:
            return self._get_manifest(snapshot_id, include_invalid, db)

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
        tp = tensor_parallel_size(body["identity"])
        segments = body.get("segments", [])
        if len(segments) != expected * tp:
            raise CorruptSnapshotError("manifest does not cover every KV block"
                                       + (f" of all {tp} TP ranks" if tp > 1 else ""))
        for rank in range(tp):
            for index, segment in enumerate(rank_segments(body, rank)):
                if segment["first_block"] != index or segment["nblocks"] != 1:
                    raise CorruptSnapshotError("invalid or overlapping block mapping")
        result = []
        for segment in segments:
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
        """Scheduler-only atomic publication of previously prepared worker bytes.

        ``prepared`` is one worker's preparation, or at TP>1 a list with one
        per rank, in rank order: the snapshot is published with every rank's
        shard or not at all.
        """
        # Shallow copies suffice: validation rebuilds every nested value, and
        # nothing below mutates the caller's lists. This runs on the scheduler
        # thread, and a deep copy of a manifest holding every session token
        # cost milliseconds per turn.
        body = _shallow(manifest)
        expected_blocks = self._validate_manifest(body)
        records = [_shallow(record) for record in (prepared if isinstance(prepared, (list, tuple)) else [prepared])]
        tp = tensor_parallel_size(body["identity"])
        if len(records) != tp:
            raise StoreError(f"a TP={tp} snapshot needs one preparation per rank, got {len(records)}")
        if records[0].get("durability") == "volatile" and "locations" in records[0]:
            return self._commit_volatile_prepared(body, records, expected_generation)
        for record in records:
            preparation_id = record.get("preparation_id", "")
            if len(preparation_id) != 32 or any(c not in "0123456789abcdef" for c in preparation_id):
                raise ValueError("invalid preparation ID")
        with self._lock, self._blob_lock:
            self._check_owner()
            paths, segments = [], []
            for rank, record in enumerate(records):
                # Use the on-disk intent as authority, not untrusted worker metadata.
                path = self.prepared_dir / (record["preparation_id"] + ".json")
                saved = json.loads(path.read_text())
                if saved != record or record["expires_at"] <= time.time():
                    raise StoreError("preparation differs from intent or expired")
                if record["durability"] != body["durability"]:
                    raise StoreError("preparation durability mismatch")
                if len(record["segments"]) != expected_blocks:
                    raise ValueError("prepared blocks do not cover the KV prefix"
                                     + (f" on TP rank {rank}" if tp > 1 else ""))
                for index, segment in enumerate(record["segments"]):
                    if segment["first_block"] != index or segment["nblocks"] != 1:
                        raise CorruptSnapshotError("invalid or overlapping block mapping")
                    segments.append(dict(segment, rank=rank) if rank else segment)
                paths.append(path)
            body["segments"] = segments
            if expected_generation is not None and body["generation"] != expected_generation:
                raise GenerationConflict("manifest generation differs from expected generation")
            # The worker wrote and fdatasynced every blob before reporting the
            # preparation, so this runs on the scheduler thread without reading
            # payloads: check each blob's header against the intent, then publish.
            entries = {}
            for segment in segments:
                entries[segment["blob_id"]] = self._blob_header(segment["blob_id"], segment)
            result = self._publish(body, entries)
            for path in paths:
                path.unlink(missing_ok=True)
            self._sync_directory(self.prepared_dir)
            return result

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
        with self._reader() as db:
            row = db.execute("SELECT reason FROM failed_saves WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            return None if row is None else row[0]

    def close(self):
        if self._lock_file.closed:
            return
        with self._lock:
            self._check_owner()
            with self._read_connections_lock:
                for db in self._read_connections:
                    db.close()
                self._read_connections.clear()
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


class BlobWriter(BlobFiles):
    """Worker filesystem data plane: no SQLite connection or manifest authority.

    Serializes with store GC through ``.blob.lock``. Preparation reservations
    protect files until publication or expiry. They cannot promise pinned memory.
    """

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
