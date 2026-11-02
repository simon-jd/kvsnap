"""Blob files: one checksummed block per file under ``blobs/``, and the disk-usage total.

The store and the worker's ``BlobWriter`` share these methods.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid

from .common import CapacityError, CorruptSnapshotError, DISK_USAGE, HEADER, MAGIC
from .crc32c import crc32c


class BlobFiles:
    """Blob-file methods shared by the store and the worker's writer."""

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
        """Check capacity and, on disk, reserve it. The caller holds the exclusive blob lock."""
        if durability == "volatile":
            extra = sum(len(data) for key, data in payloads.items() if key not in self._host)
            if sum(map(len, self._host.values())) + extra > self.host_capacity_bytes:
                raise CapacityError("host capacity exhausted; request process/durable spill explicitly")
        else:
            used = self._disk_usage()
            extra = sum(HEADER.size + len(data) for key, data in payloads.items() if not self._path(key).exists())
            if used + extra > self.disk_capacity_bytes:
                raise CapacityError("disk capacity exhausted; committed snapshots are never silently evicted")
            self._record_disk_usage(used + extra)

    def _disk_usage(self):
        """Bytes of blob files plus admitted blobs not yet written. The caller holds the blob lock.

        Admission adds a new blob's bytes before it is written, instead of
        stat-ing every blob file on every save; garbage collection and
        recovery recount from the directory. A failed write leaves its
        reservation until the next recount, which errs towards refusing a
        save. The file is only a cache: missing or unreadable, it is recounted.
        """
        try:
            return int((self.root / DISK_USAGE).read_text())
        except (OSError, ValueError):
            return self._recount_disk_usage()

    def _recount_disk_usage(self):
        used = 0
        with os.scandir(self.blob_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    used += entry.stat().st_size
        for path in self.prepared_dir.glob("*.json"):      # live intents not written yet
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if record.get("durability") == "volatile" or record.get("expires_at", 0) <= time.time():
                continue
            for segment in record.get("segments", []):
                try:
                    if not self._path(segment["blob_id"]).exists():
                        used += HEADER.size + segment["length"]
                except (CorruptSnapshotError, KeyError, TypeError):
                    continue
        self._record_disk_usage(used)
        return used

    def _record_disk_usage(self, value):
        temporary = self.root / (DISK_USAGE + ".tmp")
        temporary.write_text(str(value))
        os.replace(temporary, self.root / DISK_USAGE)
