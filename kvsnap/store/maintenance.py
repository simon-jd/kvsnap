"""Invalidation, recovery after a crash, fork, deletion, garbage collection and export."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .common import COMMITTED, CorruptSnapshotError, GenerationConflict, StoreError, _json


class Maintenance:
    """Whole-store operations: invalidation, recovery, fork, deletion, GC, export and stats."""

    def invalidate_snapshot(self, snapshot_id, reason):
        with self._lock:
            self._check_owner()
            # A load lease can outlive an explicitly deleted manifest.
            if not self.db.execute("SELECT 1 FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone():
                return
            self._invalidate(snapshot_id, reason)

    def _invalidate(self, snapshot_id, reason):
        row = self.db.execute("SELECT durability FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        # A host-memory snapshot cannot outlive the process, and recovery
        # invalidates every one of them, so its invalidation needs no fsync.
        with self._transaction("volatile" if row and row[0] == "volatile" else "durable"):
            body = self._get_manifest(snapshot_id, True)
            body["state"] = "Invalid"
            self.db.execute("UPDATE snapshots SET state='Invalid',manifest_json=?,error=? WHERE snapshot_id=?",
                            (_json(body), reason, snapshot_id))
            self._release_references(snapshot_id)

    def _release_references(self, snapshot_id):
        """Drop one snapshot's blob references: O(its blocks), not O(every snapshot).

        Every reference count equals the number of ``snapshot_blobs`` rows for
        its blob; ``_rebuild_references`` restores that from the manifests at
        recovery and in garbage collection.
        """
        self.db.execute("UPDATE blobs SET refcount=MAX(refcount-1,0) WHERE blob_id IN "
                        "(SELECT blob_id FROM snapshot_blobs WHERE snapshot_id=?)", (snapshot_id,))
        self.db.execute("DELETE FROM snapshot_blobs WHERE snapshot_id=?", (snapshot_id,))

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
            self._recount_disk_usage()
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
            self._release_references(snapshot_id)
            self.db.execute("DELETE FROM snapshots WHERE snapshot_id=?", (snapshot_id,))

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
            self._recount_disk_usage()
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
