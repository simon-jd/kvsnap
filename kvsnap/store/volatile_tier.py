"""The store's side of host-memory (volatile) snapshots: owner locations,
commits and eviction. The pinned memory itself belongs to the worker.
"""

from __future__ import annotations

import json
import time

from .common import CorruptSnapshotError, GenerationConflict, StoreError, _json


class VolatileTier:
    """Manifests and accounting for snapshots whose blocks live in a worker's pinned memory."""

    def volatile_locations(self, manifest):
        """Opaque worker handles; no process-local pointer is exposed."""
        keys = sorted({segment["blob_id"] for segment in manifest["segments"]})
        with self._reader() as db:
            result = self._committed_volatile_locations(keys, db)
        if len(result) != len(keys):
            raise CorruptSnapshotError("volatile snapshot has no worker-owned allocation")
        return result

    def _committed_volatile_locations(self, keys, db=None):
        # One statement for any number of keys: a snapshot has one row per block.
        rows = (db or self.db).execute("SELECT blob_id,location_json FROM volatile_locations "
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

    def _commit_volatile_prepared(self, body, records, expected_generation):
        """Publish host-memory preparations: one per TP rank, in rank order."""
        from kvsnap.connector.volatile import VolatileClient
        for record in records:
            ident = record.get("preparation_id", "")
            if len(ident) != 32 or any(c not in "0123456789abcdef" for c in ident):
                raise ValueError("invalid preparation ID")
            if record.get("durability") != "volatile" or "locations" not in record:
                raise StoreError("every rank's preparation must be a host-memory one")
        with self._lock, self._blob_lock:
            self._check_owner()
            paths, segments, all_locations = [], [], {}
            for rank, record in enumerate(records):
                path = self.prepared_dir / (record["preparation_id"] + ".json")
                if json.loads(path.read_text()) != record or record["expires_at"] <= time.time():
                    raise StoreError("volatile preparation differs from intent or expired")
                if (body["durability"] != "volatile" or body["session_id"] != record["session_id"] or
                        body["generation"] != record["generation"]):
                    raise GenerationConflict("volatile preparation identity or generation mismatch")
                if {segment["blob_id"] for segment in record["segments"]} != set(record["locations"]):
                    raise CorruptSnapshotError("volatile location set differs from manifest")
                segments += [dict(segment, rank=rank) if rank else segment for segment in record["segments"]]
                # Equal blob ids hold equal bytes, whichever rank's owner keeps them.
                all_locations.update(record["locations"])
                paths.append(path)
            if expected_generation is not None and body["generation"] != expected_generation:
                raise GenerationConflict("manifest generation differs from expected generation")
            body["segments"], body["state"] = segments, "HostValid"
            # Validation checks that every rank's shard covers the KV blocks.
            expected = self._validate_manifest(body)
            if any(len(record["segments"]) != expected for record in records):
                raise CorruptSnapshotError("volatile preparation does not cover KV blocks")
            # The worker hashed each block in place right after its D2H fence
            # and wrote the digests into the intent. Here the owner only has to
            # confirm, in one round trip, that it still holds every handle at
            # the recorded length; no payload crosses the socket.
            for segment in body["segments"]:
                location = all_locations[segment["blob_id"]]
                if segment["offset"] != 0 or segment["length"] != location["length"]:
                    raise CorruptSnapshotError("volatile segment length differs from its allocation")
            owners = {}
            for key, location in all_locations.items():
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
                locations = dict(all_locations)
                locations.update(self._live_volatile_locations(list(locations)))
                segments = {segment["blob_id"]: segment for segment in body["segments"]}
                self.db.executemany("INSERT INTO volatile_locations VALUES(?,?) ON CONFLICT(blob_id) DO UPDATE SET location_json=excluded.location_json",
                                    [(key, _json(location)) for key, location in locations.items()])
                self.db.executemany("INSERT INTO blobs VALUES(?,?,?,1) ON CONFLICT(blob_id) DO UPDATE SET refcount=refcount+1",
                                    [(key, segments[key]["length"], segments[key]["crc32c"]) for key in locations])
                self.db.executemany("INSERT INTO snapshot_blobs VALUES(?,?)",
                                    [(body["snapshot_id"], key) for key in locations])
                self._hit("before_manifest_commit")
            self._count_host_commit((segments[key]["length"], None) for key in locations)
            # No directory fsync: if a crash brings the intent back, it is an
            # expired temporary root that collection reclaims, and the host
            # memory it described is gone with the process anyway.
            for path in paths:
                path.unlink(missing_ok=True)
            self._hit("after_manifest_commit")
            self._hit("before_gpu_release")
            return json.loads(serialized)

    def volatile_bytes(self):
        """Bytes of distinct blobs referenced by committed host-memory snapshots."""
        with self._reader() as db:
            return db.execute(
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
        # Usually nothing has to go. This runs on the scheduler thread at every
        # turn end, and measuring the tier walks every host block, so first try
        # an upper bound: the size last measured plus every host-memory commit
        # this store has made since. Invalidations elsewhere only make it looser.
        if self._host_bytes_bound is not None and self._host_bytes_bound + needed_bytes <= capacity_bytes:
            return evicted
        used = self._host_bytes_bound = self.volatile_bytes()
        if used + needed_bytes <= capacity_bytes:
            return evicted
        with self._lock:
            self._check_owner()
            while (used := self.volatile_bytes()) + needed_bytes > capacity_bytes:
                row = self.db.execute(
                    "SELECT s.snapshot_id FROM snapshots s LEFT JOIN snapshot_access a ON a.snapshot_id=s.snapshot_id "
                    "WHERE s.durability='volatile' AND s.state='HostValid' AND s.snapshot_id NOT IN "
                    "(SELECT snapshot_id FROM leases WHERE expires_at>?) ORDER BY COALESCE(a.at,0) LIMIT 1",
                    (time.time(),)).fetchone()
                if row is None:
                    break
                self._invalidate(row[0], "evicted from the host tier (least recently used)")
                evicted.append(row[0])
            self._host_bytes_bound = used
        return evicted

    def _count_host_commit(self, entries):
        """Raise the host tier's upper bound by a committed snapshot's blocks (lengths first)."""
        if self._host_bytes_bound is not None:
            self._host_bytes_bound += sum(length for length, _ in entries)
