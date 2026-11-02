"""Constants, errors and JSON helpers shared by the store's modules."""

from __future__ import annotations

import copy
import json
import struct


HEADER = struct.Struct("<8sIQI8s")
MAGIC = b"KVSBLOB1"
DISK_USAGE = ".disk_usage"
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


def _dict(value):
    return copy.deepcopy(value.to_dict() if hasattr(value, "to_dict") else dict(value))


def _shallow(value):
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
