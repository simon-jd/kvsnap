"""Per-engine startup handshake for the worker's resolved KV identity."""
from __future__ import annotations
import asyncio
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from kvsnap.models import KVIdentity, canonical_json


def identity_path(root, runtime_id):
    if not isinstance(runtime_id, str) or not re.fullmatch(r"[0-9a-f]{32}", runtime_id):
        raise ValueError("runtime_id must be a fresh 128-bit hex ID")
    return Path(root) / "runtime" / (runtime_id + ".identity.json")


def publish_runtime_identity(root, runtime_id, identity):
    path = identity_path(root, runtime_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Validate first, including actual layout and resolved revision fields.
    identity = KVIdentity.from_dict(identity.to_dict() if hasattr(identity, "to_dict") else identity)
    body = {"schema_version": 1, "runtime_id": runtime_id,
            "worker_pid": os.getpid(), "identity": identity.to_dict()}
    fd, temporary = tempfile.mkstemp(prefix=".identity-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical_json(body) + "\n")
            stream.flush()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


async def wait_runtime_identity(root, runtime_id, *, timeout=60, expected=None):
    path = identity_path(root, runtime_id)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("identity handshake timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    while True:
        try:
            body = json.loads(path.read_text())
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("worker did not publish resolved KV identity")
            await asyncio.sleep(0.05)
            continue
        if body.get("schema_version") != 1 or body.get("runtime_id") != runtime_id:
            raise ValueError("runtime identity handshake does not match this engine")
        pid = body.get("worker_pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError("invalid identity publisher")
        try:
            os.kill(pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("identity publisher is no longer alive") from exc
        identity = KVIdentity.from_dict(body["identity"])
        if expected is not None:
            expected.require_match(identity)
        return identity
