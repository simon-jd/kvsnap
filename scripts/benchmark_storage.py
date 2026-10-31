#!/usr/bin/env python3
"""Measure bounded native file I/O and flush latency on the output filesystem.

Creates only its own temporary regular file. No mounts or global cache changes.
Page-cache advice is a hint: storage-read accounting is reported independently.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvsnap.store.native_io import get_io_backend


def accounting():
    try:
        return {key: int(value) for key, value in
                (line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def read_sample(backend, fd, payload, *, evict):
    hint = "not_requested"
    if evict:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(fd, 0, len(payload), os.POSIX_FADV_DONTNEED)
                hint = "issued"
            except OSError as exc:
                hint = f"failed_errno_{exc.errno}"
        else:
            hint = "unavailable"
    before = accounting()
    start = time.monotonic_ns()
    actual = backend.pread_all(fd, len(payload), 0)
    elapsed = time.monotonic_ns() - start
    after = accounting()
    if actual != payload:
        raise RuntimeError("native I/O read content differs from the written payload")
    physical = after.get("read_bytes", 0) - before["read_bytes"] if "read_bytes" in before and "read_bytes" in after else None
    return {"elapsed_ns": elapsed, "bytes_per_second": len(payload) * 1e9 / elapsed,
            "cache_discard_hint": hint, "storage_read_bytes": physical,
            "storage_read_accounting_covers_payload": physical is not None and physical >= len(payload),
            "checksum_verified": True}


def run(output, sizes, repeats):
    backend = get_io_backend("native")
    try:
        mount = subprocess.run(["findmnt", "-T", str(output), "-no", "SOURCE,FSTYPE,OPTIONS"],
                               capture_output=True, text=True, timeout=10)
        filesystem = {"returncode": mount.returncode, "stdout": mount.stdout, "stderr": mount.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        filesystem = {"error": str(exc)}
    samples, summaries = [], []
    for size in sizes:
        # Generate outside timing; incompressible bytes avoid a zero-data shortcut.
        payload = os.urandom(size)
        digest = hashlib.sha256(payload).hexdigest()
        for iteration in range(repeats):
            with tempfile.TemporaryFile(prefix="kvsnap-benchmark-", dir=output) as stream:
                fd = stream.fileno()
                start = time.monotonic_ns()
                backend.pwrite_all(fd, payload, 0)
                write_ns = time.monotonic_ns() - start
                start = time.monotonic_ns()
                os.fdatasync(fd)
                flush_ns = time.monotonic_ns() - start
                advised = read_sample(backend, fd, payload, evict=True)
                warm = read_sample(backend, fd, payload, evict=False)
                samples.append({"bytes": size, "iteration": iteration, "payload_sha256": digest,
                                "write_ns": write_ns, "fdatasync_ns": flush_ns,
                                "cache_discard_read": advised, "immediate_repeat_read": warm})
        group = [item for item in samples if item["bytes"] == size]
        summaries.append({"bytes": size, "samples": len(group),
                          "median_write_ns": statistics.median(x["write_ns"] for x in group),
                          "median_fdatasync_ns": statistics.median(x["fdatasync_ns"] for x in group),
                          "median_advised_read_bytes_per_second": statistics.median(
                              x["cache_discard_read"]["bytes_per_second"] for x in group),
                          "storage_read_accounting_covered_samples": sum(
                              x["cache_discard_read"]["storage_read_accounting_covers_payload"] for x in group)})
    return {"status": "measured", "evidence_tier": "real_native_file_io",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "kernel": platform.release(), "native_library": str(backend.lib._name),
            "native_library_sha256": hashlib.sha256(Path(backend.lib._name).read_bytes()).hexdigest(),
            "filesystem": filesystem, "samples": samples, "summary": summaries,
            "limitations": ["Host end-to-end native I/O timings include allocation, thread dispatch and copies.",
                            "DONTNEED is advice, not proof of eviction; read_bytes accounting is recorded separately.",
                            "Device/controller cache state is not controlled; this is not a raw-device peak benchmark.",
                            "fdatasync timing does not prove power-loss survival.",
                            "No concurrent GPU decoding, pinned bandwidth or model prefill is measured."],
            "sources": ["https://man7.org/linux/man-pages/man2/posix_fadvise.2.html",
                        "https://docs.kernel.org/filesystems/proc.html"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory on the filesystem to measure")
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or not args.sizes_mib or any(not 1 <= n <= 256 for n in args.sizes_mib):
        parser.error("use 1..20 repeats and 1..256 MiB payloads")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(output, sorted(set(n * 1024**2 for n in args.sizes_mib)), args.repeats)
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
