"""Per-operation cost of the snapshot store with many sessions (CPU only).

    python -m bench.store_stress --label after --output results/store-stress.json

Fills a store with ``--sessions`` sessions, each holding one committed
snapshot of ``--blocks`` blocks (half in host memory, half on disk at the
``process`` level), then times the operations the engine and the gateway
issue on a turn:

  commit_durable / commit_process   a new snapshot of ``--blocks`` fresh blocks on
                                    the connector's path: the worker's BlobWriter
                                    prepares it (admission, writes) and the
                                    scheduler's commit_prepared publishes it
  invalidate_host / invalidate_disk one snapshot, as eviction and failed loads do
  make_room_fits                    make_room_volatile when the tier has room, as at
                                    every host-memory turn end
  evict_one                         make_room_volatile forcing one LRU eviction
  get_manifest                      the scheduler's lookup when a resume arrives
  lease                             acquire_lease + release_lease
  get_manifest_during_commits       the same lookup while another thread commits
                                    durable snapshots, as the connector's commit
                                    thread does beside the scheduler

Payloads are ``--block-bytes`` bytes: the store's metadata work does not
depend on block size, only the writes do. ``--label`` names this measurement
in ``--output``, which collects several (for example the same tree before and
after a change, run back to back); ``kvsnap`` is imported from PYTHONPATH, so
an older tree can be measured with this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import threading
import time

from bench.environment import ROOT, environment

BLOCK_TOKENS = 16


def identity(blocks):
    from kvsnap.models import KVIdentity
    return KVIdentity("stress", "rev", "tokenizer-rev", "bf16", BLOCK_TOKENS, "stress-layout",
                      {"layers": 1, "kv_heads": 1, "head_dim": 1},
                      {"max_model_len": blocks * BLOCK_TOKENS + 64, "rope_theta": 10000})


def snapshot(name, blocks, durability, kv):
    from kvsnap.models import SamplingConfig, SnapshotManifest
    tokens = tuple(range(blocks * BLOCK_TOKENS + 1))
    return SnapshotManifest(name, name, 0, kv, tokens, blocks * BLOCK_TOKENS, tokens[-1:], 42, SamplingConfig(),
                            8, 0, durability=durability).to_dict()


def payloads(name, blocks, size):
    return [f"{name}:{i}:".encode().ljust(size, b".") for i in range(blocks)]


def describe(seconds):
    ms = sorted(s * 1e3 for s in seconds)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 3),
            "p90_ms": round(ms[min(len(ms) - 1, int(0.9 * len(ms)))], 3), "max_ms": round(ms[-1], 3)}


def timed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def run(args) -> dict:
    import kvsnap
    from kvsnap.store import BlobWriter, SnapshotStore
    root = Path(tempfile.mkdtemp(prefix="kvsnap-stress-", dir=args.store_dir))
    kv = identity(args.blocks)
    store = SnapshotStore(root, host_capacity_bytes=1 << 40, disk_capacity_bytes=1 << 40)
    fill_started = time.perf_counter()
    names = []
    for index in range(args.sessions):
        name = f"s{index:04d}"
        durability = "volatile" if index % 2 == 0 else "process"
        store.commit(snapshot(name, args.blocks, durability, kv), payloads(name, args.blocks, args.block_bytes))
        names.append((name, durability))
    fill_s = time.perf_counter() - fill_started
    result = {}
    fresh = iter(range(10**6))

    writer = BlobWriter(root, disk_capacity_bytes=1 << 40)

    def commit(durability):
        name = f"n{next(fresh):06d}"
        prepared = writer.prepare_blocks(payloads(name, args.blocks, args.block_bytes), durability)
        store.commit_prepared(snapshot(name, args.blocks, durability, kv), prepared)

    result["commit_durable"] = describe([timed(commit, "durable") for _ in range(args.repeat)])
    result["commit_process"] = describe([timed(commit, "process") for _ in range(args.repeat)])
    hosts = [name for name, durability in names if durability == "volatile"]
    disks = [name for name, durability in names if durability == "process"]
    result["get_manifest"] = describe([timed(store.get_manifest, hosts[i]) for i in range(args.repeat)])

    def lease(name):
        store.release_lease(store.acquire_lease(name))

    result["lease"] = describe([timed(lease, hosts[i]) for i in range(args.repeat)])
    result["invalidate_host"] = describe([timed(store.invalidate_snapshot, hosts[i], "stress")
                                          for i in range(args.repeat)])
    result["invalidate_disk"] = describe([timed(store.invalidate_snapshot, disks[i], "stress")
                                          for i in range(args.repeat)])
    block = args.block_bytes
    result["make_room_fits"] = describe([timed(store.make_room_volatile, block, 1 << 40)
                                         for _ in range(args.repeat)])
    evictions = []
    for _ in range(args.repeat):
        full = store.volatile_bytes()            # a tier exactly full: the next block forces one eviction
        evictions.append(timed(store.make_room_volatile, block, full))
    result["evict_one"] = describe(evictions)

    stop, committed = threading.Event(), []

    def committer():
        while not stop.is_set():
            commit("durable")
            committed.append(1)

    thread = threading.Thread(target=committer)
    thread.start()
    try:
        lookups = []
        targets = hosts[args.repeat * 2:]
        deadline = time.perf_counter() + args.contention_s
        index = 0
        while time.perf_counter() < deadline:
            lookups.append(timed(store.get_manifest, targets[index % len(targets)]))
            index += 1
            time.sleep(0.002)
    finally:
        stop.set()
        thread.join()
    result["get_manifest_during_commits"] = {**describe(lookups), "commits_meanwhile": len(committed)}
    store.close()
    for path in sorted(root.rglob("*"), reverse=True):
        path.unlink() if path.is_file() or path.is_socket() else path.rmdir()
    root.rmdir()
    package = Path(kvsnap.__file__).resolve().parent
    if package.is_relative_to(ROOT):
        package = package.relative_to(ROOT)
    return {"kvsnap": str(package), "code_commit": args.code_commit,
            "sessions": args.sessions, "blocks": args.blocks, "block_bytes": args.block_bytes,
            "repeat": args.repeat, "fill_s": round(fill_s, 2), "operations": result,
            "environment": environment(args.store_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--code-commit", default=None, help="the commit of the kvsnap tree on PYTHONPATH")
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--blocks", type=int, default=512, help="blocks per snapshot (8k tokens at 16 per block)")
    parser.add_argument("--block-bytes", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--contention-s", type=float, default=10.0)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measured = run(args)
    existing = json.loads(args.output.read_text()) if args.output.exists() else {"measurements": {}}
    existing["measurements"][args.label] = measured
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(existing, indent=1) + "\n")
    print(json.dumps(measured["operations"], indent=1))


if __name__ == "__main__":
    main()
