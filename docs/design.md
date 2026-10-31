# Design

## Problem

In a multi-turn conversation, turn *n+1* repeats the whole conversation so far
as its prompt. If the engine still holds those tokens' KV cache (vLLM's
prefix cache), turn *n+1* only computes the new tokens. But the cache is
shared by every session on the GPU, so under load an idle session's blocks
are evicted within seconds. Its next turn then re-prefills everything. That
costs O(context) compute, and attention makes it worse than linear.

The alternative is to copy a finished turn's KV out of the GPU, then copy it
back for the next turn. Whether that wins is a question of bytes versus
FLOPs:

* restoring costs about `kv_bytes_per_token × L / bandwidth`. That is
  112 KiB per token for Qwen3-1.7B and 144 KiB for Qwen3-8B, against about
  17 GB/s of pinned H2D here;
* recomputing costs a prefill of L tokens, which scales with model size and,
  through attention, with L².

Bigger models and longer contexts favour restoring. The README measures where
the crossover lands on one RTX 4090, against a recompute with vLLM's prefix
caching off, that is, a session whose blocks were evicted.

## Components

| Component | Process | Role |
|---|---|---|
| Gateway (`kvsnap/gateway`) | API server | Session state machine: create → turn → pause → resume / fork → close. Owns the session generation fence. |
| Connector, scheduler side (`kvsnap/connector/vllm.py`) | vLLM EngineCore | Decides which blocks to save or load. Owns the SQLite store handle. Publishes manifests. Evicts the host tier. |
| Connector, worker side (same file) | vLLM worker (same process with the uni executor) | Save and load jobs, blob reads and writes, the host-memory tier. |
| Store (`kvsnap/store`) | both | SQLite WAL metadata, content-addressed blob files, leases, GC, recovery. |
| Data plane (`kvsnap/connector/dataplane.py`, `csrc/dataplane`) | worker | Pinned arena, copy stream, GPU gather/scatter, layout checks. |

The connector implements vLLM 0.31.0's `KVConnectorBase_V1`. vLLM is not
modified.

## Connector protocol

The gateway attaches an instruction to every request it submits, under
`kv_transfer_params["kvsnap"]`. It carries the snapshot to resume from (if
any), the session, the generation, the KV identity and the durability level
to save at.

**Restore**

1. `get_num_new_matched_tokens` (scheduler) checks that the snapshot exists
   and that its KV identity matches this engine: model, revision, dtype and
   block size. It reports the snapshot's KV tokens as an external hit and asks
   for an **async load**. vLLM then allocates blocks, parks the request in
   `WAITING_FOR_REMOTE_KVS` and keeps scheduling the rest of the batch.
2. `update_state_after_alloc` (scheduler):
   - checks the allocated block ids are complete, distinct and not the null
     block;
   - takes a **lease** on the snapshot's blobs, which keeps GC and host-tier
     eviction away;
   - queues a load for the worker.
3. `start_load_kv` (worker) records an event on the engine's stream and
   hands the load to the loader thread. The loader:
   1. **Disk snapshot:** `pread`s each block into its slot of one pinned
      staging buffer, with 8 reads in flight, and checks its CRC32C there.
      **Host-memory snapshot:** borrows the worker's own pinned blocks.
   2. Waits for the recorded event. A block freed this step by another
      request may still be read by kernels that are already queued.
   3. Copies the blocks to the GPU and scatters them into their slots
      ([data plane](#data-plane)), then waits for the copy stream.
4. `get_finished` reports the request in `finished_recving`. vLLM schedules
   it from the next step and prefills only the suffix.

**Save**

1. `request_finished` (scheduler) builds the manifest for the settled turn:
   tokens, KV boundary, sampling state and outputs. It returns `True` so vLLM
   keeps the blocks. For a host-memory save it first evicts least recently
   used host snapshots until the new one fits. This runs on the scheduler
   thread, inside the engine step, and each eviction is a full invalidation
   (see [Known gaps](#known-gaps)). Saving only happens at a turn boundary.
2. `wait_for_save` (worker) records an event on the engine's stream and
   hands the save to a job on the save executor. The job:
   1. reserves pinned memory;
   2. waits for the event;
   3. gathers the blocks on the GPU and copies them to pinned memory;
   4. marks the save **fenced**. From then on `get_finished` reports the
      request in `finished_sending`, and vLLM frees the GPU blocks.
   5. Then, still on the save job:
      - **disk:** hash each block (SHA-256 content id, CRC32C), deduplicate,
        write and fdatasync blobs in parallel, fsync the directory once;
      - **host memory:** hash the pinned blocks in place and adopt them into
        the host tier;
      - either way, record a preparation intent.
3. `update_connector_output` (scheduler) hands the preparation to the
   **commit thread**, which publishes the manifest in one SQLite transaction.
   It checks blob headers (disk) or asks each owner, in one round trip,
   whether it still holds the new handles and the blocks already committed
   (host). It never reads the payloads. The request stays pending push work
   until its commit finishes, so vLLM keeps stepping, and an unexpected
   commit failure stops the engine at the next step. The commit holds the
   store's lock throughout, and the scheduler's own store calls during a
   step wait for it: the manifest lookup in `get_num_new_matched_tokens`,
   the lease in `update_state_after_alloc` and host-tier eviction in
   `request_finished`.

**Failure policy.** The rule is whether a DMA was issued.
- **Before any copy:**
  - a mismatched identity, an evicted or invalidated snapshot, a lease that
    can no longer be taken, a short read or a bad CRC is a **miss**;
  - the blocks are reported through `get_block_ids_with_load_errors`, and
    the engine runs with `kv_load_failure_policy="recompute"`, so vLLM
    recomputes them;
  - a corrupt snapshot is invalidated, and a request whose load failed
    misses when vLLM reschedules it.
- **A save that fails before any copy** (pinned pool full, layout mismatch)
  or while it is prepared or committed is recorded in `failed_saves`.
  - The GPU blocks are freed.
  - The gateway's `pause` raises at once instead of timing out.
- **A copy that was issued but whose completion cannot be proven** is the
  one fail-stop case. The connector raises, and keeps the pinned memory,
  because the DMA engine could still write it.

## Store

**Layout**
- `manifests.sqlite3` (WAL mode) holds:
  - sessions and their generations;
  - snapshots and their states;
  - blob reference counts and leases;
  - closed sessions;
  - failed saves;
  - snapshot access times (host-tier LRU order).
- `blobs/<sha256>.blob` holds one KV block, behind a header with a magic
  number, version, length and CRC32C.
- `prepared/<id>.json` holds worker intents not yet published.

**Durability levels**
- `volatile`: pinned host memory owned by the worker, lost when the process
  exits. This tier is a cache: when a save does not fit, the least recently
  committed or restored unleased snapshots are invalidated first, and their
  sessions recompute on the next turn.
- `process`: blob files, no fsync.
- `durable`: blob files with fdatasync, a directory fsync, and
  `synchronous=FULL` on the commit.

**Commit order (durable)**
1. Publish the intent under the blob lock and fsync it. The intent is a GC
   root for its blobs, including their temporary files.
2. Without the lock, write the blobs to temporary files, fdatasync them and
   rename them.
3. fsync the directory.
4. Insert the snapshot as `Writing`.
5. Publish it as `Durable` in one transaction.

The blob lock is exclusive in step 1, on the worker, and in steps 4 and 5,
which `commit_prepared` runs on the scheduler side. It holds the lock through
both `synchronous=FULL` transactions and the removal and directory fsync of
the intent. Disk restores hold the lock shared while they read. So a restore
never waits for a save's blob writes and fdatasyncs (steps 2 and 3), but it
can wait behind a durable commit's SQLite transactions. Host-memory commits
take the lock too; host-memory restores do not need it.

**Readers do not take the writer lock.** Every write, from any process,
takes `.writer.lock` and `BEGIN IMMEDIATE`. A read takes only its own
connection's lock: in WAL mode each statement sees one committed state.
That leaves one visible intermediate state, a durable snapshot between its
`Writing` insert and its publish, which readers treat as not there yet.

`recover()` runs at startup:
- it invalidates every snapshot that is not in a committed state;
- it re-validates the payloads of committed ones;
- it rebuilds reference counts.

`tests/crash` SIGKILLs a writer at eight points of `SnapshotStore.commit` and
checks that recovery keeps exactly the committed snapshots. The connector's
prepare → `commit_prepared` path shares the blob writer and the publish
transaction, but is not killed at its own points.

**Fencing.** Every turn advances the session generation in SQLite
(`claim_generation`). A manifest is published only if its generation is
still current, so a late save from an aborted or superseded turn cannot
replace a newer snapshot. The claim is fsync'd unless the turn saves to host
memory: that snapshot cannot outlive the process, and a process crash keeps
the claim, which is already in the WAL.

**Content addressing.** Blob ids are the SHA-256 of the block bytes, so:
- a fork shares every block with its parent;
- turn *n+1* shares the unchanged prefix blocks of turn *n*;
- GC deletes a blob only when no snapshot, lease or live intent references
  it.

**Leases** protect blobs while a worker reads them. They are not fsynced. A
lease dies with the process that holds it, and a restarted store expires
leases by TTL.

## Data plane

vLLM stores KV per layer as `[num_blocks, ...]`, so block *b* of layer *l*
occupies `layer_bytes` at `base_l + b × pitch_l`. A snapshot packs one
block's layers back to back (`block_bytes = Σ layer_bytes`, 2.25 MiB for
Qwen3-8B). Moving N blocks therefore touches `layers × N` separate 64 KiB
regions on the GPU.

`bench/h2d_patterns.py` measures each way of copying 512 such blocks
(1.2 GB, shaped like Qwen3-8B) through this data plane, and reads them back
to check them (`results/h2d-patterns.json`):

| Pattern | Consecutive block ids | Shuffled block ids |
|---|---|---|
| One contiguous copy (peak) | 19.4 GB/s | |
| One `cudaMemcpy2DAsync` per (layer, run of consecutive ids) | 17.9 GB/s | 13.0 GB/s |
| One copy per (layer, block) from per-block allocations | 13.8 GB/s | 13.8 GB/s |
| **Contiguous H2D in 64 MiB chunks + GPU scatter** | **17.3 GB/s** | **17.3 GB/s** |
| Per-block allocations, one 2.25 MiB copy each + GPU scatter | 17.1 GB/s | 17.1 GB/s |

The copy engine is efficient only for large copies. vLLM's free list
scatters block ids in a long-running engine, and the host tier keeps one
allocation per block so blocks can be deduplicated and freed one at a time.
Both force small copies.

So the data plane never asks the copy engine for small copies. A load copies
a chunk of packed blocks into a device staging buffer, in one copy (staging
buffer) or one per block (host tier). It then writes each layer's slice into
the KV cache with one `index_copy_`, on the same copy stream, through an
int64 view of that layer's storage. A save is the mirror image:
`index_select` per layer into staging, then large D2H copies. The scatter
costs an extra pass over device memory. That is why it trails the 2D copy on
consecutive ids (17.3 vs 17.9 GB/s), but the scatter does not depend on
block order.

The C++ pool (`csrc/dataplane/cuda_pool.cpp`) owns a `cudaHostAlloc` arena, a
non-blocking copy stream and an event per allocation:

* Releasing an allocation blocks on its event, so pinned memory is never
  reused while a copy into or out of it is in flight.
* If the device cannot prove completion, the pool keeps the memory instead
  of freeing it.
* A reservation that does not fit raises a distinct `PinnedCapacityError`
  before anything is copied.

`tests/native` compiles the pool against a fake CUDA runtime and injects
event-record, wait and close failures.

## Known gaps

* **Saves still cost the batch a little.** In the last two traced runs
  (`results/turn-end-traces.json`), up to three engine steps at a turn end
  took 20–27 ms instead of about 8–9 ms. The save thread's gather and D2H
  share the GPU with the decode kernels, and its hashing and bookkeeping
  share the GIL with the engine thread. A native gather that releases the
  GIL, and hashing only the blocks a turn changed, are next.
* **Some scheduler-side work is in the engine step, and some store paths
  grow with the store.** Matching, leasing and host-tier eviction run on the
  scheduler thread and can wait for an in-flight commit (see Save, step 3).
  Every invalidation, an eviction included, is a `synchronous=FULL`
  transaction that rebuilds every blob reference count: with six 8k-token
  sessions stored, the under-load benchmark's own invalidations stalled its
  event loop for 212–297 ms. A disk save's capacity check stats every blob
  file under the exclusive blob lock.
* **Admission and reporting.** Admission takes 3–15 ms per restore: the
  generation claim, a manifest lookup in the gateway and in the scheduler,
  and the lease. The async load then costs one engine step to be reported,
  about 20 ms under load, which is now the larger fixed cost.
* **Read and copy are not pipelined.** A disk restore reads every block
  before the first H2D. Chunked overlap would hide the shorter of the two.
* **Synchronous scheduling only.** `request_finished` needs settled tokens;
  vLLM's async scheduler would need an extra step of bookkeeping.
* **Exactness needs batch-invariant kernels.** With vLLM's default kernels,
  prefilling a 64-token suffix and prefilling the whole prompt are different
  floating-point computations. Logprobs then differ from recompute by up to
  a few nats, and greedy tokens sometimes diverge. vLLM's own prefix cache
  is not bit-exact without batch-invariant kernels either, for the same
  reason.
