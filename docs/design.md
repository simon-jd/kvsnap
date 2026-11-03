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
  23 GB/s from pinned memory here;
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
3. When vLLM binds the step's metadata, before the forward, the worker
   records an event on the engine's stream. `start_load_kv`, which vLLM
   calls after the forward when every load is asynchronous, hands the load
   to the loader thread. The loader:
   1. **Disk snapshot:** `pread`s each block into its slot of one pinned
      staging buffer, with 8 reads in flight, and checks its CRC32C there.
      **Host-memory snapshot:** borrows the worker's own pinned blocks.
   2. Waits for the recorded event. A block freed this step by another
      request may still be read by kernels queued before the forward; the
      forward itself never touches a parked request's blocks.
   3. Launches the restore kernel once per 64 MiB of blocks, recording an
      event after each launch ([data plane](#data-plane)), then waits for
      the last one.
4. `get_finished` reports the request in `finished_recving` once the copy is
   complete, or earlier, once every launch is queued and at most 128 MiB
   remain (`EARLY_REPORT_BYTES`). vLLM polls once a step, right after
   launching the forward, so a copy that ended just after a poll used to
   wait a whole step. An early-reported load's last event goes to the next
   forward, which waits for it on the GPU before any kernel can read the
   blocks; at the kernel's 22 GB/s, 128 MiB is about 6 ms, less than the
   step still running. The lease is released only when the copy is
   complete, and a load that fails after it was reported stops the engine.
   vLLM schedules the request from the next step and prefills only the
   suffix.

**Save**

1. `request_finished` (scheduler) builds the manifest for the settled turn:
   tokens, KV boundary, sampling state and outputs. It returns `True` so vLLM
   keeps the blocks. For a host-memory save it first evicts least recently
   used host snapshots until the new one fits. This runs on the scheduler
   thread, inside the engine step; an eviction releases only the evicted
   snapshot's blob references, without an fsync. Saving only happens at a
   turn boundary.
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
   store's writer lock throughout. The scheduler's reads during a step (the
   manifest lookup in `get_num_new_matched_tokens`, the host-tier size check
   in `request_finished`) go through their own connection and do not wait
   for it; its writes do: the lease in `update_state_after_alloc`, and an
   eviction when the host tier is full.

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
  - blob reference counts (one per snapshot that holds the blob) and leases;
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
takes `.writer.lock` and `BEGIN IMMEDIATE`. Each thread reads through its
own query-only connection, and in WAL mode each statement sees one committed
state, so a read never waits for a writer in this process or another. (A
thread that holds the writer lock reads through the writer's connection and
sees its own transaction.) That leaves one visible intermediate state, a
durable snapshot between its `Writing` insert and its publish, which readers
treat as not there yet.

**Costs that do not grow with the store.** Invalidating or deleting a
snapshot decrements the reference counts of its own blobs, found through an
index on `snapshot_blobs(blob_id)`; host-memory invalidations skip the fsync,
since recovery invalidates every host snapshot anyway. Disk admission keeps
a running total of blob bytes in `.disk_usage`: a save reserves its new
blobs' bytes under the blob lock, and garbage collection and recovery
recount the directory. `bench/store_stress.py` times each operation with
500 sessions in the store (`results/store-stress.json`).

`recover()` runs at startup:
- it invalidates every snapshot that is not in a committed state;
- it re-validates the payloads of committed ones;
- it rebuilds reference counts from the committed manifests, as garbage
  collection also does.

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
to check them (`results/h2d-patterns.json`, median of 3 rounds):

| Pattern | Consecutive block ids | Shuffled block ids |
|---|---|---|
| One contiguous DMA copy (peak for the copy engine) | 19.5 GB/s | |
| One `cudaMemcpy2DAsync` per (layer, run of consecutive ids) | 18.2 GB/s | 13.1 GB/s |
| One copy per (layer, block) from per-block allocations | 13.8 GB/s | 13.8 GB/s |
| Contiguous DMA in 64 MiB chunks + GPU scatter (the previous load path) | 17.5 GB/s | 17.5 GB/s |
| Per-block allocations, one 2.25 MiB DMA each + GPU scatter | 17.3 GB/s | 17.2 GB/s |
| **Restore kernel reading pinned memory, staging buffer** | **22.9 GB/s** | **22.9 GB/s** |
| **Restore kernel reading pinned memory, per-block allocations** | **22.9 GB/s** | **22.9 GB/s** |

The copy engine is efficient only for large copies. vLLM's free list
scatters block ids in a long-running engine, and the host tier keeps one
allocation per block so blocks can be deduplicated and freed one at a time.
Both force small copies, and on this machine even one large DMA copy
reaches only 19.5 GB/s.

So a load does not use the copy engine. The restore kernel
(`kvsnap/connector/copy_kernel.py`, Triton) takes each packed block's host
address and walks its (layer, block) spans, reading each from pinned memory
over PCIe and writing it straight into its place in the KV cache: base and
block pitch per layer, through an int64 view of the layer's storage. Pinned
memory from `cudaHostAlloc` is mapped into the device's address space under
unified addressing, so the kernel takes host pointers as they are. Nothing
is staged on the device and nothing is scattered afterwards, and block order
does not matter. Every source is bounds-checked against its allocation, under
the pool lock, before the launch. The kernel runs on 12 programs, about a
tenth of the RTX 4090's 128 SMs, so most of the GPU stays with the decoding
batch; beside 16 decode streams it sustains 21 GB/s
(`results/resume-phases.json`). It compiles when the KV caches are
registered.

A save still uses the copy engine: `index_select` per layer into device
staging, then large D2H copies from it.

The C++ pool (`csrc/dataplane/cuda_pool.cpp`) owns a `cudaHostAlloc` arena, a
non-blocking copy stream and an event per allocation:

* Releasing an allocation blocks on its event, so pinned memory is never
  reused while a copy into or out of it is in flight. The restore kernel
  runs on the same stream but outside the pool's events: a load waits for
  its last launch before its sources can be released, and a launch that
  fails drains the copy stream first.
* If the device cannot prove completion, the pool keeps the memory instead
  of freeing it.
* A reservation that does not fit raises a distinct `PinnedCapacityError`
  before anything is copied.

`tests/native` compiles the pool against a fake CUDA runtime and injects
event-record, wait and close failures. `tests/connector/test_scatter_gather.py`
checks the restore kernel's address tables on CPU tensors laid out like
Qwen3-8B, and `tests/connector/test_copy_kernel_cuda.py` runs the kernel on
the GPU and compares every byte.

## Known gaps

* **Saves still cost the batch a little.** In a trace of this code under
  load (`results/turn-end-trace.json`), 6 of 54 turn ends had engine steps
  of 22–44 ms instead of about 7 ms. The save thread's gather and D2H
  share the GPU with the decode kernels, and its hashing and bookkeeping
  share the GIL with the engine thread. A native gather that releases the
  GIL, and hashing only the blocks a turn changed, are next.
* **Some scheduler-side work is in the engine step.** Matching, leasing and
  host-tier eviction run on the scheduler thread. Reads no longer wait for
  the commit thread, but the lease and an eviction are writes, and the
  commit thread holds the writer lock for the whole of `commit_prepared`,
  owner round trips and blob-header checks included, not only for its
  transactions. An eviction also measures the host tier, which walks every
  host block's reference: 0.7 s with 128k host blocks in the store stress
  test, though pinned memory keeps a real tier to a few thousand blocks.
* **Garbage collection and recovery still scale with the store.** Both
  rebuild every reference count from the committed manifests and recount
  the blob directory. Recovery runs when a store opens; `collect_garbage()`
  runs only when called, and nothing on the serving path calls it, so
  unreferenced blob files stay on disk until something does (the benchmarks
  call it between sessions).
* **Admission and polling.** Under load, the request still spends about
  15 ms between the adapter and the scheduler's match (input processing,
  IPC, the wait for the running step, the manifest lookup) and 5 ms in the
  lease, and a reported load waits for the rest of the step that was running
  when vLLM polled, 7 ms (`results/resume-phases.json`). The gateway no
  longer reads the manifest back (it keeps the one pause committed), and a
  load is reported while its last 128 MiB is in flight.
* **Slower than LMCache from host memory.** Restoring an evicted 8k-token
  Qwen3-8B session took 109 ms from kvsnap's host tier against 83 ms with
  LMCache's CPU backend, and under load kvsnap's async load reached the
  first token 39 ms after LMCache (README). LMCache loads inside the step
  that schedules the request, so it parks nothing and polls nothing, and
  the running requests wait for its copy: their p99 decode gap was
  57 – 60 ms against kvsnap's 23 – 26 ms.
* **Read and copy are not pipelined.** A disk restore reads every block
  before the first H2D. Chunked overlap would hide the shorter of the two.
* **Synchronous scheduling only.** `request_finished` needs settled tokens;
  vLLM's async scheduler would need an extra step of bookkeeping.
* **Exactness needs batch-invariant kernels.** With vLLM's default kernels,
  prefilling a 64-token suffix and prefilling the whole prompt are different
  floating-point computations. Logprobs then differ from recompute by up to
  3.5 nats, and greedy tokens diverged in 37 of 405 restores. vLLM's own
  prefix cache is not bit-exact without batch-invariant kernels either, for
  the same reason.
