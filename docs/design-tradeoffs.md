# Design trade-offs

Most of kvsnap's behaviour follows from a handful of choices: what a
snapshot is, when it is taken, who publishes it, and which thread pays for
each step. This file takes them one at a time, with the options that were
turned down and the price of the one that was kept. Measured figures come
from `results/*.json` (described in `results/README.md`). The traced ones
come from runs of `bench/restore_under_load.py --trace-dir`, summarized by
`bench/trace_summary.py`: engine steps over 20 ms, the client's event-loop
stalls and garbage-collection pauses, and the duration of every save phase.
`results/turn-end-trace.json` traces this code, and
`results/turn-end-traces.json` an earlier design and the changes since.

## Named session snapshots, not a prefix-hash cache

**Decision.** At the end of each turn the session's whole KV prefix is saved
as a named snapshot with a generation number, and the next turn resumes from
that snapshot by id.

**Alternatives.** vLLM's prefix cache extended to CPU or disk (the
`OffloadingConnector`, LMCache), which find blocks by hashing token prefixes.

**Why.** A session has to outlive eviction and engine restarts, be forked,
and fail loudly. A named snapshot restores exactly what was saved or misses
explicitly; a prefix-hash hit depends on which blocks survived an LRU that
every session shares. The generation number fences a stale gateway or a late
save from a superseded turn, and a failed save fails the session's `pause`
instead of silently costing a recompute later.

**Cost.** Every turn saves the whole prefix again. Unchanged blocks are
deduplicated by content hash, so they are not rewritten, but they are copied
and hashed again. Clients must use the gateway's turn, pause and resume API.
And a named snapshot is not the fastest way back from host memory: with
four 8k-token Qwen3-8B sessions rotating through GPU memory, LMCache's CPU
backend restored a session in 83 ms, kvsnap's host tier in 109 ms and
vLLM's own CPU offloading in 136 ms (README).

## Save only at turn boundaries, from settled scheduler state

**Decision.** `request_finished` builds the manifest from the request's
tokens and computed-token count, so the connector requires synchronous
scheduling.

**Alternatives.** Snapshots in the middle of a generation; vLLM's async
scheduler.

**Why.** At `request_finished` every token is settled and the KV boundary is
exact. With async scheduling, output placeholders are still in flight when a
request finishes, so the boundary would have to be reconstructed one step
later.

**Cost.** No async scheduling, and a session can only pause between turns.

## Free GPU blocks at the D2H fence, hash and publish afterwards

**Decision.** A save job gathers the blocks on the GPU and copies them to
pinned memory. As soon as that copy is fenced, vLLM may free the GPU blocks.
Hashing, deduplication, writes and fdatasync follow on the same job, and the
manifest is published last.

**Alternatives.** Keep the GPU blocks until the snapshot is durable; save
synchronously inside the engine step.

**Why.** GPU blocks are the scarce resource. On Qwen3-1.7B at 8k tokens the
D2H takes 66 ms, while preparing the snapshot (SHA-256 and CRC32C of
940 MB, dedup, the intent file) takes 127 ms more (medians in
`results/turn-end-trace.json`); holding the blocks through all of it would
shrink the batch for every other request.

**Cost.** A two-phase protocol (prepare, then commit) and a pinned staging
budget. A save can fail after its blocks are gone; it is then recorded, and
the next turn recomputes.

## Async load, with the request parked

**Decision.** A restore is reported to vLLM as an external hit loaded
asynchronously: vLLM parks the request in `WAITING_FOR_REMOTE_KVS` and keeps
running the batch while a loader thread copies the blocks.

**Alternatives.** Load inside `start_load_kv`, before the forward pass, as
LMCache does.

**Why.** A synchronous load stalls every running request for the copy: a
p99 decode gap of 78 – 129 ms during resumes, against 23 – 26 ms for the
async one (README).

**Cost.** vLLM polls for finished loads once a step, so the restored request
waits for the step that is running when its load is reported: under load its
TTFT median is 11 ms above the synchronous load's (README). Reporting a load
while its last 128 MiB is still in flight, with the next forward waiting for
it on the GPU, removed part of that wait; the rest is the step itself.

## A restore kernel that reads pinned memory, never the copy engine

**Decision.** A load is one Triton kernel per 64 MiB of blocks that reads
each (layer, block) span of a packed block from pinned host memory and
writes it into place in the KV cache. A save still gathers on the GPU into
device staging (`index_select`) and copies out with large D2H copies.

**Alternatives.** One copy per (layer, block); one `cudaMemcpy2DAsync` per
(layer, run of consecutive block ids); large DMA copies into device staging
and a GPU `index_copy_` scatter, the previous load path.

**Why.** The copy engine is fast only for large copies, vLLM's free list
scatters block ids, and on this machine even one contiguous DMA copy reaches
19.5 GB/s. On 1.2 GB shaped like Qwen3-8B the kernel reaches 22.9 GB/s
whatever the block order, against 17.5 GB/s for chunks plus a scatter,
13.8 GB/s for per-(layer, block) copies and 13.1 GB/s for 2D copies on
shuffled ids (`results/h2d-patterns.json`).

**Cost.** The kernel runs on 12 of the GPU's SMs while the batch decodes:
during resumes the median decode gap is up to 0.8 ms above its baseline
(README). It needs Triton and pinned memory the device can address, and it
runs outside the pinned pool's per-allocation events, so a load waits for
its last launch before its sources may be released.

## Content-addressed, checksummed blocks

**Decision.** Each block is stored once under the SHA-256 of its bytes, with
a CRC32C checked on every read from disk.

**Alternatives.** Store each snapshot as one file; track changed blocks
incrementally.

**Why.** Forks share every block with their parent, turn *n + 1* shares the
prefix blocks of turn *n*, and GC needs only reference counts. The CRC
catches corruption before anything is copied to the GPU, where it would turn
into wrong logprobs rather than an error.

**Cost.** Every block is hashed again on every save. Preparing an 8k-token
Qwen3-1.7B snapshot, hashing included, takes about 130 ms of background CPU.

## SQLite WAL metadata: one writer at a time, readers without the lock

**Decision.** Manifests, generations, leases and blob references live in one
SQLite database in WAL mode. Writes from any process take a cross-process
file lock and `BEGIN IMMEDIATE`. Each thread reads through a query-only
connection of its own and takes no lock.

**Alternatives.** A metadata service; one file per manifest; taking the
writer lock for every read (an earlier design); reading through the
writer's connection (another earlier design).

**Why.** The gateway and the engine are separate processes, and SQLite gives
both crash-safe transactions without a service. In WAL mode each read
statement sees one committed state, so a reader never needs the writer lock.
Taking it made the gateway's snapshot polling wait out every engine commit
with its event loop blocked; in the traced run of that design, commits took
29–105 ms. Sharing the writer's connection kept that wait inside one
process: the scheduler's manifest lookups queued behind the commit thread.

**Cost.** Writers serialize across processes. Durable commits fsync
(`synchronous=FULL`); host-memory commits and invalidations do not
(`NORMAL`), since that tier does not survive the process anyway. One
connection per reading thread.

## Publish on a commit thread, not in the engine step

**Decision.** The scheduler hands each prepared save to one commit thread,
which publishes saves in order. A request stays pending push work until its
commit finishes, so vLLM keeps stepping, and an unexpected commit failure
stops the engine at the next step.

**Alternatives.** Commit on the scheduler thread inside
`update_connector_output` (an earlier design); commit from the worker's
save job.

**Why.** On the scheduler thread a commit stalled the whole batch for that
step: 29–105 ms per 8k-token save (median 62 ms) in a traced run of that
design. Most of it was avoidable work, now removed: an ownership round trip
per already committed block instead of one per owner, deep copies of a
manifest holding every token, the same manifest validated three times, and
the parent re-validated although it was validated when it was committed. The remaining
commit, 23 ms at the median in a trace of this code, runs beside the engine
instead of inside it. The worker cannot commit itself: the scheduler owns
the generation fence.

**Cost.** One more thread competes for the GIL while it commits, and a
commit finishes asynchronously to the engine; the gateway already waits for
the manifest before answering `pause`. The commit also holds the writer
lock, so a step that has to take a lease or evict waits for a commit in
flight; manifest lookups do not.

## Generation claims fsync only for snapshots that can survive

**Decision.** Every turn claims the next session generation in SQLite.
The claim is fsync'd when the turn saves to disk; a turn that saves to host
memory commits it without an fsync.

**Alternatives.** fsync every claim (an earlier design); keep generations
in memory only.

**Why.** The claim fences stale writers, so it must outlive whatever it
protects. A host-memory snapshot dies with the process, and a process crash
keeps a committed claim anyway, since it is already in the WAL; only an OS
crash can lose it, and that loses every snapshot it fenced too. The engine
fixes a turn's durability when the turn starts, so the choice is made per
turn. This change was not traced. Admission now takes 3–17 ms for a host
restore and 14–35 ms for a disk restore, whose turn fsyncs its claim
(README).

**Cost.** None for host-memory sessions. A session that switches to disk
pays the fsync from that turn on.

## The gateway's event loop does no blocking work

**Decision.** Store calls in the gateway and the engine adapter run on worker
threads, and the gateway freezes its startup heap (`gc.collect()`,
`gc.freeze()`) once the engine is up.

**Alternatives.** Call SQLite directly from coroutines; leave garbage
collection at its defaults.

**Why.** The same event loop delivers every session's tokens, so anything
that blocks it stalls them all. Traced runs of an earlier design saw the
loop blocked for up to 197 ms by store calls (an fsync, plus waits for the
engine's writer lock), and for 157 ms by one full collection walking the
vLLM client, tokenizer and model config allocated at startup. A trace of
this code saw the loop stop for at most 42 ms, and no collection over 5 ms.
vLLM freezes its engine process the same way. The engine adapter also no longer reads the manifest: the gateway has
checked it, and the scheduler looks it up again and misses if it is gone.

**Cost.** A thread hop per store call, and objects allocated before the
freeze are never collected; they live as long as the process anyway.

## Fail closed before a DMA, fail stop after one

**Decision.** Anything that goes wrong before a copy is issued is a miss,
and vLLM recomputes. Only a copy whose completion cannot be proven stops the
engine, and its pinned memory is kept.

**Alternatives.** Retry loads; treat every connector error as fatal.

**Why.** A recompute is always correct, so a mismatched, evicted or corrupt
snapshot costs time and nothing else. A DMA that may still be writing could
corrupt another request's KV or reused pinned memory, and nothing short of
stopping can rule that out.

**Cost.** A CUDA fault during a copy takes the engine down.

## The host tier is a cache in the worker's pinned memory

**Decision.** Host-memory snapshots live in the worker's pinned pool (three
quarters of it), served to other processes over an authenticated socket
(peer credentials plus a per-owner token). When a save does not fit, the
least recently used unleased snapshot is invalidated.

**Alternatives.** Copy host snapshots into a separate shared-memory segment;
refuse saves when the tier is full.

**Why.** Restores read pinned memory directly with no extra copy, and an
evicted session costs only a recompute on its next turn.

**Cost.** Host snapshots die with the worker process, and their capacity is
bounded by pinned memory.
