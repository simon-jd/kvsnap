# Results

`scripts/bench-restore.sh` runs every benchmark here except the store stress
test, the two history files and the `-a10` files. Everything else ran on one
RTX 4090 (PCIe 4.0 x16) with TP=1. The `-a10` files ran as Snowflake ML Jobs
on two NVIDIA A10-24Q vGPUs (sm_86) in one node of a Snowflake trial compute
pool, driver 570 with the CUDA 13.2 forward-compatibility driver, at TP=2 or
at TP=1 on one of the two. Every recompute runs with vLLM's prefix caching off.
Each file records, per run, the GPU, driver, clocks and PCIe link both at
start and when measuring began, the host's load average, and the git commit
(`environment` or `environment_by_run`).

| File | Commit | Date | What it measured | Runs |
|---|---|---|---|---|
| `restore-qwen3-8b.json`, `restore-qwen3-1.7b.json` | `b22a2f4` | 2026-11-03 | Turn-2 TTFT per context length and arm (recompute, host, disk warm, disk cold), with per-phase medians and `save_ms`, the time `pause` took | 3 runs x 5 sessions per cell; recompute 15 per run |
| `baseline-resident-prefix-cache-qwen3-8b.json` | `eceed39` | 2026-11-01 | Turn 2 straight after turn 1 with vLLM's prefix caching on: the KV is still on the GPU | 3 x 5 per length |
| `baseline-evicted-{native-offload,lmcache,kvsnap}-qwen3-8b.json` | `b22a2f4` | 2026-11-03 | Four 8k-token sessions rotating through 2 GiB of GPU KV, three turns each after turn 1: prefix caching plus vLLM's CPU offloading, plus LMCache, and kvsnap's host tier; interleaved run by run | 3 x 12 |
| `baseline-evicted-prefix-cache-qwen3-8b.json` | `eceed39` | 2026-11-01 | The same rotation with prefix caching alone | 3 x 12 |
| `exact-qwen3-1.7b.json`, `exact-perturbed-qwen3-1.7b.json`, `exact-qwen3-8b.json` | `b22a2f4` | 2026-11-03 | Batch-invariant runs comparing the restored turn's logprobs with recompute, and the control that flips one key vector in each disk snapshot; 2k and 8k on 1.7B, 8k only on 8B | 1 x 3 per cell |
| `under-load-qwen3-1.7b.json` | `b22a2f4` | 2026-11-03 | Decode gaps of 16 running streams during and outside resumes, and the resumed turns' TTFT | 3 runs x 54 resumes per arm |
| `under-load-{native-offload,lmcache}-qwen3-1.7b.json` | `b22a2f4` | 2026-11-03 | The same with no kvsnap: one engine with prefix caching and vLLM's CPU offloading or LMCache's CPU backend, sent each turn's whole prompt; adds `num_cached_tokens` per resume. All five under-load arms ran one run of each engine in turn | 3 runs x 54 resumes |
| `resume-phases.json` | `cb933b0` (`baseline`), `b22a2f4` (`current`) | 2026-11-03 | One traced under-load run (no probes) of the load path before `b22a2f4` and of `b22a2f4`, back to back; each async and sync resume cut into phases from submit to first token (`bench/resume_phases.py`), with the copy's rate, how many loads were reported early and the run's decode gaps | 1 x 54 resumes per arm and trace |
| `restart-qwen3-8b.json` | `b22a2f4` | 2026-11-03 | Resume after an engine restart, against recompute on the new engine | 3 x 4 sessions per length |
| `restart-lmcache-disk-qwen3-8b.json` | `dbae296` | 2026-11-02 | The same restart with LMCache's CPU and local-disk backends instead of kvsnap: turn-2 TTFT, `num_cached_tokens` (0 on every resume), the disk directory's size before the restart and the restart time | 3 x 4 sessions per length |
| `restore-tp2-qwen3-8b-a10.json`, `restore-tp1-qwen3-8b-a10.json` | `4ba6e8b` | 2026-11-04 | Turn-2 TTFT of Qwen3-8B at 2k and 8k per arm on the A10 node: TP=2 with 1 GiB of KV per rank, and TP=1 on one vGPU with 2 GiB; `ranks` and the phases of the rank whose copy finished last | 3 runs x 5 sessions per cell; recompute 15 per run |
| `restart-tp2-qwen3-8b-a10.json` | `4ba6e8b` | 2026-11-04 | Resume after an engine restart at TP=2 on the A10 node, against recompute on the new engine | 3 x 4 sessions per length |
| `restart-tp2-to-tp1-qwen3-8b-a10.json` | `4ba6e8b` | 2026-11-04 | Engine A at TP=2 saves three 2k sessions durably, engine B starts at TP=1: `restore_misses`, `miss_reasons`, `loads_after_restart`, `turn2_saves_refused` | 1 x 3 sessions (one warm-up) |
| `a10x2/<run id>/` | see below | 2026-11-04 | Each Snowflake job's `report.json` (status trace, every `RESULT` line, credit estimate), its instance log, and in `files/` every benchmark process's log and per-run output | one directory per job |
| `h2d-patterns.json` | `b22a2f4` | 2026-11-03 | H2D bandwidth of each copy pattern, GB/s: the DMA paths, including the previous load path (`scatter`), and the restore kernel (`kernel`) | 3 rounds x median of 5 |
| `turn-end-trace.json` | `eceed39` | 2026-11-01 | A traced under-load run of the current save path: engine steps over 20 ms at turn ends, event-loop stalls, save phases | 1 |
| `store-stress.json` | `3b55430` (before), `eceed39` (after), in `code_commit` | 2026-11-01 | Store operations with 500 sessions, on the store before and after its scaling changes, run back to back | 5 or 20 per operation |
| `turn-end-traces.json` | an earlier design, before this repository's history | 2026-10-30 | History: five traced runs of that design and the changes that removed its turn-end stalls | 1 each |
| `turn-end-stalls.json` | `before`: the same earlier design; `after`: this repository's first commit (`source_tree`) | 2026-10-30 | History: the under-load benchmark of that time on both | 2 and 3 |

Date is the day the file's data were committed. The 2026-11-03 files all
ran in one session that day.

**Which code.** `b22a2f4` brought in the restore kernel, the early ready
event, early load reports and the gateway's reuse of the committed
manifest. The `baseline` trace of `resume-phases.json` ran from a git
worktree of `cb933b0`, just before `current`: the load path before those
changes, with the telemetry the breakdown needs. Later commits add the
phases part to `scripts/bench-restore.sh` (the same commands that session
ran), add the copy rate, early reports and decode gaps to
`bench/resume_phases.py` (the committed file was summarized with it), and
change the docs.

**Files from older commits.** They were not rerun, because the load path is
not on their path or the code they measure did not change.

**Trees.** The 2026-11-03 session and the LMCache disk restart ran on a
clean tree (`dirty: false`). The `eceed39` files say `dirty: true` only
because the suite rewrites tracked results files as it goes; no source file
differed. The 8B model was read from its local snapshot directory.

**Runs.** Runs of one configuration use different seeds in fresh processes.
Merged files pool every sample and keep each run's median (`*_by_run`).
p99 appears only where a set has at least 100 values.

**Earlier design.** The history files ran a design that committed saves
inside the engine step, took the writer lock for every store read, and
called the store on the gateway's event loop; its code is not in this
repository. Their benchmark also differs from today's: six resumes per arm,
one long decode request per stream, and a recompute arm that invalidated
the snapshot on the client's event loop just before each turn.

**Raw output.** Per-run files, logs, connector event files and trace
directories stay in `results/raw/`, which is not committed. The A10 jobs'
per-run files and logs are committed under `a10x2/`.

**A10 jobs** (`a10x2/`, UTC run ids, credits as `run_job.py` estimated them):

| Job | Source | What happened | Credits |
|---|---|---|---|
| `20261105T015320Z-kvsnap-smoke` | `7bb6471` plus the uncommitted TP changes | vLLM 0.31.0 at TP=2 with CUDA graphs came up with NCCL's defaults (8k prefill about 3.1 s). With `NCCL_SHM_USE_CUDA_MEMCPY=1` it printed nothing in 20 minutes and was killed, but its EngineCore and workers kept their GPU memory, so the job's kvsnap steps could not start | 1.93 |
| `20261105T023433Z-kvsnap-tp2-restore` | `679a451` | The first TP=2 restore run stopped the engine at its first commit: the commit telemetry read a single preparation id. Fixed in `565aa59` | 0.76 |
| `20261105T025128Z-kvsnap-tp2-restore-b` | `4ba6e8b` | `restore-tp2-qwen3-8b-a10.json` and `restart-tp2-to-tp1-qwen3-8b-a10.json` | 1.78 |
| `20261105T061322Z-kvsnap-tp1-restore` | `4ba6e8b` | `restore-tp1-qwen3-8b-a10.json` | 1.51 |
| `20261105T063935Z-kvsnap-tp2-restart` | `4ba6e8b` | `restart-tp2-qwen3-8b-a10.json` | 1.13 |

The merged `-a10` files were made from these per-run files with each
benchmark's `--combine`; `restart-tp2-to-tp1-qwen3-8b-a10.json` is the one
run's file as written.
