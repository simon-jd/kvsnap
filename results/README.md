# Results

`scripts/bench-restore.sh` runs every benchmark here except the store stress
test and the two history files. Everything ran on one RTX 4090 (PCIe 4.0
x16) with TP=1, and every recompute runs with vLLM's prefix caching off.
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
directories stay in `results/raw/`, which is not committed.
