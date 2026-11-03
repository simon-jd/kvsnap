# Results

`scripts/bench-restore.sh` produced every file here except the store stress
test and the two history files at the bottom. It ran on one RTX 4090 (PCIe
4.0 x16) with TP=1, and every recompute runs with vLLM's prefix caching off.
Each file records, per run, the GPU, driver, clocks and PCIe link both at
start and when measuring began, the host's load average, and the git commit
(`environment` or `environment_by_run`).

**Which code.** Every file but the two history files and the four named
below recorded commit `eceed39`, and those runs report `dirty: true` only because the suite rewrites tracked
results files as it goes; no source file differed from that commit. Later
commits change how `bench/environment.py` sets that flag (it now ignores
`results/`) and how `bench/store_stress.py` records its package path; move
two load paths the connector no longer uses into `bench/h2d_patterns.py`;
add the LMCache disk-restart and offloader under-load arms; delete three
unused scripts; split the store's class across modules with its methods
unchanged; and change the docs. The four files added last,
`exact-qwen3-8b.json`, `restart-lmcache-disk-qwen3-8b.json` and
`under-load-{native-offload,lmcache}-qwen3-1.7b.json`, ran at `dbae296` with
a clean tree (`dirty: false`): the suite's commands for them, run on their
own, with the 8B model read from its local snapshot directory. Commits after it delete the scripts, split
the store, add the 8B exactness run to `scripts/bench-restore.sh` with the
arguments it ran with, and change the docs.

| File | What it holds | Runs |
|---|---|---|
| `restore-qwen3-8b.json`, `restore-qwen3-1.7b.json` | Turn-2 TTFT per context length and arm (recompute, host, disk warm, disk cold), with per-phase medians and `save_ms`, the time `pause` took | 3 runs x 5 sessions per cell; recompute 15 per run |
| `baseline-resident-prefix-cache-qwen3-8b.json` | Turn 2 straight after turn 1 with vLLM's prefix caching on: the KV is still on the GPU | 3 x 5 per length |
| `baseline-evicted-{prefix-cache,native-offload,lmcache,kvsnap}-qwen3-8b.json` | Four 8k-token sessions rotating through 2 GiB of GPU KV, three turns each after turn 1: prefix caching alone, plus vLLM's CPU offloading, plus LMCache, and kvsnap's host tier | 3 x 12 |
| `exact-qwen3-1.7b.json`, `exact-perturbed-qwen3-1.7b.json`, `exact-qwen3-8b.json` | Batch-invariant runs comparing the restored turn's logprobs with recompute, and the control that flips one key vector in each disk snapshot; 2k and 8k on 1.7B, 8k only on 8B | 1 x 3 per cell |
| `under-load-qwen3-1.7b.json` | Decode gaps of 16 running streams during and outside resumes, and the resumed turns' TTFT | 3 runs x 54 resumes per arm |
| `under-load-{native-offload,lmcache}-qwen3-1.7b.json` | The same with no kvsnap: one engine with prefix caching and vLLM's CPU offloading or LMCache's CPU backend, sent each turn's whole prompt; adds `num_cached_tokens` per resume | 3 runs x 54 resumes |
| `restart-qwen3-8b.json` | Resume after an engine restart, against recompute on the new engine | 3 x 4 sessions per length |
| `restart-lmcache-disk-qwen3-8b.json` | The same restart with LMCache's CPU and local-disk backends instead of kvsnap: turn-2 TTFT, `num_cached_tokens` (0 on every resume), the disk directory's size before the restart and the restart time | 3 x 4 sessions per length |
| `h2d-patterns.json` | H2D bandwidth of each copy pattern, GB/s | 3 rounds x median of 5 |
| `turn-end-trace.json` | A traced under-load run of this code: engine steps over 20 ms at turn ends, event-loop stalls, save phases | 1 |
| `store-stress.json` | Store operations with 500 sessions, on the store before and after its scaling changes (`code_commit`), run back to back | 5 or 20 per operation |
| `turn-end-traces.json` | History: five traced runs of an earlier design and the changes that removed its turn-end stalls | 1 each |
| `turn-end-stalls.json` | History: the under-load benchmark of that time on the earlier design (`before`) and on the code of this repository's first commit (`after`, its `source_tree`) | 2 and 3 |

Runs of one configuration use different seeds in fresh processes; merged
files pool every sample and keep each run's median (`*_by_run`). p99 appears
only where a set has at least 100 values.

**Earlier design.** The history files ran a design that committed saves
inside the engine step, took the writer lock for every store read, and
called the store on the gateway's event loop; its code is not in this
repository. Their benchmark also differs from today's: six resumes per arm,
one long decode request per stream, and a recompute arm that invalidated
the snapshot on the client's event loop just before each turn.

**Raw output.** Per-run files, logs, connector event files and trace
directories stay in `results/raw/`, which is not committed.
