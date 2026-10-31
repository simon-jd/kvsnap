# Results

Every file here was measured on one RTX 4090 with TP=1. `scripts/bench-restore.sh`
reruns all of them except the two under-load files at the bottom of the table.
Every recompute arm runs on the same engine with vLLM's prefix caching off, so it
stands for a session whose KV blocks were evicted.

| File | What it holds | Produced by |
|---|---|---|
| `restore-qwen3-8b.json`, `restore-qwen3-1.7b.json` | Turn-2 TTFT per context length and arm (recompute, host, disk warm, disk cold), with per-phase medians and `save_ms`, the time `pause` took | `bench/restore_ttft.py` |
| `exact-qwen3-1.7b.json`, `exact-perturbed-qwen3-1.7b.json` | Batch-invariant runs that compare the restored turn's logprobs with recompute, and the control that flips one key vector in each disk snapshot | `bench/restore_ttft.py --batch-invariant [--perturb]` |
| `under-load-qwen3-1.7b.json` | Decode gaps of 16 running streams during and outside resumes, and the resumed turns' TTFT | `bench/restore_under_load.py` |
| `restart-qwen3-8b.json` | Resume after an engine restart, against recompute on the new engine | `bench/restart_resume.py` |
| `h2d-patterns.json` | H2D bandwidth of each copy pattern, in GB/s | `bench/h2d_patterns.py` |
| `turn-end-stalls.json` | Repeated under-load runs: `after` on this code, `before` on an earlier design | `bench/restore_under_load.py` |
| `turn-end-traces.json` | Summary of five traced under-load runs, from that earlier design through each change that removed its turn-end stalls | `bench/trace_summary.py` over `bench/restore_under_load.py --trace-dir` output |

**Which code.** All files except `turn-end-stalls.json` (`before`) and
`turn-end-traces.json` come from one run of `scripts/bench-restore.sh` on the
code in this repository. `after.source_tree` in `turn-end-stalls.json` gives
`git rev-parse HEAD:<dir>` for `kvsnap/`, `bench/` and `csrc/`. The runs used
that code before its identifiers were renamed to kvsnap (the Python package,
native symbols and environment variables) and a few comments were edited;
nothing else differs, and `bench/trace_summary.py` was added afterwards.

**Earlier design.** `before` in `turn-end-stalls.json` and the first trace in
`turn-end-traces.json` ran a design that committed saves inside the engine
step, took the writer lock for every store read, and called the store on the
gateway's event loop. Its code is not in this repository. Each trace's `code`
field says how its code differs from the published one.

**Raw output.** Per-sample logs, connector event files and the trace
directories stay in `results/raw/`, which is not committed.
