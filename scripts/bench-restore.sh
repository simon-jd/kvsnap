#!/usr/bin/env bash
# Every result in the README. Each run holds the machine-wide GPU lock for its
# duration. Snapshots go to an SSD scratch dir. About 15 minutes on one RTX 4090.
#   scripts/bench-restore.sh          all of it
#   scripts/bench-restore.sh smoke    one short sample of each benchmark
set -euo pipefail
cd "$(dirname "$0")/.."
python=.venv-gpu/bin/python
scratch=${KVSNAP_BENCH_STORE:-$PWD/.local-data/tmp}
out=${KVSNAP_RESULTS:-results}
mkdir -p "$scratch" "$out"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD
lock=/var/tmp/rtx4090.lock
small=(--model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)
# 16 GB of BF16 weights leave room for ~14k tokens of KV on a 24 GB card.
# This cache has the snapshot but no refs/, so load it by path; the revision is
# still passed because the connector binds snapshots to a resolved revision.
rev8b=b968826d9c46dd6066d109eabc6255188de91218
large=(--model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/$rev8b" --revision $rev8b
       --kv-cache-gib 2 --max-num-batched-tokens 8448)
run() { HF_HOME=$PWD/.local-data/huggingface/cache flock "$lock" "$python" -m "$@" --store-dir "$scratch"; }

if [[ ${1:-} == smoke ]]; then
  run bench.restore_ttft "${small[@]}" --lengths 1024 --reps 1 --output "$out/smoke-ttft.json"
  run bench.restore_ttft "${small[@]}" --lengths 1024 --reps 1 --batch-invariant --output "$out/smoke-exact.json"
  run bench.restore_under_load "${small[@]}" --length 2048 --sessions 2 --streams 4 --decode-tokens 800 \
      --interval 1 --output "$out/smoke-load.json"
  run bench.restart_resume "${small[@]}" --lengths 1024 --reps 1 --output "$out/smoke-restart.json"
  exit
fi

flock "$lock" "$python" -m bench.h2d_patterns --out "$out/h2d-patterns.json"
# Turn-2 TTFT with vLLM's default kernels and CUDA graphs.
run bench.restore_ttft "${small[@]}" --lengths 1024 2048 4096 8192 16384 --reps 5 --kv-cache-gib 6 \
    --output "$out/restore-qwen3-1.7b.json"
run bench.restore_ttft "${large[@]}" --lengths 1024 2048 4096 8192 --reps 5 --output "$out/restore-qwen3-8b.json"
# Bit-exactness against recompute, and its negative control.
run bench.restore_ttft "${small[@]}" --lengths 2048 8192 --reps 3 --batch-invariant --output "$out/exact-qwen3-1.7b.json"
run bench.restore_ttft "${small[@]}" --lengths 2048 8192 --reps 3 --batch-invariant --perturb \
    --output "$out/exact-perturbed-qwen3-1.7b.json"
# What a restore costs the requests already running.
run bench.restore_under_load "${small[@]}" --output "$out/under-load-qwen3-1.7b.json"
# Resume after an engine restart.
run bench.restart_resume "${large[@]}" --lengths 2048 8192 --reps 4 --output "$out/restart-qwen3-8b.json"
