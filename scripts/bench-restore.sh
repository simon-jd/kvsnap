#!/usr/bin/env bash
# Every result in the README, on one GPU. Each benchmark process takes the
# machine-wide GPU lock for its own duration only, so other jobs can run in
# between. Each configuration runs KVSNAP_RUNS times (default 3) in fresh
# processes with different seeds; the runs land in results/raw/suite/ and are
# merged into results/*.json. About two hours on one RTX 4090, plus any wait
# for the lock.
#   scripts/bench-restore.sh             all of it
#   scripts/bench-restore.sh smoke       one short sample of each benchmark
#   scripts/bench-restore.sh PART...     only these parts: h2d ttft-small ttft-large exact
#                                        under-load restart baselines lmcache trace
set -euo pipefail
cd "$(dirname "$0")/.."
python=${KVSNAP_PYTHON:-.venv-gpu/bin/python}
lmcache_python=${KVSNAP_LMCACHE_PYTHON:-.venv-lmcache/bin/python}
scratch=${KVSNAP_BENCH_STORE:-$PWD/.local-data/tmp}   # snapshot stores: put this on an SSD
out=${KVSNAP_RESULTS:-results}
raw=$out/raw/suite
runs=${KVSNAP_RUNS:-3}
mkdir -p "$scratch" "$raw"
# Models come from the Hugging Face cache, offline (scripts/download-models.sh).
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
lock=/var/tmp/rtx4090.lock
small=(--model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)
# KVSNAP_MODEL_8B may name a local snapshot directory of the same revision.
large=(--model "${KVSNAP_MODEL_8B:-Qwen/Qwen3-8B}" --revision b968826d9c46dd6066d109eabc6255188de91218
       --kv-cache-gib 2 --max-num-batched-tokens 8448)

# One benchmark process under the lock; the timeout starts once the lock is held.
gpu() { local log=$1; shift; flock "$lock" timeout 3h "$@" > "$log" 2>&1; }

# repeat PYTHON NAME MODULE ARGS...: $runs runs, merged into $out/NAME.json.
repeat() {
  local py=$1 name=$2 module=$3
  shift 3
  local files=()
  for run in $(seq 1 "$runs"); do
    gpu "$raw/$name-run$run.log" "$py" -m "$module" "$@" --seed "$run" --store-dir "$scratch" \
      --output "$raw/$name-run$run.json"
    files+=("$raw/$name-run$run.json")
  done
  "$py" -m "$module" --combine "${files[@]}" --output "$out/$name.json"
}

part_h2d() { gpu "$raw/h2d.log" "$python" -m bench.h2d_patterns --rounds 3 --out "$out/h2d-patterns.json"; }

part_ttft-small() {
  repeat "$python" restore-qwen3-1.7b bench.restore_ttft "${small[@]}" --lengths 1024 2048 4096 8192 16384 \
    --reps 5 --kv-cache-gib 6
}

part_ttft-large() {
  repeat "$python" restore-qwen3-8b bench.restore_ttft "${large[@]}" --lengths 1024 2048 4096 8192 --reps 5
}

# Bit-exactness against recompute, and its negative control: correctness, so one run each.
part_exact() {
  gpu "$raw/exact.log" "$python" -m bench.restore_ttft "${small[@]}" --lengths 2048 8192 --reps 3 \
    --batch-invariant --store-dir "$scratch" --output "$out/exact-qwen3-1.7b.json"
  gpu "$raw/exact-perturbed.log" "$python" -m bench.restore_ttft "${small[@]}" --lengths 2048 8192 --reps 3 \
    --batch-invariant --perturb --store-dir "$scratch" --output "$out/exact-perturbed-qwen3-1.7b.json"
}

part_under-load() {
  repeat "$python" under-load-qwen3-1.7b bench.restore_under_load "${small[@]}"
  repeat "$python" under-load-native-offload-qwen3-1.7b bench.restore_under_load "${small[@]}" \
    --engines native-offload
}

part_restart() {
  repeat "$python" restart-qwen3-8b bench.restart_resume "${large[@]}" --lengths 2048 8192 --reps 4
}

# What else could serve the resumed turn: vLLM's prefix cache while the KV is
# still on the GPU, and, once four 8k sessions have pushed each other out of
# 2 GiB of GPU KV, the prefix cache alone, vLLM's own CPU offloading and kvsnap.
part_baselines() {
  repeat "$python" baseline-resident-prefix-cache-qwen3-8b bench.resume_baselines "${large[@]}" \
    --engine prefix-cache --regime resident --lengths 1024 2048 4096 8192 --reps 5
  for engine in prefix-cache native-offload kvsnap; do
    repeat "$python" "baseline-evicted-$engine-qwen3-8b" bench.resume_baselines "${large[@]}" \
      --engine "$engine" --regime evicted --lengths 8192 --sessions 4 --rounds 3
  done
}

# LMCache needs its own environment (scripts/install.sh lmcache): its CPU
# backend in the rotation and under load, and its local-disk backend across
# an engine restart.
part_lmcache() {
  repeat "$lmcache_python" baseline-evicted-lmcache-qwen3-8b bench.resume_baselines "${large[@]}" \
    --engine lmcache --regime evicted --lengths 8192 --sessions 4 --rounds 3
  repeat "$lmcache_python" under-load-lmcache-qwen3-1.7b bench.restore_under_load "${small[@]}" --engines lmcache
  repeat "$lmcache_python" restart-lmcache-disk-qwen3-8b bench.restart_resume "${large[@]}" \
    --engine lmcache-disk --lengths 2048 8192 --reps 4
}

# Where the turn-end stalls stand on this code: one traced under-load run.
part_trace() {
  rm -rf "$raw/trace"
  gpu "$raw/trace.log" "$python" -m bench.restore_under_load "${small[@]}" --engines async \
    --trace-dir "$raw/trace" --store-dir "$scratch" --output "$raw/trace/result.json"
  "$python" -m bench.trace_summary --trace "published=$raw/trace" \
    --code "published=the code in this repository, at the commit in its environment" \
    --output "$out/turn-end-trace.json"
}

smoke() {
  local s=$raw/smoke
  mkdir -p "$s"
  gpu "$s/ttft.log" "$python" -m bench.restore_ttft "${small[@]}" --lengths 1024 --reps 1 \
    --store-dir "$scratch" --output "$s/ttft.json"
  gpu "$s/load.log" "$python" -m bench.restore_under_load "${small[@]}" --length 2048 --sessions 2 --streams 4 \
    --resumes 4 --decode-tokens 200 --interval 1 --store-dir "$scratch" --output "$s/load.json"
  gpu "$s/restart.log" "$python" -m bench.restart_resume "${small[@]}" --lengths 1024 --reps 1 \
    --store-dir "$scratch" --output "$s/restart.json"
  gpu "$s/baseline.log" "$python" -m bench.resume_baselines "${small[@]}" --engine native-offload \
    --regime evicted --lengths 2048 --sessions 8 --kv-cache-gib 1 --offload-gib 4 --rounds 1 \
    --store-dir "$scratch" --output "$s/baseline.json"
}

if [[ ${1:-} == smoke ]]; then
  smoke
  exit
fi
parts=("$@")
[[ ${#parts[@]} -gt 0 ]] || parts=(h2d ttft-small ttft-large exact under-load restart baselines lmcache trace)
for part in "${parts[@]}"; do
  if [[ $part == lmcache && ! -x $lmcache_python ]]; then
    echo "skipping lmcache: $lmcache_python not found (scripts/install.sh lmcache)" >&2
    continue
  fi
  "part_$part"
done
