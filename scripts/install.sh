#!/usr/bin/env bash
# Create the environment each part of the repository uses.
#   scripts/install.sh cpu       .venv:          the package and the CPU tests
#   scripts/install.sh gpu       .venv-gpu:      vLLM 0.31.0, torch 2.13.0 (CUDA 13.0) and
#                                                the CUDA pinned pool; every benchmark
#   scripts/install.sh lmcache   .venv-lmcache:  the same vLLM and torch with LMCache 0.5.6,
#                                                for the LMCache baseline only
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-cpu}"
python="${KVSNAP_PYTHON:-python3.12}"
torch_index=https://download.pytorch.org/whl/cu130
case "$mode" in
  cpu)
    "$python" -m venv .venv
    .venv/bin/python -m pip install -e '.[test]'
    ;;
  gpu)
    "$python" -m venv .venv-gpu
    .venv-gpu/bin/python -m pip install --extra-index-url "$torch_index" -e '.[gpu,test]'
    .venv-gpu/bin/python -m pip install 'setuptools>=68' wheel
    KVSNAP_BUILD_CUDA=1 .venv-gpu/bin/python -m pip install --no-build-isolation --no-deps -e .
    cmake -S csrc/dataplane -B build/dataplane -DKVSNAP_CUDA=ON
    cmake --build build/dataplane -j2
    ;;
  lmcache)
    # LMCache pins older numpy, protobuf and OpenTelemetry than a plain vLLM
    # install picks, so it gets an environment of its own.
    "$python" -m venv .venv-lmcache
    .venv-lmcache/bin/python -m pip install --extra-index-url "$torch_index" \
      vllm==0.31.0 torch==2.13.0 lmcache==0.5.6
    .venv-lmcache/bin/python -m pip install --no-deps -e .
    ;;
  *)
    echo 'usage: scripts/install.sh [cpu|gpu|lmcache]' >&2
    exit 2
    ;;
esac
