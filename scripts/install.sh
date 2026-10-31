#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-cpu}"
case "$mode" in cpu|gpu) ;; *) echo 'usage: scripts/install.sh [cpu|gpu]' >&2; exit 2;; esac
python="${KVSNAP_PYTHON:-python3.12}"
"$python" -m venv .venv
if [[ "$mode" == gpu ]]; then
  .venv/bin/python -m pip install -e '.[gpu,test]'
  .venv/bin/python -m pip install 'setuptools>=68' wheel
  KVSNAP_BUILD_CUDA=1 .venv/bin/python -m pip install --no-build-isolation --no-deps -e .
  cmake -S csrc/dataplane -B build/dataplane -DKVSNAP_CUDA=ON
  cmake --build build/dataplane -j2
else
  .venv/bin/python -m pip install -e '.[test]'
fi
