#!/usr/bin/env bash
# Download the two models the benchmarks use, at the revisions they pin, into
# the Hugging Face cache (HF_HUB_CACHE, else HF_HOME/hub, else
# ~/.cache/huggingface/hub), which scripts/bench-restore.sh then reads
# offline. About 20 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
"${KVSNAP_PYTHON:-.venv-gpu/bin/python}" - <<'PY'
from huggingface_hub import snapshot_download

for repo, revision in (("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
                       ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")):
    print(snapshot_download(repo, revision=revision))
PY
