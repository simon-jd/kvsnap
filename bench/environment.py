"""What a result was measured on: GPU, driver, clocks, PCIe link, host, software and commit.

Every benchmark stores ``environment()`` in its output. ``gpu_state()`` is
also taken again right before measuring, because an idle GPU reports low
clocks and a PCIe link that has stepped down. The host's load average is
recorded too: the benchmarks share one machine with other work, and the
engine's scheduler and the gateway are sensitive to CPU contention.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GPU_FIELDS = ("name", "driver_version", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
              "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
              "power.limit", "temperature.gpu", "memory.used", "memory.total")


def _run(*command: str) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_state() -> dict:
    """nvidia-smi's view of the first visible GPU (CUDA_VISIBLE_DEVICES respected), and the host load."""
    index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    out = _run("nvidia-smi", f"--id={index}", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader")
    if not out:
        return {"error": "nvidia-smi unavailable"}
    state = dict(zip(GPU_FIELDS, (value.strip() for value in out.splitlines()[0].split(","))))
    state["host_load_average_1_5_15"] = [round(value, 2) for value in os.getloadavg()]
    return state


def git_state() -> dict:
    commit = _run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    status = _run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def storage(path: str | os.PathLike | None) -> dict | None:
    """Model and transport of the block device that holds ``path`` (the snapshot store)."""
    if path is None:
        return None
    source = _run("df", "--output=source", str(path))
    if not source:
        return None
    device = source.splitlines()[-1].strip()
    parent = _run("lsblk", "-no", "PKNAME", device) or Path(device).name
    described = _run("lsblk", "-dno", "MODEL,TRAN", "/dev/" + parent.splitlines()[0].strip())
    return {"device": parent.splitlines()[0].strip(), "model_transport": described}


def host() -> dict:
    cpu = None
    try:
        match = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE)
        cpu = match.group(1) if match else None
    except OSError:
        pass
    memory = None
    try:
        match = re.search(r"^MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text(), re.MULTILINE)
        memory = round(int(match.group(1)) / 1024**2, 1) if match else None
    except OSError:
        pass
    return {"cpu": cpu, "cpus": os.cpu_count(), "memory_gib": memory, "kernel": platform.release(),
            "load_average_1_5_15": [round(value, 2) for value in os.getloadavg()]}


def software() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("vllm", "torch", "lmcache"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", None)
        if name == "torch":
            versions["torch_cuda"] = module.version.cuda
    return versions


def environment(store_dir: str | os.PathLike | None = None) -> dict:
    return {"gpu_at_start": gpu_state(), "git": git_state(), "host": host(), "storage": storage(store_dir),
            "software": software()}


def model_label(model: str) -> str:
    """A Hugging Face cache path such as .../models--Qwen--Qwen3-8B/snapshots/<rev> becomes Qwen/Qwen3-8B."""
    match = re.search(r"models--([^/]+)--([^/]+)/snapshots/", model)
    return f"{match.group(1)}/{match.group(2)}" if match else model


if __name__ == "__main__":
    import json
    print(json.dumps(environment(ROOT), indent=1))
