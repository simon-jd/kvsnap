"""Shared bootstrap for the kvsnap tensor-parallel jobs on PORTFOLIO_2XA10_POOL.

Fetches the assets from @PORTFOLIO_LAB.GPU.ASSETS (or PF_LOCAL_ASSETS for a dry
run), checks them against SHA256SUMS, unpacks the CUDA 13.2 compat driver, the
frozen vLLM 0.31.0 wheelhouse, the ninja wheel, Qwen3-8B (a Hugging Face cache
directory) and a kvsnap source archive, installs a venv with --no-index
--no-deps, and builds kvsnap's two native libraries with the node's g++.
Each payload is this file followed by its own body (cat), so every payload
stays one self-contained file.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

WORK = os.environ.get("PF_WORK", "/tmp/pf-kvsnap")
LOCAL = os.environ.get("PF_LOCAL_ASSETS")
NINJA = "ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
VENV = os.path.join(WORK, "venv")
# A local dry run sets PF_GPU_LOCK=/var/tmp/rtx4090.lock: each GPU process then holds the lock for its own run only.
GPU_LOCK = os.environ.get("PF_GPU_LOCK")


def gpu(cmd):
    return ["flock", GPU_LOCK, *cmd] if GPU_LOCK else list(cmd)


def sh(cmd, env=None, timeout=900, cwd=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}
    except Exception as e:
        return {"rc": None, "out": "", "err": repr(e)[:500]}


OUTBOX = os.path.join(WORK, "outbox")
RESULTS_STAGE = "@PORTFOLIO_LAB.GPU.JOBS/kvsnap-results"


def keep(name, content):
    """Write a file that put_results sends back (a bench output or a process log)."""
    os.makedirs(OUTBOX, exist_ok=True)
    with open(os.path.join(OUTBOX, name), "w") as f:
        f.write(content)


def put_results(label):
    """PUT every kept file to the results stage (a dry run leaves them in OUTBOX)."""
    if LOCAL or not os.path.isdir(OUTBOX):
        return {"outbox": OUTBOX}
    from snowflake.snowpark import Session
    s = Session.builder.getOrCreate()
    sent = {}
    for name in sorted(os.listdir(OUTBOX)):
        try:
            r = s.file.put(os.path.join(OUTBOX, name), f"{RESULTS_STAGE}/{label}/", auto_compress=False,
                           overwrite=True)[0]
            sent[name] = r.status
        except Exception as e:
            sent[name] = repr(e)[:200]
    return sent


def text(cmd, timeout=60):
    r = sh(cmd, timeout=timeout)
    return (r["out"] or r["err"] or "").strip()[:3000]


def last_json(res):
    for line in reversed((res.get("out") or "").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                pass
    return {"rc": res.get("rc"), "err_tail": (res.get("err") or "").strip().splitlines()[-12:]}


def key_lines(res, *words, limit=30):
    blob = (res.get("out") or "") + "\n" + (res.get("err") or "")
    return [x.strip()[:300] for x in blob.splitlines() if any(w in x for w in words)][:limit]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def untar(path, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(path) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:
            t.extractall(dest)


def node_info():
    os.makedirs(WORK, exist_ok=True)
    return {
        "python": sys.version.split()[0], "cpus": os.cpu_count(),
        "cpu_model": text(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo"]),
        "mem": text(["free", "-g"]), "df_work": text(["df", "-h", WORK]),
        "findmnt_work": text(["findmnt", "-T", WORK]), "shm": text(["df", "-h", "/dev/shm"]),
        "lsblk": text(["lsblk", "-o", "NAME,SIZE,TYPE,MODEL,TRAN,MOUNTPOINTS"]),
        "gpus": text(["nvidia-smi", "-L"]), "topo": text(["nvidia-smi", "topo", "-m"]),
        "p2p_read": text(["nvidia-smi", "topo", "-p2p", "r"]),
        "query": text(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,clocks.max.sm,"
                       "pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"]),
    }


def bootstrap(src_name, out):
    """Fetch, verify and unpack every asset; returns the environment for the venv's python."""
    os.makedirs(WORK, exist_ok=True)
    names = ["cuda-compat-13-2.tar", "wheelhouse-vllm-0.31.0.tar", NINJA, "qwen3-8b.tar", src_name]
    t = time.time()
    if LOCAL:
        for n in names + ["SHA256SUMS"]:
            shutil.copy(os.path.join(LOCAL, n), WORK)
    else:
        from snowflake.snowpark import Session
        s = Session.builder.getOrCreate()
        for n in names + ["SHA256SUMS"]:
            s.file.get(f"@PORTFOLIO_LAB.GPU.ASSETS/{n}", WORK)
    out["fetch_s"] = round(time.time() - t, 1)
    want = dict(reversed(line.split()) for line in open(os.path.join(WORK, "SHA256SUMS")) if line.strip())
    t = time.time()
    out["sha256_mismatch"] = [n for n in names if sha256(os.path.join(WORK, n)) != want.get(n)]
    out["sha256_s"] = round(time.time() - t, 1)
    if out["sha256_mismatch"]:
        raise RuntimeError(f"sha256 mismatch: {out['sha256_mismatch']}")
    compat, wheels, hub, src = (os.path.join(WORK, d) for d in ("compat", "wheels", "hf/hub", "src"))
    t = time.time()
    for dest, n in ((compat, names[0]), (wheels, names[1]), (hub, names[3]), (src, names[4])):
        untar(os.path.join(WORK, n), dest)
        os.remove(os.path.join(WORK, n))
    out["untar_s"] = round(time.time() - t, 1)
    t = time.time()
    sh([sys.executable, "-m", "venv", VENV], timeout=300)
    whls = sorted(os.path.join(wheels, f) for f in os.listdir(wheels) if f.endswith(".whl"))
    if NINJA not in map(os.path.basename, whls):      # newer wheelhouse tars already carry it
        whls.append(os.path.join(WORK, NINJA))
    out["ninja_in_wheelhouse"] = os.path.join(WORK, NINJA) not in whls
    r = sh([os.path.join(VENV, "bin", "pip"), "install", "-q", "--no-index", "--no-deps", *whls], timeout=1800)
    out["install"] = {"rc": r["rc"], "wheels": len(whls), "seconds": round(time.time() - t, 1),
                      "err_tail": (r.get("err") or "").strip().splitlines()[-4:]}
    shutil.rmtree(wheels, ignore_errors=True)
    py = os.path.join(VENV, "bin", "python")
    site = sh([py, "-c", "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))"])["out"].strip()
    repo = os.path.join(src, "kvsnap")
    native = os.path.join(repo, "kvsnap", "_native")
    suffix = ".cpython-312-x86_64-linux-gnu.so"
    builds = {
        "io": sh(["g++", "-std=c++17", "-O3", "-pthread", "-shared", "-fPIC", "csrc/dataplane/io.cpp",
                  "-o", os.path.join(native, "_io_native" + suffix)], cwd=repo),
        "cuda": sh(["g++", "-std=c++17", "-O2", "-pthread", "-shared", "-fPIC", f"-I{site}/include",
                    "csrc/dataplane/cuda_pool.cpp", f"-L{site}/lib", "-l:libcudart.so.13",
                    "-o", os.path.join(native, "_cuda_native" + suffix)], cwd=repo),
    }
    out["native_build"] = {k: {"rc": v["rc"], "err_tail": (v["err"] or "").strip().splitlines()[-4:]}
                           for k, v in builds.items()}
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HOME=os.path.join(WORK, "hf"), VLLM_NO_USAGE_STATS="1",
               DO_NOT_TRACK="1", PYTHONPATH=repo,
               PATH=os.path.join(VENV, "bin") + ":" + os.environ.get("PATH", ""),
               LD_LIBRARY_PATH=":".join([compat, f"{site}/lib", os.environ.get("LD_LIBRARY_PATH", "")]))
    commit_file = os.path.join(repo, "SOURCE_COMMIT")
    if os.path.exists(commit_file):
        env["KVSNAP_SOURCE_COMMIT"] = open(commit_file).read().strip()
        out["source_commit"] = env["KVSNAP_SOURCE_COMMIT"]
    return py, env, repo


# ---- kvsnap TP smoke ---------------------------------------------------------
# One job: can vLLM 0.31.0 with CUDA graphs (default mode) run Qwen3-8B at TP=2
# on the two A10-24Q vGPUs, with NCCL's default shared-memory transport and with
# NCCL_SHM_USE_CUDA_MEMCPY=1; then one short sample each of kvsnap's restore
# benchmark at TP=2, its restart benchmark at TP=2 and its restore benchmark at
# TP=1 on GPU 0. Prints a RESULT line after every step.

TORCH_CHECK = r"""
import json, torch
n = torch.cuda.device_count()
r = {"torch": torch.__version__, "count": n, "names": [torch.cuda.get_device_name(i) for i in range(n)],
     "peer": [[torch.cuda.can_device_access_peer(i, j) if i != j else None for j in range(n)] for i in range(n)],
     "free_total_gib": [[round(x / 2**30, 2) for x in torch.cuda.mem_get_info(i)] for i in range(n)]}
print(json.dumps(r))
"""

VLLM_TP = r"""
import json, random, sys, time
from vllm import LLM, SamplingParams
tp, eager = int(sys.argv[1]), sys.argv[2] == "1"
t = time.time()
llm = LLM(model="Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218", dtype="bfloat16",
          tensor_parallel_size=tp, max_model_len=8288, block_size=16, kv_cache_memory_bytes=(2 << 30) // tp,
          gpu_memory_utilization=0.80, enable_prefix_caching=False, async_scheduling=False,
          max_num_batched_tokens=8448, enforce_eager=eager, seed=0)
load_s = time.time() - t
rng = random.Random(0)
def timed(n, max_tokens=1):
    toks = [rng.randrange(1000, 100000) for _ in range(n)]
    t = time.perf_counter()
    llm.generate([{"prompt_token_ids": toks}], SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
                 use_tqdm=False)
    return round((time.perf_counter() - t) * 1e3, 1)
r = {"tp": tp, "eager": eager, "load_s": round(load_s, 1), "prefill_ms": {}}
for n in (1024, 8192):
    timed(n)
    r["prefill_ms"][n] = [timed(n) for _ in range(3)]
timed(1024, 65)
r["prefill_plus_64_decode_ms_1k"] = [timed(1024, 65) for _ in range(3)]
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0.0), use_tqdm=False)
r["text"] = out[0].outputs[0].text
print(json.dumps(r), flush=True)
"""

WORDS = ("custom", "Custom", "P2P", "p2p", "allreduce", "all_reduce", "Graph capturing", "CUDA graph", "cudagraph",
         "Available KV", "Model loading took", "torch.compile", "via SHM", "via P2P", "NET/", "NCCL version",
         "Error", "error:", "Traceback", "OutOfMemory", "out of memory")
MODEL = ["--model", "Qwen/Qwen3-8B", "--revision", "b968826d9c46dd6066d109eabc6255188de91218",
         "--max-num-batched-tokens", "8448"]
START, BUDGET_S = time.time(), 34 * 60


def emit(step, value):
    out[step] = value
    print("RESULT " + json.dumps({"step": step, step: value}), flush=True)


def tail(r, n):
    print("\n".join(((r["out"] or "") + "\n" + (r["err"] or "")).splitlines()[-n:]), flush=True)


def bench(name, module, args, env, cwd):
    if time.time() - START > BUDGET_S:
        emit(name, {"skipped": "job time budget"})
        return
    target = os.path.join(WORK, name + ".json")
    t = time.time()
    r = sh(gpu([py, "-m", module, *MODEL, *args, "--store-dir", store, "--output", target]), env=env, timeout=1200, cwd=cwd)
    keep(name + ".log", (r["out"] or "") + "\n" + (r["err"] or ""))
    if os.path.exists(target):
        keep(name + ".json", open(target).read())
    lines = [x for x in (r["out"] or "").splitlines() if x.startswith("L=") or x.startswith("{")]
    value = {"rc": r["rc"], "seconds": round(time.time() - t, 1), "samples": lines[:40],
             "key_lines": key_lines(r, *WORDS)}
    if r["rc"] == 0:
        result = json.load(open(target))
        value["rows"] = [{k: row.get(k) for k in ("context_tokens", "arm", "cold", "n", "ttft_ms_median",
                                                  "recompute_ttft_ms_median", "tokens_match")}
                         for row in result.get("summary") or result.get("rows", [])]
        value["storage"] = result["environment"].get("storage")
    else:
        tail(r, 80)
    emit(name, value)


out = {}
try:
    emit("node", node_info())
    py, env, repo = bootstrap(sys.argv[1] if len(sys.argv) > 1 else "kvsnap-src.tar", out)
    emit("bootstrap", {k: out.get(k) for k in ("fetch_s", "sha256_s", "untar_s", "install", "native_build",
                                                "source_commit", "sha256_mismatch")})
    emit("torch", last_json(sh(gpu([py, "-c", TORCH_CHECK]), env=env, timeout=300)))
    nccl = dict(env, NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT")
    arms = {"tp2_graphs_nccl_default": {}, "tp2_graphs_nccl_shm_memcpy": {"NCCL_SHM_USE_CUDA_MEMCPY": "1"}}
    two = (out["torch"].get("count") or 0) >= 2
    for name, extra in arms.items():
        if not two:
            emit(name, {"skipped": "fewer than 2 GPUs"})
            continue
        t = time.time()
        r = sh(gpu([py, "-c", VLLM_TP, "2", "0"]), env=dict(nccl, **extra), timeout=1200)
        keep(name + ".log", (r["out"] or "") + "\n" + (r["err"] or ""))
        emit(name, {"rc": r["rc"], "seconds": round(time.time() - t, 1), "result": last_json(r),
                    "key_lines": key_lines(r, *WORDS)})
        tail(r, 40)
    ok = {n: out[n]["result"]["prefill_ms"]["8192"] for n in arms
          if out[n].get("rc") == 0 and "8192" in out[n].get("result", {}).get("prefill_ms", {})}
    if two and not ok:
        r = sh(gpu([py, "-c", VLLM_TP, "2", "1"]), env=nccl, timeout=1200)
        emit("tp2_eager_diagnostic", {"rc": r["rc"], "result": last_json(r), "key_lines": key_lines(r, *WORDS)})
    store = os.path.join(WORK, "store")
    os.makedirs(store, exist_ok=True)
    if ok:
        best = min(ok, key=lambda n: sorted(ok[n])[1])
        tp_env = dict(env, **arms[best])
        emit("tp2_nccl_choice", {"arm": best, "prefill_8k_ms": ok})
        bench("tp2_restore", "bench.restore_ttft", ["--tensor-parallel-size", "2", "--kv-cache-gib", "1",
                                                     "--lengths", "8192", "--reps", "1"], tp_env, repo)
        bench("tp2_restart", "bench.restart_resume", ["--tensor-parallel-size", "2", "--kv-cache-gib", "1",
                                                       "--lengths", "8192", "--reps", "1"], tp_env, repo)
    bench("tp1_restore", "bench.restore_ttft", ["--kv-cache-gib", "2", "--lengths", "8192", "--reps", "1"],
          dict(env, CUDA_VISIBLE_DEVICES="0"), repo)
except Exception as e:
    import traceback
    emit("error", traceback.format_exc()[-2000:])
emit("put_results", put_results(os.environ.get("PF_RESULTS_LABEL", "smoke")))
emit("elapsed_s", round(time.time() - START, 1))
