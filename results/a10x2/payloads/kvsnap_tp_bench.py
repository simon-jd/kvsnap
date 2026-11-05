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
    """Run cmd in its own process group; on timeout kill the whole group (vLLM's EngineCore and
    workers too, or they keep their GPU memory) and keep the output so far."""
    import signal
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd,
                             start_new_session=True)
    except Exception as e:
        return {"rc": None, "out": "", "err": repr(e)[:500]}
    try:
        out, err = p.communicate(timeout=timeout)
        return {"rc": p.returncode, "out": out, "err": err}
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return {"rc": None, "out": out, "err": (err or "") + f"\nTIMEOUT after {timeout} s: process group killed"}


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


def gpu_memory():
    return text(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"])


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
    for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(name, None)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HOME=os.path.join(WORK, "hf"), HF_HUB_CACHE=hub,
               VLLM_NO_USAGE_STATS="1",
               DO_NOT_TRACK="1", PYTHONPATH=repo,
               PATH=os.path.join(VENV, "bin") + ":" + os.environ.get("PATH", ""),
               LD_LIBRARY_PATH=":".join([compat, f"{site}/lib", os.environ.get("LD_LIBRARY_PATH", "")]))
    commit_file = os.path.join(repo, "SOURCE_COMMIT")
    if os.path.exists(commit_file):
        env["KVSNAP_SOURCE_COMMIT"] = open(commit_file).read().strip()
        out["source_commit"] = env["KVSNAP_SOURCE_COMMIT"]
    return py, env, repo


# ---- kvsnap TP measurements ---------------------------------------------------
# argv: <kvsnap source tar> <plan> [budget minutes]. Runs the plan's benchmark
# processes one after another (each a fresh engine), keeps every output JSON and
# process log, and PUTs them to the results stage after each step. A step is not
# started when the previous step's duration would overrun the time budget.

MODEL = ["--model", "Qwen/Qwen3-8B", "--revision", "b968826d9c46dd6066d109eabc6255188de91218",
         "--max-num-batched-tokens", "8448"]
# NCCL's defaults: with NCCL_SHM_USE_CUDA_MEMCPY=1 the TP=2 engine hung in the smoke job (no output in 20 min).
NCCL = {}
TP2 = ["--tensor-parallel-size", "2", "--kv-cache-gib", "1"]      # 2 GiB of KV in all, as at TP=1
TP1 = ["--kv-cache-gib", "2"]
RESTORE = ["--lengths", "2048", "8192", "--reps", "5"]
RESTART = ["--lengths", "2048", "8192", "--reps", "4"]
PLANS = {
    "tp2-restore": [(f"restore-tp2-run{i}", "bench.restore_ttft", TP2 + RESTORE + ["--seed", str(i)], NCCL, None)
                    for i in (1, 2, 3)]
                   + [("restart-tp2-to-tp1", "bench.restart_resume",
                       TP2 + ["--restart-tensor-parallel-size", "1", "--lengths", "2048", "--reps", "2"], NCCL, None)],
    "tp1-restore": [(f"restore-tp1-run{i}", "bench.restore_ttft", TP1 + RESTORE + ["--seed", str(i)], {}, "0")
                    for i in (1, 2, 3)],
    "tp2-restart": [(f"restart-tp2-run{i}", "bench.restart_resume", TP2 + RESTART + ["--seed", str(i)], NCCL, None)
                    for i in (1, 2, 3)],
    # A local dry run of the plumbing on one GPU.
    "dry": [("restore-dry", "bench.restore_ttft", TP1 + ["--lengths", "2048", "--reps", "1"], {}, "0"),
            ("restart-dry", "bench.restart_resume", TP1 + ["--lengths", "2048", "--reps", "1"], {}, "0")],
}
WORDS = ("Error", "error:", "Traceback", "OutOfMemory", "out of memory", "Custom allreduce", "custom_all_reduce")
START = time.time()
out = {}


def emit(step, value):
    out[step] = value
    print("RESULT " + json.dumps({"step": step, step: value}), flush=True)


try:
    src, plan = sys.argv[1], sys.argv[2]
    budget = float(sys.argv[3]) * 60 if len(sys.argv) > 3 else 33 * 60
    label = os.environ.get("PF_RESULTS_LABEL", plan)
    emit("node", node_info())
    py, env, repo = bootstrap(src, out)
    emit("bootstrap", {k: out.get(k) for k in ("fetch_s", "sha256_s", "untar_s", "install", "native_build",
                                                "source_commit", "sha256_mismatch", "ninja_in_wheelhouse")})
    store = os.path.join(WORK, "store")
    os.makedirs(store, exist_ok=True)
    longest = 0.0
    for name, module, args, extra, visible in PLANS[plan]:
        if time.time() - START + longest > budget:
            emit(name, {"skipped": f"would overrun the {budget / 60:.0f}-minute budget"})
            continue
        e = dict(env, **extra)
        if visible is not None:
            e["CUDA_VISIBLE_DEVICES"] = visible
        target = os.path.join(WORK, name + ".json")
        t = time.time()
        r = sh(gpu([py, "-m", module, *MODEL, *args, "--store-dir", store, "--output", target]), env=e,
               timeout=900, cwd=repo)
        longest = max(longest, time.time() - t)
        keep(name + ".log", (r["out"] or "") + "\n" + (r["err"] or ""))
        value = {"rc": r["rc"], "seconds": round(time.time() - t, 1), "nccl": extra, "key_lines": key_lines(r, *WORDS),
                 "gpu_memory_after": gpu_memory()}
        if r["rc"] == 0 and os.path.exists(target):
            keep(name + ".json", open(target).read())
            result = json.load(open(target))
            value["rows"] = [{k: row.get(k) for k in ("context_tokens", "arm", "cold", "n", "ttft_ms_median",
                                                      "recompute_ttft_ms_median", "tokens_match")}
                             for row in result.get("summary") or result.get("rows", [])]
            for k in ("restore_misses", "miss_reasons", "loads_after_restart", "turn2_saves_refused"):
                if k in result:
                    value[k] = result[k]
        else:
            print("\n".join(((r["out"] or "") + "\n" + (r["err"] or "")).splitlines()[-60:]), flush=True)
        emit(name, value)
        emit(name + "-put", put_results(label))
        if r["rc"] != 0:
            emit("stopped", f"{name} failed: the rest of the plan is skipped")
            break
except Exception:
    import traceback
    emit("error", traceback.format_exc()[-2000:])
emit("elapsed_s", round(time.time() - START, 1))
