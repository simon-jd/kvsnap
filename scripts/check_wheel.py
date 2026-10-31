#!/usr/bin/env python3
"""Install a CPU wheel offline into a fresh environment outside the checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/wheel-install.json"))
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        libraries = [name for name in archive.namelist() if name.endswith(".so")]
    if len(libraries) != 1 or not libraries[0].startswith("kvsnap/_native/_io_native."):
        raise RuntimeError(f"CPU wheel has unexpected native artifacts: {libraries}")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "KVSNAP_"))}
    env.update(KVSNAP_IO_BACKEND="native", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="kvsnap-wheel-") as directory:
        root = Path(directory)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = root / "venv/bin/python"
        install = subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
                                 cwd=root, env=env, capture_output=True, text=True, check=True)
        inspect = subprocess.run([str(python), "-c",
            "import json,kvsnap; from kvsnap.store.native_io import get_io_backend; "
            "b=get_io_backend(); print(json.dumps({'package':kvsnap.__file__,'backend':b.name,'library':b.lib._name}))"],
            cwd=root, env=env, capture_output=True, text=True, check=True)
        runtime = json.loads(inspect.stdout)
        if not Path(runtime["package"]).is_relative_to(root / "venv"):
            raise RuntimeError("test imported the checkout instead of installed wheel")
        if runtime["backend"] != "native" or not Path(runtime["library"]).is_relative_to(root / "venv"):
            raise RuntimeError("installed package did not discover its own native library")
        subprocess.run([str(root / "venv/bin/kvsnap"), "demo", "--root", str(root / "store"),
                        "--output", str(root / "demo.json")], cwd=root, env=env,
                       capture_output=True, text=True, check=True)
        demo = json.loads((root / "demo.json").read_text())
        if not demo["token_equal"] or demo["evidence_tier"] != "cpu_fixture":
            raise RuntimeError("installed CLI demo did not meet its CPU contract")
        report = {"status": "passed", "evidence_tier": "fresh_cpu_wheel_install",
                  "wheel": str(wheel), "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "python": sys.version.split()[0], "packaged_libraries": libraries,
                  "runtime": runtime, "install_log": install.stdout + install.stderr,
                  "demo_token_equal": demo["token_equal"], "gpu_tested": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
