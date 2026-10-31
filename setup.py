"""Build the native file I/O by default; opt into the CUDA pinned pool explicitly.

``KVSNAP_BUILD_CUDA=1`` also builds ``kvsnap._native._cuda_native`` (needs the CUDA
toolkit; ``CUDA_HOME`` defaults to /usr/local/cuda). It is a plain C ABI loaded
with ctypes, so it does not link against torch.
"""
import os
from pathlib import Path
from setuptools import Extension, setup

extensions = [Extension("kvsnap._native._io_native", ["csrc/dataplane/io.cpp"], language="c++",
                        extra_compile_args=["-std=c++17", "-O3", "-pthread"],
                        extra_link_args=["-pthread"])]
if os.environ.get("KVSNAP_BUILD_CUDA") == "1":
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda / "include/cuda_runtime_api.h").is_file():
        raise RuntimeError(f"CUDA toolkit headers not found under {cuda}; set CUDA_HOME")
    extensions.append(Extension("kvsnap._native._cuda_native", ["csrc/dataplane/cuda_pool.cpp"],
                                language="c++", include_dirs=[str(cuda / "include")],
                                library_dirs=[str(cuda / "lib64")], runtime_library_dirs=[str(cuda / "lib64")],
                                libraries=["cudart"], extra_compile_args=["-std=c++17", "-O2", "-pthread"],
                                extra_link_args=["-pthread"]))

setup(ext_modules=extensions)
