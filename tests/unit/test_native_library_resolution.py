"""Native-library selection without importing or starting CUDA."""
import importlib.machinery
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvsnap.connector import dataplane


class NativeLibrarySelection(unittest.TestCase):
    def test_current_interpreter_artifact_wins_over_an_older_neighbor(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            module = project / "kvsnap/connector/dataplane.py"
            module.parent.mkdir(parents=True)
            module.touch()
            native = project / "kvsnap/_native"
            native.mkdir()
            # Create an older candidate first. The selected file must match
            # this interpreter, independent of directory enumeration order.
            (native / "_cuda_native.cpython-300-x86_64-linux-gnu.so").write_bytes(b"older fixture")
            expected = native / ("_cuda_native" + importlib.machinery.EXTENSION_SUFFIXES[0])
            expected.write_bytes(b"current fixture")
            with patch.object(dataplane, "__file__", str(module)), \
                 patch.dict(os.environ, {}, clear=True), \
                 patch.object(dataplane.C, "CDLL", side_effect=AssertionError("path lookup must not dlopen")):
                self.assertEqual(dataplane.resolve_cuda_library(), expected)

    def test_explicit_and_environment_override_paths_are_preserved_without_loading(self):
        with patch.dict(os.environ, {"KVSNAP_CUDA_LIBRARY": "/tmp/env-choice.so"}), \
             patch.object(dataplane.C, "CDLL", side_effect=AssertionError("path lookup must not dlopen")):
            self.assertEqual(dataplane.resolve_cuda_library(), Path("/tmp/env-choice.so"))
            self.assertEqual(dataplane.resolve_cuda_library("/tmp/explicit-choice.so"),
                             Path("/tmp/explicit-choice.so"))


if __name__ == "__main__":
    unittest.main()
