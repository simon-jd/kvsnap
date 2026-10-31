"""Real SIGKILL of the CPU storage protocol; it tests neither power loss nor the GPU."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from kvsnap.store import CorruptSnapshotError, SnapshotStore


WORKER = r'''
import os, signal, sys
from kvsnap.models import KVIdentity, SamplingConfig, SnapshotManifest
from kvsnap.store import SnapshotStore
root, point, level = sys.argv[1:]
def fail(name):
    if name == point:
        os.kill(os.getpid(), signal.SIGKILL)
identity = KVIdentity("fixture", "rev", "tokenizer-rev", "bf16", 2, "fixture-BHTD",
                      {"layers":1}, {"max_model_len":32})
m = SnapshotManifest("snapshot", "session", 0, identity, (10,11,12,13), 3, (13,),
                     42, SamplingConfig(), 8, 0, durability=level)
store = SnapshotStore(root, failpoint=fail)
store.commit(m, [b"prefix", b"tail"])
if point == "during_h2d":
    store.load("snapshot")
raise RuntimeError("failpoint not reached")
'''


class StoreCrashTests(unittest.TestCase):
    def test_sigkill_at_all_eight_boundaries_for_persistent_levels(self):
        points = ("after_d2h", "during_blob_write", "before_fdatasync", "after_rename",
                  "before_manifest_commit", "after_manifest_commit", "before_gpu_release", "during_h2d")
        committed = {"after_manifest_commit", "before_gpu_release", "during_h2d"}
        for level in ("process", "durable"):
            for point in points:
                with self.subTest(level=level, point=point), tempfile.TemporaryDirectory() as directory:
                    process = subprocess.run([sys.executable, "-c", WORKER, directory, point, level],
                                             cwd=Path(__file__).resolve().parents[2],
                                             capture_output=True, timeout=20)
                    self.assertEqual(process.returncode, -signal.SIGKILL, process.stderr.decode())
                    with SnapshotStore(directory) as store:
                        if point in committed:
                            self.assertEqual(store.load("snapshot"), [b"prefix", b"tail"])
                        else:
                            with self.assertRaises(CorruptSnapshotError):
                                store.load("snapshot")
                            self.assertEqual(store.get_manifest("snapshot", include_invalid=True)["state"], "Invalid")
                        store.collect_garbage()
                        for row in store.export()["blobs"]:
                            self.assertGreater(row["refcount"], 0)


if __name__ == "__main__":
    unittest.main()
