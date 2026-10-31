"""Independent storage syscall-order checks; these do not simulate power loss."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvsnap.store import SnapshotStore
from kvsnap.store.store import BlobWriter


class StoreDirectoryReviewTests(unittest.TestCase):
    def check_directory_publication(self, factory):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "new-parent" / "store"
            synced = []
            real_fsync = os.fsync

            def record_fsync(fd):
                synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
                return real_fsync(fd)

            with patch("os.fsync", side_effect=record_fsync):
                instance = factory(root)
            try:
                # fsync(child) alone does not persist its directory entry in
                # its parent. Every newly created ancestor must be published.
                self.assertIn(base, synced, "new-parent entry was never synced")
                self.assertIn(root.parent, synced, "store entry was never synced")
                self.assertIn(root, synced, "blob/prepared entries were never synced")
            finally:
                if isinstance(instance, SnapshotStore):
                    instance.close()

    def test_snapshot_store_persists_new_ancestor_directory_entries(self):
        self.check_directory_publication(SnapshotStore)

    def test_worker_writer_persists_new_ancestor_directory_entries(self):
        self.check_directory_publication(BlobWriter)


if __name__ == "__main__":
    unittest.main()
