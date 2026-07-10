import json
import tempfile
import unittest
from pathlib import Path

from src.core.artifacts import ArtifactManager
from src.core.errors import ArtifactError


class ArtifactManagerTest(unittest.TestCase):
    def test_prepare_recreates_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "user-file.txt"
            marker.write_text("keep", encoding="utf-8")

            ArtifactManager(root).prepare()

            self.assertFalse(marker.exists())
            self.assertTrue(root.is_dir())
            self.assertFalse((root / "final").exists())
            self.assertFalse((root / "debug").exists())
            self.assertFalse((root / "intermediate").exists())
            self.assertFalse((root / "logs").exists())

    def test_prepares_empty_directory_and_writes_atomic_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = ArtifactManager(root)
            manager.prepare()
            manager.write_manifest(manager.build_manifest_stub("job-1"))

            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], "job_manifest_v1")
            self.assertEqual(manifest["job_id"], "job-1")
            self.assertFalse((root / ".manifest.json.tmp").exists())

    def test_prepare_removes_registered_and_unknown_old_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = ArtifactManager(root)
            manager.prepare()
            owned = root / "intermediate" / "owned.txt"
            owned.parent.mkdir(parents=True, exist_ok=True)
            owned.write_text("owned", encoding="utf-8")
            foreign = root / "keep.txt"
            foreign.write_text("keep", encoding="utf-8")
            manager.register("intermediate", owned, "intermediate")
            manager.write_manifest(manager.build_manifest_stub("job-1"))

            ArtifactManager(root).prepare()

            self.assertFalse(owned.exists())
            self.assertFalse(foreign.exists())

    def test_existing_manifest_is_recreated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = ArtifactManager(root)
            manager.prepare()
            owned = root / "intermediate" / "old.txt"
            owned.parent.mkdir(parents=True)
            owned.write_text("old", encoding="utf-8")
            manager.register("intermediate", owned, "intermediate")
            manager.write_manifest(manager.build_manifest_stub("job-1"))

            ArtifactManager(root).prepare()

            self.assertFalse(owned.exists())

    def test_cleanup_keeps_intermediate_after_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(Path(temp_dir))
            manager.prepare()
            path = manager.root / "intermediate" / "raw.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
            manager.register("nifti", path, "intermediate")

            warnings = manager.cleanup(success=False)

            self.assertEqual(warnings, [])
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
