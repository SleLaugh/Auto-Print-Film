import tempfile
import unittest
from pathlib import Path

from src.core.pipeline import Pipeline
from src.models import Request
from src.utils.Logger_utils import _logger_manager
from src.config import LoggingConfig
from src.utils import (
    configure_logging,
    is_debug_file_enabled,
    is_debug_log_file_enabled,
)


class DebugArtifactRetentionTest(unittest.TestCase):
    def test_cleanup_debug_artifacts_removes_intermediate_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("metadata", "nifti", "segmentation", "横状位"):
                (root / name).mkdir()
                (root / name / "marker").write_text("x", encoding="utf-8")

            Pipeline._cleanup_debug_artifacts(
                Request(dicom_dir="/tmp/dicom", output_dir=temp_dir)
            )

            self.assertFalse((root / "metadata").exists())
            self.assertFalse((root / "nifti").exists())
            self.assertFalse((root / "segmentation").exists())
            self.assertTrue((root / "横状位").exists())

    def test_debug_file_switch_is_read_from_logger_manager(self):
        original = _logger_manager.debug_file_enabled
        try:
            _logger_manager.debug_file_enabled = False
            self.assertFalse(is_debug_file_enabled())
            _logger_manager.debug_file_enabled = True
            self.assertTrue(is_debug_file_enabled())
        finally:
            _logger_manager.debug_file_enabled = original

    def test_debug_file_and_debug_log_switches_are_independent(self):
        original_file = _logger_manager.debug_file_enabled
        original_log = _logger_manager.debug_log_file_enabled
        try:
            configure_logging(LoggingConfig(
                debug_file_enabled=True,
                debug_log_file_enabled=False,
            ))
            self.assertTrue(is_debug_file_enabled())
            self.assertFalse(is_debug_log_file_enabled())

            configure_logging(LoggingConfig(
                debug_file_enabled=False,
                debug_log_file_enabled=True,
            ))
            self.assertFalse(is_debug_file_enabled())
            self.assertTrue(is_debug_log_file_enabled())
        finally:
            _logger_manager.debug_file_enabled = original_file
            _logger_manager.debug_log_file_enabled = original_log


if __name__ == "__main__":
    unittest.main()
