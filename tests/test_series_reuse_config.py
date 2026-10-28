"""跨序列复用开关与调用协议的兼容性。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import main
from src.config import PipelineConfig, load_app_config
from src.models import Request


class SeriesReuseConfigTest(unittest.TestCase):
    def test_disabled_by_default_in_typed_and_repository_config(self):
        self.assertFalse(PipelineConfig().series_reuse_enabled)
        self.assertFalse(load_app_config(environ={}, defaults={}).pipeline.series_reuse_enabled)
        self.assertFalse(load_app_config(environ={}).pipeline.series_reuse_enabled)

    def test_explicit_environment_and_file_precedence(self):
        sources = {
            "explicit": {"pipeline": {"series_reuse_enabled": False}},
            "environ": {"AUTOPRINTFILM_SERIES_REUSE_ENABLED": "true"},
            "defaults": {"pipeline": {"series_reuse_enabled": False}},
        }
        for expected in (False, True, False):
            config = load_app_config(**sources)
            self.assertIs(config.pipeline.series_reuse_enabled, expected)
            self.assertIs(config.summary()["pipeline"]["series_reuse_enabled"], expected)
            if sources["explicit"]:
                sources["explicit"].clear()
            else:
                sources["environ"].clear()

    def test_json_pipeline_section_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "nnuNet_config.json"
            config_path.write_text(json.dumps({
                "pipeline": {"series_reuse_enabled": True},
            }), encoding="utf-8")
            config = load_app_config(environ={}, default_path=config_path)
        self.assertTrue(config.pipeline.series_reuse_enabled)

    def test_boolean_switch_rejects_ambiguous_values(self):
        for value in (1, 0, "yes", "no", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "series_reuse_enabled"):
                load_app_config(environ={}, defaults={
                    "pipeline": {"series_reuse_enabled": value},
                })
        for value in ("true", 1, None):
            with self.subTest(typed_value=value), self.assertRaisesRegex(ValueError, "series_reuse_enabled"):
                PipelineConfig(series_reuse_enabled=value)


class SeriesReuseRequestTest(unittest.TestCase):
    @staticmethod
    def request_data():
        return {"dicom_dir": "/input", "output_dir": "/output", "ScanThickness": 1.5}

    def test_old_requests_keep_empty_parent_and_serialization(self):
        request = Request.from_dict(self.request_data())
        self.assertEqual(request.series_cache_parent, "")
        self.assertNotIn("series_cache_parent", request.to_dict())
        self.assertTrue(request.validate()[0])

    def test_parent_aliases_round_trip_to_canonical_field(self):
        for field_name in ("series_cache_parent", "SeriesCacheParent"):
            request = Request.from_dict({**self.request_data(), field_name: "/check/10000166"})
            self.assertEqual(request.series_cache_parent, "/check/10000166")
            self.assertEqual(request.to_dict()["series_cache_parent"], "/check/10000166")
            self.assertTrue(request.validate()[0])

    def test_direct_path_and_null_parent(self):
        request = Request("/input", "/output", series_cache_parent=Path("/check"))
        self.assertEqual(request.to_dict()["series_cache_parent"], "/check")
        self.assertTrue(request.validate()[0])
        self.assertEqual(Request("/input", "/output", series_cache_parent=None).series_cache_parent, "")

    def test_parent_must_be_a_path(self):
        for value in (True, 1, [], {}):
            with self.subTest(value=value):
                request = Request("/input", "/output", series_cache_parent=value)
                valid, error = request.validate()
                self.assertFalse(valid)
                self.assertIn("series_cache_parent", error)

    def test_cli_parent_flag_is_optional_and_accepts_both_spellings(self):
        base_arguments = ["main.py", "-i", "/input", "-o", "/output", "--scan_thickness", "1.5"]
        with patch.object(sys, "argv", base_arguments):
            self.assertIsNone(main.parse_arguments().series_cache_parent)
        for flag in ("--series-cache-parent", "--series_cache_parent"):
            with self.subTest(flag=flag), patch.object(sys, "argv", [*base_arguments, flag, "/check"]):
                self.assertEqual(main.parse_arguments().series_cache_parent, "/check")


if __name__ == "__main__":
    unittest.main()
