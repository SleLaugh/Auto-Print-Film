"""Executable releases use editable config and leave external inference isolated."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src import runtime
from src.config import load_app_config, load_slice_part_config
from src.core.nnunet_command import nnunet_command_prefix
from src.core.preflight import PreflightValidator
from src.models import Request
from src.models.window import load_window_presets
from src.stages.stage_localization import LocalizationStage
from src.stages.stage_segmentation import SegmentationStage
from src.utils.Logger_utils import LoggerManager


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class FrozenRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "moved executable package"
        self.root.mkdir()
        self.executable = self.root / "AutoPrintFilm"
        self.executable.write_text("executable fixture", encoding="utf-8")
        self.config_dir = self.root / "src/config"
        self.config_dir.mkdir(parents=True)
        for path in (SOURCE_ROOT / "src/config").glob("*.json"):
            shutil.copy2(path, self.config_dir / path.name)
        self._write_json("nnuNet_config.json", {
            "_configuration": "3d_fullres", "_fold": "0", "_continue": True,
            "nnUNet_raw": "models/raw",
            "nnUNet_preprocessed": "models/preprocessed",
            "nnUNet_results": "models/results",
            "nnUNet_predict_command": "inference/predict",
        })
        self._write_json("logger_config.json", {"console_enabled": False})
        self._write_json("window_presets.json", [
            {"name": "外置窗位", "window_width": 1234, "window_center": 321},
        ])
        self.unrelated_cwd = Path(self.temporary.name) / "worker directory"
        self.unrelated_cwd.mkdir()
        (self.unrelated_cwd / "config").mkdir()
        for filename in ("nnuNet_config.json", "logger_config.json", "slice_part_config.json"):
            (self.unrelated_cwd / "config" / filename).write_text("invalid JSON", encoding="utf-8")

    def _write_json(self, filename, value):
        (self.config_dir / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _frozen(self):
        stack = ExitStack()
        stack.enter_context(patch.object(sys, "frozen", True, create=True))
        stack.enter_context(patch.object(sys, "_MEIPASS", str(self.root / "_internal"), create=True))
        stack.enter_context(patch.object(sys, "executable", str(self.executable)))
        stack.enter_context(patch.object(Path, "cwd", return_value=self.unrelated_cwd))
        return stack

    def test_source_root_and_config_override_order_remain_unchanged(self):
        with patch.object(sys, "frozen", False, create=True), patch.object(Path, "cwd", return_value=self.unrelated_cwd):
            self.assertEqual(runtime.application_root(), SOURCE_ROOT)
            self.assertEqual(runtime.config_candidates("logger_config.json"), [
                self.unrelated_cwd / "config/logger_config.json",
                self.unrelated_cwd / "logger_config.json",
                SOURCE_ROOT / "src/config/logger_config.json",
            ])

    def test_frozen_root_tracks_the_moved_executable_and_ignores_internal_directory(self):
        with self._frozen():
            self.assertEqual(runtime.application_root(), self.root)
            self.assertEqual(runtime.config_candidates("logger_config.json"), [
                self.config_dir / "logger_config.json",
            ])
        moved = self.root.with_name("another moved package")
        self.root.rename(moved)
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(moved / self.executable.name)):
            self.assertEqual(runtime.application_root(), moved)
            self.assertEqual(runtime.config_path("window_presets.json"), moved / "src/config/window_presets.json")

    def test_config_loader_and_slice_rules_use_external_json(self):
        with self._frozen():
            config = load_app_config(environ={})
            slice_config = load_slice_part_config()
        self.assertEqual(config.nnunet.results, "models/results")
        self.assertFalse(config.logging.console_enabled)
        self.assertEqual(config.slice_part_config, slice_config)
        self.assertIn("腰椎", slice_config["slice_parts"])

    def test_missing_external_slice_config_does_not_fall_back_to_source(self):
        (self.config_dir / "slice_part_config.json").unlink()
        with self._frozen(), self.assertRaises(FileNotFoundError):
            load_slice_part_config()

    def test_window_presets_use_editable_file_and_honor_explicit_override(self):
        override = self.root / "override-windows.json"
        override.write_text('[{"name":"调用方窗位","window_width":99,"window_center":5}]', encoding="utf-8")
        with self._frozen():
            presets = load_window_presets()
            overridden = load_window_presets(override)
        self.assertEqual(set(presets), {"外置窗位"})
        self.assertEqual(presets["外置窗位"].window_width, 1234)
        self.assertEqual(set(overridden), {"调用方窗位"})

    def test_logger_and_direct_stages_use_the_same_external_config(self):
        logger = object.__new__(LoggerManager)
        with self._frozen(), redirect_stdout(io.StringIO()):
            logger._load_config()
            segmentation = SegmentationStage()
            localization = LocalizationStage()
        self.assertFalse(logger.console_enabled)
        self.assertEqual(segmentation._nnunet_env_vars["nnUNet_results"], str(self.root / "models/results"))
        self.assertEqual(localization._nnunet_paths["nnUNet_results"], str(self.root / "models/results"))

    def test_relative_external_command_and_injected_model_paths_use_executable_root(self):
        with self._frozen():
            prefix = nnunet_command_prefix("inference/predict")
            config = load_app_config(environ={})
            segmentation = SegmentationStage({
                "nnUNet_raw": config.nnunet.raw,
                "nnUNet_preprocessed": config.nnunet.preprocessed,
                "nnUNet_results": config.nnunet.results,
                "slice_part_config": config.slice_part_config,
            })
        self.assertEqual(prefix, [str(self.root / "inference/predict")])
        self.assertEqual(segmentation._nnunet_env_vars["nnUNet_raw"], str(self.root / "models/raw"))

    def test_model_preflight_and_stages_resolve_paths_without_worker_cwd(self):
        for name in ("raw", "preprocessed", "results"):
            (self.root / "models" / name).mkdir(parents=True)
        command = self.root / "inference/predict"
        command.parent.mkdir()
        command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        command.chmod(0o755)
        data = json.loads((self.config_dir / "nnuNet_config.json").read_text(encoding="utf-8"))
        data["nnUNet_preprocessed"] = "$AUTOPRINT_MODEL_TEST_ROOT/preprocessed"
        self._write_json("nnuNet_config.json", data)
        input_dir = self.unrelated_cwd / "dicom"
        input_dir.mkdir()
        output_dir = self.unrelated_cwd / "output"
        with self._frozen(), patch.dict(os.environ, {"AUTOPRINT_MODEL_TEST_ROOT": str(self.root / "models")}):
            config = load_app_config(environ={})
            PreflightValidator().validate(Request(str(input_dir), str(output_dir)), config)
            segmentation = SegmentationStage(data)
            localization = LocalizationStage(data)
        self.assertEqual(segmentation._nnunet_env_vars, localization._nnunet_paths)
        self.assertEqual(segmentation._nnunet_env_vars["nnUNet_preprocessed"], str(self.root / "models/preprocessed"))
        self.assertFalse(output_dir.exists())

    def test_development_subprocess_environment_is_an_unchanged_independent_copy(self):
        environment = {"LD_LIBRARY_PATH": "/development/lib", "LD_LIBRARY_PATH_ORIG": "/older/lib", "PATH": "/external/bin"}
        with patch.object(sys, "frozen", False, create=True):
            child = runtime.external_process_environment(environment)
        self.assertEqual(child, environment)
        self.assertIsNot(child, environment)

    def test_frozen_external_environment_restores_original_path_or_removes_bootloader_path(self):
        for original in (None, "", "/external/cuda/lib:/system/lib"):
            with self.subTest(original=original):
                environment = {"LD_LIBRARY_PATH": str(self.root / "_internal"), "PATH": "/external/bin", "nnUNet_results": "/external/models"}
                if original is not None:
                    environment["LD_LIBRARY_PATH_ORIG"] = original
                before = dict(environment)
                with self._frozen(), patch.object(sys, "platform", "linux"):
                    child = runtime.external_process_environment(environment)
                if original is None:
                    self.assertNotIn("LD_LIBRARY_PATH", child)
                else:
                    self.assertEqual(child["LD_LIBRARY_PATH"], original)
                self.assertEqual(child["PATH"], "/external/bin")
                self.assertEqual(child["nnUNet_results"], "/external/models")
                self.assertEqual(environment, before)

    def test_segmentation_restores_child_library_environment_without_mutating_parent(self):
        environment = {"LD_LIBRARY_PATH": str(self.root / "_internal"), "LD_LIBRARY_PATH_ORIG": "/external/lib", "PATH": "/external/bin"}
        with self._frozen(), patch.object(sys, "platform", "linux"), patch.dict(os.environ, environment, clear=True), redirect_stdout(io.StringIO()):
            stage = SegmentationStage()
            output = self.root / "output/seg.nii.gz"
            with patch("src.stages.stage_segmentation.run_inference_process") as run, patch.object(stage, "_normalize_output_file"):
                run.return_value = subprocess.CompletedProcess([], 0, "", "")
                stage._run_nnunet_inference(self.root / "input", self.root / "input/raw_0000.nii.gz", output, "292", "3d_fullres", "0", True)
            child = run.call_args.kwargs["env"]
            self.assertEqual(child["LD_LIBRARY_PATH"], "/external/lib")
            self.assertEqual(child["nnUNet_results"], str(self.root / "models/results"))
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], environment["LD_LIBRARY_PATH"])


if __name__ == "__main__":
    unittest.main()
