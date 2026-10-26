"""nnUNet runs in its separately configured inference environment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.config import AppConfig, NnUNetConfig, load_app_config
from src.core import nnunet_command
from src.core.errors import DependencyError
from src.core.pipeline import Pipeline
from src.core.preflight import PreflightValidator
from src.models import Request
from src.stages.stage_segmentation import SegmentationStage


class ExternalNnUNetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.command = self.root / "external inference/nnUNetv2_predict"
        self.command.parent.mkdir()
        self.command.write_text("#!/bin/sh\nexit 0\n")
        self.command.chmod(0o755)

    def test_default_command_uses_the_external_path(self):
        self.assertEqual(nnunet_command.nnunet_command_prefix(), ["nnUNetv2_predict"])
        finder = Mock(return_value="/external/bin/nnUNetv2_predict")
        nnunet_command.validate_nnunet_command(finder)
        finder.assert_called_once_with("nnUNetv2_predict")

    def test_explicit_external_shell_command_needs_no_python_or_path_lookup(self):
        finder = Mock(return_value=None)
        nnunet_command.validate_nnunet_command(finder, predict_command=str(self.command))
        finder.assert_not_called()
        self.assertEqual(
            nnunet_command.nnunet_command_prefix(str(self.command)), [str(self.command)],
        )

    def test_external_interpreter_without_script_calls_official_entry_there(self):
        prefix = nnunet_command.nnunet_command_prefix(python_executable=sys.executable)
        self.assertEqual(prefix[:2], [sys.executable, "-c"])
        self.assertIn("predict_entry_point", prefix[2])
        finder = Mock(return_value=None)
        nnunet_command.validate_nnunet_command(finder, python_executable=sys.executable)
        finder.assert_not_called()

    def test_shell_wrapper_with_python_reports_configuration_error(self):
        with self.assertRaisesRegex(DependencyError, "清空 nnUNet_python"):
            nnunet_command.validate_nnunet_command(
                predict_command=str(self.command), python_executable=sys.executable,
            )
        with self.assertRaisesRegex(DependencyError, "shell 包装脚本"):
            nnunet_command.nnunet_command_prefix(str(self.command), sys.executable)

    def test_external_python_symlink_keeps_venv_path(self):
        interpreter = self.root / "external venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        prefix = nnunet_command.nnunet_command_prefix(python_executable=str(interpreter))
        self.assertEqual(prefix[0], str(interpreter))
        self.assertNotEqual(prefix[0], str(interpreter.resolve()))

    def test_interpreter_can_run_readable_script_without_execute_bit(self):
        script = self.root / "external inference/predict.py"
        script.write_text("print('external inference')\n")
        script.chmod(0o644)
        nnunet_command.validate_nnunet_command(
            predict_command=str(script), python_executable=sys.executable,
        )
        self.assertEqual(
            nnunet_command.nnunet_command_prefix(str(script), sys.executable),
            [sys.executable, str(script)],
        )

    def test_relative_external_command_is_based_on_application_directory(self):
        with patch.object(nnunet_command, "application_root", return_value=self.root):
            self.assertEqual(
                nnunet_command.nnunet_command_prefix("external inference/nnUNetv2_predict"),
                [str(self.command)],
            )

    def test_missing_configured_command_fails_before_output_is_created(self):
        input_dir = self.root / "dicom"
        input_dir.mkdir()
        output = self.root / "output"
        config = AppConfig(nnunet=NnUNetConfig(predict_command=str(self.root / "missing")))
        with self.assertRaisesRegex(DependencyError, "外部可执行文件"):
            PreflightValidator(check_model_directories=False).validate(
                Request(dicom_dir=str(input_dir), output_dir=str(output)), config,
            )
        self.assertFalse(output.exists())

    def test_loader_priorities_and_pipeline_pass_external_fields_to_segmentation(self):
        config = load_app_config(
            explicit={"nnunet": {"python_executable": sys.executable}},
            environ={"nnUNet_predict_command": str(self.command)},
            defaults={"nnUNet_python": "/old/python", "nnUNet_predict_command": "/old/command"},
        )
        self.assertEqual(config.nnunet.python_executable, sys.executable)
        self.assertEqual(config.nnunet.predict_command, str(self.command))
        stage = Pipeline(config=config).stages[1]
        self.assertEqual(stage._predict_python, sys.executable)
        self.assertEqual(stage._predict_command, str(self.command))

    def test_non_string_runner_config_is_rejected(self):
        for field in ("nnUNet_predict_command", "nnUNet_python"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "路径字符串"):
                    load_app_config(defaults={field: False}, environ={})

    def test_segmentation_keeps_arguments_and_model_environment_for_external_command(self):
        stage = SegmentationStage({
            "nnUNet_predict_command": str(self.command), "nnUNet_results": "/external/models",
        })
        input_dir = self.root / "中文 input"
        output = self.root / "segmentation/seg.nii.gz"
        with (
            patch("src.stages.stage_segmentation.run_inference_process") as run,
            patch.object(stage, "_normalize_output_file"),
        ):
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            stage._run_nnunet_inference(
                input_dir, input_dir / "raw_0000.nii.gz", output, "292", "3d_fullres", "0", True,
            )
        self.assertEqual(run.call_args.args[0], [
            str(self.command), "-i", str(input_dir), "-o", str(output.parent),
            "-d", "292", "-c", "3d_fullres", "-f", "0",
            "-step_size", "0.5", "-npp", "1", "-nps", "1", "--continue_prediction",
        ])
        self.assertEqual(run.call_args.kwargs["env"]["nnUNet_results"], "/external/models")

    def test_external_python_entry_propagates_arguments_and_exit_code(self):
        package = self.root / "stub/nnunetv2/inference"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "predict_from_raw_data.py").write_text(
            "import json, sys\n"
            "def predict_entry_point():\n"
            "    print(json.dumps(sys.argv, ensure_ascii=False))\n"
            "    return 17\n",
        )
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = str(package.parents[1])
        arguments = ["-i", "中文 DICOM", "-o", "output path", "--continue_prediction"]
        prefix = nnunet_command.nnunet_command_prefix(python_executable=sys.executable)
        # Only a stub entry runs; no real nnUNet/torch imports or model inference.
        result = subprocess.run(
            [prefix[0], "-S", *prefix[1:], *arguments], env=environment,
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(json.loads(result.stdout), ["nnUNetv2_predict", *arguments])


if __name__ == "__main__":
    unittest.main()
