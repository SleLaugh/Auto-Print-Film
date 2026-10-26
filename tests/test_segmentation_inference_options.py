"""验证推理选项传到外部命令，使用模拟输出避免运行真实模型。"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import Request
from src.stages.stage_segmentation import SegmentationStage


class SegmentationInferenceOptionsTest(unittest.TestCase):
    def run_inference(self, options):
        stage = SegmentationStage({"_continue": False, **options})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "nifti" / "raw_0000.nii.gz"
            input_path.parent.mkdir()
            input_path.touch()
            # 每次使用独立缓存，确保已有分割不会跳过待验证的推理命令。
            request = Request(
                dicom_dir=str(root / "dicom"),
                output_dir=str(root / "output"),
                cache_dir=str(root / "cache"),
                scan_thickness=1.0,
            )
            context = ProcessingContext(request)
            context.set_result("preprocessing", {"nifti_path": str(input_path)})

            def predict(command, **kwargs):
                output_dir = Path(command[command.index("-o") + 1])
                sitk.WriteImage(
                    sitk.Image([2, 2, 2], sitk.sitkUInt8), str(output_dir / "raw.nii.gz"),
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch(
                "src.stages.stage_segmentation.run_inference_process", side_effect=predict,
            ) as run:
                stage.execute(context)
            run.assert_called_once()
            command = run.call_args.args[0]
            output_path = Path(context.get_result("segmentation")["seg_path"])
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.name, "seg.nii.gz")
            self.assertEqual(command[command.index("-c") + 1], "3d_fullres")
            self.assertEqual(command[command.index("-f") + 1], "0")
            return command

    def test_defaults_pass_step_size_and_keep_tta_enabled(self):
        command = self.run_inference({})
        self.assertEqual(command.count("-step_size"), 1)
        self.assertEqual(command[command.index("-step_size") + 1], "0.5")
        self.assertNotIn("--disable_tta", command)
        self.assertEqual(command[command.index("-npp") + 1], "1")
        self.assertEqual(command[command.index("-nps") + 1], "1")

    def test_worker_options_reach_subprocess(self):
        options = {"num_processes_preprocessing": 2, "num_processes_segmentation_export": 4}
        for config in (options, {"nnunet": options}):
            with self.subTest(config=config):
                command = self.run_inference(config)
                self.assertEqual(command.count("-npp"), 1)
                self.assertEqual(command.count("-nps"), 1)
                self.assertEqual(command[command.index("-npp") + 1], "2")
                self.assertEqual(command[command.index("-nps") + 1], "4")

    def test_worker_text_values_are_normalized_before_subprocess(self):
        command = self.run_inference({
            "num_processes_preprocessing": "2", "num_processes_segmentation_export": "3",
        })
        self.assertEqual(command[command.index("-npp") + 1], "2")
        self.assertEqual(command[command.index("-nps") + 1], "3")

    def test_invalid_worker_options_are_rejected_before_inference(self):
        for field in ("num_processes_preprocessing", "num_processes_segmentation_export"):
            for value in (0, -1, True, False, 1.5, 1.0, "1.5", "nan", float("inf")):
                for options in ({field: value}, {"nnunet": {field: value}}):
                    with self.subTest(field=field, value=value, options=options):
                        with patch("src.stages.stage_segmentation.run_inference_process") as run:
                            with self.assertRaisesRegex(ValueError, field):
                                SegmentationStage(options)
                            run.assert_not_called()

    def test_flat_and_nested_options_reach_subprocess(self):
        options = {"disable_tta": True, "step_size": 0.75}
        for config in (options, {"nnunet": options}):
            with self.subTest(config=config):
                command = self.run_inference(config)
                self.assertEqual(command.count("--disable_tta"), 1)
                self.assertEqual(command[command.index("-step_size") + 1], "0.75")

    def test_text_values_do_not_pass_disable_tta_for_false(self):
        for value, disabled in (("false", False), ("true", True)):
            with self.subTest(value=value):
                command = self.run_inference({
                    "nnunet": {"disable_tta": value, "step_size": "0.75"},
                })
                self.assertEqual("--disable_tta" in command, disabled)
                self.assertEqual(command[command.index("-step_size") + 1], "0.75")

    def test_invalid_step_size_is_rejected_before_inference(self):
        values = (0, -0.1, 1.1, True, False, float("nan"), float("inf"), "nan", "inf", "fast")
        for value in values:
            for options in ({"step_size": value}, {"nnunet": {"step_size": value}}):
                with self.subTest(options=options):
                    with patch("src.stages.stage_segmentation.run_inference_process") as run:
                        with self.assertRaisesRegex(ValueError, "step_size"):
                            SegmentationStage(options)
                        run.assert_not_called()

    def test_invalid_disable_tta_is_rejected_before_inference(self):
        for value in (0, 1, "yes", "0"):
            with self.subTest(value=value):
                with patch("src.stages.stage_segmentation.run_inference_process") as run:
                    with self.assertRaisesRegex(ValueError, "disable_tta"):
                        SegmentationStage({"disable_tta": value})
                    run.assert_not_called()

    def test_none_slice_part_config_still_uses_default_part_rules(self):
        command = self.run_inference({"slice_part_config": None, "disable_tta": True})

        self.assertEqual(command[command.index("-d") + 1], "292")
        self.assertIn("--disable_tta", command)

    def test_inference_options_do_not_parse_unrelated_application_settings(self):
        command = self.run_inference({
            "logging": {"console_enabled": "not-a-boolean"},
            "pipeline": {"run_preflight": "not-a-boolean"},
            "nnunet": {
                "python_executable": False,
                "disable_tta": True,
                "step_size": 0.75,
            },
        })

        self.assertIn("--disable_tta", command)
        self.assertEqual(command[command.index("-step_size") + 1], "0.75")
