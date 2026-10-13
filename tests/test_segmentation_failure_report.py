"""推理失败的业务消息与完整技术日志保持分离。"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import AppConfig, PipelineConfig
from src.core.base_stage import BaseStage
from src.core.errors import ExternalProcessError
from src.core.pipeline import Pipeline
from src.models import Request
from src.stages.stage_segmentation import SegmentationStage


class FakePreprocessingStage(BaseStage):
    """仅传递输入路径，不执行 DICOM 转换或 GPU 推理。"""

    def execute(self, context):
        context.set_result("preprocessing", {"nifti_path": "/tmp/raw_0000.nii.gz"})
        return context


class SegmentationFailureReportTest(unittest.TestCase):
    def setUp(self):
        self.error_log = self.enterContext(patch("src.stages.stage_segmentation.ErrorLog"))
        self.enterContext(patch("src.stages.stage_segmentation.DebugLog"))
        self.enterContext(patch("src.core.pipeline.ErrorLog"))
        self.enterContext(patch("src.core.pipeline.RunLog"))

    @staticmethod
    def missing_trace(model_id):
        return (
            'Traceback (most recent call last):\n  File "predict.py", line 48\n'
            f"RuntimeError: Could not find a dataset with the ID {model_id}. "
            "Make sure the requested dataset ID exists.\n"
            "Here are your currently defined folders:\n...\n"
            "If something is not right, adapt your environment variables.\n"
        )

    def test_missing_dataset_message_uses_actual_id_and_effective_paths(self):
        for model_id in (7, 298, 1234):
            with self.subTest(model_id=model_id):
                error = SegmentationStage._describe_inference_failure(
                    model_id, 1, self.missing_trace(model_id),
                    {"nnUNet_results": "/models/trained"},
                )
                self.assertIsInstance(error, ExternalProcessError)
                self.assertIn(f"未找到模型 {model_id}", str(error))
                self.assertIn(f"Dataset{model_id:03d}*", str(error))
                self.assertIn("nnUNet_results=/models/trained", str(error))
                self.assertIn("nnUNet_raw=未配置", str(error))
                self.assertNotIn("Traceback", str(error))
                self.assertEqual(error.details["reason"], "dataset_not_found")

    def test_other_failures_are_not_reported_as_missing_model(self):
        for stderr in (
            self.missing_trace(297),
            "Traceback (most recent call last):\nRuntimeError: CUDA out of memory",
            "FileNotFoundError: checkpoint_final.pth",
            "",
            "x" * 2000,
        ):
            with self.subTest(stderr=stderr[:60]):
                error = SegmentationStage._describe_inference_failure(298, 2, stderr, {})
                self.assertNotIn("未找到模型", str(error))
                self.assertNotIn("Traceback", str(error))
                self.assertIn("退出码 2", str(error))
                self.assertLess(len(str(error)), 400)
                self.assertEqual(error.details["reason"], "inference_failed")

    def test_inference_logs_full_output_and_uses_merged_environment(self):
        stage = SegmentationStage({"nnUNet_results": "/configured/results"})
        stderr = self.missing_trace(298)
        with patch.dict("os.environ", {
            "nnUNet_results": "/old/results", "nnUNet_raw": "/inherited/raw",
        }), patch("src.stages.stage_segmentation.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, "predict startup", stderr)
            with self.assertRaises(ExternalProcessError) as caught:
                stage._run_nnunet_inference(
                    Path("/tmp/input"), Path("/tmp/input/raw_0000.nii.gz"),
                    Path("/tmp/output/seg.nii.gz"), 298, "3d_fullres", "0", True,
                )
        message = str(caught.exception)
        self.assertIn("nnUNet_results=/configured/results", message)
        self.assertIn("nnUNet_raw=/inherited/raw", message)
        self.assertNotIn("/old/results", message)
        self.assertNotIn("Traceback", message)
        logged = "\n".join(str(call.args[0]) for call in self.error_log.call_args_list)
        self.assertIn(stderr, logged)
        self.assertIn("predict startup", logged)
        self.assertEqual(run.call_args.kwargs["env"]["nnUNet_results"], "/configured/results")

    def test_pipeline_writes_readable_missing_model_message_to_report(self):
        stage = SegmentationStage({"nnUNet_results": "/models/results"})
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[FakePreprocessingStage("preprocessing"), stage],
                config=AppConfig(pipeline=PipelineConfig(run_preflight=False)),
            )
            request = Request("/tmp/dicom", temp_dir, scan_thickness=5.0)
            with patch("src.stages.stage_segmentation.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 1, "", self.missing_trace(298))
                response = pipeline.run(request)
            report = json.loads((Path(temp_dir) / "report.json").read_text(encoding="utf-8"))
            self.assertFalse(response.success)
            self.assertFalse(report["Success"])
            self.assertEqual(report["PlaneList"], [])
            self.assertIn("未找到模型 298", report["Message"])
            self.assertIn("nnUNet_results=/models/results", report["Message"])
            self.assertNotIn("Traceback", report["Message"])

    def test_successful_inference_keeps_output_normalization(self):
        stage = SegmentationStage({"_fold": "0"})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "raw.nii.gz").write_bytes(b"fake segmentation")
            with patch("src.stages.stage_segmentation.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, "ok", "")
                stage._run_nnunet_inference(
                    root, root / "raw_0000.nii.gz", root / "seg.nii.gz",
                    298, "3d_fullres", "0", False,
                )
            self.assertEqual((root / "seg.nii.gz").read_bytes(), b"fake segmentation")
            self.error_log.assert_not_called()

    def test_output_normalization_rejects_multiple_unrelated_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("old_case_a.nii.gz", "old_case_b.nii.gz"):
                (root / name).write_bytes(name.encode())
            output = root / "seg.nii.gz"
            with self.assertRaisesRegex(RuntimeError, "多个候选结果"):
                SegmentationStage._normalize_output_file(root / "current_case.nii.gz", output)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(root.glob("*.nii.gz"))), 2)

    def test_output_normalization_keeps_known_output_with_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "seg.nii.gz"
            output.write_bytes(b"current segmentation")
            (root / "old_case.nii.gz").write_bytes(b"unrelated segmentation")
            SegmentationStage._normalize_output_file(root / "current_case.nii.gz", output)
            self.assertEqual(output.read_bytes(), b"current segmentation")
