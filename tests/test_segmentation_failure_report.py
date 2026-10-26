"""推理失败的业务消息与完整技术日志保持分离。"""

import json
import shlex
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
    WORKER_FAILURE = (
        "RuntimeError: Background workers died. Look for the error message further up! "
        "If there is none then your RAM was full and the worker was killed by the OS. "
        "Use fewer workers or get more RAM in that case!"
    )

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

    def test_worker_failure_keeps_specific_error_before_generic_tail(self):
        for cause in (
            "FileNotFoundError: [Errno 2] No such file or directory: 'raw_0000.nii.gz'",
            "MemoryError: unable to allocate preprocessing buffer",
            "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 8.00 GiB",
            "RuntimeError: The size of tensor a (512) must match tensor b (256)",
        ):
            with self.subTest(cause=cause):
                stderr = (
                    'Traceback (most recent call last):\n  File "worker.py", line 42\n'
                    f"{cause}\n\n"
                    'Traceback (most recent call last):\n  File "predict.py", line 48\n'
                    f"{self.WORKER_FAILURE}\n"
                )
                error = SegmentationStage._describe_inference_failure(292, 1, stderr, {})
                self.assertIn(cause, str(error))
                self.assertNotIn("Background workers died", str(error))
                self.assertNotIn("RAM was full", str(error))
                self.assertNotIn("Traceback", str(error))
                self.assertEqual(error.details["reason"], "inference_failed")

    def test_normal_stdout_progress_does_not_hide_stderr_failure(self):
        for stderr, cause in (
            ("Traceback (most recent call last):\nRuntimeError: CUDA out of memory",
             "RuntimeError: CUDA out of memory"),
            ("Native allocation failed", "Native allocation failed"),
        ):
            with self.subTest(stderr=stderr):
                error = SegmentationStage._describe_inference_failure(
                    292, 1, stderr, {}, stdout="using device: cuda\nPredicting raw\n",
                )
                self.assertIn(cause, str(error))
                self.assertNotIn("Predicting raw", str(error))

    def test_nonworker_failure_uses_final_stderr_exception(self):
        stderr = (
            "ValueError: initial validation failed\n"
            "During handling of the above exception, another exception occurred:\n"
            "RuntimeError: final prediction failure\n"
            "cleanup finished\n"
        )
        error = SegmentationStage._describe_inference_failure(
            292, 1, stderr, {}, stdout="ValueError: earlier stdout diagnostic",
        )
        self.assertIn("RuntimeError: final prediction failure", str(error))
        self.assertNotIn("initial validation", str(error))
        self.assertNotIn("earlier stdout", str(error))

    def test_nonworker_failure_finds_specific_exception_in_stdout(self):
        cause = "FileNotFoundError: checkpoint_final.pth"
        error = SegmentationStage._describe_inference_failure(
            292, 1, "progress bar finished", {}, stdout=f"{cause}\ncleanup finished",
        )
        self.assertIn(cause, str(error))

    def test_worker_failure_finds_specific_exception_in_stdout(self):
        cause = "ValueError: Input image and segmentation must have the same shape"
        stdout = (
            "preprocessing case raw\n"
            'Traceback (most recent call last):\n  File "worker.py", line 42\n'
            f"{cause}\n"
        )
        error = SegmentationStage._describe_inference_failure(
            292, 1, self.WORKER_FAILURE, {}, stdout=stdout,
        )
        self.assertIn(cause, str(error))
        self.assertNotIn("Background workers died", str(error))
        self.assertEqual(error.details["reason"], "inference_failed")

    def test_worker_only_failure_reports_possible_causes_without_asserting_oom(self):
        error = SegmentationStage._describe_inference_failure(
            292, 1, self.WORKER_FAILURE, {},
        )
        message = str(error)
        for expected in ("模型 292", "退出码 1", "后台", "进程", "异常退出", "可能", "内存"):
            self.assertIn(expected, message)
        for unexpected in ("Background workers died", "RAM was full", "内存已满", "内存耗尽"):
            self.assertNotIn(unexpected, message)
        self.assertEqual(error.details["reason"], "background_worker_failed")

    def test_secondary_pipe_errors_do_not_hide_recoverable_worker_failure(self):
        for secondary in ("EOFError", "BrokenPipeError: [Errno 32] Broken pipe",
                          "ConnectionResetError: connection reset"):
            error = SegmentationStage._describe_inference_failure(
                292, 1, f"{secondary}\n{self.WORKER_FAILURE}", {},
            )
            self.assertEqual(error.details["reason"], "background_worker_failed")

    def test_specific_stdout_failure_takes_priority_over_secondary_pipe_error(self):
        cause = "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 8.00 GiB"
        for secondary in ("EOFError", "BrokenPipeError: [Errno 32] Broken pipe"):
            with self.subTest(secondary=secondary):
                error = SegmentationStage._describe_inference_failure(
                    292, 1, f"{secondary}\n{self.WORKER_FAILURE}", {}, stdout=cause,
                )
                self.assertIn(cause, str(error))
                self.assertNotIn(secondary, str(error))

    def test_missing_dataset_in_stdout_still_uses_specific_model_diagnosis(self):
        error = SegmentationStage._describe_inference_failure(
            292, 1, self.WORKER_FAILURE, {"nnUNet_results": "/models/results"},
            stdout=self.missing_trace(292),
        )
        self.assertIn("未找到模型 292", str(error))
        self.assertIn("nnUNet_results=/models/results", str(error))
        self.assertNotIn("Background workers died", str(error))
        self.assertEqual(error.details["reason"], "dataset_not_found")

    def test_inference_logs_full_output_and_uses_merged_environment(self):
        stage = SegmentationStage({"nnUNet_results": "/configured/results"})
        stderr = self.missing_trace(298)
        with patch.dict("os.environ", {
            "nnUNet_results": "/old/results", "nnUNet_raw": "/inherited/raw",
        }), patch("src.stages.stage_segmentation.run_inference_process") as run:
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
        command = run.call_args.args[0]
        self.assertIn("-npp", command)
        self.assertIn("-nps", command)
        self.assertIn(shlex.join(command), logged)
        self.assertEqual(run.call_args.kwargs["env"]["nnUNet_results"], "/configured/results")

    def test_inference_logs_both_raw_streams_and_reports_stdout_cause(self):
        stage = SegmentationStage({"nnUNet_results": "/configured/results"})
        cause = "FileNotFoundError: [Errno 2] Missing input: '/tmp/input case/raw_0000.nii.gz'"
        stdout = f'preprocessing started\nTraceback (most recent call last):\n{cause}\n'
        stderr = f"progress: 1/1\n{self.WORKER_FAILURE}\n"
        with patch("src.stages.stage_segmentation.run_inference_process") as run:
            run.return_value = subprocess.CompletedProcess([], 1, stdout, stderr)
            with self.assertRaises(ExternalProcessError) as caught:
                stage._run_nnunet_inference(
                    Path("/tmp/input case"), Path("/tmp/input case/raw_0000.nii.gz"),
                    Path("/tmp/output case/seg.nii.gz"), 292, "3d_fullres", "0", False,
                )
        self.assertIn(cause, str(caught.exception))
        self.assertNotIn("Background workers died", str(caught.exception))
        logged = "\n".join(str(call.args[0]) for call in self.error_log.call_args_list)
        self.assertIn(stdout, logged)
        self.assertIn(stderr, logged)
        command = run.call_args.args[0]
        self.assertIn("-npp", command)
        self.assertIn("-nps", command)
        self.assertIn(shlex.join(command), logged)

    def test_pipeline_writes_readable_missing_model_message_to_report(self):
        stage = SegmentationStage({"nnUNet_results": "/models/results"})
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[FakePreprocessingStage("preprocessing"), stage],
                config=AppConfig(pipeline=PipelineConfig(run_preflight=False)),
            )
            request = Request("/tmp/dicom", temp_dir, scan_thickness=5.0)
            with patch("src.stages.stage_segmentation.run_inference_process") as run:
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
            with patch("src.stages.stage_segmentation.run_inference_process") as run:
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
