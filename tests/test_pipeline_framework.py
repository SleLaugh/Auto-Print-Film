import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import AppConfig, LoggingConfig, PipelineConfig
from src.core.base_stage import BaseStage
from src.core.lifecycle import JobStatus, StageStatus
from src.core.pipeline import Pipeline
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum


class SuccessfulStage(BaseStage):
    """汇总 `SuccessfulStage` 相关测试场景。"""

    def execute(self, context):
        """为测试准备 `execute` 所需的数据或记录行为。"""

        context.set_result(self.name, {"ok": True})
        return context


class FailingStage(BaseStage):
    """汇总 `FailingStage` 相关测试场景。"""

    def execute(self, context):
        """为测试准备 `execute` 所需的数据或记录行为。"""

        raise RuntimeError("boom")


class InterruptingStage(BaseStage):
    """模拟用户中断或进程管理器发送的可捕获中止。"""

    def execute(self, context):
        """抛出 KeyboardInterrupt，验证流水线最外层兜底。"""

        raise KeyboardInterrupt()


class DebugArtifactStage(BaseStage):
    """模拟业务阶段直接在 metadata 目录写调试文件。"""

    def execute(self, context):
        """为测试准备 `execute` 所需的数据或记录行为。"""

        metadata_dir = Path(context.request.output_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "调试文件.txt").write_text("debug", encoding="utf-8")
        context.set_result(self.name, {"ok": True})
        return context


class FailingPreflight:
    """模拟输出目录清理前发生的预检失败。"""

    @staticmethod
    def validate(request, config):
        """抛出固定错误，验证早期失败报告。"""

        raise RuntimeError("preflight boom")


def make_request(output_dir):
    """为测试准备 `make_request` 所需的数据或记录行为。"""

    return Request(
        dicom_dir="/tmp/dicom",
        output_dir=output_dir,
        plane_configs=[PlaneConfig(
            PlaneTypeEnum.横状位,
            1.0,
            1.0,
            ["骨窗"],
        )],
    )


def make_pipeline(stages):
    """为测试准备 `make_pipeline` 所需的数据或记录行为。"""

    return Pipeline(
        stages=stages,
        config=AppConfig(pipeline=PipelineConfig(run_preflight=False)),
    )


class PipelineFrameworkTest(unittest.TestCase):
    """汇总 `PipelineFrameworkTest` 相关测试场景。"""

    def setUp(self):
        """隔离框架日志，测试中的 `one/boom` 不得写入真实日志目录。"""

        self.run_log = self.enterContext(patch("src.core.pipeline.RunLog"))
        self.error_log = self.enterContext(patch("src.core.pipeline.ErrorLog"))

    def test_default_pipeline_uses_eight_responsibility_stages(self):
        """默认主链应把旧定位阶段拆成校正、定位和解剖分析。"""

        pipeline = Pipeline(
            config=AppConfig(pipeline=PipelineConfig(run_preflight=False))
        )

        self.assertEqual(
            [stage.name for stage in pipeline.stages],
            [
                "preprocessing",
                "segmentation",
                "spatial_correction",
                "localization",
                "anatomical_analysis",
                "planning",
                "reconstruction",
                "reporting",
            ],
        )

    def test_debug_enabled_keeps_intermediate_artifacts_after_success(self):
        """调试文件开关应保留成功任务的中间诊断产物。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[DebugArtifactStage("debug_stage")],
                config=AppConfig(
                    logging=LoggingConfig(debug_file_enabled=True),
                    pipeline=PipelineConfig(run_preflight=False),
                ),
            )

            response = pipeline.run(make_request(temp_dir))

            self.assertTrue(response.success)
            self.assertTrue(
                (Path(temp_dir) / "metadata" / "调试文件.txt").exists()
            )

    def test_debug_log_alone_does_not_keep_debug_artifacts_or_manifest(self):
        """只开启 DebugLog.txt 时不得生成或保留调试产物。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[DebugArtifactStage("debug_stage")],
                config=AppConfig(
                    logging=LoggingConfig(
                        debug_file_enabled=False,
                        debug_log_file_enabled=True,
                    ),
                    pipeline=PipelineConfig(run_preflight=False),
                ),
            )

            response = pipeline.run(make_request(temp_dir))

            self.assertTrue(response.success)
            self.assertFalse((Path(temp_dir) / "metadata").exists())
            self.assertFalse((Path(temp_dir) / "manifest.json").exists())

    def test_successful_pipeline_writes_succeeded_manifest(self):
        """验证 `test_successful_pipeline_writes_succeeded_manifest` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[SuccessfulStage("one"), SuccessfulStage("two")],
                config=AppConfig(
                    logging=LoggingConfig(debug_file_enabled=True),
                    pipeline=PipelineConfig(run_preflight=False),
                ),
            )

            response = pipeline.run(make_request(temp_dir))

            manifest = json.loads(
                (Path(temp_dir) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(response.success)
            self.assertIs(response.context.job_status, JobStatus.SUCCEEDED)
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(
                [item["status"] for item in manifest["stages"]],
                ["succeeded", "succeeded"],
            )

    def test_pipeline_logs_each_stage_lifecycle(self):
        """控制台和调试日志应明确记录阶段序号、稳定名称和完成状态。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([
                SuccessfulStage("one"),
                SuccessfulStage("two"),
            ]).run(make_request(temp_dir))

        self.assertTrue(response.success)
        messages = [call.args[0] for call in self.run_log.call_args_list]
        self.assertTrue(any(
            message.startswith("[资源] stage=one phase=start pid=")
            for message in messages
        ))
        self.assertTrue(any(
            message.startswith("[资源] stage=two phase=complete pid=")
            for message in messages
        ))
        self.assertIn("Pipeline 开始执行，共 2 个阶段", messages)
        self.assertIn(
            "[Pipeline] 阶段 1/2 开始: one（自定义阶段）",
            messages,
        )
        self.assertTrue(any(
            message.startswith(
                "[Pipeline] 阶段 1/2 完成: one（自定义阶段），耗时 "
            )
            for message in messages
        ))
        self.assertIn(
            "[Pipeline] 阶段 2/2 开始: two（自定义阶段）",
            messages,
        )
        self.assertTrue(any(
            message.startswith(
                "[Pipeline] 阶段 2/2 完成: two（自定义阶段），耗时 "
            )
            for message in messages
        ))

    def test_failed_stage_log_contains_context(self):
        """失败日志必须包含序号、阶段标识、职责和错误原因。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([
                FailingStage("anatomical_analysis"),
                SuccessfulStage("planning"),
            ]).run(make_request(temp_dir))

        self.assertFalse(response.success)
        message, error = self.error_log.call_args.args
        self.assertIn("阶段 1/2 执行失败", message)
        self.assertIn("阶段标识: anatomical_analysis", message)
        self.assertIn("阶段职责: 解剖分析", message)
        self.assertIn("错误原因: boom", message)
        self.assertIsInstance(error, RuntimeError)

    def test_failed_stage_always_writes_failure_report(self):
        """任一业务阶段失败都必须发布外部可轮询的最终报告。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([
                FailingStage("anatomical_analysis"),
                SuccessfulStage("planning"),
            ]).run(make_request(temp_dir))

            report = json.loads(
                (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertFalse(response.success)
        self.assertFalse(report["Success"])
        self.assertEqual(report["Message"], "boom")
        self.assertEqual(report["SlicePart"], "腰椎")
        self.assertEqual(report["PlaneList"], [])

    def test_preflight_failure_writes_failure_report_before_pipeline_starts(self):
        """预检异常发生在上下文创建前，也必须发布失败报告。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[SuccessfulStage("one")],
                config=AppConfig(
                    pipeline=PipelineConfig(run_preflight=True),
                ),
                preflight=FailingPreflight(),
            )

            with self.assertRaisesRegex(RuntimeError, "preflight boom"):
                pipeline.run(make_request(temp_dir))

            report = json.loads(
                (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertFalse(report["Success"])
        self.assertEqual(report["Message"], "preflight boom")
        self.assertEqual(report["PlaneList"], [])

    def test_keyboard_interrupt_also_writes_failure_report(self):
        """不属于普通 Exception 的可捕获中止也必须形成报告。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = make_pipeline([InterruptingStage("one")])

            with self.assertRaises(KeyboardInterrupt):
                pipeline.run(make_request(temp_dir))

            report = json.loads(
                (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertFalse(report["Success"])
        self.assertEqual(report["Message"], "任务被中断")
        self.assertEqual(report["PlaneList"], [])

    def test_successful_custom_pipeline_writes_minimal_success_report(self):
        """没有第八阶段的自定义流水线也必须形成成功状态报告。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([SuccessfulStage("one")]).run(
                make_request(temp_dir)
            )
            report = json.loads(
                (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertTrue(response.success)
        self.assertTrue(report["Success"])
        self.assertEqual(report["Message"], "")
        self.assertEqual(report["PlaneList"], [])

    def test_non_debug_pipeline_does_not_write_manifest(self):
        """验证 `test_non_debug_pipeline_does_not_write_manifest` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([SuccessfulStage("one")]).run(
                make_request(temp_dir)
            )

            self.assertTrue(response.success)
            self.assertFalse((Path(temp_dir) / "manifest.json").exists())

    def test_failed_stage_marks_remaining_stages_skipped(self):
        """验证 `test_failed_stage_marks_remaining_stages_skipped` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([
                FailingStage("one"),
                SuccessfulStage("two"),
            ]).run(make_request(temp_dir))

            self.assertFalse(response.success)
            self.assertEqual(response.status, "failed")
            self.assertIs(
                response.context.stage_records["one"].status,
                StageStatus.FAILED,
            )
            self.assertIs(
                response.context.stage_records["two"].status,
                StageStatus.SKIPPED,
            )

    def test_response_keeps_legacy_fields_and_adds_job_fields(self):
        """验证 `test_response_keeps_legacy_fields_and_adds_job_fields` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([SuccessfulStage("one")]).run(
                make_request(temp_dir)
            )

            data = response.to_dict()

            self.assertIn("axial_count", data)
            self.assertEqual(data["job_id"], response.context.job_id)
            self.assertEqual(data["status"], "succeeded")
            self.assertEqual(data["stages"]["one"], "succeeded")


if __name__ == "__main__":
    unittest.main()
