import json
import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig, LoggingConfig, PipelineConfig
from src.core.base_stage import BaseStage
from src.core.lifecycle import JobStatus, StageStatus
from src.core.pipeline import Pipeline
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum


class SuccessfulStage(BaseStage):
    def execute(self, context):
        context.set_result(self.name, {"ok": True})
        return context


class FailingStage(BaseStage):
    def execute(self, context):
        raise RuntimeError("boom")


class DebugArtifactStage(BaseStage):
    """模拟旧阶段在 metadata 目录写调试文件。"""

    def execute(self, context):
        metadata_dir = Path(context.request.output_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "调试文件.txt").write_text("debug", encoding="utf-8")
        context.set_result(self.name, {"ok": True})
        return context


def make_request(output_dir):
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
    return Pipeline(
        stages=stages,
        config=AppConfig(pipeline=PipelineConfig(run_preflight=False)),
    )


class PipelineFrameworkTest(unittest.TestCase):
    def test_debug_enabled_keeps_intermediate_artifacts_after_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[DebugArtifactStage("debug_stage")],
                config=AppConfig(
                    logging=LoggingConfig(debug_log_file_enabled=True),
                    pipeline=PipelineConfig(run_preflight=False),
                ),
            )

            response = pipeline.run(make_request(temp_dir))

            self.assertTrue(response.success)
            self.assertTrue(
                (Path(temp_dir) / "metadata" / "调试文件.txt").exists()
            )

    def test_successful_pipeline_writes_succeeded_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                stages=[SuccessfulStage("one"), SuccessfulStage("two")],
                config=AppConfig(
                    logging=LoggingConfig(debug_log_file_enabled=True),
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

    def test_non_debug_pipeline_does_not_write_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            response = make_pipeline([SuccessfulStage("one")]).run(
                make_request(temp_dir)
            )

            self.assertTrue(response.success)
            self.assertFalse((Path(temp_dir) / "manifest.json").exists())

    def test_failed_stage_marks_remaining_stages_skipped(self):
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
