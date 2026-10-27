"""Capacity instrumentation must be optional and preserve actual task outcomes."""

import json
from pathlib import Path
import subprocess
from unittest.mock import Mock, patch

import pytest

from src import capacity
from src.config import AppConfig, LoggingConfig, PipelineConfig
from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.core.pipeline import Pipeline
from src.models import Request
from src.stages.stage_segmentation import SegmentationStage


def request_for(root):
    return Request(
        dicom_dir=str(root / "private-input"), output_dir=str(root / "output"),
        cache_dir=str(root / "private-cache"), scan_thickness=1,
    )


def records(request):
    path = Path(request.output_dir) / "log/DebugLog.txt"
    return [json.loads(line[len("[Capacity] "):])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("[Capacity] ")]


def test_disabled_capacity_does_not_construct_sampler_or_touch_files(tmp_path):
    request = request_for(tmp_path)
    with patch.object(capacity, "CapacitySession") as session:
        with capacity.capacity_session(request, enabled=False):
            capacity.record_capacity_details(model_id="292")
            with capacity.capacity_inference(model_id="292", attempt=1):
                assert not capacity.capacity_enabled()
        session.assert_not_called()
    assert not Path(request.output_dir).exists()


def test_repeated_runs_have_distinct_ids_and_do_not_log_private_paths(tmp_path):
    request = request_for(tmp_path)
    for _ in range(2):
        with capacity.capacity_session(request, enabled=True) as session:
            capacity.record_capacity_details(model_id="292", segmentation_mode="direct_inference")
            with capacity.capacity_inference(model_id="292", attempt=1) as result:
                result["outcome"] = "succeeded"
            session.status = "succeeded"
    data = records(request)
    assert [row["event"] for row in data] == [
        "task_start", "inference_start", "inference_end", "task_end",
    ] * 2
    assert data[0]["run_id"] != data[4]["run_id"]
    assert data[2]["outcome"] == "succeeded"
    assert data[3]["duration_seconds"] >= 0
    assert data[3]["series_key"] == capacity.sequence_key(request.cache_dir)
    assert "private-input" not in json.dumps(data)
    assert "private-cache" not in json.dumps(data)
    assert not (Path(request.output_dir) / "log/RunLog.txt").exists()


def test_logging_failure_preserves_processing_exception_and_resets_context(tmp_path):
    original = RuntimeError("processing failed")
    with pytest.raises(RuntimeError) as caught:
        with capacity.capacity_session(request_for(tmp_path), enabled=True) as session:
            session.stream.close()
            session.stream = Mock()
            session.stream.write.side_effect = OSError("disk full")
            raise original
    assert caught.value is original
    assert not capacity.capacity_enabled()


class SummaryStage(BaseStage):
    def __init__(self, fail):
        super().__init__("summary_stage")
        self.fail = fail

    def execute(self, context):
        if self.fail:
            raise ValueError("expected failure")
        return context


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_pipeline_summary_follows_debug_flag_and_report_result(tmp_path, enabled, fail):
    request = request_for(tmp_path)
    config = AppConfig(
        logging=LoggingConfig(debug_log_file_enabled=enabled),
        pipeline=PipelineConfig(run_preflight=False),
    )
    with patch("src.core.pipeline.RunLog"), patch("src.core.pipeline.DebugLog"), \
            patch("src.core.pipeline.ErrorLog"):
        result = Pipeline(stages=[SummaryStage(fail)], config=config).run(request)
    assert result.success is not fail
    path = Path(request.output_dir) / "log/DebugLog.txt"
    assert path.exists() is enabled
    if enabled:
        end = records(request)[-1]
        assert end["status"] == ("failed" if fail else "succeeded")
        assert end["stage_seconds"]["summary_stage"] >= 0


@pytest.mark.parametrize("reuse,mode", [
    ([True], "cache_hit"), ([False, True], "wait_reuse"),
    ([False, False], "direct_inference"),
])
def test_segmentation_summary_distinguishes_cache_wait_from_real_inference(tmp_path, reuse, mode):
    request = request_for(tmp_path)
    context = ProcessingContext(request)
    context.set_result("preprocessing", {"nifti_path": str(tmp_path / "raw_0000.nii.gz")})
    stage = SegmentationStage()
    with capacity.capacity_session(request, enabled=True), \
            patch.object(stage, "_can_reuse_segmentation", side_effect=reuse), \
            patch.object(stage, "_generate_segmentation") as generate, \
            patch("src.stages.stage_segmentation.DebugLog"):
        stage.execute(context)
    end = records(request)[-1]
    assert end["segmentation_mode"] == mode
    assert end["cache_wait_seconds"] >= 0
    assert generate.call_count == (mode == "direct_inference")


@pytest.mark.parametrize("returncode", [0, 1])
def test_inference_events_surround_only_subprocess_execution(tmp_path, returncode):
    request = request_for(tmp_path)
    stage = SegmentationStage({"export_resampling_threads": 1})
    def run(command, **kwargs):
        assert records(request)[-1]["event"] == "inference_start"
        return subprocess.CompletedProcess(command, returncode, "", "expected test failure")
    with capacity.capacity_session(request, enabled=True), \
            patch("src.stages.stage_segmentation.run_inference_process", side_effect=run), \
            patch.object(stage, "_normalize_output_file"), \
            patch("src.stages.stage_segmentation.RunLog"), \
            patch("src.stages.stage_segmentation.ErrorLog"), \
            patch("src.stages.stage_segmentation.DebugLog"):
        if returncode:
            with pytest.raises(Exception):
                stage._run_nnunet_inference(
                    tmp_path, tmp_path / "raw_0000.nii.gz", tmp_path / "seg.nii.gz",
                    "292", "3d_fullres", "0", True,
                )
        else:
            stage._run_nnunet_inference(
                tmp_path, tmp_path / "raw_0000.nii.gz", tmp_path / "seg.nii.gz",
                "292", "3d_fullres", "0", True,
            )
    end = next(row for row in records(request) if row["event"] == "inference_end")
    assert end["outcome"] == ("failed" if returncode else "succeeded")
