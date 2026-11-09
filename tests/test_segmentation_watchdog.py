"""Recover failed inference with fresh outputs; never restart an unclean process tree."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from src.config import AppConfig, PipelineConfig
from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.core.errors import ExternalProcessError
from src.core.inference_process import InferenceProcessAborted
from src.core.pipeline import Pipeline
from src.models import Request, SlicePartEnum
from src.stages.stage_segmentation import SegmentationStage


@pytest.fixture(autouse=True)
def isolate_model_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "nnUNet_results"))


def _context(root):
    source = root / "nifti" / "raw_0000.nii.gz"
    source.parent.mkdir()
    source.touch()
    request = Request(
        dicom_dir=str(root / "dicom"), output_dir=str(root / "output"),
        cache_dir=str(root / "cache"), slice_part=SlicePartEnum.颈椎, scan_thickness=1.0,
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {"nifti_path": str(source)})
    return context


def _aborted(reason="timeout", cleanup_succeeded=True):
    return InferenceProcessAborted(
        reason=reason, timeout_seconds=12,
        stdout="partial progress", stderr="fatal worker detail",
        cleanup_succeeded=cleanup_succeeded,
    )


@pytest.mark.parametrize("failure", ["timeout", "native_worker_crash", "signal", "worker"])
def test_retry_gets_clean_workspace_and_publishes_only_complete_result(tmp_path, failure):
    context = _context(tmp_path)
    stage = SegmentationStage({"inference_timeout_seconds_by_model": {"292": 12}})
    workspaces = []
    log_paths = []

    def infer(command, **options):
        work = Path(command[command.index("-o") + 1])
        workspaces.append(work)
        assert not (work / "raw.nii.gz").exists()
        assert options["timeout_seconds"] == 12
        assert options["env"]["PYTHONFAULTHANDLER"] == "1"
        assert options["env"]["PYTHONUNBUFFERED"] == "1"
        for stream in ("stdout", "stderr"):
            path = options[f"{stream}_path"]
            assert path.parent == tmp_path / "output" / "log"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"attempt {len(workspaces)} {stream}", encoding="utf-8")
            log_paths.append(path)
        output = work / "raw.nii.gz"
        if len(workspaces) == 1:
            output.write_bytes(b"incomplete")
            if failure in {"timeout", "native_worker_crash"}:
                raise _aborted(failure)
            if failure == "signal":
                return subprocess.CompletedProcess(command, -11, "", "native crash")
            return subprocess.CompletedProcess(command, 1, "", "RuntimeError: Background workers died")
        output.write_bytes(b"complete segmentation")
        return subprocess.CompletedProcess(command, 0, "done", "")

    with patch("src.stages.stage_segmentation.run_inference_process", side_effect=infer) as run:
        stage.execute(context)
    assert run.call_count == 2
    assert workspaces[0] != workspaces[1]
    assert not any(work.exists() for work in workspaces)
    result = Path(context.get_result("segmentation")["seg_path"])
    assert result.read_bytes() == b"complete segmentation"
    assert all(path.is_file() for path in log_paths)


@pytest.mark.parametrize("cleanup_succeeded", [True, False])
def test_terminal_failure_never_publishes_partial_segmentation(tmp_path, cleanup_succeeded):
    context = _context(tmp_path)
    stage = SegmentationStage({"inference_max_retries": 1})
    with patch(
        "src.stages.stage_segmentation.run_inference_process",
        side_effect=_aborted(cleanup_succeeded=cleanup_succeeded),
    ) as run:
        with pytest.raises(ExternalProcessError) as caught:
            stage.execute(context)
    assert run.call_count == (2 if cleanup_succeeded else 1)
    assert caught.value.details["cleanup_succeeded"] is cleanup_succeeded
    assert not list((tmp_path / "cache").rglob("seg.nii.gz"))
    assert not list((tmp_path / "cache").rglob("model.json"))


def test_missing_model_is_not_retried_and_disabled_retry_is_respected(tmp_path):
    for index, retries in enumerate((0, 1)):
        root = tmp_path / str(index)
        root.mkdir()
        context = _context(root)
        stage = SegmentationStage({"inference_max_retries": retries})
        with patch("src.stages.stage_segmentation.run_inference_process") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 1, "", "RuntimeError: Could not find a dataset with the ID 292",
            )
            with pytest.raises(ExternalProcessError) as caught:
                stage.execute(context)
        assert caught.value.details["reason"] == "dataset_not_found"
        assert run.call_count == 1
    stage = SegmentationStage({"inference_max_retries": 0})
    with patch("src.stages.stage_segmentation.run_inference_process", side_effect=_aborted()) as run:
        with pytest.raises(ExternalProcessError):
            stage.execute(context)
    assert run.call_count == 1


def test_model_timeout_override_does_not_change_other_model_limit(tmp_path):
    stage = SegmentationStage({
        "inference_timeout_seconds": 900,
        "inference_timeout_seconds_by_model": {"315": 180},
    })
    with (
        patch("src.stages.stage_segmentation.run_inference_process") as run,
        patch.object(stage, "_normalize_output_file"),
    ):
        run.return_value = subprocess.CompletedProcess([], 0, "", "")
        for model_id, timeout in ((315, 180), (292, 900)):
            stage._run_nnunet_inference(
                tmp_path, tmp_path / "raw_0000.nii.gz", tmp_path / "seg.nii.gz",
                model_id, "3d_fullres", "0", True,
            )
            assert run.call_args.kwargs["timeout_seconds"] == timeout


def test_exhausted_timeout_is_written_to_failure_report(tmp_path):
    class Preprocessing(BaseStage):
        def execute(self, context):
            context.set_result("preprocessing", {"nifti_path": str(tmp_path / "raw_0000.nii.gz")})
            return context

    pipeline = Pipeline(
        stages=[Preprocessing("preprocessing"), SegmentationStage({"inference_max_retries": 1})],
        config=AppConfig(pipeline=PipelineConfig(run_preflight=False)),
    )
    request = Request(
        "/tmp/fake-dicom", str(tmp_path / "output"),
        slice_part=SlicePartEnum.颈椎, scan_thickness=1,
    )
    with patch("src.stages.stage_segmentation.run_inference_process", side_effect=_aborted()) as run:
        response = pipeline.run(request)
    assert run.call_count == 2
    assert not response.success
    report = json.loads((tmp_path / "output" / "report.json").read_text(encoding="utf-8"))
    assert report["Success"] is False
    assert report["PlaneList"] == []
    assert "推理时限" in report["Message"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX verified process-group retry")
def test_real_stub_worker_crash_recovers_through_stage_and_preserves_logs(tmp_path):
    """Exercise stage, launcher, faulthandler and cache publication together without a model."""
    context = _context(tmp_path)
    script = tmp_path / "fake_predict.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "source = pathlib.Path(sys.argv[sys.argv.index('-i') + 1])\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "attempt_file = source / 'attempts'\n"
        "attempt = int(attempt_file.read_text()) + 1 if attempt_file.exists() else 1\n"
        "attempt_file.write_text(str(attempt))\n"
        "if attempt == 1:\n"
        "    (output / 'raw.nii.gz').write_bytes(b'partial')\n"
        "    sys.stderr.write('progress 50%'); sys.stderr.flush()\n"
        "    worker = 'import os, resource, signal; resource.setrlimit(resource.RLIMIT_CORE, (0,0)); os.kill(os.getpid(), signal.SIGSEGV)'\n"
        "    subprocess.Popen([sys.executable, '-c', worker])\n"
        "    time.sleep(60)\n"
        "else:\n"
        "    assert not (output / 'raw.nii.gz').exists()\n"
        "    (output / 'raw.nii.gz').write_bytes(b'complete')\n"
        "    print('second attempt completed')\n",
        encoding="utf-8",
    )
    stage = SegmentationStage({
        "nnUNet_predict_command": str(script), "nnUNet_python": sys.executable,
        "inference_timeout_seconds": 10, "inference_max_retries": 1,
    })
    started = time.monotonic()
    stage.execute(context)
    assert time.monotonic() - started < 8
    assert Path(context.get_result("segmentation")["seg_path"]).read_bytes() == b"complete"
    logs = tmp_path / "output" / "log"
    assert len(list(logs.glob("*.stderr.log"))) == 2
    assert "Fatal Python error: Segmentation fault" in next(
        logs.glob("*attempt-1.stderr.log")
    ).read_text(encoding="utf-8")
    assert "second attempt completed" in next(
        logs.glob("*attempt-2.stdout.log")
    ).read_text(encoding="utf-8")
