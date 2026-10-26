"""Run independent tasks against one series cache without DICOM data or a GPU.

The real Pipeline, cache stages, image IO, logs, reports and cleanup run in fresh
interpreters. Only DICOM decoding, model inference and reconstruction are small
synthetic replacements. Barriers demonstrate actual overlap rather than timings.
"""

import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import traceback
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.config import AppConfig, LoggingConfig, PipelineConfig
from src.core.base_stage import BaseStage
from src.core.pipeline import Pipeline
from src.models import Request
from src.stages import stage_preprocessing as preprocessing
from src.stages.stage_preprocessing import PreprocessingStage
from src.stages.stage_segmentation import SegmentationStage
from src.utils import RunLog, configure_logging


pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux process-lock regression")
_WAIT_SECONDS = 20


def _synthetic_image():
    image = sitk.GetImageFromArray(np.arange(24, dtype=np.int16).reshape(2, 3, 4))
    image.SetSpacing((0.75, 0.75, 1.5))
    return image


class _ModelFreeSegmentation(SegmentationStage):
    def __init__(self, task_name, model_id, events, inference_barrier):
        super().__init__({"_continue": False})
        self.task_name = task_name
        self.model_id = model_id
        self.events = events
        self.inference_barrier = inference_barrier

    def _match_model(self, **kwargs):
        return {"model_id": self.model_id, "configuration": "3d_fullres", "fold": "0"}

    def _run_nnunet_inference(self, **kwargs):
        source = Path(kwargs["input_path"])
        output = Path(kwargs["output_path"])
        image = sitk.ReadImage(str(source))
        assert image.GetSize() == (4, 3, 2)
        self.events.put(("inference", self.task_name, self.model_id, str(source), str(output)))
        # Different models must both reach inference before either can finish.
        # A whole-series lock would make this barrier fail.
        if self.inference_barrier is not None:
            self.inference_barrier.wait(timeout=_WAIT_SECONDS)
        sitk.WriteImage(image > 10, str(output))


class _IndependentReconstruction(BaseStage):
    def __init__(self, task_name, events, barrier, *, fail, failure_finished):
        super().__init__("reconstruction")
        self.task_name = task_name
        self.events = events
        self.barrier = barrier
        self.fail = fail
        self.failure_finished = failure_finished

    def execute(self, context):
        root = Path(context.request.output_dir)
        shared_raw = Path(context.get_result("preprocessing")["nifti_path"])
        shared_seg = Path(context.get_result("segmentation")["seg_path"])
        # Stand in for request-specific plans/intermediates and delivered images.
        scratch = root / "segmentation" / "local-plan.json"
        scratch.parent.mkdir()
        scratch.write_text(self.task_name, encoding="utf-8")
        result_path = root / "reconstructed.json"
        payload = {
            "task": self.task_name,
            "model": context.metadata["model_id"],
            "nifti": str(shared_raw),
            "segmentation": str(shared_seg),
            "size": list(sitk.ReadImage(str(shared_seg)).GetSize()),
        }
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        RunLog(f"task={self.task_name} reconstruction entered")
        self.events.put(("reconstruction", self.task_name))
        # Same-model waiters must leave the cache transaction and both progress
        # into their own reconstruction before either task is allowed to finish.
        self.barrier.wait(timeout=_WAIT_SECONDS)
        if self.fail:
            raise RuntimeError(f"intentional reconstruction failure: {self.task_name}")
        if self.failure_finished is not None:
            assert self.failure_finished.wait(timeout=_WAIT_SECONDS), "failed sibling did not finish"
            assert shared_raw.is_file() and shared_seg.is_file()
            assert scratch.read_text(encoding="utf-8") == self.task_name
            assert json.loads(result_path.read_text(encoding="utf-8"))["task"] == self.task_name
        context.set_result(self.name, payload)
        RunLog(f"task={self.task_name} reconstruction complete")
        return context


def _pipeline_worker(root, task_name, model_id, events, start, downstream, *,
                     inference_barrier=None, fail=False, failure_finished=None):
    root = Path(root)
    output = root / "outputs" / task_name
    # Spawned children do not inherit pytest's logger patch. Bind every real
    # application log to this test-owned task output before running any stage.
    config = AppConfig(
        logging=LoggingConfig(
            debug_file_enabled=True,
            console_enabled=False,
            run_log_file_enabled=True,
            debug_log_file_enabled=True,
            error_log_file_enabled=True,
        ),
        pipeline=PipelineConfig(run_preflight=False, keep_intermediate_on_failure=False),
    )
    configure_logging(config.logging, output_dir=output)

    def read_series(_stage, _directory):
        events.put(("conversion", task_name))
        return {"image": _synthetic_image(), "series_file_names": ["synthetic.dcm"]}

    try:
        with (
            patch.object(PreprocessingStage, "_read_dicom_series", read_series),
            patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
        ):
            request = Request(
                dicom_dir=str(root / "same-series"),
                output_dir=str(output),
                cache_dir=str(root / "cache"),
                scan_thickness=1.5,
            )
            pipeline = Pipeline([
                PreprocessingStage(),
                _ModelFreeSegmentation(task_name, model_id, events, inference_barrier),
                _IndependentReconstruction(
                    task_name, events, downstream, fail=fail,
                    failure_finished=failure_finished,
                ),
            ], config=config)
            start.wait(timeout=_WAIT_SECONDS)
            response = pipeline.run(request)
            events.put(("done", task_name, response.success, response.context.job_id))
            if fail and failure_finished is not None:
                failure_finished.set()
    except BaseException:
        events.put(("error", task_name, traceback.format_exc()))


def _run_pair(root, model_ids, *, failed_task=None):
    mp = multiprocessing.get_context("spawn")
    events = mp.Queue()
    start, downstream = mp.Barrier(2), mp.Barrier(2)
    inference = mp.Barrier(2) if len(set(model_ids)) == 2 else None
    failure_finished = mp.Event() if failed_task is not None else None
    children = [mp.Process(
        target=_pipeline_worker,
        args=(str(root), name, model_id, events, start, downstream),
        kwargs={
            "inference_barrier": inference,
            "fail": name == failed_task,
            "failure_finished": failure_finished,
        },
    ) for name, model_id in zip(("task-a", "task-b"), model_ids)]
    try:
        # Import-time application messages contain Chinese. Give each child a
        # UTF-8 console while leaving its locale/default file encoding intact.
        with patch.dict(os.environ, {"PYTHONIOENCODING": "utf-8"}):
            for child in children:
                child.start()
        for child in children:
            child.join(timeout=_WAIT_SECONDS + 10)
            assert not child.is_alive(), f"pipeline worker {child.pid} did not finish"
            assert child.exitcode == 0
        messages = []
        while True:
            try:
                messages.append(events.get(timeout=0.1))
            except Empty:
                break
        assert not [message for message in messages if message[0] == "error"], messages
        assert len([message for message in messages if message[0] == "done"]) == 2, messages
        return messages
    finally:
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.kill()
                child.join(timeout=5)
        events.close()
        events.join_thread()


@pytest.mark.parametrize("model_ids", [(315, 315), (292, 315)], ids=["same-model", "different-models"])
def test_same_series_tasks_share_cache_and_reconstruct_in_independent_outputs(tmp_path, model_ids):
    messages = _run_pair(tmp_path, model_ids)
    assert sum(message[0] == "conversion" for message in messages) == 1
    inferences = [message for message in messages if message[0] == "inference"]
    assert len(inferences) == len(set(model_ids))
    assert {message[3] for message in inferences} == {str(tmp_path / "cache/nifti/raw_0000.nii.gz")}
    assert {message[1] for message in messages if message[0] == "reconstruction"} == {"task-a", "task-b"}
    completions = [message for message in messages if message[0] == "done"]
    assert all(message[2] is True for message in completions), messages
    assert len({message[3] for message in completions}) == 2, "each task needs its own job identity"

    for task, model_id in zip(("task-a", "task-b"), model_ids):
        output = tmp_path / "outputs" / task
        result = json.loads((output / "reconstructed.json").read_text(encoding="utf-8"))
        assert result["task"] == task and result["model"] == str(model_id)
        assert result["size"] == [4, 3, 2]
        assert json.loads((output / "report.json").read_text(encoding="utf-8"))["Success"] is True
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "succeeded"
        assert all(stage["status"] == "succeeded" for stage in manifest["stages"])
        log = (output / "log/RunLog.txt").read_text(encoding="utf-8")
        assert f"task={task} reconstruction complete" in log
        other_task = "task-b" if task == "task-a" else "task-a"
        assert f"task={other_task} " not in log, "task logs must not cross into sibling output"
        segmentation = tmp_path / "cache" / str(model_id) / "segmentation"
        assert sitk.ReadImage(str(segmentation / "seg.nii.gz")).GetSize() == (4, 3, 2)
        assert json.loads((segmentation / "model.json").read_text(encoding="utf-8")) == {"model_id": str(model_id)}
    assert not list((tmp_path / "cache").rglob(".segmentation-*"))
    assert not list((tmp_path / "cache/.work").iterdir())


def test_failed_same_series_task_cleanup_preserves_running_sibling_and_shared_cache(tmp_path):
    messages = _run_pair(tmp_path, (315, 315), failed_task="task-a")
    assert {message[1]: message[2] for message in messages if message[0] == "done"} == {
        "task-a": False, "task-b": True,
    }, messages
    assert sum(message[0] == "conversion" for message in messages) == 1
    assert sum(message[0] == "inference" for message in messages) == 1
    failed, sibling = tmp_path / "outputs/task-a", tmp_path / "outputs/task-b"
    assert not (failed / "segmentation").exists(), "failed task intermediate cleanup must run"
    assert (sibling / "segmentation/local-plan.json").read_text(encoding="utf-8") == "task-b"
    assert json.loads((sibling / "reconstructed.json").read_text(encoding="utf-8"))["task"] == "task-b"
    assert json.loads((failed / "report.json").read_text(encoding="utf-8"))["Success"] is False
    assert json.loads((sibling / "report.json").read_text(encoding="utf-8"))["Success"] is True
    assert json.loads((failed / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads((sibling / "manifest.json").read_text(encoding="utf-8"))["status"] == "succeeded"
    assert "intentional reconstruction failure: task-a" in (failed / "log/ErrorLog.txt").read_text(encoding="utf-8")
    assert "task=task-b reconstruction complete" in (sibling / "log/RunLog.txt").read_text(encoding="utf-8")
    assert not (sibling / "log/ErrorLog.txt").exists()
    assert sitk.ReadImage(str(tmp_path / "cache/nifti/raw_0000.nii.gz")).GetSize() == (4, 3, 2)
    assert sitk.ReadImage(str(tmp_path / "cache/315/segmentation/seg.nii.gz")).GetSize() == (4, 3, 2)
