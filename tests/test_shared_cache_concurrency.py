"""Exercise shared segmentation caches with real processes and fake inference.

Only test-owned files are used. Payloads stand in for NIfTI files because this
stage's cache transaction does not parse their contents; no models are imported
or invoked. Spawn avoids relying on inherited monkeypatches or file descriptors.
"""

import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import sys
import time
import traceback

import pytest

from src.core.context import ProcessingContext
from src.core.cache_workspace import cleanup_cache_workspaces
from src.models import Request, SlicePartEnum
from src.stages import stage_segmentation as segmentation_module
from src.stages.stage_segmentation import SegmentationStage


pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux process-lock regression tests")
_COMPLETE_SEGMENTATION = b"complete fake segmentation\n"
_WAIT_SECONDS = 20


class _FakeSegmentationStage(SegmentationStage):
    def __init__(
        self, model_id, events, *, fail=False, inference_started=None,
        inference_release=None, inference_barrier=None, marker_started=None,
        marker_release=None, marker_barrier=None, forbid_inference=False,
        forbid_marker=False,
    ):
        super().__init__({"_continue": False})
        self.fake_model_id = model_id
        self.events = events
        self.fail = fail
        self.inference_started = inference_started
        self.inference_release = inference_release
        self.inference_barrier = inference_barrier
        self.marker_started = marker_started
        self.marker_release = marker_release
        self.marker_barrier = marker_barrier
        self.forbid_inference = forbid_inference
        self.forbid_marker = forbid_marker

    def _match_model(self, **kwargs):
        return {"model_id": self.fake_model_id, "configuration": "3d_fullres", "fold": "0"}

    def _run_nnunet_inference(self, **kwargs):
        if self.forbid_inference:
            raise AssertionError("a warm cache must not run inference")
        output_path = Path(kwargs["output_path"])
        self.events.put(("inference", os.getpid(), str(output_path)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"partial")
        if self.inference_started is not None:
            self.inference_started.set()
        if self.inference_barrier is not None:
            self.inference_barrier.wait(timeout=_WAIT_SECONDS)
        if self.inference_release is not None:
            if not self.inference_release.wait(timeout=_WAIT_SECONDS):
                raise TimeoutError("fake inference was not released")
        if self.fail:
            raise RuntimeError("intentional fake inference failure")
        output_path.write_bytes(_COMPLETE_SEGMENTATION)

    def _write_model_marker(self, output_dir, dataset_id):
        if self.forbid_marker:
            raise AssertionError("a warm cache must not rewrite its marker")
        self.events.put(("marker", os.getpid(), str(output_dir)))
        if self.marker_started is not None:
            self.marker_started.set()
        if self.marker_barrier is not None:
            self.marker_barrier.wait(timeout=_WAIT_SECONDS)
        if self.marker_release is not None:
            if not self.marker_release.wait(timeout=_WAIT_SECONDS):
                raise TimeoutError("fake marker was not released")
        super()._write_model_marker(output_dir, dataset_id)


def _segmentation_worker(root, model_id, events, start_barrier=None, post_barrier=None, **options):
    # pytest's log fixture lives in the parent; explicitly silence each spawned
    # child so regression tests cannot create application log files.
    segmentation_module.DebugLog = lambda *args, **kwargs: None
    segmentation_module.ErrorLog = lambda *args, **kwargs: None
    try:
        root = Path(root)
        stage = _FakeSegmentationStage(model_id, events, **options)
        request = Request(
            dicom_dir=str(root / "unused-dicom"),
            output_dir=str(root / f"output-{os.getpid()}"),
            cache_dir=str(root / "cache"),
            slice_part=SlicePartEnum.颈椎,
            scan_thickness=1.0,
        )
        context = ProcessingContext(request)
        context.set_result("preprocessing", {"nifti_path": str(root / "input" / "raw_0000.nii.gz")})
        if start_barrier is not None:
            start_barrier.wait(timeout=_WAIT_SECONDS)
        stage.execute(context)
        result_path = context.get_result("segmentation")["seg_path"]
        events.put(("postprocessing", os.getpid(), result_path))
        if post_barrier is not None:
            # Represents independent reconstruction after the cache transaction.
            # Every process must get here before any downstream task finishes.
            post_barrier.wait(timeout=_WAIT_SECONDS)
        events.put(("done", os.getpid(), result_path))
    except BaseException as error:
        events.put(("error", os.getpid(), type(error).__name__, str(error), traceback.format_exc()))


@pytest.fixture
def processes():
    started = []
    yield started
    for process in started:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


@pytest.fixture
def mp_context():
    return multiprocessing.get_context("spawn")


def _start(processes, mp_context, root, events, *, model_id=292, **kwargs):
    process = mp_context.Process(
        target=_segmentation_worker,
        args=(str(root), model_id, events),
        kwargs=kwargs,
    )
    processes.append(process)
    process.start()
    return process


def _join(processes):
    for process in processes:
        process.join(timeout=_WAIT_SECONDS + 5)
        assert not process.is_alive(), f"worker {process.pid} did not finish"
        assert process.exitcode == 0, f"worker {process.pid} exited {process.exitcode}"


def _messages(events):
    result = []
    while True:
        try:
            result.append(events.get(timeout=0.1))
        except Empty:
            return result


def _assert_success(messages, worker_count):
    assert not [message for message in messages if message[0] == "error"], messages
    assert sum(message[0] == "done" for message in messages) == worker_count, messages


def _cache_output(root, model_id=292):
    return root / "cache" / str(model_id) / "segmentation" / "seg.nii.gz"


def _assert_published(root, model_id=292):
    output = _cache_output(root, model_id)
    assert output.read_bytes() == _COMPLETE_SEGMENTATION
    assert json.loads((output.parent / "model.json").read_text()) == {"model_id": str(model_id)}
    return output


def test_three_cold_cache_processes_infer_once_and_publish_only_complete_pair(tmp_path, processes, mp_context):
    events = mp_context.Queue()
    start = mp_context.Barrier(3)
    inference_started = mp_context.Event()
    inference_release = mp_context.Event()
    marker_started = mp_context.Event()
    marker_release = mp_context.Event()
    for _ in range(3):
        _start(
            processes, mp_context, tmp_path, events,
            start_barrier=start, inference_started=inference_started,
            inference_release=inference_release, marker_started=marker_started,
            marker_release=marker_release,
        )

    assert inference_started.wait(timeout=_WAIT_SECONDS)
    output = _cache_output(tmp_path)
    assert not output.exists(), "an incomplete inference must not look like a legacy cache hit"
    assert not (output.parent / "model.json").exists()
    inference_release.set()
    assert marker_started.wait(timeout=_WAIT_SECONDS)
    assert not output.exists(), "the marker must be complete before publishing the final segmentation"
    marker_release.set()
    _join(processes)
    messages = _messages(events)
    _assert_success(messages, 3)
    inference_paths = [Path(message[2]) for message in messages if message[0] == "inference"]
    assert len(inference_paths) == 1, messages
    assert inference_paths[0] != output
    assert inference_paths[0].parent.parent == output.parent.parent
    assert len([message for message in messages if message[0] == "marker"]) == 1, messages
    assert not inference_paths[0].parent.exists(), "successful temporary work must be cleaned up"
    assert {message[2] for message in messages if message[0] == "done"} == {str(output)}
    _assert_published(tmp_path)


@pytest.mark.parametrize("legacy", [False, True], ids=["with-marker", "legacy-no-marker"])
def test_hot_cache_is_read_only_and_following_work_can_overlap(tmp_path, processes, mp_context, legacy):
    output = _cache_output(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(_COMPLETE_SEGMENTATION)
    marker = output.parent / "model.json"
    if not legacy:
        marker.write_text('{"model_id": "292"}\n')
    before = {
        path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in (output, marker) if path.exists()
    }
    events = mp_context.Queue()
    start = mp_context.Barrier(3)
    downstream = mp_context.Barrier(3)
    for _ in range(3):
        _start(
            processes, mp_context, tmp_path, events, start_barrier=start,
            post_barrier=downstream, forbid_inference=True, forbid_marker=True,
        )
    _join(processes)
    messages = _messages(events)
    _assert_success(messages, 3)
    assert sum(message[0] == "postprocessing" for message in messages) == 3
    assert not [message for message in messages if message[0] in {"inference", "marker"}]
    assert not (tmp_path / "cache" / ".locks").exists(), "a warm fast path need not mutate lock state"
    for path, unchanged in before.items():
        assert (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_ctime_ns) == unchanged
    if legacy:
        assert not marker.exists(), "historical cache reuse must not create a marker"


def test_failed_inference_does_not_publish_and_releases_lock_for_retry(tmp_path, processes, mp_context):
    events = mp_context.Queue()
    first = _start(processes, mp_context, tmp_path, events, fail=True)
    _join([first])
    failed_messages = _messages(events)
    errors = [message for message in failed_messages if message[0] == "error"]
    assert len(errors) == 1 and errors[0][2:4] == ("RuntimeError", "intentional fake inference failure"), failed_messages
    output = _cache_output(tmp_path)
    assert not output.exists()
    assert not (output.parent / "model.json").exists()
    incomplete = [Path(message[2]) for message in failed_messages if message[0] == "inference"]
    assert len(incomplete) == 1
    assert not incomplete[0].parent.exists(), "normal failures must remove their private partial output"

    second = _start(processes, mp_context, tmp_path, events)
    _join([second])
    retried = _messages(events)
    _assert_success(retried, 1)
    assert sum(message[0] == "inference" for message in retried) == 1
    _assert_published(tmp_path)


def test_killed_producer_releases_process_lock_and_partial_output_is_not_reused(tmp_path, processes, mp_context):
    events = mp_context.Queue()
    started = mp_context.Event()
    release = mp_context.Event()
    producer = _start(
        processes, mp_context, tmp_path, events,
        inference_started=started, inference_release=release,
    )
    assert started.wait(timeout=_WAIT_SECONDS)
    lock_path = tmp_path / "cache" / ".locks" / "segmentation-292.lock"
    assert lock_path.is_file()
    output = _cache_output(tmp_path)
    assert not output.exists()
    abandoned = list(output.parent.parent.glob(".segmentation-*"))
    assert len(abandoned) == 1
    producer.kill()
    producer.join(timeout=5)
    assert producer.exitcode is not None and producer.exitcode < 0
    # A killed process cannot run TemporaryDirectory cleanup. Its private partial
    # must nevertheless remain invisible to normal cache reuse.
    events_after_kill = _messages(events)
    replacement = _start(processes, mp_context, tmp_path, events)
    _join([replacement])
    retried = _messages(events)
    _assert_success(retried, 1)
    assert sum(message[0] == "inference" for message in retried) == 1, (events_after_kill, retried)
    assert not abandoned[0].exists(), "retry must reclaim the dead producer's private workspace"
    _assert_published(tmp_path)


def _external_prediction_worker(root):
    """Exercise the actual subprocess boundary with a small model-free script."""
    root = Path(root)
    segmentation_module.DebugLog = lambda *args, **kwargs: None
    segmentation_module.ErrorLog = lambda *args, **kwargs: None
    stage = SegmentationStage({
        "nnUNet_predict_command": str(root / "fake_predict.py"),
        "nnUNet_python": sys.executable,
        **{key: str(root / key) for key in (
            "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results",
        )},
    })
    request = Request(
        dicom_dir=str(root / "unused-dicom"),
        output_dir=str(root / "output"),
        cache_dir=str(root / "cache"),
        slice_part=SlicePartEnum.颈椎,
        scan_thickness=1.0,
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {"nifti_path": str(root / "input/raw_0000.nii.gz")})
    stage.execute(context)


def test_real_prediction_child_keeps_workspace_protected_after_parent_is_killed(
    tmp_path, processes, mp_context,
):
    script = tmp_path / "fake_predict.py"
    script.write_text(
        "import json, os, pathlib, sys, time\n"
        "root = pathlib.Path(__file__).parent\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "leases = []\n"
        "for fd in pathlib.Path('/proc/self/fd').iterdir():\n"
        "    try:\n"
        "        target = os.readlink(fd)\n"
        "        if target.endswith('.cache-workspace.lease'): leases.append(target)\n"
        "    except OSError: pass\n"
        "(output / 'raw.nii.gz').write_bytes(b'child is still writing')\n"
        "(root / 'child-ready.json').write_text(json.dumps({'pid': os.getpid(), 'output': str(output), 'leases': leases}))\n"
        "deadline = time.monotonic() + 20\n"
        "while not (root / 'release-child').exists():\n"
        "    if time.monotonic() > deadline: raise TimeoutError('child was not released')\n"
        "    time.sleep(0.01)\n"
        "(output / 'raw.nii.gz').write_bytes(b'complete fake segmentation')\n"
    )
    parent = mp_context.Process(target=_external_prediction_worker, args=(str(tmp_path),))
    processes.append(parent)
    child_pid = None
    try:
        parent.start()
        ready = tmp_path / "child-ready.json"
        deadline = time.monotonic() + _WAIT_SECONDS
        while not ready.exists():
            assert parent.is_alive(), f"producer exited {parent.exitcode} before its child started"
            assert time.monotonic() < deadline, "fake prediction child did not start"
            time.sleep(0.01)
        info = json.loads(ready.read_text())
        child_pid = info["pid"]
        workspace = Path(info["output"])
        assert info["leases"], "the real prediction subprocess must inherit its workspace lease"
        parent.kill()
        parent.join(timeout=5)
        cleanup_cache_workspaces(workspace.parent, ".segmentation-")
        assert workspace.is_dir(), "an orphan prediction child must retain its active workspace"
        assert (workspace / "raw.nii.gz").read_bytes() == b"child is still writing"

        (tmp_path / "release-child").touch()
        deadline = time.monotonic() + _WAIT_SECONDS
        while workspace.exists():
            cleanup_cache_workspaces(workspace.parent, ".segmentation-")
            assert time.monotonic() < deadline, "workspace was not reclaimed after the child exited"
            time.sleep(0.01)
    finally:
        (tmp_path / "release-child").touch()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_different_models_infer_and_write_private_markers_concurrently(tmp_path, processes, mp_context):
    events = mp_context.Queue()
    inference_together = mp_context.Barrier(2)
    markers_together = mp_context.Barrier(2)
    for model_id in (292, 315):
        _start(
            processes, mp_context, tmp_path, events, model_id=model_id,
            inference_barrier=inference_together, marker_barrier=markers_together,
        )
    _join(processes)
    messages = _messages(events)
    _assert_success(messages, 2)
    marker_dirs = [Path(message[2]) for message in messages if message[0] == "marker"]
    assert len(marker_dirs) == len(set(marker_dirs)) == 2
    for model_id in (292, 315):
        output = _assert_published(tmp_path, model_id)
        assert output.parent not in marker_dirs, "markers must be prepared in private working directories"
        assert (tmp_path / "cache" / ".locks" / f"segmentation-{model_id}.lock").is_file()
    assert not list((tmp_path / "cache").rglob(".model.json*")), "marker temporary files must not leak"


def _marker_worker(output_dir, model_id, events, release):
    real_dump = segmentation_module.json.dump

    def paused_dump(data, file, *args, **kwargs):
        # Pause after the temporary file has been opened. Every writer must own
        # a separate temporary name even when targeting the same marker.
        events.put(("marker-open", os.getpid()))
        if not release.wait(timeout=_WAIT_SECONDS):
            raise TimeoutError("concurrent marker writes were not released")
        return real_dump(data, file, *args, **kwargs)

    segmentation_module.json.dump = paused_dump
    try:
        SegmentationStage._write_model_marker(Path(output_dir), model_id)
        events.put(("done", os.getpid(), str(output_dir)))
    except BaseException as error:
        events.put(("error", os.getpid(), type(error).__name__, str(error), traceback.format_exc()))


def test_concurrent_marker_writers_use_distinct_temporary_files(tmp_path, processes, mp_context):
    marker_dir = tmp_path / "marker-only"
    marker_dir.mkdir()
    events = mp_context.Queue()
    release = mp_context.Event()
    for model_id in (292, 297, 315):
        process = mp_context.Process(
            target=_marker_worker,
            args=(str(marker_dir), model_id, events, release),
        )
        processes.append(process)
        process.start()
    opened = [events.get(timeout=_WAIT_SECONDS) for _ in range(3)]
    assert all(message[0] == "marker-open" for message in opened), opened
    assert len(list(marker_dir.iterdir())) == 3, "each marker writer needs its own temporary file"
    assert not (marker_dir / "model.json").exists()
    release.set()
    _join(processes)
    messages = _messages(events)
    _assert_success(messages, 3)
    assert json.loads((marker_dir / "model.json").read_text())["model_id"] in {"292", "297", "315"}
    assert list(marker_dir.iterdir()) == [marker_dir / "model.json"]


@pytest.mark.parametrize("failure_point", ["model.json", "seg.nii.gz"], ids=["marker-publication", "seg-publication"])
def test_failed_publication_of_replacement_cache_is_not_reused_and_can_retry(tmp_path, monkeypatch, failure_point):
    output = _cache_output(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"complete but invalid old model segmentation\n")
    marker = output.parent / "model.json"
    marker.write_text('{"model_id": "297"}\n', encoding="utf-8")
    assert not SegmentationStage._can_reuse_segmentation(output.parent, output, 292)

    def context():
        request = Request(
            dicom_dir=str(tmp_path / "unused-dicom"),
            output_dir=str(tmp_path / "independent-output"),
            cache_dir=str(tmp_path / "cache"),
            slice_part=SlicePartEnum.颈椎,
            scan_thickness=1.0,
        )
        result = ProcessingContext(request)
        result.set_result("preprocessing", {"nifti_path": str(tmp_path / "input/raw_0000.nii.gz")})
        return result

    events = Queue()
    stage = _FakeSegmentationStage(292, events)
    original_replace = Path.replace
    failed_publications = []

    def fail_publication(source, target):
        if Path(target) == output.parent / failure_point:
            failed_publications.append((source, Path(target)))
            raise OSError("intentional cache publication failure")
        return original_replace(source, target)

    with monkeypatch.context() as publication:
        publication.setattr(Path, "replace", fail_publication)
        with pytest.raises(OSError, match="intentional cache publication failure"):
            stage.execute(context())

    assert len(failed_publications) == 1
    assert not output.exists(), "an invalid old seg must not pair with a newly published marker"
    assert not SegmentationStage._can_reuse_segmentation(output.parent, output, 292)
    assert not list(output.parent.parent.glob(".segmentation-*")), "failed private work must be cleaned up"
    failed_events = _messages(events)
    assert sum(event[0] == "inference" for event in failed_events) == 1
    assert sum(event[0] == "marker" for event in failed_events) == 1
    if failure_point == "seg.nii.gz":
        assert json.loads(marker.read_text()) == {"model_id": "292"}

    retried_context = context()
    stage.execute(retried_context)
    retry_events = _messages(events)
    assert sum(event[0] == "inference" for event in retry_events) == 1, "retry must generate rather than reuse the failed publication"
    assert retried_context.get_result("segmentation")["seg_path"] == str(output)
    _assert_published(tmp_path)
    assert SegmentationStage._can_reuse_segmentation(output.parent, output, 292)

    # Once the replacement is complete, the same fixture must become read-only.
    before = {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_ctime_ns) for path in (output, marker)}
    stage.forbid_inference = True
    stage.forbid_marker = True
    stage.execute(context())
    assert _messages(events) == []
    assert {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_ctime_ns) for path in before} == before
