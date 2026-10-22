"""Shared preprocessing cache tests using spawned processes and synthetic images only."""

from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import Request
from src.stages import stage_preprocessing as preprocessing
from src.stages.stage_preprocessing import PreprocessingStage


def _image():
    image = sitk.GetImageFromArray(np.arange(24, dtype=np.int16).reshape(2, 3, 4) - 12)
    image.SetSpacing((0.75, 0.75, 1.5))
    image.SetOrigin((1.0, 2.0, 3.0))
    return image


def _context(root):
    return ProcessingContext(Request(
        dicom_dir=str(root / "synthetic-input"),
        output_dir=str(root / "output"),
        cache_dir=str(root / "cache"),
        scan_thickness=1.5,
    ))


def _read_result():
    return {"image": _image(), "series_file_names": ["synthetic-1.dcm", "synthetic-2.dcm"]}


def _preprocessing_worker(root, start, entered, release, calls, attempts, results):
    """A fresh interpreter executes the actual stage and actual OS cache lock."""
    root = Path(root)
    original_lock = preprocessing.cache_lock
    original_write = sitk.WriteImage

    @contextmanager
    def observed_lock(path):
        attempts.put("attempt")
        with original_lock(path):
            yield

    def read_series(_stage, _directory):
        with calls.get_lock():
            calls.value += 1
        return _read_result()

    def write_image(image, path):
        # Keep a real partial file visible only inside the writer's work directory.
        Path(path).write_bytes(b"synthetic partial NIfTI")
        entered.set()
        if not release.wait(15):
            raise TimeoutError("Synthetic writer was not released")
        return original_write(image, path)

    try:
        with (
            patch.object(preprocessing, "cache_lock", observed_lock),
            patch.object(preprocessing, "DebugLog"),
            patch.object(preprocessing, "ErrorLog"),
            patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
            patch.object(preprocessing, "is_debug_file_enabled", return_value=True),
            patch.object(PreprocessingStage, "_read_dicom_series", read_series),
            patch.object(preprocessing.sitk, "WriteImage", write_image),
        ):
            if not start.wait(15):
                raise TimeoutError("Synthetic worker was not started")
            context = _context(root)
            PreprocessingStage().execute(context)
            result = context.get_result("preprocessing")
            actual = sitk.ReadImage(result["nifti_path"])
            results.put(("ok", actual.GetSize(), int(sitk.GetArrayFromImage(actual).sum()), Path(result["metadata_path"]).is_file()))
    except BaseException as error:
        results.put(("error", repr(error)))
        raise


def test_spawned_processes_convert_once_and_publish_complete_cache(tmp_path):
    mp = multiprocessing.get_context("spawn")
    start, entered, release = mp.Event(), mp.Event(), mp.Event()
    calls = mp.Value("i", 0)
    attempts, results = mp.Queue(), mp.Queue()
    children = [mp.Process(target=_preprocessing_worker, args=(
        str(tmp_path), start, entered, release, calls, attempts, results,
    )) for _ in range(3)]
    try:
        for child in children:
            child.start()
        start.set()
        assert entered.wait(15), "A synthetic writer must acquire the cache lock"
        # All contenders reach the actual lock while its owner is paused.
        assert [attempts.get(timeout=15) for _ in children] == ["attempt"] * 3
        assert calls.value == 1
        assert not (tmp_path / "cache/nifti/raw_0000.nii.gz").exists()
        partials = list((tmp_path / "cache/.work").glob("*/raw_0000.nii.gz"))
        assert len(partials) == 1
        assert partials[0].read_bytes() == b"synthetic partial NIfTI"
        release.set()
        outcomes = [results.get(timeout=15) for _ in children]
        for child in children:
            child.join(15)
            assert child.exitcode == 0
        assert outcomes == [("ok", (4, 3, 2), -12, True)] * 3
        assert calls.value == 1
        cached = sitk.ReadImage(str(tmp_path / "cache/nifti/raw_0000.nii.gz"))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(cached), sitk.GetArrayFromImage(_image()))
        assert cached.GetSpacing() == _image().GetSpacing()
        assert cached.GetOrigin() == _image().GetOrigin()
        assert list((tmp_path / "cache/nifti").iterdir()) == [tmp_path / "cache/nifti/raw_0000.nii.gz"]
        assert list((tmp_path / "cache/.work").iterdir()) == []
        assert (tmp_path / "cache/.locks/preprocessing.lock").is_file()
        metadata = json.loads((tmp_path / "cache/metadata/dicom_metadata.json").read_text())
        assert metadata["size_xyz"] == [4, 3, 2]
    finally:
        release.set()
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.terminate()
                child.join(5)
        for queue in (attempts, results):
            queue.close()
            queue.join_thread()


@pytest.mark.parametrize("debug", [False, True])
def test_legacy_cache_hit_does_not_write_or_lock(tmp_path, debug):
    output = tmp_path / "cache/nifti/raw_0000.nii.gz"
    output.parent.mkdir(parents=True)
    sitk.WriteImage(_image(), str(output))
    metadata = tmp_path / "cache/metadata/dicom_metadata.json"
    metadata.parent.mkdir()
    metadata.write_text('{"legacy": true}', encoding="utf-8")
    snapshots = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output, metadata)}
    with (
        patch.object(preprocessing, "cache_lock", side_effect=AssertionError("Cache hit must be read-only")),
        patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
        patch.object(preprocessing, "is_debug_file_enabled", return_value=debug),
        patch.object(PreprocessingStage, "_read_dicom_series", side_effect=AssertionError("Unexpected conversion")),
        patch.object(preprocessing.sitk, "WriteImage", side_effect=AssertionError("Unexpected cache write")),
    ):
        context = _context(tmp_path)
        PreprocessingStage().execute(context)
    assert context.get_result("preprocessing")["nifti_path"] == str(output)
    assert context.get_result("preprocessing")["metadata_path"] == str(metadata)
    assert not (tmp_path / "cache/.work").exists()
    assert not (tmp_path / "cache/.locks").exists()
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in snapshots} == snapshots


def test_failed_partial_write_is_not_published_and_can_retry(tmp_path):
    output = tmp_path / "cache/nifti/raw_0000.nii.gz"
    failed_path = []

    def fail_after_partial_write(_image, path):
        failed_path.append(Path(path))
        Path(path).write_bytes(b"synthetic incomplete image")
        assert Path(path).parent.parent == tmp_path / "cache/.work"
        assert not output.exists()
        raise OSError("synthetic write failure")

    with (
        patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
        patch.object(preprocessing, "is_debug_file_enabled", return_value=False),
        patch.object(PreprocessingStage, "_read_dicom_series", return_value=_read_result()) as read,
    ):
        with patch.object(preprocessing.sitk, "WriteImage", side_effect=fail_after_partial_write):
            with pytest.raises(OSError, match="synthetic write failure"):
                PreprocessingStage().execute(_context(tmp_path))
        assert not output.exists()
        assert not failed_path[0].exists()
        assert list((tmp_path / "cache/.work").iterdir()) == []
        PreprocessingStage().execute(_context(tmp_path))
        assert read.call_count == 2
    np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(output))), sitk.GetArrayFromImage(_image()))


def test_debug_metadata_failure_does_not_publish_raw(tmp_path):
    output = tmp_path / "cache/nifti/raw_0000.nii.gz"
    with (
        patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
        patch.object(preprocessing, "is_debug_file_enabled", return_value=True),
        patch.object(PreprocessingStage, "_read_dicom_series", return_value=_read_result()),
    ):
        with patch.object(preprocessing.json, "dump", side_effect=OSError("synthetic metadata failure")):
            with pytest.raises(OSError, match="synthetic metadata failure"):
                PreprocessingStage().execute(_context(tmp_path))
        assert not output.exists()
        assert not (tmp_path / "cache/metadata/dicom_metadata.json").exists()
        assert list((tmp_path / "cache/.work").iterdir()) == []
        PreprocessingStage().execute(_context(tmp_path))
    assert output.is_file()
    assert (tmp_path / "cache/metadata/dicom_metadata.json").is_file()


def test_debug_metadata_is_published_before_final_raw(tmp_path):
    output = tmp_path / "cache/nifti/raw_0000.nii.gz"
    metadata = tmp_path / "cache/metadata/dicom_metadata.json"
    original_replace = Path.replace
    publications = []

    def observe_replace(source, target):
        target = Path(target)
        if target == output:
            assert metadata.is_file()
            assert not output.exists()
            assert sitk.ReadImage(str(source)).GetSize() == (4, 3, 2)
        publications.append(target)
        return original_replace(source, target)

    with (
        patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
        patch.object(preprocessing, "is_debug_file_enabled", return_value=True),
        patch.object(PreprocessingStage, "_read_dicom_series", return_value=_read_result()),
        patch.object(Path, "replace", observe_replace),
    ):
        PreprocessingStage().execute(_context(tmp_path))
    assert publications == [metadata, output]


def _abandoned_preprocessing_worker(root, connection):
    def interrupted_write(_image, path):
        Path(path).write_bytes(b"synthetic interrupted preprocessing")
        connection.send(str(path))
        multiprocessing.Event().wait(30)
        raise TimeoutError("Synthetic producer was not stopped")

    with (
        patch.object(preprocessing, "DebugLog"),
        patch.object(preprocessing, "ErrorLog"),
        patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
        patch.object(preprocessing, "is_debug_file_enabled", return_value=False),
        patch.object(PreprocessingStage, "_read_dicom_series", return_value=_read_result()),
        patch.object(preprocessing.sitk, "WriteImage", side_effect=interrupted_write),
    ):
        PreprocessingStage().execute(_context(Path(root)))


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGKILL recovery")
@pytest.mark.parametrize("hot_cache", [False, True], ids=["cold-retry", "warm-reuse"])
def test_next_cache_use_reclaims_killed_preprocessing_workspace(tmp_path, hot_cache):
    mp = multiprocessing.get_context("spawn")
    receiver, sender = mp.Pipe(duplex=False)
    child = mp.Process(target=_abandoned_preprocessing_worker, args=(str(tmp_path), sender))
    child.start()
    sender.close()
    try:
        assert receiver.poll(15)
        partial = Path(receiver.recv())
        assert partial.is_file()
        child.kill()
        child.join(5)
        assert child.exitcode is not None and child.exitcode < 0
        assert partial.is_file(), "SIGKILL must leave the synthetic workspace for recovery"
        output = tmp_path / "cache/nifti/raw_0000.nii.gz"
        assert not output.exists()
        if hot_cache:
            sitk.WriteImage(_image(), str(output))
        with (
            patch.object(preprocessing, "read_source_dicom_metadata", return_value={}),
            patch.object(preprocessing, "is_debug_file_enabled", return_value=False),
            patch.object(PreprocessingStage, "_read_dicom_series", return_value=_read_result()) as read,
        ):
            PreprocessingStage().execute(_context(tmp_path))
        assert read.call_count == (0 if hot_cache else 1)
        assert not partial.parent.exists()
        assert list((tmp_path / "cache/.work").iterdir()) == []
        assert sitk.ReadImage(str(output)).GetSize() == (4, 3, 2)
    finally:
        receiver.close()
        if child.is_alive():
            child.kill()
        child.join(5)
