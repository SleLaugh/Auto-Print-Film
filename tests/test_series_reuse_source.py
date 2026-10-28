"""Only opted-in cross-series reuse requires current DICOM source provenance."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import Request
from src.stages import stage_preprocessing as preprocessing
from src.stages.stage_preprocessing import PreprocessingStage, SourceChangedError


def _image(value):
    return sitk.GetImageFromArray(np.full((2, 3, 4), value, dtype=np.int16))


def _bump_mtime(path):
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def source_case(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "1.dcm"
    source.write_text("7")
    monkeypatch.setattr(preprocessing, "read_source_dicom_metadata", lambda *args: {})
    monkeypatch.setattr(preprocessing, "is_debug_file_enabled", lambda: False)
    stage = PreprocessingStage()
    calls = []

    def read_series(directory):
        calls.append(directory)
        return {"image": _image(int(source.read_text(encoding="utf-8"))), "series_file_names": [str(source)]}

    monkeypatch.setattr(stage, "_read_dicom_series", read_series)

    def context(enabled=True):
        current = ProcessingContext(Request(
            dicom_dir=str(source_dir), output_dir=str(tmp_path / "output"),
            cache_dir=str(tmp_path / "cache"),
        ))
        if enabled:
            current.metadata["series_reuse_requested"] = True
        return current

    raw = tmp_path / "cache/nifti/raw_0000.nii.gz"
    proof = tmp_path / "cache/metadata/source_provenance.json"
    return stage, context, source, raw, proof, calls


def test_disabled_keeps_legacy_cache_without_check_or_write(source_case, monkeypatch):
    stage, context, source, raw, proof, calls = source_case
    raw.parent.mkdir(parents=True)
    sitk.WriteImage(_image(99), str(raw))
    before = (raw.read_bytes(), raw.stat())

    def no_signature(*args):
        pytest.fail("disabled mode must keep its original read-only cache behavior")

    monkeypatch.setattr(preprocessing, "_source_signature", no_signature)
    stage.execute(context(False))

    assert calls == []
    assert (raw.read_bytes(), raw.stat()) == before
    assert not proof.exists()


def test_enabled_rebuilds_unproven_legacy_raw_once_then_reuses_read_only(source_case, monkeypatch):
    stage, context, source, raw, proof, calls = source_case
    raw.parent.mkdir(parents=True)
    sitk.WriteImage(_image(99), str(raw))

    first = context()
    stage.execute(first)
    assert len(calls) == 1
    assert np.all(sitk.GetArrayFromImage(sitk.ReadImage(str(raw))) == 7)
    provenance = json.loads(proof.read_text(encoding="utf-8"))
    assert provenance["source"]["file_count"] == 1
    assert provenance["raw_generation"]["inode"] == raw.stat().st_ino
    assert first.metadata["series_reuse_source_validated"] is True
    before = (raw.read_bytes(), raw.stat(), proof.read_bytes(), proof.stat())

    def no_lock(*args):
        pytest.fail("a validated cache hit should not acquire the writer lock")

    monkeypatch.setattr(preprocessing, "cache_lock", no_lock)
    stage.execute(context())
    assert len(calls) == 1
    assert (raw.read_bytes(), raw.stat(), proof.read_bytes(), proof.stat()) == before


@pytest.mark.parametrize("change", ["modify", "add", "remove", "rename", "raw", "proof"])
def test_source_or_raw_generation_change_forces_reconversion(source_case, change):
    stage, context, source, raw, proof, calls = source_case
    stage.execute(context())
    if change == "modify":
        source.write_text("8")
        _bump_mtime(source)
    elif change == "add":
        (source.parent / "2.dcm").write_text("8")
    elif change == "remove":
        extra = source.parent / "2.dcm"
        extra.write_text("8")
        stage.execute(context())
        extra.unlink()
    elif change == "rename":
        extra = source.parent / "2.dcm"
        extra.write_text("8")
        stage.execute(context())
        extra.rename(source.parent / "3.dcm")
    elif change == "raw":
        sitk.WriteImage(_image(99), str(raw))
        _bump_mtime(raw)
    else:
        proof.write_text("invalid proof")
    before = len(calls)

    stage.execute(context())

    assert len(calls) == before + 1
    expected = int(source.read_text(encoding="utf-8"))
    assert np.all(sitk.GetArrayFromImage(sitk.ReadImage(str(raw))) == expected)


def test_source_change_during_conversion_publishes_neither_raw_nor_proof(source_case, monkeypatch):
    stage, context, source, raw, proof, calls = source_case
    read_series = stage._read_dicom_series

    def changing_read(directory):
        result = read_series(directory)
        source.write_text("9")
        _bump_mtime(source)
        return result

    monkeypatch.setattr(stage, "_read_dicom_series", changing_read)
    with pytest.raises(SourceChangedError):
        stage.execute(context())
    assert not raw.exists()
    assert not proof.exists()


def test_source_change_at_proof_publication_removes_proof(source_case, monkeypatch):
    stage, context, source, raw, proof, calls = source_case
    replace = Path.replace

    def changing_replace(path, target):
        result = replace(path, target)
        if Path(target) == proof:
            source.write_text("9")
            _bump_mtime(source)
        return result

    monkeypatch.setattr(Path, "replace", changing_replace)
    with pytest.raises(SourceChangedError):
        stage.execute(context())
    assert not proof.exists()


def test_source_generation_rechecked_inside_existing_writer_lock(source_case, monkeypatch):
    stage, context, source, raw, proof, calls = source_case
    entered, release = threading.Event(), threading.Event()
    read_series = stage._read_dicom_series

    def paused_read(directory):
        result = read_series(directory)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(stage, "_read_dicom_series", paused_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(stage.execute, context())
        assert entered.wait(5)
        second = pool.submit(stage.execute, context())
        release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert len(calls) == 1
    assert proof.is_file()


def test_source_file_path_is_supported(source_case):
    stage, context, source, raw, proof, calls = source_case
    current = context()
    current.request.dicom_dir = str(source)
    stage.execute(current)
    assert json.loads(proof.read_text(encoding="utf-8"))["source"]["file_count"] == 1
    second = context()
    second.request.dicom_dir = str(source)
    stage.execute(second)
    assert len(calls) == 1
