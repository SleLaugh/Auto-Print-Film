"""Keep segmentation sidecars consistent while replacing an invalid cache.

Inference is a subprocess stub that only writes test payloads. These tests never
load a model or parse patient images; cache publication operates on opaque files.
"""

import json
from pathlib import Path
import subprocess

import pytest

from src.core.context import ProcessingContext
from src.core.errors import ExternalProcessError
from src.models import Request
from src.stages import stage_segmentation as segmentation_module
from src.stages.stage_segmentation import SegmentationStage


_MODEL = 292
_OLD_SEG = b"invalid-old-segmentation"
_NEW_SEG = b"new-complete-segmentation"


@pytest.fixture
def cache_case(tmp_path):
    input_path = tmp_path / "nifti" / "raw_0000.nii.gz"
    input_path.parent.mkdir()
    input_path.touch()
    output_dir = tmp_path / "cache" / str(_MODEL) / "segmentation"
    output_dir.mkdir(parents=True)
    (output_dir / "seg.nii.gz").write_bytes(_OLD_SEG)
    (output_dir / "model.json").write_text('{"model_id":"315"}\n')
    (output_dir / "dataset.json").write_text('{"labels":{"stale-label":99}}\n')
    (output_dir / "old-probabilities.npz").write_bytes(b"obsolete")
    old_directory = output_dir / "obsolete-directory"
    old_directory.mkdir()
    (old_directory / "old.json").write_text("obsolete")
    # Cleaning the old directory must unlink links without following them.
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "keep.txt").write_text("unrelated file")
    (output_dir / "old-link").symlink_to(outside, target_is_directory=True)

    request = Request(
        dicom_dir=str(tmp_path / "unused-dicom"),
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        scan_thickness=1.0,
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {"nifti_path": str(input_path)})
    stage = SegmentationStage({"_continue": False})
    return stage, context, output_dir, outside


def _fake_command(monkeypatch, sidecars=None, returncode=0):
    called = []

    def predict(command, **kwargs):
        output_dir = Path(command[command.index("-o") + 1])
        called.append(output_dir)
        assert output_dir.name != "segmentation", "the command must write into its private workspace"
        (output_dir / "raw.nii.gz").write_bytes(_NEW_SEG)
        for name, content in (sidecars or {}).items():
            (output_dir / name).write_bytes(content)
        return subprocess.CompletedProcess(command, returncode, "", "intentional fake failure" if returncode else "")

    monkeypatch.setattr(segmentation_module.subprocess, "run", predict)
    return called


def _snapshot_tree(path):
    result = {}
    for child in path.rglob("*"):
        stat = child.lstat()
        result[str(child.relative_to(path))] = (
            child.readlink().as_posix() if child.is_symlink() else child.read_bytes() if child.is_file() else None,
            stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns,
        )
    return result


def _assert_hit(output_dir):
    assert SegmentationStage._can_reuse_segmentation(output_dir, output_dir / "seg.nii.gz", _MODEL)
    assert (output_dir / "seg.nii.gz").read_bytes() == _NEW_SEG
    assert json.loads((output_dir / "model.json").read_text()) == {"model_id": str(_MODEL)}


def test_invalid_cache_removes_old_sidecars_when_command_only_writes_segmentation(cache_case, monkeypatch):
    stage, context, output_dir, outside = cache_case
    calls = _fake_command(monkeypatch)

    stage.execute(context)

    assert len(calls) == 1
    _assert_hit(output_dir)
    assert {path.name for path in output_dir.iterdir()} == {"seg.nii.gz", "model.json"}
    assert (outside / "keep.txt").read_text() == "unrelated file"
    assert context.get_result("segmentation")["seg_path"] == str(output_dir / "seg.nii.gz")


def test_new_sidecars_are_complete_before_final_segmentation_becomes_visible(cache_case, monkeypatch):
    stage, context, output_dir, outside = cache_case
    new_sidecars = {
        "dataset.json": b'{"labels":{"new-label":1}}\n',
        "plans.json": b'{"plans_name":"current"}\n',
        "predict_from_raw_data_args.json": b'{"step_size":0.5,"use_mirroring":true}\n',
    }
    _fake_command(monkeypatch, new_sidecars)
    real_replace = Path.replace
    published = []

    def checked_replace(source, target):
        if Path(target) == output_dir / "seg.nii.gz":
            assert not (output_dir / "seg.nii.gz").exists(), "old segmentation must be removed before marker replacement"
            assert json.loads((output_dir / "model.json").read_text()) == {"model_id": str(_MODEL)}
            for name, expected in new_sidecars.items():
                assert (output_dir / name).read_bytes() == expected
            assert not (output_dir / "old-probabilities.npz").exists()
            published.append(True)
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", checked_replace)
    stage.execute(context)

    assert published == [True]
    _assert_hit(output_dir)
    assert {path.name for path in output_dir.iterdir()} == {"seg.nii.gz", "model.json", *new_sidecars}
    assert (outside / "keep.txt").is_file()


@pytest.mark.parametrize("failure_phase", ["inference", "marker", "publish"])
def test_failed_replacement_never_pairs_invalid_old_segmentation_with_new_marker(cache_case, monkeypatch, failure_phase):
    stage, context, output_dir, outside = cache_case
    initial_tree = _snapshot_tree(output_dir)
    _fake_command(monkeypatch, {"dataset.json": b'{"labels":{"new-label":1}}'}, returncode=1 if failure_phase == "inference" else 0)
    expected_exception = ExternalProcessError if failure_phase == "inference" else OSError

    if failure_phase == "marker":
        def failing_marker(working_dir, dataset_id):
            (working_dir / "model.json").write_text(json.dumps({"model_id": str(dataset_id)}))
            raise OSError("intentional marker failure")
        monkeypatch.setattr(stage, "_write_model_marker", failing_marker)
    elif failure_phase == "publish":
        real_replace = Path.replace

        def failing_segmentation_replace(source, target):
            if Path(target) == output_dir / "seg.nii.gz":
                raise OSError("intentional final publication failure")
            return real_replace(source, target)
        monkeypatch.setattr(Path, "replace", failing_segmentation_replace)

    with pytest.raises(expected_exception):
        stage.execute(context)

    assert not stage._can_reuse_segmentation(output_dir, output_dir / "seg.nii.gz", _MODEL)
    if failure_phase in {"inference", "marker"}:
        assert _snapshot_tree(output_dir) == initial_tree, "failed private generation must leave the old invalid cache unchanged"
    else:
        assert not (output_dir / "seg.nii.gz").exists(), "an interrupted publish must not expose the old segmentation with a new marker"
    assert (outside / "keep.txt").read_text() == "unrelated file"


def test_hot_cache_keeps_all_sidecars_read_only(cache_case, monkeypatch):
    stage, context, output_dir, outside = cache_case
    (output_dir / "model.json").write_text(json.dumps({"model_id": str(_MODEL)}))
    before = _snapshot_tree(output_dir)
    output_dir_stat = output_dir.stat()

    def forbidden(*args, **kwargs):
        raise AssertionError("a warm cache must not infer or rewrite its marker")

    monkeypatch.setattr(stage, "_run_nnunet_inference", forbidden)
    monkeypatch.setattr(stage, "_write_model_marker", forbidden)
    stage.execute(context)

    assert _snapshot_tree(output_dir) == before
    assert (output_dir.stat().st_ino, output_dir.stat().st_mtime_ns, output_dir.stat().st_ctime_ns) == (
        output_dir_stat.st_ino, output_dir_stat.st_mtime_ns, output_dir_stat.st_ctime_ns,
    )
    assert not (Path(context.request.cache_dir) / ".locks").exists()
    assert (outside / "keep.txt").is_file()
