"""Model selection, source generations, and fail-closed reuse provenance."""

import json
import os

import pydicom
import pytest

from src.config import AppConfig, NnUNetConfig
from src.core import series_reuse
from src.stages.stage_segmentation import SegmentationStage
from test_series_reuse_end_to_end import (
    _pipeline,
    _request,
    lung_reuse_case as lung_reuse_case,
)


@pytest.fixture
def model_case(tmp_path):
    root = tmp_path / "models"
    folder = root / "Dataset315_Lung" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    fold = folder / "fold_0"
    fold.mkdir(parents=True)
    (folder / "dataset.json").write_text('{"labels":{"background":0,"lung":1}}')
    (folder / "plans.json").write_text('{"plans_name":"nnUNetPlans","version":1}')
    (fold / "checkpoint_final.pth").write_bytes(b"synthetic-model-v1")
    config = AppConfig(nnunet=NnUNetConfig(results=str(root)))
    model = {"model_id": "315", "configuration": "3d_fullres", "fold": "0"}
    return root, folder, config, model


def _ready_markers(parent, name):
    return list((parent / name / "cache" / "series_reuse" / "v1").glob("*/*/*/ready.json"))


def _mutable_model_identity(monkeypatch):
    current = {"checkpoint": "version-1"}
    monkeypatch.setattr(series_reuse, "_model_identity", lambda *args: dict(current))
    # Isolate model provenance from unrelated source edits while this test runs.
    monkeypatch.setattr(series_reuse, "_code_identity", lambda: {"test-code": "version-1"})
    return current


def test_model_identity_selects_current_configuration_fold_and_final_checkpoint(model_case):
    root, folder, config, model = model_case
    unrelated = folder / "fold_1"
    unrelated.mkdir()
    (unrelated / "checkpoint_final.pth").write_bytes(b"other-fold")
    (folder / "fold_0" / "checkpoint_best.pth").write_bytes(b"other-checkpoint")
    other_trainer = root / "Dataset315_Lung" / "OtherTrainer__nnUNetPlans__3d_fullres"
    other_trainer.mkdir()
    (other_trainer / "checkpoint_final.pth").write_bytes(b"other-trainer")

    actual = series_reuse._model_identity(config, model)

    assert set(actual) == {
        str((folder / "dataset.json").resolve()),
        str((folder / "plans.json").resolve()),
        str((folder / "fold_0" / "checkpoint_final.pth").resolve()),
    }
    (unrelated / "checkpoint_final.pth").write_bytes(b"unrelated changed")
    assert series_reuse._model_identity(config, model) == actual
    other_fold = dict(model, fold="1")
    assert series_reuse._model_identity(config, other_fold) != actual


@pytest.mark.parametrize("name", ["dataset.json", "plans.json", "fold_0/checkpoint_final.pth"])
def test_selected_model_generation_or_label_plan_content_changes_identity(model_case, name):
    _root, folder, config, model = model_case
    before = series_reuse._model_identity(config, model)
    selected = folder / name
    if selected.suffix == ".json":
        payload = json.loads(selected.read_text(encoding="utf-8"))
        payload["changed"] = True
        selected.write_text(json.dumps(payload))
    else:
        previous = selected.stat()
        replacement = selected.with_suffix(".replacement")
        replacement.write_bytes(b"synthetic-model-v2")
        assert replacement.stat().st_size == previous.st_size
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        os.replace(replacement, selected)
    assert series_reuse._model_identity(config, model) != before


def test_empty_config_results_uses_actual_environment_model_directory(model_case, monkeypatch):
    root, _folder, config, model = model_case
    monkeypatch.setenv("nnUNet_results", str(root))
    from_environment = AppConfig(nnunet=NnUNetConfig(results=""))
    assert series_reuse._model_identity(from_environment, model) == series_reuse._model_identity(config, model)


def test_explicit_results_directory_takes_precedence_over_environment(model_case, monkeypatch):
    root, _folder, config, model = model_case
    monkeypatch.setenv("nnUNet_results", str(root / "missing-environment-models"))
    assert len(series_reuse._model_identity(config, model)) == 3


@pytest.mark.parametrize("name", ["dataset.json", "plans.json", "fold_0/checkpoint_final.pth"])
def test_missing_selected_model_file_rejects_reuse(model_case, name):
    _root, folder, config, model = model_case
    (folder / name).unlink()
    with pytest.raises(ValueError, match="来源文件缺失"):
        series_reuse._model_identity(config, model)


def test_unknown_results_directory_rejects_empty_model_identity(monkeypatch):
    monkeypatch.delenv("nnUNet_results", raising=False)
    config = AppConfig(nnunet=NnUNetConfig(results=""))
    model = {"model_id": "315", "configuration": "3d_fullres", "fold": "0"}
    with pytest.raises(ValueError, match="无法确定实际"):
        series_reuse._model_identity(config, model)


def test_ambiguous_dataset_directory_rejects_reuse(model_case):
    root, _folder, config, model = model_case
    (root / "Dataset315_Duplicate").mkdir()
    with pytest.raises(ValueError, match="无法唯一确定"):
        series_reuse._model_identity(config, model)


def test_new_model_generation_misses_old_full_and_segmentation_cache(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    model = _mutable_model_identity(monkeypatch)
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    model["checkpoint"] = "version-2"

    second = _pipeline().run(_request(parent, "B"))

    assert second.success, second.errors
    assert len(inference_calls) == 2
    assert not second.context.metadata["series_reuse_session"].hit
    old_marker = json.loads(_ready_markers(parent, "A")[0].read_text(encoding="utf-8"))
    new_marker = json.loads(_ready_markers(parent, "B")[0].read_text(encoding="utf-8"))
    assert old_marker["segmentation_key"] != new_marker["segmentation_key"]


def test_model_change_during_inference_does_not_publish_cache(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    model = _mutable_model_identity(monkeypatch)
    inference = SegmentationStage._run_nnunet_inference

    def changing_inference(stage, **kwargs):
        result = inference(stage, **kwargs)
        model["checkpoint"] = "version-2"
        return result

    monkeypatch.setattr(SegmentationStage, "_run_nnunet_inference", changing_inference)
    response = _pipeline().run(_request(parent, "A"))

    assert response.success, response.errors
    assert len(inference_calls) == 1
    assert not _ready_markers(parent, "A")
    assert not (parent / "A" / "cache" / "315" / "segmentation" / "series_reuse.json").exists()


def test_model_change_during_reporting_does_not_publish_ready(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    model = _mutable_model_identity(monkeypatch)
    pipeline = _pipeline()
    reporting = next(stage for stage in pipeline.stages if stage.name == "reporting")
    execute = reporting.execute

    def changing_report(context):
        result = execute(context)
        model["checkpoint"] = "version-2"
        return result

    monkeypatch.setattr(reporting, "execute", changing_report)
    response = pipeline.run(_request(parent, "A"))

    assert response.success, response.errors
    assert len(inference_calls) == 1
    assert not _ready_markers(parent, "A")
    subsequent = _pipeline().run(_request(parent, "B"))
    assert subsequent.success, subsequent.errors
    assert len(inference_calls) == 2
    assert _ready_markers(parent, "B")


def test_session_initialization_failure_never_pairs_old_seg_with_rebuilt_raw(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    _mutable_model_identity(monkeypatch)
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    proof = parent / "A" / "cache" / "metadata" / "source_provenance.json"
    old_proof = proof.read_bytes()
    marker = parent / "A" / "cache" / "315" / "segmentation" / "series_reuse.json"
    assert marker.is_file()
    source = parent / "A" / "dicom" / "0020.dcm"
    image = pydicom.dcmread(source)
    pixels = image.pixel_array.copy()
    pixels[20, 20] += 1
    image.PixelData = pixels.astype("<i2", copy=False).tobytes()
    image.save_as(source, enforce_file_format=True)

    def unknown_model(*args):
        raise ValueError("synthetic model source unavailable")

    monkeypatch.setattr(series_reuse, "_model_identity", unknown_model)
    response = _pipeline().run(_request(parent, "A"))

    assert response.success, response.errors
    assert proof.read_bytes() != old_proof
    assert response.context.metadata["series_reuse_requested"] is True
    assert "series_reuse_session" not in response.context.metadata
    assert len(inference_calls) == 2
    assert not marker.exists()
