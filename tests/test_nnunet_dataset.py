"""Numeric ID prefixes must not select another dataset's weights or labels."""

import pytest

from src.core.errors import DependencyError
from src.core.nnunet_dataset import resolve_nnunet_dataset_name
from src.stages.stage_localization import LocalizationStage


def test_exact_id_boundary_ignores_longer_ids_and_deduplicates_roots(tmp_path):
    raw, results = tmp_path / "raw", tmp_path / "results"
    for root in (raw, results):
        (root / "Dataset292_Liver").mkdir(parents=True)
    (raw / "Dataset29222_ctver").mkdir()
    assert resolve_nnunet_dataset_name(292, [str(raw), str(results)]) == "Dataset292_Liver"
    assert resolve_nnunet_dataset_name("29222", [str(raw), str(results)]) == "Dataset29222_ctver"


def test_symlink_deployment_is_resolved_by_dataset_name(tmp_path):
    source = tmp_path / "external-model"
    source.mkdir()
    root = tmp_path / "results"
    root.mkdir()
    (root / "Dataset29222_ctver").symlink_to(source, target_is_directory=True)
    assert resolve_nnunet_dataset_name(29222, [str(root)]) == "Dataset29222_ctver"


def test_true_duplicate_id_remains_an_explicit_error(tmp_path):
    for name in ("Dataset292_Liver", "Dataset292_Spine", "Dataset29222_ctver"):
        (tmp_path / name).mkdir()
    with pytest.raises(DependencyError, match="对应多个数据集名称") as caught:
        resolve_nnunet_dataset_name(292, [str(tmp_path)])
    assert caught.value.details["candidates"] == ["Dataset292_Liver", "Dataset292_Spine"]


def test_longer_id_alone_must_not_fall_back_to_wrong_upstream_match(tmp_path):
    (tmp_path / "Dataset29222_ctver").mkdir()
    with pytest.raises(DependencyError, match="不属于该模型"):
        resolve_nnunet_dataset_name(292, [str(tmp_path)])


def test_missing_dataset_retains_external_cli_diagnosis(tmp_path):
    assert resolve_nnunet_dataset_name(292, ["", str(tmp_path / "missing")]) == "292"


def test_small_id_is_zero_padded_and_relative_roots_are_resolved(tmp_path, monkeypatch):
    from src.core import nnunet_dataset

    (tmp_path / "Dataset001_Test").mkdir()
    monkeypatch.setattr(nnunet_dataset, "resolve_application_path", lambda value: str(tmp_path / value))
    assert resolve_nnunet_dataset_name(1, ["."]) == "Dataset001_Test"


def test_localization_never_reads_labels_from_longer_numeric_id(tmp_path):
    wrong = tmp_path / "Dataset29222_ctver/dataset.json"
    wrong.parent.mkdir()
    wrong.write_text('{}')
    correct = tmp_path / "Dataset292_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"
    correct.parent.mkdir(parents=True)
    correct.write_text('{}')
    stage = LocalizationStage({
        "nnUNet_results": str(tmp_path), "nnUNet_raw": str(tmp_path),
        "nnUNet_preprocessed": str(tmp_path),
    })
    assert stage._find_dataset_json({"model_id": 292}) == correct
    correct.unlink()
    with pytest.raises(FileNotFoundError, match="未找到模型"):
        stage._find_dataset_json({"model_id": 292})
