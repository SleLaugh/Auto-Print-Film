"""取景上下文只扩展同类标签，不能混入多器官模型的其他结构。"""

from dataclasses import FrozenInstanceError
import json

import pytest

from src.reconstruction.framing_context import FramingContext, load_framing_context


def _dataset(tmp_path, labels):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return path


def test_spine_context_reads_region_ids_and_excludes_other_anatomy(tmp_path):
    path = _dataset(tmp_path, {
        "background": 0, "vertebrae_T12": 7, "vertebrae_L1": [0, 8, 9],
        "L2": 10, "vertebrae_S1": 11, "sacrum": 12,
        "liver": 13, "hip_left": 14, "vertebrae_C8": 15, "vertebrae_T13": 16,
    })
    context = load_framing_context(path, {"vertebrae_L1": {"label_ids": [8, 9]}})
    assert context == FramingContext((7, 8, 9, 10, 11))
    assert context.axis_xyz == (0.0, 0.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        context.label_ids = ()


@pytest.mark.parametrize("target", ["C1", "C7", "T1", "T12", "L1", "L5", "S1"])
def test_short_names_enable_only_recognized_spine_family(tmp_path, target):
    path = _dataset(tmp_path, {"background": 0, "vertebrae_L1": 1, "vertebrae_L2": 2})
    assert load_framing_context(path, {target: {"label_ids": [1]}}).label_ids == (1, 2)


def test_sacrum_is_kept_only_when_explicitly_selected(tmp_path):
    path = _dataset(tmp_path, {"background": 0, "vertebrae_L5": 2, "sacrum": [3, 4]})
    coords = {"vertebrae_L5": {"label_ids": [2]}, "sacrum": {"label_ids": [3, 4]}}
    assert load_framing_context(path, coords).label_ids == (2, 3, 4)
    assert load_framing_context(path, {"vertebrae_L5": coords["vertebrae_L5"]}).label_ids == (2,)


@pytest.mark.parametrize("other", ["lung", "liver", "hip_left", "C8", "T13", "L6", "S2"])
def test_non_spine_target_does_not_enable_context_or_read_sidecar(tmp_path, other):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    coords = {"vertebrae_L1": {"label_ids": [1]}, other: {"label_ids": [2]}}
    assert load_framing_context(path, coords) is None


def test_zero_or_absent_targets_do_not_enable_context(tmp_path):
    path = _dataset(tmp_path, {"vertebrae_L1": 1, "vertebrae_L2": 2})
    assert load_framing_context(path, {}) is None
    assert load_framing_context(path, {"liver": {"label_ids": [0]}}) is None
    assert load_framing_context(path, {
        "vertebrae_L1": {"label_ids": [1]}, "unselected": {"label_ids": [0]},
    }).label_ids == (1, 2)


@pytest.mark.parametrize("path", [None, ""])
def test_missing_sidecar_argument_preserves_existing_framing(path):
    assert load_framing_context(path, {"vertebrae_L1": {"label_ids": [1]}}) is None


def test_absent_sidecar_file_preserves_existing_framing(tmp_path):
    assert load_framing_context(
        tmp_path / "missing.json", {"vertebrae_L1": {"label_ids": [1]}},
    ) is None


@pytest.mark.parametrize("content", ["{", "null", "[]", "{}", '{"labels": []}'])
def test_existing_corrupt_sidecar_raises(tmp_path, content):
    path = tmp_path / "dataset.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_framing_context(path, {"vertebrae_L1": {"label_ids": [1]}})


@pytest.mark.parametrize("value", [[], "1", 1.5, -1, True, [1, "2"], [None]])
def test_malformed_label_values_raise_instead_of_silently_ignoring_context(tmp_path, value):
    path = _dataset(tmp_path, {"vertebrae_L1": value})
    with pytest.raises(ValueError):
        load_framing_context(path, {"vertebrae_L1": {"label_ids": [1]}})


def test_no_matching_context_labels_returns_none(tmp_path):
    path = _dataset(tmp_path, {"background": 0, "liver": 2, "vertebrae_C8": 3})
    assert load_framing_context(path, {"vertebrae_L1": {"label_ids": [1]}}) is None
