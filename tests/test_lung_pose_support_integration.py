"""肺姿态支持只来自本次模型的明确语义标签，不改变原切割目标。"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import LocalizationOperations


@pytest.mark.parametrize("name", ("pleural effusion", "Pleural_Effusion", "pleural   effusion"))
def test_support_resolves_model_labels_without_including_other_foreground(tmp_path, name):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"labels": {
        "background": 0, "lung": 7, name: 13, "pericardial effusion": 23,
        "other": 2,
    }}), encoding="utf-8")
    labels = np.array([[[0, 7, 7, 13, 23, 2]]], dtype=np.uint8)
    target = labels == 7
    original = target.copy()
    support, source = AnatomicalAnalysisStage._lung_pose_support(target, labels, str(path))
    np.testing.assert_array_equal(support, (labels == 7) | (labels == 13))
    np.testing.assert_array_equal(target, original)
    assert source == (name,)


@pytest.mark.parametrize("label_map", (
    {"background": 0, "lung": 7, "pericardial effusion": 13},
    {"background": 0, "lung": 7, "pleural effusion": 0},
    {"background": 0, "lung": 7, "pleural effusion": 15},
))
def test_missing_or_empty_support_keeps_lung_only_path(tmp_path, label_map):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"labels": label_map}), encoding="utf-8")
    labels = np.array([[[0, 7, 13]]], dtype=np.uint8)
    assert AnatomicalAnalysisStage._lung_pose_support(labels == 7, labels, str(path)) == (
        None, (),
    )
    assert AnatomicalAnalysisStage._lung_pose_support(labels == 7, labels, None) == (None, ())


def test_geometry_bridge_passes_support_separately_from_target():
    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    support = target.copy()
    support[1, 1, 2] = True
    sentinel_image = object()
    expected = (object(), {"pose_source": "pleural_envelope"})
    with patch(
        "src.stages.stage_localization.estimate_lung_frame_with_diagnostics",
        return_value=expected,
    ) as estimate:
        result = LocalizationOperations._build_anatomical_geometry(
            "肺部", target, sentinel_image,
            lung_pose_support_mask=support,
            lung_pose_support_labels=("pleural effusion",),
        )
    assert result is expected
    estimate.assert_called_once_with(
        target, sentinel_image, pose_support_mask=support,
        pose_support_source=("pleural effusion",),
    )
