"""C2 的宽基底不能因齿突长轴和关节翼而被全部丢弃。"""

from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import LabelTarget
from src.stages.stage_localization import LocalizationStage


def _c2_with_dens_and_articular_wings():
    array = np.zeros((52, 100, 200), dtype=np.uint8)
    array[4:24, 10:40, 70:130] = 25  # 下方主体
    array[24:47, 18:33, 88:112] = 25  # 上方细长齿突
    array[10:26, 20:40, 20:180] = 25  # 两侧宽关节翼
    array[4:30, 60:85, 20:180] = 25  # 后方附件
    array[15, 39:61, 99:101] = 25  # 原标签内的细桥
    return array


def test_failed_c2_selection_recovers_existing_inferior_body_only():
    array = _c2_with_dens_and_articular_wings()
    original = array.copy()
    diagnostics = {}

    body = LocalizationStage._vertebral_body_candidate_mask(
        array, LabelTarget("vertebrae_C2", [25]),
        spacing_zyx=(1.0, 0.5, 0.5), diagnostics=diagnostics,
    )

    assert diagnostics["method"] == "c2_inferior_body_recovery"
    assert body[8, 24, 100]  # 恢复下方主体，不能只留下上方齿突。
    assert not np.any(body[26:])
    assert not np.any(body[:, 60:])
    assert np.all(~body | (array == 25))
    np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("label", ("vertebrae_C3", "vertebrae_L1"))
def test_same_failed_shape_does_not_relax_other_vertebrae(label):
    with patch.object(LocalizationStage, "_recover_c2_inferior_body", side_effect=AssertionError):
        body = LocalizationStage._vertebral_body_candidate_mask(
            _c2_with_dens_and_articular_wings(), LabelTarget(label, [25]),
            spacing_zyx=(1.0, 0.5, 0.5),
        )
    assert not np.any(body)


def test_successful_existing_c2_selection_never_enters_recovery():
    array = np.zeros((40, 50, 60), dtype=np.uint8)
    array[4:24, 8:30, 18:42] = 25
    with patch.object(LocalizationStage, "_recover_c2_inferior_body", side_effect=AssertionError):
        body = LocalizationStage._vertebral_body_candidate_mask(
            array, LabelTarget("vertebrae_C2", [25]), spacing_zyx=(1.0, 0.5, 0.5),
        )
    np.testing.assert_array_equal(body, array == 25)


@pytest.mark.parametrize("shape", ("body_only", "dens_only", "detached_superior_fragment"))
def test_missing_connected_body_and_dens_evidence_does_not_recover(shape):
    mask = np.zeros((60, 50, 80), dtype=bool)
    if shape != "dens_only":
        mask[4:24, 8:30, 20:60] = True
    if shape == "dens_only":
        mask[4:46, 14:24, 34:46] = True
    if shape == "detached_superior_fragment":
        mask[50:56, 14:24, 34:46] = True
    original = mask.copy()
    recovered = LocalizationStage._recover_c2_inferior_body(mask, spacing_zyx=(1.0, 0.5, 0.5))
    assert recovered is None
    np.testing.assert_array_equal(mask, original)


@pytest.mark.parametrize("axis,spacing", (
    (0, (0.5, 0.5, 0.5)), (1, (1.0, 0.25, 0.5)), (2, (1.0, 0.5, 0.25)),
))
def test_recovery_follows_physical_support_when_one_axis_is_resampled(axis, spacing):
    mask = _c2_with_dens_and_articular_wings() == 25
    original = LocalizationStage._recover_c2_inferior_body(mask, spacing_zyx=(1.0, 0.5, 0.5))
    resampled = LocalizationStage._recover_c2_inferior_body(np.repeat(mask, 2, axis=axis), spacing_zyx=spacing)
    assert original is not None
    np.testing.assert_array_equal(resampled, np.repeat(original, 2, axis=axis))


@pytest.mark.parametrize("spacing", ((1.0, 0.5, 0.5), (0.5, 0.5, 0.5)))
def test_recovery_maps_identical_body_back_when_si_index_direction_is_reversed(spacing):
    mask = _c2_with_dens_and_articular_wings() == 25
    if spacing[0] == 0.5:
        mask = np.repeat(mask, 2, axis=0)
    forward_diagnostics = {}
    reversed_diagnostics = {}
    forward = LocalizationStage._recover_c2_inferior_body(
        mask, spacing_zyx=spacing, diagnostics=forward_diagnostics,
    )
    reversed_mask = mask[::-1].copy()
    original = reversed_mask.copy()
    reverse = LocalizationStage._recover_c2_inferior_body(
        reversed_mask, spacing_zyx=spacing, diagnostics=reversed_diagnostics,
    )

    assert forward is not None
    np.testing.assert_array_equal(reverse, forward[::-1])
    np.testing.assert_array_equal(reversed_mask, original)
    assert reversed_diagnostics["superior_body_limit_voxel"] == (
        len(mask) - 1 - forward_diagnostics["superior_body_limit_voxel"]
    )


def test_stage5_keeps_existing_annotations_when_c2_is_recovered():
    array = np.zeros((120, 100, 200), dtype=np.uint8)
    array[60:112] = _c2_with_dens_and_articular_wings()
    array[32:52, 10:40, 70:130] = 24
    array[4:24, 10:40, 70:130] = 23
    original = array.copy()
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))
    targets = {f"vertebrae_C{i}": LabelTarget(f"vertebrae_C{i}", [27 - i])
               for i in (2, 3, 4)}
    with patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        with patch.object(LocalizationStage, "_recover_c2_inferior_body", return_value=None):
            baseline = LocalizationStage._build_spine_body_geometry("颈椎", image, array, targets)
        actual = LocalizationStage._build_spine_body_geometry("颈椎", image, array, targets)

    assert actual["filtered_annotation_centroids"] == baseline["filtered_annotation_centroids"]
    assert actual["filtered_annotation_centroids"]["vertebrae_C2"]["found"] is False
    recovered = actual["interspace_recovered_annotations"]["vertebrae_C2"]
    assert recovered["found"] and recovered["prism_found"]
    assert actual["interspace_body_pairs"]["C3_C4"] == baseline["interspace_body_pairs"]["C3_C4"]
    assert "C2_C3" in actual["interspace_body_pairs"]
    np.testing.assert_array_equal(array, original)
