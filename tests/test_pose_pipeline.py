"""回正方向必须贯穿坐标估计、体积采样和所有部位的三状位计划。"""

import math
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import LabelTarget, PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.stages.stage_localization import LocalizationStage


def _rotation():
    yaw, roll = np.deg2rad((17.0, -23.0))
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ]) @ np.array([
        [np.cos(roll), 0.0, np.sin(roll)],
        [0.0, 1.0, 0.0],
        [-np.sin(roll), 0.0, np.cos(roll)],
    ])


@pytest.mark.parametrize("direction", (_rotation(), _rotation() @ np.diag((-1., 1., 1.))))
def test_xy_resampling_is_invariant_to_oblique_and_reflected_storage(direction):
    """相同局部图像的输出不能随着存储方向而改变或混入相邻z层。"""
    array = np.zeros((5, 65, 65), dtype=np.uint16)
    array[:, 18:45, 27:38] = np.arange(1, 6)[:, None, None] * 100
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 1.2, 2.0))
    image.SetOrigin((12., -51., 32.))
    expected = LocalizationStage._rotate_image_xy(image, 19.0, sitk.sitkNearestNeighbor, 0)
    image.SetDirection(direction.ravel().tolist())
    actual = LocalizationStage._rotate_image_xy(image, 19.0, sitk.sitkNearestNeighbor, 0)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(expected))
    assert actual.GetDirection() == image.GetDirection()
    assert actual.GetSpacing() == image.GetSpacing()


def test_spine_candidate_excludes_truncated_and_missing_labels_before_fitting():
    segmentation = np.zeros((12, 40, 40), dtype=np.uint8)
    segmentation[0:3, 10:20, 10:20] = 1
    segmentation[4:7, 10:20, 10:20] = 2
    segmentation[8:11, 10:20, 10:20] = 3
    targets = {f"vertebrae_L{value}": LabelTarget(f"vertebrae_L{value}", [value]) for value in range(1, 5)}
    stage = LocalizationStage()
    with patch("src.stages.stage_localization.estimate_spine_axial_symmetry", return_value={
        "should_correct": False, "angle_degrees": 0.0, "skipped_label_names": [],
    }) as fit:
        result = stage._build_xy_correction_candidate(
            segmentation, targets, image=sitk.GetImageFromArray(segmentation),
        )
    assert set(fit.call_args.args[0]) == {"vertebrae_L2", "vertebrae_L3"}
    assert result["skipped_label_names"] == ["vertebrae_L1", "vertebrae_L4"]


@pytest.mark.parametrize("part", ("头骨", "髋骨", "肺部"))
def test_non_spine_production_defers_pose_without_allocating_a_pca_point_cloud(part):
    stage = LocalizationStage()
    segmentation = np.ones((5, 20, 20), dtype=np.uint8)
    with patch.object(stage, "_detect_axial_skew", side_effect=AssertionError("不再计算旧PCA")):
        result = stage._build_xy_correction_candidate(
            segmentation, {"target": LabelTarget("target", [1])},
            slice_part=part, image=sitk.GetImageFromArray(segmentation),
        )
    assert result["correction_status"] == "deferred_to_anatomical_frame"
    assert not result["should_correct"]


def test_legacy_pca_aggregation_cannot_reverse_when_the_long_axis_switches():
    segmentation = np.zeros((12, 100, 100), dtype=np.uint8)
    y, x = np.mgrid[:100, :100]
    angle = math.radians(5.0)
    for label, center_y, radii, z_range in ((1, 25, (20, 6), slice(2, 5)), (2, 70, (6, 20), slice(7, 10))):
        local_x = (x - 50) * math.cos(angle) + (y - center_y) * math.sin(angle)
        local_y = -(x - 50) * math.sin(angle) + (y - center_y) * math.cos(angle)
        mask = (local_x / radii[0]) ** 2 + (local_y / radii[1]) ** 2 <= 1
        segmentation[z_range, mask] = label
    targets = {f"vertebrae_L{value}": LabelTarget(f"vertebrae_L{value}", [value]) for value in (1, 2)}
    result = LocalizationStage()._detect_vertebrae_axial_skew(segmentation, targets)
    assert len(result["used_label_names"]) == 2
    assert result["angle_degrees"] == pytest.approx(5.0, abs=0.75)


@pytest.mark.parametrize("part", (SlicePartEnum.头骨, SlicePartEnum.肺部, SlicePartEnum.颈椎,
                                  SlicePartEnum.胸椎, SlicePartEnum.腰椎, SlicePartEnum.髋骨))
def test_every_part_preserves_all_three_anatomical_plane_axes(part):
    axes = _rotation()

    def point(values):
        return dict(zip(("x", "y", "z"), values.tolist()))

    localization = {
        "correction_scope_bbox": {f"{axis}_{side}": value for axis in ("x", "y", "z") for side, value in (("min", 5), ("max", 55))},
        "image": {"size_xyz": [64] * 3, "spacing_xyz": [1.] * 3, "origin_xyz": [0.] * 3, "direction": np.eye(3).ravel().tolist()},
        "anatomical_geometry": {"global_frame": {
            "origin": {"x": 30., "y": 30., "z": 30.},
            "left_right_axis": point(axes[:, 0]),
            "anterior_posterior_axis": point(axes[:, 1]),
            "superior_inferior_axis": point(axes[:, 2]),
            "bounds": {name: {"minimum": 10., "maximum": 50.} for name in ("left_right", "anterior_posterior", "superior_inferior")},
            "method": "test_pose", "confidence": 1., "applied": True, "warnings": [],
        }, "local_frames": {}},
    }
    request = Request("/tmp/dicom", "/tmp/output", slice_part=part)
    expected_axes = ((axes[:, 0], axes[:, 1], axes[:, 2]),
                     (axes[:, 1], -axes[:, 2], -axes[:, 0]),
                     (axes[:, 0], -axes[:, 2], axes[:, 1]))
    for plane, expected in zip((PlaneTypeEnum.横状位, PlaneTypeEnum.矢状位, PlaneTypeEnum.冠状位), expected_axes):
        plan = build_whole_range_plan_from_bbox(request, PlaneConfig(plane, 4., 5., ["骨窗"]), localization)
        for actual, wanted in zip((plan.slices[0].row_dir, plan.slices[0].col_dir, plan.slices[0].normal), expected):
            np.testing.assert_allclose([actual.x, actual.y, actual.z], wanted, atol=1e-6)
