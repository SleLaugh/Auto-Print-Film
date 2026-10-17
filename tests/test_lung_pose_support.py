"""胸膜腔支持只能补救肺形态质量不足，不能扩展目标或绕过其他拒绝条件。"""

from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.anatomy import lung


def _pair(yaw=10.0, roll=6.0):
    shape = (70, 72, 110)
    xyz = np.indices(shape).reshape(3, -1).T[:, ::-1].astype(float)
    yaw, roll = np.radians([yaw, roll])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[np.cos(roll), 0.0, -np.sin(roll)],
                   [0.0, 1.0, 0.0], [np.sin(roll), 0.0, np.cos(roll)]])
    pose = rz @ ry
    local = (xyz - [55.0, 36.0, 35.0]) @ pose
    mask = np.zeros(len(local), dtype=bool)
    for cx in (-28.0, 28.0):
        mask |= np.sum(((local - [cx, 0.0, 0.0]) / [17.0, 25.0, 28.0]) ** 2,
                       axis=1) <= 1.0
    mask = mask.reshape(shape)
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    pitch = np.radians(14.0)
    reference = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)],
                          [0.0, np.sin(pitch), np.cos(pitch)]])
    image.SetDirection(tuple(reference.ravel()))
    return mask, image, reference, reference @ pose[:, 0]


def _quality_rejection(points, axes):
    return axes[:, 0], np.mean(points, axis=0), {
        "accepted": False, "status": "rejected", "angle_degrees": 8.0,
        # 文案故意不沿用生产中文，重试必须依据稳定原因码。
        "confidence": 0.0, "reason": "Reflection quality below configured limits",
        "rejection_code": "reflection_quality",
        "warnings": ["目标镜像匹配质量不足，保留输入方向"],
    }


def test_supported_pose_uses_strict_fit_but_lung_controls_center_bounds_and_pitch():
    support, image, reference, expected_lr = _pair()
    target = support.copy()
    target[:24, :, 55:] = False
    real_estimator = lung.estimate_reflection_symmetry_plane
    calls = []

    def first_quality_rejection(points, axes):
        calls.append(len(points))
        return (_quality_rejection(points, axes) if len(calls) == 1
                else real_estimator(points, axes))

    with patch.object(lung, "estimate_reflection_symmetry_plane", first_quality_rejection):
        result, diagnostic = lung.estimate_lung_frame_with_diagnostics(
            target, image, pose_support_mask=support,
            pose_support_source=("pleural effusion",),
        )
    frame = result.global_frame
    assert len(calls) == 2
    assert frame.applied, diagnostic
    assert frame.method == "lung_pleural_envelope_symmetry"
    assert diagnostic["lung_symmetry"]["status"] == "rejected"
    assert diagnostic["pose_support"]["source_labels"] == ["pleural effusion"]
    assert diagnostic["pose_support"]["fallback_source"] == "lung_reflection_quality"
    assert diagnostic["pose_support"]["bilateral_evidence"]["reliable"]
    actual_lr = np.array(list(frame.left_right_axis.to_dict().values()))
    assert actual_lr @ expected_lr > np.cos(np.radians(1.0))
    axes = np.column_stack([list(getattr(frame, key).to_dict().values()) for key in
                            ("left_right_axis", "anterior_posterior_axis",
                             "superior_inferior_axis")])
    expected_ap = reference[:, 1] - (reference[:, 1] @ actual_lr) * actual_lr
    np.testing.assert_allclose(axes[:, 1], expected_ap / np.linalg.norm(expected_ap))
    points = lung.target_mask_physical_points(target, image)
    np.testing.assert_allclose(list(frame.origin.to_dict().values()), points.mean(axis=0))
    assert diagnostic["point_count"] == int(target.sum())
    assert diagnostic["retained_point_count"] == int(target.sum())
    assert diagnostic["pose_support"]["retained_point_count"] > int(target.sum())
    for index, bound in enumerate((frame.bounds.left_right, frame.bounds.anterior_posterior,
                                   frame.bounds.superior_inferior)):
        assert bound.minimum == pytest.approx((points @ axes[:, index]).min())
        assert bound.maximum == pytest.approx((points @ axes[:, index]).max())


def test_reliable_original_lung_pose_does_not_consult_support():
    mask, image, _, _ = _pair()
    expected, _ = lung.estimate_lung_frame_with_diagnostics(mask, image)
    with patch.object(lung, "_try_pose_support", side_effect=AssertionError("unneeded retry")):
        actual, diagnostic = lung.estimate_lung_frame_with_diagnostics(
            mask, image, pose_support_mask=mask,
        )
    assert expected.to_dict() == actual.to_dict()
    assert diagnostic["pose_support"]["status"] == "not_needed"


@pytest.mark.parametrize("invalid", ["shape", "missing_lung", "connected"])
def test_invalid_support_keeps_existing_fallback(invalid):
    mask, image, reference, _ = _pair()
    support = mask.copy()
    if invalid == "shape":
        support = support[:-1]
    elif invalid == "missing_lung":
        support[:, :, 55:] = False
    else:
        support[32:39, 32:40, 25:85] = True
    with patch.object(lung, "estimate_reflection_symmetry_plane", _quality_rejection):
        result, diagnostic = lung.estimate_lung_frame_with_diagnostics(
            mask, image, pose_support_mask=support,
        )
    assert not result.global_frame.applied
    assert diagnostic["pose_support"]["status"] == "rejected"
    np.testing.assert_allclose(list(result.global_frame.left_right_axis.to_dict().values()),
                               reference[:, 0])


def test_failed_support_does_not_override_original_rejection():
    mask, image, _, _ = _pair()
    with patch.object(lung, "estimate_reflection_symmetry_plane",
                      side_effect=_quality_rejection) as estimator:
        result, diagnostic = lung.estimate_lung_frame_with_diagnostics(
            mask, image, pose_support_mask=mask,
        )
    assert estimator.call_count == 2
    assert not result.global_frame.applied
    assert result.global_frame.warnings
    assert diagnostic["pose_support"]["status"] == "rejected"
    assert diagnostic["correction_degrees"] == 0.0


def test_other_rejection_does_not_retry_with_support():
    mask, image, _, _ = _pair()

    def reject_at_limit(points, axes):
        normal, center, diagnostic = _quality_rejection(points, axes)
        diagnostic["reason"] = "镜像面候选触及角度或位移边界，保留输入方向"
        diagnostic["rejection_code"] = ""
        return normal, center, diagnostic

    with patch.object(lung, "estimate_reflection_symmetry_plane", reject_at_limit), \
            patch.object(lung, "_try_pose_support", side_effect=AssertionError("unsafe retry")):
        result, diagnostic = lung.estimate_lung_frame_with_diagnostics(
            mask, image, pose_support_mask=mask,
        )
    assert not result.global_frame.applied
    assert diagnostic["pose_support"]["status"] == "not_needed"


def test_unchanged_support_does_not_claim_applied_rotation():
    mask, image, _, _ = _pair(yaw=0.0, roll=0.0)
    real_estimator = lung.estimate_reflection_symmetry_plane
    calls = []

    def reject_first(points, axes):
        calls.append(len(points))
        return (_quality_rejection(points, axes) if len(calls) == 1
                else real_estimator(points, axes))

    with patch.object(lung, "estimate_reflection_symmetry_plane", reject_first):
        result, diagnostic = lung.estimate_lung_frame_with_diagnostics(
            mask, image, pose_support_mask=mask,
        )
    assert diagnostic["pose_support"]["status"] == "unchanged"
    assert not result.global_frame.applied
    assert diagnostic["correction_degrees"] == 0.0
    assert not result.global_frame.warnings
