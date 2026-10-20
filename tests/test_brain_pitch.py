"""脑形态俯仰的物理方向、回退策略及仅脑区切片范围契约。"""

from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.anatomy.brain import estimate_brain_frame_with_diagnostics
from src.anatomy.orientation import patient_reference_axes


def _rotation_x(angle):
    cosine, sine = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _oblique_direction():
    yaw, side_tilt = np.radians((19.0, -13.0))
    rz = np.asarray([[np.cos(yaw), -np.sin(yaw), 0.0],
                     [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    ry = np.asarray([[np.cos(side_tilt), 0.0, np.sin(side_tilt)],
                     [0.0, 1.0, 0.0], [-np.sin(side_tilt), 0.0, np.cos(side_tilt)]])
    return rz @ ry @ _rotation_x(8.0)


def _brain_case(pitch=20.0, *, radii=(22.0, 36.0, 19.0), oblique=False, flips=(False,) * 3):
    shape = (72, 100, 80)
    spacing = np.asarray((1.1, 0.9, 1.3))
    indices = np.indices(shape).reshape(3, -1).T[:, ::-1]
    center = (np.asarray(shape[::-1]) - 1.0) * spacing / 2.0
    local = indices * spacing - center
    radius = np.sum(((local @ _rotation_x(pitch)) / radii) ** 2, axis=1)
    mask = (radius <= 1.0).reshape(shape)
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing(tuple(spacing))
    image.SetOrigin((183.0, -297.0, 401.0))
    direction = _oblique_direction() if oblique else np.eye(3)
    image.SetDirection(tuple(direction.ravel()))
    if any(flips):
        image = sitk.Flip(image, list(flips))
        mask = sitk.GetArrayFromImage(image) != 0
    return image, mask, direction @ _rotation_x(pitch)


def _axes(frame):
    return np.column_stack([
        list(axis.to_dict().values())
        for axis in (frame.left_right_axis, frame.anterior_posterior_axis, frame.superior_inferior_axis)
    ])


@pytest.mark.parametrize("pitch", (-37.0, -24.0, 20.0, 37.0))
def test_default_pitch_recovers_known_physical_axis_without_symmetry_improvement(pitch):
    image, mask, expected = _brain_case(pitch)
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    frame = result.global_frame
    actual = _axes(frame)
    assert diagnostics["pitch"]["accepted"], diagnostics
    assert not diagnostics["symmetry"]["accepted"], diagnostics
    assert diagnostics["pitch_policy"] == "brain_shape_principal_axis"
    assert diagnostics["pitch"]["angle_degrees"] == pytest.approx(pitch, abs=0.2)
    assert diagnostics["pitch"]["midline_disagreement_degrees"] < 0.2
    np.testing.assert_allclose(actual, expected, atol=0.004)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)
    assert np.linalg.det(actual) == pytest.approx(1.0)
    assert frame.applied
    assert frame.confidence > 0.0
    assert frame.method == "brain_shape_principal_axis"


@pytest.mark.parametrize("flips", tuple(product((False, True), repeat=3)))
def test_oblique_and_reflected_storage_preserve_the_same_physical_pitch(flips):
    image, mask, expected = _brain_case(-18.0, oblique=True, flips=flips)
    original_direction = image.GetDirection()
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    assert diagnostics["pitch"]["accepted"], diagnostics
    np.testing.assert_allclose(_axes(result.global_frame), expected, atol=0.004)
    assert result.global_frame.superior_inferior_axis.z > 0.0
    assert image.GetDirection() == original_direction
    indices = np.argwhere(mask)[:, ::-1]
    points = np.asarray(image.GetOrigin()) + (indices * image.GetSpacing()) @ np.asarray(original_direction).reshape(3, 3).T
    np.testing.assert_allclose(list(result.global_frame.origin.to_dict().values()), points.mean(axis=0), atol=1e-9)


@pytest.mark.parametrize("case,code", (
    ("round", "low_anisotropy"),
    ("large_angle", "angle_out_of_bounds"),
    ("clipped", "brain_touches_image_boundary"),
    ("tiny", "insufficient_voxels"),
))
def test_unreliable_shape_keeps_symmetry_plane_pitch_and_explains_rejection(case, code):
    image, mask, _ = _brain_case(48.0 if case == "large_angle" else 20.0,
                               radii=(22.0, 25.0, 25.0) if case == "round" else (22.0, 36.0, 19.0))
    if case == "clipped":
        mask[0, 50, 40] = True
    elif case == "tiny":
        mask[:] = False
        mask[30:32, 45:48, 36:40] = True
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    pitch = diagnostics["pitch"]
    assert not pitch["accepted"]
    assert pitch["angle_degrees"] == 0.0
    assert pitch["rejection_code"] == code
    assert pitch["reason"]
    assert any("脑形态俯仰未采用" in warning for warning in result.global_frame.warnings)
    if case == "clipped":
        assert diagnostics["coverage"]["truncated"]
        assert diagnostics["coverage"]["source_boundary_contacts"]["z_min"]
        assert any("仅覆盖已采集脑区" in warning for warning in result.global_frame.warnings)
    reference = patient_reference_axes(np.asarray(image.GetDirection()).reshape(3, 3))
    lr = _axes(result.global_frame)[:, 0]
    expected_ap = reference[:, 1] - (reference[:, 1] @ lr) * lr
    np.testing.assert_allclose(_axes(result.global_frame)[:, 1], expected_ap / np.linalg.norm(expected_ap), atol=1e-12)


def test_zero_pitch_keeps_reference_axes_without_claiming_an_applied_correction():
    image, mask, _ = _brain_case(0.0)
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    assert diagnostics["pitch"]["status"] == "unchanged"
    assert diagnostics["pitch"]["rejection_code"] == "below_minimum_correction"
    assert not result.global_frame.applied
    np.testing.assert_allclose(_axes(result.global_frame), np.eye(3), atol=1e-12)


@pytest.mark.parametrize(("input_pitch", "shape_pitch"), ((65., 30.), (-65., -30.)))
def test_pitch_cannot_reverse_patient_superior_direction_in_oblique_input(input_pitch, shape_pitch):
    image, mask, _ = _brain_case(shape_pitch)
    image.SetDirection(tuple(_rotation_x(input_pitch).ravel()))
    baseline, _ = estimate_brain_frame_with_diagnostics(mask, image, pitch_enabled=False)
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    pitch = diagnostics["pitch"]
    assert pitch["candidate_angle_degrees"] == pytest.approx(shape_pitch, abs=0.2)
    assert pitch["status"] == "rejected"
    assert pitch["rejection_code"] == "patient_head_direction_conflict"
    assert not pitch["accepted"]
    assert pitch["angle_degrees"] == 0.
    assert pitch["confidence"] == 0.
    assert result.global_frame.superior_inferior_axis.z > 0.
    np.testing.assert_allclose(_axes(result.global_frame), _axes(baseline.global_frame), atol=1e-12)
    assert any("头脚方向" in warning for warning in result.global_frame.warnings)


def test_oblique_input_still_accepts_pitch_when_patient_superior_direction_is_preserved():
    image, mask, _ = _brain_case(25.)
    image.SetDirection(tuple(_rotation_x(55.).ravel()))
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    assert diagnostics["pitch"]["accepted"], diagnostics
    assert result.global_frame.superior_inferior_axis.z > 0.
    np.testing.assert_allclose(_axes(result.global_frame), _rotation_x(80.), atol=0.004)


def test_disabled_pitch_preserves_the_input_ap_projected_into_the_symmetry_plane():
    image, mask, _ = _brain_case(20.0, oblique=True)
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image, pitch_enabled=False)
    assert diagnostics["pitch"]["status"] == "disabled"
    assert not diagnostics["pitch"]["accepted"]
    np.testing.assert_allclose(_axes(result.global_frame), _oblique_direction(), atol=1e-12)


def test_pitch_rotates_only_ap_si_after_correcting_the_left_right_symmetry_plane():
    image, mask, _ = _brain_case(oblique=True)
    yaw, tilt = np.radians((10.0, 8.0))
    rz = np.asarray([[np.cos(yaw), -np.sin(yaw), 0.0],
                     [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    ry = np.asarray([[np.cos(tilt), 0.0, np.sin(tilt)],
                     [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])
    pose = rz @ ry @ _rotation_x(20.0)
    spacing = np.asarray(image.GetSpacing())
    coordinates = np.indices(mask.shape).reshape(3, -1).T[:, ::-1] * spacing
    coordinates -= (np.asarray(mask.shape[::-1]) - 1.0) * spacing / 2.0
    mask = (np.sum(((coordinates @ pose) / (22.0, 36.0, 19.0)) ** 2, axis=1) <= 1.0).reshape(mask.shape)
    baseline, _ = estimate_brain_frame_with_diagnostics(mask, image, pitch_enabled=False)
    result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    assert diagnostics["symmetry"]["accepted"], diagnostics
    assert diagnostics["pitch"]["accepted"], diagnostics
    np.testing.assert_array_equal(_axes(result.global_frame)[:, 0], _axes(baseline.global_frame)[:, 0])
    expected = _oblique_direction() @ pose
    for index in range(3):
        assert _axes(result.global_frame)[:, index] @ expected[:, index] > np.cos(np.radians(1.0))


def test_all_brain_voxel_corners_fit_final_bounds_in_each_axis():
    image, brain, _ = _brain_case(-20.0, oblique=True, flips=(True, False, True))
    baseline, _ = estimate_brain_frame_with_diagnostics(brain, image, pitch_enabled=False)
    result, diagnostics = estimate_brain_frame_with_diagnostics(brain, image)
    frame = result.global_frame
    assert frame.origin == baseline.global_frame.origin
    assert diagnostics["coverage"]["policy"] == "brain_only"
    assert diagnostics["coverage"]["labels"] == ["brain"]
    assert diagnostics["coverage"]["voxel_count"] == np.count_nonzero(brain)
    assert not any(diagnostics["coverage"]["source_boundary_contacts"].values())
    assert not diagnostics["coverage"]["truncated"]
    axes = _axes(frame)
    indices = np.argwhere(brain)[:, ::-1]
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for corner in product((-0.5, 0.5), repeat=3):
        points = np.asarray(image.GetOrigin()) + ((indices + corner) * image.GetSpacing()) @ direction.T
        projected = points @ axes
        lower = np.minimum(lower, projected.min(axis=0))
        upper = np.maximum(upper, projected.max(axis=0))
    for index, bounds in enumerate((frame.bounds.left_right, frame.bounds.anterior_posterior, frame.bounds.superior_inferior)):
        assert bounds.minimum == pytest.approx(lower[index], abs=1e-9)
        assert bounds.maximum == pytest.approx(upper[index], abs=1e-9)


def test_midline_shape_must_agree_with_whole_brain():
    image, mask, _ = _brain_case()
    spacing = np.asarray(image.GetSpacing())
    xyz = np.indices(mask.shape).reshape(3, -1).T[:, ::-1] * spacing
    local = xyz - (np.asarray(mask.shape[::-1]) - 1.0) * spacing / 2.0
    radius_lr = (local[:, 0] / 28.0) ** 2
    inner = np.abs(local[:, 0]) <= 10.0
    radius = np.empty(len(local))
    for selected, angle in ((inner, -20.0), (~inner, 28.0)):
        coordinates = local[selected] @ _rotation_x(angle)
        radius[selected] = radius_lr[selected] + (coordinates[:, 1] / 36.0) ** 2 + (coordinates[:, 2] / 18.0) ** 2
    mask = (radius <= 1.0).reshape(mask.shape)
    _, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    assert diagnostics["pitch"]["rejection_code"] == "midline_disagreement", diagnostics
    assert diagnostics["pitch"]["midline_disagreement_degrees"] > 5.0


def test_missing_midline_evidence_rejects_pitch():
    image, mask, _ = _brain_case()
    mask[:, :, 25:55] = False
    _, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    assert diagnostics["pitch"]["rejection_code"] == "insufficient_midline_voxels", diagnostics


def test_coordinate_arrays_are_bounded_by_one_layer():
    image, mask, _ = _brain_case()
    original_argwhere = np.argwhere

    def require_layer(array):
        assert array.ndim == 2, "不得构造全卷体素坐标数组"
        return original_argwhere(array)

    with patch("src.anatomy.brain.np.argwhere", side_effect=require_layer):
        _, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
    assert diagnostics["target_point_count"] == np.count_nonzero(mask)
    assert diagnostics["symmetry"]["sample_count"] <= 6000
