"""矢状位 CPR 的冠状投影采样与定位图保持同一几何。"""

import copy

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.curved_stack import CurvedStackResampler
from src.reconstruction.spine_localizer import build_spine_localizer_surface
from src.stages.stage_reconstruction import ReconstructionStage


_SPINE_PARTS = ("颈椎", "胸椎", "腰椎", "脊柱")


def _xyz(vector):
    return dict(zip("xyz", map(float, vector)))


def _case(part="脊柱", *, oblique=False, thickness=0.1):
    """同时包含左右侧弯和前后弯曲，避免平直中心线掩盖几何不一致。"""

    z, y, x = np.indices((64, 64, 64), dtype=float)
    image = sitk.GetImageFromArray(np.asarray(x + 10.0 * y + 100.0 * z, dtype=np.float32))
    image.SetOrigin((-20.0, -30.0, -40.0))
    image.SetSpacing((0.8, 1.1, 1.3))
    axes = np.eye(3)
    if oblique:
        yaw, pitch = np.deg2rad((17.0, 13.0))
        rz = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                       (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
        rx = np.array(((1.0, 0.0, 0.0), (0.0, np.cos(pitch), -np.sin(pitch)),
                       (0.0, np.sin(pitch), np.cos(pitch))))
        axes = rz @ rx
    image.SetDirection(tuple(axes.ravel()))
    indices = np.array(((29.0, 23.0, 47.0), (35.0, 39.0, 32.0),
                        (30.0, 26.0, 17.0)))
    curve = np.array([
        image.TransformContinuousIndexToPhysicalPoint(tuple(index))
        for index in indices
    ])
    reference = np.array(image.TransformIndexToPhysicalPoint((32, 31, 32)))
    lr, ap, si = axes.T
    projected = curve - np.outer((curve - reference) @ ap, ap)
    height = float(np.linalg.norm(np.diff(projected, axis=0), axis=1).sum())
    offsets = (-8.0, 0.0, 8.0)
    plan = {
        "plan_id": f"{part}_CPR_矢状位",
        "slice_part": part,
        "slice_type": "CPR",
        "plane": "矢状位",
        "geometry_type": "curved_stack",
        "localizer_plane": "冠状位",
        "localizer_reference": _xyz(reference),
        "curve_points": [_xyz(point) for point in curve],
        "curve_projection_axis": _xyz(ap),
        "curve_spacing_mm": 0.8,
        "offset_axis": _xyz(lr),
        "offsets_mm": list(offsets),
        "extent_mm": {"width": 3.2, "height": height},
        "output_spacing_mm": {"x": 0.8, "y": 0.8, "z": 0.0},
        "slice_thickness": thickness,
        "slice_spacing": 8.0,
        "window_presets": ["骨窗"],
        "slices": [{
            "index": index + 1,
            "offset_mm": offset,
            "center": _xyz(reference + offset * lr),
            "normal": _xyz(-lr),
            "row_dir": _xyz(ap),
            "col_dir": _xyz(-si),
        } for index, offset in enumerate(offsets)],
    }
    return image, plan, indices, ap, reference


class _CapturingResampler(CurvedStackResampler):
    def __init__(self):
        super().__init__()
        self.sampled_grids = []

    def _sample_linear_grid(self, image, source, points):
        self.sampled_grids.append(points.copy())
        return np.zeros(points.shape[:2], dtype=np.float32)


@pytest.mark.parametrize("part", _SPINE_PARTS)
@pytest.mark.parametrize("oblique", (False, True))
@pytest.mark.parametrize("thickness", (0.1, 4.0))
def test_projected_localizer_matches_actual_sampling_for_every_offset(part, oblique, thickness):
    """定位线逐行匹配实际中心采样，厚层也不能沿 AP 移动。"""

    image, plan, _, ap, reference = _case(part, oblique=oblique, thickness=thickness)
    original_plan = copy.deepcopy(plan)
    projected = ReconstructionStage._curved_localizer_projected_curves(plan=plan, fixed_axis=1)
    sampler = _CapturingResampler()
    for layer in plan["slices"]:
        sampler.sampled_grids.clear()
        sampler.sample_slice(image, plan, layer)
        grids = np.asarray(sampler.sampled_grids)
        middle_column = grids.shape[2] // 2
        centers = grids[:, :, middle_column, :].mean(axis=0)
        np.testing.assert_allclose(centers, projected[layer["offset_mm"]], atol=1e-10)
        np.testing.assert_allclose(
            grids[:, :, middle_column, :] @ ap, reference @ ap, atol=1e-10,
        )
        np.testing.assert_allclose(np.asarray(projected[layer["offset_mm"]]) @ ap,
                                   reference @ ap, atol=1e-10)
    assert plan == original_plan


@pytest.mark.parametrize("part", _SPINE_PARTS)
@pytest.mark.parametrize("oblique", (False, True))
def test_localizer_background_keeps_original_anterior_posterior_depth(part, oblique):
    """冠状背景仍读取原始 AP 深度，不能跟着正式采样投影变成固定层。"""

    image, plan, indices, _, _ = _case(part, oblique=oblique)
    surface = build_spine_localizer_surface(image, plan)
    assert surface is not None
    expected_depths = np.interp(np.arange(image.GetSize()[2]), indices[::-1, 2], indices[::-1, 1])
    np.testing.assert_allclose(surface.depth_indices, expected_depths, atol=1e-10)
    assert np.ptp(surface.depth_indices) > 10.0
    sampled = sitk.GetArrayFromImage(surface.sample(image))[:, 0, :]
    for raw_index in indices:
        x, y, z = raw_index.astype(int)
        assert sampled[z, x] == pytest.approx(x + 10.0 * y + 100.0 * z)


def test_legacy_plan_without_projection_keeps_three_dimensional_localizer_curve():
    image, plan, _, ap, _ = _case(oblique=True)
    del plan["curve_projection_axis"]
    rows = round(plan["extent_mm"]["height"] / plan["output_spacing_mm"]["y"]) + 1
    raw_curve = np.array([[point[axis] for axis in "xyz"] for point in plan["curve_points"]])
    samples = CurvedStackResampler._resample_curve(raw_curve, rows)
    expected = CurvedStackResampler._offset_curve_samples(
        samples,
        np.array([plan["offset_axis"][axis] for axis in "xyz"]),
        plan["offsets_mm"],
        support_mm=8.0 + plan["slice_thickness"] / 2.0,
    )
    projected = ReconstructionStage._curved_localizer_projected_curves(plan=plan, fixed_axis=1)
    for offset, curve in expected.items():
        np.testing.assert_allclose(projected[offset], curve, atol=1e-10)
    assert np.ptp(np.asarray(projected[0.0]) @ ap) > 10.0
    sampler = _CapturingResampler()
    sampler.sample_slice(image, plan, plan["slices"][1])
    np.testing.assert_allclose(sampler.sampled_grids[0][:, 2, :], projected[0.0], atol=1e-10)
    ReconstructionStage._validate_plan(plan)


@pytest.mark.parametrize("axis", (
    {"x": 0.0, "y": 0.0, "z": 0.0},
    {"x": np.nan, "y": 1.0, "z": 0.0},
    {"x": 0.0, "y": np.inf, "z": 0.0},
    {"x": 0.0, "y": 1.0},
    [0.0, 1.0, 0.0],
))
def test_projection_axis_validation_rejects_invalid_vectors(axis):
    _, plan, _, _, _ = _case()
    plan["curve_projection_axis"] = axis
    with pytest.raises(ValueError, match="curve_projection_axis"):
        ReconstructionStage._validate_plan(plan)


@pytest.mark.parametrize("slice_type,geometry,plane", (
    ("MPR", "plane_stack", "矢状位"),
    ("CPR", "curved_stack", "冠状位"),
    ("CPR", "curved_stack", "横状位"),
))
def test_projection_is_only_allowed_for_sagittal_cpr(slice_type, geometry, plane):
    _, plan, _, _, _ = _case()
    plan.update(slice_type=slice_type, geometry_type=geometry, plane=plane)
    with pytest.raises(ValueError, match="curve_projection_axis 仅支持矢状位 CPR"):
        ReconstructionStage._validate_plan(plan)


@pytest.mark.parametrize("reference", (
    {"x": np.nan, "y": 0.0, "z": 0.0},
    {"x": 0.0, "y": 0.0},
))
def test_projection_requires_finite_slice_center_reference(reference):
    _, plan, _, _, _ = _case()
    plan["slices"][0]["center"] = reference
    with pytest.raises(ValueError, match="参考 center"):
        ReconstructionStage._validate_plan(plan)


def test_valid_projection_plan_passes_validation():
    _, plan, _, _, _ = _case(oblique=True)
    ReconstructionStage._validate_plan(plan)
