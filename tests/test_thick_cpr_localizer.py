"""厚层 CPR 与脊柱曲面定位图使用细显示网格，物理路径和背景保持对齐。"""

from copy import deepcopy

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.curved_stack import CurvedStackResampler
from src.stages.stage_reconstruction import ReconstructionStage


def _xyz(values):
    return dict(zip("xyz", map(float, values)))


def _image(thickness, reversed_z):
    z, y, x = np.indices((13, 49, 49), dtype=float)
    image = sitk.GetImageFromArray((x + 10.0 * y + 100.0 * z).astype(np.float32))
    image.SetSpacing((0.75, 0.75, thickness))
    image.SetOrigin((-21.0, 13.0, 77.0))
    angle = np.deg2rad(23.0)
    axes = np.array(((np.cos(angle), -np.sin(angle), 0.0),
                     (np.sin(angle), np.cos(angle), 0.0),
                     (0.0, 0.0, -1.0 if reversed_z else 1.0)))
    image.SetDirection(tuple(axes.ravel()))
    return image, axes


def _physical(image, index):
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint(tuple(index)))


def _display_indices(background, physical_points):
    """从实际 DICOM 参考几何独立投影，返回显示图的列、行连续索引。"""

    origin = np.asarray(list(map(float, background.GetMetaData("0020|0032").split("\\"))))
    orientation = np.asarray(list(map(float, background.GetMetaData("0020|0037").split("\\"))))
    delta = np.asarray(physical_points) - origin
    return np.column_stack((delta @ orientation[:3] / background.GetSpacing()[0],
                            delta @ orientation[3:] / background.GetSpacing()[1]))


def _bilinear(array, column, row):
    column = float(np.clip(column, 0.0, array.shape[1] - 1))
    row = float(np.clip(row, 0.0, array.shape[0] - 1))
    left, top = int(np.floor(column)), int(np.floor(row))
    right, bottom = min(left + 1, array.shape[1] - 1), min(top + 1, array.shape[0] - 1)
    dx, dy = column - left, row - top
    return ((1.0 - dy) * ((1.0 - dx) * array[top, left] + dx * array[top, right])
            + dy * ((1.0 - dx) * array[bottom, left] + dx * array[bottom, right]))


def _assert_reference_grid(image, render, fixed_axis):
    background = render.background_image
    assert background.GetSize() == (49, round(12 * image.GetSpacing()[2] / 0.75) + 1)
    np.testing.assert_allclose(background.GetSpacing(), (0.75, 0.75), atol=1e-12)
    z_first = 12 if image.GetDirection()[8] > 0.0 else 0
    first = [0.0, 0.0, float(z_first)]
    last = [48.0, 48.0, float(12 - z_first)]
    first[fixed_axis] = last[fixed_axis] = 24.0
    indices = _display_indices(background, [_physical(image, first), _physical(image, last)])
    np.testing.assert_allclose(indices, ((0.0, 0.0),
                                       (background.GetWidth() - 1, background.GetHeight() - 1)),
                               atol=1e-8)
    assert render.line_alpha.shape == (background.GetHeight(), background.GetWidth())


def _assert_fine_paths(render, physical_paths):
    """实际采样点附近必须有线，线外的大部分解剖背景必须保持可见。"""

    alpha = render.line_alpha
    paths = [_display_indices(render.background_image, points) for points in physical_paths]
    for path in paths:
        inside = path[(path[:, 0] >= 1.0) & (path[:, 0] < alpha.shape[1] - 2)
                      & (path[:, 1] >= 1.0) & (path[:, 1] < alpha.shape[0] - 2)]
        assert len(inside) > 10
        for column, row in inside:
            left, top = int(np.floor(column)), int(np.floor(row))
            assert np.max(alpha[top:top + 2, left:left + 2]) > 0.15

    # 允许双线性抗锯齿边缘，但不能把粗层的整行白带放大到细网格上。
    active = np.argwhere(alpha > 0.05)[:, ::-1]
    expected = np.concatenate(paths)
    distances_squared = np.sum((active[:, None, :] - expected[None, :, :]) ** 2, axis=2)
    assert np.max(np.min(distances_squared, axis=1)) < 1.8 ** 2
    assert np.mean(alpha > 1e-4) < 0.20
    assert np.mean(alpha == 0.0) > 0.75


class _CapturingResampler(CurvedStackResampler):
    def __init__(self):
        super().__init__()
        self.grids = []

    def _sample_linear_grid(self, image, source, points):
        self.grids.append(points.copy())
        return np.zeros(points.shape[:2], dtype=np.float32)


def _actual_cpr_paths(image, plan):
    sampler = _CapturingResampler()
    paths = []
    for layer in plan["slices"]:
        sampler.grids.clear()
        sampler.sample_slice(image, plan, layer)
        # 单列、对称薄 slab 的中心就是正式 CPR 实际采样的中心路径。
        paths.append(np.asarray(sampler.grids).mean(axis=(0, 2)))
    return paths


def _cpr_plan(image, axes, *, spine):
    indices = ((20.0, 9.0, 2.0), (28.0, 32.0, 6.0), (22.0, 14.0, 10.0)) if spine else (
        (24.0, 6.0, 3.0), (24.0, 24.0, 5.0), (24.0, 42.0, 3.0))
    curve = np.asarray([_physical(image, index) for index in indices])
    reference = _physical(image, (24.0, 24.0, 6.0))
    normal = axes[:, 0] if spine else axes[:, 2]
    projected = curve - np.outer((curve - reference) @ axes[:, 1], axes[:, 1]) if spine else curve
    offsets = (-6.0, 0.0, 6.0)
    plan = {
        "plan_id": "thick_spine_cpr" if spine else "thick_ordinary_cpr",
        "slice_part": "胸腰椎" if spine else "肺部", "slice_type": "CPR",
        "plane": "矢状位" if spine else "冠状位", "geometry_type": "curved_stack",
        "localizer_plane": "冠状位" if spine else "矢状位",
        "localizer_reference": _xyz(reference), "curve_points": [_xyz(point) for point in curve],
        "curve_spacing_mm": 0.75, "offset_axis": _xyz(normal), "offsets_mm": list(offsets),
        "extent_mm": {"width": 0.0, "height": float(np.linalg.norm(np.diff(projected, axis=0), axis=1).sum())},
        "output_spacing_mm": {"x": 0.75, "y": 0.75, "z": 0.0},
        "slice_thickness": 0.1, "slice_spacing": 6.0, "window_presets": ["骨窗"],
        "slices": [{"index": index + 1, "offset_mm": offset,
                    "center": _xyz(reference + normal * offset), "normal": _xyz(normal),
                    "row_dir": _xyz(axes[:, 1] if spine else axes[:, 0]),
                    "col_dir": _xyz(axes[:, 2] if spine else axes[:, 1])}
                   for index, offset in enumerate(offsets)],
    }
    if spine:
        plan["curve_projection_axis"] = _xyz(axes[:, 1])
    return plan


@pytest.mark.parametrize("thickness", (10.0, 20.0))
@pytest.mark.parametrize("reversed_z", (False, True))
def test_thick_ordinary_cpr_refines_reference_grid_before_drawing_physical_paths(thickness, reversed_z):
    image, axes = _image(thickness, reversed_z)
    plan = _cpr_plan(image, axes, spine=False)
    original = deepcopy(plan)
    render = ReconstructionStage._build_plan_localizer_render(image, plan)

    _assert_reference_grid(image, render, fixed_axis=0)
    assert render.background_kind == "orthogonal_plane"
    _assert_fine_paths(render, _actual_cpr_paths(image, plan))
    assert plan == original
    np.testing.assert_allclose(image.GetSpacing(), (0.75, 0.75, thickness))
    # 原始 x=24 参考面上的解剖灰度也必须跟细化后的物理位置对齐。
    background = sitk.GetArrayFromImage(render.background_image)
    column, row = _display_indices(render.background_image, [_physical(image, (24.0, 13.0, 6.0))])[0]
    assert _bilinear(background, column, row) == pytest.approx(24.0 + 130.0 + 600.0, abs=1e-3)


@pytest.mark.parametrize("thickness", (10.0, 20.0))
@pytest.mark.parametrize("reversed_z", (False, True))
@pytest.mark.parametrize("part", ("胸腰椎", "全脊柱"))
def test_thick_spine_cpr_preserves_following_background_and_actual_offset_paths(thickness, reversed_z, part):
    image, axes = _image(thickness, reversed_z)
    plan = _cpr_plan(image, axes, spine=True)
    plan["slice_part"] = part
    original = deepcopy(plan)
    render = ReconstructionStage._build_plan_localizer_render(image, plan)

    _assert_reference_grid(image, render, fixed_axis=1)
    assert render.background_kind == "spine_following_coronal"
    assert "AP sampling depth varies" in render.background_image.GetMetaData("0008|2111")
    _assert_fine_paths(render, _actual_cpr_paths(image, plan))
    background = sitk.GetArrayFromImage(render.background_image)
    column, row = _display_indices(render.background_image, [_physical(image, (13.0, 24.0, 6.0))])[0]
    # y=32 的曲面背景必须保留，不能退化为 y=24 的固定冠状层。
    assert _bilinear(background, column, row) == pytest.approx(13.0 + 320.0 + 600.0, abs=1e-3)
    assert plan == original


@pytest.mark.parametrize("thickness", (10.0, 20.0))
@pytest.mark.parametrize("reversed_z", (False, True))
@pytest.mark.parametrize("part", ("胸腰椎", "全脊柱"))
def test_thick_spine_mpr_scales_true_surface_intersection_into_fine_reference_grid(thickness, reversed_z, part):
    image, axes = _image(thickness, reversed_z)
    reference = _physical(image, (24.0, 24.0, 6.0))
    normal = axes @ np.asarray((1.0, 0.5, 0.0)) / np.sqrt(1.25)
    row_direction = axes @ np.asarray((-0.5, 1.0, 0.0)) / np.sqrt(1.25)
    labels = ("vertebrae_C1" if part == "全脊柱" else "vertebrae_C7",
              "vertebrae_T6", "vertebrae_S1")
    anatomy = {"spine_geometry": {"filtered_annotation_centroids": {
        label: {"found": True, "prism_found": True,
                "prism_centroid_physical": _xyz(_physical(image, index))}
        for label, index in zip(labels, ((24.0, 9.0, 2.0), (24.0, 32.0, 6.0), (24.0, 14.0, 10.0)))
    }}}
    plan = {
        "plan_id": "thick_spine_mpr", "slice_part": part, "slice_type": "MPR",
        "plane": "矢状位", "geometry_type": "plane_stack", "localizer_plane": "冠状位",
        "localizer_reference": _xyz(reference), "source_labels": list(labels),
        "slices": [{"index": 1, "center": _xyz(reference), "normal": _xyz(normal),
                    "row_dir": _xyz(row_direction), "col_dir": _xyz(axes[:, 2])}],
    }
    original = deepcopy(plan)
    render = ReconstructionStage._build_plan_localizer_render(image, plan, anatomical_analysis=anatomy)

    _assert_reference_grid(image, render, fixed_axis=1)
    assert render.background_kind == "spine_following_coronal"
    source_z = np.linspace(0.0, 12.0, round(12 * thickness / 0.75) + 1)
    source_y = np.interp(source_z, (2.0, 6.0, 10.0), (9.0, 32.0, 14.0))
    # n·(p-center)=0 与曲面 y(z) 相交，故 x(z)=24-0.5*(y(z)-24)。
    source_x = 24.0 - 0.5 * (source_y - 24.0)
    physical_path = np.asarray([_physical(image, point) for point in zip(source_x, source_y, source_z)])
    _assert_fine_paths(render, [physical_path])
    background = sitk.GetArrayFromImage(render.background_image)
    column, row = _display_indices(render.background_image, [_physical(image, (13.0, 24.0, 6.0))])[0]
    assert _bilinear(background, column, row) == pytest.approx(933.0, abs=1e-3)
    assert plan == original
