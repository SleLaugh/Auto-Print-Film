"""完整源盒保护的原间距契约，以及贴源边缘人体的安全性。"""

import copy
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.reconstruction.framing import TargetFramer
from src.stages.stage_reconstruction import ReconstructionStage


def _point(value):
    return dict(zip("xyz", map(float, value)))


def _vector(value):
    return np.array([value[axis] for axis in "xyz"])


def _fixture(part, plane, percent, tilted):
    size = np.array([88, 66, 48])
    spacing = np.array([0.9, 1.2, 1.5])
    direction = np.eye(3)
    if tilted:
        yaw, pitch = np.deg2rad([23, 17])
        direction = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1],
        ]) @ np.array([
            [1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]) @ np.diag([-1.0, 1.0, -1.0])
    row_axis, col_axis, depth_axis = {
        "横状位": (0, 1, 2), "冠状位": (0, 2, 1), "矢状位": (1, 2, 0),
    }[plane]
    row, base_col = direction[:, row_axis], direction[:, col_axis]
    col = base_col.copy()
    if tilted:
        # 真正相对源网格斜切；仅旋转源 direction 和计划轴不能覆盖此情形。
        angle = np.deg2rad(13)
        col = base_col * np.cos(angle) + np.cross(row, base_col) * np.sin(angle)
    normal = np.cross(row, col)
    axes = np.column_stack((row, col, normal))
    labels = np.zeros(tuple(size[::-1]), dtype=np.uint8)

    def label_box(row_bounds, col_bounds, depth_bounds):
        ranges = [slice(None)] * 3
        for axis, bounds in ((row_axis, row_bounds), (col_axis, col_bounds), (depth_axis, depth_bounds)):
            ranges[2 - axis] = slice(*bounds)
        labels[tuple(ranges)] = 1

    label_box((20, 45), (12, 26), (7, 13))
    label_box((35, 60), (20, 36), (19, 25))
    label_box((46, 52), (30, 34), (33, 35))
    # 高 HU 外壳不属于任何选定标签：模拟皮肤/肋骨和后方骶尾结构。
    indices = np.indices(labels.shape)[::-1]
    outer = np.any((indices < 6) | (indices >= size[:, None, None, None] - 6), axis=0)
    raw_array = np.full(labels.shape, 40, dtype=np.float32)
    raw_array[outer] = 3000
    raw_array[labels > 0] = -700 if part == "肺部" else 700
    raw, seg = map(sitk.GetImageFromArray, (raw_array, labels))
    for image in (raw, seg):
        image.SetSpacing(tuple(spacing))
        image.SetOrigin((-51.0, 19.0, 7.0))
        image.SetDirection(tuple(direction.ravel()))
    plan = {
        "plan_id": "complete_source", "slice_part": part, "slice_type": "MPR",
        "plane": plane, "geometry_type": "plane_stack", "group_name": "whole_range",
        "source_labels": ["target"], "fill_percent": percent,
        "slice_thickness": 2.4, "slice_spacing": 12.0,
        "window_presets": ["骨窗", "软组织窗"],
        "extent_mm": dict(zip(("width", "height"), (size[[row_axis, col_axis]] - 1) * spacing[[row_axis, col_axis]])),
        "output_spacing_mm": dict(zip("xyz", [*spacing[[row_axis, col_axis]], 0.0])),
        "slices": [],
    }
    for index, depth in enumerate((10, 22, 34, 43), 1):
        center_index = (size - 1) / 2
        center_index[depth_axis] = depth
        center_index[row_axis] += (index - 1) * 1.3
        center_index[col_axis] -= (index - 1) * 0.4
        plan["slices"].append({
            "index": index,
            "center": _point(raw.TransformContinuousIndexToPhysicalPoint(tuple(center_index))),
            "row_dir": _point(row), "col_dir": _point(col), "normal": _point(normal),
        })
    plan["range_start"], plan["range_end"] = plan["slices"][0]["center"], plan["slices"][-1]["center"]
    plan["path_points"] = [plan["range_start"], plan["range_end"]]
    return raw, seg, plan, {"target": {"label_ids": [1]}}, axes


def _source_slab_vertices(source, center, normal, thickness):
    """独立求源连续体素盒与 slab 的交集顶点，不调用生产范围函数。"""
    corners_index = np.array(list(product(*[(-0.5, size - 0.5) for size in source.GetSize()])))
    voxel_axes = np.array(source.GetDirection()).reshape(3, 3) * source.GetSpacing()
    corners = corners_index @ voxel_axes.T + source.GetOrigin()
    depths = (corners - center) @ normal
    half = thickness / 2
    vertices = list(corners[np.abs(depths) <= half + 1e-10])
    for first, second in combinations(range(8), 2):
        if np.count_nonzero(corners_index[first] != corners_index[second]) != 1:
            continue
        span = depths[second] - depths[first]
        if abs(span) < 1e-12:
            continue
        for boundary in (-half, half):
            ratio = (boundary - depths[first]) / span
            if 0 <= ratio <= 1:
                vertices.append(corners[first] + ratio * (corners[second] - corners[first]))
    return np.array(vertices)


def _target_centers(seg, plan, axes):
    """由源标签体素推导中心，避免把 framing 自报的中心作为真值。"""
    indices = np.argwhere(sitk.GetArrayViewFromImage(seg) > 0)[:, ::-1]
    voxel_axes = np.array(seg.GetDirection()).reshape(3, 3) * seg.GetSpacing()
    points = indices @ voxel_axes.T + seg.GetOrigin()
    half_voxel = np.abs(voxel_axes.T @ axes).sum(axis=0) / 2
    projected = points @ axes[:, :2]
    group = (projected.min(axis=0) - half_voxel[:2], projected.max(axis=0) + half_voxel[:2])
    centers = np.array([_vector(item["center"]) for item in plan["slices"]])
    bounds = []
    for center in centers:
        selected = np.abs((points - center) @ axes[:, 2]) < half_voxel[2] + plan["slice_thickness"] / 2 - 1e-7
        current = projected[selected]
        bounds.append((current.min(axis=0) - half_voxel[:2], current.max(axis=0) + half_voxel[:2]) if len(current) else None)
    if not plan.get("fill_percent"):
        projected_centers = np.broadcast_to((group[0] + group[1]) / 2, (len(centers), 2))
    elif plan["slice_part"] == "肺部":
        valid = [index for index, item in enumerate(bounds) if item is not None]
        reference = max(valid, key=lambda index: round(np.prod(bounds[index][1] - bounds[index][0]), 8))
        lower, upper = bounds[reference]
        shift = (lower + upper) / 2 - centers[reference] @ axes[:, :2]
        projected_centers = centers @ axes[:, :2] + shift
    else:
        projected_centers = np.array([(item[0] + item[1]) / 2 if item is not None else (group[0] + group[1]) / 2 for item in bounds])
    return centers + (projected_centers - centers @ axes[:, :2]) @ axes[:, :2].T


def _peripheral_points(source, center, axes):
    """取中心切面四向的无标签亮壳坐标，直接反查最终像素是否仍然可见。"""
    voxel_axes = np.array(source.GetDirection()).reshape(3, 3) * source.GetSpacing()
    inverse = np.linalg.inv(voxel_axes)
    middle_index = (np.array(source.GetSize()) - 1) / 2
    middle = np.array(source.GetOrigin()) + voxel_axes @ middle_index
    middle += axes[:, 2] * ((center - middle) @ axes[:, 2])
    source_middle_index = inverse @ (middle - source.GetOrigin())
    lower, upper = np.full(3, 3.0), np.array(source.GetSize()) - 4.0
    points = []
    for axis in axes[:, :2].T:
        delta = inverse @ axis
        for sign in (-1, 1):
            ray = delta * sign
            distance = np.inf
            for dimension in range(3):
                if abs(ray[dimension]) > 1e-10:
                    boundary = upper[dimension] if ray[dimension] > 0 else lower[dimension]
                    distance = min(distance, (boundary - source_middle_index[dimension]) / ray[dimension])
            if distance > 0:
                points.append(middle + axis * sign * distance)
    return points


def _run_stage(tmp_path, raw, seg, plan):
    raw_path, seg_path = tmp_path / "raw.mha", tmp_path / "seg.mha"
    sitk.WriteImage(raw, str(raw_path))
    sitk.WriteImage(seg, str(seg_path))
    request = Request(str(tmp_path / "source"), str(tmp_path / "out"),
                      slice_part=SlicePartEnum(plan["slice_part"]),
                      plane_configs=[PlaneConfig(PlaneTypeEnum(plan["plane"]), plan["slice_thickness"],
                                                 plan["slice_spacing"], plan["window_presets"],
                                                 fill_percent=plan["fill_percent"])])
    context = ProcessingContext(request)
    context.set_result("localization", {"raw_for_reconstruction_path": str(raw_path),
                                       "seg_for_reconstruction_path": str(seg_path),
                                       "coords": {"target": {"label_ids": [1]}}})
    context.set_result("planning", {"plans": [plan]})
    ReconstructionStage({"parallel": plan["fill_percent"] == 100}).execute(context)
    return context.get_result("reconstruction")["outputs"][0]


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("横状位", "冠状位", "矢状位"))
@pytest.mark.parametrize("percent", (80, 100))
@pytest.mark.parametrize("tilted", (False, True))
def test_final_dicom_preserves_body_anatomy_touching_original_source_edges(tmp_path, part, plane, percent, tilted):
    raw, seg, plan, _, axes = _fixture(part, plane, percent, tilted)
    original = copy.deepcopy(plan)
    expected_centers = _target_centers(seg, plan, axes)
    source_vertices = [_source_slab_vertices(raw, _vector(item["center"]), axes[:, 2], plan["slice_thickness"])
                       for item in plan["slices"]]
    source_required = np.max([2 * np.max(np.abs((points - center) @ axes[:, :2]), axis=0)
                              for points, center in zip(source_vertices, expected_centers)], axis=0)
    output = _run_stage(tmp_path, raw, seg, plan)
    geometries = []
    for window in plan["window_presets"]:
        for item, expected_center, vertices in zip(plan["slices"], expected_centers, source_vertices):
            ds = pydicom.dcmread(Path(output["window_outputs"][window]) / f"{item['index']}.dcm")
            size = np.array([int(ds.Columns), int(ds.Rows)])
            spacing = np.array(ds.PixelSpacing, dtype=float)[::-1]
            row, col = np.array(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
            origin = np.array(ds.ImagePositionPatient, dtype=float)
            center = origin + (size[0] - 1) * spacing[0] * row / 2 + (size[1] - 1) * spacing[1] * col / 2
            if percent == 100:
                details = output["framing"][item["index"] - 1]
                np.testing.assert_allclose(_vector(details["target_center_before_skin_edge"]), expected_center, atol=1e-5)
            else:
                np.testing.assert_allclose(center, expected_center, atol=1e-5)
            # 本例身体实心填满整个源盒，故源边缘同样是须保留的人体边缘。
            local_vertices = (vertices - center) @ axes[:, :2]
            assert np.all(np.abs(local_vertices) <= size * spacing / 2 + 1e-5)
            for peripheral in _peripheral_points(raw, _vector(item["center"]), axes):
                pixel = (peripheral - origin) @ np.column_stack((row, col)) / spacing
                assert np.all(pixel >= -0.5) and np.all(pixel <= size - 0.5)
                x, y = np.rint(pixel).astype(int)
                neighborhood = ds.pixel_array[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                assert neighborhood.size > 0 and neighborhood.max() > 60000
            assert abs((center - _vector(item["center"])) @ axes[:, 2]) < 1e-5
            geometries.append((*size, *spacing))
    np.testing.assert_allclose(geometries, [geometries[0]] * len(geometries))
    for detail in output["framing"]:
        assert detail["source_preservation"]["mode"] == "complete_body_slab"
        if percent != 100:
            assert np.all(np.asarray(detail["source_preservation"]["required_size_mm"]) >= source_required - 1e-6)
            assert detail["source_preservation"]["zoom_limited"] is True
        assert "vertical_source_support" not in detail
        if part == "肺部":
            assert detail["center_source"] == ("group_skin_edge" if percent == 100 else "group_reference_slice")
        else:
            assert detail["center_source"] in ({"slice_skin_edge"} if percent == 100 else {"slice_target", "group_fallback"})
    assert plan == original


@pytest.mark.parametrize("part,percent", [(part, 80) for part in ("颅脑", "颈椎", "胸椎", "腰椎", "脊柱")])
def test_source_preservation_isolated_from_other_parts(part, percent):
    raw, seg, plan, coords, _ = _fixture(part, "横状位", percent, True)
    baseline = TargetFramer(seg, coords, plan)
    guarded = TargetFramer(seg, coords, plan, source_image=raw)
    for item in plan["slices"]:
        assert guarded.plane_geometry(plan, item) == baseline.plane_geometry(plan, item)


def test_source_image_is_not_inspected_for_cpr():
    _, seg, plan, coords, _ = _fixture("肺部", "横状位", 100, True)
    plan["geometry_type"] = "curved_stack"
    plan["slice_type"] = "CPR"

    class UnreadableSource:
        def __getattribute__(self, name):
            raise AssertionError(f"CPR 不能读取平面源保护几何: {name}")

    framer = TargetFramer(seg, coords, plan, source_image=UnreadableSource())
    assert framer.mask.GetSize() == seg.GetSize()


def test_zero_fill_source_preservation_is_independent_of_hu():
    raw, seg, plan, coords, _ = _fixture("肺部", "横状位", 0, True)
    empty = sitk.Image(raw.GetSize(), sitk.sitkFloat32) - 1024.0
    empty.CopyInformation(raw)
    bright = TargetFramer(seg, coords, plan, source_image=raw)
    dark = TargetFramer(seg, coords, plan, source_image=empty)
    for item in plan["slices"]:
        assert bright.plane_geometry(plan, item) == dark.plane_geometry(plan, item)


def test_explicit_body_support_keeps_independent_api_geometry():
    from src.reconstruction.body_framing import BodyFramingSupport

    raw, seg, plan, coords, axes = _fixture("髋骨", "横状位", 80, True)
    voxel_axes = np.array(seg.GetDirection()).reshape(3, 3) * seg.GetSpacing()
    points = np.argwhere(sitk.GetArrayViewFromImage(seg) > 0)[:, ::-1] @ voxel_axes.T + seg.GetOrigin()
    body = BodyFramingSupport(points, voxel_axes, tuple(axes[:, 2]))
    explicit = TargetFramer(seg, coords, plan, body_support=body)
    also_source = TargetFramer(seg, coords, plan, body_support=body, source_image=raw)
    for item in plan["slices"]:
        assert explicit.plane_geometry(plan, item) == also_source.plane_geometry(plan, item)


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
def test_zero_fill_source_preservation_uses_raw_geometry_when_segmentation_grid_differs(part):
    raw, seg, plan, coords, axes = _fixture(part, "横状位", 0, True)
    expanded = sitk.Image([size + 10 for size in raw.GetSize()], sitk.sitkFloat32)
    expanded.SetSpacing(tuple(np.array(raw.GetSpacing()) * 1.1))
    expanded.SetDirection(raw.GetDirection())
    voxel_axes = np.array(raw.GetDirection()).reshape(3, 3) * raw.GetSpacing()
    expanded.SetOrigin(tuple(np.array(raw.GetOrigin()) - voxel_axes @ np.full(3, 3)))
    centers = _target_centers(seg, plan, axes)
    source = TargetFramer(seg, coords, plan, source_image=expanded)
    old_source = TargetFramer(seg, coords, plan, source_image=raw)
    expected_required = np.zeros(2)
    for item, center in zip(plan["slices"], centers):
        vertices = _source_slab_vertices(expanded, _vector(item["center"]), axes[:, 2], plan["slice_thickness"])
        expected_required = np.maximum(expected_required, 2 * np.max(np.abs((vertices - center) @ axes[:, :2]), axis=0))
    for item in plan["slices"]:
        _, framed = source.plane_geometry(plan, item)
        _, previous = old_source.plane_geometry(plan, item)
        details = framed["framing"]["source_preservation"]
        np.testing.assert_allclose(details["required_size_mm"], expected_required, atol=1e-6)
        assert np.all(np.array(details["required_size_mm"]) > previous["framing"]["source_preservation"]["required_size_mm"])
        np.testing.assert_allclose(_vector(framed["center"]), _vector(previous["center"]))


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
def test_slice_entirely_outside_source_does_not_expand_group_field_of_view(part):
    raw, seg, plan, coords, axes = _fixture(part, "横状位", 100, True)
    extended = copy.deepcopy(plan)
    empty = copy.deepcopy(plan["slices"][-1])
    empty["index"] += 1
    empty["center"] = _point(_vector(empty["center"]) + axes @ np.full(3, 10000.0))
    extended["slices"].append(empty)
    original = TargetFramer(seg, coords, plan, source_image=raw)
    guarded = TargetFramer(seg, coords, extended, source_image=raw)
    for item in plan["slices"]:
        actual_plan, actual_slice = guarded.plane_geometry(extended, item)
        expected_plan, expected_slice = original.plane_geometry(plan, item)
        assert actual_slice == expected_slice
        for field in ("extent_mm", "output_spacing_mm"):
            assert actual_plan[field] == expected_plan[field]
    framed_plan, framed_empty = guarded.plane_geometry(extended, empty)
    original_plan, framed_original = original.plane_geometry(plan, plan["slices"][0])
    assert framed_plan["extent_mm"] == original_plan["extent_mm"]
    empty_preservation = framed_empty["framing"]["source_preservation"]
    first_preservation = framed_original["framing"]["source_preservation"]
    for key, value in first_preservation.items():
        if key != "center_adjustment_mm":
            assert empty_preservation[key] == value
    if part == "肺部":
        # 完全无源数据的末层不参与定视野，但仍应沿用整组肺的共同取景平移。
        np.testing.assert_allclose(empty_preservation["center_adjustment_mm"], first_preservation["center_adjustment_mm"])
        empty_shift = (_vector(framed_empty["center"]) - _vector(empty["center"])) @ axes[:, :2]
        first_shift = (_vector(framed_original["center"]) - _vector(plan["slices"][0]["center"])) @ axes[:, :2]
        np.testing.assert_allclose(empty_shift, first_shift, atol=1e-8)


def test_source_slab_touching_box_without_positive_overlap_has_no_support():
    from src.reconstruction.source_framing import SourceImageSupport

    source = sitk.Image([10, 12, 14], sitk.sitkFloat32)
    support = SourceImageSupport(source)
    plan = {"slice_thickness": 1.0}
    axes = np.eye(3)
    for depth in (-1.0, 14.0):
        assert support.slab_bounds(plan, np.array([4.5, 5.5, depth]), axes) is None
        inside = depth + (1e-5 if depth < 0 else -1e-5)
        assert support.slab_bounds(plan, np.array([4.5, 5.5, inside]), axes) is not None


def test_near_axis_guard_preserves_sampled_source_centers_outside_nominal_thickness():
    from src.reconstruction.source_framing import SourceImageSupport

    source = sitk.Image([8, 10, 4], sitk.sitkFloat32)
    source.SetSpacing((1.0, 1.0, 5.0))
    angle = 0.001
    row, col = np.array([np.cos(angle), 0, -np.sin(angle)]), np.array([0, 1, 0])
    normal = np.cross(row, col)
    axes = np.column_stack((row, col, normal))
    center = np.array([3.5, 4.5, 2.4])
    thickness = 1.0
    assert 0 < 1 - normal[2] <= 1e-6  # HU 采样会采用源层重叠分支。
    actual_bounds = SourceImageSupport(source).slab_bounds({"slice_thickness": thickness}, center, axes)
    nominal = _source_slab_vertices(source, center, normal, thickness) @ axes[:, :2]
    sampled_vertices = []
    for depth in range(source.GetSize()[2]):
        physical = np.array(source.TransformIndexToPhysicalPoint((0, 0, depth)))
        offset = (physical - center) @ normal
        overlap = min(offset + 2.5, thickness / 2) - max(offset - 2.5, -thickness / 2)
        if overlap > 0:
            assert abs(offset) > thickness / 2
            # 实际采样平面经过层中心；只按名义 slab 截交会漏掉其外缘。
            sampled_vertices.extend(_source_slab_vertices(source, center + normal * offset, normal, 0.0))
    projected = np.array(sampled_vertices) @ axes[:, :2]
    assert projected[:, 0].min() < nominal[:, 0].min() - 1e-4
    assert projected[:, 0].max() > nominal[:, 0].max() + 1e-4
    assert np.all(projected.min(axis=0) >= actual_bounds[0] - 1e-9)
    assert np.all(projected.max(axis=0) <= actual_bounds[1] + 1e-9)


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("横状位", "冠状位", "矢状位"))
@pytest.mark.parametrize("percent,tilted", ((0, False), (None, True)))
def test_zero_fill_preserves_source_with_common_larger_matrix_and_original_spacing(tmp_path, part, plane, percent, tilted):
    raw, seg, plan, _, axes = _fixture(part, plane, percent, tilted)
    centers = _target_centers(seg, plan, axes)
    output = _run_stage(tmp_path, raw, seg, plan)
    original_spacing = np.array([plan["output_spacing_mm"][axis] for axis in "xy"])
    original_size = np.rint(np.array([plan["extent_mm"][axis] for axis in ("width", "height")]) / original_spacing).astype(int) + 1
    geometries = []
    for window in plan["window_presets"]:
        for item, expected_center in zip(plan["slices"], centers):
            ds = pydicom.dcmread(Path(output["window_outputs"][window]) / f"{item['index']}.dcm")
            size = np.array([int(ds.Columns), int(ds.Rows)])
            spacing = np.array(ds.PixelSpacing, dtype=float)[::-1]
            row, col = np.array(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
            origin = np.array(ds.ImagePositionPatient, dtype=float)
            center = origin + (size[0] - 1) * spacing[0] * row / 2 + (size[1] - 1) * spacing[1] * col / 2
            np.testing.assert_allclose(center, expected_center, atol=1e-5)
            np.testing.assert_allclose(spacing, original_spacing, rtol=1e-7)
            assert np.all(size >= original_size)
            assert np.any(size > original_size)
            np.testing.assert_array_equal((size - original_size) % 2, [0, 0])
            vertices = _source_slab_vertices(raw, _vector(item["center"]), axes[:, 2], plan["slice_thickness"])
            assert np.all(np.abs((vertices - center) @ axes[:, :2]) <= size * spacing / 2 + 1e-5)
            for peripheral in _peripheral_points(raw, _vector(item["center"]), axes):
                pixel = (peripheral - origin) @ np.column_stack((row, col)) / spacing
                assert np.all(pixel >= -0.5) and np.all(pixel <= size - 0.5)
                x, y = np.rint(pixel).astype(int)
                assert ds.pixel_array[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max() > 60000
            geometries.append((*size, *spacing))
    np.testing.assert_allclose(geometries, [geometries[0]] * len(geometries))
    for details in output["framing"]:
        assert details["scale"] == 1.0
        assert details["center_source"] == "group_target"
        assert details["source_preservation"]["mode"] == "complete_source_slab"
        assert details["source_preservation"]["zoom_limited"] is False
        assert "vertical_source_support" not in details
