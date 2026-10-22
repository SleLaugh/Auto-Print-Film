"""肺/髋最终 DICOM 按选定目标居中、共同倍率适配胶片格。"""

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.reconstruction.framing import TargetFramer
from src.reconstruction.framing_context import FramingContext
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image
from src.stages.stage_reconstruction import ReconstructionStage


def _point(value):
    return dict(zip("xyz", map(float, value)))


def _fixture(part, plane, percent, *, oblique=False, drifting=False):
    spacing = np.array([0.8, 1.1, 1.5])
    direction = np.eye(3)
    if oblique:
        yaw, pitch = np.deg2rad([19.0, 13.0])
        direction = np.array([
            [np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1],
        ]) @ np.array([
            [1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)],
        ]) @ np.diag([-1.0, 1.0, -1.0])
    origin = np.array([-45.0, 20.0, 10.0])
    row_axis, col_axis, normal_axis = {
        "横状位": (0, 1, 2), "冠状位": (0, 2, 1), "矢状位": (1, 2, 0),
    }[plane]
    row, col = direction[:, row_axis], direction[:, col_axis]
    normal = np.cross(row, col)
    array = np.zeros((50, 80, 100), dtype=np.uint8)

    def box(row_range, col_range, depth_range):
        indices = [slice(None)] * 3
        for axis, bounds in ((row_axis, row_range), (col_axis, col_range), (normal_axis, depth_range)):
            indices[2 - axis] = slice(*bounds)
        array[tuple(indices)] = 1

    box((18, 58), (12, 32), (8, 16))
    # 第二个主体可跨层平移；倍率不能把这种位移误算成更大的目标尺寸。
    box((40, 80) if drifting else (18, 58), (23, 43) if drifting else (12, 32), (17, 21))
    box((50, 55), (31, 36), (23, 25))
    seg = sitk.GetImageFromArray(array)
    seg.SetSpacing(tuple(spacing))
    seg.SetOrigin(tuple(origin))
    seg.SetDirection(tuple(direction.ravel()))
    # 偏心、很宽的体表和扫描床都不是此次选定部位。
    raw_array = np.full(array.shape, -1024.0, dtype=np.float32)
    raw_array[:, 3:76, 3:96] = 60.0
    raw_array[:, 77:79, 1:99] = 700.0
    raw_array[array == 1] = -600.0 if part == "肺部" else 1200.0
    raw = sitk.GetImageFromArray(raw_array)
    raw.CopyInformation(seg)
    size = np.array(raw.GetSize())
    in_plane_size = size[[row_axis, col_axis]]
    plan = {
        "plan_id": "target_canvas", "slice_part": part, "slice_type": "MPR",
        "plane": plane, "geometry_type": "plane_stack", "group_name": "whole_range",
        "source_labels": ["target"], "fill_percent": percent,
        "slice_thickness": 0.4, "slice_spacing": 3.0,
        "window_presets": ["肺窗", "纵隔窗"] if part == "肺部" else ["骨窗", "软组织窗"],
        "extent_mm": dict(zip(("width", "height"), (in_plane_size - 1) * spacing[[row_axis, col_axis]])),
        "output_spacing_mm": dict(zip("xyz", [*spacing[[row_axis, col_axis]], 0.0])),
        "slices": [],
    }
    for index, depth in enumerate((12, 18, 23, 29), 1):
        source_index = (size - 1) / 2
        source_index[normal_axis] = depth
        if drifting:
            # 计划本身的面内路径仍须保留；固定的是取景平移，不能冻结绝对中心。
            source_index[row_axis] += (index - 1) * 1.5
            source_index[col_axis] -= (index - 1) * 0.5
        plan["slices"].append({
            "index": index, "center": _point(seg.TransformContinuousIndexToPhysicalPoint(tuple(source_index))),
            "row_dir": _point(row), "col_dir": _point(col), "normal": _point(normal),
        })
    plan["range_start"], plan["range_end"] = plan["slices"][0]["center"], plan["slices"][-1]["center"]
    plan["path_points"] = [plan["range_start"], plan["range_end"]]
    return raw, seg, plan, {"target": {"label_ids": [1]}}, in_plane_size


def _target_truth(seg, plan):
    """从源标签体素独立计算全组与各真实厚层边界，不读取 framing。"""

    indices = np.argwhere(sitk.GetArrayViewFromImage(seg) == 1)[:, ::-1]
    voxel_axes = np.array(seg.GetDirection()).reshape(3, 3) * seg.GetSpacing()
    points = indices @ voxel_axes.T + seg.GetOrigin()
    axes = np.column_stack([
        [plan["slices"][0][key][axis] for axis in "xyz"] for key in ("row_dir", "col_dir", "normal")
    ])
    half_voxel = np.abs(voxel_axes.T @ axes).sum(axis=0) / 2

    def bounds(selected):
        projected = selected @ axes[:, :2]
        return projected.min(axis=0) - half_voxel[:2], projected.max(axis=0) + half_voxel[:2]

    group = bounds(points)
    slabs = []
    for item in plan["slices"]:
        center = np.array([item["center"][axis] for axis in "xyz"])
        selected = np.abs((points - center) @ axes[:, 2]) < half_voxel[2] + plan["slice_thickness"] / 2 - 1e-7
        slabs.append(bounds(points[selected]) if np.any(selected) else None)
    return axes, group, slabs


def _run_stage(tmp_path, raw, seg, plan):
    raw_path, seg_path = tmp_path / "raw.mha", tmp_path / "seg.mha"
    sitk.WriteImage(raw, str(raw_path))
    sitk.WriteImage(seg, str(seg_path))
    request = Request(
        str(tmp_path / "source"), str(tmp_path / "out"), slice_part=SlicePartEnum(plan["slice_part"]),
        plane_configs=[PlaneConfig(
            PlaneTypeEnum(plan["plane"]), plan["slice_thickness"], plan["slice_spacing"],
            plan["window_presets"], fill_percent=plan["fill_percent"],
        )],
    )
    context = ProcessingContext(request)
    context.set_result("localization", {
        "raw_for_reconstruction_path": str(raw_path), "seg_for_reconstruction_path": str(seg_path),
        "coords": {"target": {"label_ids": [1]}},
    })
    context.set_result("planning", {"plans": [plan]})
    with patch("src.reconstruction.body_framing.build_body_framing_support", side_effect=AssertionError("体表不参与正式取景")):
        ReconstructionStage({"parallel": plan["fill_percent"] == 100}).execute(context)
    return context.get_result("reconstruction")["outputs"][0]


def _expected_centers_and_size(plan, axes, group, slabs):
    planned = np.array([[item["center"][axis] for axis in "xyz"] for item in plan["slices"]]) @ axes[:, :2]
    valid = [index for index, bounds in enumerate(slabs) if bounds is not None]
    if plan["slice_part"] == "肺部" and valid:
        # 独立目标真值决定参考截面；相同物理面积只忽略浮点投影末尾误差。
        reference = max(valid, key=lambda index: round(float(np.prod(slabs[index][1] - slabs[index][0])), 8))
        lower, upper = slabs[reference]
        shift = (lower + upper) / 2 - planned[reference]
        centers = planned + shift
        maximum = np.max([
            2 * np.maximum(slabs[index][1] - centers[index], centers[index] - slabs[index][0])
            for index in valid
        ], axis=0)
        return centers, maximum, reference, shift
    centers = np.array([(bounds[0] + bounds[1]) / 2 if bounds is not None else (group[0] + group[1]) / 2 for bounds in slabs])
    maximum = np.max([slabs[index][1] - slabs[index][0] for index in valid], axis=0) if valid else group[1] - group[0]
    return centers, maximum, None, None


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("横状位", "冠状位", "矢状位"))
@pytest.mark.parametrize("percent,oblique,drifting", (
    (80, False, False), (100, False, False), (80, True, True), (100, True, True),
))
def test_final_dicom_centers_selected_target_and_fits_film_cell(tmp_path, part, plane, percent, oblique, drifting):
    raw, seg, plan, coords, original_size = _fixture(part, plane, percent, oblique=oblique, drifting=drifting)
    original = copy.deepcopy(plan)
    axes, group, slabs = _target_truth(seg, plan)
    expected_centers, maximum, reference, shift = _expected_centers_and_size(plan, axes, group, slabs)
    sizes = np.array([upper - lower for bounds in slabs if bounds is not None for lower, upper in [bounds]])
    base_spacing = np.array([plan["output_spacing_mm"][axis] for axis in "xy"])
    expected_spacing = base_spacing * max(maximum / (original_size * base_spacing)) / (percent / 100)
    output = _run_stage(tmp_path, raw, seg, plan)
    geometries = []
    display_occupancy = {cell: [] for cell in ((188, 112), (136, 104), (112, 188))}
    framer = TargetFramer(seg, coords, plan)
    for window in plan["window_presets"]:
        for item, bounds, expected_center in zip(plan["slices"], slabs, expected_centers):
            ds = pydicom.dcmread(Path(output["window_outputs"][window]) / f"{item['index']}.dcm")
            size = np.array([int(ds.Columns), int(ds.Rows)])
            spacing = np.array(ds.PixelSpacing, dtype=float)[::-1]
            row, col = np.array(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
            center = np.array(ds.ImagePositionPatient, dtype=float) + row * (size[0] - 1) * spacing[0] / 2 + col * (size[1] - 1) * spacing[1] / 2
            lower, upper = group if bounds is None else bounds
            np.testing.assert_allclose(center @ axes[:, :2], expected_center, atol=1e-5)
            np.testing.assert_allclose(spacing, expected_spacing, rtol=1e-6)
            np.testing.assert_array_equal((original_size - size) % 2, [0, 0])
            assert np.all(size <= original_size)
            assert abs((center - np.array(list(item["center"].values()))) @ axes[:, 2]) < 1e-5
            if bounds is not None:
                assert np.all(center @ axes[:, :2] - size * spacing / 2 <= lower + 1e-5)
                assert np.all(center @ axes[:, :2] + size * spacing / 2 >= upper - 1e-5)
                for cell, occupancy in display_occupancy.items():
                    fit = min(np.array(cell) / (size * spacing))
                    occupancy.append(max((upper - lower) * fit / cell))
            # 最终 HU 像素必须对应居中后的整张画布，不允许事后单侧裁空。
            sampling_plan, framed = framer.plane_geometry(plan, item)
            sampled = PlaneStackResampler().sample_slice(raw, sampling_plan, framed)
            expected = sitk.GetArrayFromImage(apply_window_to_image(sampled, window))[0]
            np.testing.assert_array_equal(ds.pixel_array, expected)
            geometries.append((ds.Columns, ds.Rows, *spacing))
    np.testing.assert_allclose(geometries, [geometries[0]] * len(geometries))
    for cell, occupancy in display_occupancy.items():
        ideal_fov = maximum / (percent / 100)
        ideal_fit = min(np.array(cell) / ideal_fov)
        expected_fill = max(np.max(sizes * ideal_fit / cell, axis=1))
        assert max(occupancy) == pytest.approx(expected_fill, abs=0.035)
        assert max(occupancy) <= percent / 100 + 1e-5
    for details, bounds in zip(output["framing"], slabs):
        assert "body_support_source" not in details
        assert "vertical_source_support" not in details
        if part == "肺部":
            assert details["center_source"] == "group_reference_slice"
            assert details["reference_slice_index"] == plan["slices"][reference]["index"]
            assert details["reference_slice_index"] == 1  # 两个同面积主体并列，取最先计划层。
            np.testing.assert_allclose(details["reference_shift_mm"], shift, atol=1e-8)
            np.testing.assert_allclose(details["reference_target_size_mm"], slabs[reference][1] - slabs[reference][0])
        else:
            assert details["center_source"] == ("group_fallback" if bounds is None else "slice_target")
            assert "reference_slice_index" not in details
    assert plan == original


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("percent", (None, 0))
def test_zero_fill_does_not_compact_target_canvas(part, percent):
    _, seg, plan, coords, original_size = _fixture(part, "横状位", percent)
    framed_plan, framed = TargetFramer(seg, coords, plan).plane_geometry(plan, plan["slices"][0])
    assert framed_plan["extent_mm"] == plan["extent_mm"]
    assert framed_plan["output_spacing_mm"] == plan["output_spacing_mm"]
    assert framed["framing"]["output_size_xy"] == list(original_size)
    assert framed["framing"]["center_source"] == "group_target"


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
def test_empty_planned_slabs_keep_finite_group_target_fallback(part):
    _, seg, plan, coords, _ = _fixture(part, "横状位", 100, drifting=True)
    plan["slices"] = [{**plan["slices"][-1], "index": 1}]
    axes, group, slabs = _target_truth(seg, plan)
    assert slabs == [None]
    framed_plan, framed = TargetFramer(seg, coords, plan).plane_geometry(plan, plan["slices"][0])
    np.testing.assert_allclose(framed["framing"]["framing_target_size_mm"], group[1] - group[0])
    np.testing.assert_allclose(np.array(list(framed["center"].values())) @ axes[:, :2], (group[0] + group[1]) / 2)
    assert framed["framing"]["center_source"] == "group_fallback"
    assert "reference_slice_index" not in framed["framing"]
    assert all(np.isfinite(framed_plan["output_spacing_mm"][axis]) for axis in "xy")


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
def test_zero_fill_stage_keeps_existing_vertical_source_crop(tmp_path, part):
    raw, seg, plan, _, _ = _fixture(part, "冠状位", 0)
    output = _run_stage(tmp_path, raw, seg, plan)
    assert any(frame.get("vertical_source_support", {}).get("crop_top_rows", 0) > 0
               for frame in output["framing"])


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
def test_target_canvas_includes_only_longitudinally_bounded_context(part):
    raw, seg, plan, coords, _ = _fixture(part, "横状位", 100, oblique=True, drifting=True)
    array = sitk.GetArrayFromImage(seg)
    array[12, 35:40, 83:88] = 2
    array[40, 60:65, 90:95] = 2  # 原目标纵向范围外的上下文不能扩大取景。
    contextual = sitk.GetImageFromArray(array)
    contextual.CopyInformation(seg)
    normal = tuple(plan["slices"][0]["normal"][axis] for axis in "xyz")
    framer = TargetFramer(contextual, coords, plan, context=FramingContext((1, 2), normal))
    oracle_array = array.copy()
    oracle_array[40, 60:65, 90:95] = 0
    oracle = sitk.GetImageFromArray((oracle_array > 0).astype(np.uint8))
    oracle.CopyInformation(seg)
    axes, group, slabs = _target_truth(oracle, plan)
    centers, maximum, _, _ = _expected_centers_and_size(plan, axes, group, slabs)
    for item, bounds, expected_center in zip(plan["slices"], slabs, centers):
        sampling_plan, framed = framer.plane_geometry(plan, item)
        np.testing.assert_allclose(framed["framing"]["framing_target_size_mm"], maximum)
        if bounds is not None:
            center = np.array([framed["center"][axis] for axis in "xyz"])
            np.testing.assert_allclose(center @ axes[:, :2], expected_center)
        compact = PlaneStackResampler().sample_slice(raw, sampling_plan, framed)
        # 对称整数裁图须复用旧完整矩阵的采样相位，不发生半像素重采样。
        full_plan = {**sampling_plan, "extent_mm": {
            extent: (raw.GetSize()[axis] - 1) * sampling_plan["output_spacing_mm"][dimension]
            for axis, extent, dimension in ((0, "width", "x"), (1, "height", "y"))
        }}
        full = PlaneStackResampler().sample_slice(raw, full_plan, framed)
        left, top = (np.array(full.GetSize()[:2]) - compact.GetSize()[:2]) // 2
        expected = sitk.GetArrayViewFromImage(full)[:, top:top + compact.GetSize()[1], left:left + compact.GetSize()[0]]
        np.testing.assert_allclose(sitk.GetArrayViewFromImage(compact), expected, atol=1e-4, rtol=0)


def test_lung_reference_uses_largest_actual_section_not_first_or_last():
    _, seg, plan, coords, _ = _fixture("肺部", "横状位", 100, oblique=True, drifting=True)
    array = sitk.GetArrayFromImage(seg)
    array[17:21, 23:44, 40:80] = 1  # 第二个主体比第一个稍高，面积明确最大。
    expanded = sitk.GetImageFromArray(array)
    expanded.CopyInformation(seg)
    axes, group, slabs = _target_truth(expanded, plan)
    centers, required, reference, shift = _expected_centers_and_size(plan, axes, group, slabs)
    assert reference == 1
    framer = TargetFramer(expanded, coords, plan)
    for item, expected_center in zip(plan["slices"], centers):
        _, framed = framer.plane_geometry(plan, item)
        details = framed["framing"]
        assert details["reference_slice_index"] == 2
        np.testing.assert_allclose(details["reference_shift_mm"], shift)
        np.testing.assert_allclose(details["framing_target_size_mm"], required)
        center = np.array([framed["center"][axis] for axis in "xyz"])
        np.testing.assert_allclose(center @ axes[:, :2], expected_center)


def test_hip_does_not_use_lung_reference_strategy():
    _, seg, plan, coords, _ = _fixture("髋骨", "横状位", 100, drifting=True)
    axes, group, slabs = _target_truth(seg, plan)
    framer = TargetFramer(seg, coords, plan)
    with patch.object(framer, "_lung_reference_geometry", side_effect=AssertionError("髋骨保持逐层目标居中")):
        for item, bounds in zip(plan["slices"], slabs):
            _, framed = framer.plane_geometry(plan, item)
            expected = group if bounds is None else bounds
            np.testing.assert_allclose(np.array(list(framed["center"].values())) @ axes[:, :2], (expected[0] + expected[1]) / 2)
            assert "reference_slice_index" not in framed["framing"]
