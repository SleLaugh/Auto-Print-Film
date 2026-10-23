"""Fill100 把完整人体的共同画布收紧到皮肤，其他 Fill 保持原像素。"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.reconstruction.body_framing import build_body_framing_support
from src.reconstruction.framing import TargetFramer
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image
from tests.test_body_preserving_canvas import (
    _body_vertices, _fixture, _geometry, _stage, _vector,
)
from tests.test_source_preserving_framing import _source_slab_vertices


def _skin_expected(part, centers, bounds):
    """独立几何解：肺统一平移，髋选择离原中心最近的完整容纳位置。"""
    centers = np.asarray(centers, dtype=float)
    lowers, uppers = (np.asarray([item[index] for item in bounds]) for index in (0, 1))
    if part == "肺部":
        lower = np.min(lowers - centers, axis=0)
        upper = np.max(uppers - centers, axis=0)
        shift = (lower + upper) / 2
        return centers + shift, upper - lower
    size = np.max(uppers - lowers, axis=0)
    minimum = uppers - size / 2
    maximum = lowers + size / 2
    return np.minimum(np.maximum(centers, minimum), maximum), size


def _truth_body_bounds(raw, mask, plan, axes):
    """实体人体真值加2mm余量，源画布以外不增加虚构身体。"""
    bounds = []
    for item in plan["slices"]:
        vertices = _body_vertices(mask, raw, item, axes, plan["slice_thickness"])
        projected = vertices @ axes[:, :2]
        source = _source_slab_vertices(raw, _vector(item["center"]), axes[:, 2], plan["slice_thickness"]) @ axes[:, :2]
        bounds.append((np.maximum(projected.min(axis=0) - 2, source.min(axis=0)),
                       np.minimum(projected.max(axis=0) + 2, source.max(axis=0))))
    return bounds


def _target_slab_bounds(seg, plan, axes):
    indices = np.argwhere(sitk.GetArrayViewFromImage(seg) > 0)[:, ::-1]
    voxel_axes = np.asarray(seg.GetDirection()).reshape(3, 3) * seg.GetSpacing()
    points = indices @ voxel_axes.T + seg.GetOrigin()
    half = np.abs(voxel_axes.T @ axes).sum(axis=0) / 2
    result = []
    for item in plan["slices"]:
        selected = points[np.abs((points - _vector(item["center"])) @ axes[:, 2]) < plan["slice_thickness"] / 2 + half[2] - 1e-7]
        if len(selected):
            projected = selected @ axes[:, :2]
            result.append((projected.min(axis=0) - half[:2], projected.max(axis=0) + half[:2]))
        else:
            result.append(None)
    return result


def _skin_fixture(part, plane, flipped=False):
    """人体分层变窄，目标偏前且冠位SI偏心；较窄层允许保持原横向中心。"""
    raw, _, plan, coords, _, axes = _fixture(part, 100, plane=plane)
    shape = tuple(reversed(raw.GetSize()))
    body = np.zeros(shape, dtype=bool)
    body[:7, 35:115, 30:175] = True
    body[7:14, 35:115, 48:160] = True
    body[14:, 35:115, 65:150] = True
    labels = np.zeros(shape, dtype=np.uint8)
    labels[2:7, 48:76, 65:128] = 1
    labels[7:12, 53:80, 95:130] = 1
    labels[14:16, 61:66, 111:116] = 1
    array = np.where(body, 40.0, -1024.0).astype(np.float32)
    array[labels > 0] = -800 if part == "肺部" else 1000
    array[:, 137:140, 3:197] = 1500
    source, seg = (sitk.GetImageFromArray(value) for value in (array, labels))
    source.CopyInformation(raw)
    seg.CopyInformation(raw)
    if flipped:
        # 转存改变存储方向，计划仍是同一组患者物理切面。
        source, seg = (sitk.Flip(value, [True, True, True]) for value in (source, seg))
        body = body[::-1, ::-1, ::-1].copy()
    return source, seg, plan, coords, body, axes


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("横状位", "冠状位"))
@pytest.mark.parametrize("flipped", (False, True))
def test_final_skin_edge_canvas_is_tight_complete_and_preserves_group_policy(tmp_path, part, plane, flipped):
    raw, seg, plan, coords, truth, axes = _skin_fixture(part, plane, flipped)
    original = copy.deepcopy(plan)
    baseline = TargetFramer(seg, coords, plan)
    previous = [baseline.plane_geometry(plan, item)[1] for item in plan["slices"]]
    previous_centers = np.asarray([_vector(item["center"]) for item in previous])
    body_bounds = _truth_body_bounds(raw, truth, plan, axes)
    targets = _target_slab_bounds(seg, plan, axes)
    combined = [(np.minimum(lower, target[0]), np.maximum(upper, target[1])) if target is not None else (lower, upper)
                for (lower, upper), target in zip(body_bounds, targets)]
    expected_centers, expected_fov = _skin_expected(part, previous_centers @ axes[:, :2], combined)
    old_required = np.max([2 * np.maximum(upper - center, center - lower)
                           for (lower, upper), center in zip(combined, previous_centers @ axes[:, :2])], axis=0)
    assert np.prod(expected_fov) < np.prod(old_required) * 0.95
    output = _stage(tmp_path, raw, seg, plan)
    geometries, actual_centers = [], []
    tissue = sitk.GetImageFromArray(truth.astype(np.float32))
    tissue.CopyInformation(raw)
    framer = TargetFramer(seg, coords, plan, source_image=raw)
    for window in plan["window_presets"]:
        low_edges, high_edges = [], []
        for index, (item, bounds) in enumerate(zip(plan["slices"], combined)):
            ds = pydicom.dcmread(Path(output["window_outputs"][window]) / f"{item['index']}.dcm")
            size, spacing, output_axes, origin, center = _geometry(ds)
            np.testing.assert_allclose(center @ axes[:, :2], expected_centers[index], atol=1e-5)
            fov = size * spacing
            assert np.all(fov >= expected_fov - 1e-5)
            assert np.all(fov - expected_fov <= 2 * spacing + 1e-5)
            vertices = _body_vertices(truth, raw, item, axes, plan["slice_thickness"])
            assert np.all(np.abs((vertices - center) @ axes[:, :2]) <= fov / 2 + 1e-5)
            lower, upper = bounds
            low_edges.append(lower - center @ axes[:, :2] + fov / 2)
            high_edges.append(center @ axes[:, :2] + fov / 2 - upper)
            # 不仅几何声明完整：最终像素须沿独立人体真值的四个方向到达体表。
            sample_plan, sample_item = framer.plane_geometry(plan, item)
            sampled_truth = PlaneStackResampler().sample_slice(tissue, sample_plan, sample_item)
            visible = np.argwhere(sitk.GetArrayViewFromImage(sampled_truth)[0] > 0.05)
            assert len(visible) > 0
            actual_lower = origin @ output_axes + visible.min(axis=0)[::-1] * spacing
            actual_upper = origin @ output_axes + visible.max(axis=0)[::-1] * spacing
            expected_projection = vertices @ output_axes
            assert np.all(actual_lower <= expected_projection.min(axis=0) + 2.5 * spacing)
            assert np.all(actual_upper >= expected_projection.max(axis=0) - 2.5 * spacing)
            solid = np.argwhere(sitk.GetArrayViewFromImage(sampled_truth)[0] > 0.95)
            assert len(solid) > 0
            for axis in (0, 1):
                for extreme in (np.argmin, np.argmax):
                    y, x = solid[extreme(solid[:, axis])]
                    assert ds.pixel_array[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max() > 1000
            normal = axes[:, 2]
            assert abs((center - _vector(item["center"])) @ normal) < 1e-5
            geometries.append((*size, *spacing))
            actual_centers.append(center)
        # 决定共同宽/高的真实层应贴边，不能额外留下由目标偏心造成的空气。
        assert np.all(np.min(low_edges, axis=0) <= 2 * spacing + 1e-5)
        assert np.all(np.min(high_edges, axis=0) <= 2 * spacing + 1e-5)
    np.testing.assert_allclose(geometries, [geometries[0]] * len(geometries))
    if part == "肺部":
        planned = np.asarray([_vector(item["center"]) for item in plan["slices"]])
        offsets = np.asarray(actual_centers[:len(planned)]) - planned
        np.testing.assert_allclose(offsets, [offsets[0]] * len(offsets), atol=1e-5)
    else:
        # 髋的较窄层已有可完整容纳皮肤的横向中心时，不能强制改到体表中心。
        if plane == "横状位":
            allowed = [index for index, ((lower, upper), center) in enumerate(zip(combined, previous_centers @ axes[:, :2]))
                       if upper[0] - expected_fov[0] / 2 < center[0] < lower[0] + expected_fov[0] / 2]
            assert allowed
            for index in allowed:
                assert expected_centers[index, 0] == pytest.approx(previous_centers[index] @ axes[:, 0])
    for index, details in enumerate(output["framing"]):
        assert details["center_source"] == ("group_skin_edge" if part == "肺部" else "slice_skin_edge")
        assert details["target_center_source"] == previous[index]["framing"]["center_source"]
        np.testing.assert_allclose(_vector(details["target_center_before_skin_edge"]), previous_centers[index])
        preservation = details["source_preservation"]
        assert preservation["canvas_mode"] == "skin_edge"
        assert preservation["centering_policy"] == ("group_offset" if part == "肺部" else "nearest_body_fitting_center")
        np.testing.assert_allclose(preservation["center_adjustment_mm"], expected_centers[index] - previous_centers[index] @ axes[:, :2], atol=1e-6)
        if part == "肺部":
            np.testing.assert_allclose(details["reference_target_shift_mm"], previous[index]["framing"]["reference_shift_mm"])
            np.testing.assert_allclose(details["reference_shift_mm"],
                                       (_vector(details["center"]) - _vector(plan["slices"][index]["center"])) @ axes[:, :2])
    assert plan == original


# 调整100之前保存的原图几何与双窗PixelData基线；避免把新实现作为自身真值。
_UNCHANGED_HASHES = {
    ("肺部", "横状位", 80): "c4dd84f598c7bf4104e75a81181ecaf485718f9f7358720dd7105ee27ed98619",
    ("肺部", "横状位", 0): "5ba9e8afe3f450e72e8a5cdc678b326880a1c38a82b4319c68894840e4ad2adf",
    ("肺部", "横状位", None): "5ba9e8afe3f450e72e8a5cdc678b326880a1c38a82b4319c68894840e4ad2adf",
    ("肺部", "冠状位", 80): "90340df375501ad17f5e726c3da69034898e5337de628ec4b1952cdaff52d7b1",
    ("肺部", "冠状位", 0): "b468dd75ef8a23b316a09cbf518bf4f8c7ea1e462cab53a1db9c8925d14d81be",
    ("肺部", "冠状位", None): "b468dd75ef8a23b316a09cbf518bf4f8c7ea1e462cab53a1db9c8925d14d81be",
    ("髋骨", "横状位", 80): "b9868d618f04471b03d1d32ff7bd002f180cc48600375055cb1b1b0ee6086891",
    ("髋骨", "横状位", 0): "c8d88255b17b612c475399f7ee621123513dba93db018a504d142d46fcfc3311",
    ("髋骨", "横状位", None): "c8d88255b17b612c475399f7ee621123513dba93db018a504d142d46fcfc3311",
    ("髋骨", "冠状位", 80): "98b55c214a9fdcaeaa4c718eb900250d7b7390764417d3c2c2875b5ccb3355a2",
    ("髋骨", "冠状位", 0): "35ee2891bd429d2b35ff06dc637dc2c4140bbe34226c8faa6d4480c370e0e098",
    ("髋骨", "冠状位", None): "35ee2891bd429d2b35ff06dc637dc2c4140bbe34226c8faa6d4480c370e0e098",
}


@pytest.mark.parametrize("part,plane,percent", _UNCHANGED_HASHES)
def test_non_100_geometry_and_pixels_match_the_previous_body_guard(part, plane, percent):
    raw, seg, plan, coords, _, _ = _fixture(part, percent, plane=plane, tilted=True)
    framer = TargetFramer(seg, coords, plan, source_image=raw)
    digest = hashlib.sha256()
    for item in plan["slices"]:
        sampling_plan, framed = framer.plane_geometry(plan, item)
        geometry = [framed["framing"]["output_size_xy"],
                    [sampling_plan["output_spacing_mm"][key] for key in "xy"],
                    [framed["center"][key] for key in "xyz"]]
        digest.update(json.dumps(geometry, separators=(",", ":")).encode())
        image = PlaneStackResampler().sample_slice(raw, sampling_plan, framed)
        for window in plan["window_presets"]:
            pixels = sitk.GetArrayFromImage(apply_window_to_image(image, window))
            digest.update(np.asarray(pixels, dtype="<u2").tobytes())
        assert "canvas_mode" not in framed["framing"]["source_preservation"]
        assert "target_center_before_skin_edge" not in framed["framing"]
    assert digest.hexdigest() == _UNCHANGED_HASHES[(part, plane, percent)]


def test_explicit_body_support_api_does_not_enable_skin_edge_mode():
    raw, seg, plan, coords, _, _ = _fixture("肺部", 100)
    support = build_body_framing_support(raw)
    framer = TargetFramer(seg, coords, plan, body_support=support, source_image=raw)
    for item in plan["slices"]:
        _, framed = framer.plane_geometry(plan, item)
        assert framed["framing"]["center_source"] == "slice_body_contour"
        assert "source_preservation" not in framed["framing"]
        assert "target_center_before_skin_edge" not in framed["framing"]


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
def test_skin_edge_union_keeps_target_outside_the_recognized_body_contour(part):
    raw, seg, plan, coords, _, axes = _skin_fixture(part, "横状位")
    labels = sitk.GetArrayFromImage(seg)
    labels[3:6, 59:67, 185:194] = 1
    selected = sitk.GetImageFromArray(labels)
    selected.CopyInformation(seg)
    # 此区域在低 HU 原图中不形成外轮廓；目标标签仍须独立参与保护。
    array = sitk.GetArrayFromImage(raw)
    array[3:6, 59:67, 185:194] = -800
    source = sitk.GetImageFromArray(array)
    source.CopyInformation(raw)
    targets = _target_slab_bounds(selected, plan, axes)
    framer = TargetFramer(selected, coords, plan, source_image=source)
    for item, target in zip(plan["slices"], targets):
        if target is None:
            continue
        framed_plan, framed = framer.plane_geometry(plan, item)
        center = _vector(framed["center"]) @ axes[:, :2]
        fov = np.asarray(framed["framing"]["output_size_xy"]) * [framed_plan["output_spacing_mm"][key] for key in "xy"]
        assert np.all(center - fov / 2 <= target[0] + 1e-6)
        assert np.all(center + fov / 2 >= target[1] - 1e-6)


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
def test_unknown_source_layer_keeps_full_source_priority_over_skin_edge_zoom(part):
    from scipy import ndimage

    raw, seg, plan, coords, _, axes = _skin_fixture(part, "横状位")
    array = sitk.GetArrayFromImage(raw)
    y, x = np.ogrid[:array.shape[1], :array.shape[2]]
    large = ((x - 99) / 94) ** 2 + ((y - 79) / 72) ** 2 <= 1
    shell = large & ~ndimage.binary_erosion(large, iterations=2)
    array[4] = -1024
    array[4, large] = -850
    array[4, shell] = 40
    array[4, 65:100, 80:120] = 40
    source = sitk.GetImageFromArray(array)
    source.CopyInformation(raw)
    framer = TargetFramer(seg, coords, plan, source_image=source)
    item = plan["slices"][0]
    framed_plan, framed = framer.plane_geometry(plan, item)
    center = _vector(framed["center"])
    fov = np.asarray(framed["framing"]["output_size_xy"]) * [framed_plan["output_spacing_mm"][key] for key in "xy"]
    vertices = _source_slab_vertices(source, _vector(item["center"]), axes[:, 2], plan["slice_thickness"])
    assert np.all(np.abs((vertices - center) @ axes[:, :2]) <= fov / 2 + 1e-6)
    assert framed["framing"]["source_preservation"]["fallback_source_slice_count"] >= 1
