"""肺部按胸廓共同放大，肺底小目标和空目标层仍须容纳完整躯干。"""

from dataclasses import dataclass

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.body_framing import build_body_framing_support
from src.reconstruction.framing import TargetFramer
from src.reconstruction.plane_stack import PlaneStackResampler


def _point(vector):
    return dict(zip("xyz", map(float, vector)))


@dataclass
class _Thorax:
    raw: sitk.Image
    target: sitk.Image
    trunk: sitk.Image
    plan: dict
    coords: dict
    axes: np.ndarray


def _thorax(*, attached=False, spacing=(1.0, 1.0, 1.5), oblique=False,
            flipped=False, fill_percent=100, separable=True):
    """分别定义胸廓、上臂、骨芯和肺；各源层有独立的躯干外边界真值。

    体位随层变化，因此整卷 bbox 比任一层都宽；共同倍率必须比较各层
    重新居中后的宽度。上臂可在肩部与胸廓相连，末端两层完全没有肺标签。
    """

    shape = (12, 190, 340)
    y, x = np.ogrid[:shape[1], :shape[2]]
    trunk = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.uint8)
    raw = np.full(shape, -1024, dtype=np.int16)
    for z in range(shape[0]):
        cx, cy = 139 + 3 * z, 78 + z
        rx, ry = 64 + 2 * min(z, 10), 46 + z // 2
        body = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        trunk[z] = body
        raw[z, body] = 45
        for side in (-1, 1):
            arm_cx = cx + side * (rx + 26)
            arm = ((x - arm_cx) / 18) ** 2 + ((y - cy) / 29) ** 2 <= 1
            raw[z, arm] = 45
            if attached and (z <= 6 or not separable):
                bridge = (np.abs(y - cy) <= 10) & (x >= min(cx, arm_cx)) & (x <= max(cx, arm_cx))
                raw[z, bridge] = 45
            if separable:
                bone = (x - arm_cx) ** 2 + (y - cy) ** 2 <= 12 ** 2
                raw[z, bone] = 1100
        # 脊柱骨芯保留在胸廓内，不能把它和上臂骨芯同等处理。
        spine = ((x - cx) / 9) ** 2 + ((y - (cy + 29)) / 9) ** 2 <= 1
        raw[z, spine] = 950
        if 1 <= z <= 6:
            for side, label in ((-1, 1), (1, 2)):
                lung = ((x - (cx + side * 29)) / 20) ** 2 + ((y - (cy - 3)) / 25) ** 2 <= 1
                labels[z, lung] = label
                raw[z, lung] = -850
        elif z == 7:
            labels[z, cy + 20:cy + 23, cx + 35:cx + 40] = 2
            raw[z, labels[z] > 0] = -850
    # 薄扫描床独立于身体和上臂。
    raw[:, 176:179, 2:-2] = 650
    direction = np.eye(3)
    if oblique:
        yaw, pitch = np.deg2rad((23, 14))
        direction = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1],
        ]) @ np.array([
            [1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ])
    images = [sitk.GetImageFromArray(array) for array in (raw, labels, trunk.astype(np.uint8))]
    for image in images:
        image.SetSpacing(spacing)
        image.SetOrigin((-123.0, 17.0, 46.0))
        image.SetDirection(tuple(direction.ravel()))
    physical = images[0].TransformContinuousIndexToPhysicalPoint
    depths = (2, 6, 7, 9, 10)
    plan = {
        "plan_id": "lung_trunk", "slice_part": "肺部", "slice_type": "MPR",
        "plane": "横状位", "geometry_type": "plane_stack", "fill_percent": fill_percent,
        "source_labels": ["lung_left", "lung_right"],
        "slice_thickness": 1.8 * spacing[2], "slice_spacing": spacing[2],
        "extent_mm": {"width": 319.0, "height": 319.0},
        "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
        "slices": [{
            "index": number, "center": _point(physical((150.0, 80.0, float(z)))),
            "row_dir": _point(direction[:, 0]), "col_dir": _point(direction[:, 1]),
            "normal": _point(direction[:, 2]),
        } for number, z in enumerate(depths, 1)],
    }
    if flipped:
        images = [sitk.Flip(image, [True, True, True]) for image in images]
    coords = {"lung_left": {"label_ids": [1]}, "lung_right": {"label_ids": [2]}}
    return _Thorax(*images, plan, coords, direction)


def _physical_points(mask):
    indices = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)[:, ::-1]
    voxel_axes = np.asarray(mask.GetDirection()).reshape(3, 3) * mask.GetSpacing()
    return indices @ voxel_axes.T + mask.GetOrigin(), voxel_axes


def _slab_bounds(case, item):
    """由真值实心躯干体素计算正厚度相交范围，不读取取景器的元数据。"""

    points, voxel_axes = _physical_points(case.trunk)
    center = np.array([item["center"][axis] for axis in "xyz"])
    normal = case.axes[:, 2]
    half_support = case.plan["slice_thickness"] / 2 + np.abs(voxel_axes.T @ normal).sum() / 2
    points = points[np.abs((points - center) @ normal) < half_support - 1e-7]
    projected = points @ case.axes[:, :2]
    half_voxel = np.abs(voxel_axes.T @ case.axes[:, :2]).sum(axis=0) / 2
    return projected.min(axis=0) - half_voxel, projected.max(axis=0) + half_voxel


@pytest.mark.parametrize("attached", (False, True))
@pytest.mark.parametrize("spacing,oblique,flipped", (
    ((1.0, 1.0, 1.5), False, False),
    ((0.9, 1.2, 1.7), True, False),
    ((0.9, 1.2, 1.7), True, True),
))
@pytest.mark.parametrize("fill_percent", (80, 100))
def test_lung_trunk_uses_common_scale_without_arm_padding_and_keeps_terminal_contour(
    attached, spacing, oblique, flipped, fill_percent,
):
    case = _thorax(attached=attached, spacing=spacing, oblique=oblique,
                   flipped=flipped, fill_percent=fill_percent)
    support = build_body_framing_support(case.raw, target_mask=case.target, exclude_arms=True)
    assert support is not None
    framer = TargetFramer(case.target, case.coords, case.plan, body_support=support)
    common_spacings, occupancy = [], []
    sampler = PlaneStackResampler()
    for item in case.plan["slices"]:
        plan, framed = framer.plane_geometry(case.plan, item)
        image = sampler.sample_slice(case.raw, plan, framed)
        lower, upper = _slab_bounds(case, item)
        image_center = np.asarray(image.TransformContinuousIndexToPhysicalPoint(
            ((image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2, 0),
        ))
        local_center = image_center @ case.axes[:, :2]
        output_spacing = np.asarray(image.GetSpacing()[:2])
        fov = np.asarray(image.GetSize()[:2]) * output_spacing
        common_spacings.append(output_spacing)
        occupancy.append(np.max((upper - lower) / fov))
        # 上臂允许裁切；独立真值胸廓四个外边界始终应在最终图像内。
        assert np.all(local_center - fov / 2 <= lower + 1e-6)
        assert np.all(local_center + fov / 2 >= upper - 1e-6)
        # 连接肩桥没有唯一的软组织分界，允许保守保留的桥使联合bbox中心
        # 移动两个源像素；完整胸廓的四向覆盖仍须满足上面的精确断言。
        center_tolerance = (2 if attached else 1) * max(spacing[:2])
        np.testing.assert_allclose(local_center, (lower + upper) / 2, atol=center_tolerance)

        # 用最终采样器验证像素覆盖；独立胸廓mask避开允许裁切的手臂像素。
        sampled_truth = sampler.sample_slice(case.trunk, plan, framed)
        visible = np.argwhere(sitk.GetArrayViewFromImage(sampled_truth)[0] > 0.01)
        assert len(visible) > 100
        visible_lower = np.asarray(image.GetOrigin()) @ case.axes[:, :2] + visible.min(axis=0)[::-1] * output_spacing
        visible_upper = np.asarray(image.GetOrigin()) @ case.axes[:, :2] + visible.max(axis=0)[::-1] * output_spacing
        tolerance = 2.5 * output_spacing
        assert np.all(visible_lower <= lower + tolerance)
        assert np.all(visible_upper >= upper - tolerance)
        assert framed["framing"]["center_source"] == "slice_body_contour"
    np.testing.assert_allclose(common_spacings, [common_spacings[0]] * len(common_spacings), atol=1e-10)
    # 分离的臂有确切胸廓边界，最宽层应达到设定占比。连接肩部的软组织
    # 分界没有独立HU边缘，允许保守保留小段连接组织，但不能再按整臂缩小。
    expected_occupancy = fill_percent / 100
    if attached:
        assert 0.9 * expected_occupancy <= max(occupancy) <= expected_occupancy + 0.015
    else:
        assert max(occupancy) == pytest.approx(expected_occupancy, abs=0.015)


def test_lung_keeps_conservative_body_support_when_arm_separation_is_not_reliable():
    case = _thorax(attached=True, separable=False)
    original = build_body_framing_support(case.raw)
    attempted = build_body_framing_support(case.raw, target_mask=case.target, exclude_arms=True)
    assert original is not None and attempted is not None
    original_framer = TargetFramer(case.target, case.coords, case.plan, body_support=original)
    attempted_framer = TargetFramer(case.target, case.coords, case.plan, body_support=attempted)
    for item in case.plan["slices"]:
        expected_plan, expected = original_framer.plane_geometry(case.plan, item)
        actual_plan, actual = attempted_framer.plane_geometry(case.plan, item)
        assert actual_plan == expected_plan
        assert actual["center"] == expected["center"]


def test_coarse_target_geometry_and_raw_storage_flips_keep_the_same_physical_trunk_framing():
    case = _thorax(spacing=(0.9, 1.2, 1.7), oblique=True)
    factor = 2
    coarse_size = [int(np.ceil((size - 1) / factor)) + 1 for size in case.target.GetSize()]
    coarse_reference = sitk.Image(coarse_size, sitk.sitkUInt8)
    coarse_reference.SetOrigin(case.target.GetOrigin())
    coarse_reference.SetDirection(case.target.GetDirection())
    coarse_reference.SetSpacing(tuple(factor * value for value in case.target.GetSpacing()))
    coarse_target = sitk.Resample(case.target, coarse_reference, sitk.Transform(),
                                 sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    assert coarse_target.GetSpacing() != case.raw.GetSpacing()
    assert np.any(sitk.GetArrayViewFromImage(coarse_target) > 0)
    baseline = []
    for flipped in (False, True):
        # 原始CT存储方向可和分割完全不同；分割保持同一个粗物理网格。
        raw = sitk.Flip(case.raw, [True, True, True]) if flipped else case.raw
        support = build_body_framing_support(raw, target_mask=coarse_target, exclude_arms=True)
        assert support is not None
        framer = TargetFramer(coarse_target, case.coords, case.plan, body_support=support)
        spacings, occupancy = [], []
        for index, item in enumerate(case.plan["slices"]):
            plan, framed = framer.plane_geometry(case.plan, item)
            image = PlaneStackResampler().sample_slice(raw, plan, framed)
            lower, upper = _slab_bounds(case, item)
            center = np.asarray(image.TransformContinuousIndexToPhysicalPoint(
                ((image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2, 0),
            ))
            local_center = center @ case.axes[:, :2]
            spacing = np.asarray(image.GetSpacing()[:2])
            fov = np.asarray(image.GetSize()[:2]) * spacing
            assert np.all(local_center - fov / 2 <= lower + 1e-6)
            assert np.all(local_center + fov / 2 >= upper - 1e-6)
            np.testing.assert_allclose(local_center, (lower + upper) / 2, atol=1.2)
            spacings.append(spacing)
            occupancy.append(np.max((upper - lower) / fov))
            if flipped:
                expected_center, expected_spacing, expected_size, expected_pixels = baseline[index]
                np.testing.assert_allclose(center, expected_center, atol=1e-8)
                np.testing.assert_allclose(spacing, expected_spacing, atol=1e-10)
                assert image.GetSize() == expected_size
                # 不同存储次序重采样同一CT，最终HU像素也须保持一致。
                np.testing.assert_allclose(sitk.GetArrayFromImage(image), expected_pixels, atol=0.005)
            else:
                baseline.append((center, spacing, image.GetSize(), sitk.GetArrayFromImage(image)))
        np.testing.assert_allclose(spacings, [spacings[0]] * len(spacings), atol=1e-10)
        assert max(occupancy) == pytest.approx(1.0, abs=0.015)


def test_native_xy_axis_permutation_keeps_physical_trunk_support_and_final_pixels():
    case = _thorax(attached=True, spacing=(0.9, 1.2, 1.7), oblique=True)
    baseline_support = build_body_framing_support(case.raw, target_mask=case.target, exclude_arms=True)
    transposed_raw = sitk.PermuteAxes(case.raw, [1, 0, 2])
    transposed_target = sitk.PermuteAxes(case.target, [1, 0, 2])
    transposed_support = build_body_framing_support(
        transposed_raw, target_mask=transposed_target, exclude_arms=True,
    )
    assert baseline_support is not None and transposed_support is not None
    # 点集按物理坐标比较，不能把原数组x列当成患者左右方向。
    expected_points = {tuple(point) for point in np.round(baseline_support.points_xyz, 7)}
    actual_points = {tuple(point) for point in np.round(transposed_support.points_xyz, 7)}
    assert actual_points == expected_points
    baseline = TargetFramer(case.target, case.coords, case.plan, body_support=baseline_support)
    actual = TargetFramer(transposed_target, case.coords, case.plan, body_support=transposed_support)
    sampler = PlaneStackResampler()
    for item in case.plan["slices"]:
        expected_plan, expected_slice = baseline.plane_geometry(case.plan, item)
        actual_plan, actual_slice = actual.plane_geometry(case.plan, item)
        expected_image = sampler.sample_slice(case.raw, expected_plan, expected_slice)
        actual_image = sampler.sample_slice(transposed_raw, actual_plan, actual_slice)
        np.testing.assert_allclose(actual_image.GetOrigin(), expected_image.GetOrigin(), atol=1e-8)
        np.testing.assert_allclose(actual_image.GetSpacing(), expected_image.GetSpacing(), atol=1e-10)
        assert actual_image.GetSize() == expected_image.GetSize()
        np.testing.assert_allclose(sitk.GetArrayFromImage(actual_image),
                                   sitk.GetArrayFromImage(expected_image), atol=0.005)
