"""显式传入身体支持时的独立取景 API；正式流水线只使用目标标签。"""

import copy
from itertools import product

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

from src.reconstruction.body_framing import (
    BodyFramingSupport,
    build_body_framing_support,
)
from src.reconstruction.framing import TargetFramer
from src.reconstruction.plane_stack import PlaneStackResampler


def _point(values):
    return dict(zip("xyz", map(float, values)))


def _vector(point):
    return np.array([point[axis] for axis in "xyz"], dtype=float)


def _fixture(part="肺部", fill_percent=80, *, oblique=False, flipped=False):
    """身体、肺和床有独立真值；源方向不必与患者轴重合。"""

    spacing = np.array([0.9, 1.2, 1.7])
    direction = np.eye(3)
    if oblique:
        yaw, pitch = np.deg2rad([23.0, 13.0])
        direction = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0],
        ]) @ np.array([
            [1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ])
    if flipped:
        direction = direction @ np.diag([-1.0, -1.0, 1.0])
    origin = np.array([-35.0, 12.0, 6.0])

    def physical(index_xyz):
        return origin + direction @ (np.asarray(index_xyz) * spacing)

    y, x = np.ogrid[:112, :144]
    body = ((x - 68.0) / 55.0) ** 2 + ((y - 48.0) / 35.0) ** 2 <= 1
    labels = np.zeros((10, 112, 144), dtype=np.uint8)
    left = ((x - 48.0) / 13.0) ** 2 + ((y - 44.0) / 17.0) ** 2 <= 1
    right = ((x - 88.0) / 13.0) ** 2 + ((y - 44.0) / 17.0) ** 2 <= 1
    labels[2:6, left] = 1
    labels[2:6, right] = 2
    labels[6, 71:74, 98:102] = 1  # 末层只剩右后方一小块目标。
    raw_array = np.where(np.broadcast_to(body, labels.shape), 100.0, -1024.0)
    if part == "肺部":
        raw_array[labels > 0] = -800.0  # 肺内气体不能把整个胸廓掏空。
    else:
        raw_array[labels > 0] = 1200.0
    raw_array[:, 104:109, 2:142] = 700.0  # 与身体分离的宽床板。
    raw = sitk.GetImageFromArray(raw_array.astype(np.float32))
    raw.SetSpacing(tuple(spacing))
    raw.SetDirection(tuple(direction.ravel()))
    raw.SetOrigin(tuple(origin))
    seg = sitk.GetImageFromArray(labels)
    seg.CopyInformation(raw)
    plan = {
        "plan_id": "body_outline", "slice_part": part, "slice_type": "MPR",
        "plane": "横状位", "geometry_type": "plane_stack", "group_name": "whole_range",
        "source_labels": ["target_left", "target_right"], "fill_percent": fill_percent,
        "slice_thickness": 0.5, "slice_spacing": 1.7,
        "window_presets": ["软组织窗"],
        "extent_mm": {"width": 143 * spacing[0], "height": 111 * spacing[1]},
        "output_spacing_mm": {"x": spacing[0], "y": spacing[1], "z": 0.0},
        "slices": [{
            "index": index, "center": _point(physical((60.0, 42.0, depth))),
            "row_dir": _point(direction[:, 0]), "col_dir": _point(direction[:, 1]),
            "normal": _point(direction[:, 2]),
        } for index, depth in enumerate((3, 6, 7), 1)],
    }
    plan["range_start"] = plan["slices"][0]["center"]
    plan["range_end"] = plan["slices"][-1]["center"]
    plan["path_points"] = [plan["range_start"], plan["range_end"]]
    coords = {"target_left": {"label_ids": [1]}, "target_right": {"label_ids": [2]}}
    return raw, seg, plan, coords, body, physical, direction


def _assert_body_visible(image, source_body, depth, physical, direction):
    """从源体素和最终图像各自计算覆盖，不读取 framing 的范围元数据。"""

    center = np.array(image.TransformContinuousIndexToPhysicalPoint(
        ((image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2, 0),
    ))
    expected_center = physical((68.0, 48.0, depth))
    np.testing.assert_allclose(center, expected_center, atol=1e-6)
    yx = np.argwhere(source_body)
    xyz = np.column_stack((yx[:, 1], yx[:, 0], np.full(len(yx), depth)))
    points = np.array([physical(index) for index in xyz])
    axes = direction[:, :2]
    projected = points @ axes
    half_voxel = np.array([0.9, 1.2]) / 2
    lower = projected.min(axis=0) - half_voxel
    upper = projected.max(axis=0) + half_voxel
    fov_half = np.array(image.GetSize()[:2]) * image.GetSpacing()[:2] / 2
    assert np.all(center @ axes - fov_half <= lower + 1e-6)
    assert np.all(center @ axes + fov_half >= upper - 1e-6)

    # 实际体表像素须到达源轮廓的四个方向；插值只允许一个输出像素量化误差。
    visible = np.argwhere(sitk.GetArrayViewFromImage(image)[0] > -400.0)
    assert len(visible) > 100
    pixel_lower, pixel_upper = visible.min(axis=0)[::-1], visible.max(axis=0)[::-1]
    image_origin = np.array(image.GetOrigin()) @ axes
    visible_lower = image_origin + pixel_lower * image.GetSpacing()[:2]
    visible_upper = image_origin + pixel_upper * image.GetSpacing()[:2]
    tolerance = 1.5 * np.array(image.GetSpacing()[:2]) + half_voxel
    assert np.all(visible_lower <= lower + tolerance)
    assert np.all(visible_upper >= upper - tolerance)
    # 床板并未被拿来居中或扩大视野；上下体表周围都留有空气（Fill100可贴左右边）。
    assert np.any(sitk.GetArrayViewFromImage(image)[0, 0] < -1000)
    assert np.any(sitk.GetArrayViewFromImage(image)[0, -1] < -1000)


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("fill_percent", (80, 100))
@pytest.mark.parametrize("oblique,flipped", ((False, False), (True, False), (True, True)))
def test_body_outline_survives_small_and_empty_terminal_targets(part, fill_percent, oblique, flipped):
    raw, seg, plan, coords, body, physical, direction = _fixture(
        part, fill_percent, oblique=oblique, flipped=flipped,
    )
    original = copy.deepcopy(plan)
    support = build_body_framing_support(raw)
    assert isinstance(support, BodyFramingSupport)
    framer = TargetFramer(seg, coords, plan, body_support=support)
    spacings = []
    normal_positions = []
    for item, depth in zip(plan["slices"], (3, 6, 7)):
        framed_plan, framed = framer.plane_geometry(plan, item)
        image = PlaneStackResampler().sample_slice(raw, framed_plan, framed)
        _assert_body_visible(image, body, depth, physical, direction)
        spacing = np.array(image.GetSpacing()[:2])
        np.testing.assert_allclose(spacing[0] / spacing[1], 0.9 / 1.2)
        spacings.append(spacing)
        assert framed["framing"]["center_source"] == "slice_body_contour"
        assert framed["framing"]["body_support_source"] == "ct_body_contour"
        for field in ("normal", "row_dir", "col_dir"):
            assert framed[field] == item[field]
        normal = _vector(item["normal"])
        assert abs((_vector(framed["center"]) - _vector(item["center"])) @ normal) < 1e-7
        normal_positions.append(_vector(framed["center"]) @ normal)
        assert framed_plan["slice_thickness"] == plan["slice_thickness"]
        assert framed_plan["slice_spacing"] == plan["slice_spacing"]
    np.testing.assert_allclose(spacings, [spacings[0]] * 3)
    np.testing.assert_allclose(np.diff(normal_positions), np.array([3, 1]) * 1.7)
    assert plan == original


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("condition", ("thin_shell", "broken_shell", "one_bad_layer"))
def test_partial_body_recognition_cannot_crop_a_previously_visible_outer_shell(part, condition):
    y, x = np.ogrid[:280, :300]
    body = ((x - 150.0) / 130.0) ** 2 + ((y - 135.0) / 115.0) ** 2 <= 1
    shell = body & ~ndimage.binary_erosion(body, iterations=6)
    raw_array = np.full((3, 280, 300), -1024, dtype=np.int16)
    raw_array[:, body] = -850
    raw_array[:, shell] = 30
    raw_array[:, 195:230, 130:170] = 50
    if condition == "broken_shell":
        raw_array[:, 20:40, 150] = -850
    elif condition == "one_bad_layer":
        raw_array[0, body] = 30
        raw_array[2, body] = 30
    labels = np.zeros(raw_array.shape, dtype=np.uint8)
    labels[:, 125:145, 35:130] = 1
    labels[:, 125:145, 170:265] = 2
    raw, seg = (sitk.GetImageFromArray(array) for array in (raw_array, labels))
    plan = {
        "plan_id": "partial_outer_shell", "slice_part": part, "slice_type": "MPR",
        "plane": "横状位", "geometry_type": "plane_stack", "fill_percent": 80,
        "source_labels": ["target_left", "target_right"],
        "slice_thickness": 0.5, "slice_spacing": 1.0,
        "extent_mm": {"width": 319.0, "height": 319.0},
        "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
        "slices": [{
            "index": z + 1, "center": _point((149.5, 134.5, z)),
            "row_dir": _point((1, 0, 0)), "col_dir": _point((0, 1, 0)),
            "normal": _point((0, 0, 1)),
        } for z in range(3)],
    }
    coords = {"target_left": {"label_ids": [1]}, "target_right": {"label_ids": [2]}}
    support = build_body_framing_support(raw)
    supported = TargetFramer(seg, coords, plan, body_support=support)
    body_yx = np.argwhere(body)
    lower = body_yx.min(axis=0)[::-1] - 0.5
    upper = body_yx.max(axis=0)[::-1] + 0.5
    for item in plan["slices"]:
        framed_plan, framed = supported.plane_geometry(plan, item)
        image = PlaneStackResampler().sample_slice(raw, framed_plan, framed)
        center = np.array(image.TransformContinuousIndexToPhysicalPoint(
            ((image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2, 0),
        ))[:2]
        half_fov = np.array(image.GetSize()[:2]) * image.GetSpacing()[:2] / 2
        if support is None:
            # 被可靠性检查拒绝的外壳不构成显示支持；此时恢复目标取景，
            # 不再要求未选中的体表被目标画布完整容纳。
            target = np.argwhere(labels[0] > 0)[:, ::-1]
            target_lower, target_upper = target.min(axis=0) - 0.5, target.max(axis=0) + 0.5
            np.testing.assert_allclose(center, (target_lower + target_upper) / 2)
            assert np.all(center - half_fov <= target_lower + 1e-6)
            assert np.all(center + half_fov >= target_upper - 1e-6)
            assert framed["framing"]["center_source"] == ("group_reference_slice" if part == "肺部" else "slice_target")
            assert "body_support_source" not in framed["framing"]
            continue
        assert np.all(center - half_fov <= lower + 1e-6)
        assert np.all(center + half_fov >= upper - 1e-6)

        # 从最终采样像素反查体表四向覆盖，不能只相信 framing 的范围声明。
        visible = np.argwhere(sitk.GetArrayViewFromImage(image)[0] > -400)
        assert len(visible) > 0
        pixel_lower = visible.min(axis=0)[::-1]
        pixel_upper = visible.max(axis=0)[::-1]
        spacing = np.array(image.GetSpacing()[:2])
        visible_lower = np.array(image.GetOrigin()[:2]) + pixel_lower * spacing
        visible_upper = np.array(image.GetOrigin()[:2]) + pixel_upper * spacing
        tolerance = 2 * spacing + 0.5
        assert np.all(visible_lower <= lower + tolerance)
        assert np.all(visible_upper >= upper - tolerance)
        if framed["framing"]["center_source"] == "slice_body_contour":
            # 混合好/坏层的共用倍率可能掩盖裁切，但内部块仍不能冒充身体居中。
            np.testing.assert_allclose(center, (lower + upper) / 2, atol=1.0)

@pytest.mark.parametrize("part,fill_percent", (
    ("肺部", 0), ("肺部", None), ("髋骨", 0), ("髋骨", None),
    ("颅脑", 80), ("颈椎", 80), ("胸椎", 80), ("腰椎", 80), ("胸腰椎", 80), ("全脊柱", 80),
))
def test_body_support_does_not_change_existing_non_body_framing_contract(part, fill_percent):
    raw, seg, plan, coords, _, _, _ = _fixture(part, fill_percent, oblique=True, flipped=True)
    support = build_body_framing_support(raw)
    baseline = TargetFramer(seg, coords, plan)
    supported = TargetFramer(seg, coords, plan, body_support=support)
    for item in plan["slices"]:
        expected_plan, expected = baseline.plane_geometry(plan, item)
        actual_plan, actual = supported.plane_geometry(plan, item)
        assert actual_plan == expected_plan
        assert actual == expected


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("冠状位", "矢状位"))
@pytest.mark.parametrize("fill_percent", (80, 100))
@pytest.mark.parametrize("oblique", (False, True))
def test_other_mpr_planes_keep_complete_in_plane_body_cross_section(part, plane, fill_percent, oblique):
    raw, seg, plan, coords, body, physical, direction = _fixture(
        part, fill_percent, oblique=oblique, flipped=oblique,
    )
    if plane == "冠状位":
        row_axis, normal_axis, depths = 0, 1, (44, 72, 80)
    else:
        row_axis, normal_axis, depths = 1, 0, (48, 100, 116)
    row, col = direction[:, row_axis], direction[:, 2]
    normal = np.cross(row, col)
    plan["plane"] = plane
    source_spacing = np.array(raw.GetSpacing())
    plan["output_spacing_mm"] = {"x": source_spacing[row_axis], "y": source_spacing[2], "z": 0.0}
    plan["extent_mm"] = {
        "width": (raw.GetSize()[row_axis] - 1) * source_spacing[row_axis],
        "height": (raw.GetSize()[2] - 1) * source_spacing[2],
    }
    for item, depth in zip(plan["slices"], depths):
        index = np.array([60.0, 42.0, 4.0])
        index[normal_axis] = depth
        item.update(center=_point(physical(index)), row_dir=_point(row),
                    col_dir=_point(col), normal=_point(normal))
    support = build_body_framing_support(raw)
    framer = TargetFramer(seg, coords, plan, body_support=support)
    yx = np.argwhere(body)
    # 冠/矢只要求原肺 SI 范围的身体横截面完整，身体不能扩展切割纵向范围。
    source_indices = np.concatenate([
        np.column_stack((yx[:, 1], yx[:, 0], np.full(len(yx), z))) for z in range(2, 7)
    ])
    source_points = np.array([physical(index) for index in source_indices])
    axes = np.column_stack((row, col))
    spacings = []
    for item, depth in zip(plan["slices"], depths):
        framed_plan, framed = framer.plane_geometry(plan, item)
        image = PlaneStackResampler().sample_slice(raw, framed_plan, framed)
        center = np.array(image.TransformContinuousIndexToPhysicalPoint(
            ((image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2, 0),
        ))
        selected = np.abs(source_indices[:, normal_axis] - depth) < 0.5
        projected = source_points[selected] @ axes
        half_voxel = np.array([source_spacing[row_axis], source_spacing[2]]) / 2
        lower, upper = projected.min(axis=0) - half_voxel, projected.max(axis=0) + half_voxel
        fov_half = np.array(image.GetSize()[:2]) * image.GetSpacing()[:2] / 2
        assert np.all(center @ axes - fov_half <= lower + 1e-6)
        assert np.all(center @ axes + fov_half >= upper - 1e-6)
        np.testing.assert_allclose(center @ axes, (lower + upper) / 2, atol=1e-6)
        assert framed["framing"]["center_source"] == "slice_body_contour"
        np.testing.assert_allclose(np.array(image.GetSpacing()[:2]) / source_spacing[[row_axis, 2]],
                                   [image.GetSpacing()[0] / source_spacing[row_axis]] * 2)
        actual = np.argwhere(sitk.GetArrayViewFromImage(image)[0] > -400)
        assert len(actual) > 0
        actual_width = (actual[:, 1].max() - actual[:, 1].min()) * image.GetSpacing()[0]
        assert actual_width >= upper[0] - lower[0] - 3 * image.GetSpacing()[0]
        assert abs((center - _vector(item["center"])) @ normal) < 1e-7
        spacings.append(image.GetSpacing())
    np.testing.assert_allclose(spacings, [spacings[0]] * len(spacings))


def _clipped_body_voxel_bounds(source_body, image, planned_slice, thickness):
    """独立裁切实体体素的盒角和棱边，求厚层内真实体素交集的投影边界。"""

    zyx = np.argwhere(np.broadcast_to(source_body, tuple(reversed(image.GetSize()))))
    voxel_axes = np.array(image.GetDirection()).reshape(3, 3) * image.GetSpacing()
    points = zyx[:, ::-1] @ voxel_axes.T + image.GetOrigin()
    row, col, normal = (_vector(planned_slice[key]) for key in ("row_dir", "col_dir", "normal"))
    center_depth = _vector(planned_slice["center"]) @ normal
    lower_depth, upper_depth = center_depth - thickness / 2, center_depth + thickness / 2
    half_support = np.abs(voxel_axes.T @ normal).sum() / 2
    depths = points @ normal
    points = points[(depths + half_support > lower_depth) & (depths - half_support < upper_depth)]
    signs = np.array(list(product((-0.5, 0.5), repeat=3)))
    corners = points[:, None, :] + (signs @ voxel_axes.T)[None, :, :]
    corner_depths = corners @ normal
    inside = (corner_depths >= lower_depth - 1e-10) & (corner_depths <= upper_depth + 1e-10)
    vertices = [corners[inside]]
    # 三维盒的每条棱恰好只改变一个角点符号；求它与两层界面的交点。
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(signs[first] != signs[second]) != 1:
                continue
            denominator = corner_depths[:, second] - corner_depths[:, first]
            if abs(float(denominator[0])) < 1e-12:
                continue
            edge = corners[:, second] - corners[:, first]
            for plane_depth in (lower_depth, upper_depth):
                fraction = (plane_depth - corner_depths[:, first]) / denominator
                crossing = (fraction >= -1e-10) & (fraction <= 1 + 1e-10)
                vertices.append(corners[crossing, first] + fraction[crossing, None] * edge[crossing])
    projected = np.concatenate(vertices) @ np.column_stack((row, col))
    return projected.min(axis=0), projected.max(axis=0)


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("fill_percent,thickness", ((80, 4.0), (100, 6.0)))
@pytest.mark.parametrize("coarse_segmentation", (False, True))
def test_relative_oblique_thick_slabs_cover_real_body_voxel_intersections(
    part, fill_percent, thickness, coarse_segmentation,
):
    raw, seg, plan, coords, body, physical, _ = _fixture(part, fill_percent)
    if coarse_segmentation:
        # 身体支持来自细间距 HU，目标来自较粗分割；不能共用分割的半体素支撑。
        seg = sitk.Resample(
            seg, [int((value - 1) // 2 + 1) for value in seg.GetSize()], sitk.Transform(),
            sitk.sitkNearestNeighbor, seg.GetOrigin(), tuple(2 * np.array(seg.GetSpacing())),
            seg.GetDirection(), 0, sitk.sitkUInt8,
        )
    # 源网格保持 identity，只有计划发生三维旋转，不能退化成共同旋转坐标系。
    yaw, pitch, roll = np.deg2rad([23.0, 19.0, 13.0])
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0],
    ]) @ np.array([
        [np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)],
    ]) @ np.array([
        [1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll), np.cos(roll)],
    ])
    assert np.all(np.abs(rotation[:, 2]) > 0.01)
    plan["slice_thickness"] = thickness
    for item, depth in zip(plan["slices"], (3, 6, 7)):
        item.update(center=_point(physical((68.0, 48.0, depth))),
                    row_dir=_point(rotation[:, 0]), col_dir=_point(rotation[:, 1]),
                    normal=_point(rotation[:, 2]))
    original = copy.deepcopy(plan)
    framer = TargetFramer(seg, coords, plan, body_support=build_body_framing_support(raw))
    spacings = []
    centers = []
    for item in plan["slices"]:
        lower, upper = _clipped_body_voxel_bounds(body, raw, item, thickness)
        framed_plan, framed = framer.plane_geometry(plan, item)
        image = PlaneStackResampler().sample_slice(raw, framed_plan, framed)
        center = np.array(image.TransformContinuousIndexToPhysicalPoint(
            ((image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2, 0),
        ))
        fov_half = np.array(image.GetSize()[:2]) * image.GetSpacing()[:2] / 2
        projected_center = center @ rotation[:, :2]
        assert np.all(projected_center - fov_half <= lower + 1e-6)
        assert np.all(projected_center + fov_half >= upper - 1e-6)
        assert framed["framing"]["center_source"] == "slice_body_contour"
        for field in ("row_dir", "col_dir", "normal"):
            assert framed[field] == item[field]
        assert abs((center - _vector(item["center"])) @ rotation[:, 2]) < 1e-7
        assert framed_plan["slice_thickness"] == thickness
        spacing = np.array(image.GetSpacing()[:2])
        np.testing.assert_allclose(spacing[0] / spacing[1], 0.9 / 1.2)
        spacings.append(spacing)
        centers.append(center)
    np.testing.assert_allclose(spacings, [spacings[0]] * len(spacings))
    actual_depth_steps = np.diff(np.array(centers) @ rotation[:, 2])
    original_depth_steps = np.diff([_vector(item["center"]) @ rotation[:, 2] for item in plan["slices"]])
    np.testing.assert_allclose(actual_depth_steps, original_depth_steps, atol=1e-7)
    assert plan == original
