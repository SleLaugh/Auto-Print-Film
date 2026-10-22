"""显式身体支持 API 的肺横状位画布压紧；生产目标取景另行验收。"""

import copy

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.body_framing import build_body_framing_support
from src.reconstruction.framing import TargetFramer, frame_curved_slice
from src.reconstruction.plane_stack import PlaneStackResampler


def _point(values):
    return dict(zip("xyz", map(float, values)))


def _vector(point):
    return np.array([point[key] for key in "xyz"], dtype=float)


def _fixture(fill_percent=100, *, oblique=False):
    """真值来自逐层实体躯干，不使用取景元数据反算预期范围。"""

    spacing = np.array([1.0, 1.0, 2.5])
    origin = np.array([-70.0, -100.0, 10.0])
    yaw, pitch = np.deg2rad([12.0, 8.0] if oblique else [0.0, 0.0])
    direction = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0],
    ]) @ np.array([
        [1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)],
    ])

    def physical(index):
        return origin + direction @ (np.asarray(index) * spacing)

    y, x = np.ogrid[:200, :200]
    widths = (50, 62, 74, 68, 60, 55, 50, 45, 40)
    # 最大高度令未经相位保护的裁后 Rows 为奇数，原画布为偶数；
    # 因而本例能实际检测半像素采样移动，而不只是验证最终矩阵大小。
    heights = (27, 30, 34, 34, 32, 30, 27, 26, 25)
    bodies = np.array([
        ((x - (90 + 2 * z)) / rx) ** 2 + ((y - (75 + 3 * z)) / ry) ** 2 <= 1
        for z, (rx, ry) in enumerate(zip(widths, heights))
    ])
    labels = np.zeros(bodies.shape, dtype=np.uint8)
    for z in (1, 2, 3):
        cx, cy = 90 + 2 * z, 75 + 3 * z
        labels[z, ((x - cx + 25) / 16) ** 2 + ((y - cy) / 19) ** 2 <= 1] = 1
        labels[z, ((x - cx - 25) / 16) ** 2 + ((y - cy) / 19) ** 2 <= 1] = 2
    labels[4, 97:99, 126:129] = 2  # 肺底只剩偏心小目标；下一层没有肺。
    raw_array = np.where(bodies, 50.0, -1024.0).astype(np.float32)
    raw_array[labels > 0] = -800.0
    raw_array[:, 190:194, 5:195] = 700.0  # 分离床板不能扩大画布。
    raw = sitk.GetImageFromArray(raw_array)
    raw.SetSpacing(tuple(spacing))
    raw.SetDirection(tuple(direction.ravel()))
    raw.SetOrigin(tuple(origin))
    seg = sitk.GetImageFromArray(labels)
    seg.CopyInformation(raw)
    depths = (1, 2, 3, 4, 5)
    plan = {
        "plan_id": "lung_canvas", "slice_part": "肺部", "slice_type": "MPR",
        "plane": "横状位", "geometry_type": "plane_stack", "group_name": "whole_range",
        "source_labels": ["left_lung", "right_lung"], "fill_percent": fill_percent,
        "slice_thickness": 0.5, "slice_spacing": 2.5,
        "window_presets": ["纵隔窗", "肺窗"],
        "extent_mm": {"width": 199.0, "height": 199.0},
        "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
        "slices": [{
            "index": index, "center": _point(physical((100, 100, z))),
            "row_dir": _point(direction[:, 0]), "col_dir": _point(direction[:, 1]),
            "normal": _point(direction[:, 2]),
        } for index, z in enumerate(depths, start=1)],
    }
    plan["range_start"] = plan["slices"][0]["center"]
    plan["range_end"] = plan["slices"][-1]["center"]
    plan["path_points"] = [plan["range_start"], plan["range_end"]]
    coords = {"left_lung": {"label_ids": [1]}, "right_lung": {"label_ids": [2]}}
    return raw, seg, plan, coords, bodies, depths, physical, direction


def _truth(bodies, depths):
    bounds = []
    for z in depths:
        indices = np.argwhere(bodies[z])[:, ::-1]
        bounds.append((indices.min(axis=0) - 0.5, indices.max(axis=0) + 0.5))
    maximum = np.max([upper - lower for lower, upper in bounds], axis=0)
    return bounds, maximum


@pytest.mark.parametrize("fill_percent", (80, 100))
@pytest.mark.parametrize("oblique", (False, True))
def test_lung_canvas_removes_air_without_changing_common_sampling_scale(fill_percent, oblique):
    raw, seg, plan, coords, bodies, depths, physical, direction = _fixture(
        fill_percent, oblique=oblique,
    )
    original = copy.deepcopy(plan)
    bounds, maximum = _truth(bodies, depths)
    expected_spacing = max(maximum / 200) / (fill_percent / 100)
    expected_size = np.minimum(200, np.ceil(maximum / (fill_percent / 100) / expected_spacing - 1e-9)).astype(int)
    expected_size += (200 - expected_size) % 2
    framer = TargetFramer(seg, coords, plan, body_support=build_body_framing_support(raw))
    images = []
    for item, z, (lower, upper) in zip(plan["slices"], depths, bounds):
        framed_plan, framed = framer.plane_geometry(plan, item)
        image = PlaneStackResampler().sample_slice(raw, framed_plan, framed)
        images.append(image)
        size, spacing = np.array(image.GetSize()[:2]), np.array(image.GetSpacing()[:2])
        np.testing.assert_array_equal(size, expected_size)
        np.testing.assert_array_equal((200 - size) % 2, (0, 0))
        np.testing.assert_allclose(spacing, expected_spacing, atol=1e-12)
        np.testing.assert_allclose(framed["framing"]["scale"], 1 / expected_spacing)
        center = np.array(image.TransformContinuousIndexToPhysicalPoint(
            ((size[0] - 1) / 2, (size[1] - 1) / 2, 0),
        ))
        expected_center = physical((*((lower + upper) / 2), z))
        np.testing.assert_allclose(center, expected_center, atol=1e-7)
        projected_center = (center - raw.GetOrigin()) @ direction[:, :2]
        assert np.all(projected_center - size * spacing / 2 <= lower + 1e-7)
        assert np.all(projected_center + size * spacing / 2 >= upper - 1e-7)
        for key in ("row_dir", "col_dir", "normal"):
            assert framed[key] == item[key]
        assert abs((center - _vector(item["center"])) @ direction[:, 2]) < 1e-7
        assert framed_plan["slice_thickness"] == plan["slice_thickness"]
        assert framed_plan["slice_spacing"] == plan["slice_spacing"]

        # 画布只应从同倍率旧图两侧各裁整数像素，不能引入半像素重采样。
        old_canvas_plan = copy.deepcopy(framed_plan)
        old_canvas_plan["extent_mm"] = dict(zip(("width", "height"), 199 * spacing))
        old_canvas = PlaneStackResampler().sample_slice(raw, old_canvas_plan, framed)
        left, top = (200 - size) // 2
        old_crop = sitk.GetArrayViewFromImage(old_canvas)[
            :, top:top + size[1], left:left + size[0]
        ]
        np.testing.assert_allclose(sitk.GetArrayViewFromImage(image), old_crop, rtol=0, atol=1e-4)

        # 椭圆尖端只有单个源体素，窗化/插值可使其不达阈值；分别计入
        # 两端源像素及输出像素的量化差，完整物理覆盖已在上方独立严格验证。
        visible_yx = np.argwhere(sitk.GetArrayViewFromImage(image)[0] > -400)
        visible_span = (visible_yx.max(axis=0) - visible_yx.min(axis=0))[::-1] * spacing
        assert np.all(visible_span >= upper - lower - 2 - 3 * spacing)
    assert plan == original
    assert expected_size[1] < 110 < raw.GetSize()[1]
    np.testing.assert_allclose(expected_size[0] / expected_size[1], maximum[0] / maximum[1], atol=0.05)
    extra_air = expected_size - maximum / (fill_percent / 100) / expected_spacing
    assert np.all(extra_air >= -1e-9)
    assert np.all(extra_air < 2 + 1e-9)

    # 模拟实际胶片宽格按完整画布 fit，不能仅凭 DICOM 内部 FOV 宣称达到 100%。
    screen_scale = min(136 / expected_size[0], 104 / expected_size[1])
    widest_display_fraction = maximum[0] / expected_spacing * screen_scale / 136
    np.testing.assert_allclose(widest_display_fraction, fill_percent / 100, atol=1e-10)
    old_screen_scale = min(136 / 200, 104 / 200)
    assert maximum[0] / expected_spacing * old_screen_scale / 136 < fill_percent / 100 * 0.8
    np.testing.assert_allclose([image.GetSpacing() for image in images], [images[0].GetSpacing()] * len(images))


@pytest.mark.parametrize("part,plane,percent,with_body", (
    ("肺部", "横状位", 0, True), ("肺部", "横状位", None, True),
    ("髋骨", "横状位", 100, True),
    ("肺部", "冠状位", 100, True), ("肺部", "矢状位", 100, True),
))
def test_explicit_body_canvas_compaction_keeps_its_original_scope(part, plane, percent, with_body):
    raw, seg, plan, coords, _, _, _, _ = _fixture(percent)
    plan.update(slice_part=part, plane=plane)
    framer = TargetFramer(
        seg, coords, plan, body_support=build_body_framing_support(raw) if with_body else None,
    )
    for item in plan["slices"]:
        sampling_plan, sampling_slice = framer.plane_geometry(plan, item)
        image = PlaneStackResampler().sample_slice(raw, sampling_plan, sampling_slice)
        assert image.GetSize() == raw.GetSize()[:2] + (1,)


@pytest.mark.parametrize("fill_percent", (80, 100))
def test_cpr_keeps_its_unfolded_canvas(fill_percent):
    raw, seg, _, _, _, _, _, _ = _fixture(fill_percent)
    image = raw[:, :, 2:3]
    mask = sitk.Cast(seg[:, :, 2:3] > 0, sitk.sitkUInt8)
    framed, _ = frame_curved_slice(image, mask, fill_percent)
    assert framed.GetSize() == image.GetSize()
