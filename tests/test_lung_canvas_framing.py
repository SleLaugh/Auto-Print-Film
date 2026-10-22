"""肺横状位去除画布空气后，胶片宽格仍按共同倍率显示完整躯干。"""

import copy
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.reconstruction.body_framing import build_body_framing_support
from src.reconstruction.framing import TargetFramer, frame_curved_slice
from src.reconstruction.plane_stack import PlaneStackResampler
from src.stages.stage_reconstruction import ReconstructionStage


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
    ("肺部", "横状位", 100, False), ("髋骨", "横状位", 100, True),
    ("肺部", "冠状位", 100, True), ("肺部", "矢状位", 100, True),
))
def test_air_canvas_compaction_is_limited_to_supported_positive_lung_axial_mpr(part, plane, percent, with_body):
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


@pytest.mark.parametrize("fill_percent", (80, 100))
def test_reconstruction_dicom_windows_share_compact_matrix_scale_and_full_terminal_contour(tmp_path: Path, fill_percent):
    raw, seg, plan, coords, bodies, depths, physical, direction = _fixture(fill_percent, oblique=True)
    raw_path, seg_path = tmp_path / "raw.nii.gz", tmp_path / "seg.nii.gz"
    sitk.WriteImage(raw, str(raw_path))
    sitk.WriteImage(seg, str(seg_path))
    request = Request(
        str(tmp_path / "source"), str(tmp_path / "out"), slice_part=SlicePartEnum.肺部,
        plane_configs=[PlaneConfig(
            PlaneTypeEnum.横状位, 0.5, 2.5, ["纵隔窗", "肺窗"], fill_percent=fill_percent,
        )],
    )
    context = ProcessingContext(request)
    context.set_result("localization", {
        "raw_for_reconstruction_path": str(raw_path), "seg_for_reconstruction_path": str(seg_path),
        "coords": coords,
    })
    context.set_result("planning", {"plans": [plan]})
    context.set_result("preprocessing", {"source_dicom_metadata": {"0020|000d": "1.2.3.4.5"}})
    ReconstructionStage({"parallel": False}).execute(context)
    output = context.get_result("reconstruction")["outputs"][0]
    bounds, maximum = _truth(bodies, depths)
    expected_spacing = max(maximum / 200) / (fill_percent / 100)
    geometry = []
    for window in ("纵隔窗", "肺窗"):
        directory = Path(output["window_outputs"][window])
        for item, z, (lower, upper) in zip(plan["slices"], depths, bounds):
            ds = pydicom.dcmread(directory / f"{item['index']}.dcm")
            row, col = np.array(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
            spacing_yx = np.array(ds.PixelSpacing, dtype=float)
            center = np.array(ds.ImagePositionPatient, dtype=float)
            center += row * (ds.Columns - 1) * spacing_yx[1] / 2
            center += col * (ds.Rows - 1) * spacing_yx[0] / 2
            np.testing.assert_allclose(center, physical((*((lower + upper) / 2), z)), atol=1e-5)
            np.testing.assert_allclose(np.column_stack((row, col)), direction[:, :2], atol=1e-7)
            np.testing.assert_allclose(spacing_yx, expected_spacing, rtol=1e-6)
            assert ds.Rows < 110
            assert ds.Columns == 200
            assert float(ds.SliceThickness) == plan["slice_thickness"]
            projected_center = (center - raw.GetOrigin()) @ direction[:, :2]
            half_fov = np.array([ds.Columns, ds.Rows]) * spacing_yx[::-1] / 2
            assert np.all(projected_center - half_fov <= lower + 1e-5)
            assert np.all(projected_center + half_fov >= upper - 1e-5)
            # 两种窗都检查实际四向体表，防止只有几何声明完整。
            visible_yx = np.argwhere(ds.pixel_array > 100)
            span = (visible_yx.max(axis=0) - visible_yx.min(axis=0))[::-1] * spacing_yx[::-1]
            assert np.all(span >= upper - lower - 2 - 3 * spacing_yx[::-1])
            geometry.append((ds.Columns, ds.Rows, *spacing_yx, *ds.ImagePositionPatient, *ds.ImageOrientationPatient))
    np.testing.assert_allclose(geometry[:len(depths)], geometry[len(depths):], atol=1e-8)
