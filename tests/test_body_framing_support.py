"""原始 HU 身体轮廓提取：完整外壳、扫描床排除和物理坐标保持。"""

from itertools import product

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

from src.reconstruction.body_framing import build_body_framing_support


def _image(array, spacing=(1.0, 1.0, 2.0)):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin((17.0, -42.0, 101.0))
    return image


def _case():
    raw = np.full((4, 200, 240), -1024, dtype=np.int16)
    raw[1:3, 30:140, 35:205] = 30
    raw[1:3, 50:120, 65:175] = -900  # 肺空腔不应产生内部边界。
    # 弯曲床面 bbox 很大，但床板自身只有 2mm 厚；不能按 bbox 当成人体。
    raw[:, 175:177, 2:238] = 500
    raw[:, 150:177, 2:4] = 500
    raw[:, 150:177, 236:238] = 500
    return _image(raw)


def _indices(support, image):
    origin = np.asarray(image.GetOrigin())
    return np.rint((support.points_xyz - origin) @ np.linalg.inv(support.voxel_axes.T)).astype(int)


def test_lung_cavities_are_filled_and_large_curved_table_is_excluded():
    image = _case()
    support = build_body_framing_support(image)
    xyz = _indices(support, image)
    np.testing.assert_array_equal(xyz.min(axis=0), (35, 30, 1))
    np.testing.assert_array_equal(xyz.max(axis=0), (204, 139, 2))
    assert np.all((xyz[:, 0] < 65) | (xyz[:, 0] >= 175) | (xyz[:, 1] < 50) | (xyz[:, 1] >= 120))
    # 两层矩形完整的一体素厚外壳；内部实心点不需要展开。
    assert len(xyz) == 2 * (2 * 170 + 2 * 110 - 4)
    assert len(xyz) < 0.04 * 2 * 170 * 110
    assert not support.points_xyz.flags.writeable


@pytest.mark.parametrize("flips", tuple(product((False, True), repeat=3)))
def test_storage_flips_and_oblique_scan_keep_identical_physical_boundary(flips):
    image = _case()
    yaw, tilt = np.deg2rad((23.0, 17.0))
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]])
    direction = rz @ rx
    image.SetDirection(direction.ravel())
    baseline = build_body_framing_support(image)
    flipped = sitk.Flip(image, list(flips))
    actual = build_body_framing_support(flipped)
    expected_points = {tuple(point) for point in np.round(baseline.points_xyz, 7)}
    assert {tuple(point) for point in np.round(actual.points_xyz, 7)} == expected_points
    np.testing.assert_allclose(actual.longitudinal_axis_xyz, direction[:, 2], atol=1e-12)
    np.testing.assert_allclose(actual.voxel_axes, np.asarray(flipped.GetDirection()).reshape(3, 3) * flipped.GetSpacing())


@pytest.mark.parametrize("spacing", [(0.5, 2.0, 1.5), (2.0, 0.5, 3.0)])
def test_component_qualification_and_body_bounds_use_physical_in_plane_spacing(spacing):
    raw = np.full((2, 180, 180), -1024, dtype=np.int16)
    raw[:, 20:120, 25:125] = -100
    raw[:, 50:90, 55:95] = -950
    # 物理宽度 2mm 的扫描床；交换两个面内像素间距也不能被纳入。
    table_thickness = max(1, int(round(2.0 / spacing[1])))
    raw[:, 150:150 + table_thickness, :] = 1000
    image = _image(raw, spacing)
    support = build_body_framing_support(image)
    xyz = _indices(support, image)
    np.testing.assert_array_equal(xyz.min(axis=0), (25, 20, 0))
    np.testing.assert_array_equal(xyz.max(axis=0), (124, 119, 1))


def test_disconnected_thighs_are_both_kept_without_the_scan_table():
    raw = np.full((2, 180, 240), -1024, dtype=np.int16)
    raw[:, 30:120, 25:95] = 40
    raw[:, 30:120, 145:215] = 40
    raw[:, 150:153, :] = 500
    image = _image(raw)
    xyz = _indices(build_body_framing_support(image), image)
    assert np.any(xyz[:, 0] == 25) and np.any(xyz[:, 0] == 214)
    assert not np.any((xyz[:, 0] > 94) & (xyz[:, 0] < 145))
    assert xyz[:, 1].max() == 119


def test_close_table_is_not_joined_to_body_by_morphological_closing():
    raw = np.full((1, 160, 200), -1024, dtype=np.int16)
    raw[:, 20:120, 30:170] = 50
    raw[:, 121:124, :] = 500  # 仅 1mm 空气间隙，整图闭合会把床接到身体。
    image = _image(raw)
    xyz = _indices(build_body_framing_support(image), image)
    np.testing.assert_array_equal(xyz.min(axis=0), (30, 20, 0))
    np.testing.assert_array_equal(xyz.max(axis=0), (169, 119, 0))


def test_body_at_image_edge_keeps_native_visible_contour():
    raw = np.full((1, 160, 180), -1024, dtype=np.int16)
    raw[:, 20:130, :150] = 30
    image = _image(raw)
    xyz = _indices(build_body_framing_support(image), image)
    np.testing.assert_array_equal(xyz.min(axis=0), (0, 20, 0))
    np.testing.assert_array_equal(xyz.max(axis=0), (149, 129, 0))
    assert sum(xyz[:, 0] == 0) == 110


@pytest.mark.parametrize("kind", ["air", "table", "tiny_bone", "thin_ring"])
def test_absent_body_returns_none_instead_of_mistaking_hardware_or_small_targets(kind):
    raw = np.full((2, 160, 200), -1024, dtype=np.int16)
    if kind == "table":
        raw[:, 130:133, :] = 500
        raw[:, 70:133, :3] = 500
        raw[:, 70:133, -3:] = 500
    elif kind == "tiny_bone":
        raw[:, 70:76, 90:96] = 1000
    elif kind == "thin_ring":
        raw[:, 30:130, 50:150] = 800
        raw[:, 32:128, 52:148] = -1024
    assert build_body_framing_support(_image(raw)) is None


def test_non_volume_input_is_rejected():
    with pytest.raises(ValueError, match="三维"):
        build_body_framing_support(sitk.Image(32, 32, sitk.sitkInt16))


@pytest.mark.parametrize("condition", ("thin_shell", "broken_shell", "one_bad_layer"))
def test_incomplete_outer_shell_never_promotes_inner_tissue_to_body_contour(condition):
    y, x = np.ogrid[:280, :300]
    body = ((x - 150.0) / 130.0) ** 2 + ((y - 135.0) / 115.0) ** 2 <= 1
    shell = body & ~ndimage.binary_erosion(body, iterations=6)
    raw = np.full((3, 280, 300), -1024, dtype=np.int16)
    raw[:, body] = -850
    raw[:, shell] = 30
    raw[:, 195:230, 130:170] = 50  # 内部组织够厚，但不代表身体外轮廓。
    if condition == "broken_shell":
        raw[:, 20:40, 150] = -850  # 一条像素宽的开口贯穿前侧外壁。
    elif condition == "one_bad_layer":
        raw[0, body] = 30
        raw[2, body] = 30
    image = _image(raw)
    support = build_body_framing_support(image)
    if support is None:
        return  # 无可靠外轮廓时允许安全回退，不能把内部块当成身体。

    xyz = _indices(support, image)
    body_yx = np.argwhere(body)
    expected_lower = body_yx.min(axis=0)[::-1]
    expected_upper = body_yx.max(axis=0)[::-1]
    for z in np.unique(xyz[:, 2]):
        points = xyz[xyz[:, 2] == z, :2]
        # 按独立椭圆真值检查四向外边界；单像素开口最多影响一个边界像素。
        assert np.all(points.min(axis=0) <= expected_lower + 1)
        assert np.all(points.max(axis=0) >= expected_upper - 1)


def test_unrelated_thin_ring_does_not_discard_reliable_body_support():
    y, x = np.ogrid[:200, :360]
    body = ((x - 100.0) / 70.0) ** 2 + ((y - 100.0) / 60.0) ** 2 <= 1
    raw = np.full((2, 200, 360), -1024, dtype=np.int16)
    raw[:, body] = 30
    # 独立薄环和身体互不包围；过滤环不能连带丢掉可靠的人体。
    raw[:, 50:150, 240:330] = 800
    raw[:, 53:147, 243:327] = -1024
    image = _image(raw)
    support = build_body_framing_support(image)
    assert support is not None
    xyz = _indices(support, image)
    np.testing.assert_array_equal(xyz.min(axis=0), (30, 40, 0))
    np.testing.assert_array_equal(xyz.max(axis=0), (170, 160, 1))
