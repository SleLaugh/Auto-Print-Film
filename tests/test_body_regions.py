"""身体范围必须在区域规划前完整、保守地求解，不能由器官包围盒代替。"""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.anatomy.body_regions import apply_body_transverse_regions
from src.models import AnatomicalBounds, AnatomicalFrame, AxisBounds, Point3D
from src.reconstruction.body_framing import build_body_framing_support


def _image(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(np.asarray(direction).ravel())
    return image


def _region(si, axes=None):
    axes = np.eye(3) if axes is None else axes
    frame = AnatomicalFrame(
        origin=Point3D(10, 11, 12),
        left_right_axis=Point3D(*axes[:, 0]),
        anterior_posterior_axis=Point3D(*axes[:, 1]),
        superior_inferior_axis=Point3D(*axes[:, 2]),
        bounds=AnatomicalBounds(AxisBounds(45, 55), AxisBounds(45, 55), AxisBounds(*si)),
        method="hip", confidence=0.9, applied=True, warnings=["原回正诊断"],
    )
    return {"global_frame": frame.to_dict(), "local_frames": {}, "constrained": False,
            "target_labels": ["liver", "hip_left", "hip_right"]}


def _apply(image, region, target=None):
    return apply_body_transverse_regions(
        image, target, {"abdomen": region}, {"abdomen": {"transverse_source": "body"}},
    )["abdomen"]


def _bounds(region, frame="global_frame"):
    bounds = region[frame]["bounds"]
    return np.asarray([(bounds[key]["minimum"], bounds[key]["maximum"])
                       for key in ("left_right", "anterior_posterior", "superior_inferior")])


def test_body_width_is_measured_only_inside_si_region_and_preserves_inputs():
    raw = np.full((10, 100, 120), -1024, dtype=np.int16)
    raw[:, 20:80, 30:90] = 30
    raw[0, 5:95, 5:115] = 30
    raw[9, 5:95, 5:115] = 30
    region = _region((2.5, 6.5))
    original = deepcopy(region)
    actual = _apply(_image(raw), region)
    np.testing.assert_allclose(_bounds(actual), ((29.5, 89.5), (19.5, 79.5), (2.5, 6.5)))
    assert actual["constrained"] is True
    assert actual["global_frame"]["origin"] == region["global_frame"]["origin"]
    assert actual["global_frame"]["warnings"] == ["原回正诊断"]
    assert actual["transverse_support"]["frames"]["global_frame"]["status"] == "body"
    assert region == original


@pytest.mark.parametrize("connected", [False, True])
def test_only_reliably_separate_arms_are_removed_and_connected_arms_are_kept(connected):
    raw = np.full((3, 160, 240), -1024, dtype=np.int16)
    raw[:, 30:130, 70:170] = 30
    raw[:, 40:100, 10:50] = 30
    raw[:, 40:100, 190:230] = 30
    if connected:
        raw[:, 60:80, 49:71] = 30
        raw[:, 60:80, 169:191] = 30
    target = np.zeros_like(raw, dtype=np.uint8)
    target[:, 55:105, 100:140] = 1
    actual = _apply(_image(raw), _region((-0.5, 2.5)), _image(target))
    expected_lr = (9.5, 229.5) if connected else (69.5, 169.5)
    np.testing.assert_allclose(_bounds(actual)[:2], (expected_lr, (29.5, 129.5)))
    assert actual["transverse_support"]["support_region"] == ("body" if connected else "trunk")


def test_pelvis_peripheral_thick_bones_are_never_used_to_split_connected_body():
    yy, xx = np.ogrid[:160, :240]
    raw = np.full((3, 160, 240), -1024, dtype=np.int16)
    raw[:, 20:140, 30:210] = 30
    bones = ((xx - 53) ** 2 + (yy - 80) ** 2 < 18 ** 2) | ((xx - 187) ** 2 + (yy - 80) ** 2 < 18 ** 2)
    raw[:, bones] = 900
    target = np.zeros_like(raw, dtype=np.uint8)
    target[:, 65:95, 100:140] = 1
    actual = _apply(_image(raw), _region((-0.5, 2.5)), _image(target))
    np.testing.assert_allclose(_bounds(actual)[:2], ((29.5, 209.5), (19.5, 139.5)))


def test_independent_curved_table_with_four_to_eight_mm_radius_does_not_expand_abdomen():
    raw = np.full((3, 200, 240), -1024, dtype=np.int16)
    raw[:, 20:120, 60:180] = 30
    # 最大内切半径大于明确薄床的4mm，但仍小于身体主体的8mm门槛。
    # preserve_small_components 会把此类已拒绝的外围组件重新纳入。
    raw[:, 155:165, 10:230] = 500
    raw[:, 130:165, 10:20] = 500
    raw[:, 130:165, 220:230] = 500
    target = np.zeros_like(raw, dtype=np.uint8)
    target[:, 50:90, 100:140] = 1
    image = _image(raw)
    permissive = build_body_framing_support(image, preserve_small_components=True)
    assert permissive.points_xyz[:, 1].max() == 164
    actual = _apply(image, _region((-0.5, 2.5)), _image(target))
    np.testing.assert_allclose(_bounds(actual)[:2], ((59.5, 179.5), (19.5, 119.5)))
    assert actual["transverse_support"]["frames"]["global_frame"]["fallback_source_slice_count"] == 0


@pytest.mark.parametrize("flips", [(False, False, False), (True, False, True), (False, True, False)])
def test_physical_direction_spacing_origin_and_storage_flips_are_respected(flips):
    yaw, tilt = np.deg2rad((23, 17))
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    axes = rz @ rx
    spacing, origin = (1.3, 1.7, 2.4), (17.0, -43.0, 101.0)
    raw = np.full((5, 100, 120), -1024, dtype=np.int16)
    raw[:, 20:80, 30:90] = 30
    image = sitk.Flip(_image(raw, spacing, origin, axes), flips)
    projected_origin = np.asarray(origin) @ axes
    si = projected_origin[2] + np.asarray((1.5, 3.5)) * spacing[2]
    actual = _apply(image, _region(si, axes))
    expected_lr = projected_origin[0] + np.asarray((29.5, 89.5)) * spacing[0]
    expected_ap = projected_origin[1] + np.asarray((19.5, 79.5)) * spacing[1]
    np.testing.assert_allclose(_bounds(actual), (expected_lr, expected_ap, si), atol=1e-9)


def test_oblique_frame_uses_physical_si_not_stored_z():
    angle = np.deg2rad(30)
    axes = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    raw = np.full((6, 100, 120), -1024, dtype=np.int16)
    raw[:, 20:80, 30:90] = 30
    # 物理 SI 区间仅穿过身体前侧；不能把完整 y=20..79 都算入。
    si = (-13.0, -8.0)
    actual = _apply(_image(raw), _region(si, axes))
    points = np.argwhere(raw > -500)[:, ::-1] @ axes
    half = np.abs(axes).sum(axis=0) / 2
    selected = (points[:, 2] + half[2] > si[0] + 1e-7) & (points[:, 2] - half[2] < si[1] - 1e-7)
    expected = np.column_stack((points[selected, :2].min(axis=0) - half[:2],
                                points[selected, :2].max(axis=0) + half[:2]))
    np.testing.assert_allclose(_bounds(actual)[:2], expected, atol=1e-9)
    assert _bounds(actual)[1, 1] < 45


@pytest.mark.parametrize("bad_layers,expected_status,count", [
    ([2], "body_with_source_fallback", 1),
    ([0, 1, 2, 3, 4], "source_fallback", 3),
])
def test_unknown_body_layers_use_source_fov_not_small_organ_bounds(bad_layers, expected_status, count):
    raw = np.full((5, 100, 120), -1024, dtype=np.int16)
    raw[:, 20:80, 30:90] = 30
    raw[bad_layers] = -1024
    actual = _apply(_image(raw), _region((0.5, 3.5)))
    np.testing.assert_allclose(_bounds(actual), ((-0.5, 119.5), (-0.5, 99.5), (0.5, 3.5)))
    diagnostic = actual["transverse_support"]["frames"]["global_frame"]
    assert diagnostic["status"] == expected_status
    assert diagnostic["fallback_source_slice_count"] == count
    assert diagnostic["warnings"]
    assert actual["global_frame"]["warnings"] == ["原回正诊断", *diagnostic["warnings"]]


def test_unknown_layers_outside_si_do_not_force_source_fov_fallback():
    raw = np.full((5, 100, 120), -1024, dtype=np.int16)
    raw[1:4, 20:80, 30:90] = 30
    actual = _apply(_image(raw), _region((0.5, 3.5)))
    np.testing.assert_allclose(_bounds(actual)[:2], ((29.5, 89.5), (19.5, 79.5)))
    assert actual["transverse_support"]["frames"]["global_frame"]["fallback_source_slice_count"] == 0


def test_fallback_source_box_is_intersected_with_oblique_si_interval():
    angle = np.deg2rad(30)
    axes = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    raw = np.full((4, 100, 120), -1024, dtype=np.int16)
    actual = _apply(_image(raw), _region((-1.0, 1.0), axes))
    # y 范围仅约 0..8 mm，远小于原FOV的100mm，证明回退盒先做SI截交。
    # AP 下界在 y=z=-0.5；上界在 z=3.5 与 SI=-1 的交线上。
    expected_ap = (-0.5 * (np.cos(angle) + np.sin(angle)),
                   (3.5 + np.cos(angle)) / np.sin(angle))
    np.testing.assert_allclose(_bounds(actual)[1], expected_ap, atol=1e-9)
    np.testing.assert_allclose(_bounds(actual)[0], (-0.5, 119.5))


def test_body_extraction_is_shared_across_regions_and_local_frames():
    raw = np.full((5, 100, 120), -1024, dtype=np.int16)
    raw[:, 20:80, 30:90] = 30
    first = _region((0.5, 2.5))
    first["local_frames"]["axial"] = deepcopy(first["global_frame"])
    regions = {"first": first, "second": _region((2.5, 4.5)), "labels": _region((0.5, 4.5))}
    with patch("src.anatomy.body_regions.build_body_framing_support", wraps=build_body_framing_support) as build:
        actual = apply_body_transverse_regions(_image(raw), None, regions, {
            "first": {"transverse_source": "body"}, "second": {"transverse_source": "body"}, "labels": {},
        })
    assert build.call_count == 1
    assert actual["labels"] is regions["labels"]
    assert actual["first"]["global_frame"] == actual["first"]["local_frames"]["axial"]


def test_label_only_regions_do_not_read_image_or_extract_body():
    region = _region((0.5, 2.5))
    with patch("src.anatomy.body_regions.build_body_framing_support", side_effect=AssertionError):
        assert apply_body_transverse_regions(None, None, {"selected": region}, {"selected": {}}) == {"selected": region}


def test_mask_geometry_mismatch_and_no_source_intersection_are_explicit_errors():
    image = _image(np.full((5, 100, 120), -1024, dtype=np.int16))
    wrong = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    wrong.SetOrigin((10, 0, 0))
    with pytest.raises(ValueError, match="空间不一致"):
        _apply(image, _region((0, 4)), wrong)
    with pytest.raises(ValueError, match="身体区域 abdomen.*没有可用交集"):
        _apply(image, _region((20, 30)))
