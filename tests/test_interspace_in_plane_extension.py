"""椎间隙六面配置到成片的回归：面内加宽保持原切面及采样格。"""

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import PlaneConfig, Point3D, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.straight import CutSegment, StraightPlaneFrame, build_straight_slice_plan
from src.reconstruction.framing import TargetFramer, prepare_interspace_framing
from src.reconstruction.plane_stack import PlaneStackResampler


_ZERO_EXTENSION = dict.fromkeys(
    ("top_mm", "bottom_mm", "front_mm", "back_mm", "left_mm", "right_mm"), 0.0,
)


def _vector(point):
    return np.asarray([point[axis] for axis in "xyz"], dtype=float)


def _fixture(*, flipped=False, oblique=False):
    spacing = np.asarray([0.7, 1.1, 1.3])
    size = np.asarray([64, 48, 33])
    direction = np.diag([-1.0, -1.0, -1.0]) if flipped else np.eye(3)
    origin = (size - 1) * spacing if flipped else np.zeros(3)
    z, y, x = np.indices(tuple(size[::-1]))
    # 多方向变化的真实源强度，使采样相位变化不能被恒值图掩盖。
    raw = (x * 3 + y * 7 + z * 13 + np.sin(x / 3) * 17).astype(np.float32)
    labels = np.zeros(raw.shape, dtype=np.uint8)
    labels[11:23, 17:32, 20:43] = 1
    source, segmentation = map(sitk.GetImageFromArray, (raw, labels))
    for image in (source, segmentation):
        image.SetSpacing(tuple(spacing))
        image.SetOrigin(tuple(origin))
        image.SetDirection(tuple(direction.ravel()))
    if oblique:
        angle, tilt = np.deg2rad([31.0, 23.0])
        row = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        col = np.asarray([
            -np.sin(angle) * np.cos(tilt), np.cos(angle) * np.cos(tilt), np.sin(tilt),
        ])
    else:
        row, col = np.eye(3)[:2]
    normal = np.cross(row, col)
    frame = StraightPlaneFrame(*(Point3D(*axis) for axis in (row, col, normal)))
    center = np.asarray(source.TransformContinuousIndexToPhysicalPoint((31.5, 24.0, 16.0)))
    segment = CutSegment(
        group_name="L1_L2",
        target_start=Point3D(*(center - normal * 2.25)),
        target_end=Point3D(*(center + normal * 2.25)),
        target_region_bbox={
            "x_min": 20, "x_max": 42, "y_min": 17, "y_max": 31,
            "z_min": 11, "z_max": 22,
        },
        source_labels=("disc",), frame=frame, recenter_target_in_plane=False,
    )
    localization = {"image": {
        "spacing_xyz": spacing.tolist(), "size_xyz": size.tolist(),
        "origin_xyz": origin.tolist(), "direction": direction.ravel().tolist(),
    }}
    return source, segmentation, {"disc": {"label_ids": [1]}}, localization, segment, frame


def _plan(fixture, extension, *, fill_percent=80, group_name=None):
    _, _, _, localization, segment, frame = fixture
    localization = deepcopy(localization)
    localization["slice_rules"] = {"横状位": {"椎间隙": {
        "face_extension_mm": {**_ZERO_EXTENSION, **extension},
    }}}
    if group_name is not None:
        segment = replace(segment, group_name=group_name)
    config = PlaneConfig(
        PlaneTypeEnum.横状位, 1.0, 1.5, ["骨窗"], fill_percent=fill_percent,
        slice_type=SliceType.椎间隙,
    )
    request = Request(
        dicom_dir="/dicom", output_dir="/output", slice_part=SlicePartEnum.腰椎,
        plane_configs=[config],
    )
    return build_straight_slice_plan(
        request, SliceType.椎间隙.value, config, localization, segment, frame,
    ).to_dict()


def _assert_same_layer_group(base, expanded):
    original = deepcopy(expanded)
    original.pop("framing_in_plane_extension_mm", None)
    assert original == base


def _assert_preserved_sampling(fixture, base, expanded, lower_mm, upper_mm, *, shared=None):
    source, segmentation, coords, *_ = fixture
    base_shared, expanded_shared = shared or (None, None)
    baseline = TargetFramer(segmentation, coords, base, shared_geometry=base_shared)
    padded = TargetFramer(segmentation, coords, expanded, shared_geometry=expanded_shared)
    resampler = PlaneStackResampler()
    for old_slice, new_slice in zip(base["slices"], expanded["slices"]):
        base_plan, base_slice = baseline.plane_geometry(base, old_slice)
        padded_plan, padded_slice = padded.plane_geometry(expanded, new_slice)
        original = resampler.sample_slice(source, base_plan, base_slice)
        output = resampler.sample_slice(source, padded_plan, padded_slice)
        spacing = np.asarray(original.GetSpacing()[:2])
        lower = np.ceil(np.asarray(lower_mm) / spacing).astype(int)
        upper = np.ceil(np.asarray(upper_mm) / spacing).astype(int)
        np.testing.assert_array_equal(
            output.GetSize()[:2], np.asarray(original.GetSize()[:2]) + lower + upper,
        )
        assert output.GetSpacing() == original.GetSpacing()
        assert output.GetDirection() == original.GetDirection()
        assert padded_slice["framing"]["scale"] == base_slice["framing"]["scale"]
        # 单侧扩展必须向该侧移动画布中心，不能从另一侧裁走原图。
        row, col, normal = (_vector(old_slice[key]) for key in ("row_dir", "col_dir", "normal"))
        delta = _vector(padded_slice["center"]) - _vector(base_slice["center"])
        np.testing.assert_allclose(delta, (row * (upper[0] - lower[0]) * spacing[0]
                                         + col * (upper[1] - lower[1]) * spacing[1]) / 2,
                                   atol=1e-12)
        assert float(delta @ normal) == pytest.approx(0.0, abs=1e-12)
        expected_origin = (np.asarray(original.GetOrigin()) - row * lower[0] * spacing[0]
                           - col * lower[1] * spacing[1])
        np.testing.assert_allclose(output.GetOrigin(), expected_origin, atol=1e-12)
        array = sitk.GetArrayFromImage(output)[0]
        width, height = original.GetSize()[:2]
        retained = array[lower[1]:lower[1] + height, lower[0]:lower[0] + width]
        # 原点断言已严格检查同一采样格；允许 float32 厚层累计的舍入误差。
        np.testing.assert_allclose(retained, sitk.GetArrayFromImage(original)[0], rtol=0, atol=3e-4)
        assert padded_slice["framing"]["output_size_xy"] == list(output.GetSize()[:2])


@pytest.mark.parametrize(
    "face, fill_percent, expected",
    [
        ("left_mm", 0, {"row_min": 0.0, "row_max": 3.17, "col_min": 0.0, "col_max": 0.0}),
        ("right_mm", 80, {"row_min": 3.17, "row_max": 0.0, "col_min": 0.0, "col_max": 0.0}),
        ("front_mm", 0, {"row_min": 0.0, "row_max": 0.0, "col_min": 3.17, "col_max": 0.0}),
        ("back_mm", 80, {"row_min": 0.0, "row_max": 0.0, "col_min": 0.0, "col_max": 3.17}),
    ],
)
def test_single_patient_face_extension_reaches_sampled_canvas(face, fill_percent, expected):
    fixture = _fixture()
    base = _plan(fixture, {}, fill_percent=fill_percent)
    expanded = _plan(fixture, {face: 3.17}, fill_percent=fill_percent)

    assert expanded["framing_in_plane_extension_mm"] == expected
    _assert_same_layer_group(base, expanded)
    _assert_preserved_sampling(
        fixture, base, expanded,
        [expected["row_min"], expected["col_min"]],
        [expected["row_max"], expected["col_max"]],
    )


def test_flipped_source_and_oblique_cut_project_all_faces_without_extending_layer_group():
    fixture = _fixture(flipped=True, oblique=True)
    extension = {"left_mm": 3.17, "right_mm": 1.23, "front_mm": 2.71, "back_mm": 4.39}
    base = _plan(fixture, {})
    expanded = _plan(fixture, extension)
    # 三条源轴翻转后，患者左仍为物理 +X、前仍为物理 -Y。
    offsets = np.asarray(list(product((-1.23, 3.17), (-2.71, 4.39), (0.0,))))
    frame = fixture[-1]
    axes = np.column_stack([[getattr(axis, key) for key in "xyz"]
                            for axis in (frame.row_dir, frame.col_dir)])
    projected = offsets @ axes
    lower, upper = -projected.min(axis=0), projected.max(axis=0)
    np.testing.assert_allclose(
        [expanded["framing_in_plane_extension_mm"][key]
         for key in ("row_min", "row_max", "col_min", "col_max")],
        [lower[0], upper[0], lower[1], upper[1]],
    )
    _assert_same_layer_group(base, expanded)
    _assert_preserved_sampling(fixture, base, expanded, lower, upper)


def test_shared_geometry_uses_per_side_maximum_and_preserves_each_original_canvas():
    fixture = _fixture()
    _, segmentation, coords, *_ = fixture
    extensions = [
        {"left_mm": 3.17, "front_mm": 1.23},
        {"right_mm": 4.39, "back_mm": 2.71},
        {"left_mm": 1.11, "front_mm": 0.51},
    ]
    base = [_plan(fixture, {}, group_name=f"gap_{index}") for index in range(3)]
    expanded = [_plan(fixture, value, group_name=f"gap_{index}")
                for index, value in enumerate(extensions)]
    untouched = deepcopy(expanded)
    base_geometry = prepare_interspace_framing(segmentation, coords, base)
    geometry = prepare_interspace_framing(segmentation, coords, expanded)
    lower, upper = np.asarray([4.39, 1.23]), np.asarray([3.17, 2.71])
    expected_size = (np.asarray(base_geometry.size_xy)
                     + np.ceil(lower / base_geometry.spacing_xy).astype(int)
                     + np.ceil(upper / base_geometry.spacing_xy).astype(int))
    assert geometry.size_xy == tuple(expected_size)
    assert geometry.spacing_xy == base_geometry.spacing_xy
    assert geometry.scale == base_geometry.scale
    assert geometry.target_size_mm == base_geometry.target_size_mm
    for original, padded in zip(base, expanded):
        _assert_same_layer_group(original, padded)
        _assert_preserved_sampling(
            fixture, original, padded, lower, upper, shared=(base_geometry, geometry),
        )
    assert expanded == untouched


def test_top_bottom_keep_their_existing_layer_extension_independent_of_in_plane_padding():
    fixture = _fixture()
    unextended = _plan(fixture, {})
    base = _plan(fixture, {"top_mm": 2.6, "bottom_mm": 1.3})
    expanded = _plan(fixture, {
        "top_mm": 2.6, "bottom_mm": 1.3,
        "left_mm": 3.17, "right_mm": 1.23, "front_mm": 2.71, "back_mm": 4.39,
    })

    assert base["target_start"] != unextended["target_start"]
    assert base["target_end"] != unextended["target_end"]
    assert base["slice_image_count"] > unextended["slice_image_count"]
    _assert_same_layer_group(base, expanded)
    _assert_preserved_sampling(fixture, base, expanded, [1.23, 2.71], [3.17, 4.39])
