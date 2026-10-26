"""粗源网格不能使定位线变成遮挡背景的粗白带。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.reconstruction.plane_stack import PlaneStackResampler
from src.stages.stage_reconstruction import ReconstructionStage


def _direction(name):
    if name == "reverse_z":
        return np.diag([1.0, 1.0, -1.0])
    if name == "rotated":
        a, b = np.radians([27.0, 11.0])
        rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)],
                       [0.0, np.sin(b), np.cos(b)]])
        return rz @ rx
    return np.eye(3)


def _case(spacing=(0.75, 0.75, 10.0), *, localizer_plane="矢状位",
          direction="identity", slice_count=None):
    z, y, x = np.indices((37, 48, 32))
    image = sitk.GetImageFromArray((5 * x + y + 3 * z - 900).astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin((11.0, -25.0, 80.0))
    matrix = _direction(direction)
    image.SetDirection(tuple(matrix.ravel()))
    theta = np.radians(19.0)
    if localizer_plane == "矢状位":
        row = matrix[:, 0]
        column = matrix[:, 1] * np.cos(theta) + matrix[:, 2] * np.sin(theta)
    else:
        row = matrix[:, 0] * np.cos(theta) + matrix[:, 2] * np.sin(theta)
        column = matrix[:, 1]
    normal = np.cross(row, column)
    midpoint = np.array(image.TransformContinuousIndexToPhysicalPoint((15.5, 23.5, 18.0)))
    if slice_count is None:
        slice_count = min(36, max(3, int(36 * spacing[2] * 0.85 * np.cos(theta) / 8.0) + 1))
    offsets = (np.arange(slice_count) - (slice_count - 1) / 2.0) * 8.0
    reference = image.TransformIndexToPhysicalPoint((8, 17, 18))
    plan = {
        "plan_id": "thick_lung_localizer", "slice_part": "肺部",
        "slice_type": "MPR", "plane": "横状位", "geometry_type": "plane_stack",
        "group_name": "whole_range", "fill_percent": 90.0,
        "extent_mm": {"width": 31 * spacing[0], "height": 47 * spacing[1]},
        "output_spacing_mm": {"x": 0.75, "y": 0.75, "z": 8.0},
        "slice_thickness": 8.0, "slice_spacing": 8.0,
        "window_presets": ["肺窗", "纵隔窗"], "localizer_plane": localizer_plane,
        "localizer_reference": dict(zip("xyz", reference)),
        "slices": [
            {"index": index, "center": dict(zip("xyz", midpoint + normal * offset)),
             "row_dir": dict(zip("xyz", row)), "col_dir": dict(zip("xyz", column)),
             "normal": dict(zip("xyz", normal))}
            for index, offset in enumerate(offsets, 1)
        ],
    }
    return image, plan


def _geometry(image):
    return {
        "size": image.GetSize(), "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(), "direction": image.GetDirection(),
        "ipp": image.GetMetaData("0020|0032"), "iop": image.GetMetaData("0020|0037"),
    }


def _patient_geometry(image):
    origin = np.array([float(value) for value in image.GetMetaData("0020|0032").split("\\")])
    direction = np.array([float(value) for value in image.GetMetaData("0020|0037").split("\\")])
    return origin, direction[:3], direction[3:]


def _groups(indices):
    return 0 if not len(indices) else 1 + int(np.count_nonzero(np.diff(indices) > 1))


def _assert_patient_geometry_and_background(source, plan, render):
    background = render.background_image
    origin, row, column = _patient_geometry(background)
    fixed_axis = 0 if plan["localizer_plane"] == "矢状位" else 1
    horizontal_axis = 1 if fixed_axis == 0 else 0
    fixed_index = 8 if fixed_axis == 0 else 17
    direction = np.array(source.GetDirection()).reshape(3, 3)
    reverse = direction[2, 2] > 0.0
    first_index = [0, 0, 36 if reverse else 0]
    last_index = [0, 0, 0 if reverse else 36]
    first_index[fixed_axis] = last_index[fixed_axis] = fixed_index
    last_index[horizontal_axis] = source.GetSize()[horizontal_axis] - 1
    np.testing.assert_allclose(origin, source.TransformIndexToPhysicalPoint(first_index), atol=1e-7)
    last_pixel = (origin + row * (background.GetSize()[0] - 1) * background.GetSpacing()[0]
                  + column * (background.GetSize()[1] - 1) * background.GetSpacing()[1])
    np.testing.assert_allclose(last_pixel, source.TransformIndexToPhysicalPoint(last_index), atol=1e-7)
    np.testing.assert_allclose(row, direction[:, horizontal_axis], atol=1e-9)
    np.testing.assert_allclose(column, direction[:, 2] * (-1 if reverse else 1), atol=1e-9)
    # Independent interpolation of a known source field catches movement to
    # another reference layer as well as geometry/pixel mismatches after resize.
    rows, columns = np.indices(render.line_alpha.shape)
    points = (origin + columns[..., None] * row * background.GetSpacing()[0]
              + rows[..., None] * column * background.GetSpacing()[1])
    indices = ((points - source.GetOrigin()) @ direction) / source.GetSpacing()
    expected = 5 * indices[..., 0] + indices[..., 1] + 3 * indices[..., 2] - 900
    np.testing.assert_allclose(sitk.GetArrayFromImage(background), expected, atol=1e-3)


def _assert_separate_physical_lines(plan, render):
    background, alpha = render.background_image, render.line_alpha
    origin, row, column = _patient_geometry(background)
    spacing_x, spacing_y = background.GetSpacing()
    assert np.count_nonzero(alpha > 1e-4) / alpha.size < 0.4
    for column_index in (0, alpha.shape[1] // 2, alpha.shape[1] - 1):
        profile = alpha[:, column_index]
        painted_rows = np.flatnonzero(profile > 1e-4)
        assert _groups(painted_rows) == len(plan["slices"])
        components = np.split(painted_rows, np.flatnonzero(np.diff(painted_rows) > 1) + 1)
        # At this angle a one-pixel antialiased line reaches at most three
        # rows; increasing the source thickness must not increase that width.
        assert max(map(len, components)) <= 3
        base = origin + row * column_index * spacing_x
        centers = []
        for planned_slice in plan["slices"]:
            center = np.array(list(planned_slice["center"].values()))
            normal = np.array(list(planned_slice["normal"].values()))
            centers.append(center)
            true_row = float((center - base) @ normal / ((column * spacing_y) @ normal))
            nearby = np.arange(max(0, int(np.floor(true_row)) - 2),
                               min(len(profile), int(np.ceil(true_row)) + 3))
            weights = profile[nearby]
            assert weights.max() > 0.25
            painted_row = float(nearby @ weights / weights.sum())
            assert abs(painted_row - true_row) <= 1.0
        # Every neighboring pair has unpainted background between the lines.
        for first, second in zip(centers, centers[1:]):
            midpoint_row = float(((first + second) / 2 - base) @ normal
                                 / ((column * spacing_y) @ normal))
            assert profile[int(round(midpoint_row))] < 1e-6


@pytest.mark.parametrize("slice_spacing", (2.0, 5.0, 10.0, 20.0))
@pytest.mark.parametrize("localizer_plane", ("矢状位", "冠状位"))
@pytest.mark.parametrize("direction", ("identity", "reverse_z", "rotated"))
def test_coarse_source_localizers_have_fine_separate_physically_aligned_lines(
    slice_spacing, localizer_plane, direction,
):
    image, plan = _case(spacing=(0.75, 0.75, slice_spacing),
                        localizer_plane=localizer_plane, direction=direction)
    before = deepcopy(plan)
    original_pixels = sitk.GetArrayFromImage(image)
    original_geometry = (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())
    render = ReconstructionStage._build_plan_localizer_render(image, plan)
    assert render is not None
    assert render.background_image.GetSpacing()[1] <= 0.75 + 1e-9
    assert render.background_image.GetSpacing()[0] == pytest.approx(0.75)
    assert render.background_image.GetSize()[1] > image.GetSize()[2]
    assert render.background_image.GetSize()[0] == image.GetSize()[1 if localizer_plane == "矢状位" else 0]
    _assert_patient_geometry_and_background(image, plan, render)
    _assert_separate_physical_lines(plan, render)
    assert plan == before
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), original_pixels)
    assert (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()) == original_geometry


@pytest.mark.parametrize("localizer_plane", ("矢状位", "冠状位"))
def test_isotropic_five_mm_source_caps_both_display_axes_at_one_mm(localizer_plane):
    image, plan = _case(spacing=(5.0, 5.0, 5.0), localizer_plane=localizer_plane, slice_count=7)
    render = ReconstructionStage._build_plan_localizer_render(image, plan)
    assert render is not None
    assert max(render.background_image.GetSpacing()) <= 1.0
    _assert_patient_geometry_and_background(image, plan, render)
    _assert_separate_physical_lines(plan, render)


def test_display_refinement_leaves_fill_and_mpr_sampling_unchanged():
    image, original = _case(slice_count=3)
    resampler = PlaneStackResampler()
    mpr_before = resampler.sample_slice(image, original, original["slices"][0])
    captures = []
    for fill in (0.0, 50.0, 90.0, 100.0):
        plan = deepcopy(original)
        plan["fill_percent"] = fill
        # Fill changes finite MPR extent and in-plane center; neither changes
        # an infinite plane's Localizer intersection or fixed reference layer.
        plan["extent_mm"] = {"width": 15.0 + fill, "height": 20.0 + fill}
        for planned_slice in plan["slices"]:
            planned_slice["center"]["x"] += fill / 10.0
        before = deepcopy(plan)
        render = ReconstructionStage._build_plan_localizer_render(image, plan)
        assert render is not None
        assert plan == before
        captures.append((_geometry(render.background_image),
                         sitk.GetArrayFromImage(render.background_image), render.line_alpha))
    for geometry, background, alpha in captures[1:]:
        assert geometry == captures[0][0]
        np.testing.assert_array_equal(background, captures[0][1])
        np.testing.assert_array_equal(alpha, captures[0][2])
    mpr_after = resampler.sample_slice(image, original, original["slices"][0])
    assert mpr_after.GetSize() == mpr_before.GetSize()
    assert mpr_after.GetSpacing() == mpr_before.GetSpacing()
    assert mpr_after.GetOrigin() == mpr_before.GetOrigin()
    assert mpr_after.GetDirection() == mpr_before.GetDirection()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(mpr_after), sitk.GetArrayFromImage(mpr_before))


def test_mildly_anisotropic_thin_slice_localizer_keeps_original_grid():
    image, plan = _case(spacing=(0.8, 0.9, 1.2), slice_count=3)
    render = ReconstructionStage._build_plan_localizer_render(image, plan)
    assert render is not None
    assert render.background_image.GetSize() == (48, 37)
    assert render.background_image.GetSpacing() == (0.9, 1.2)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(render.background_image),
                                  sitk.GetArrayFromImage(image)[:, :, 8][::-1])


def test_thick_localizer_dicom_windows_share_refined_patient_geometry(tmp_path):
    image, plan = _case(slice_count=3)
    before = deepcopy(plan)
    render = ReconstructionStage._build_plan_localizer_render(image, plan)
    assert render is not None
    result = ReconstructionStage({"parallel": False})._execute_plan(
        image=image, plan=plan, output_root=tmp_path,
        study_uid="1.2.826.0.1.3680043.2.1125.1026", source_metadata={},
    )
    assert plan == before
    localizers = [pydicom.dcmread(str(Path(directory) / "Localizer.dcm"))
                  for directory in result["window_outputs"].values()]
    assert len(localizers) == 2
    first = localizers[0]
    assert float(first.PixelSpacing[0]) <= 1.0
    assert (first.Columns, first.Rows) == render.background_image.GetSize()
    np.testing.assert_allclose(first.ImagePositionPatient, _patient_geometry(render.background_image)[0])
    np.testing.assert_allclose(first.PixelSpacing, render.background_image.GetSpacing()[::-1])
    for localizer in localizers[1:]:
        for field in ("Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient"):
            assert getattr(localizer, field) == getattr(first, field)
    assert not np.array_equal(first.pixel_array, localizers[1].pixel_array)
    assert np.all(first.pixel_array[render.line_alpha > 0.5] >= 32767)
