"""Anatomy-region occupancy uses a fixed box center and original CT sampling."""

import copy
from itertools import product
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.reconstruction.framing import TargetFramer
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image
from tests.test_body_preserving_canvas import _fixture, _geometry, _stage, _vector


POLICY = {"basis": "selected_region", "margin_mm": 5.0}


def _region_fixture(part="髋骨", percent=100, *, tilted=False, plane="冠状位"):
    raw, seg, plan, coords, body, axes = _fixture(part, percent, plane=plane, tilted=tilted)
    plan["framing_policy"] = copy.deepcopy(POLICY)
    return raw, seg, plan, coords, body, axes


def _whole_label_vertices(seg):
    """Independent physical corners of all selected voxels, without slab filtering."""
    indices = np.argwhere(sitk.GetArrayViewFromImage(seg) == 1)[:, ::-1]
    voxel_axes = np.asarray(seg.GetDirection()).reshape(3, 3) * seg.GetSpacing()
    centers = indices @ voxel_axes.T + seg.GetOrigin()
    offsets = np.asarray(list(product((-0.5, 0.5), repeat=3))) @ voxel_axes.T
    return (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)


def _expected_region(seg, axes, margin=5.0):
    projected = _whole_label_vertices(seg) @ axes[:, :2]
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    return (lower + upper) / 2, upper - lower + 2 * margin, lower, upper


@pytest.mark.parametrize("tilted", (False, True))
@pytest.mark.parametrize("plane", ("横状位", "冠状位", "矢状位"))
@pytest.mark.parametrize("percent", (100, 80, 50))
def test_selected_region_has_global_bounds_and_static_center_even_for_small_or_empty_slabs(tilted, plane, percent):
    raw, seg, plan, coords, _, axes = _region_fixture(percent=percent, tilted=tilted, plane=plane)
    original = copy.deepcopy(plan)
    expected_center, required, lower, upper = _expected_region(seg, axes)
    required /= percent / 100
    framer = TargetFramer(seg, coords, plan, source_image=raw)
    geometries = []
    for item in plan["slices"]:
        sampling_plan, framed = framer.plane_geometry(plan, item)
        details = framed["framing"]
        center = _vector(framed["center"])
        spacing = np.asarray([sampling_plan["output_spacing_mm"][axis] for axis in "xy"])
        size = np.asarray(details["output_size_xy"])
        fov = size * spacing
        np.testing.assert_allclose(center @ axes[:, :2], expected_center, atol=1e-8)
        assert abs((center - _vector(item["center"])) @ axes[:, 2]) < 1e-8
        assert np.all(fov >= required - 1e-8)
        assert np.all(fov - required <= 2 * spacing + 1e-8)
        assert np.all(lower >= expected_center - fov / 2 + 5 - 1e-8)
        assert np.all(upper <= expected_center + fov / 2 - 5 + 1e-8)
        assert details["center_source"] == "group_selected_region"
        assert details["selected_region"]["basis"] == "selected_region"
        np.testing.assert_allclose(details["selected_region"]["lower_in_plane_mm"], lower)
        np.testing.assert_allclose(details["selected_region"]["upper_in_plane_mm"], upper)
        assert "source_preservation" not in details
        assert "reference_slice_index" not in details
        geometries.append((*size, *spacing))
    np.testing.assert_allclose(geometries, [geometries[0]] * len(geometries), atol=1e-10)
    # The fixture includes small targets and target-free plans; neither may move the box.
    if plane == "冠状位" and not tilted:
        depths = np.argwhere(sitk.GetArrayViewFromImage(seg) == 1)[:, 1]
        assert depths.max() < 90
        assert len(plan["slices"]) == 4  # final two slabs are outside the selected labels
    assert plan == original


def test_region_frame_may_crop_unselected_body_instead_of_expanding_to_skin():
    raw, seg, plan, coords, body, axes = _region_fixture()
    framer = TargetFramer(seg, coords, plan, source_image=raw)
    sampling_plan, framed = framer.plane_geometry(plan, plan["slices"][0])
    spacing = np.asarray([sampling_plan["output_spacing_mm"][axis] for axis in "xy"])
    fov = np.asarray(framed["framing"]["output_size_xy"]) * spacing
    center = _vector(framed["center"]) @ axes[:, :2]
    body_indices = np.argwhere(body)[:, ::-1]
    body_points = body_indices * raw.GetSpacing() + raw.GetOrigin()
    outside = np.any(np.abs(body_points @ axes[:, :2] - center) > fov / 2, axis=1)
    assert outside.sum() > 1000
    selected = _whole_label_vertices(seg) @ axes[:, :2]
    assert np.all(np.abs(selected - center) <= fov / 2 + 1e-8)


def test_region_dicom_keeps_unlabelled_midline_bone_and_shared_windows_without_body_extraction(tmp_path, monkeypatch):
    import src.reconstruction.framing as framing_module
    import src.stages.stage_reconstruction as stage_module

    raw, seg, plan, _, _, axes = _region_fixture()
    array = sitk.GetArrayFromImage(raw)
    # Posterior, target-free slabs still contain a central bone inside the global
    # hip box. It is intentionally absent from segmentation, like sacral context.
    landmark = (slice(10, 14), slice(86, 114), slice(96, 103))
    assert not np.any(sitk.GetArrayViewFromImage(seg)[landmark])
    array[landmark] = 1800
    source = sitk.GetImageFromArray(array)
    source.CopyInformation(raw)

    class ForbiddenBodySupport:
        def __init__(self, *args, **kwargs):
            pytest.fail("explicit selected_region must not extract or inspect whole body")

    def forbidden_vertical_crop(*args, **kwargs):
        pytest.fail("region canvas must retain its global in-plane center")

    monkeypatch.setattr(stage_module, "BodyImageSupport", ForbiddenBodySupport)
    monkeypatch.setattr(framing_module, "BodyImageSupport", ForbiddenBodySupport)
    monkeypatch.setattr(stage_module, "crop_vertical_padding", forbidden_vertical_crop)
    monkeypatch.setattr(stage_module, "load_framing_context", lambda *args, **kwargs: pytest.fail(
        "explicit selected_region must not read automatic neighboring-label context"))
    output = _stage(tmp_path, source, seg, plan)
    expected_center, required, _, _ = _expected_region(seg, axes)
    all_geometry = []
    for item in plan["slices"]:
        datasets = [pydicom.dcmread(Path(output["window_outputs"][window]) / f"{item['index']}.dcm")
                    for window in plan["window_presets"]]
        first = datasets[0]
        for other in datasets[1:]:
            for tag in ("Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient", "SliceThickness"):
                np.testing.assert_array_equal(getattr(first, tag), getattr(other, tag))
        size, spacing, output_axes, origin, center = _geometry(first)
        np.testing.assert_allclose(center @ axes[:, :2], expected_center, atol=1e-5)
        np.testing.assert_allclose(output_axes, axes[:, :2], atol=1e-7)
        assert np.all(size * spacing >= required - 1e-5)
        all_geometry.append((*size, *spacing))
        if item["index"] in (3, 4):
            # Project the known central landmark to final DICOM coordinates,
            # rather than using production label/body framing as pixel truth.
            middle = np.asarray(source.TransformContinuousIndexToPhysicalPoint((99.0, float((90, 108)[item['index'] - 3]), 11.5)))
            xy = (middle - origin) @ output_axes / spacing
            x, y = np.rint(xy).astype(int)
            assert 2 <= x < int(first.Columns) - 2
            assert 2 <= y < int(first.Rows) - 2
            for dataset in datasets:
                assert dataset.pixel_array[y-2:y+3, x-2:x+3].max() > 60000
    np.testing.assert_allclose(all_geometry, [all_geometry[0]] * len(all_geometry), atol=1e-8)


@pytest.mark.parametrize("part", ("肺部", "髋骨", "颈椎"))
@pytest.mark.parametrize("percent", (None, 0))
def test_zero_or_null_ignores_policy_and_preserves_exact_geometry_and_window_pixels(part, percent):
    raw, seg, configured, coords, _, _ = _region_fixture(part, percent, tilted=True)
    baseline = copy.deepcopy(configured)
    baseline.pop("framing_policy")
    before = TargetFramer(seg, coords, baseline, source_image=raw)
    after = TargetFramer(seg, coords, configured, source_image=raw)
    for old_item, new_item in zip(baseline["slices"], configured["slices"]):
        old_plan, old_slice = before.plane_geometry(baseline, old_item)
        new_plan, new_slice = after.plane_geometry(configured, new_item)
        stripped = copy.deepcopy(new_plan)
        stripped.pop("framing_policy")
        assert stripped == old_plan
        assert new_slice == old_slice
        old_sample = PlaneStackResampler().sample_slice(raw, old_plan, old_slice)
        new_sample = PlaneStackResampler().sample_slice(raw, new_plan, new_slice)
        for window in configured["window_presets"]:
            np.testing.assert_array_equal(
                sitk.GetArrayFromImage(apply_window_to_image(old_sample, window)),
                sitk.GetArrayFromImage(apply_window_to_image(new_sample, window)),
            )


@pytest.mark.parametrize("part", ("肺部", "颅脑", "颈椎", "胸椎", "腰椎", "骶骨"))
def test_explicit_region_policy_is_generic_and_opt_in(part):
    raw, seg, configured, coords, _, axes = _region_fixture(part)
    expected_center, expected_size, _, _ = _expected_region(seg, axes)
    framer = TargetFramer(seg, coords, configured, source_image=raw)
    for item in configured["slices"]:
        _, framed = framer.plane_geometry(configured, item)
        np.testing.assert_allclose(_vector(framed["center"]) @ axes[:, :2], expected_center)
        np.testing.assert_allclose(framed["framing"]["framing_target_size_mm"], expected_size)
    unconfigured = copy.deepcopy(configured)
    unconfigured.pop("framing_policy")
    ordinary = TargetFramer(seg, coords, unconfigured, source_image=raw)
    for item in unconfigured["slices"]:
        _, framed = ordinary.plane_geometry(unconfigured, item)
        assert "selected_region" not in framed["framing"]
        assert framed["framing"]["center_source"] != "group_selected_region"


@pytest.mark.parametrize("overrides", ({"slice_type": "椎间隙"}, {"geometry_type": "curved_stack", "slice_type": "CPR"}))
def test_region_policy_does_not_enable_non_mpr_paths(overrides):
    raw, seg, plan, coords, _, _ = _region_fixture("颈椎")
    plan.update(overrides)
    framer = TargetFramer(seg, coords, plan, source_image=raw)
    assert framer._selected_region is None


def test_lower_occupancy_reveals_original_hu_outside_100_region_without_precropping(tmp_path):
    raw, seg, original_plan, coords, _, axes = _region_fixture()
    source_array = sitk.GetArrayFromImage(raw)
    markers = [(58, 2100.0), (44, 2300.0)]
    for x, hu in markers:
        source_array[10:14, 64:73, x-2:x+3] = hu
        assert not np.any(sitk.GetArrayViewFromImage(seg)[10:14, 64:73, x-2:x+3])
    source = sitk.GetImageFromArray(source_array)
    source.CopyInformation(raw)
    original_array = sitk.GetArrayFromImage(source)
    original_geometry = (source.GetOrigin(), source.GetDirection(), source.GetSpacing(), source.GetSize())
    centers, fovs, matrix, spacings = [], [], [], []
    for percent in (100, 80, 50):
        plan = copy.deepcopy(original_plan)
        plan["fill_percent"] = percent
        directory = tmp_path / str(percent)
        directory.mkdir()
        output = _stage(directory, source, seg, plan)
        item = plan["slices"][1]  # y=68: both HU sentinels lie in this original slab
        dataset = pydicom.dcmread(Path(output["window_outputs"]["骨窗"]) / "2.dcm")
        size, spacing, output_axes, origin, center = _geometry(dataset)
        centers.append(center @ axes[:, :2]);fovs.append(size * spacing)
        matrix.append(size);spacings.append(spacing)
        # Independently inspect the production sampler's HU at physical marker
        # coordinates. A cropped/reused 100% image could only add padding here.
        sampler = TargetFramer(seg, coords, plan, source_image=source)
        sampling_plan, sampling_item = sampler.plane_geometry(plan, item)
        sampled = PlaneStackResampler().sample_slice(source, sampling_plan, sampling_item)
        hu_pixels = sitk.GetArrayFromImage(sampled)[0]
        expected_visible = (False, False) if percent == 100 else ((True, False) if percent == 80 else (True, True))
        for (x_index, expected_hu), visible in zip(markers, expected_visible):
            point = np.asarray(source.TransformContinuousIndexToPhysicalPoint((float(x_index), 68.0, 11.5)))
            xy = (point - origin) @ output_axes / spacing
            x, y = np.rint(xy).astype(int)
            inside = 0 <= x < int(dataset.Columns) and 0 <= y < int(dataset.Rows)
            assert inside == visible
            if visible:
                assert hu_pixels[y, x] == pytest.approx(expected_hu, abs=1e-4)
                assert dataset.pixel_array[y, x] > 65000
        if percent == 100:
            assert dataset.pixel_array.max() < 60000  # neither outside sentinel leaked into 100%
    np.testing.assert_allclose(centers, [centers[0]] * 3, atol=1e-5)
    np.testing.assert_array_equal(matrix, [matrix[0]] * 3)
    np.testing.assert_allclose(spacings[1], spacings[0] * 1.25, atol=1e-8)
    np.testing.assert_allclose(spacings[2], spacings[0] * 2, atol=1e-8)
    np.testing.assert_allclose(fovs[1], fovs[0] * 1.25, atol=1e-8)
    np.testing.assert_allclose(fovs[2], fovs[0] * 2, atol=1e-8)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(source), original_array)
    assert (source.GetOrigin(), source.GetDirection(), source.GetSpacing(), source.GetSize()) == original_geometry
