"""Anatomy-region occupancy uses a fixed box center and original CT sampling."""

import copy
from itertools import product
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.reconstruction.framing import TargetFramer, selected_region_framing
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image
from tests.test_body_preserving_canvas import _body_vertices, _fixture, _geometry, _stage, _vector


POLICY = {"basis": "selected_region", "margin_mm": 5.0}
LUNG_HIP_MODES = (
    ("肺部", "横状位", "reference_slice"),
    ("肺部", "冠状位", "reference_slice"),
    ("肺部", "矢状位", "reference_slice"),
    ("髋骨", "横状位", "slice_target"),
    ("髋骨", "矢状位", "slice_target"),
    ("髋骨", "冠状位", "group"),
)


def _region_fixture(part="髋骨", percent=100, *, tilted=False, plane="冠状位", centering=None):
    raw, seg, plan, coords, body, axes = _fixture(part, percent, plane=plane, tilted=tilted)
    plan["framing_policy"] = copy.deepcopy(POLICY)
    if centering is not None:
        plan["framing_policy"]["centering"] = centering
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
        assert details["selected_region"]["centering"] == "group"
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
@pytest.mark.parametrize("centering", (None, "group", "reference_slice", "slice_target"))
@pytest.mark.parametrize("canvas", (None, "original"))
def test_zero_or_null_ignores_policy_and_preserves_exact_geometry_and_window_pixels(part, percent, centering, canvas):
    raw, seg, configured, coords, _, _ = _region_fixture(part, percent, tilted=True, centering=centering)
    if canvas is not None:
        configured["framing_policy"]["canvas"] = canvas
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


@pytest.mark.parametrize("centering", (None, "", "skin", "GROUP", True, 0, [], {}))
def test_runtime_policy_rejects_invalid_centering_in_external_plan(centering):
    _, _, plan, _, _, _ = _region_fixture()
    plan["framing_policy"]["centering"] = centering
    with pytest.raises(ValueError, match="centering"):
        selected_region_framing(plan)


@pytest.mark.parametrize("centering", ("group", "reference_slice", "slice_target"))
def test_runtime_policy_preserves_explicit_centering_and_legacy_default_shape(centering):
    _, _, plan, _, _, _ = _region_fixture()
    assert selected_region_framing(plan) == POLICY
    plan["framing_policy"]["centering"] = centering
    assert selected_region_framing(plan) == {**POLICY, "centering": centering}


@pytest.mark.parametrize("part,plane,centering", LUNG_HIP_MODES)
@pytest.mark.parametrize("tilted", (False, True))
def test_lung_hip_region_modes_preserve_target_centers_and_all_slab_labels_across_fill(part, plane, centering, tilted):
    raw, seg, original, coords, _, axes = _region_fixture(
        part, plane=plane, centering=centering, tilted=tilted,
    )
    # The unchanged no-source, non-100 path is an independent legacy centering
    # contract. It has no skin adjustment and uses the original target mode.
    legacy = copy.deepcopy(original)
    legacy.pop("framing_policy")
    legacy["fill_percent"] = 80
    old = TargetFramer(seg, coords, legacy)
    expected_centers = []
    for item in legacy["slices"]:
        _, framed = old.plane_geometry(legacy, item)
        expected_centers.append(_vector(framed["center"]))
    if centering == "group":
        group_center, _, _, _ = _expected_region(seg, axes)
        expected_centers = [
            _vector(item["center"]) + axes[:, :2] @ (
                group_center - _vector(item["center"]) @ axes[:, :2]
            ) for item in original["slices"]
        ]
    truth = [
        _body_vertices(sitk.GetArrayViewFromImage(seg) == 1, seg, item, axes, original["slice_thickness"])
        for item in original["slices"]
    ]
    recorded = []
    for percent in (100, 80, 50):
        plan = copy.deepcopy(original)
        plan["fill_percent"] = percent
        framer = TargetFramer(seg, coords, plan, source_image=raw)
        records = []
        for number, item in enumerate(plan["slices"]):
            sampling, framed = framer.plane_geometry(plan, item)
            details = framed["framing"]
            center = _vector(framed["center"])
            spacing = np.asarray([sampling["output_spacing_mm"][axis] for axis in "xy"])
            size = np.asarray(details["output_size_xy"])
            fov = size * spacing
            np.testing.assert_allclose(center, expected_centers[number], atol=1e-8)
            np.testing.assert_allclose((center - _vector(item["center"])) @ axes[:, 2], 0, atol=1e-8)
            if len(truth[number]):
                # Exact clipped voxel boxes provide label truth independently
                # of the production target/slab bounding-box implementation.
                projected = (truth[number] - center) @ axes[:, :2]
                assert np.all(np.abs(projected) <= fov / 2 - 5. + 1e-7)
            assert details["selected_region"]["centering"] == centering
            assert details["context_label_ids"] == []
            assert not {"source_preservation", "target_center_before_skin_edge", "vertical_source_support"} & details.keys()
            if centering == "reference_slice":
                assert details["center_source"] == "group_reference_slice"
                if plane == "横状位" and not tilted:
                    assert details["reference_slice_index"] == 2
                    if number == len(plan["slices"]) - 1:
                        assert details["slice_target_size_mm"] is None
            elif centering == "slice_target":
                expected_source = "slice_target" if len(truth[number]) else "group_fallback"
                assert details["center_source"] == expected_source
            else:
                assert details["center_source"] == "group_selected_region"
            records.append((center, spacing, size, fov))
        for _, spacing, size, _ in records[1:]:
            np.testing.assert_allclose(spacing, records[0][1], atol=1e-10)
            np.testing.assert_array_equal(size, records[0][2])
        if centering == "reference_slice":
            shifts = [(record[0] - _vector(item["center"])) @ axes[:, :2]
                      for record, item in zip(records, plan["slices"])]
            np.testing.assert_allclose(shifts, [shifts[0]] * len(shifts), atol=1e-8)
        recorded.append(records)
    for records, factor in zip(recorded[1:], (1.25, 2.)):
        for current, base in zip(records, recorded[0]):
            np.testing.assert_allclose(current[0], base[0], atol=1e-8)
            np.testing.assert_allclose(current[1], base[1] * factor, atol=1e-8)
            np.testing.assert_array_equal(current[2], base[2])
            np.testing.assert_allclose(current[3], base[3] * factor, atol=1e-8)


@pytest.mark.parametrize("centering", ("reference_slice", "slice_target"))
def test_target_free_group_uses_full_region_fallback_without_collapsing_or_moving_center(centering):
    raw, seg, original, coords, _, axes = _region_fixture("肺部", centering=centering)
    group_center, group_size, _, _ = _expected_region(seg, axes)
    for item in original["slices"]:
        center = _vector(item["center"]) + axes[:, 2] * 1000.
        item["center"] = dict(zip("xyz", map(float, center)))
    for percent in (100, 80, 50):
        plan = copy.deepcopy(original)
        plan["fill_percent"] = percent
        framer = TargetFramer(seg, coords, plan, source_image=raw)
        for item in plan["slices"]:
            sampling, framed = framer.plane_geometry(plan, item)
            details = framed["framing"]
            spacing = np.asarray([sampling["output_spacing_mm"][a] for a in "xy"])
            fov = np.asarray(details["output_size_xy"]) * spacing
            np.testing.assert_allclose(_vector(framed["center"]) @ axes[:, :2], group_center)
            np.testing.assert_allclose(details["framing_target_size_mm"], group_size)
            assert np.all(fov >= group_size / (percent / 100) - 1e-8)
            assert details["center_source"] == "group_fallback"
            assert details["slice_target_size_mm"] is None
            assert "reference_slice_index" not in details


@pytest.mark.parametrize("part,plane,centering", LUNG_HIP_MODES)
def test_lower_occupancy_reveals_original_hu_outside_100_region_without_precropping(
    tmp_path, monkeypatch, part, plane, centering,
):
    import src.reconstruction.framing as framing_module
    import src.stages.stage_reconstruction as stage_module

    raw, seg, original_plan, coords, _, axes = _region_fixture(part, plane=plane, centering=centering)
    source_array = sitk.GetArrayFromImage(raw)
    markers = [(34, 2100.0), (26, 2300.0)] if plane == "矢状位" else [(58, 2100.0), (44, 2300.0)]
    marker_points = []
    for index, hu in markers:
        if plane == "横状位":
            region = (slice(7, 12), slice(64, 73), slice(index-2, index+3))
            point_index = (float(index), 68., 9.)
        elif plane == "冠状位":
            region = (slice(10, 14), slice(64, 73), slice(index-2, index+3))
            point_index = (float(index), 68., 11.5)
        else:
            region = (slice(10, 14), slice(index-1, index+2), slice(94, 103))
            point_index = (98., float(index), 11.5)
        source_array[region] = hu
        assert not np.any(sitk.GetArrayViewFromImage(seg)[region])
        marker_points.append(np.asarray(raw.TransformContinuousIndexToPhysicalPoint(point_index)))
    source = sitk.GetImageFromArray(source_array)
    source.CopyInformation(raw)

    class ForbiddenBodySupport:
        def __init__(self, *args, **kwargs):
            pytest.fail("selected-region zoom must not inspect body or skin")

    def forbidden_automatic_framing(*args, **kwargs):
        pytest.fail("selected-region zoom must not add neighboring labels or crop padding")

    monkeypatch.setattr(framing_module, "BodyImageSupport", ForbiddenBodySupport)
    monkeypatch.setattr(stage_module, "BodyImageSupport", ForbiddenBodySupport)
    monkeypatch.setattr(stage_module, "load_framing_context", forbidden_automatic_framing)
    monkeypatch.setattr(stage_module, "crop_vertical_padding", forbidden_automatic_framing)
    original_array = sitk.GetArrayFromImage(source)
    original_geometry = (source.GetOrigin(), source.GetDirection(), source.GetSpacing(), source.GetSize())
    centers, fovs, matrix, spacings = [], [], [], []
    for percent in (100, 80, 50):
        plan = copy.deepcopy(original_plan)
        plan["fill_percent"] = percent
        directory = tmp_path / str(percent)
        directory.mkdir()
        output = _stage(directory, source, seg, plan)
        item = plan["slices"][1]  # Both HU sentinels lie in this original slab.
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
        for point, (_, expected_hu), visible in zip(marker_points, markers, expected_visible):
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
