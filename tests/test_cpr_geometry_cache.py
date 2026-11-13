"""CPR 几何去重必须保留采样、方向翻转及缓存失效的原有结果。"""

import copy
from unittest import mock

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.curved_stack import CurvedStackResampler


def _plan():
    return {
        "curve_points": [
            {"x": 7.0, "y": 5.0, "z": 2.0},
            {"x": 7.0, "y": 5.0, "z": 7.0},
            {"x": 7.0, "y": 7.0, "z": 12.0},
        ],
        "curve_spacing_mm": 1.0,
        "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        "offsets_mm": [-2.0000001, 2.0000001],
        "extent_mm": {"width": 4.0, "height": 10.0},
        "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
        "slice_thickness": 1.0,
        "slices": [{"index": 1, "offset_mm": -2.0000001}],
    }


def _image():
    z, y, x = np.indices((16, 16, 16))
    return sitk.GetImageFromArray((x * 3 + y * 17 + z * 101).astype(np.int16))


def _assert_same_image(actual, expected):
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(expected))
    assert actual.GetOrigin() == expected.GetOrigin()
    assert actual.GetSpacing() == expected.GetSpacing()
    assert actual.GetDirection() == expected.GetDirection()


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
def test_parallel_field_reuses_zero_smoothing_without_building_surface(axis):
    samples = np.column_stack((np.zeros(5), np.zeros(5), np.arange(5.0)))
    axis = np.asarray(axis)
    expected = CurvedStackResampler._curve_frame(samples, axis)[2]
    with mock.patch.object(
        CurvedStackResampler, "_surface_frame", side_effect=AssertionError("unused surface"),
    ), mock.patch.object(
        CurvedStackResampler, "_transport_offset_directions",
        wraps=CurvedStackResampler._transport_offset_directions,
    ) as transport:
        actual = CurvedStackResampler._parallel_offset_directions(samples, axis, 80.0)
    np.testing.assert_array_equal(actual, expected)
    assert transport.call_count == 1


def test_surface_frame_is_shared_by_mask_and_hu_without_retaining_row_flip():
    plan, image = _plan(), _image()
    mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(image) % 3 == 0).astype(np.uint8))
    mask.CopyInformation(image)
    original = plan["slices"][0]
    flipped = {**original, "row_dir": {"x": 0.0, "y": -1.0, "z": 0.0}}
    calls = [(mask, original, True), (image, flipped, False), (image, original, False)]
    expected = [
        CurvedStackResampler().sample_slice(source, plan, layer, is_mask=is_mask)
        for source, layer, is_mask in calls
    ]
    sampler = CurvedStackResampler()
    with mock.patch.object(sampler, "_surface_frame", wraps=sampler._surface_frame) as frame:
        actual = [sampler.sample_slice(source, plan, layer, is_mask=is_mask)
                  for source, layer, is_mask in calls]
    assert frame.call_count == 1
    for output, baseline in zip(actual, expected):
        _assert_same_image(output, baseline)
    assert actual[1].GetDirection() != actual[2].GetDirection()


@pytest.mark.parametrize("change", ["curve", "rows", "axis", "offsets", "support", "projection"])
def test_surface_cache_invalidates_when_prepared_geometry_changes(change):
    plan, image = _plan(), _image()
    sampler = CurvedStackResampler()
    sampler.sample_slice(image, plan, plan["slices"][0])
    changed = copy.deepcopy(plan)
    if change == "curve":
        changed["curve_points"][1]["y"] += 0.5
    elif change == "rows":
        changed["output_spacing_mm"]["y"] = 0.5
    elif change == "axis":
        changed["offset_axis"] = {"x": 1.0, "y": 0.2, "z": 0.0}
    elif change == "offsets":
        changed["offsets_mm"].append(3.0)
    elif change == "support":
        changed["slice_thickness"] = 2.0
    else:
        changed["curve_projection_axis"] = {"x": 0.0, "y": 1.0, "z": 0.0}
        changed["slices"][0]["center"] = {"x": 7.0, "y": 6.0, "z": 7.0}
    with mock.patch.object(sampler, "_surface_frame", wraps=sampler._surface_frame) as frame:
        actual = sampler.sample_slice(image, changed, changed["slices"][0])
    expected = CurvedStackResampler().sample_slice(image, changed, changed["slices"][0])
    assert frame.call_count == 1
    _assert_same_image(actual, expected)


def test_localizer_offsets_reuse_base_field_for_rounding_and_added_zero():
    plan, image = _plan(), _image()
    sampler = CurvedStackResampler()
    sampler.sample_slice(image, plan, plan["slices"][0])
    axis = np.asarray([1.0, 0.0, 0.0])
    offsets = [-2.0, 0.0, 2.0]
    support = sampler._maximum_offset_extent_mm(plan)
    expected = sampler._offset_curve_samples(
        sampler._resample_curve(sampler._sampling_curve(plan), 11),
        axis, offsets, support_mm=support,
    )
    with mock.patch.object(
        sampler, "_offset_curve_geometry", side_effect=AssertionError("duplicate field"),
    ):
        actual = sampler.offset_curves_for_plan(
            plan, rows=11, offset_axis=axis, offsets_mm=offsets, support_mm=support,
        )
    for offset in offsets:
        np.testing.assert_array_equal(actual[offset], expected[offset])


@pytest.mark.parametrize("change", ["curve", "rows", "axis", "support", "offset_support"])
def test_localizer_field_mismatch_falls_back_without_invalidating_hu_cache(change):
    plan, image = _plan(), _image()
    sampler = CurvedStackResampler()
    baseline = sampler.sample_slice(image, plan, plan["slices"][0])
    original_key = sampler._prepared_geometry_key
    changed = copy.deepcopy(plan)
    rows, axis, offsets, support = 11, np.asarray([1.0, 0.0, 0.0]), [0.0], 3.0
    if change == "curve":
        changed["curve_points"][1]["y"] += 0.5
    elif change == "rows":
        rows = 12
    elif change == "axis":
        axis = np.asarray([0.0, 1.0, 0.0])
    elif change == "support":
        support = 90.0
    else:
        offsets = [90.0]
    expected = sampler._offset_curve_samples(
        sampler._resample_curve(sampler._sampling_curve(changed), rows),
        axis, offsets, support_mm=support,
    )
    with mock.patch.object(
        sampler, "_offset_curve_geometry", wraps=sampler._offset_curve_geometry,
    ) as geometry:
        actual = sampler.offset_curves_for_plan(
            changed, rows=rows, offset_axis=axis, offsets_mm=offsets, support_mm=support,
        )
    assert geometry.call_count == 1
    for offset in offsets:
        np.testing.assert_array_equal(actual[offset], expected[offset])
    assert sampler._prepared_geometry_key == original_key
    with mock.patch.object(sampler, "_surface_frame", side_effect=AssertionError("lost cache")):
        repeated = sampler.sample_slice(image, plan, plan["slices"][0])
    _assert_same_image(repeated, baseline)
