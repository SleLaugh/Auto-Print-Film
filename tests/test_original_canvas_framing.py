"""Lung/hip zoom retains the real planner canvas and one physical group center."""

import copy
from itertools import product
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.config import load_slice_part_config
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.planning.base import select_planner
from src.reconstruction.framing import TargetFramer, selected_region_framing
from src.reconstruction.plane_stack import PlaneStackResampler
from tests.test_body_preserving_canvas import _fixture, _geometry, _stage, _vector
from tests.test_selected_region_framing import _whole_label_vertices


def _planned_fixture(part, plane, flipped=False, *, thickness=1., slice_spacing=2.):
    base_raw, base_seg, _, _, _, _ = _fixture(part, tilted=flipped)
    # A genuine anisotropic source, rather than a hand-written square plan.
    # Pad the source longitudinally so the selected region has real surrounding
    # CT in all four in-plane directions, including coronal/sagittal views.
    images = [sitk.GetImageFromArray(np.pad(
        sitk.GetArrayFromImage(image), ((32, 32), (0, 0), (0, 0)),
        constant_values=padding,
    )) for image, padding in ((base_raw, -1024.), (base_seg, 0))]
    raw, seg = images
    for image in images:
        image.SetSpacing((0.8, 1.1, 1.7))
        image.SetDirection(base_raw.GetDirection())
        image.SetOrigin(base_raw.GetOrigin())
    indices = np.argwhere(sitk.GetArrayViewFromImage(seg) == 1)
    bbox = {f"{axis}_{end}": int(getattr(indices[:, column], end)())
            for column, axis in enumerate("zyx") for end in ("min", "max")}
    coords = {"target": {"label_ids": [1]}}
    localization = {
        "image": {"size_xyz": list(raw.GetSize()), "spacing_xyz": list(raw.GetSpacing()),
                  "origin_xyz": list(raw.GetOrigin()), "direction": list(raw.GetDirection())},
        "correction_scope_bbox": bbox, "coords": coords,
        "slice_rules": load_slice_part_config()["slice_parts"][part]["slice_rules"],
    }
    request = Request(
        dicom_dir="/tmp/unused_source", output_dir="/tmp/unused_output", slice_part=SlicePartEnum(part),
        plane_configs=[PlaneConfig(PlaneTypeEnum(plane), thickness, slice_spacing,
                                   ["骨窗", "软组织窗"], fill_percent=100)],
    )
    plan = select_planner(part, "MPR").build_plans(request, localization).plans[0].to_dict()
    return raw, seg, coords, plan


def _axes(plan):
    return np.column_stack([_vector(plan["slices"][0][key])
                            for key in ("row_dir", "col_dir", "normal")])


def _original_matrix(plan):
    spacing = np.asarray([plan["output_spacing_mm"][a] for a in "xy"])
    extent = np.asarray([plan["extent_mm"][a] for a in ("width", "height")])
    return np.rint(extent / spacing).astype(int) + 1, spacing


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("横状位", "冠状位", "矢状位"))
@pytest.mark.parametrize("flipped", (False, True))
def test_original_canvas_keeps_source_plan_aspect_and_static_region_framing(part, plane, flipped):
    raw, seg, coords, original = _planned_fixture(part, plane, flipped)
    assert original["framing_policy"] == {
        "basis": "selected_region", "margin_mm": 5.0, "canvas": "original",
    }
    axes = _axes(original)
    size, base_spacing = _original_matrix(original)
    source_centers = np.asarray([
        raw.TransformIndexToPhysicalPoint(corner)
        for corner in product(*[(0, n - 1) for n in raw.GetSize()])
    ])
    # Verify the production Stage6 plan actually came from the source canvas,
    # rather than silently constructing a narrow label-bbox-shaped matrix.
    projected_source = source_centers @ axes[:, :2]
    np.testing.assert_allclose(
        [original["extent_mm"][a] for a in ("width", "height")],
        np.ptp(projected_source, axis=0), atol=1e-7,
    )
    selected = _whole_label_vertices(seg) @ axes[:, :2]
    lower, upper = selected.min(axis=0), selected.max(axis=0)
    group_center = (lower + upper) / 2
    records = []
    empty_tail_seen = False
    for percent in (100, 70, 50):
        plan = copy.deepcopy(original)
        plan["fill_percent"] = percent
        framer = TargetFramer(seg, coords, plan, source_image=raw)
        geometry = []
        for item in plan["slices"]:
            sampling, framed = framer.plane_geometry(plan, item)
            details = framed["framing"]
            spacing = np.asarray([sampling["output_spacing_mm"][a] for a in "xy"])
            fov = np.asarray(details["output_size_xy"]) * spacing
            center = _vector(framed["center"])
            np.testing.assert_array_equal(details["output_size_xy"], size)
            np.testing.assert_allclose(center @ axes[:, :2], group_center, atol=1e-8)
            np.testing.assert_allclose((center - _vector(item["center"])) @ axes[:, 2], 0, atol=1e-8)
            np.testing.assert_allclose(spacing / base_spacing, spacing[0] / base_spacing[0], atol=1e-10)
            assert np.all(lower >= group_center - fov / 2 + 5. - 1e-7)
            assert np.all(upper <= group_center + fov / 2 - 5. + 1e-7)
            np.testing.assert_allclose(fov[0] / fov[1], (size * base_spacing)[0] / (size * base_spacing)[1])
            assert details["center_source"] == "group_selected_region"
            assert details["selected_region"]["centering"] == "group"
            assert details["selected_region"]["canvas"] == "original"
            assert not {"source_preservation", "reference_slice_index", "vertical_source_support"} & details.keys()
            empty_tail_seen |= details["slice_target_size_mm"] is None
            geometry.append((center, spacing, fov))
        np.testing.assert_allclose([row[1] for row in geometry], [geometry[0][1]] * len(geometry), atol=1e-10)
        records.append(geometry)
    for other, factor in zip(records[1:], (1 / .7, 2.)):
        for current, base in zip(other, records[0]):
            np.testing.assert_allclose(current[0], base[0], atol=1e-8)
            np.testing.assert_allclose(current[1], base[1] * factor, atol=1e-8)
            np.testing.assert_allclose(current[2], base[2] * factor, atol=1e-8)
    if part == "肺部" and plane == "横状位":
        assert empty_tail_seen, "real extended source plans include an empty end layer"


@pytest.mark.parametrize("canvas", (None, "", "source", "ORIGINAL", True, 0, [], {}))
def test_runtime_rejects_invalid_canvas_from_external_plan(canvas):
    _, _, _, plan = _planned_fixture("肺部", "横状位")
    plan["framing_policy"]["canvas"] = canvas
    with pytest.raises(ValueError, match="canvas"):
        selected_region_framing(plan)


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("横状位", "冠状位", "矢状位"))
def test_original_canvas_reveals_raw_hu_above_below_left_and_right(tmp_path, monkeypatch, part, plane):
    import src.reconstruction.framing as framing_module
    import src.stages.stage_reconstruction as stage_module

    raw, seg, coords, original = _planned_fixture(part, plane, thickness=4., slice_spacing=4.)
    axes = _axes(original)
    selected = _whole_label_vertices(seg)
    projected = selected @ axes[:, :2]
    group_xy = (projected.min(axis=0) + projected.max(axis=0)) / 2
    normal_depth = float(np.mean(selected @ axes[:, 2]))
    item = min(original["slices"], key=lambda value: abs(_vector(value["center"]) @ axes[:, 2] - normal_depth))
    physical_center = axes[:, :2] @ group_xy + axes[:, 2] * (_vector(item["center"]) @ axes[:, 2])
    size, base_spacing = _original_matrix(original)
    # Independently place real HU blocks in concentric rings: one visible at
    # 70%, and one only at 50%. Each ring has all four in-plane directions.
    required = np.ptp(projected, axis=0) + 10.
    fov100 = size * base_spacing * np.max(required / (size * base_spacing))
    pixels = sitk.GetArrayFromImage(raw)
    probes = []
    source_direction = np.asarray(raw.GetDirection()).reshape(3, 3)
    normal_axis = int(np.argmax(np.abs(source_direction.T @ axes[:, 2])))
    for ring, multiplier in enumerate((1.2, 1.75)):
        for axis, sign in product((0, 1), (-1, 1)):
            point = physical_center + axes[:, axis] * sign * multiplier * fov100[axis] / 2
            index = np.rint(raw.TransformPhysicalPointToContinuousIndex(tuple(point))).astype(int)
            radius = np.ones(3, dtype=int)
            radius[normal_axis] = 5
            assert np.all(index - radius >= 0) and np.all(index + radius < np.asarray(raw.GetSize()))
            region = tuple(slice(i-r, i+r+1) for i, r in zip(index[::-1], radius[::-1]))
            hu = 2500. + len(probes) * 100.
            pixels[region] = hu
            probes.append((np.asarray(raw.TransformIndexToPhysicalPoint(tuple(map(int, index)))), hu, ring))
    source = sitk.GetImageFromArray(pixels)
    source.CopyInformation(raw)
    original_pixels = sitk.GetArrayFromImage(source)

    class ForbiddenBodySupport:
        def __init__(self, *args, **kwargs):
            pytest.fail("original target zoom must not inspect body or skin")

    def forbidden_auxiliary_framing(*args, **kwargs):
        pytest.fail("original target zoom must not load context or crop canvas rows")

    monkeypatch.setattr(framing_module, "BodyImageSupport", ForbiddenBodySupport)
    monkeypatch.setattr(stage_module, "BodyImageSupport", ForbiddenBodySupport)
    monkeypatch.setattr(stage_module, "load_framing_context", forbidden_auxiliary_framing)
    monkeypatch.setattr(stage_module, "crop_vertical_padding", forbidden_auxiliary_framing)
    recorded = []
    for percent in (100, 70, 50):
        plan = copy.deepcopy(original)
        plan["fill_percent"] = percent
        directory = tmp_path / str(percent)
        directory.mkdir()
        output = _stage(directory, source, seg, plan)
        dataset = pydicom.dcmread(Path(output["window_outputs"]["骨窗"]) / f"{item['index']}.dcm")
        out_size, spacing, output_axes, origin, center = _geometry(dataset)
        np.testing.assert_array_equal(out_size, size)
        np.testing.assert_allclose(center @ axes[:, :2], group_xy, atol=1e-5)
        for window in plan["window_presets"]:
            other = pydicom.dcmread(Path(output["window_outputs"][window]) / f"{item['index']}.dcm")
            for tag in ("Rows", "Columns", "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
                np.testing.assert_array_equal(getattr(other, tag), getattr(dataset, tag))
        sampling_plan, framed = TargetFramer(seg, coords, plan, source_image=source).plane_geometry(plan, item)
        hu = sitk.GetArrayFromImage(PlaneStackResampler().sample_slice(source, sampling_plan, framed))[0]
        for point, expected_hu, ring in probes:
            xy = np.rint((point - origin) @ output_axes / spacing).astype(int)
            inside = bool(np.all(xy >= 0) and np.all(xy < out_size))
            assert inside == (percent == 50 or (percent == 70 and ring == 0))
            if inside:
                assert hu[xy[1], xy[0]] == pytest.approx(expected_hu, abs=1e-3)
                assert dataset.pixel_array[xy[1], xy[0]] > 65000
        recorded.append((center, spacing, out_size))
    for other, factor in zip(recorded[1:], (1 / .7, 2.)):
        np.testing.assert_allclose(other[0], recorded[0][0], atol=1e-5)
        np.testing.assert_allclose(other[1], recorded[0][1] * factor, atol=1e-8)
        np.testing.assert_array_equal(other[2], recorded[0][2])
    np.testing.assert_array_equal(sitk.GetArrayFromImage(source), original_pixels)
