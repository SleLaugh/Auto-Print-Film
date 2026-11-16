"""CPR 规划复用不得跨 build 留存分割或改变终端标签覆盖。"""

from copy import deepcopy
import weakref

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import PlaneConfig, Point3D, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning import spine_cpr
from tests import test_stage6_slice_planning as fixtures


def _planning_case(tmp_path, plane):
    labels = ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3"]
    localization = fixtures.Stage6SlicePlanningTest._spine_localization_result(
        ("L1", "L2", "L3"), "腰椎",
    )
    array = np.zeros((40, 120, 120), dtype=np.uint8)
    annotations = {}
    for label_id, name in enumerate(labels, 1):
        center = localization["coords"][name]["centroid"]
        z = int(center["z"])
        array[z - 1:z + 2, 48:53, 58:63] = label_id
        annotations[name] = {
            "found": True,
            "centroid_physical": dict(center),
            "local_superior_inferior_face_centroids": {
                "min": {**center, "z": z + 0.5},
                "max": {**center, "z": z - 0.5},
            },
        }
    path = tmp_path / "seg.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
    localization["seg_for_reconstruction_path"] = str(path)
    localization["spine_geometry"] = {"filtered_annotation_centroids": annotations}
    localization["spine_cpr_config"]["full_label"] = True
    config = PlaneConfig(plane, 1.0, 3.0, ["骨窗"], slice_type=SliceType.CPR)
    request = Request(
        "/tmp/dicom", str(tmp_path / "output"),
        slice_part=SlicePartEnum.腰椎, plane_configs=[config],
    )
    return request, config, localization, labels, array


def _build(case):
    plans, warnings = spine_cpr.build_spine_sagittal_cpr_plans(*case[:4])
    return [plan.to_dict() for plan in plans], warnings


@pytest.mark.parametrize("plane", [PlaneTypeEnum.冠状位, PlaneTypeEnum.矢状位])
def test_build_reuses_one_read_and_keeps_uncached_plan_and_input(tmp_path, monkeypatch, plane):
    case = _planning_case(tmp_path, plane)
    original_localization = deepcopy(case[2])
    image_refs = []
    extraction_labels = []
    read_image = sitk.ReadImage
    extract_points = spine_cpr._raw_label_physical_points

    def read(*args, **kwargs):
        image = read_image(*args, **kwargs)
        image_refs.append(weakref.ref(image))
        return image

    def extract(image, label_ids):
        extraction_labels.append(frozenset(label_ids))
        return extract_points(image, label_ids)

    monkeypatch.setattr(sitk, "ReadImage", read)
    monkeypatch.setattr(spine_cpr, "_raw_label_physical_points", extract)
    with monkeypatch.context() as uncached:
        uncached.setattr(spine_cpr, "_CprSegmentationCache", lambda: None)
        expected = _build(case)
    assert len(image_refs) == 3
    image_refs.clear()
    extraction_labels.clear()

    assert _build(case) == expected
    assert len(image_refs) == 1
    assert extraction_labels == [frozenset({1}), frozenset({3}), frozenset({1, 2, 3})]
    assert all(reference() is None for reference in image_refs)
    assert case[2] == original_localization


def test_new_build_reads_replaced_segmentation_at_same_path(tmp_path, monkeypatch):
    case = _planning_case(tmp_path, PlaneTypeEnum.冠状位)
    before = _build(case)
    case[4][17:20, 48:53, 58:63] = 1
    sitk.WriteImage(sitk.GetImageFromArray(case[4]), case[2]["seg_for_reconstruction_path"])
    after = _build(case)
    with monkeypatch.context() as uncached:
        uncached.setattr(spine_cpr, "_CprSegmentationCache", lambda: None)
        expected = _build(case)
    assert after == expected
    assert after != before


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_failed_read_is_retried_and_changed_path_does_not_reuse_image(monkeypatch, error_type):
    first = sitk.Image([2, 3, 4], sitk.sitkUInt8)
    second = sitk.Image([4, 3, 2], sitk.sitkUInt8)
    calls = []

    def read(path):
        calls.append(path)
        if len(calls) == 1:
            raise error_type("temporary read failure")
        return first if path == "first.nii.gz" else second

    monkeypatch.setattr(sitk, "ReadImage", read)
    cache = spine_cpr._CprSegmentationCache()
    localization = {"seg_for_reconstruction_path": "first.nii.gz"}
    assert spine_cpr._read_original_segmentation(localization, segmentation_cache=cache) is None
    assert spine_cpr._read_original_segmentation(localization, segmentation_cache=cache) is first
    assert spine_cpr._read_original_segmentation(localization, segmentation_cache=cache) is first
    localization["seg_for_reconstruction_path"] = "second.nii.gz"
    assert spine_cpr._read_original_segmentation(localization, segmentation_cache=cache) is second
    assert calls == ["first.nii.gz", "first.nii.gz", "second.nii.gz"]


@pytest.mark.parametrize("plane", [PlaneTypeEnum.冠状位, PlaneTypeEnum.矢状位])
@pytest.mark.parametrize("all_reads_fail", [False, True])
def test_build_read_failures_keep_uncached_plans_and_warnings(
    tmp_path, monkeypatch, plane, all_reads_fail,
):
    case = _planning_case(tmp_path, plane)
    read_image = sitk.ReadImage
    calls = []

    def read(path):
        calls.append(path)
        if all_reads_fail or len(calls) == 1:
            raise RuntimeError("unavailable segmentation")
        return read_image(path)

    monkeypatch.setattr(sitk, "ReadImage", read)
    with monkeypatch.context() as uncached:
        uncached.setattr(spine_cpr, "_CprSegmentationCache", lambda: None)
        expected = _build(case)
    assert len(calls) == 3
    calls.clear()

    actual = _build(case)
    assert actual == expected
    assert actual[0], "读取失败时仍应保留原有的规划回退结果"
    assert len(calls) == (3 if all_reads_fail else 2)


@pytest.mark.parametrize("edge", ["same_label_ids", "empty_terminal", "endpoint_conflict"])
def test_terminal_aliases_missing_voxels_and_conflicts_keep_independent_helper_result(
    monkeypatch, edge,
):
    array = np.zeros((30, 16, 16), dtype=np.uint8)
    array[20:24, 6:10, 6:10] = 1
    if edge == "endpoint_conflict":
        # 不同标签都更靠近首端；原策略仍把末标签交给末端切线处理。
        array[24:28, 6:10, 10:14] = 3
    image = sitk.GetImageFromArray(array)
    curve = [Point3D(8.0, 8.0, z) for z in (21.0, 12.0, 5.0)]
    end_ids = {1} if edge == "same_label_ids" else {3}
    first_index = spine_cpr._endpoint_index_nearest_label(curve, image, {1})
    last_index = spine_cpr._endpoint_index_nearest_label(curve, image, end_ids)
    assert first_index == 0
    assert last_index == (None if edge == "empty_terminal" else 0)

    expected = list(curve)
    expected[0] = spine_cpr._extend_endpoint_to_label_projection(
        curve[0], curve[1], image, {1},
    )
    if edge != "empty_terminal":
        expected[-1] = spine_cpr._extend_endpoint_to_label_projection(
            curve[-1], curve[-2], image, end_ids,
        )
    assert expected[0].z > curve[0].z
    assert expected[-1] == curve[-1]

    extracted_ids = []
    extract_points = spine_cpr._raw_label_physical_points

    def extract(segmentation, label_ids):
        extracted_ids.append(frozenset(label_ids))
        return extract_points(segmentation, label_ids)

    monkeypatch.setattr(sitk, "ReadImage", lambda _path: image)
    monkeypatch.setattr(spine_cpr, "_raw_label_physical_points", extract)
    localization = {
        "seg_for_reconstruction_path": "seg.nii.gz",
        "coords": {"first": {"label_ids": [1]}, "last": {"label_ids": list(end_ids)}},
    }
    assert spine_cpr._extend_curve_endpoints_to_terminal_raw_labels(
        localization, ["first", "last"], curve,
    ) == expected
    assert extracted_ids == (
        [frozenset({1})] if edge == "same_label_ids" else [frozenset({1}), frozenset({3})]
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_terminal_points_match_independent_reads_in_oblique_reflected_grid(
    monkeypatch, reverse,
):
    array = np.zeros((30, 16, 16), dtype=np.uint8)
    array[20:24, 6:10, 6:10] = 1
    array[3:7, 6:10, 6:10] = 3
    image = sitk.GetImageFromArray(array)
    angle = np.deg2rad(23.0)
    image.SetSpacing((0.7, 1.2, 2.0))
    image.SetOrigin((-2.0, 3.0, 5.0))
    image.SetDirection((np.cos(angle), -np.sin(angle), 0.0,
                        np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, -1.0))
    curve = [Point3D(*image.TransformIndexToPhysicalPoint((8, 8, z))) for z in (21, 12, 5)]
    if reverse:
        curve.reverse()
    projection = Point3D(*np.asarray(image.GetDirection()).reshape(3, 3)[:, 1])
    expected = list(curve)
    for ids in ({1}, {3}):
        index = spine_cpr._endpoint_index_nearest_label(curve, image, ids)
        expected[index] = spine_cpr._extend_endpoint_to_label_projection(
            expected[index], spine_cpr._adjacent_endpoint_point(expected, index),
            image, ids, projection,
        )

    monkeypatch.setattr(sitk, "ReadImage", lambda _path: image)
    localization = {
        "seg_for_reconstruction_path": "seg.nii.gz",
        "coords": {"first": {"label_ids": [1]}, "last": {"label_ids": [3]}},
    }
    assert spine_cpr._extend_curve_endpoints_to_terminal_raw_labels(
        localization, ["first", "last"], curve, projection_axis=projection,
    ) == expected
