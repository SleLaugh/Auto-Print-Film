"""逐椎体姿态只允许在同一工作分割快照内复用。"""

from copy import deepcopy

import numpy as np
import pytest
import SimpleITK as sitk

import src.anatomy.spine_pose as spine_pose
from src.anatomy.spine_pose import (
    SpineLabelSymmetryCache,
    estimate_spine_axial_symmetry,
    estimate_spine_label_symmetry,
)
from src.models import LabelTarget
from src.stages.stage_localization import LocalizationStage


def _record_fits(monkeypatch, status="unchanged"):
    calls = []

    def fit(points, axes, *, max_points):
        calls.append((points.copy(), max_points))
        return axes[:, 0].copy(), points.mean(axis=0), {
            "status": status, "accepted": status != "rejected", "nested": [1],
        }

    monkeypatch.setattr(spine_pose, "estimate_reflection_symmetry_plane", fit)
    return calls


@pytest.mark.parametrize("status", ["accepted", "unchanged", "rejected"])
def test_pose_cache_preserves_all_statuses_and_isolates_mutable_results(monkeypatch, status):
    calls = _record_fits(monkeypatch, status)
    image = sitk.Image((20, 20, 20), sitk.sitkUInt8)
    indices = np.indices((8, 8, 8)).reshape(3, -1).T
    cache = SpineLabelSymmetryCache(np.zeros((20, 20, 20), dtype=np.uint8))
    first = estimate_spine_label_symmetry(indices, image, pose_cache=cache)
    expected = deepcopy(first)
    first["symmetry_diagnostics"]["nested"].append(99)
    second = estimate_spine_label_symmetry(indices, image, pose_cache=cache)
    assert second == expected
    second["symmetry_plane_center"]["x"] = -999.0
    assert estimate_spine_label_symmetry(indices, image, pose_cache=cache) == expected
    assert len(calls) == 1
    assert expected["reliable"] is (status != "rejected")


@pytest.mark.parametrize("change", ["indices", "spacing", "origin", "direction", "max_points"])
def test_pose_cache_checks_index_identity_geometry_and_sampling_parameters(monkeypatch, change):
    calls = _record_fits(monkeypatch)
    image = sitk.Image((20, 20, 20), sitk.sitkUInt8)
    indices = np.indices((8, 8, 8)).reshape(3, -1).T
    cache = SpineLabelSymmetryCache(np.zeros((20, 20, 20), dtype=np.uint8))
    estimate_spine_label_symmetry(indices, image, pose_cache=cache)
    if change == "indices":
        indices = indices.copy()
    elif change == "spacing":
        image.SetSpacing((0.5, 1.0, 2.0))
    elif change == "origin":
        image.SetOrigin((1.0, 2.0, 3.0))
    elif change == "direction":
        image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    max_points = 256 if change == "max_points" else 3000
    changed = estimate_spine_label_symmetry(
        indices, image, max_points=max_points, pose_cache=cache,
    )
    assert len(calls) == 2
    assert estimate_spine_label_symmetry(
        indices, image, max_points=max_points, pose_cache=cache,
    ) == changed
    assert len(calls) == 2
    assert estimate_spine_label_symmetry(indices, image, max_points=max_points) == changed


@pytest.mark.parametrize("status", ["unchanged", "rejected"])
def test_aggregate_retains_cached_labels_when_no_rotation_is_selected(monkeypatch, status):
    calls = _record_fits(monkeypatch, status)
    image = sitk.Image((20, 20, 20), sitk.sitkUInt8)
    indices = np.indices((8, 8, 8)).reshape(3, -1).T
    cache = SpineLabelSymmetryCache(np.zeros((20, 20, 20), dtype=np.uint8))
    aggregate = estimate_spine_axial_symmetry({"L1": indices}, image, pose_cache=cache)
    assert not aggregate["should_correct"]
    if aggregate["label_angles"]:
        aggregate["label_angles"][0]["symmetry_diagnostics"]["nested"].append(99)
    pose = estimate_spine_label_symmetry(indices, image, pose_cache=cache)
    assert pose["symmetry_diagnostics"]["nested"] == [1]
    assert len(calls) == 1


@pytest.mark.parametrize("replace_segmentation", [False, True])
def test_spatial_candidate_and_body_geometry_share_only_matching_snapshot(monkeypatch, replace_segmentation):
    calls = _record_fits(monkeypatch)
    seg_array = np.zeros((28, 24, 24), dtype=np.uint8)
    seg_array[3:11, 6:16, 7:17] = 7
    seg_array[16:24, 6:16, 7:17] = 9
    image = sitk.GetImageFromArray(seg_array)
    targets = {
        "vertebrae_L1": LabelTarget("vertebrae_L1", [7]),
        "vertebrae_S1": LabelTarget("vertebrae_S1", [9]),
    }
    stage = LocalizationStage()
    regions = stage._build_label_voxel_regions(seg_array, targets)
    cache = SpineLabelSymmetryCache(seg_array)
    stage._build_xy_correction_candidate(
        seg_array, targets, slice_part="腰椎", label_regions=regions,
        image=image, pose_cache=cache,
    )
    assert len(calls) == 2
    if replace_segmentation:
        seg_array = seg_array.copy()
    geometry = stage._build_spine_body_geometry(
        "腰椎", image, seg_array, targets, label_regions=regions, pose_cache=cache,
    )
    assert len(calls) == (4 if replace_segmentation else 2)
    assert geometry == stage._build_spine_body_geometry(
        "腰椎", image, seg_array, targets, label_regions=regions,
    )
