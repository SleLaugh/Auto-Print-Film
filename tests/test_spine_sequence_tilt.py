"""Anterior-body recovery must preserve ownership through LR side tilt."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.stages.spine_label_sequence import (
    _reassociate_anterior_foreground,
    repair_lumbar_body_sequence,
)
from src.stages.stage_localization import LocalizationStage
from tests.test_spine_sequence_integration import _sequence_fixture


@pytest.mark.parametrize("tilt_sign", (-1, 1))
def test_twenty_two_degree_side_tilt_keeps_each_physical_body_owner(tilt_sign):
    seg, raw, image, labels, targets, masks, _ = _sequence_fixture(shifted=True)
    padding = ((32, 32), (0, 0), (0, 0))

    def tilt(array):
        result = np.pad(array, padding)
        for x in range(result.shape[2]):
            # dz/dx in physical units = .25 * 1.2 / .75 = .4 (~21.8 degrees).
            result[:, :, x] = np.roll(
                result[:, :, x], round((x-43) * .25 * tilt_sign), axis=0,
            )
        return result

    seg, raw = tilt(seg), tilt(raw)
    masks = {label: tilt(mask) for label, mask in masks.items()}
    tilted_image = sitk.GetImageFromArray(seg)
    tilted_image.SetSpacing(image.GetSpacing())
    tilted_image.SetOrigin(image.GetOrigin())
    tilted_image.SetDirection(image.GetDirection())

    def extract(array, target):
        if target.label_name.endswith("_S1"):
            return LocalizationStage._s1_body_candidate_mask(array, target)
        return LocalizationStage._vertebral_body_candidate_mask(
            array, target, spacing_zyx=tuple(reversed(tilted_image.GetSpacing())),
        )

    original = seg.copy()
    repaired, diagnostic = repair_lumbar_body_sequence(
        seg, raw, tilted_image, labels, targets, extract,
    )

    assert repaired is not None, diagnostic
    assert diagnostic["status"] == "repaired"
    np.testing.assert_array_equal(seg, original)
    np.testing.assert_array_equal(repaired != 0, original != 0)
    for source, target in ((6, 5), (7, 6), (8, 7), (9, 8)):
        assert np.all(repaired[masks[source]] == target), (source, np.unique(repaired[masks[source]], return_counts=True))
    anchor_foreground = np.isin(original, (1, 2, 3, 4))
    np.testing.assert_array_equal(repaired[anchor_foreground], original[anchor_foreground])
    np.testing.assert_array_equal(repaired[:, 58:, :], original[:, 58:, :])
    np.testing.assert_array_equal(repaired[original == 10], original[original == 10])


def test_overlapping_sagittal_projections_use_lr_distance_for_unowned_cap():
    array = np.zeros((24, 24, 36), dtype=np.uint8)
    anchor = np.array([(z, y, x) for z in range(10, 13)
                       for y in range(10, 13) for x in range(5, 8)])
    shifted = anchor + (0, 0, 15)
    array[tuple(anchor.T)] = 4
    array[tuple(shifted.T)] = 6
    # Both points have the same sagittal projection. LR proximity distinguishes
    # the cap on the anchor from the cap on the recovered body's opposite side.
    array[11, 11, 8] = 5
    array[11, 11, 23] = 5
    array[11, 11, 32] = 6  # remote original fragment remains unchanged

    def record(indices, source_name, target_name, ids):
        return {"indices": indices, "minimum": indices.min(axis=0),
                "maximum": indices.max(axis=0), "name": source_name,
                "canonical_name": target_name, "ids": ids}

    records = [record(anchor, "vertebrae_L4", "vertebrae_L4", [4]),
               record(shifted, "vertebrae_L2", "vertebrae_L3", [6])]
    repaired, _ = _reassociate_anterior_foreground(
        array, records, np.ones(3),
        {"vertebrae_L4": [4], "vertebrae_L3": [5]}, [5, 6],
    )

    assert np.all(repaired[tuple(shifted.T)] == 5)
    assert np.all(repaired[tuple(anchor.T)] == 4)
    assert repaired[11, 11, 8] == 4
    assert repaired[11, 11, 23] == 5
    assert repaired[11, 11, 32] == 6
    np.testing.assert_array_equal(repaired != 0, array != 0)


def _cap_component_case(cap_points):
    array = np.zeros((80, 42, 50), dtype=np.uint8)
    anchor = np.array([(z, y, x) for z in range(10, 21)
                       for y in range(10, 21) for x in range(10, 31)])
    shifted = anchor + (30, 0, 0)
    array[tuple(anchor.T)] = 4
    array[tuple(shifted.T)] = 6
    array[tuple(cap_points.T)] = 5
    records = [
        {"indices": points, "minimum": points.min(axis=0), "maximum": points.max(axis=0),
         "name": source, "canonical_name": target, "ids": ids}
        for points, source, target, ids in (
            (anchor, "vertebrae_L4", "vertebrae_L4", [4]),
            (shifted, "vertebrae_L2", "vertebrae_L3", [6]),
        )
    ]
    diagnostics = {}
    repaired, _ = _reassociate_anterior_foreground(
        array, records, np.ones(3), {"vertebrae_L4": [4], "vertebrae_L3": [5]},
        [5, 6], diagnostics=diagnostics,
    )
    np.testing.assert_array_equal(repaired != 0, array != 0)
    np.testing.assert_array_equal(repaired[tuple(anchor.T)], array[tuple(anchor.T)])
    assert np.all(repaired[tuple(shifted.T)] == 5)
    return array, repaired, diagnostics["cap_components"]


def test_confirmed_connected_cap_keeps_outer_corner_beyond_seed_distance():
    cap = np.array([(z, y, x) for z in (21, 22)
                    for y in range(10, 21) for x in range(10, 31)] + [(23, 10, 10)])
    remote = np.array([(65, 33, x) for x in range(40, 45)])
    _, repaired, components = _cap_component_case(np.concatenate((cap, remote)))
    assert np.all(repaired[tuple(cap.T)] == 4)
    # This corner has no seed within 2 mm. Its connected cap has a reliable owner.
    assert repaired[23, 10, 10] == 4
    assert np.all(repaired[tuple(remote.T)] == 5)
    assert {component["status"] for component in components} == {"reassociated", "preserved"}
    assert next(component for component in components if component["status"] == "preserved")["seed_voxel_count"] == 0


def test_component_connected_to_posterior_foreground_is_not_partially_reassigned():
    component = np.array([(21, y, x) for y in range(10, 36) for x in range(10, 31)])
    _, repaired, diagnostics = _cap_component_case(component)
    assert np.all(repaired[tuple(component.T)] == 5)
    assert diagnostics[0]["reason"] == "outside_anterior_body_corridor"


def test_component_with_two_neighbor_seed_owners_is_preserved():
    component = np.array([(z, 15, 15) for z in range(21, 40)])
    _, repaired, diagnostics = _cap_component_case(component)
    assert np.all(repaired[tuple(component.T)] == 5)
    assert diagnostics[0]["reason"] == "ambiguous_3d_attachment"
    assert diagnostics[0]["seed_owner_fraction"] == .5


def test_component_with_fewer_than_ten_percent_nearby_seeds_is_preserved():
    component = np.array([(21, 20, 30)]
                         + [(21, y, x) for y in range(21, 36) for x in range(30, 46)])
    _, repaired, diagnostics = _cap_component_case(component)
    assert np.all(repaired[tuple(component.T)] == 5)
    assert diagnostics[0]["seed_voxel_count"] < len(component)*.1
    assert diagnostics[0]["reason"] == "insufficient_3d_seeds"


def _anisotropic_attachment_case(anchor, neighbor, cap):
    array = np.zeros((32, 40, 32), dtype=np.uint8)
    array[tuple(anchor.T)] = 4
    array[tuple(neighbor.T)] = 6
    array[tuple(cap.T)] = 5
    records = [
        {"indices": points, "minimum": points.min(axis=0), "maximum": points.max(axis=0),
         "name": source, "canonical_name": target, "ids": ids}
        for points, source, target, ids in (
            (anchor, "vertebrae_L4", "vertebrae_L4", [4]),
            (neighbor, "vertebrae_L2", "vertebrae_L3", [6]),
        )
    ]
    diagnostics = {}
    repaired, _ = _reassociate_anterior_foreground(
        array, records, np.array((3., .5, .5)),
        {"vertebrae_L4": [4], "vertebrae_L3": [5]}, [5, 6], diagnostics=diagnostics,
    )
    return repaired, diagnostics["cap_components"]


def test_two_mm_lr_gap_is_not_a_grid_neighbor_with_thick_si_sampling():
    anchor, neighbor = np.array([[10, 10, 10]]), np.array([[20, 20, 20]])
    cap = np.array([[10, 10, 14]])  # LR four voxels / 2 mm; SI spacing is 3 mm.
    repaired, diagnostics = _anisotropic_attachment_case(anchor, neighbor, cap)
    assert repaired[10, 10, 14] == 5
    assert diagnostics[0]["seed_voxel_count"] == 1
    assert diagnostics[0]["reason"] == "ambiguous_3d_attachment"


def test_dominant_owner_cannot_borrow_another_owners_grid_contact():
    anchor = np.array([(10, y, 10) for y in range(10, 31)])
    neighbor = np.array([[10, 31, 12]])
    cap = np.array([(10, y, 12) for y in range(10, 31)])
    repaired, diagnostics = _anisotropic_attachment_case(anchor, neighbor, cap)
    assert np.all(repaired[tuple(cap.T)] == 5)
    assert diagnostics[0]["seed_owner_fraction"] >= .9
    assert diagnostics[0]["candidate_target_name"] == "vertebrae_L4"
    assert diagnostics[0]["reason"] == "ambiguous_3d_attachment"


def test_projected_ghost_cap_cannot_trigger_public_sequence_repair():
    seg, raw, image, labels, targets, _, cap = _sequence_fixture(shifted=True)
    seg[cap] = 4
    moved = np.roll(cap, 4, axis=0)
    seg[moved] = 5
    raw[moved] = 390

    def tilt(array):
        result = np.pad(array, ((32, 32), (0, 0), (0, 0)))
        for x in range(result.shape[2]):
            result[:, :, x] = np.roll(result[:, :, x], round((x-43)*.25), axis=0)
        return result

    seg, raw = tilt(seg), tilt(raw)
    tilted_image = sitk.GetImageFromArray(seg)
    tilted_image.SetSpacing(image.GetSpacing())

    def extract(array, target):
        if target.label_name.endswith("_S1"):
            return LocalizationStage._s1_body_candidate_mask(array, target)
        return LocalizationStage._vertebral_body_candidate_mask(
            array, target, spacing_zyx=tuple(reversed(image.GetSpacing())),
        )

    original = seg.copy()
    result, diagnostic = repair_lumbar_body_sequence(
        seg, raw, tilted_image, labels, targets, extract,
    )
    assert result is None, diagnostic
    assert diagnostic["status"] == "skipped"
    assert diagnostic["slot_support"]["neighbor_projection_overlap_fraction"] == 1.
    assert diagnostic["ct_evidence"]["valley_count"] == 1
    assert all(component["status"] == "preserved" for component in diagnostic["cap_components"])
    np.testing.assert_array_equal(seg, original)
