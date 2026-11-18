"""Pair-local CT endplates may recover a failed envelope, never shared geometry."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import LabelTarget
from src.planning.spine_common import resolve_spine_interspace_segments
from src.stages.interspace_geometry import (
    _collision_recovery_groups,
    prism_collision_recovery_allowed,
    validate_spine_interspace_geometry,
)
from src.stages.stage_localization import LocalizationOperations


def _case(*, central_ct_supported=True, invalid=True, direction=None, spacing=(1., 1., 1.)):
    seg = np.zeros((145, 64, 60), np.uint8)
    raw = np.full(seg.shape, -30., np.float32)
    names = [f"vertebrae_L{i}" for i in range(1, 5)]
    targets, bodies, coords = {}, {}, {}
    for value, name, low in zip(range(1, 5), names, (110, 80, 50, 20)):
        seg[low:low + 20, 12:46, 12:45] = value
        raw[low:low + 20, 12:46, 12:45] = 500.
        indices = np.argwhere(seg == value)
        bodies[name] = indices
        targets[name] = LabelTarget(name, [value])
        coords[name] = {
            "found": True,
            "centroid": dict(zip("xyz", indices.mean(axis=0)[::-1])),
            # Full labels can include posterior elements with overlapping Z;
            # their framing boxes are deliberately not interspace evidence.
            "bbox": {"x_min": 12, "x_max": 44, "y_min": 12, "y_max": 55,
                     "z_min": low - 12, "z_max": low + 32},
        }
    if not central_ct_supported:
        raw[70:80, 12:46, 12:45] = 500.
    image = sitk.GetImageFromArray(seg)
    image.SetSpacing(spacing)
    if direction is not None:
        image.SetDirection(tuple(np.asarray(direction).ravel()))
        image.SetOrigin((-25., 31., 90.))
    with patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        shared = LocalizationOperations._build_spine_body_geometry("腰椎", image, seg, targets)
    annotations = shared["filtered_annotation_centroids"]
    if invalid:
        for name, partner in ((names[1], names[2]), (names[2], names[1])):
            annotations[name]["prism_found"] = False
            annotations[name]["prism_voxel_count"] = 0
            annotations[name]["prism_failure_reasons"] = [{
                "reason": "projected_separator_unavailable", "partner_label": partner,
            }]
    regions = {name: SimpleNamespace(indices_zyx=points) for name, points in bodies.items()}
    pairs = [(f"L{i}_L{i + 1}", names[i - 1], names[i]) for i in range(1, 4)]
    result = {"coords": coords, "spine_geometry": shared,
              "image": {"size_xyz": image.GetSize(), "spacing_xyz": image.GetSpacing(),
                        "origin_xyz": image.GetOrigin(), "direction": image.GetDirection()}}
    return image, seg, raw, targets, regions, bodies, shared, pairs, result


def _validate(case, loader=None, enabled=True):
    image, seg, raw, targets, regions, bodies, shared, pairs, _ = case
    return validate_spine_interspace_geometry(
        LocalizationOperations, image, seg, targets, regions, bodies,
        shared["filtered_annotation_centroids"], pairs,
        raw_array_loader=loader if loader is not None else lambda: raw,
        collision_recovery_enabled=enabled,
    )


def test_outer_groups_recover_independently_without_reviving_central_or_shared_geometry():
    case = _case(central_ct_supported=False)
    _, seg, raw, _, _, bodies, shared, pairs, result = case
    before = deepcopy(shared)
    original = seg.copy()
    for points in bodies.values():
        points.setflags(write=False)
    previous, _ = resolve_spine_interspace_segments(result, pairs)
    assert previous == []
    loader = Mock(return_value=raw)
    validated, local_pairs, diagnostic = _validate(case, loader)
    assert set(validated) == {"L1_L2", "L3_L4"}, diagnostic
    assert diagnostic["endplates"]["L2_L3"]["status"] == "kept_original"
    assert local_pairs == {}  # No changed body and no old nearest-point correction.
    loader.assert_called_once()
    assert shared == before
    np.testing.assert_array_equal(seg, original)
    assert not validated["L1_L2"]["vertebrae_L2"]["prism_found"]
    result["spine_geometry"] = {**shared, "interspace_validated_annotations": validated}
    with patch("src.planning.spine_common._translate_interspace_to_body_pair",
               side_effect=AssertionError("CT position must stay fixed")), \
         patch("src.planning.spine_common._refine_interspace_symmetry_geometry",
               side_effect=AssertionError("CT orientation must stay fixed")):
        segments, warnings = resolve_spine_interspace_segments(result, pairs)
    assert [segment.group_name for segment in segments] == ["L1_L2", "L3_L4"]
    assert any("L2_L3" in warning and warning.startswith("跳过") for warning in warnings)
    for segment, expected_midpoint in zip(segments, (104.5, 44.5)):
        assert (segment.target_start.z + segment.target_end.z) / 2 == pytest.approx(expected_midpoint)
        assert segment.frame.normal.z == pytest.approx(-1.)
        assert not segment.recenter_target_in_plane
    validated["L1_L2"]["vertebrae_L2"]["centroid_voxel"]["x"] += 100
    assert shared == before


@pytest.mark.parametrize("kind", ("missing", "flat", "misaligned"))
def test_ct_unavailable_or_unverified_does_not_recover(kind):
    case = _case()
    if kind == "misaligned":
        def load():
            raise ValueError("原图与分割空间不一致")
    else:
        def load():
            return None if kind == "missing" else np.zeros_like(case[2])
    before = deepcopy(case[6])
    validated, local_pairs, _ = _validate(case, load)
    assert validated == local_pairs == {}
    assert case[6] == before


@pytest.mark.parametrize("sequence", (
    None, {}, {"status": "skipped"}, {"status": "repaired"},
    {"status": "unchanged"},
    {"status": "unchanged", "mixed_body_labels": {"status": "skipped"}},
))
def test_sequence_uncertainty_disables_recovery(sequence):
    assert not prism_collision_recovery_allowed("腰椎", sequence)
    case = _case()
    loader = Mock(side_effect=AssertionError("Sequence uncertainty must not load CT"))
    assert _validate(case, loader, enabled=prism_collision_recovery_allowed("腰椎", sequence))[0] == {}
    loader.assert_not_called()


def test_only_unchanged_lumbar_sequence_enables_new_path():
    sequence = {"status": "unchanged", "mixed_body_labels": {"status": "unchanged"}}
    assert prism_collision_recovery_allowed("腰椎", sequence)
    for part in ("颈椎", "胸椎", "胸腰椎", "全脊柱"):
        assert not prism_collision_recovery_allowed(part, sequence)


def test_normal_geometry_does_not_load_ct_even_when_recovery_is_enabled():
    case = _case(invalid=False)
    loader = Mock(side_effect=AssertionError("Normal geometry must not load CT"))
    assert _validate(case, loader)[0] == {}
    loader.assert_not_called()


def _use_body_boxes(case, names):
    """让指定主体的旧 bbox 回退已有正间隙，不改变 CT 或恢复候选。"""

    for name in names:
        minimum, maximum = case[5][name].min(axis=0), case[5][name].max(axis=0)
        case[8]["coords"][name]["bbox"] = {
            f"{axis}_{side}": int(point[index])
            for index, axis in enumerate("zyx")
            for side, point in (("min", minimum), ("max", maximum))
        }


def test_valid_bbox_fallback_is_preserved_even_with_valid_collision_endplates():
    case = _case(central_ct_supported=False)
    _use_body_boxes(case, case[5])
    result, pairs = case[8], case[7]
    previous, previous_warnings = resolve_spine_interspace_segments(result, pairs)
    assert len(previous) == 3
    assert all(segment.frame is None and segment.recenter_target_in_plane for segment in previous)
    validated, _, _ = _validate(case)
    assert set(validated) == {"L1_L2", "L3_L4"}
    result["spine_geometry"] = {**case[6], "interspace_validated_annotations": validated}
    before = deepcopy(result)
    with patch("src.planning.spine_common._resolve_collision_endplate_geometry",
               side_effect=AssertionError("An existing interval must never use new geometry")):
        actual, warnings = resolve_spine_interspace_segments(result, pairs)
    assert [asdict(segment) for segment in actual] == [asdict(segment) for segment in previous]
    assert warnings == previous_warnings
    assert result == before


def test_collision_recovery_only_fills_missing_groups_among_existing_bbox_intervals():
    from src.planning.spine_common import _resolve_collision_endplate_geometry

    case = _case(central_ct_supported=False)
    _use_body_boxes(case, ("vertebrae_L1", "vertebrae_L2"))
    result, pairs = case[8], case[7]
    previous, previous_warnings = resolve_spine_interspace_segments(result, pairs)
    previous_by_group = {segment.group_name: asdict(segment) for segment in previous}
    assert "L1_L2" in previous_by_group and "L3_L4" not in previous_by_group
    validated, _, _ = _validate(case)
    result["spine_geometry"] = {**case[6], "interspace_validated_annotations": validated}
    with patch("src.planning.spine_common._resolve_collision_endplate_geometry",
               wraps=_resolve_collision_endplate_geometry) as recover:
        actual, warnings = resolve_spine_interspace_segments(result, pairs)
    actual_by_group = {segment.group_name: asdict(segment) for segment in actual}
    for group, original in previous_by_group.items():
        assert actual_by_group[group] == original
    assert set(actual_by_group) == {*previous_by_group, "L3_L4"}
    assert [call.args[1] for call in recover.call_args_list] == ["L3_L4"]
    assert not any(warning.startswith("跳过") and "L3_L4" in warning for warning in warnings)
    assert [warning for warning in warnings if "L1_L2" in warning] == [
        warning for warning in previous_warnings if "L1_L2" in warning
    ]


def test_normal_complete_intervals_never_enter_collision_geometry():
    case = _case(invalid=False)
    previous, previous_warnings = resolve_spine_interspace_segments(case[8], case[7])
    assert len(previous) == 3
    with patch("src.planning.spine_common._resolve_collision_endplate_geometry",
               side_effect=AssertionError("Normal intervals must stay on the original path")):
        actual, warnings = resolve_spine_interspace_segments(case[8], case[7])
    assert [asdict(segment) for segment in actual] == [asdict(segment) for segment in previous]
    assert warnings == previous_warnings


@pytest.mark.parametrize("reason", (None, "actual_body_voxel_overlap", "unverified_prism_overlap"))
def test_unknown_or_actual_body_collision_cannot_use_recovery(reason):
    case = _case()
    annotations = case[6]["filtered_annotation_centroids"]
    for name in ("vertebrae_L2", "vertebrae_L3"):
        annotations[name]["prism_failure_reasons"] = [] if reason is None else [{"reason": reason}]
    assert _collision_recovery_groups(annotations, case[7]) == set()
    loader = Mock(side_effect=AssertionError("Unverified failure must not load CT"))
    assert _validate(case, loader)[0] == {}


def test_recovery_annotations_are_bound_to_group_and_do_not_fall_back_to_invalid_quad():
    case = _case()
    validated, _, _ = _validate(case)
    result, pairs = case[8], case[7]
    # Keep the validated provenance but make its independent LR support invalid.
    # The old quad and bbox must not be allowed to rescue this candidate.
    altered = deepcopy(validated)
    altered["L1_L2"]["vertebrae_L2"]["sagittal_body_endplates"][0]["ct_profile_row_range_mm"] = [1., 1.]
    result["spine_geometry"] = {**case[6], "interspace_validated_annotations": altered}
    segments, warnings = resolve_spine_interspace_segments(result, pairs[:1])
    assert segments == []
    assert any("独立 CT 端板几何无效" in warning for warning in warnings)
    # A successful group cannot be published under another group's key.
    wrong = {"L1_L2": validated["L2_L3"]}
    result["spine_geometry"] = {**case[6], "interspace_validated_annotations": wrong}
    assert resolve_spine_interspace_segments(result, pairs[:1])[0] == []


def test_separator_records_failure_cause_without_changing_existing_decision():
    # Deep LR tilt defeats both existing projection separators, while body voxels
    # remain disjoint. The metadata must describe this, not revive the prisms.
    left, right = tuple(np.asarray([
        (z + (12 if x < 10 else 0), y, x)
        for x in range(20) for z in range(start, start + 10) for y in range(10)
    ]) for start in (0, 12))
    names = ("vertebrae_L3", "vertebrae_L4")
    combined = np.concatenate((left, right))
    minimum = combined.min(axis=0)
    shape = combined.max(axis=0) - minimum + 1
    regions = {name: (minimum, np.ones(tuple(shape), bool)) for name in names}
    evidence = {}
    actual = LocalizationOperations._resolve_non_overlapping_prism_regions(
        regions, dict(zip(names, (left, right))), diagnostics=evidence,
    )
    assert all(value is None for value in actual.values())
    assert all(item[0]["reason"] == "projected_separator_unavailable" for item in evidence.values())
    collided = np.concatenate((right, left[:1]))
    evidence.clear()
    actual = LocalizationOperations._resolve_non_overlapping_prism_regions(
        regions, dict(zip(names, (left, collided))), diagnostics=evidence,
    )
    assert all(value is None for value in actual.values())
    assert all(item[0]["reason"] == "actual_body_voxel_overlap" for item in evidence.values())


@pytest.mark.parametrize("direction", (
    np.array([[1., 0., 0.], [0., np.cos(.3), -np.sin(.3)], [0., np.sin(.3), np.cos(.3)]]),
    np.diag((-1., 1., 1.)), np.diag((1., -1., -1.)),
))
def test_recovery_retains_physical_endplates_in_oblique_and_reflected_source(direction):
    case = _case(direction=direction, spacing=(.7, 1.2, 1.5))
    validated, _, diagnostic = _validate(case)
    assert len(validated) == 3, diagnostic
    result = case[8]
    result["spine_geometry"] = {**case[6], "interspace_validated_annotations": validated}
    segments, _ = resolve_spine_interspace_segments(result, case[7])
    assert len(segments) == 3
    for segment, expected_z in zip(segments, (104.5, 74.5, 44.5)):
        start = np.array([getattr(segment.target_start, axis) for axis in "xyz"])
        end = np.array([getattr(segment.target_end, axis) for axis in "xyz"])
        midpoint = case[0].TransformPhysicalPointToContinuousIndex(tuple((start + end) * .5))
        assert midpoint[2] == pytest.approx(expected_z)
        normal = np.array([getattr(segment.frame.normal, axis) for axis in "xyz"])
        assert normal @ direction[:, 2] == pytest.approx(-1.)


@pytest.mark.parametrize("mismatch", (None, "shape", "spacing", "origin", "direction"))
def test_stage_checks_actual_raw_geometry_before_collision_recovery(tmp_path, monkeypatch, mismatch):
    from src.core.context import ProcessingContext
    from src.models import PlaneConfig, Request
    from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
    from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
    from src.stages.stage_localization import _LOCALIZATION_WORKSPACE_KEY

    image, seg, raw, targets, _, _, _, pairs, _ = _case()
    stage = AnatomicalAnalysisStage()
    seg_path, raw_path = str(tmp_path / "seg.nii.gz"), str(tmp_path / "raw.nii.gz")
    raw_image = sitk.GetImageFromArray(raw if mismatch != "shape" else raw[:-1])
    raw_image.SetSpacing(image.GetSpacing())
    raw_image.SetOrigin(image.GetOrigin())
    raw_image.SetDirection(image.GetDirection())
    if mismatch == "spacing":
        raw_image.SetSpacing((1.1, 1., 1.))
    elif mismatch == "origin":
        raw_image.SetOrigin((1., 0., 0.))
    elif mismatch == "direction":
        raw_image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
    sitk.WriteImage(raw_image, raw_path)
    context = ProcessingContext(Request(
        dicom_dir=str(tmp_path), output_dir=str(tmp_path / "out"), slice_part=SlicePartEnum.腰椎,
        plane_configs=[PlaneConfig(plane=PlaneTypeEnum.横状位, slice_type=SliceType.椎间隙,
                                  slice_thickness=5., slice_spacing=5., window_presets=["软组织窗"])],
    ))
    labels = list(targets)
    pose = {"method": "spine", "labels": ["L1", "L2", "L3", "L4"],
            "roles": {"start": "L1", "end": "L4"}}
    cut = {"regions": {"default": {
        "type": "whole_range", "target_labels": labels,
        "bounds": {
            "upper": {"labels": labels, "edge": "upper", "reduce": "max", "offset_mm": 0.0},
            "lower": {"labels": labels, "edge": "lower", "reduce": "min", "offset_mm": 0.0},
        },
    }}}
    context.set_result("localization", {
        "pose_config": pose, "resolved_pose_config": stage._resolve_pose_aliases(pose),
        "seg_for_reconstruction_path": seg_path, "raw_for_reconstruction_path": raw_path,
        "resolved_cut_config": cut, "resolved_cut_configs": {
            "whole_range": cut, "interspace": {"intervals": [
                {"region": group, "start_labels": [upper], "end_labels": [lower]}
                for group, upper, lower in pairs]},
        }, "localization_result_path": "",
        "spine_label_sequence": {"status": "unchanged", "mixed_body_labels": {"status": "unchanged"}},
    })
    target_mask = stage._build_target_mask(seg, targets)
    regions = stage._build_label_voxel_regions(seg, targets)
    context.metadata[_LOCALIZATION_WORKSPACE_KEY] = {
        "seg_path": seg_path, "image": image, "seg_array": seg, "label_targets": targets,
        "label_regions": regions, "label_coordinates": stage._extract_label_coordinates(seg, targets, label_regions=regions),
        "target_mask": target_mask, "correction_scope_bbox": stage._build_target_bbox(target_mask),
    }
    original = stage._build_spine_body_prism_regions

    def mark_collision(*args, **kwargs):
        bodies, prisms = original(*args, **kwargs)
        for name, partner in (("vertebrae_L2", "vertebrae_L3"), ("vertebrae_L3", "vertebrae_L2")):
            prisms[name] = None
            kwargs["prism_diagnostics"][name] = [{
                "reason": "projected_separator_unavailable", "partner_label": partner,
            }]
        return bodies, prisms

    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    with patch.object(AnatomicalAnalysisStage, "_build_spine_body_prism_regions", side_effect=mark_collision), \
         patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        stage.execute(context)
    geometry = context.get_result("anatomical_analysis")["spine_geometry"]
    assert not geometry["filtered_annotation_centroids"]["vertebrae_L2"]["prism_found"]
    if mismatch is None:
        assert set(geometry["interspace_validated_annotations"]) == {group for group, _, _ in pairs}
    else:
        assert "interspace_validated_annotations" not in geometry
        endplates = geometry["interspace_validation"]["endplates"]
        assert all(value["reason"] == "ct_unavailable" for value in endplates.values())
        assert any("空间不一致" in value.get("raw_error", "") for value in endplates.values())
