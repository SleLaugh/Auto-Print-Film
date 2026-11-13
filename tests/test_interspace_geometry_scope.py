"""Interspace validation must not change shared body, prism or CPR geometry."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import SimpleITK as sitk

from src.models import LabelTarget
from src.stages.interspace_geometry import validate_spine_interspace_geometry
from src.stages.stage_localization import LocalizationStage


def _case():
    segmentation = np.zeros((65, 55, 65), dtype=np.uint8)
    segmentation[39:54, 12:34, 17:44] = 1
    segmentation[10:25, 12:34, 17:44] = 2
    labels = ("vertebrae_L2", "vertebrae_L3")
    targets = {name: LabelTarget(name, [value]) for value, name in enumerate(labels, 1)}
    bodies = {name: np.argwhere(segmentation == value)
              for value, name in enumerate(labels, 1)}
    regions = {name: SimpleNamespace(indices_zyx=indices)
               for name, indices in bodies.items()}
    image = sitk.GetImageFromArray(segmentation)
    image.SetSpacing((0.8, 0.8, 1.0))
    image.SetOrigin((-24.0, 13.0, -97.0))
    with patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        shared = LocalizationStage._build_spine_body_geometry(
            "腰椎", image, segmentation, targets,
        )
    pairs = [("L2_L3", *labels)]
    return image, segmentation, targets, regions, bodies, shared, pairs


def _build(image, segmentation, targets, raw, **kwargs):
    with patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        return LocalizationStage._build_spine_body_geometry(
            "腰椎", image, segmentation, targets, raw_array_loader=raw, **kwargs,
        )


def test_shared_geometry_builder_keeps_validation_off_by_default():
    image, segmentation, targets, _, _, shared, _ = _case()
    original = segmentation.copy()
    raw = Mock(side_effect=AssertionError("Normal geometry must not load CT"))
    with patch("src.stages.interspace_geometry.validate_spine_interspace_geometry") as validate:
        actual = _build(image, segmentation, targets, raw)
    validate.assert_not_called()
    raw.assert_not_called()
    assert actual == shared
    np.testing.assert_array_equal(segmentation, original)


def test_enabling_validation_keeps_normal_shared_centroids_prisms_and_pairs_identical():
    image, segmentation, targets, _, _, shared, _ = _case()
    original = segmentation.copy()
    raw = Mock(side_effect=AssertionError("No conflict should not load CT"))
    actual = _build(image, segmentation, targets, raw, validate_interspaces=True)
    raw.assert_not_called()
    assert "interspace_validated_annotations" not in actual
    assert "interspace_validated_body_pairs" not in actual
    assert actual["interspace_validation"]["body_candidate_count"] == 0
    for key, value in shared.items():
        assert actual[key] == value, key
    np.testing.assert_array_equal(segmentation, original)


def test_label_region_indices_are_reused_and_input_body_arrays_are_read_only():
    image, segmentation, targets, regions, bodies, shared, pairs = _case()
    annotations = shared["filtered_annotation_centroids"]
    previous = deepcopy(annotations)
    for indices in bodies.values():
        indices.setflags(write=False)
    raw = Mock(side_effect=AssertionError("Normal pair must not load CT"))
    with patch("src.stages.interspace_geometry.np.isin", side_effect=AssertionError("Full volume scan")):
        validated, validated_pairs, diagnostic = validate_spine_interspace_geometry(
            LocalizationStage, image, segmentation, targets, regions,
            bodies, annotations, pairs, raw_array_loader=raw,
        )
    assert validated == {} and validated_pairs == {}
    assert diagnostic["body_candidate_count"] == 0
    assert annotations == previous
    raw.assert_not_called()


def _one_candidate(label_mask, body_mask, spacing, *, vertebra_name, diagnostics):
    if vertebra_name != "vertebrae_L2":
        return None
    # A narrower interior is enough to exercise replacement annotation creation;
    # this test isolates publishing and CT rejection from candidate selection.
    replacement = label_mask.copy()
    replacement[:, :, :3] = False
    diagnostics.update(replaced=True, reason="test_candidate")
    return replacement


def test_candidate_rejected_by_ct_does_not_publish_any_body_or_pair_change():
    image, segmentation, targets, regions, bodies, shared, pairs = _case()
    annotations = shared["filtered_annotation_centroids"]
    previous = deepcopy(shared)
    original_indices = {name: indices.copy() for name, indices in bodies.items()}
    original_segmentation = segmentation.copy()
    raw = Mock(return_value=np.zeros(segmentation.shape, dtype=np.int16))
    with patch("src.stages.interspace_body.validate_interspace_body", side_effect=_one_candidate), \
            patch("src.stages.interspace_endplates._fit_facing_face", return_value=None) as fit:
        validated, validated_pairs, diagnostic = validate_spine_interspace_geometry(
            LocalizationStage, image, segmentation, targets, regions,
            bodies, annotations, pairs, raw_array_loader=raw,
        )
    raw.assert_called_once()
    assert fit.call_count == 2
    assert diagnostic["body_candidate_count"] == 1
    assert diagnostic["endplates"]["L2_L3"]["status"] == "kept_original"
    assert validated == {} and validated_pairs == {}
    assert shared == previous
    np.testing.assert_array_equal(segmentation, original_segmentation)
    for name, indices in bodies.items():
        np.testing.assert_array_equal(indices, original_indices[name])


def test_accepted_candidate_publishes_separate_annotations_and_local_pair_only():
    image, segmentation, targets, regions, bodies, shared, pairs = _case()
    annotations = shared["filtered_annotation_centroids"]
    previous = deepcopy(shared)
    original_indices = {name: indices.copy() for name, indices in bodies.items()}

    def accept(image, candidate_bodies, source, candidate_annotations, groups, **kwargs):
        assert candidate_bodies["vertebrae_L2"] is not bodies["vertebrae_L2"]
        assert candidate_bodies["vertebrae_L3"] is bodies["vertebrae_L3"]
        assert candidate_annotations["vertebrae_L2"] is not annotations["vertebrae_L2"]
        return {"L2_L3": deepcopy(candidate_annotations)}, {"L2_L3": {"status": "validated"}}

    with patch("src.stages.interspace_body.validate_interspace_body", side_effect=_one_candidate), \
            patch("src.stages.interspace_endplates.validate_interspace_endplates", side_effect=accept):
        validated, validated_pairs, diagnostic = validate_spine_interspace_geometry(
            LocalizationStage, image, segmentation, targets, regions,
            bodies, annotations, pairs,
        )
    assert set(validated) == {"L2_L3"} and set(validated_pairs) == {"L2_L3"}
    assert diagnostic["body_candidate_count"] == 1
    assert shared == previous
    # Mutating a published candidate afterwards must not affect global geometry.
    validated["L2_L3"]["vertebrae_L2"]["centroid_voxel"]["x"] += 100
    assert shared == previous
    for name, indices in bodies.items():
        np.testing.assert_array_equal(indices, original_indices[name])


def test_stage_geometry_raw_callback_is_loaded_once_and_cached():
    image, segmentation, targets, _, _, shared, _ = _case()
    raw_array = np.zeros(segmentation.shape, dtype=np.int16)
    raw = Mock(return_value=raw_array)

    def validate(*args, raw_array_loader):
        assert raw_array_loader() is raw_array
        assert raw_array_loader() is raw_array
        return {}, {}, {"body_candidate_count": 0}

    with patch("src.stages.interspace_geometry.validate_spine_interspace_geometry", side_effect=validate):
        actual = _build(image, segmentation, targets, raw, validate_interspaces=True)
    raw.assert_called_once()
    for key, value in shared.items():
        assert actual[key] == value


def test_stage_geometry_wrong_raw_shape_is_rejected_without_publishing_changes():
    image, segmentation, targets, _, _, shared, _ = _case()
    raw = Mock(return_value=np.zeros((2, 2, 2), dtype=np.int16))

    def validate(*args, raw_array_loader):
        try:
            raw_array_loader()
        except ValueError as error:
            assert "形状不一致" in str(error)
            return {}, {}, {"reason": "ct_unavailable"}
        raise AssertionError("Shape mismatch must fail before fitting")

    with patch("src.stages.interspace_geometry.validate_spine_interspace_geometry", side_effect=validate):
        actual = _build(image, segmentation, targets, raw, validate_interspaces=True)
    assert "interspace_validated_annotations" not in actual
    for key, value in shared.items():
        assert actual[key] == value
