"""New C2 pairs use only the natural, CT-supported faces toward the gap."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.models import LabelTarget, Point3D
from src.planning.spine_common import _supported_sagittal_endplate_edge
from src.planning.straight import StraightPlaneFrame
from src.stages.spine_body_core import fit_body_endplates_from_core
from src.stages.stage_localization import LocalizationStage


def test_single_requested_face_does_not_require_unrelated_boundary_inside_bone():
    source = np.zeros((80, 50, 60), dtype=bool)
    raw = np.zeros(source.shape, dtype=np.float32)
    for y in range(10, 40):
        top = 40 + y // 2
        source[20:top, y, 10:50] = True
        raw[10:top, y, 10:50] = 600.0
    indices = np.argwhere(source)
    bounds = {"min": [26, 14, 20], "max": [36, 35, 40]}
    original_source, original_raw = source.copy(), raw.copy()

    # The label's inferior face is artificial and still lies inside CT bone.
    assert fit_body_endplates_from_core(
        indices, (1, 1, 1), bounds, source_indices_zyx=indices, raw_array=raw,
    ) is None
    faces = fit_body_endplates_from_core(
        indices, (1, 1, 1), bounds, source_indices_zyx=indices, raw_array=raw,
        required_sides=("max",),
    )

    assert faces is not None and len(faces) == 1
    assert faces[0]["side"] == "max"
    np.testing.assert_allclose(
        faces[0]["normal_zyx"], np.array([1, -0.5, 0]) / np.sqrt(1.25), atol=0.02,
    )
    assert faces[0]["ct_validation"] == "density_drop"
    assert faces[0]["source"] == "original_label"
    assert fit_body_endplates_from_core(
        indices, (1, 1, 1), bounds, required_sides=(None,),
    ) is None
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(raw, original_raw)


def test_single_face_requires_explicit_c2_pair_support_marker():
    annotation = {"sagittal_body_endplates": [{
        "endpoints": [{"x": 0.0, "y": 10.0, "z": 20.0},
                      {"x": 0.0, "y": 30.0, "z": 20.0}],
        "normal": {"x": 0.0, "y": 0.0, "z": -1.0},
    }]}
    frame = StraightPlaneFrame(Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1))
    assert _supported_sagittal_endplate_edge(annotation, frame, (0, -1)) is None
    annotation["sagittal_endplate_support"] = {"method": "adjacent_recovered_body"}
    assert _supported_sagittal_endplate_edge(annotation, frame, (0, -1)) is None
    annotation["sagittal_endplate_support"] = {
        "method": "c2_interspace_endplates", "required_endplate_side": "min",
    }
    assert _supported_sagittal_endplate_edge(annotation, frame, (0, -1)) is not None


def test_stage5_attaches_natural_c3_face_only_to_recovered_c2_partner():
    array = np.zeros((120, 100, 200), dtype=np.uint8)
    array[64:84, 10:40, 70:130] = 25
    array[84:107, 18:33, 88:112] = 25
    array[70:76, 20:40, 45:155] = 25
    array[64:90, 60:85, 20:180] = 25
    array[75, 39:61, 99:101] = 25
    for y in range(10, 40):
        array[29:47 + y // 4, y, 70:130] = 24
    raw = np.where(array != 0, 600.0, 0.0).astype(np.float32)
    original = array.copy()
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))
    targets = {f"vertebrae_C{i}": LabelTarget(f"vertebrae_C{i}", [27 - i])
               for i in (2, 3)}

    def body(array, target, **kwargs):
        if target.label_name == "vertebrae_C3":
            result = array == 24
            # Crop the original tilted superior face to an artificial flat face.
            result[18:] = False
            return result
        return LocalizationStage._recover_c2_inferior_body(array == 25, **kwargs)

    with patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        with patch.object(LocalizationStage, "_vertebral_body_candidate_mask", side_effect=body):
            geometry = LocalizationStage._build_spine_body_geometry(
                "颈椎", image, array, targets, raw_array_loader=lambda: raw,
            )

    recovered = geometry["interspace_recovered_annotations"]["vertebrae_C2"]
    partner = recovered["interspace_partner_annotations"]["vertebrae_C3"]
    global_c3 = geometry["filtered_annotation_centroids"]["vertebrae_C3"]
    assert "sagittal_body_endplates" not in global_c3
    assert len(partner["sagittal_body_endplates"]) == 1
    face = partner["sagittal_body_endplates"][0]
    assert face["side"] == "max" and face["normal"]["y"] < -0.1
    assert face["ct_validation"] == "density_drop"
    unmodified = deepcopy(partner)
    unmodified.pop("sagittal_body_endplates")
    unmodified.pop("sagittal_endplate_support")
    assert unmodified == global_c3
    np.testing.assert_array_equal(array, original)
