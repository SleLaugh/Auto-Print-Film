"""椎间隙端板复核不改变普通主体，且不能把自然倾斜拉成水平。"""

from copy import deepcopy
from unittest import mock

import numpy as np
import pytest
import SimpleITK as sitk

from src.planning.spine_common import (
    _interspace_annotation_view,
    _resolve_interspace_prism_geometry,
    regularize_cervical_interspace_segments,
)
from src.stages.interspace_endplates import (
    _positive_facing_gap,
    validate_interspace_endplates,
)


def _xyz(point):
    return dict(zip("xyz", map(float, point)))


def _fixture(sidewall=True, slope=0.5, reflected=False):
    image = sitk.Image((56, 100, 140), sitk.sitkUInt8)
    if reflected:
        image.SetDirection((1., 0., 0., 0., -1., 0., 0., 0., -1.))
    names = ("vertebrae_L5", "vertebrae_S1")
    bounds = ((10, 45, 40, 61), (35, 70, 28, 46)) if sidewall else ((10, 45, 48, 68), (10, 45, 20, 40))
    bodies, annotations = {}, {}
    raw = np.full((140, 100, 56), -30, dtype=np.float32)
    for name, (y0, y1, z0, z1) in zip(names, bounds):
        indices = np.array([(z, y, x) for y in range(y0, y1 + 1)
                            for z in range(int(np.ceil(z0 + slope * (y - y0))), int(np.floor(z1 + slope * (y - y0))) + 1)
                            for x in range(12, 43)], dtype=np.int32)
        bodies[name] = indices
        raw[tuple(indices.T)] = 500
        vertices_voxel = [(27, y0, z0), (27, y1, z0 + slope * (y1 - y0)),
                          (27, y1, z1 + slope * (y1 - y0)), (27, y0, z1)]
        points = [image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, p))) for p in vertices_voxel]
        center = np.mean(points, axis=0)
        annotations[name] = {
            "found": True, "prism_found": True, "voxel_count": len(indices),
            "centroid_physical": _xyz(center),
            "sagittal_body_quadrilateral": {
                "center": _xyz(center), "vertices": [_xyz(p) for p in points],
                "left_right_extent_mm": {"min": -15., "max": 15.},
            },
        }
    result = {"image": {"size_xyz": image.GetSize(), "spacing_xyz": image.GetSpacing(),
                        "origin_xyz": image.GetOrigin(), "direction": image.GetDirection()},
              "spine_geometry": {"filtered_annotation_centroids": annotations}}
    return image, bodies, annotations, raw, result, [("L5_S1", *names)]


@pytest.mark.parametrize("reflected", (False, True))
def test_sidewall_conflict_uses_original_ct_endplates_and_preserves_source(reflected):
    image, bodies, annotations, raw, result, pairs = _fixture(reflected=reflected)
    original = deepcopy(annotations)
    replacements, diagnostics = validate_interspace_endplates(
        image, bodies, bodies, annotations, pairs, raw_array_loader=lambda: raw,
    )
    assert diagnostics["L5_S1"]["status"] == "validated", diagnostics
    assert annotations == original
    result["spine_geometry"]["interspace_validated_annotations"] = replacements
    view = _interspace_annotation_view(result, pairs[0][1:], "L5_S1")
    geometry = _resolve_interspace_prism_geometry(view, *pairs[0][1:])
    assert geometry is not None
    normal = np.array(list(vars(geometry[2].normal).values()))
    si = np.asarray(image.GetDirection()).reshape(3, 3)[:, 2]
    assert np.degrees(np.arccos(abs(normal @ si))) == pytest.approx(np.degrees(np.arctan(.5)), abs=2)
    # S1 只有面向 L5 的一面；下端面从未成为拟合前提。
    lower = replacements["L5_S1"]["vertebrae_S1"]
    assert len(lower["sagittal_body_endplates"]) == 1
    assert lower["sagittal_body_endplates"][0]["side"] == "max"
    assert lower["sagittal_body_endplates"][0]["ct_validation"] == "density_drop"
    assert result["spine_geometry"]["filtered_annotation_centroids"] == original


@pytest.mark.parametrize("slope", (0.0, 0.5, 1.2))
def test_normal_including_steep_endplates_never_load_ct_or_change_geometry(slope):
    image, bodies, annotations, _, result, pairs = _fixture(sidewall=False, slope=slope)
    loader = mock.Mock(side_effect=AssertionError("normal geometry must not load CT"))
    replacements, diagnostics = validate_interspace_endplates(
        image, bodies, bodies, annotations, pairs, raw_array_loader=loader,
    )
    assert replacements == {} and diagnostics == {}
    loader.assert_not_called()
    assert _interspace_annotation_view(result, pairs[0][1:], "L5_S1") is result


def test_ct_unavailable_and_nonbone_ct_keep_original_and_do_not_publish_candidate():
    image, bodies, annotations, raw, _, pairs = _fixture()
    annotations["vertebrae_L5"]["interspace_body_recheck"] = {"replaced": True}
    for loader in (None, lambda: np.zeros_like(raw)):
        replacements, diagnostics = validate_interspace_endplates(
            image, bodies, bodies, annotations, pairs, raw_array_loader=loader,
        )
        assert replacements == {}
        assert diagnostics["L5_S1"]["status"] == "kept_original"


def test_crossing_endplates_rejected_and_group_annotation_cannot_leak():
    image, bodies, annotations, raw, result, pairs = _fixture()
    replacements, _ = validate_interspace_endplates(
        image, bodies, bodies, annotations, pairs, raw_array_loader=lambda: raw,
    )
    faces = [replacements["L5_S1"][n]["sagittal_body_endplates"][0] for n in pairs[0][1:]]
    assert _positive_facing_gap(image, faces)
    crossed = deepcopy(faces)
    for point in crossed[1]["endpoints"]:
        point["z"] += 40
    assert not _positive_facing_gap(image, crossed)
    result["spine_geometry"]["interspace_validated_annotations"] = replacements
    assert _interspace_annotation_view(result, pairs[0][1:], "L4_L5") is result
    untrusted = deepcopy(replacements)
    untrusted["L5_S1"]["vertebrae_L5"]["sagittal_body_endplates"][0]["ct_validation"] = "unavailable"
    result["spine_geometry"]["interspace_validated_annotations"] = untrusted
    assert _interspace_annotation_view(result, pairs[0][1:], "L5_S1") is result


def test_ct_confirmed_direction_is_not_relimited_by_low_body_volume():
    from src.planning.straight import CutSegment

    image, bodies, annotations, raw, result, pairs = _fixture()
    replacements, _ = validate_interspace_endplates(
        image, bodies, bodies, annotations, pairs, raw_array_loader=lambda: raw,
    )
    result["spine_geometry"]["interspace_validated_annotations"] = replacements
    # 让旧体积保护必然进入；独立端板应优先于体积/中心连线的间接推断。
    annotations["vertebrae_C3"] = {"found": True, "voxel_count": 100000}
    annotations["vertebrae_L5"]["voxel_count"] = 1
    replacements["L5_S1"]["vertebrae_L5"]["voxel_count"] = 1
    replacements["L5_S1"]["vertebrae_L5"]["centroid_physical"] = _xyz((27, 0, 70))
    replacements["L5_S1"]["vertebrae_S1"]["centroid_physical"] = _xyz((27, 80, 70))
    local = _interspace_annotation_view(result, pairs[0][1:], "L5_S1")
    start, end, frame = _resolve_interspace_prism_geometry(local, *pairs[0][1:])
    segment = CutSegment("L5_S1", start, end, {}, pairs[0][1:], frame)
    actual, warnings = regularize_cervical_interspace_segments(result, [segment])
    assert actual == [segment]
    assert warnings == []


def test_local_body_pair_override_is_limited_to_the_validated_group():
    image, bodies, annotations, raw, result, pairs = _fixture()
    replacements, _ = validate_interspace_endplates(
        image, bodies, bodies, annotations, pairs, raw_array_loader=lambda: raw,
    )
    old_pair = {"found": True, "minimum_distance_mm": 3.0}
    new_pair = {"found": True, "minimum_distance_mm": 7.0}
    result["spine_geometry"].update(
        interspace_validated_annotations=replacements,
        interspace_body_pairs={"L5_S1": old_pair},
        interspace_validated_body_pairs={"L5_S1": new_pair},
    )
    local = _interspace_annotation_view(result, pairs[0][1:], "L5_S1")
    assert local["spine_geometry"]["interspace_body_pairs"]["L5_S1"] == new_pair
    assert result["spine_geometry"]["interspace_body_pairs"]["L5_S1"] == old_pair
    assert _interspace_annotation_view(result, pairs[0][1:], "other") is result
