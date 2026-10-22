"""Recovered body endplates must not inherit a protruding envelope corner."""

import copy

import numpy as np
import pytest
import SimpleITK as sitk

from src.planning.spine_common import _resolve_interspace_prism_geometry
from src.stages.stage_localization import LocalizationStage


def _case(direction):
    image = sitk.Image(45, 45, 45, sitk.sitkUInt8)
    image.SetSpacing((.8, .7, 1.2))
    image.SetOrigin((-17., 23., -51.))
    image.SetDirection(tuple(np.asarray(direction).ravel()))

    def physical(y, z):
        return dict(zip("xyz", image.TransformContinuousIndexToPhysicalPoint((20., float(y), float(z)))))

    def annotation(points):
        return {
            "prism_found": True,
            "sagittal_body_quadrilateral": {
                "vertices": [physical(y, z) for y, z in points],
                "left_right_extent_mm": {"min": -4., "max": 4.},
            },
        }

    upper = annotation(((10, 26), (30, 18), (30, 34), (10, 34)))
    upper["body_extraction"] = {"method": "density_core_recovery"}
    lower = annotation(((10, 6), (30, 6), (30, 16), (10, 16)))
    upper["sagittal_body_endplates"] = [
        {"endpoints": [physical(14, z), physical(26, z)],
         "normal": dict(zip("xyz", np.asarray(direction)[:, 2] * sign))}
        for z, sign in ((24., -1.), (32., 1.))
    ]
    result = {
        "image": {"size_xyz": list(image.GetSize()), "spacing_xyz": list(image.GetSpacing()),
                  "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection())},
        "spine_geometry": {"filtered_annotation_centroids": {"upper": upper, "lower": lower}},
    }
    return image, result


angle = np.deg2rad(23.)
ROTATED = np.array(((1., 0., 0.), (0., np.cos(angle), -np.sin(angle)), (0., np.sin(angle), np.cos(angle))))


@pytest.mark.parametrize("direction", (np.eye(3), ROTATED, np.diag((-1., 1., -1.))))
def test_supported_face_uses_true_boundary_and_preserves_image_space(direction):
    image, result = _case(direction)
    start, end, frame = _resolve_interspace_prism_geometry(result, "upper", "lower")
    for point, expected_z in ((start, 24.), (end, 16.)):
        index = image.TransformPhysicalPointToContinuousIndex((point.x, point.y, point.z))
        np.testing.assert_allclose(index, (20., 20., expected_z), atol=1e-6)
    np.testing.assert_allclose((frame.normal.x, frame.normal.y, frame.normal.z), -direction[:, 2], atol=1e-6)


@pytest.mark.parametrize("invalid", ([], [{}, {}], [{"endpoints": []}, {}]))
def test_invalid_supported_faces_keep_existing_quad_geometry(invalid):
    _, result = _case(np.eye(3))
    annotation = result["spine_geometry"]["filtered_annotation_centroids"]["upper"]
    annotation.pop("sagittal_body_endplates")
    baseline = _resolve_interspace_prism_geometry(result, "upper", "lower")
    changed = copy.deepcopy(result)
    changed["spine_geometry"]["filtered_annotation_centroids"]["upper"]["sagittal_body_endplates"] = invalid
    assert _resolve_interspace_prism_geometry(changed, "upper", "lower") == baseline


def test_nonperpendicular_endplate_normal_is_rejected():
    _, result = _case(np.eye(3))
    annotation = result["spine_geometry"]["filtered_annotation_centroids"]["upper"]
    faces = annotation.pop("sagittal_body_endplates")
    baseline = _resolve_interspace_prism_geometry(result, "upper", "lower")
    faces[0]["normal"] = {"x": 0., "y": 1., "z": 1.}
    annotation["sagittal_body_endplates"] = faces
    assert _resolve_interspace_prism_geometry(result, "upper", "lower") == baseline


def test_supported_faces_do_not_change_a_pair_without_recovered_body():
    _, result = _case(np.eye(3))
    annotation = result["spine_geometry"]["filtered_annotation_centroids"]["upper"]
    faces = annotation.pop("sagittal_body_endplates")
    annotation.pop("body_extraction")
    baseline = _resolve_interspace_prism_geometry(result, "upper", "lower")
    annotation["sagittal_body_endplates"] = faces
    assert _resolve_interspace_prism_geometry(result, "upper", "lower") == baseline


def test_stage5_endplate_uses_source_boundary_in_global_oblique_space():
    source = np.array([(z + 20, y + 30, x + 40)
                       for y in range(10, 30) for z in range(y, y + 12)
                       for x in range(12, 32)])
    # Recovery has a flat artificial limit; the original tilted boundary remains
    # the direction evidence. All indices and core bounds include a crop offset.
    recovered = source[(source[:, 0] >= 42) & (source[:, 0] <= 49)]
    image = sitk.Image(100, 100, 100, sitk.sitkUInt8)
    image.SetSpacing((1., 1., 2.))
    image.SetOrigin((-41., 17., -27.))
    image.SetDirection(tuple(ROTATED.ravel()))
    raw = np.zeros((100, 100, 100), np.float32)
    raw[tuple(source.T)] = 600.
    faces = LocalizationStage._estimate_supported_body_endplates(
        recovered, image,
        {"method": "density_core_recovery",
         "core_bbox_zyx": {"min": [40, 44, 58], "max": [48, 55, 65]}},
        source_indices_zyx=source,
        raw_array=raw,
    )
    assert faces is not None
    expected = ROTATED @ (np.array((0., 2., -1.)) / np.sqrt(5))
    np.testing.assert_allclose(list(faces[0]["normal"].values()), expected, atol=1e-6)
    for point in faces[0]["endpoints"]:
        x, y, z = image.TransformPhysicalPointToContinuousIndex(tuple(point[a] for a in "xyz"))
        assert abs(z - (y - 10)) < 1e-6
