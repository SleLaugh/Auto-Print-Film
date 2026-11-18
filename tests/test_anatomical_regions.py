"""验证区域边界在最终切片方向中的语义及现有默认范围的兼容性。"""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.anatomy.regions import _LabelProjector, resolve_anatomical_regions
from src.models import AnatomicalBounds, AnatomicalFrame, AnatomicalGeometryResult, AxisBounds, Point3D


def _frame(axes=None, bounds=((-10, 40), (-20, 50), (-30, 60))):
    axes = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
    return AnatomicalFrame(
        origin=Point3D(101, 202, 303),
        left_right_axis=Point3D(*axes[:, 0]),
        anterior_posterior_axis=Point3D(*axes[:, 1]),
        superior_inferior_axis=Point3D(*axes[:, 2]),
        bounds=AnatomicalBounds(*(AxisBounds(*pair) for pair in bounds)),
        method="existing_pose",
        confidence=0.75,
        applied=True,
        warnings=["保留已有诊断"],
    )


def _image(spacing=(1, 1, 1), direction=None, origin=(0, 0, 0)):
    image = sitk.Image([64, 64, 64], sitk.sitkUInt8)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(np.asarray(direction, dtype=float).ravel().tolist())
    return image


def _boundary(labels, edge, reduce=None, offset=0):
    result = {"labels": labels, "edge": edge, "offset_mm": offset}
    if reduce is not None:
        result["reduce"] = reduce
    return result


def _region(targets, upper=None, lower=None):
    return {
        "type": "whole_range",
        "target_labels": targets,
        "bounds": {
            "upper": upper or _boundary(targets, "upper", "max"),
            "lower": lower or _boundary(targets, "lower", "min"),
        },
    }


def _resolve(region, indices, *, pose_labels=None, geometry=None, image=None, method="spine"):
    return resolve_anatomical_regions(
        {"selected": region},
        geometry or AnatomicalGeometryResult(_frame()),
        image or _image(),
        indices,
        pose_labels or [*indices, "pose_support"],
        method,
    )["selected"]


def _bounds(result, axis="superior_inferior", frame="global_frame"):
    value = result[frame]["bounds"][axis]
    return value["minimum"], value["maximum"]


def _cube(z, y, x, size=3):
    return np.indices((size, size, size)).reshape(3, -1).T + np.array([z, y, x])


def test_boundary_edge_is_independent_from_upper_or_lower_region_side():
    indices = {
        "upper_marker": np.array([[10, 1, 1], [20, 2, 2]]),
        "lower_marker": np.array([[2, 1, 1], [5, 2, 2]]),
        "target": np.array([[1, 4, 3], [30, 6, 8]]),
    }
    result = _resolve(_region(
        ["target"],
        upper=_boundary(["upper_marker"], "lower"),
        lower=_boundary(["lower_marker"], "upper"),
    ), indices)
    assert _bounds(result) == (5, 10)
    assert _bounds(result, "left_right") == (3, 8)
    assert _bounds(result, "anterior_posterior") == (4, 6)
    assert result["constrained"] is True


@pytest.mark.parametrize("upper_reduce,lower_reduce,expected", [
    ("max", "min", (0, 23)),
    ("min", "max", (2, 13)),
])
def test_reduce_each_label_edge_before_applying_signed_millimeter_offset(
    upper_reduce, lower_reduce, expected,
):
    indices = {
        "left": np.array([[2, 1, 1], [10, 2, 2]]),
        "right": np.array([[4, 3, 3], [20, 4, 4]]),
    }
    labels = ["left", "right"]
    result = _resolve(_region(
        labels,
        upper=_boundary(labels, "upper", upper_reduce, 3),
        lower=_boundary(labels, "lower", lower_reduce, -2),
    ), indices)
    assert _bounds(result) == expected


def test_projection_respects_image_direction_origin_spacing_and_final_axes():
    c = np.sqrt(0.5)
    axes = np.array([[1, 0, 0], [0, c, -c], [0, c, c]])
    image = _image(
        spacing=(2, 3, 5),
        direction=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        origin=(10, 20, 30),
    )
    indices = {"target": np.array([[1, 2, 3], [4, 5, 6]])}
    result = _resolve(
        _region(["target"]), indices,
        image=image, geometry=AnatomicalGeometryResult(_frame(axes)),
    )
    np.testing.assert_allclose(_bounds(result, "left_right"), (-5, 4))
    np.testing.assert_allclose(_bounds(result, "anterior_posterior"), (61 * c, 82 * c))
    np.testing.assert_allclose(_bounds(result), (9 * c, 18 * c))
    assert result["global_frame"]["origin"] == {"x": 101, "y": 202, "z": 303}
    assert result["global_frame"]["warnings"] == ["保留已有诊断"]


def test_pose_support_does_not_enlarge_target_extent():
    indices = {
        "target": np.array([[2, 3, 4], [8, 9, 10]]),
        "support": np.array([[55, 56, 57]]),
    }
    result = _resolve(_region(["target"]), indices, pose_labels=["target", "support"])
    assert _bounds(result, "left_right") == (4, 10)
    assert _bounds(result, "anterior_posterior") == (3, 9)
    assert _bounds(result) == (2, 8)


def test_cervical_local_frame_reprojects_region_after_selecting_its_direction():
    local = _frame([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    geometry = AnatomicalGeometryResult(_frame(), {"axial_mpr": local})
    indices = {"target": np.array([[2, 10, 3], [8, 20, 7]])}
    result = _resolve(_region(["target"]), indices, geometry=geometry)
    assert _bounds(result) == (2, 8)
    local_bounds = result["local_frames"]["axial_mpr"]["bounds"]
    assert local_bounds["superior_inferior"] == {"minimum": 10, "maximum": 20}
    assert local_bounds["anterior_posterior"] == {"minimum": -8, "maximum": -2}


@pytest.mark.parametrize("method", ["brain", "lung", "hip", "spine"])
def test_complete_default_region_reuses_all_existing_bounds_without_voxel_scan(method):
    frame = _frame(bounds=((-10.25, 40.75), (-20.5, 50.125), (-30.625, 60.875)))
    local = replace(frame, bounds=AnatomicalBounds(
        AxisBounds(-11, 42), AxisBounds(-22, 54), AxisBounds(-33, 66),
    ))
    geometry = AnatomicalGeometryResult(frame, {"axial_mpr": local})
    # 包含任意远端点，证明默认边界未被原始噪点或体素中心重新计算。
    indices = {"left": np.array([[1, 1, 1], [63, 63, 63]]), "right": np.array([[2, 2, 2]])}
    with patch.object(_LabelProjector, "bounds", side_effect=AssertionError("不应重复投影")):
        result = _resolve(
            _region(["left", "right"]), indices, pose_labels=["left", "right"],
            geometry=geometry, method=method,
        )
    assert result["global_frame"] == frame.to_dict()
    assert result["local_frames"]["axial_mpr"] == local.to_dict()
    assert result["constrained"] is False


def test_offset_on_full_union_preserves_filtered_bounds_without_new_projection():
    indices = {"lung": np.array([[1, 1, 1], [63, 63, 63]])}
    region = _region(["lung"], upper=_boundary(["lung"], "upper", offset=30))
    with patch.object(_LabelProjector, "bounds", side_effect=AssertionError("不应重复投影")):
        result = _resolve(region, indices, pose_labels=["lung"], method="lung")
    assert _bounds(result) == (-30, 90)
    assert _bounds(result, "left_right") == (-10, 40)
    assert result["constrained"] is True


def test_brain_custom_region_preserves_voxel_outer_support():
    indices = {"target": np.array([[2, 3, 4], [6, 7, 8]])}
    result = _resolve(
        _region(["target"]), indices, method="brain", image=_image(spacing=(2, 3, 5)),
    )
    assert _bounds(result, "left_right") == (7, 17)
    assert _bounds(result, "anterior_posterior") == (7.5, 22.5)
    assert _bounds(result) == (7.5, 32.5)


def test_hip_subset_keeps_largest_component_and_drops_distant_noise():
    indices = {
        "left": np.concatenate([_cube(5, 5, 5), [[50, 50, 50]]]),
        "right": _cube(10, 5, 20),
    }
    result = _resolve(
        _region(["left"]), indices, pose_labels=["left", "right"], method="hip",
    )
    assert _bounds(result) == (5, 7)
    assert _bounds(result, "left_right") == (5, 7)
    assert _bounds(result, "anterior_posterior") == (5, 7)


def test_lung_subset_cleans_union_once_then_preserves_per_label_support():
    from src.anatomy.lung import retain_dominant_lung_components

    indices = {
        "left": np.concatenate([_cube(5, 5, 5), [[50, 50, 50]]]),
        "right": _cube(10, 5, 20),
    }
    with patch("src.anatomy.lung.retain_dominant_lung_components", wraps=retain_dominant_lung_components) as clean:
        results = resolve_anatomical_regions(
            {"left": _region(["left"]), "right": _region(["right"])},
            AnatomicalGeometryResult(_frame()), _image(), indices, ["left", "right"], "lung",
        )
    assert clean.call_count == 1
    assert _bounds(results["left"]) == (5, 7)
    assert _bounds(results["right"]) == (10, 12)
    assert _bounds(results["right"], "left_right") == (20, 22)


def test_hip_multilabel_role_cleans_combined_side_before_resolving_label_subset():
    from src.anatomy.hip import _retain_largest_component

    # left_b 的 8 点与 left_a 相连，远端 12 点本身更大，但不是左侧最大主体。
    connected_b = _cube(5, 5, 8, size=2)
    distant_b = np.indices((2, 2, 3)).reshape(3, -1).T + np.array([40, 40, 40])
    indices = {
        "left_a": _cube(5, 5, 5),
        "left_b": np.concatenate([connected_b, distant_b]),
        "right": _cube(10, 5, 20),
    }
    with patch("src.anatomy.hip._retain_largest_component", wraps=_retain_largest_component) as clean:
        results = resolve_anatomical_regions(
            {"b": _region(["left_b"]), "a": _region(["left_a"])},
            AnatomicalGeometryResult(_frame()), _image(), indices, list(indices), "hip",
            pose_roles={"left": ["left_a", "left_b"], "right": ["right"]},
        )
    assert clean.call_count == 1
    assert _bounds(results["b"]) == (5, 6)
    assert _bounds(results["b"], "left_right") == (8, 9)
    assert _bounds(results["a"]) == (5, 7)


def test_projections_are_cached_across_named_regions_and_equal_frame_directions(monkeypatch):
    indices = {"target": np.array([[2, 3, 4], [8, 9, 10]])}
    calls = []
    original = _LabelProjector._validate_chunk

    def capture(self, chunk, label):
        calls.append(label)
        original(self, chunk, label)

    monkeypatch.setattr(_LabelProjector, "_validate_chunk", capture)
    resolve_anatomical_regions(
        {"first": _region(["target"]), "second": _region(["target"])},
        AnatomicalGeometryResult(_frame(), {"same_axes": _frame()}),
        _image(), indices, ["target", "support"], "spine",
    )
    assert calls == ["target"]


@pytest.mark.parametrize("indices,error", [
    ({}, "缺失"),
    ({"target": np.empty((0, 3))}, "为空"),
    ({"target": np.array([1, 2, 3])}, "N×3"),
    ({"target": np.array([[1.5, 2, 3]])}, "有限整数"),
    ({"target": np.array([[64, 2, 3]])}, "有限整数"),
    ({"target": np.array([[np.nan, 2, 3]])}, "有限整数"),
])
def test_missing_empty_or_invalid_label_indices_fail_explicitly(indices, error):
    with pytest.raises(ValueError, match=error):
        _resolve(_region(["target"]), indices, pose_labels=["target", "support"])


@pytest.mark.parametrize("upper,lower,error", [
    (_boundary(["target"], "upper", offset=-6), _boundary(["target"], "lower"), "范围为空"),
    (_boundary(["target"], "lower"), _boundary(["target"], "upper"), "反转"),
    (_boundary(["target"], "middle"), _boundary(["target"], "lower"), "edge"),
    (_boundary(["target", "other"], "upper"), _boundary(["target"], "lower"), "reduce"),
    (_boundary(["target"], "upper", offset=float("inf")), _boundary(["target"], "lower"), "有限"),
    (_boundary(["target"], "upper", offset=True), _boundary(["target"], "lower"), "有限"),
])
def test_invalid_boundary_configuration_fails_explicitly(upper, lower, error):
    indices = {"target": np.array([[2, 3, 4], [8, 9, 10]])}
    with pytest.raises(ValueError, match=error):
        _resolve(_region(["target"], upper, lower), indices)
