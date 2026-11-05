"""C2 的补充端面只修复缺失间隙，不能进入其它区间或定位图的统计。"""

from copy import deepcopy

import numpy as np

from src.planning.spine_common import (
    regularize_cervical_interspace_segments,
    resolve_spine_interspace_segments,
    resolve_spine_localizer_reference,
)


def _case():
    coords, annotations = {}, {}
    for name, low, high in (("C2", 65, 75), ("C3", 45, 60),
                            ("C4", 25, 40), ("C5", 5, 20)):
        label = f"vertebrae_{name}"
        center = {"x": 20.0, "y": 17.5, "z": (low + high) / 2}
        bbox = {"x_min": 10, "x_max": 30, "y_min": 10, "y_max": 25,
                "z_min": low, "z_max": high}
        coords[label] = {"found": True, "centroid": center, "bbox": bbox}
        annotations[label] = {
            "found": True, "prism_found": True, "voxel_count": 5000,
            "body_bbox": bbox, "centroid_voxel": center, "centroid_physical": center,
            "sagittal_body_quadrilateral": {
                "center": center,
                "vertices": [{"x": 20.0, "y": y, "z": z}
                             for y, z in ((10, low), (25, low), (25, high), (10, high))],
                "left_right_extent_mm": {"min": -10.0, "max": 10.0},
            },
        }
    recovered = deepcopy(annotations["vertebrae_C2"])
    recovered["body_extraction"] = {"method": "c2_inferior_body_recovery"}
    recovered["sagittal_endplate_support"] = {
        "method": "c2_interspace_endplates", "required_endplate_side": "min",
    }
    recovered["sagittal_body_endplates"] = [{
        "endpoints": [{"x": 20.0, "y": y, "z": 65.0} for y in (10.0, 25.0)],
        "normal": {"x": 0.0, "y": 0.0, "z": -1.0},
        "source": "original_label", "ct_validation": "density_drop",
    }]
    partner = deepcopy(annotations["vertebrae_C3"])
    partner["sagittal_endplate_support"] = {
        "method": "c2_interspace_endplates", "required_endplate_side": "max",
    }
    partner["sagittal_body_endplates"] = [{
        "endpoints": [{"x": 20.0, "y": y, "z": 60.0} for y in (10.0, 25.0)],
        "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        "source": "original_label", "ct_validation": "density_drop",
    }]
    recovered["interspace_partner_annotations"] = {"vertebrae_C3": partner}
    annotations["vertebrae_C2"] = {"found": False, "prism_found": False, "voxel_count": 0}
    # 完整 C2 的附件与 C3 在头脚方向重叠；bbox 回退原本会拒绝此间隙。
    coords["vertebrae_C2"]["bbox"] = {**coords["vertebrae_C2"]["bbox"], "z_min": 50}
    result = {
        "image": {"size_xyz": [80, 80, 100], "spacing_xyz": [1.0, 1.0, 1.0],
                  "origin_xyz": [0.0, 0.0, 0.0],
                  "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        "coords": coords,
        "spine_geometry": {"filtered_annotation_centroids": annotations},
    }
    intervals = [(f"C{i}_C{i + 1}", f"vertebrae_C{i}", f"vertebrae_C{i + 1}")
                 for i in range(2, 5)]
    return result, recovered, intervals


def test_recovered_c2_adds_only_its_interval_without_mutating_shared_geometry():
    result, recovered, intervals = _case()
    baseline, _ = resolve_spine_interspace_segments(result, intervals)
    reference = resolve_spine_localizer_reference(result, list(result["coords"]))
    result["spine_geometry"]["interspace_recovered_annotations"] = {"vertebrae_C2": recovered}
    original = deepcopy(result)

    segments, warnings = resolve_spine_interspace_segments(result, intervals)

    assert [segment.group_name for segment in segments] == ["C2_C3", "C3_C4", "C4_C5"]
    assert segments[1:] == baseline
    assert not any(warning.startswith("跳过脊柱区间 ") for warning in warnings)
    assert resolve_spine_localizer_reference(result, list(result["coords"])) == reference
    assert regularize_cervical_interspace_segments(result, segments)[0][1:] == (
        regularize_cervical_interspace_segments(result, baseline)[0]
    )
    assert result == original


def test_existing_nonempty_c2_cannot_be_replaced_by_recovery_evidence():
    result, recovered, intervals = _case()
    result["spine_geometry"]["filtered_annotation_centroids"]["vertebrae_C2"] = recovered
    baseline = resolve_spine_interspace_segments(result, intervals)
    invalid = {**deepcopy(recovered), "sagittal_body_quadrilateral": {}}
    result["spine_geometry"]["interspace_recovered_annotations"] = {"vertebrae_C2": invalid}

    assert resolve_spine_interspace_segments(result, intervals) == baseline


def test_adjacent_natural_endplate_is_used_only_with_the_recovered_pair():
    result, recovered, intervals = _case()
    result["coords"]["vertebrae_C3"]["bbox"] = {
        **result["coords"]["vertebrae_C3"]["bbox"], "z_max": 64,
    }
    result["spine_geometry"]["interspace_recovered_annotations"] = {"vertebrae_C2": recovered}
    baseline, _ = resolve_spine_interspace_segments(result, intervals)
    partner = deepcopy(recovered["interspace_partner_annotations"]["vertebrae_C3"])
    normal = np.array((0.0, -4.0, 15.0)) / np.sqrt(241.0)
    partner["sagittal_body_endplates"] = [
        {"endpoints": [{"x": 20.0, "y": 10.0, "z": 60.0},
                       {"x": 20.0, "y": 25.0, "z": 64.0}],
         "normal": dict(zip("xyz", normal)),
         "source": "original_label", "ct_validation": "density_drop"},
    ]
    recovered["interspace_partner_annotations"] = {"vertebrae_C3": partner}
    original = deepcopy(result)

    actual, _ = resolve_spine_interspace_segments(result, intervals)

    assert actual[0].frame != baseline[0].frame
    assert actual[1:] == baseline[1:]
    assert result == original


def test_missing_ct_evidence_on_either_facing_endplate_keeps_the_original_skip():
    for missing in ("upper", "lower", "partner"):
        result, recovered, intervals = _case()
        baseline = resolve_spine_interspace_segments(result, intervals)
        if missing == "partner":
            recovered.pop("interspace_partner_annotations")
        else:
            annotation = recovered if missing == "upper" else (
                recovered["interspace_partner_annotations"]["vertebrae_C3"]
            )
            annotation["sagittal_body_endplates"][0].pop("ct_validation")
        result["spine_geometry"]["interspace_recovered_annotations"] = {"vertebrae_C2": recovered}

        assert resolve_spine_interspace_segments(result, intervals) == baseline
