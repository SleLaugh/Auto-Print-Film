"""主体分割缺损不能把前后侧壁当成椎间端面。"""

import copy

import numpy as np
import pytest

from src.planning.spine_common import resolve_spine_interspace_segments


def _xyz(values):
    return dict(zip("xyz", map(float, values)))


def _fragmented_lumbar_case(direction):
    # 来自实际失败病例的矢状轮廓：L3 只剩后方小块，与 L2/L4 的
    # 前后范围没有交集。保留物理几何，移除患者信息及原始影像路径。
    bodies = (
        ("L1", 139202, [(-115.915, -445.022), (-90.441, -451.320),
                         (-84.143, -423.679), (-115.170, -422.096)]),
        ("L2", 156952, [(-126.567, -479.800), (-96.418, -479.800),
                         (-93.496, -454.152), (-125.837, -452.593)]),
        ("L3", 5817, [(-85.951, -488.800), (-74.714, -488.800),
                       (-74.714, -479.558), (-85.556, -474.953)]),
        ("L4", 169106, [(-131.438, -513.712), (-100.007, -510.713),
                         (-100.007, -486.284), (-131.832, -487.786)]),
        ("L5", 168627, [(-125.379, -547.407), (-98.640, -538.142),
                         (-98.640, -511.825), (-127.051, -518.059)]),
        ("S1", 233791, [(-100.520, -573.932), (-74.296, -543.746),
                         (-85.833, -527.366), (-119.795, -551.587)]),
    )
    coords, annotations = {}, {}
    for short_name, count, polygon in bodies:
        label = f"vertebrae_{short_name}"
        vertices = np.array([(0.0, y, z) for y, z in polygon])
        center = vertices.mean(axis=0)
        coords[label] = {
            "found": True,
            "centroid": _xyz(center),
            "bbox": {
                "x_min": -20, "x_max": 20,
                "y_min": float(vertices[:, 1].min()),
                "y_max": float(vertices[:, 1].max()),
                "z_min": float(vertices[:, 2].min()),
                "z_max": float(vertices[:, 2].max()),
            },
        }
        annotations[label] = {
            "found": True, "prism_found": True, "voxel_count": count,
            "centroid_physical": _xyz(direction @ center),
            "sagittal_body_quadrilateral": {
                "center": _xyz(direction @ center),
                "vertices": [_xyz(direction @ vertex) for vertex in vertices],
                "left_right_extent_mm": {"min": -20.0, "max": 20.0},
            },
        }
    localization = {
        "image": {
            "size_xyz": [512, 512, 200], "spacing_xyz": [1.0, 1.0, 1.0],
            "origin_xyz": [0.0, 0.0, 0.0], "direction": direction.ravel().tolist(),
        },
        "coords": coords,
        "spine_geometry": {"filtered_annotation_centroids": annotations},
    }
    names = [name for name, _, _ in bodies]
    intervals = [(f"{upper}_{lower}", f"vertebrae_{upper}", f"vertebrae_{lower}")
                 for upper, lower in zip(names, names[1:])]
    return localization, intervals


def _directions():
    angle = np.deg2rad(23.0)
    rotate = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle), np.cos(angle)]])
    return (np.eye(3), rotate, np.diag((-1.0, 1.0, 1.0)), np.diag((1.0, 1.0, -1.0)))


@pytest.mark.parametrize("direction", _directions())
def test_posterior_fragment_does_not_produce_coronal_interspace(direction):
    localization, intervals = _fragmented_lumbar_case(direction)
    segments, warnings = resolve_spine_interspace_segments(localization, intervals)

    assert [segment.group_name for segment in segments] == ["L1_L2", "L4_L5", "L5_S1"]
    skipped = [warning for warning in warnings if warning.startswith("跳过脊柱区间 ")]
    assert len(skipped) == 2
    for group in ("L2_L3", "L3_L4"):
        assert any(group in warning and "vertebrae_L3" in warning
                   and "侧壁" in warning for warning in skipped)

    # 正常三个间隙的真实倾角和位置不能因坏 L3 而移动。
    legacy = copy.deepcopy(localization)
    for body in legacy["spine_geometry"]["filtered_annotation_centroids"].values():
        body.pop("voxel_count")
    previous, _ = resolve_spine_interspace_segments(legacy, intervals)
    previous = {segment.group_name: segment for segment in previous}
    for segment in segments:
        assert segment == previous[segment.group_name]


def test_small_body_with_overlapping_ap_support_is_not_rejected():
    localization, intervals = _fragmented_lumbar_case(np.eye(3))
    body = localization["spine_geometry"]["filtered_annotation_centroids"]["vertebrae_L3"]
    body["sagittal_body_quadrilateral"] = {
        "center": _xyz((0, -110, -484.5)),
        "vertices": [_xyz((0, y, z)) for y, z in
                     ((-125, -487), (-95, -487), (-95, -482), (-125, -482))],
        "left_right_extent_mm": {"min": -20.0, "max": 20.0},
    }
    body["centroid_physical"] = _xyz((0, -110, -484.5))
    segments, warnings = resolve_spine_interspace_segments(localization, intervals[:3])

    assert [segment.group_name for segment in segments] == ["L1_L2", "L2_L3", "L3_L4"]
    assert not any("主体分割不足" in warning for warning in warnings)


def test_displaced_oblique_gap_is_not_clamped_or_rejected_for_small_volume_alone():
    localization, _ = _fragmented_lumbar_case(np.eye(3))
    annotations = localization["spine_geometry"]["filtered_annotation_centroids"]
    for label, polygon, center in (
        ("vertebrae_L2", [(0, 10), (2, 10), (2, 6), (0, 6)], (0, 1, 8)),
        ("vertebrae_L3", [(4, 4), (6, 4), (6, 0), (4, 0)], (0, 5, 2)),
    ):
        annotations[label]["centroid_physical"] = _xyz(center)
        annotations[label]["sagittal_body_quadrilateral"] = {
            "center": _xyz(center),
            "vertices": [_xyz((0, y, z)) for y, z in polygon],
            "left_right_extent_mm": {"min": -20.0, "max": 20.0},
        }
    segments, warnings = resolve_spine_interspace_segments(
        localization, [("L2_L3", "vertebrae_L2", "vertebrae_L3")],
    )

    assert len(segments) == 1
    assert segments[0].frame.normal.y == pytest.approx(2 ** -0.5)
    assert segments[0].frame.normal.z == pytest.approx(-(2 ** -0.5))
    assert not any("主体分割不足" in warning for warning in warnings)


@pytest.mark.parametrize("count", (None, "unknown", 0, float("inf")))
def test_missing_or_invalid_volume_evidence_keeps_legacy_result(count):
    localization, intervals = _fragmented_lumbar_case(np.eye(3))
    body = localization["spine_geometry"]["filtered_annotation_centroids"]["vertebrae_L3"]
    if count is None:
        body.pop("voxel_count")
    else:
        body["voxel_count"] = count
    segments, warnings = resolve_spine_interspace_segments(localization, intervals)

    assert len(segments) == 5
    assert not any("主体分割不足" in warning for warning in warnings)
