#!/src/planning/spine_common.py
# -*- coding: utf-8 -*-

"""脊柱类 planner 的公共方法。"""

from dataclasses import replace
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.models import Point3D
from src.models.enums import PlaneTypeEnum
from src.planning.geometry import (
    normalize_vector,
    subtract_points,
    vector_length,
    voxel_index_to_physical,
)
from src.planning.straight import CutSegment, StraightPlaneFrame, default_straight_frame


_CERVICAL_INTERSPACE_MAX_CENTERLINE_DEVIATION_DEGREES = 10.0
_CERVICAL_INTERSPACE_LOW_BODY_VOLUME_FRACTION = 0.75
_INTERSPACE_BODY_PAIR_MIN_SUPPORT_POINTS = 8
_INTERSPACE_PROJECTED_GAP_UNDERESTIMATE_MM = 1.0
_CERVICAL_TERMINAL_CENTER_OFFSET_TRIGGER_MM = 0.25
_CERVICAL_TERMINAL_INFERIOR_MARGIN_MM = 0.5
_INTERSPACE_SIDEWALL_MAX_BODY_VOLUME_RATIO = 0.25
_INTERSPACE_SIDEWALL_MAX_SI_ALIGNMENT = 0.5


def voxel_centroid_to_physical(
    centroid: Dict[str, float],
    image_info: Dict[str, Any],
) -> Point3D:
    """
    把第四阶段输出的 z/y/x 体素质心转成 x/y/z 物理坐标。

    这里必须同时使用 spacing、origin 和 direction，避免带方向矩阵的
    NIfTI 在第六阶段生成错误的物理坐标。
    """

    return voxel_index_to_physical(
        Point3D(centroid["x"], centroid["y"], centroid["z"]),
        image_info,
    )


def get_label_center(
    localization_result: Dict[str, Any],
    label_name: str,
) -> Optional[Point3D]:
    """读取某个 label 的中心点，未找到时返回 None。"""

    coord = localization_result.get("coords", {}).get(label_name)
    if not coord or not coord.get("found") or not coord.get("centroid"):
        return None
    return voxel_centroid_to_physical(
        coord["centroid"],
        localization_result["image"],
    )


def _label_bbox(
    localization_result: Dict[str, Any],
    label_name: str,
) -> Optional[Dict[str, float]]:
    """读取一个已找到 label 的体素 bbox，缺失或未找到时返回 None。"""

    coord = localization_result.get("coords", {}).get(label_name)
    if not coord or not coord.get("found") or not coord.get("bbox"):
        return None
    return dict(coord["bbox"])


def union_bboxes(*bboxes: Dict[str, float]) -> Dict[str, float]:
    """合并一个或多个体素 bbox，返回能够覆盖全部输入的最小轴对齐框。"""

    if not bboxes:
        raise ValueError("合并体素 bbox 时至少需要一个输入")
    return {
        "x_min": min(float(bbox["x_min"]) for bbox in bboxes),
        "x_max": max(float(bbox["x_max"]) for bbox in bboxes),
        "y_min": min(float(bbox["y_min"]) for bbox in bboxes),
        "y_max": max(float(bbox["y_max"]) for bbox in bboxes),
        "z_min": min(float(bbox["z_min"]) for bbox in bboxes),
        "z_max": max(float(bbox["z_max"]) for bbox in bboxes),
    }


def _bbox_head_foot_boundary_points(
    bbox: Dict[str, float],
    in_plane_bbox: Dict[str, float],
    image_info: Dict[str, Any],
) -> Tuple[Point3D, Point3D]:
    """返回 bbox 在患者物理 z 方向上的足侧、头侧外边界中心点。"""

    x_center = (
        float(in_plane_bbox["x_min"]) + float(in_plane_bbox["x_max"])
    ) / 2.0
    y_center = (
        float(in_plane_bbox["y_min"]) + float(in_plane_bbox["y_max"])
    ) / 2.0
    first_point = voxel_index_to_physical(
        Point3D(x_center, y_center, float(bbox["z_min"]) - 0.5),
        image_info,
    )
    second_point = voxel_index_to_physical(
        Point3D(x_center, y_center, float(bbox["z_max"]) + 0.5),
        image_info,
    )
    if first_point.z <= second_point.z:
        return first_point, second_point
    return second_point, first_point


def _parse_xyz_point(value: Any) -> Optional[Point3D]:
    """解析一个包含有限 x/y/z 的点，坐标语义由调用方决定。"""

    if not isinstance(value, dict):
        return None
    try:
        coordinates = tuple(float(value[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        return None
    return Point3D(*coordinates)


def _filtered_spine_annotations(localization_result: Dict[str, Any]) -> Dict[str, Any]:
    """安全读取第五阶段过滤几何，损坏容器按缺失处理。"""

    spine_geometry = localization_result.get("spine_geometry")
    if not isinstance(spine_geometry, dict):
        return {}
    annotations = spine_geometry.get("filtered_annotation_centroids")
    if not isinstance(annotations, dict):
        return {}
    return annotations


def _parse_voxel_bbox(value: Any) -> Optional[Dict[str, float]]:
    """解析一个有限且非反向的 x/y/z 体素包围盒。"""

    if not isinstance(value, dict):
        return None
    try:
        bbox = {
            f"{axis}_{side}": float(value[f"{axis}_{side}"])
            for axis in ("x", "y", "z")
            for side in ("min", "max")
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(coordinate) for coordinate in bbox.values()):
        return None
    if any(
        bbox[f"{axis}_min"] > bbox[f"{axis}_max"]
        for axis in ("x", "y", "z")
    ):
        return None
    return bbox


def _voxel_point_or_bbox_center(
    value: Any,
    bbox: Dict[str, float],
) -> Point3D:
    """读取体素点；字段损坏时使用对应包围盒中心。"""

    point = _parse_xyz_point(value)
    if point is not None:
        return point
    return Point3D(*(
        (bbox[f"{axis}_min"] + bbox[f"{axis}_max"]) / 2.0
        for axis in ("x", "y", "z")
    ))


def _spine_label_voxel_geometry(
    localization_result: Dict[str, Any],
    label_name: str,
) -> Optional[Tuple[Dict[str, float], Point3D]]:
    """按过滤主体、四棱柱、全量标签顺序读取定位图选层几何。"""

    annotation = _filtered_spine_annotations(localization_result).get(label_name)
    if isinstance(annotation, dict) and annotation.get("found") is not False:
        body_bbox = _parse_voxel_bbox(annotation.get("body_bbox"))
        if body_bbox is not None:
            return body_bbox, _voxel_point_or_bbox_center(
                annotation.get("centroid_voxel"),
                body_bbox,
            )
        if annotation.get("prism_found"):
            prism_bbox = _parse_voxel_bbox(annotation.get("prism_bbox"))
            if prism_bbox is not None:
                return prism_bbox, _voxel_point_or_bbox_center(
                    annotation.get("prism_centroid_voxel"),
                    prism_bbox,
                )

    coord = localization_result.get("coords", {}).get(label_name)
    if not isinstance(coord, dict) or not coord.get("found"):
        return None
    bbox = _parse_voxel_bbox(coord.get("bbox"))
    if bbox is None:
        return None
    return bbox, _voxel_point_or_bbox_center(coord.get("centroid"), bbox)


def _median(values: Sequence[float]) -> float:
    """返回非空有限序列的中位数。"""

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _shared_spine_axis_coordinate(
    geometries: Sequence[Tuple[Dict[str, float], Point3D]],
    axis: str,
) -> float:
    """选择穿过全部椎体，或在无交集时覆盖椎体数最多的轴坐标。"""

    lower = max(bbox[f"{axis}_min"] for bbox, _ in geometries)
    upper = min(bbox[f"{axis}_max"] for bbox, _ in geometries)
    if lower <= upper:
        return (lower + upper) / 2.0

    center = _median([getattr(point, axis) for _, point in geometries])
    candidates = {center}
    for bbox, _ in geometries:
        minimum = bbox[f"{axis}_min"]
        maximum = bbox[f"{axis}_max"]
        candidates.update((minimum, maximum, (minimum + maximum) / 2.0))

    def score(candidate: float) -> Tuple[int, float]:
        coverage = sum(
            bbox[f"{axis}_min"] <= candidate <= bbox[f"{axis}_max"]
            for bbox, _ in geometries
        )
        return coverage, -abs(candidate - center)

    return max(sorted(candidates), key=score)


def resolve_spine_localizer_reference(
    localization_result: Dict[str, Any],
    label_names: Sequence[str],
) -> Optional[Point3D]:
    """计算一张尽量同时穿过指定椎体主体的统一定位图参考点。

    x、y 分别决定原始矢状层和冠状参考层；启用脊柱曲面背景时 y 只确定
    显示参考网格与单层回退，实际深度随主体变化。此点不参与正式切片
    中心、方向、范围或层间距计算。
    """

    geometries = [
        geometry
        for label_name in label_names
        if (geometry := _spine_label_voxel_geometry(
            localization_result,
            label_name,
        )) is not None
    ]
    if not geometries:
        return None

    voxel_reference = Point3D(
        _shared_spine_axis_coordinate(geometries, "x"),
        _shared_spine_axis_coordinate(geometries, "y"),
        _median([point.z for _, point in geometries]),
    )
    return voxel_index_to_physical(
        voxel_reference,
        localization_result["image"],
    )


def _annotation_physical_center(
    localization_result: Dict[str, Any],
    label_name: str,
) -> Optional[Point3D]:
    """按四棱柱、过滤主体、全量质心顺序取得椎体中心。"""

    annotation = _filtered_spine_annotations(localization_result).get(label_name, {})
    if annotation.get("prism_found"):
        point = _parse_xyz_point(annotation.get("prism_centroid_physical"))
        if point is not None:
            return point
    point = _parse_xyz_point(annotation.get("centroid_physical"))
    if point is not None:
        return point
    return get_label_center(localization_result, label_name)


def _face_candidates(annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """返回可参与椎间端面选择的矢状头脚两面。"""

    faces = annotation.get("sagittal_body_faces")
    if not isinstance(faces, list):
        return []
    candidates = []
    for face in faces:
        if (
            not isinstance(face, dict)
            or face.get("axis_index") != 0
            or face.get("side") not in {"min", "max"}
        ):
            continue
        center = _parse_xyz_point(face.get("center"))
        normal = _parse_xyz_point(face.get("normal"))
        if center is None or normal is None or vector_length(normal) == 0.0:
            continue
        try:
            support = int(face.get("point_count", 0))
        except (TypeError, ValueError):
            support = 0
        if support <= 0:
            continue
        candidates.append({
            "center": center,
            "normal": normalize_vector(normal),
            "point_count": support,
            "side": face["side"],
        })
    if len(candidates) < 2 or {face["side"] for face in candidates} != {"min", "max"}:
        return []
    return candidates


def _cross(first: Point3D, second: Point3D) -> Point3D:
    """计算两个物理方向的叉积。"""

    return Point3D(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    )


def _dot(first: Point3D, second: Point3D) -> float:
    """计算两个物理向量的点积。"""

    return first.x * second.x + first.y * second.y + first.z * second.z


def _is_finite_orthonormal_frame(
    frame: StraightPlaneFrame,
    require_right_handed: bool = False,
) -> bool:
    """检查切片坐标系的三个方向均有限、单位化且两两正交。"""

    directions = (frame.row_dir, frame.col_dir, frame.normal)
    if not all(
        math.isfinite(value)
        for direction in directions
        for value in (direction.x, direction.y, direction.z)
    ):
        return False
    if any(abs(vector_length(direction) - 1.0) > 1e-6 for direction in directions):
        return False
    if any(
        abs(_dot(first, second)) > 1e-6
        for first, second in (
            (frame.row_dir, frame.col_dir),
            (frame.row_dir, frame.normal),
            (frame.col_dir, frame.normal),
        )
    ):
        return False
    handedness = _dot(_cross(frame.row_dir, frame.col_dir), frame.normal)
    if abs(abs(handedness) - 1.0) > 1e-6:
        return False
    return not require_right_handed or handedness > 0.0


def _cross_2d(first: tuple[float, float], second: tuple[float, float]) -> float:
    """计算二维向量叉积的标量 z 分量。"""

    return first[0] * second[1] - first[1] * second[0]


def _is_strictly_convex_polygon_2d(
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """检查有序多边形有限、非退化且保持一致转向。"""

    if len(polygon) != 4 or not all(
        math.isfinite(value)
        for point in polygon
        for value in point
    ):
        return False
    turns = []
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        next_point = polygon[(index + 2) % len(polygon)]
        turn = _cross_2d(
            (following[0] - current[0], following[1] - current[1]),
            (next_point[0] - following[0], next_point[1] - following[1]),
        )
        if abs(turn) <= 1e-8:
            return False
        turns.append(turn)
    return all(turn > 0.0 for turn in turns) or all(turn < 0.0 for turn in turns)


def _point_to_segment_2d(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """返回点到二维线段的平方距离及线段上的最近点。"""

    direction = (end[0] - start[0], end[1] - start[1])
    length_squared = direction[0] ** 2 + direction[1] ** 2
    if length_squared <= 1e-12:
        nearest = start
    else:
        parameter = max(0.0, min(1.0, (
            (point[0] - start[0]) * direction[0]
            + (point[1] - start[1]) * direction[1]
        ) / length_squared))
        nearest = (
            start[0] + parameter * direction[0],
            start[1] + parameter * direction[1],
        )
    delta = (point[0] - nearest[0], point[1] - nearest[1])
    return delta[0] ** 2 + delta[1] ** 2, nearest, point


def _point_inside_convex_polygon_2d(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """判断点是否在有序凸多边形内（含边界）。"""

    signs = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        signs.append(_cross_2d(
            (end[0] - start[0], end[1] - start[1]),
            (point[0] - start[0], point[1] - start[1]),
        ))
    return all(value >= -1e-8 for value in signs) or all(
        value <= 1e-8 for value in signs
    )


def _segments_intersect_2d(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    """判断两条二维闭线段是否相交或接触。"""

    tolerance = 1e-8

    def orientation(
        start: tuple[float, float],
        end: tuple[float, float],
        point: tuple[float, float],
    ) -> float:
        return _cross_2d(
            (end[0] - start[0], end[1] - start[1]),
            (point[0] - start[0], point[1] - start[1]),
        )

    def on_segment(
        start: tuple[float, float],
        end: tuple[float, float],
        point: tuple[float, float],
    ) -> bool:
        return (
            min(start[0], end[0]) - tolerance
            <= point[0]
            <= max(start[0], end[0]) + tolerance
            and min(start[1], end[1]) - tolerance
            <= point[1]
            <= max(start[1], end[1]) + tolerance
        )

    values = (
        orientation(first_start, first_end, second_start),
        orientation(first_start, first_end, second_end),
        orientation(second_start, second_end, first_start),
        orientation(second_start, second_end, first_end),
    )
    first_straddles = (
        (values[0] > tolerance and values[1] < -tolerance)
        or (values[0] < -tolerance and values[1] > tolerance)
    )
    second_straddles = (
        (values[2] > tolerance and values[3] < -tolerance)
        or (values[2] < -tolerance and values[3] > tolerance)
    )
    if first_straddles and second_straddles:
        return True
    return any((
        abs(values[0]) <= tolerance
        and on_segment(first_start, first_end, second_start),
        abs(values[1]) <= tolerance
        and on_segment(first_start, first_end, second_end),
        abs(values[2]) <= tolerance
        and on_segment(second_start, second_end, first_start),
        abs(values[3]) <= tolerance
        and on_segment(second_start, second_end, first_end),
    ))


def _closest_convex_polygon_pair_2d(
    upper: Sequence[tuple[float, float]],
    lower: Sequence[tuple[float, float]],
) -> Optional[tuple[tuple[float, float], tuple[float, float], float]]:
    """求两个凸多边形边界之间的最近点对。"""

    if len(upper) < 3 or len(lower) < 3:
        return None
    if any(
        _point_inside_convex_polygon_2d(point, lower)
        for point in upper
    ) or any(
        _point_inside_convex_polygon_2d(point, upper)
        for point in lower
    ):
        return None
    for upper_index, upper_start in enumerate(upper):
        upper_end = upper[(upper_index + 1) % len(upper)]
        for lower_index, lower_start in enumerate(lower):
            lower_end = lower[(lower_index + 1) % len(lower)]
            if _segments_intersect_2d(
                upper_start,
                upper_end,
                lower_start,
                lower_end,
            ):
                return None
    best: Optional[tuple[tuple[float, float], tuple[float, float], float]] = None
    for first_polygon, second_polygon in ((upper, lower), (lower, upper)):
        for point in first_polygon:
            for edge_index, start in enumerate(second_polygon):
                end = second_polygon[(edge_index + 1) % len(second_polygon)]
                distance_squared, nearest, source = _point_to_segment_2d(
                    point,
                    start,
                    end,
                )
                pair = (source, nearest, distance_squared)
                if first_polygon is lower:
                    pair = (nearest, source, distance_squared)
                if best is None or pair[2] < best[2]:
                    best = pair
    return best


def _facing_convex_polygon_boundary_pair_2d(
    upper: Sequence[tuple[float, float]],
    lower: Sequence[tuple[float, float]],
    permit_inverted_gap: bool = False,
    *,
    upper_boundary=None,
    lower_boundary=None,
) -> Optional[tuple[tuple[float, float], tuple[float, float], float]]:
    """在两条相向端边的共同覆盖中段构造代表性边界对。"""

    if len(upper) < 3 or len(lower) < 3:
        return None

    upper_center = (
        sum(point[0] for point in upper) / len(upper),
        sum(point[1] for point in upper) / len(upper),
    )
    lower_center = (
        sum(point[0] for point in lower) / len(lower),
        sum(point[1] for point in lower) / len(lower),
    )
    interval_direction = (
        lower_center[0] - upper_center[0],
        lower_center[1] - upper_center[1],
    )
    direction_length_squared = (
        interval_direction[0] ** 2 + interval_direction[1] ** 2
    )
    if direction_length_squared <= 1e-12:
        return None
    direction_length = math.sqrt(direction_length_squared)
    interval_unit = (
        interval_direction[0] / direction_length,
        interval_direction[1] / direction_length,
    )

    def signed_area(
        polygon: Sequence[tuple[float, float]],
    ) -> float:
        return sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        ) / 2.0

    def facing_edge(
        polygon: Sequence[tuple[float, float]],
        toward: tuple[float, float],
    ) -> Optional[
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ]:
        area = signed_area(polygon)
        if abs(area) <= 1e-12:
            return None
        candidates = []
        for index in range(len(polygon)):
            start = polygon[index]
            end = polygon[(index + 1) % len(polygon)]
            edge = (end[0] - start[0], end[1] - start[1])
            edge_length = math.hypot(edge[0], edge[1])
            if edge_length <= 1e-12:
                continue
            if area > 0.0:
                outward = (edge[1] / edge_length, -edge[0] / edge_length)
            else:
                outward = (-edge[1] / edge_length, edge[0] / edge_length)
            alignment = outward[0] * toward[0] + outward[1] * toward[1]
            candidates.append((alignment, start, end, outward))
        if not candidates:
            return None
        _, start, end, outward = max(candidates, key=lambda item: item[0])
        return start, end, outward

    upper_edge = upper_boundary or facing_edge(upper, interval_unit)
    lower_edge = lower_boundary or facing_edge(
        lower,
        (-interval_unit[0], -interval_unit[1]),
    )
    if upper_edge is None or lower_edge is None:
        return None
    upper_normal = upper_edge[2]
    lower_normal = (-lower_edge[2][0], -lower_edge[2][1])
    normal_sum = (
        upper_normal[0] + lower_normal[0],
        upper_normal[1] + lower_normal[1],
    )
    normal_length = math.hypot(normal_sum[0], normal_sum[1])
    if normal_length <= 1e-12:
        boundary_normal = interval_unit
    else:
        boundary_normal = (
            normal_sum[0] / normal_length,
            normal_sum[1] / normal_length,
        )
    alignment = (
        boundary_normal[0] * interval_unit[0]
        + boundary_normal[1] * interval_unit[1]
    )
    if alignment < 0.5:
        boundary_normal = interval_unit

    tangent = (-boundary_normal[1], boundary_normal[0])

    def tangent_range(
        edge: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> tuple[float, float]:
        values = (
            edge[0][0] * tangent[0] + edge[0][1] * tangent[1],
            edge[1][0] * tangent[0] + edge[1][1] * tangent[1],
        )
        return min(values), max(values)

    upper_range = tangent_range(upper_edge)
    lower_range = tangent_range(lower_edge)
    overlap_minimum = max(upper_range[0], lower_range[0])
    overlap_maximum = min(upper_range[1], lower_range[1])
    if overlap_maximum - overlap_minimum <= 1e-8:
        return None

    def point_at_tangent(
        edge: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        coordinate: float,
    ) -> Optional[tuple[float, float]]:
        start, end, _ = edge
        start_coordinate = start[0] * tangent[0] + start[1] * tangent[1]
        end_coordinate = end[0] * tangent[0] + end[1] * tangent[1]
        denominator = end_coordinate - start_coordinate
        if abs(denominator) <= 1e-12:
            return None
        parameter = (coordinate - start_coordinate) / denominator
        return (
            start[0] + (end[0] - start[0]) * parameter,
            start[1] + (end[1] - start[1]) * parameter,
        )

    def boundary_pair_at(
        coordinate: float,
    ) -> Optional[
        tuple[tuple[float, float], tuple[float, float], float]
    ]:
        upper_point = point_at_tangent(upper_edge, coordinate)
        lower_point = point_at_tangent(lower_edge, coordinate)
        if upper_point is None or lower_point is None:
            return None
        gap = (
            (lower_point[0] - upper_point[0]) * boundary_normal[0]
            + (lower_point[1] - upper_point[1]) * boundary_normal[1]
        )
        return upper_point, lower_point, gap

    minimum_pair = boundary_pair_at(overlap_minimum)
    maximum_pair = boundary_pair_at(overlap_maximum)
    if minimum_pair is None or maximum_pair is None:
        return None
    minimum_gap = minimum_pair[2]
    maximum_gap = maximum_pair[2]

    # 共同覆盖中点仍有正间隙时先保留候选。旧逻辑会先按零交点拆出正间隙
    # 子区间，再取子区间中点，可能让轻微相交的四棱柱把中心拉向一端。
    # 最终仍需与正区间代表间隙比较，避免为了居中而明显缩短有效范围。
    midpoint_coordinate = (overlap_minimum + overlap_maximum) / 2.0
    midpoint_pair = boundary_pair_at(midpoint_coordinate)
    boundaries = [overlap_minimum, overlap_maximum]
    if minimum_gap * maximum_gap < 0.0:
        crossing = overlap_minimum - minimum_gap * (
            overlap_maximum - overlap_minimum
        ) / (maximum_gap - minimum_gap)
        if overlap_minimum < crossing < overlap_maximum:
            boundaries.append(crossing)
    boundaries.sort()

    positive_intervals = []
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2.0
        pair = boundary_pair_at(midpoint)
        if pair is not None and pair[2] > 1e-8:
            positive_intervals.append((end - start, midpoint, pair))
    if positive_intervals:
        _, _, selected_pair = max(
            positive_intervals,
            key=lambda item: (item[0], item[2][2]),
        )
        if (
            midpoint_pair is not None
            and midpoint_pair[2] > 1e-8
            and midpoint_pair[2] >= selected_pair[2] * 0.75
        ):
            selected_pair = midpoint_pair
        upper_boundary, lower_boundary, gap = selected_pair
        return upper_boundary, lower_boundary, gap * gap

    if not permit_inverted_gap:
        return None
    pair = midpoint_pair
    if pair is None or abs(pair[2]) <= 1e-8:
        return None
    raw_upper, raw_lower, raw_gap = pair
    midpoint = (
        (raw_upper[0] + raw_lower[0]) / 2.0,
        (raw_upper[1] + raw_lower[1]) / 2.0,
    )
    half_distance = abs(raw_gap) / 2.0
    ordered_upper = (
        midpoint[0] - boundary_normal[0] * half_distance,
        midpoint[1] - boundary_normal[1] * half_distance,
    )
    ordered_lower = (
        midpoint[0] + boundary_normal[0] * half_distance,
        midpoint[1] + boundary_normal[1] * half_distance,
    )
    return ordered_upper, ordered_lower, raw_gap * raw_gap


def _body_centroid_physical(
    localization_result: Dict[str, Any],
    label_name: str,
) -> Optional[Point3D]:
    """读取过滤主体质心，缺失时回退到完整标签质心。"""

    annotation = _filtered_spine_annotations(localization_result).get(
        label_name,
        {},
    )
    if isinstance(annotation, dict):
        point = _parse_xyz_point(annotation.get("centroid_physical"))
        if point is not None:
            return point
    return get_label_center(localization_result, label_name)


def _limited_normal_toward_centerline(
    normal: Point3D,
    centerline: Point3D,
    maximum_deviation_degrees: float,
) -> tuple[Point3D, float]:
    """把法向限制在主体中心线附近，并返回修正前的夹角。"""

    aligned_centerline = centerline
    dot_value = _dot(normal, aligned_centerline)
    if dot_value < 0.0:
        aligned_centerline = Point3D(
            -aligned_centerline.x,
            -aligned_centerline.y,
            -aligned_centerline.z,
        )
        dot_value = -dot_value
    dot_value = max(-1.0, min(1.0, dot_value))
    deviation_degrees = math.degrees(math.acos(dot_value))
    if deviation_degrees <= maximum_deviation_degrees:
        return normal, deviation_degrees

    tangent_component = Point3D(
        normal.x - aligned_centerline.x * dot_value,
        normal.y - aligned_centerline.y * dot_value,
        normal.z - aligned_centerline.z * dot_value,
    )
    tangent_direction = normalize_vector(tangent_component)
    limited_radians = math.radians(maximum_deviation_degrees)
    limited = normalize_vector(Point3D(
        aligned_centerline.x * math.cos(limited_radians)
        + tangent_direction.x * math.sin(limited_radians),
        aligned_centerline.y * math.cos(limited_radians)
        + tangent_direction.y * math.sin(limited_radians),
        aligned_centerline.z * math.cos(limited_radians)
        + tangent_direction.z * math.sin(limited_radians),
    ))
    return limited, deviation_degrees


def regularize_cervical_interspace_segments(
    localization_result: Dict[str, Any],
    segments: Sequence[CutSegment],
) -> tuple[List[CutSegment], List[str]]:
    """仅对低可信颈椎主体造成的异常椎间法向做有限度约束。"""

    annotations = _filtered_spine_annotations(localization_result)
    body_volumes = []
    for annotation in annotations.values():
        if not isinstance(annotation, dict):
            continue
        try:
            voxel_count = int(annotation.get("voxel_count", 0))
        except (TypeError, ValueError):
            continue
        if annotation.get("found") is not False and voxel_count > 0:
            body_volumes.append(float(voxel_count))
    if len(body_volumes) < 3:
        return list(segments), []
    median_body_volume = _median(body_volumes)

    corrected_segments: List[CutSegment] = []
    warnings: List[str] = []
    for segment in segments:
        if segment.frame is None or len(segment.source_labels) != 2:
            corrected_segments.append(segment)
            continue

        endpoint_volumes = []
        for label_name in segment.source_labels:
            annotation = annotations.get(label_name)
            try:
                voxel_count = int(
                    annotation.get("voxel_count", 0)
                    if isinstance(annotation, dict)
                    else 0
                )
            except (TypeError, ValueError):
                voxel_count = 0
            endpoint_volumes.append(voxel_count)
        low_volume_threshold = (
            median_body_volume
            * _CERVICAL_INTERSPACE_LOW_BODY_VOLUME_FRACTION
        )
        if not endpoint_volumes or min(endpoint_volumes) >= low_volume_threshold:
            corrected_segments.append(segment)
            continue

        upper_center = _body_centroid_physical(
            localization_result,
            segment.source_labels[0],
        )
        lower_center = _body_centroid_physical(
            localization_result,
            segment.source_labels[1],
        )
        if upper_center is None or lower_center is None:
            corrected_segments.append(segment)
            continue
        try:
            centerline = subtract_points(lower_center, upper_center)
            row_component = _dot(centerline, segment.frame.row_dir)
            centerline = normalize_vector(Point3D(
                centerline.x - segment.frame.row_dir.x * row_component,
                centerline.y - segment.frame.row_dir.y * row_component,
                centerline.z - segment.frame.row_dir.z * row_component,
            ))
            limited_normal, deviation_degrees = (
                _limited_normal_toward_centerline(
                    segment.frame.normal,
                    centerline,
                    _CERVICAL_INTERSPACE_MAX_CENTERLINE_DEVIATION_DEGREES,
                )
            )
        except ValueError:
            corrected_segments.append(segment)
            continue
        if deviation_degrees <= (
            _CERVICAL_INTERSPACE_MAX_CENTERLINE_DEVIATION_DEGREES
        ):
            corrected_segments.append(segment)
            continue

        target_vector = subtract_points(segment.target_end, segment.target_start)
        target_length = vector_length(target_vector)
        midpoint = Point3D(
            (segment.target_start.x + segment.target_end.x) / 2.0,
            (segment.target_start.y + segment.target_end.y) / 2.0,
            (segment.target_start.z + segment.target_end.z) / 2.0,
        )
        half_length = target_length / 2.0
        target_start = Point3D(
            midpoint.x - limited_normal.x * half_length,
            midpoint.y - limited_normal.y * half_length,
            midpoint.z - limited_normal.z * half_length,
        )
        target_end = Point3D(
            midpoint.x + limited_normal.x * half_length,
            midpoint.y + limited_normal.y * half_length,
            midpoint.z + limited_normal.z * half_length,
        )
        try:
            col_dir = normalize_vector(_cross(
                limited_normal,
                segment.frame.row_dir,
            ))
        except ValueError:
            corrected_segments.append(segment)
            continue
        corrected_segments.append(replace(
            segment,
            target_start=target_start,
            target_end=target_end,
            frame=StraightPlaneFrame(
                row_dir=segment.frame.row_dir,
                col_dir=col_dir,
                normal=limited_normal,
            ),
        ))
        warnings.append(
            f"脊柱区间 {segment.group_name} 主体体积偏小且四棱柱法向偏离"
            f"主体中心线 {deviation_degrees:.1f}°，已限制为 "
            f"{_CERVICAL_INTERSPACE_MAX_CENTERLINE_DEVIATION_DEGREES:.1f}°"
        )
    return corrected_segments, warnings


def _translate_interspace_to_body_pair(
    localization_result: Dict[str, Any],
    group_name: str,
    target_start: Point3D,
    target_end: Point3D,
    frame: StraightPlaneFrame,
) -> tuple[Point3D, Point3D, Optional[str]]:
    """用主体间隙和颈椎末端中面参考只校正切片位置。"""

    spine_geometry = localization_result.get("spine_geometry")
    if not isinstance(spine_geometry, dict):
        return target_start, target_end, None
    body_pairs = spine_geometry.get("interspace_body_pairs")
    if not isinstance(body_pairs, dict):
        return target_start, target_end, None
    body_pair = body_pairs.get(group_name)
    if not isinstance(body_pair, dict) or body_pair.get("found") is not True:
        return target_start, target_end, None
    try:
        minimum_distance = float(body_pair["minimum_distance_mm"])
        support_point_count = int(body_pair["support_point_count"])
    except (KeyError, TypeError, ValueError):
        return target_start, target_end, None
    midpoint_reference = _parse_xyz_point(body_pair.get("midpoint_physical"))
    if (
        midpoint_reference is None
        or not math.isfinite(minimum_distance)
        or minimum_distance <= 0.0
        or support_point_count < _INTERSPACE_BODY_PAIR_MIN_SUPPORT_POINTS
    ):
        return target_start, target_end, None

    target_length = vector_length(subtract_points(target_end, target_start))
    midpoint = Point3D(
        (target_start.x + target_end.x) / 2.0,
        (target_start.y + target_end.y) / 2.0,
        (target_start.z + target_end.z) / 2.0,
    )
    gap_underestimated = minimum_distance - target_length > (
        _INTERSPACE_PROJECTED_GAP_UNDERESTIMATE_MM
    )
    warning_reason: Optional[str] = None
    if gap_underestimated:
        translation_mm = _dot(
            subtract_points(midpoint_reference, midpoint),
            frame.normal,
        )
        warning_reason = (
            f"四棱柱投影范围 {target_length:.1f}mm 小于主体最近间隙 "
            f"{minimum_distance:.1f}mm"
        )
    elif _is_terminal_cervical_interspace(localization_result, group_name):
        annotations = _filtered_spine_annotations(localization_result)
        upper_annotation = annotations.get(
            str(body_pair.get("upper_label", "")),
        )
        lower_annotation = annotations.get(
            str(body_pair.get("lower_label", "")),
        )
        upper_center = _parse_xyz_point(
            upper_annotation.get("centroid_physical")
            if isinstance(upper_annotation, dict)
            else None
        )
        lower_center = _parse_xyz_point(
            lower_annotation.get("centroid_physical")
            if isinstance(lower_annotation, dict)
            else None
        )
        if upper_center is None or lower_center is None:
            return target_start, target_end, None
        body_midpoint = Point3D(
            (upper_center.x + lower_center.x) / 2.0,
            (upper_center.y + lower_center.y) / 2.0,
            (upper_center.z + lower_center.z) / 2.0,
        )
        center_offset_mm = _dot(
            subtract_points(body_midpoint, midpoint),
            frame.normal,
        )
        if center_offset_mm <= (
            _CERVICAL_TERMINAL_CENTER_OFFSET_TRIGGER_MM
        ):
            return target_start, target_end, None
        translation_mm = (
            center_offset_mm + _CERVICAL_TERMINAL_INFERIOR_MARGIN_MM
        )
        warning_reason = (
            f"中心位于相邻主体中面上方 {center_offset_mm:.1f}mm，"
            f"已向 C7 侧保留 {_CERVICAL_TERMINAL_INFERIOR_MARGIN_MM:.1f}mm 余量"
        )
    else:
        return target_start, target_end, None

    if not math.isfinite(translation_mm) or abs(translation_mm) <= 1e-8:
        return target_start, target_end, None

    translation = Point3D(
        frame.normal.x * translation_mm,
        frame.normal.y * translation_mm,
        frame.normal.z * translation_mm,
    )
    translated_start = Point3D(
        target_start.x + translation.x,
        target_start.y + translation.y,
        target_start.z + translation.z,
    )
    translated_end = Point3D(
        target_end.x + translation.x,
        target_end.y + translation.y,
        target_end.z + translation.z,
    )
    warning = (
        f"脊柱区间 {group_name} {warning_reason}，已沿法向平移 "
        f"{abs(translation_mm):.1f}mm"
    )
    return translated_start, translated_end, warning


def _is_terminal_cervical_interspace(
    localization_result: Dict[str, Any],
    group_name: str,
) -> bool:
    """判断当前区域是否是配置中的最后一个颈椎间隙。"""

    if localization_result.get("slice_part") != "颈椎":
        return False
    resolved_configs = localization_result.get("resolved_cut_configs")
    interspace_config = (
        resolved_configs.get("interspace")
        if isinstance(resolved_configs, dict)
        else None
    )
    intervals = (
        interspace_config.get("intervals")
        if isinstance(interspace_config, dict)
        else None
    )
    if not isinstance(intervals, list) or not intervals:
        return False
    terminal = intervals[-1]
    return isinstance(terminal, dict) and terminal.get("region") == group_name


def _overlapping_convex_polygon_boundary_pair_2d(
    upper: Sequence[tuple[float, float]],
    lower: Sequence[tuple[float, float]],
) -> Optional[tuple[tuple[float, float], tuple[float, float], float]]:
    """外包络轻微穿入时用相向端边构造稳定的有向边界对。"""

    if len(upper) < 3 or len(lower) < 3:
        return None
    upper_inside = [
        _point_inside_convex_polygon_2d(point, lower)
        for point in upper
    ]
    lower_inside = [
        _point_inside_convex_polygon_2d(point, upper)
        for point in lower
    ]
    if not any(upper_inside) and not any(lower_inside):
        # 顶点互不包含而边界相交是十字穿越，不能伪造椎间间隙。
        return None
    if all(upper_inside) or all(lower_inside):
        # 完整包含不是相邻椎体外包络的轻微穿入。
        return None
    return _facing_convex_polygon_boundary_pair_2d(
        upper,
        lower,
        permit_inverted_gap=True,
    )


def _parse_sagittal_prism(
    annotation: Dict[str, Any],
    frame: StraightPlaneFrame,
) -> Optional[tuple[list[tuple[float, float]], tuple[float, float]]]:
    """读取第五阶段保存的四棱柱矢状轮廓及左右范围。"""

    geometry = annotation.get("sagittal_body_quadrilateral")
    if not isinstance(geometry, dict):
        return None
    if not _is_finite_orthonormal_frame(frame):
        return None
    raw_vertices = geometry.get("vertices")
    if not isinstance(raw_vertices, list) or len(raw_vertices) != 4:
        return None
    vertices = [_parse_xyz_point(value) for value in raw_vertices]
    if any(vertex is None for vertex in vertices):
        return None
    typed_vertices = [vertex for vertex in vertices if vertex is not None]
    center = _parse_xyz_point(geometry.get("center"))
    if center is None:
        center = Point3D(
            sum(vertex.x for vertex in typed_vertices) / len(typed_vertices),
            sum(vertex.y for vertex in typed_vertices) / len(typed_vertices),
            sum(vertex.z for vertex in typed_vertices) / len(typed_vertices),
        )
    extent = geometry.get("left_right_extent_mm")
    if not isinstance(extent, dict):
        return None
    try:
        minimum = float(extent["min"]) + _dot(center, frame.row_dir)
        maximum = float(extent["max"]) + _dot(center, frame.row_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (minimum, maximum)) or minimum >= maximum:
        return None
    polygon = [
        (_dot(vertex, frame.col_dir), _dot(vertex, frame.normal))
        for vertex in typed_vertices
    ]
    if not _is_strictly_convex_polygon_2d(polygon):
        return None
    return polygon, (minimum, maximum)


def _supported_sagittal_endplate_edge(
    annotation: Dict[str, Any],
    frame: StraightPlaneFrame,
    toward: tuple[float, float],
):
    """读取主体恢复后独立拟合的端面，覆盖包络仍由四棱柱负责。"""

    faces = annotation.get("sagittal_body_endplates")
    if not isinstance(faces, list) or len(faces) != 2:
        return None
    toward_length = math.hypot(*toward)
    if toward_length <= 1e-8:
        return None
    candidates = []
    for face in faces:
        if not isinstance(face, dict):
            return None
        endpoints = face.get("endpoints")
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            return None
        points = [_parse_xyz_point(point) for point in endpoints]
        normal = _parse_xyz_point(face.get("normal"))
        if any(point is None for point in points) or normal is None:
            return None
        start, end = [(_dot(point, frame.col_dir), _dot(point, frame.normal)) for point in points]
        vector = (end[0] - start[0], end[1] - start[1])
        length = math.hypot(*vector)
        normal_2d = (_dot(normal, frame.col_dir), _dot(normal, frame.normal))
        normal_length = math.hypot(*normal_2d)
        if length <= 1e-8 or normal_length <= 1e-8:
            return None
        normal_2d = (normal_2d[0] / normal_length, normal_2d[1] / normal_length)
        if abs(vector[0] * normal_2d[0] + vector[1] * normal_2d[1]) > length * 1e-5:
            return None
        alignment = (normal_2d[0] * toward[0] + normal_2d[1] * toward[1]) / toward_length
        candidates.append((alignment, start, end, normal_2d))
    best = max(candidates, key=lambda item: item[0])
    return best[1:] if best[0] > 0.0 else None


def _resolve_interspace_prism_geometry(
    localization_result: Dict[str, Any],
    upper_label: str,
    lower_label: str,
) -> Optional[Tuple[Point3D, Point3D, StraightPlaneFrame]]:
    """用两个四棱柱的凸四边形最近点生成椎间切法向。"""

    filtered = _filtered_spine_annotations(localization_result)
    upper_annotation = filtered.get(upper_label)
    lower_annotation = filtered.get(lower_label)
    if not isinstance(upper_annotation, dict) or not isinstance(lower_annotation, dict):
        return None
    if (
        upper_annotation.get("prism_found") is not True
        or lower_annotation.get("prism_found") is not True
    ):
        return None
    try:
        default_frame = default_straight_frame(
            PlaneTypeEnum.横状位,
            localization_result["image"],
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not _is_finite_orthonormal_frame(default_frame):
        return None
    upper_data = _parse_sagittal_prism(upper_annotation, default_frame)
    lower_data = _parse_sagittal_prism(lower_annotation, default_frame)
    if upper_data is None or lower_data is None:
        return None
    upper_polygon, upper_lr = upper_data
    lower_polygon, lower_lr = lower_data
    overlap_minimum = max(upper_lr[0], lower_lr[0])
    overlap_maximum = min(upper_lr[1], lower_lr[1])
    # 只有左右范围重叠时，最近点对才落在同一张横切平面内；否则该标注
    # 不能代表本椎间的横切端面，交给旧 bbox 逻辑回退并报警。
    if overlap_minimum >= overlap_maximum:
        return None
    upper_center = tuple(sum(p[axis] for p in upper_polygon) / 4 for axis in (0, 1))
    lower_center = tuple(sum(p[axis] for p in lower_polygon) / 4 for axis in (0, 1))
    toward_lower = (lower_center[0] - upper_center[0], lower_center[1] - upper_center[1])
    recovered_pair = any(
        isinstance(annotation.get("body_extraction"), dict)
        and annotation["body_extraction"].get("method") == "density_core_recovery"
        for annotation in (upper_annotation, lower_annotation)
    )
    upper_boundary = _supported_sagittal_endplate_edge(
        upper_annotation, default_frame, toward_lower,
    ) if recovered_pair else None
    lower_boundary = _supported_sagittal_endplate_edge(
        lower_annotation, default_frame, (-toward_lower[0], -toward_lower[1]),
    ) if recovered_pair else None
    closest = None
    if upper_boundary is not None or lower_boundary is not None:
        closest = _facing_convex_polygon_boundary_pair_2d(
            upper_polygon, lower_polygon,
            upper_boundary=upper_boundary, lower_boundary=lower_boundary,
        )
    if closest is None:
        closest = _closest_convex_polygon_pair_2d(upper_polygon, lower_polygon)
        if closest is not None:
            representative = _facing_convex_polygon_boundary_pair_2d(
                upper_polygon,
                lower_polygon,
            )
            if representative is not None:
                closest = representative
        else:
            closest = _overlapping_convex_polygon_boundary_pair_2d(
                upper_polygon,
                lower_polygon,
            )
    if closest is None or closest[2] <= 1e-8:
        return None
    upper_2d, lower_2d, _ = closest
    row_coordinate = (overlap_minimum + overlap_maximum) / 2.0

    def lift(point: tuple[float, float]) -> Point3D:
        return Point3D(
            default_frame.row_dir.x * row_coordinate
            + default_frame.col_dir.x * point[0]
            + default_frame.normal.x * point[1],
            default_frame.row_dir.y * row_coordinate
            + default_frame.col_dir.y * point[0]
            + default_frame.normal.y * point[1],
            default_frame.row_dir.z * row_coordinate
            + default_frame.col_dir.z * point[0]
            + default_frame.normal.z * point[1],
        )

    target_start = lift(upper_2d)
    target_end = lift(lower_2d)
    try:
        normal = normalize_vector(subtract_points(target_end, target_start))
        col_dir = normalize_vector(_cross(normal, default_frame.row_dir))
    except ValueError:
        return None
    candidate_frame = StraightPlaneFrame(
        row_dir=default_frame.row_dir,
        col_dir=col_dir,
        normal=normal,
    )
    if not _is_finite_orthonormal_frame(
        candidate_frame,
        require_right_handed=True,
    ):
        return None
    if _dot(normal, subtract_points(target_end, target_start)) <= 0.0:
        return None
    return (
        target_start,
        target_end,
        candidate_frame,
    )


def _interspace_sidewall_geometry_warning(
    localization_result: Dict[str, Any],
    upper_label: str,
    lower_label: str,
    frame: StraightPlaneFrame,
) -> Optional[str]:
    """拒绝小残块与相邻主体侧壁之间的伪椎间隙，不凭空补出缺失主体。

    体积悬殊、前后投影无交集、候选法向远离头脚轴必须同时成立，避免
    仅因椎体偏小或正常斜切就丢弃间隙。缺少体积证据的旧结果保持兼容。
    """

    annotations = _filtered_spine_annotations(localization_result)
    labels = (upper_label, lower_label)
    bodies = [annotations.get(label) for label in labels]
    if not all(isinstance(body, dict) for body in bodies):
        return None
    try:
        counts = [int(body["voxel_count"]) for body in bodies]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if min(counts) <= 0 or min(counts) / max(counts) >= (
        _INTERSPACE_SIDEWALL_MAX_BODY_VOLUME_RATIO
    ):
        return None
    reference = default_straight_frame(PlaneTypeEnum.横状位, localization_result["image"])
    if abs(_dot(frame.normal, reference.normal)) >= _INTERSPACE_SIDEWALL_MAX_SI_ALIGNMENT:
        return None
    prisms = [_parse_sagittal_prism(body, reference) for body in bodies]
    if any(prism is None for prism in prisms):
        return None
    anterior_posterior_ranges = [
        (min(point[0] for point in prism[0]), max(point[0] for point in prism[0]))
        for prism in prisms
    ]
    if max(interval[0] for interval in anterior_posterior_ranges) <= (
        min(interval[1] for interval in anterior_posterior_ranges) + 1e-6
    ):
        return None
    smaller_index = counts.index(min(counts))
    return (
        f"{labels[smaller_index]} 主体分割不足（{min(counts)} 体素，"
        f"相邻主体 {max(counts)} 体素），四棱柱前后范围不重叠，"
        "候选切面来自侧壁，无法可靠确定椎间隙"
    )


def _resolve_interspace_face_geometry(
    localization_result: Dict[str, Any],
    upper_label: str,
    lower_label: str,
) -> Optional[Tuple[Point3D, Point3D, StraightPlaneFrame]]:
    """从相邻椎体相向端面生成椎间边界和独立斜横切坐标系。"""

    filtered = _filtered_spine_annotations(localization_result)
    upper_annotation = filtered.get(upper_label)
    lower_annotation = filtered.get(lower_label)
    if not isinstance(upper_annotation, dict) or not isinstance(lower_annotation, dict):
        return None
    # prism_found 会显式标记碰撞等原因造成的四棱柱失效；只有字段缺失时
    # 才允许尝试端面回退，不能把已判无效的几何重新复活。
    if (
        upper_annotation.get("prism_found") is False
        or lower_annotation.get("prism_found") is False
    ):
        return None

    upper_center = _annotation_physical_center(localization_result, upper_label)
    lower_center = _annotation_physical_center(localization_result, lower_label)
    if upper_center is None or lower_center is None:
        return None
    try:
        interval_direction = normalize_vector(
            subtract_points(lower_center, upper_center)
        )
    except ValueError:
        return None

    selected_faces = []
    for annotation_index, annotation in enumerate((upper_annotation, lower_annotation)):
        candidates = _face_candidates(annotation)
        if not candidates:
            return None
        center = upper_center if annotation_index == 0 else lower_center
        # 上位椎体选择沿 interval_direction 最远的一面（朝下位），
        # 下位椎体选择投影最小的一面（朝上位）。
        selected_faces.append(min(
            candidates,
            key=lambda face: (
                -_dot(
                    subtract_points(face["center"], center),
                    interval_direction,
                ) if annotation_index == 0 else _dot(
                    subtract_points(face["center"], center),
                    interval_direction,
                ),
                -face["point_count"],
            ),
        ))

    upper_face, lower_face = selected_faces
    upper_normal = upper_face["normal"]
    lower_normal = lower_face["normal"]
    if _dot(upper_normal, interval_direction) < 0.0:
        upper_normal = Point3D(-upper_normal.x, -upper_normal.y, -upper_normal.z)
    if _dot(lower_normal, interval_direction) < 0.0:
        lower_normal = Point3D(-lower_normal.x, -lower_normal.y, -lower_normal.z)
    try:
        normal = normalize_vector(Point3D(
            upper_normal.x + lower_normal.x,
            upper_normal.y + lower_normal.y,
            upper_normal.z + lower_normal.z,
        ))
    except ValueError:
        return None

    try:
        default_frame = default_straight_frame(
            PlaneTypeEnum.横状位,
            localization_result["image"],
        )
        if not all(
            math.isfinite(value)
            for direction in (
                default_frame.row_dir,
                default_frame.col_dir,
                default_frame.normal,
            )
            for value in (direction.x, direction.y, direction.z)
        ):
            return None
        normal_in_plane = Point3D(
            normal.x - default_frame.row_dir.x * _dot(normal, default_frame.row_dir),
            normal.y - default_frame.row_dir.y * _dot(normal, default_frame.row_dir),
            normal.z - default_frame.row_dir.z * _dot(normal, default_frame.row_dir),
        )
        normal = normalize_vector(normal_in_plane)
        if not all(
            math.isfinite(value)
            for value in (normal.x, normal.y, normal.z)
        ):
            return None
        if _dot(normal, interval_direction) < 0.0:
            normal = Point3D(-normal.x, -normal.y, -normal.z)
        col_dir = normalize_vector(_cross(normal, default_frame.row_dir))
    except (KeyError, TypeError, ValueError):
        return None
    return (
        upper_face["center"],
        lower_face["center"],
        StraightPlaneFrame(
            row_dir=default_frame.row_dir,
            col_dir=col_dir,
            normal=normal,
        ),
    )


def _annotation_symmetry_axis(annotation: Any) -> Optional[Point3D]:
    """只读取通过第五阶段质量检查的物理左右轴。"""

    if not isinstance(annotation, dict):
        return None
    diagnostics = annotation.get("symmetry_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("status") not in {
        "accepted", "unchanged",
    }:
        return None
    axis = _parse_xyz_point(
        annotation.get("symmetry_left_right_axis")
    )
    if axis is None:
        return None
    try:
        return normalize_vector(axis)
    except ValueError:
        return None


def _refine_interspace_symmetry_geometry(
    localization_result: Dict[str, Any],
    upper_label: str,
    lower_label: str,
    target_start: Point3D,
    target_end: Point3D,
    frame: Optional[StraightPlaneFrame],
) -> tuple[Point3D, Point3D, Optional[StraightPlaneFrame], Optional[str]]:
    """在保留真实间隙中点和长度的同时，补上局部 yaw 与左右侧倾。

    四棱柱的矢状端面仍决定前后倾斜；左右轴由两个完整椎体的可靠对称面
    决定，法向随后正交化。缺少端面时只使用真实主体质心连线作为带警告的
    方向回退，不创建假的端面，也不改变已有 bbox 所给出的间隙范围。
    """

    annotations = _filtered_spine_annotations(localization_result)
    axes = [_annotation_symmetry_axis(annotations.get(label))
            for label in (upper_label, lower_label)]
    if any(axis is None for axis in axes):
        return target_start, target_end, frame, "缺少可靠左右对称轴，保留原椎间方向"
    reference = default_straight_frame(PlaneTypeEnum.横状位, localization_result["image"])
    upper_axis, lower_axis = axes
    # 同一无向对称轴用输入左右方向统一符号，避免平均时相互抵消。
    upper_axis = upper_axis if _dot(upper_axis, reference.row_dir) >= 0.0 else Point3D(
        -upper_axis.x, -upper_axis.y, -upper_axis.z,
    )
    lower_axis = lower_axis if _dot(lower_axis, reference.row_dir) >= 0.0 else Point3D(
        -lower_axis.x, -lower_axis.y, -lower_axis.z,
    )
    if _dot(upper_axis, lower_axis) < math.cos(math.radians(25.0)):
        return target_start, target_end, frame, "相邻椎体左右轴差异过大，保留原椎间方向"
    try:
        row_dir = normalize_vector(Point3D(
            upper_axis.x + lower_axis.x,
            upper_axis.y + lower_axis.y,
            upper_axis.z + lower_axis.z,
        ))
        warning = None
        if frame is not None:
            source_normal = frame.normal
        else:
            upper_center = _annotation_physical_center(localization_result, upper_label)
            lower_center = _annotation_physical_center(localization_result, lower_label)
            if upper_center is None or lower_center is None:
                return target_start, target_end, frame, "缺少真实主体质心，保留原椎间方向"
            source_normal = normalize_vector(subtract_points(lower_center, upper_center))
            warning = "端面不足，使用真实主体质心连线与左右对称轴确定椎间方向"
        row_component = _dot(source_normal, row_dir)
        normal = normalize_vector(Point3D(
            source_normal.x - row_dir.x * row_component,
            source_normal.y - row_dir.y * row_component,
            source_normal.z - row_dir.z * row_component,
        ))
        if _dot(normal, source_normal) < math.cos(math.radians(30.0)):
            return target_start, target_end, frame, "局部对称轴要求的切面变化过大，保留原椎间方向"
        col_dir = normalize_vector(_cross(normal, row_dir))
        target_vector = subtract_points(target_end, target_start)
        if _dot(normal, target_vector) < 0.0:
            normal = Point3D(-normal.x, -normal.y, -normal.z)
            col_dir = Point3D(-col_dir.x, -col_dir.y, -col_dir.z)
        candidate_frame = StraightPlaneFrame(row_dir=row_dir, col_dir=col_dir, normal=normal)
        if not _is_finite_orthonormal_frame(candidate_frame, require_right_handed=True):
            raise ValueError("局部椎间坐标无效")
        half_length = vector_length(target_vector) / 2.0
        midpoint = Point3D(
            (target_start.x + target_end.x) / 2.0,
            (target_start.y + target_end.y) / 2.0,
            (target_start.z + target_end.z) / 2.0,
        )
        return (
            Point3D(midpoint.x - normal.x * half_length,
                    midpoint.y - normal.y * half_length,
                    midpoint.z - normal.z * half_length),
            Point3D(midpoint.x + normal.x * half_length,
                    midpoint.y + normal.y * half_length,
                    midpoint.z + normal.z * half_length),
            candidate_frame,
            warning,
        )
    except ValueError:
        return target_start, target_end, frame, "无法构造可靠局部椎间坐标，保留原椎间方向"


def resolve_spine_interspace_segments(
    localization_result: Dict[str, Any],
    intervals: Sequence[Tuple[str, str, str]],
) -> Tuple[List[CutSegment], List[str]]:
    """把脊柱相邻 label 解析成多个独立待切区间及其 warning。"""

    segments: List[CutSegment] = []
    warnings: List[str] = []
    image_info = localization_result["image"]
    fallback_bbox = localization_result.get("correction_scope_bbox")

    for group_name, upper_label, lower_label in intervals:
        upper_center = get_label_center(localization_result, upper_label)
        lower_center = get_label_center(localization_result, lower_label)
        if upper_center is None or lower_center is None:
            warnings.append(f"跳过脊柱区间 {group_name}: 缺少端点 label")
            continue

        filtered_annotations = _filtered_spine_annotations(localization_result)
        quadrilateral_data_present = any(
            isinstance(filtered_annotations.get(label_name), dict)
            and "sagittal_body_quadrilateral" in filtered_annotations[label_name]
            for label_name in (upper_label, lower_label)
        )
        prism_geometry = _resolve_interspace_prism_geometry(
            localization_result,
            upper_label,
            lower_label,
        )
        if quadrilateral_data_present:
            face_geometry = prism_geometry
            using_prism_geometry = prism_geometry is not None
        else:
            face_geometry = _resolve_interspace_face_geometry(
                localization_result,
                upper_label,
                lower_label,
            )
            using_prism_geometry = False
        face_data_present = quadrilateral_data_present or any(
            isinstance(filtered_annotations.get(label_name), dict)
            and "sagittal_body_faces" in filtered_annotations[label_name]
            for label_name in (upper_label, lower_label)
        )
        invalid_face_interval = False
        if face_geometry is not None:
            candidate_start, candidate_end, candidate_frame = face_geometry
            invalid_face_interval = _dot(
                subtract_points(candidate_end, candidate_start),
                candidate_frame.normal,
            ) <= 1e-6
            if invalid_face_interval:
                face_geometry = None
                using_prism_geometry = False
            elif using_prism_geometry:
                quality_warning = _interspace_sidewall_geometry_warning(
                    localization_result, upper_label, lower_label, candidate_frame,
                )
                if quality_warning:
                    # 已确认是残块侧壁，不能再用同一坏标签的 bbox/质心
                    # 伪造另一套间隙，也不能把切面强行压回固定横状位。
                    warnings.append(f"跳过脊柱区间 {group_name}: {quality_warning}")
                    continue
        upper_bbox = _label_bbox(localization_result, upper_label)
        lower_bbox = _label_bbox(localization_result, lower_label)
        if upper_bbox is not None and lower_bbox is not None:
            target_region_bbox = union_bboxes(upper_bbox, lower_bbox)
            if face_geometry is not None:
                target_start, target_end, frame = face_geometry
                target_start, target_end, frame, pose_warning = (
                    _refine_interspace_symmetry_geometry(
                        localization_result, upper_label, lower_label,
                        target_start, target_end, frame,
                    )
                )
                if pose_warning:
                    warnings.append(f"脊柱区间 {group_name} {pose_warning}")
                if using_prism_geometry:
                    (
                        target_start,
                        target_end,
                        position_warning,
                    ) = _translate_interspace_to_body_pair(
                        localization_result,
                        group_name,
                        target_start,
                        target_end,
                        frame,
                    )
                    if position_warning is not None:
                        warnings.append(position_warning)
            else:
                if face_data_present or invalid_face_interval:
                    warnings.append(
                        f"脊柱区间 {group_name} 四棱柱端面几何无效，已回退到 bbox"
                    )
                upper_foot, _ = _bbox_head_foot_boundary_points(
                    upper_bbox,
                    target_region_bbox,
                    image_info,
                )
                _, lower_head = _bbox_head_foot_boundary_points(
                    lower_bbox,
                    target_region_bbox,
                    image_info,
                )
                target_start, target_end = upper_foot, lower_head
                frame = None
        else:
            if fallback_bbox is None:
                warnings.append(f"跳过脊柱区间 {group_name}: 缺少可用 bbox")
                continue
            using_prism_geometry = False
            target_region_bbox = dict(fallback_bbox)
            target_start, target_end = upper_center, lower_center
            frame = None
            warnings.append(
                f"脊柱区间 {group_name} 缺少椎体 bbox，已回退到质心范围"
            )

        # 方向估计不能把原先没有有效头脚间隙的 bbox 重新变成有效区间。
        if frame is None and target_start.z > target_end.z:
            target_start, target_end, frame, pose_warning = (
                _refine_interspace_symmetry_geometry(
                    localization_result, upper_label, lower_label,
                    target_start, target_end, frame,
                )
            )
            if pose_warning:
                warnings.append(f"脊柱区间 {group_name} {pose_warning}")

        if frame is not None:
            target_vector = subtract_points(target_end, target_start)
            valid_interval = _dot(target_vector, frame.normal) > 1e-6
        else:
            valid_interval = target_start.z > target_end.z
        if not valid_interval:
            warnings.append(
                f"跳过脊柱区间 {group_name}: 相向边界没有形成有效头脚区间"
            )
            continue
        segments.append(CutSegment(
            group_name=group_name,
            target_start=target_start,
            target_end=target_end,
            target_region_bbox=target_region_bbox,
            source_labels=(upper_label, lower_label),
            frame=frame,
            recenter_target_in_plane=not using_prism_geometry,
        ))

    return segments, warnings
