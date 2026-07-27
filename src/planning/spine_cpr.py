"""脊柱冠状位 CPR 规划。"""

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.models import ExtentMM, PlannedSlice, Point3D, SlicePlan
from src.models.enums import PlaneTypeEnum
from src.planning.geometry import (
    image_axis_direction,
    normalize_vector,
    subtract_points,
    vector_length,
    voxel_index_to_physical,
)
from src.planning.spine_common import get_label_center, union_bboxes


SPINE_CPR_LABELS = {
    "颈椎": [f"vertebrae_C{index}" for index in range(2, 8)],
    "胸椎": [f"vertebrae_T{index}" for index in range(1, 13)],
    "腰椎": [
        "vertebrae_L1",
        "vertebrae_L2",
        "vertebrae_L3",
        "vertebrae_L4",
        "vertebrae_L5",
        "vertebrae_S1",
    ],
}


def build_spine_sagittal_cpr_plan(
    request: Any,
    plane_config: Any,
    localization_result: Dict[str, Any],
    label_names: Iterable[str],
) -> Tuple[SlicePlan | None, List[str]]:
    """为脊柱冠状位生成 CPR plan；矢状位继续交给旧直线 MPR 逻辑。"""

    if plane_config.plane != PlaneTypeEnum.冠状位:
        return None, []

    centers: list[tuple[str, Point3D]] = []
    for label_name in label_names:
        center = get_label_center(localization_result, label_name)
        if center is not None:
            centers.append((label_name, center))

    if len(centers) < 3:
        return None, [
            f"CPR 降级: 有效椎体少于 3 个，当前 {len(centers)} 个"
        ]

    image_info = localization_result["image"]
    source_labels = [label_name for label_name, _ in centers]
    centroid_curve_points = _smooth_curve([center for _, center in centers])
    center = _midpoint(centroid_curve_points)
    curve_points = _extend_curve_to_endpoint_bboxes(
        localization_result,
        source_labels,
        centroid_curve_points,
    )
    curve_length = _polyline_length(curve_points)
    if curve_length <= 0.0:
        return None, ["CPR 降级: 中心曲线长度无效"]

    offset_axis = _offset_axis_for_plane(image_info, plane_config.plane)
    tangent_dir = normalize_vector(subtract_points(curve_points[-1], curve_points[0]))
    row_dir = normalize_vector(_cross(tangent_dir, offset_axis))
    offsets = _resolve_offsets_mm(
        localization_result,
        source_labels,
        curve_points,
        offset_axis,
        float(plane_config.slice_spacing),
    )
    slices = [
        PlannedSlice(
            index=index + 1,
            center=_add(center, _scale(offset_axis, offset_mm)),
            normal=offset_axis,
            row_dir=row_dir,
            col_dir=tangent_dir,
            offset_mm=offset_mm,
        )
        for index, offset_mm in enumerate(offsets)
    ]

    return SlicePlan(
        plan_id=f"{request.body_part.value}_CPR_{plane_config.plane.value}",
        body_part=request.body_part.value,
        cut_mode="cpr",
        plane=plane_config.plane.value,
        group_name="CPR",
        geometry_type="curved_stack",
        range_start=curve_points[0],
        range_end=curve_points[-1],
        path_points=curve_points,
        slices=slices,
        extent_mm=ExtentMM(
            width=_resolve_cpr_width_extent_mm(
                localization_result,
                source_labels,
                row_dir,
            ),
            height=curve_length,
        ),
        output_spacing_mm=Point3D(
            _minimum_image_spacing(image_info),
            _minimum_image_spacing(image_info),
            0.0,
        ),
        slice_thickness=plane_config.slice_thickness,
        slice_spacing=plane_config.slice_spacing,
        window_presets=list(plane_config.window_presets),
        source_labels=source_labels,
        curve_points=curve_points,
        curve_spacing_mm=_minimum_image_spacing(image_info),
        offset_axis=offset_axis,
        offsets_mm=offsets,
        target_start=curve_points[0],
        target_end=curve_points[-1],
        localizer_reference=center,
        localizer_plane="矢状位",
        confidence=1.0,
        warnings=[],
    ), []


def found_spine_label_names(
    localization_result: Dict[str, Any],
    label_names: Iterable[str],
) -> List[str]:
    """按解剖顺序返回当前定位结果里真正找到的脊柱 label。"""

    coords = localization_result.get("coords", {})
    return [
        label_name
        for label_name in label_names
        if coords.get(label_name, {}).get("found")
    ]


def spine_target_bbox(
    localization_result: Dict[str, Any],
    label_names: Iterable[str],
) -> Tuple[Dict[str, float] | None, List[str]]:
    """用指定椎体 label 的 bbox 并集限定当前脊柱目标段。"""

    coords = localization_result.get("coords", {})
    used_labels = []
    bboxes = []
    for label_name in label_names:
        coord = coords.get(label_name, {})
        bbox = coord.get("bbox")
        if not coord.get("found") or not bbox:
            continue
        used_labels.append(label_name)
        bboxes.append(dict(bbox))
    if not bboxes:
        return None, []
    return union_bboxes(*bboxes), used_labels


def _smooth_curve(points: List[Point3D]) -> List[Point3D]:
    """第一版采用保守三点平滑，偏离原质心超过 5mm 时保留原点。"""

    if len(points) < 3:
        return list(points)
    smoothed = [points[0]]
    for index in range(1, len(points) - 1):
        candidate = Point3D(
            (points[index - 1].x + points[index].x + points[index + 1].x) / 3.0,
            (points[index - 1].y + points[index].y + points[index + 1].y) / 3.0,
            (points[index - 1].z + points[index].z + points[index + 1].z) / 3.0,
        )
        drift = vector_length(subtract_points(candidate, points[index]))
        smoothed.append(points[index] if drift > 5.0 else candidate)
    smoothed.append(points[-1])
    return smoothed


def _extend_curve_to_endpoint_bboxes(
    localization_result: Dict[str, Any],
    label_names: List[str],
    curve_points: List[Point3D],
) -> List[Point3D]:
    """把 CPR 曲线首尾从各自质心沿头脚轴延到椎体端面。"""

    if len(curve_points) < 2 or len(label_names) < 2:
        return list(curve_points)

    superior_inferior_axis = _superior_inferior_axis(localization_result)
    start_point = _endpoint_boundary_point(
        localization_result,
        label_names[0],
        curve_points[0],
        superior_inferior_axis,
        choose_minimum=True,
    )
    end_point = _endpoint_boundary_point(
        localization_result,
        label_names[-1],
        curve_points[-1],
        superior_inferior_axis,
        choose_minimum=False,
    )
    return [start_point, *curve_points, end_point]


def _superior_inferior_axis(localization_result: Dict[str, Any]) -> Point3D:
    """优先使用阶段三解剖头脚轴，缺失时回退到图像 z 轴。"""

    axis = (
        localization_result
        .get("anatomical_geometry", {})
        .get("global_frame", {})
        .get("superior_inferior_axis")
    )
    if axis:
        return normalize_vector(Point3D(
            float(axis["x"]),
            float(axis["y"]),
            float(axis["z"]),
        ))
    return normalize_vector(image_axis_direction(localization_result["image"], 2))


def _endpoint_boundary_point(
    localization_result: Dict[str, Any],
    label_name: str,
    center_point: Point3D,
    axis_dir: Point3D,
    choose_minimum: bool,
) -> Point3D:
    """沿给定头脚轴取首末椎体 bbox 的外侧端面点。"""

    bbox = (
        localization_result
        .get("coords", {})
        .get(label_name, {})
        .get("bbox")
    )
    if not bbox:
        return center_point

    center_projection = _dot(center_point, axis_dir)
    projections = [
        _dot(point, axis_dir) - center_projection
        for point in _bbox_corners(bbox, localization_result["image"])
    ]
    boundary_delta = (
        max(projections)
        if choose_minimum
        else min(projections)
    )
    return _add(center_point, _scale(axis_dir, boundary_delta))


def _resolve_offsets_mm(
    localization_result: Dict[str, Any],
    label_names: List[str],
    curve_points: List[Point3D],
    offset_axis: Point3D,
    slice_spacing: float,
) -> List[float]:
    """根据椎体 bbox 在左右方向的覆盖生成从左到右的 offset。"""

    spacing = max(float(slice_spacing), 1e-6)
    center_projection = float(np.median([
        _dot(point, offset_axis)
        for point in curve_points
    ]))
    projections: list[float] = []
    for label_name in label_names:
        bbox = (
            localization_result
            .get("coords", {})
            .get(label_name, {})
            .get("bbox")
        )
        if not bbox:
            continue
        projections.extend(
            _dot(point, offset_axis) - center_projection
            for point in _bbox_corners(bbox, localization_result["image"])
        )

    if not projections:
        projections = [-spacing, 0.0, spacing]

    start = math.floor(min(projections) / spacing) * spacing
    end = math.ceil(max(projections) / spacing) * spacing
    offsets = []
    current = start
    while current <= end + spacing * 1e-6:
        offsets.append(round(current, 6))
        current += spacing
    if not any(abs(offset) <= 1e-6 for offset in offsets):
        offsets.append(0.0)
        offsets.sort()
    return offsets


def _bbox_corners(bbox: Dict[str, float], image_info: Dict[str, Any]) -> List[Point3D]:
    """把体素 bbox 八个角转换成物理点。"""

    xs = [float(bbox["x_min"]), float(bbox["x_max"])]
    ys = [float(bbox["y_min"]), float(bbox["y_max"])]
    zs = [float(bbox["z_min"]), float(bbox["z_max"])]
    return [
        voxel_index_to_physical(Point3D(x, y, z), image_info)
        for x in xs
        for y in ys
        for z in zs
    ]


def _resolve_ap_extent_mm(
    localization_result: Dict[str, Any],
    row_dir: Point3D,
) -> float:
    """用整体 bbox 在前后显示方向上的投影宽度作为 CPR 图像宽度。"""

    bbox = localization_result.get("correction_scope_bbox")
    if not bbox:
        return 120.0
    corners = _bbox_corners(bbox, localization_result["image"])
    projections = [_dot(point, row_dir) for point in corners]
    extent = max(projections) - min(projections)
    return max(float(extent), _minimum_image_spacing(localization_result["image"]))


def _offset_axis_for_plane(
    image_info: Dict[str, Any],
    plane: PlaneTypeEnum,
) -> Point3D:
    """根据 CPR 输出状位选择曲面堆叠方向。"""

    if plane == PlaneTypeEnum.矢状位:
        return normalize_vector(image_axis_direction(image_info, 0))
    if plane == PlaneTypeEnum.冠状位:
        return normalize_vector(image_axis_direction(image_info, 1))
    raise ValueError(f"不支持的脊柱 CPR 状位: {plane.value}")


def _resolve_cpr_width_extent_mm(
    localization_result: Dict[str, Any],
    label_names: List[str],
    row_dir: Point3D,
) -> float:
    """按目标椎体 bbox 在 CPR 横向显示方向上的投影决定图像宽度。"""

    corners: list[Point3D] = []
    for label_name in label_names:
        bbox = (
            localization_result
            .get("coords", {})
            .get(label_name, {})
            .get("bbox")
        )
        if bbox:
            corners.extend(_bbox_corners(bbox, localization_result["image"]))
    if not corners:
        return _resolve_ap_extent_mm(localization_result, row_dir)
    projections = [_dot(point, row_dir) for point in corners]
    extent = max(projections) - min(projections)
    return max(float(extent), _minimum_image_spacing(localization_result["image"]))


def _polyline_length(points: List[Point3D]) -> float:
    """计算折线总长度。"""

    return sum(
        vector_length(subtract_points(current, previous))
        for previous, current in zip(points, points[1:])
    )


def _minimum_image_spacing(image_info: Dict[str, Any]) -> float:
    """读取图像最小体素间距，作为 CPR 默认像素间距。"""

    return min(float(value) for value in image_info["spacing_xyz"])


def _midpoint(points: List[Point3D]) -> Point3D:
    """返回曲线采样点物理中心。"""

    middle = np.mean([[point.x, point.y, point.z] for point in points], axis=0)
    return Point3D(float(middle[0]), float(middle[1]), float(middle[2]))


def _dot(left: Point3D, right: Point3D) -> float:
    """计算两个 Point3D 的点积。"""

    return left.x * right.x + left.y * right.y + left.z * right.z


def _scale(vector: Point3D, factor: float) -> Point3D:
    """缩放三维向量。"""

    return Point3D(vector.x * factor, vector.y * factor, vector.z * factor)


def _add(left: Point3D, right: Point3D) -> Point3D:
    """相加三维点或向量。"""

    return Point3D(left.x + right.x, left.y + right.y, left.z + right.z)


def _cross(left: Point3D, right: Point3D) -> Point3D:
    """计算三维叉积。"""

    return Point3D(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )
