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

    centers, center_warnings = _get_spine_annotation_centers(
        localization_result,
        label_names,
    )

    if len(centers) < 3:
        return None, center_warnings + [
            f"CPR 降级: 有效椎体少于 3 个，当前 {len(centers)} 个"
        ]

    image_info = localization_result["image"]
    source_labels = [label_name for label_name, _ in centers]
    annotation_centers = [center for _, center in centers]
    center = _midpoint(annotation_centers)
    anchor_points = _build_annotation_centroid_path(
        localization_result,
        centers,
    )
    # 阶段四只保存“首端面 + 全部标注质心 + 末端面”的稀疏锚点折线。
    # 最终图像行由阶段五按弧长加密；这里不再做额外平滑，避免端部切线退化。
    curve_points = anchor_points
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
        warnings=center_warnings,
    ), center_warnings


def _get_spine_annotation_centers(
    localization_result: Dict[str, Any],
    label_names: Iterable[str],
) -> Tuple[List[Tuple[str, Point3D]], List[str]]:
    """读取过滤标注质心；旧结果缺少过滤结果时回退到全量质心。"""

    filtered_centroids = (
        localization_result.get("spine_geometry", {})
        .get("filtered_annotation_centroids", {})
    )
    centers: list[tuple[str, Point3D]] = []
    warnings: list[str] = []
    has_filtered_geometry = bool(filtered_centroids)
    coords = localization_result.get("coords", {})
    for requested_label_name in label_names:
        label_name = _resolve_available_spine_label(
            requested_label_name,
            filtered_centroids,
            coords,
        )
        annotation = filtered_centroids.get(label_name)
        physical = annotation.get("centroid_physical") if annotation else None
        if annotation and annotation.get("found") and physical:
            centers.append((
                label_name,
                Point3D(
                    float(physical["x"]),
                    float(physical["y"]),
                    float(physical["z"]),
                ),
            ))
            continue

        full_center = get_label_center(localization_result, label_name)
        if full_center is None:
            continue
        if has_filtered_geometry:
            warnings.append(
                f"{label_name} 缺少过滤标注质心，CPR 不使用该椎体"
            )
            continue
        centers.append((label_name, full_center))

    return centers, warnings


def _resolve_available_spine_label(
    requested_label_name: str,
    filtered_centroids: Dict[str, Any],
    coords: Dict[str, Any],
) -> str:
    """在当前定位结果中解析模型标签别名。"""

    available_labels = set(filtered_centroids) | set(coords)
    if requested_label_name in available_labels:
        return requested_label_name
    if requested_label_name == "vertebrae_S1" and "sacrum" in available_labels:
        return "sacrum"
    return requested_label_name


def _build_annotation_centroid_path(
    localization_result: Dict[str, Any],
    centers: List[Tuple[str, Point3D]],
) -> List[Point3D]:
    """在全部过滤标注质心首尾补入主体端面中点。"""

    centroid_path = [center for _, center in centers]
    if len(centroid_path) < 2:
        return centroid_path

    filtered_centroids = (
        localization_result.get("spine_geometry", {})
        .get("filtered_annotation_centroids", {})
    )
    if not filtered_centroids:
        return centroid_path

    first_direction = normalize_vector(
        subtract_points(centroid_path[1], centroid_path[0])
    )
    last_direction = normalize_vector(
        subtract_points(centroid_path[-1], centroid_path[-2])
    )
    first_face = _resolve_endpoint_face_center(
        filtered_centroids.get(centers[0][0]),
        centroid_path[0],
        first_direction,
        toward_start=True,
    )
    last_face = _resolve_endpoint_face_center(
        filtered_centroids.get(centers[-1][0]),
        centroid_path[-1],
        last_direction,
        toward_start=False,
    )

    path = list(centroid_path)
    if first_face is not None:
        path.insert(0, first_face)
    if last_face is not None:
        path.append(last_face)
    return path


def _resolve_endpoint_face_center(
    annotation: Dict[str, Any] | None,
    centroid: Point3D,
    reference_direction: Point3D,
    toward_start: bool,
) -> Point3D | None:
    """按相邻质心连线方向选取首/末椎体的主体端面中心。"""

    if not annotation or not annotation.get("found"):
        return None

    body_face = _resolve_sagittal_body_face_center(
        annotation,
        reference_direction,
        toward_start,
    )
    if body_face is not None:
        return body_face

    candidates = annotation.get(
        "local_superior_inferior_face_candidates"
    )
    if isinstance(candidates, list):
        extension_direction = (
            _scale(reference_direction, -1.0)
            if toward_start
            else reference_direction
        )
        angle_tolerance_cosine = math.cos(math.radians(25.0))
        aligned_candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            physical = candidate.get("physical")
            if not isinstance(physical, dict):
                continue
            if candidate.get("face_valid", True) is False:
                continue
            if not all(axis in physical for axis in ("x", "y", "z")):
                continue
            point = Point3D(
                float(physical["x"]),
                float(physical["y"]),
                float(physical["z"]),
            )
            delta = subtract_points(point, centroid)
            distance = vector_length(delta)
            if distance <= 1e-6:
                continue
            alignment = _dot(
                normalize_vector(delta),
                extension_direction,
            )
            if alignment <= 0.0:
                continue
            extension_distance = _dot(delta, extension_direction)
            aligned_candidates.append((
                alignment,
                extension_distance,
                distance,
                _candidate_point_count(candidate),
                point,
            ))
        if aligned_candidates:
            angle_compatible_candidates = [
                candidate
                for candidate in aligned_candidates
                if candidate[0] >= angle_tolerance_cosine
            ]
            if angle_compatible_candidates:
                stable_candidates = _filter_sparse_endpoint_candidates(
                    angle_compatible_candidates
                )
                # 角度在容差内时，优先取沿质心连线方向最远的真实端面。
                return max(
                    stable_candidates,
                    key=lambda item: (item[1], item[0], item[2]),
                )[4]
            return max(
                aligned_candidates,
                key=lambda item: (item[0], item[1], item[2]),
            )[4]

    face_centroids = annotation.get("local_superior_inferior_face_centroids")
    if not isinstance(face_centroids, dict):
        return None
    candidates = []
    for key in ("min", "max"):
        physical = face_centroids.get(key)
        if not isinstance(physical, dict):
            continue
        if not all(axis in physical for axis in ("x", "y", "z")):
            continue
        point = Point3D(
            float(physical["x"]),
            float(physical["y"]),
            float(physical["z"]),
        )
        candidates.append((
            _dot(subtract_points(point, centroid), reference_direction),
            point,
        ))
    if len(candidates) != 2:
        return None
    return min(candidates, key=lambda item: item[0])[1] if toward_start else max(
        candidates,
        key=lambda item: item[0],
    )[1]


def _resolve_sagittal_body_face_center(
    annotation: Dict[str, Any],
    reference_direction: Point3D,
    toward_start: bool,
) -> Point3D | None:
    """在主体四面中选择质心入射面的对侧端面。"""

    faces = annotation.get("sagittal_body_faces")
    if not isinstance(faces, list):
        return None

    extension_direction = (
        _scale(reference_direction, -1.0)
        if toward_start
        else reference_direction
    )
    parsed_faces = []
    for face in faces:
        if not isinstance(face, dict):
            continue
        center = face.get("center")
        normal = face.get("normal")
        if not isinstance(center, dict) or not isinstance(normal, dict):
            continue
        if not all(axis in center for axis in ("x", "y", "z")):
            continue
        if not all(axis in normal for axis in ("x", "y", "z")):
            continue
        axis_index = face.get("axis_index")
        side = face.get("side")
        if side not in {"min", "max"}:
            continue
        try:
            axis_index = int(axis_index)
        except (TypeError, ValueError):
            continue
        normal_vector = normalize_vector(Point3D(
            float(normal["x"]),
            float(normal["y"]),
            float(normal["z"]),
        ))
        incoming_alignment = _dot(normal_vector, extension_direction)
        point_count = face.get("point_count")
        try:
            support = int(point_count) if point_count is not None else 0
        except (TypeError, ValueError):
            support = 0
        parsed_faces.append((
            incoming_alignment,
            support,
            axis_index,
            str(side),
            Point3D(
                float(center["x"]),
                float(center["y"]),
                float(center["z"]),
            ),
        ))

    if not parsed_faces:
        return None
    # 面法线和延伸方向最相反的面，是相邻质心线进入椎体的那一面；
    # CPR 端点要落在这条入射方向对应的对侧大面中心，而不是角点或斜边。
    entrance = min(parsed_faces, key=lambda item: (item[0], -item[1]))
    opposite_side = "max" if entrance[3] == "min" else "min"
    opposite_faces = [
        face
        for face in parsed_faces
        if face[2] == entrance[2] and face[3] == opposite_side
    ]
    if opposite_faces:
        return max(opposite_faces, key=lambda item: (item[0], item[1]))[4]
    return max(parsed_faces, key=lambda item: (item[0], item[1]))[4]


def _candidate_point_count(candidate: Dict[str, Any]) -> int | None:
    point_count = candidate.get("point_count")
    if point_count is None:
        return None
    try:
        return int(point_count)
    except (TypeError, ValueError):
        return None


def _filter_sparse_endpoint_candidates(candidates: List[Tuple]) -> List[Tuple]:
    """剔除终末端只有少量体素支撑的角点候选。"""

    point_counts = [
        candidate[3]
        for candidate in candidates
        if isinstance(candidate[3], int) and candidate[3] > 0
    ]
    if not point_counts:
        return candidates

    min_supported_points = max(point_counts) * 0.25
    stable_candidates = [
        candidate
        for candidate in candidates
        if not isinstance(candidate[3], int)
        or candidate[3] >= min_supported_points
    ]
    return stable_candidates or candidates


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


def _resolve_offsets_mm(
    localization_result: Dict[str, Any],
    label_names: List[str],
    curve_points: List[Point3D],
    offset_axis: Point3D,
    slice_spacing: float,
) -> List[float]:
    """根据全量椎体 bbox 沿 CPR 堆叠轴的覆盖生成 offset。"""

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
    """用整体 bbox 在 CPR 横向显示方向上的投影宽度作为图像宽度。"""

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
