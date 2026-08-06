"""脊柱冠状位 CPR 规划。"""

import math
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import SimpleITK as sitk

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
from src.planning.straight import SliceRangeExtension, resolve_slice_range_extension
from src.utils import DebugLog


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

SPINE_CONFIG_LABEL_ORDER = (
    [f"vertebrae_C{index}" for index in range(1, 8)]
    + [f"vertebrae_T{index}" for index in range(1, 13)]
    + [f"vertebrae_L{index}" for index in range(1, 6)]
    + ["vertebrae_S1"]
)

CPR_RANGE_MODE_FULL_LABEL = "full_label"
CPR_RANGE_MODE_BODY_PRISM = "body_prism"


def build_spine_sagittal_cpr_plan(
    request: Any,
    plane_config: Any,
    localization_result: Dict[str, Any],
    label_names: Iterable[str],
) -> Tuple[SlicePlan | None, List[str]]:
    """兼容旧入口：返回当前配置生成的第一套 CPR plan。"""

    plans, warnings = build_spine_sagittal_cpr_plans(
        request=request,
        plane_config=plane_config,
        localization_result=localization_result,
        label_names=label_names,
    )
    return (plans[0] if plans else None), warnings


def build_spine_sagittal_cpr_plans(
    request: Any,
    plane_config: Any,
    localization_result: Dict[str, Any],
    label_names: Iterable[str],
) -> Tuple[List[SlicePlan], List[str]]:
    """为脊柱冠状位生成 CPR plan；矢状位继续交给旧直线 MPR 逻辑。"""

    perf_start = time.perf_counter()
    if plane_config.plane != PlaneTypeEnum.冠状位:
        return [], []

    centers, center_warnings = _get_spine_annotation_centers(
        localization_result,
        label_names,
    )

    if len(centers) < 3:
        return [], center_warnings + [
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
    curve_points = _smooth_cpr_centroid_path_with_straight_endpoints(
        anchor_points,
        annotation_centers,
        _minimum_image_spacing(image_info),
        normalize_vector(image_axis_direction(image_info, 2)),
    )
    curve_points = _extend_curve_endpoints_to_terminal_raw_labels(
        localization_result,
        source_labels,
        curve_points,
    )
    curve_length = _polyline_length(curve_points)
    if curve_length <= 0.0:
        return [], ["CPR 降级: 中心曲线长度无效"]

    offset_axis = _offset_axis_for_plane(image_info, plane_config.plane)
    tangent_dir = normalize_vector(subtract_points(curve_points[-1], curve_points[0]))
    row_dir = normalize_vector(_cross(tangent_dir, offset_axis))

    range_modes = _configured_cpr_range_modes(localization_result)
    include_range_suffix = range_modes != ["full_label"]
    plans: list[SlicePlan] = []
    for range_mode in range_modes:
        offsets = _resolve_offsets_mm(
            localization_result,
            source_labels,
            curve_points,
            offset_axis,
            float(plane_config.slice_thickness),
            float(plane_config.slice_spacing),
            range_mode=range_mode,
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
        plan_id, group_name = _cpr_plan_identity(
            request.body_part.value,
            plane_config.plane.value,
            range_mode,
            include_range_suffix,
        )
        plans.append(SlicePlan(
            plan_id=plan_id,
            body_part=request.body_part.value,
            cut_mode="cpr",
            plane=plane_config.plane.value,
            group_name=group_name,
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
            cpr_range_mode=range_mode,
            target_start=curve_points[0],
            target_end=curve_points[-1],
            localizer_reference=center,
            localizer_plane="矢状位",
            confidence=1.0,
            warnings=center_warnings,
        ))

    DebugLog(
        "    [CPR规划性能] "
        f"body_part={request.body_part.value}, "
        f"plane={plane_config.plane.value}, "
        f"labels={len(source_labels)}, "
        f"curve_points={len(curve_points)}, "
        f"range_modes={range_modes}, "
        f"offset_count={[len(plan.offsets_mm or []) for plan in plans]}, "
        f"curve_length={curve_length:.3f}mm, "
        f"耗时={time.perf_counter() - perf_start:.3f}s"
    )

    return plans, center_warnings


def _configured_cpr_range_modes(
    localization_result: Dict[str, Any],
) -> List[str]:
    """从阶段三配置解析 CPR 范围模式。"""

    config = localization_result.get("spine_cpr_config")
    if not isinstance(config, dict):
        config = localization_result.get("spine_cpr", {})
    if not isinstance(config, dict):
        config = {}
    return (
        [CPR_RANGE_MODE_FULL_LABEL]
        if _config_bool(config.get("full_label"), default=True)
        else [CPR_RANGE_MODE_BODY_PRISM]
    )


def _config_bool(value: Any, default: bool) -> bool:
    """把配置中的布尔或常见字符串布尔值规整成 bool。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _cpr_plan_identity(
    body_part: str,
    plane: str,
    range_mode: str,
    include_range_suffix: bool,
) -> tuple[str, str]:
    """按范围模式生成稳定 plan_id 和输出组名。"""

    base_plan_id = f"{body_part}_CPR_{plane}"
    if not include_range_suffix:
        return base_plan_id, "CPR"
    if range_mode == CPR_RANGE_MODE_BODY_PRISM:
        return f"{base_plan_id}_body_prism", "CPR_主椎体"
    return f"{base_plan_id}_full_label", "CPR_整体范围"


def configured_spine_cpr_labels(
    body_part: str,
    localization_result: Dict[str, Any],
    fallback_labels: Iterable[str],
) -> List[str]:
    """按阶段三解析后的整切 start/end 配置生成 CPR 椎体列表。"""

    configured_labels = _configured_whole_range_spine_labels(
        localization_result,
    )
    if configured_labels is None:
        return list(fallback_labels)
    if body_part == "颈椎":
        configured_labels = [
            label_name
            for label_name in configured_labels
            if label_name != "vertebrae_C1"
        ]
    return configured_labels or list(fallback_labels)


def _configured_whole_range_spine_labels(
    localization_result: Dict[str, Any],
) -> List[str] | None:
    resolved_cut_config = localization_result.get("resolved_cut_config")
    if not isinstance(resolved_cut_config, dict):
        return None

    start_label = _first_configured_label(
        resolved_cut_config.get("start_labels")
    )
    end_label = _first_configured_label(
        resolved_cut_config.get("end_labels")
    )
    if start_label is None or end_label is None:
        return None

    try:
        start_index = SPINE_CONFIG_LABEL_ORDER.index(start_label)
        end_index = SPINE_CONFIG_LABEL_ORDER.index(end_label)
    except ValueError:
        return None

    if start_index <= end_index:
        return list(SPINE_CONFIG_LABEL_ORDER[start_index:end_index + 1])
    return list(reversed(SPINE_CONFIG_LABEL_ORDER[end_index:start_index + 1]))


def _first_configured_label(value: Any) -> str | None:
    if isinstance(value, list) and value:
        first_value = value[0]
        return str(first_value) if first_value else None
    if isinstance(value, str) and value:
        return value
    return None


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
        if annotation:
            has_prism_contract = "prism_found" in annotation
            prism_point = _parse_physical_point(
                annotation.get("prism_centroid_physical")
            )
            if annotation.get("prism_found") and prism_point is not None:
                centers.append((label_name, prism_point))
                continue

            body_point = _parse_physical_point(
                annotation.get("centroid_physical")
            )
            if annotation.get("found") and body_point is not None:
                if has_prism_contract:
                    warnings.append(
                        f"{label_name} 四棱柱生成失败，"
                        "CPR 回退到过滤主体质心"
                    )
                centers.append((label_name, body_point))
                continue
            warnings.append(
                f"{label_name} 缺少有效过滤标注质心，CPR 不使用该椎体"
            )
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


def _parse_physical_point(value: Any) -> Point3D | None:
    """把外部几何字段解析为三轴有限物理点。"""

    if not isinstance(value, dict):
        return None
    try:
        coordinates = tuple(float(value[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        return None
    return Point3D(*coordinates)


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
    left_right_axis = image_axis_direction(
        localization_result["image"],
        0,
    )
    first_face = _correct_endpoint_face_to_local_arc(
        first_face,
        centroid_path[0],
        centroid_path[1],
        centroid_path[2] if len(centroid_path) >= 3 else None,
        left_right_axis,
    )
    last_face = _correct_endpoint_face_to_local_arc(
        last_face,
        centroid_path[-1],
        centroid_path[-2],
        centroid_path[-3] if len(centroid_path) >= 3 else None,
        left_right_axis,
    )
    coords = localization_result.get("coords", {})
    original_segmentation = _read_original_segmentation(
        localization_result,
    )
    first_face = _extend_existing_endpoint_to_original_label_exit(
        first_face,
        centroid_path[0],
        coords.get(centers[0][0], {}).get("label_ids"),
        original_segmentation,
    )
    last_face = _extend_existing_endpoint_to_original_label_exit(
        last_face,
        centroid_path[-1],
        coords.get(centers[-1][0], {}).get("label_ids"),
        original_segmentation,
    )

    path = list(centroid_path)
    if (
        first_face is not None
        and vector_length(
            subtract_points(first_face, centroid_path[0])
        ) > 1e-6
    ):
        path.insert(0, first_face)
    if (
        last_face is not None
        and vector_length(
            subtract_points(last_face, centroid_path[-1])
        ) > 1e-6
    ):
        path.append(last_face)
    return path


def _read_original_segmentation(
    localization_result: Dict[str, Any],
) -> sitk.Image | None:
    """读取端点沿线检查所需的原始分割；不可用时安全跳过延伸。"""

    segmentation_path = localization_result.get(
        "seg_for_reconstruction_path"
    )
    if not segmentation_path:
        return None
    try:
        return sitk.ReadImage(str(segmentation_path))
    except (OSError, RuntimeError):
        return None


def _extend_existing_endpoint_to_original_label_exit(
    endpoint: Point3D | None,
    centroid: Point3D,
    label_ids: Any,
    segmentation: sitk.Image | None,
) -> Point3D | None:
    """沿已有端线延长到白线连续穿出原始标签的位置。"""

    if endpoint is None or segmentation is None:
        return endpoint

    endpoint_delta = subtract_points(endpoint, centroid)
    if vector_length(endpoint_delta) <= 1e-6:
        return endpoint

    if not isinstance(label_ids, (list, tuple, set)) or not label_ids:
        return endpoint
    try:
        target_label_ids = {int(label_id) for label_id in label_ids}
    except (TypeError, ValueError):
        return endpoint

    endpoint_direction = normalize_vector(endpoint_delta)
    spacing = [abs(float(value)) for value in segmentation.GetSpacing()]
    if not spacing or any(value <= 1e-6 for value in spacing):
        return endpoint

    if not _point_has_any_label(
        segmentation,
        endpoint,
        target_label_ids,
    ):
        return endpoint

    sample_step = min(spacing) * 0.5
    exit_clearance = math.sqrt(sum(value ** 2 for value in spacing))
    image_diagonal = math.sqrt(sum(
        ((size - 1) * axis_spacing) ** 2
        for size, axis_spacing in zip(segmentation.GetSize(), spacing)
    ))
    maximum_distance = image_diagonal + exit_clearance

    last_inside = endpoint
    outside_distance = 0.0
    distance = sample_step
    while distance <= maximum_distance + sample_step * 1e-6:
        sample_point = _add(
            endpoint,
            _scale(endpoint_direction, distance),
        )
        if _point_has_any_label(
            segmentation,
            sample_point,
            target_label_ids,
        ):
            last_inside = sample_point
            outside_distance = 0.0
        else:
            outside_distance += sample_step
            if outside_distance >= exit_clearance:
                return last_inside
        distance += sample_step
    return endpoint


def _point_has_any_label(
    segmentation: sitk.Image,
    point: Point3D,
    label_ids: set[int],
) -> bool:
    """判断物理点的最近体素是否属于目标原始标签。"""

    try:
        index = segmentation.TransformPhysicalPointToIndex(
            (point.x, point.y, point.z)
        )
    except RuntimeError:
        return False
    if any(
        coordinate < 0 or coordinate >= size
        for coordinate, size in zip(index, segmentation.GetSize())
    ):
        return False
    return int(segmentation.GetPixel(index)) in label_ids


def _correct_endpoint_face_to_local_arc(
    face: Point3D | None,
    centroid: Point3D,
    adjacent_centroid: Point3D,
    interior_centroid: Point3D | None,
    left_right_axis: Point3D,
) -> Point3D | None:
    """把越出局部质心弧线方向区间的端面最短投影到合法边界。"""

    if face is None:
        return None

    outward_segment = _project_away_from_axis(
        subtract_points(centroid, adjacent_centroid),
        left_right_axis,
    )
    if vector_length(outward_segment) <= 1e-6:
        return face

    outward = normalize_vector(outward_segment)
    face_delta = subtract_points(face, centroid)
    sagittal_face_delta = _project_away_from_axis(
        face_delta,
        left_right_axis,
    )
    outward_component = _dot(sagittal_face_delta, outward)
    target_angle = None
    if (
        interior_centroid is not None
        and vector_length(sagittal_face_delta) > 1e-6
    ):
        inner_segment = _project_away_from_axis(
            subtract_points(adjacent_centroid, interior_centroid),
            left_right_axis,
        )
        if vector_length(inner_segment) > 1e-6:
            left_right = normalize_vector(left_right_axis)
            inner_angle = _signed_angle_around_axis(
                normalize_vector(inner_segment),
                outward,
                left_right,
            )
            face_angle = _signed_angle_around_axis(
                outward,
                normalize_vector(sagittal_face_delta),
                left_right,
            )
            minimum_angle = min(0.0, inner_angle)
            maximum_angle = max(0.0, inner_angle)
            if face_angle < minimum_angle - 1e-6:
                target_angle = 0.0
            elif face_angle > maximum_angle + 1e-6:
                target_angle = 0.0

    if outward_component < -1e-6:
        target_angle = 0.0
    if target_angle is None:
        return face

    left_right_delta = subtract_points(face_delta, sagittal_face_delta)
    boundary_ray = normalize_vector(
        _rotate_around_axis(
            outward,
            normalize_vector(left_right_axis),
            target_angle,
        )
    )
    projected_sagittal_delta = _scale(
        boundary_ray,
        max(_dot(sagittal_face_delta, boundary_ray), 0.0),
    )
    return _add(
        centroid,
        _add(left_right_delta, projected_sagittal_delta),
    )


def _project_away_from_axis(
    vector: Point3D,
    axis: Point3D,
) -> Point3D:
    """移除向量在指定轴上的分量。"""

    unit_axis = normalize_vector(axis)
    return subtract_points(
        vector,
        _scale(unit_axis, _dot(vector, unit_axis)),
    )


def _signed_angle_around_axis(
    first: Point3D,
    second: Point3D,
    axis: Point3D,
) -> float:
    """返回从 first 到 second 绕 axis 的带符号角。"""

    return math.atan2(
        _dot(_cross(first, second), axis),
        _dot(first, second),
    )


def _rotate_around_axis(
    vector: Point3D,
    axis: Point3D,
    angle_radians: float,
) -> Point3D:
    """使用 Rodrigues 公式把向量绕单位轴旋转指定角度。"""

    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return _add(
        _add(
            _scale(vector, cosine),
            _scale(_cross(axis, vector), sine),
        ),
        _scale(
            axis,
            _dot(axis, vector) * (1.0 - cosine),
        ),
    )


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
    slice_thickness: float,
    slice_spacing: float,
    range_mode: str = CPR_RANGE_MODE_FULL_LABEL,
) -> List[float]:
    """按原始椎体标注体素外缘和层厚生成 CPR offset。"""

    if range_mode == CPR_RANGE_MODE_BODY_PRISM:
        filtered_interval = _filtered_annotation_offset_interval_mm(
            localization_result,
            label_names,
            curve_points,
            offset_axis,
        )
        if filtered_interval is not None:
            filtered_interval = _extend_cpr_offset_interval(
                localization_result,
                filtered_interval,
            )
            return _offsets_covering_interval(
                filtered_interval[0],
                filtered_interval[1],
                slice_thickness,
                slice_spacing,
            )

    raw_interval = _raw_label_offset_interval_mm(
        localization_result,
        label_names,
        curve_points,
        offset_axis,
    )
    if raw_interval is not None:
        raw_interval = _extend_cpr_offset_interval(
            localization_result,
            raw_interval,
        )
        return _offsets_covering_interval(
            raw_interval[0],
            raw_interval[1],
            slice_thickness,
            slice_spacing,
        )
    return _bbox_offsets_mm(
        localization_result,
        label_names,
        curve_points,
        offset_axis,
        slice_thickness,
        slice_spacing,
    )


def _filtered_annotation_offset_interval_mm(
    localization_result: Dict[str, Any],
    label_names: List[str],
    curve_points: List[Point3D],
    offset_axis: Point3D,
) -> Tuple[float, float] | None:
    """用阶段三过滤后的四棱柱/主体 bbox 计算 CPR offset 区间。"""

    annotations = (
        localization_result
        .get("spine_geometry", {})
        .get("filtered_annotation_centroids", {})
    )
    if not isinstance(annotations, dict) or not annotations:
        return None

    curve = np.asarray([
        [point.x, point.y, point.z]
        for point in curve_points
    ], dtype=float)
    if len(curve) < 2:
        return None
    axis = np.asarray(
        [offset_axis.x, offset_axis.y, offset_axis.z],
        dtype=float,
    )
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-6 or not math.isfinite(axis_length):
        return None
    axis = axis / axis_length
    curve_without_axis = curve - np.outer(curve @ axis, axis)

    projections: list[float] = []
    for label_name in label_names:
        annotation = annotations.get(label_name)
        if not isinstance(annotation, dict) or not annotation.get("found"):
            continue
        annotation_points = _filtered_annotation_support_points(
            localization_result,
            annotation,
        )
        if annotation_points is None:
            continue
        points_without_axis = annotation_points - np.outer(
            annotation_points @ axis,
            axis,
        )
        nearest_indices = _nearest_curve_indices(
            points_without_axis,
            curve_without_axis,
        )
        projections.extend(
            float(value)
            for value in np.sum(
                (annotation_points - curve[nearest_indices]) * axis,
                axis=1,
            )
        )
    if not projections:
        return None
    return min(projections), max(projections)


def _filtered_annotation_support_points(
    localization_result: Dict[str, Any],
    annotation: Dict[str, Any],
) -> np.ndarray | None:
    """返回主体四棱柱的有方向支撑点，旧结果缺字段时回退 bbox。"""

    sagittal_geometry = annotation.get("sagittal_body_quadrilateral")
    vertices = (
        sagittal_geometry.get("vertices")
        if isinstance(sagittal_geometry, dict)
        else None
    )
    extent = (
        sagittal_geometry.get("left_right_extent_mm")
        if isinstance(sagittal_geometry, dict)
        else None
    )
    if isinstance(vertices, list) and len(vertices) >= 4 and isinstance(extent, dict):
        try:
            sagittal_points = np.asarray([
                [
                    float(point["x"]),
                    float(point["y"]),
                    float(point["z"]),
                ]
                for point in vertices
            ], dtype=float)
            left_right_min = float(extent["min"])
            left_right_max = float(extent["max"])
            direction = np.asarray(
                localization_result["image"].get("direction"),
                dtype=float,
            ).reshape(3, 3)
            left_right_axis = direction[:, 0]
            axis_length = float(np.linalg.norm(left_right_axis))
            if (
                sagittal_points.shape[1] == 3
                and np.all(np.isfinite(sagittal_points))
                and math.isfinite(left_right_min)
                and math.isfinite(left_right_max)
                and axis_length > 1e-6
                and math.isfinite(axis_length)
            ):
                left_right_axis /= axis_length
                return np.vstack([
                    sagittal_points + left_right_axis * left_right_min,
                    sagittal_points + left_right_axis * left_right_max,
                ])
        except (KeyError, TypeError, ValueError):
            pass

    bbox = annotation.get("prism_bbox") if annotation.get("prism_found") else None
    if not bbox:
        bbox = annotation.get("body_bbox")
    if not bbox:
        return None
    return np.asarray([
        [point.x, point.y, point.z]
        for point in _bbox_corners(bbox, localization_result["image"])
    ], dtype=float)


def _extend_cpr_offset_interval(
    localization_result: Dict[str, Any],
    interval: Tuple[float, float],
) -> Tuple[float, float]:
    """用 whole_range 冠状位扩展增加 CPR offset 覆盖范围。"""

    extension = _resolve_cpr_offset_range_extension(localization_result)
    return (
        float(interval[0]) - extension.start_mm,
        float(interval[1]) + extension.end_mm,
    )


def _resolve_cpr_offset_range_extension(
    localization_result: Dict[str, Any],
) -> SliceRangeExtension:
    """CPR 冠状位定位竖线左右范围沿 offset 轴扩展。"""

    return resolve_slice_range_extension(
        localization_result,
        "whole_range",
        PlaneTypeEnum.冠状位,
    )


def _raw_label_offset_interval_mm(
    localization_result: Dict[str, Any],
    label_names: List[str],
    curve_points: List[Point3D],
    offset_axis: Point3D,
) -> Tuple[float, float] | None:
    """计算原始标签体素相对现有 CPR 中线的前后偏移区间。"""

    perf_start = time.perf_counter()
    segmentation = _read_original_segmentation(localization_result)
    if segmentation is None:
        return None
    label_ids = _configured_label_ids(localization_result, label_names)
    if not label_ids:
        return None

    curve = np.asarray([
        [point.x, point.y, point.z]
        for point in curve_points
    ], dtype=float)
    if len(curve) < 2:
        return None
    axis = np.asarray(
        [offset_axis.x, offset_axis.y, offset_axis.z],
        dtype=float,
    )
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-6 or not math.isfinite(axis_length):
        return None
    axis = axis / axis_length

    mask_points = _raw_label_physical_points(segmentation, label_ids)
    if mask_points.size == 0:
        return None

    nearest_start = time.perf_counter()
    curve_without_axis = curve - np.outer(curve @ axis, axis)
    points_without_axis = mask_points - np.outer(mask_points @ axis, axis)
    nearest_indices = _nearest_curve_indices(
        points_without_axis,
        curve_without_axis,
    )
    nearest_elapsed = time.perf_counter() - nearest_start
    offsets = np.sum((mask_points - curve[nearest_indices]) * axis, axis=1)
    if offsets.size == 0:
        return None

    safety_margin = _voxel_safety_margin_mm(segmentation)
    interval = (
        float(np.min(offsets) - safety_margin),
        float(np.max(offsets) + safety_margin),
    )
    DebugLog(
        "    [CPR规划性能] 原始标注offset区间: "
        f"labels={sorted(label_ids)}, "
        f"raw_points={len(mask_points)}, "
        f"curve_points={len(curve)}, "
        f"nearest耗时={nearest_elapsed:.3f}s, "
        f"interval=[{interval[0]:.3f}, {interval[1]:.3f}]mm, "
        f"总耗时={time.perf_counter() - perf_start:.3f}s"
    )
    return interval


def _extend_curve_endpoints_to_terminal_raw_labels(
    localization_result: Dict[str, Any],
    label_names: List[str],
    curve_points: List[Point3D],
) -> List[Point3D]:
    """沿当前首尾切线延长 CPR 中线到终端原始标签投影外缘。"""

    perf_start = time.perf_counter()
    if len(curve_points) < 2 or not label_names:
        return list(curve_points)
    segmentation = _read_original_segmentation(localization_result)
    if segmentation is None:
        return list(curve_points)

    extended = list(curve_points)
    start_ids = _configured_label_ids(localization_result, [label_names[0]])
    end_ids = _configured_label_ids(localization_result, [label_names[-1]])
    start_endpoint_index = _endpoint_index_nearest_label(
        extended,
        segmentation,
        start_ids,
    )
    end_endpoint_index = _endpoint_index_nearest_label(
        extended,
        segmentation,
        end_ids,
    )
    if (
        start_endpoint_index is not None
        and end_endpoint_index is not None
        and start_endpoint_index == end_endpoint_index
    ):
        start_endpoint_index = 0
        end_endpoint_index = -1

    if start_ids and start_endpoint_index is not None:
        extended[start_endpoint_index] = _extend_endpoint_to_label_projection(
            endpoint=extended[start_endpoint_index],
            adjacent=_adjacent_endpoint_point(extended, start_endpoint_index),
            segmentation=segmentation,
            label_ids=start_ids,
        )
    if end_ids and end_endpoint_index is not None:
        extended[end_endpoint_index] = _extend_endpoint_to_label_projection(
            endpoint=extended[end_endpoint_index],
            adjacent=_adjacent_endpoint_point(extended, end_endpoint_index),
            segmentation=segmentation,
            label_ids=end_ids,
        )
    DebugLog(
        "    [CPR规划性能] 端点延伸到原始标注: "
        f"start_labels={sorted(start_ids)}, "
        f"end_labels={sorted(end_ids)}, "
        f"start_endpoint_index={start_endpoint_index}, "
        f"end_endpoint_index={end_endpoint_index}, "
        f"耗时={time.perf_counter() - perf_start:.3f}s"
    )
    return extended


def _endpoint_index_nearest_label(
    curve_points: List[Point3D],
    segmentation: sitk.Image,
    label_ids: set[int],
) -> int | None:
    """返回距离指定终端原始标签最近的曲线端点下标。"""

    if not label_ids:
        return None
    label_points = _raw_label_physical_points(segmentation, label_ids)
    if label_points.size == 0:
        return None
    endpoint_arrays = [
        np.asarray(
            [curve_points[0].x, curve_points[0].y, curve_points[0].z],
            dtype=float,
        ),
        np.asarray(
            [curve_points[-1].x, curve_points[-1].y, curve_points[-1].z],
            dtype=float,
        ),
    ]
    mean_distances = [
        float(np.mean(np.linalg.norm(label_points - endpoint, axis=1)))
        for endpoint in endpoint_arrays
    ]
    return 0 if mean_distances[0] <= mean_distances[1] else -1


def _adjacent_endpoint_point(
    curve_points: List[Point3D],
    endpoint_index: int,
) -> Point3D:
    """返回指定首/尾端点的相邻曲线点。"""

    return curve_points[1] if endpoint_index == 0 else curve_points[-2]


def _extend_endpoint_to_label_projection(
    endpoint: Point3D,
    adjacent: Point3D,
    segmentation: sitk.Image,
    label_ids: set[int],
) -> Point3D:
    """把端点沿远离相邻点的方向延伸到标签体素投影外缘。"""

    outward_delta = subtract_points(endpoint, adjacent)
    if vector_length(outward_delta) <= 1e-6:
        return endpoint
    label_points = _raw_label_physical_points(segmentation, label_ids)
    if label_points.size == 0:
        return endpoint

    outward = normalize_vector(outward_delta)
    outward_array = np.asarray([outward.x, outward.y, outward.z], dtype=float)
    endpoint_array = np.asarray([endpoint.x, endpoint.y, endpoint.z], dtype=float)
    projections = (label_points - endpoint_array) @ outward_array
    max_extension = (
        float(np.max(projections))
        + _terminal_endpoint_padding_mm(segmentation)
    )
    if max_extension <= 1e-6 or not math.isfinite(max_extension):
        return endpoint
    return _add(endpoint, _scale(outward, max_extension))


def _terminal_endpoint_padding_mm(image: sitk.Image) -> float:
    """终端标签头足兜底额外多留一个最小体素间距，避免视觉上刚好擦边。"""

    spacing = [abs(float(value)) for value in image.GetSpacing()]
    return _voxel_safety_margin_mm(image) + min(spacing) * 1.5


def _configured_label_ids(
    localization_result: Dict[str, Any],
    label_names: List[str],
) -> set[int]:
    """收集当前 CPR 配置范围内真实存在的原始分割 label id。"""

    label_ids: set[int] = set()
    coords = localization_result.get("coords", {})
    for label_name in label_names:
        values = coords.get(label_name, {}).get("label_ids")
        if not isinstance(values, (list, tuple, set)):
            continue
        for value in values:
            try:
                label_ids.add(int(value))
            except (TypeError, ValueError):
                continue
    return label_ids


def _raw_label_physical_points(
    segmentation: sitk.Image,
    label_ids: set[int],
) -> np.ndarray:
    """返回原始标签体素中心的物理坐标数组。"""

    perf_start = time.perf_counter()
    array = sitk.GetArrayViewFromImage(segmentation)
    mask = np.isin(array, list(label_ids))
    indices_zyx = np.argwhere(mask)
    if indices_zyx.size == 0:
        DebugLog(
            "    [CPR规划性能] 读取原始标注体素: "
            f"labels={sorted(label_ids)}, voxels=0, "
            f"耗时={time.perf_counter() - perf_start:.3f}s"
        )
        return np.empty((0, 3), dtype=float)
    indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
    origin = np.asarray(segmentation.GetOrigin(), dtype=float)
    spacing = np.asarray(segmentation.GetSpacing(), dtype=float)
    direction = np.asarray(
        segmentation.GetDirection(),
        dtype=float,
    ).reshape(3, 3)
    points = origin + (indices_xyz * spacing) @ direction.T
    DebugLog(
        "    [CPR规划性能] 读取原始标注体素: "
        f"labels={sorted(label_ids)}, "
        f"voxels={len(points)}, "
        f"image_size={segmentation.GetSize()}, "
        f"耗时={time.perf_counter() - perf_start:.3f}s"
    )
    return points


def _nearest_curve_indices(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """分块查找每个原始体素点最近的 CPR 曲线采样点。"""

    nearest = np.empty((len(points),), dtype=int)
    chunk_size = 4096
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        distances = np.sum(
            (
                points[start:stop, np.newaxis, :]
                - curve[np.newaxis, :, :]
            ) ** 2,
            axis=2,
        )
        nearest[start:stop] = np.argmin(distances, axis=1)
    return nearest


def _voxel_safety_margin_mm(image: sitk.Image) -> float:
    """用半个体素对角线覆盖体素中心到体素外缘的安全余量。"""

    spacing = [abs(float(value)) for value in image.GetSpacing()]
    return 0.5 * math.sqrt(sum(value ** 2 for value in spacing))


def _offsets_covering_interval(
    d_min: float,
    d_max: float,
    slice_thickness: float,
    slice_spacing: float,
) -> List[float]:
    """生成不少包 `[d_min, d_max]` 的 CPR 厚层中心 offset。"""

    thickness = float(slice_thickness)
    spacing = float(slice_spacing)
    if thickness <= 0.0:
        raise ValueError("slice_thickness 必须大于 0")
    if spacing <= 0.0:
        raise ValueError("slice_spacing 必须大于 0")

    half_thickness = thickness / 2.0
    neg_count = int(
        math.ceil(max(0.0, -float(d_min) - half_thickness) / spacing)
    )
    pos_count = int(
        math.ceil(max(0.0, float(d_max) - half_thickness) / spacing)
    )
    while -neg_count * spacing - half_thickness > float(d_min) + 1e-6:
        neg_count += 1
    while pos_count * spacing + half_thickness < float(d_max) - 1e-6:
        pos_count += 1
    return [
        round(index * spacing, 6)
        for index in range(-neg_count, pos_count + 1)
    ]


def _bbox_offsets_mm(
    localization_result: Dict[str, Any],
    label_names: List[str],
    curve_points: List[Point3D],
    offset_axis: Point3D,
    slice_thickness: float,
    slice_spacing: float,
) -> List[float]:
    """原始分割不可用时，用完整 bbox 角点作为保守回退。"""

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
        spacing = max(float(slice_spacing), 1e-6)
        projections = [-spacing, 0.0, spacing]
    interval = _extend_cpr_offset_interval(
        localization_result,
        (min(projections), max(projections)),
    )
    return _offsets_covering_interval(
        interval[0],
        interval[1],
        slice_thickness,
        slice_spacing,
    )


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


def _correct_reversing_cpr_centroids(
    centroid_points: List[Point3D],
    spacing_mm: float,
    longitudinal_axis: Point3D,
) -> List[Point3D]:
    """回正与局部曲率趋势相反的 CPR 中间质心。"""

    if len(centroid_points) < 5:
        return list(centroid_points)

    minimum_residual = max(float(spacing_mm), 1e-6)
    arrays = np.array([
        [point.x, point.y, point.z]
        for point in centroid_points
    ], dtype=float)
    axis = np.array([
        longitudinal_axis.x,
        longitudinal_axis.y,
        longitudinal_axis.z,
    ], dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return list(centroid_points)

    axis /= axis_norm
    if float((arrays[-1] - arrays[0]) @ axis) < 0.0:
        axis = -axis
    longitudinal = arrays @ axis
    if np.any(np.diff(longitudinal) <= 1e-6):
        return list(centroid_points)

    expected = arrays.copy()
    residuals = np.zeros_like(arrays)
    valid = np.zeros(len(arrays), dtype=bool)
    for index in range(1, len(arrays) - 1):
        denominator = longitudinal[index + 1] - longitudinal[index - 1]
        if denominator <= 1e-6:
            return list(centroid_points)
        weight = (longitudinal[index] - longitudinal[index - 1]) / denominator
        expected[index] = (
            arrays[index - 1] * (1.0 - weight)
            + arrays[index + 1] * weight
        )
        residuals[index] = arrays[index] - expected[index]
        residuals[index] -= axis * float(residuals[index] @ axis)
        valid[index] = (
            float(np.linalg.norm(residuals[index])) >= minimum_residual
        )

    corrected = arrays.copy()
    for index in range(1, len(arrays) - 1):
        if not valid[index]:
            continue
        reference = _neighbor_cpr_residual_trend(residuals, valid, index)
        if reference is None:
            if _is_dominant_cpr_local_corner(
                arrays,
                residuals,
                valid,
                index,
            ):
                corrected[index] = expected[index]
            continue
        if _residual_is_weaker_than_neighbor_trend(residuals, valid, index):
            continue
        if (
            _has_one_sided_cpr_residual_reference(valid, index)
            and not _is_terminal_side_cpr_centroid(residuals, index)
            and _raw_path_reverses_across_residual(arrays, residuals, index)
        ):
            continue
        if float(residuals[index] @ reference) < 0.0:
            corrected[index] = expected[index]

    return [
        Point3D(float(point[0]), float(point[1]), float(point[2]))
        for point in corrected
    ]


def _neighbor_cpr_residual_trend(
    residuals: np.ndarray,
    valid: np.ndarray,
    index: int,
) -> np.ndarray | None:
    """返回当前残差可参考的相邻曲率趋势。"""

    neighbors: list[np.ndarray] = []
    if index - 1 >= 1 and bool(valid[index - 1]):
        neighbors.append(residuals[index - 1])
    if index + 1 <= len(residuals) - 2 and bool(valid[index + 1]):
        neighbors.append(residuals[index + 1])
    if not neighbors:
        return None
    if len(neighbors) == 2 and float(neighbors[0] @ neighbors[1]) <= 0.0:
        return None
    reference = np.sum(np.array(neighbors), axis=0)
    if float(np.linalg.norm(reference)) <= 1e-6:
        return None
    return reference


def _residual_is_weaker_than_neighbor_trend(
    residuals: np.ndarray,
    valid: np.ndarray,
    index: int,
) -> bool:
    """避免被更强的相邻异常残差拖动。"""

    neighbor_norms: list[float] = []
    if index - 1 >= 1 and bool(valid[index - 1]):
        neighbor_norms.append(float(np.linalg.norm(residuals[index - 1])))
    if index + 1 <= len(residuals) - 2 and bool(valid[index + 1]):
        neighbor_norms.append(float(np.linalg.norm(residuals[index + 1])))
    if not neighbor_norms:
        return False
    return float(np.linalg.norm(residuals[index])) < max(neighbor_norms)


def _has_one_sided_cpr_residual_reference(
    valid: np.ndarray,
    index: int,
) -> bool:
    """判断当前点是否只有一侧残差可作趋势参考。"""

    usable_neighbors = 0
    if index - 1 >= 1 and bool(valid[index - 1]):
        usable_neighbors += 1
    if index + 1 <= len(valid) - 2 and bool(valid[index + 1]):
        usable_neighbors += 1
    return usable_neighbors == 1


def _is_terminal_side_cpr_centroid(
    residuals: np.ndarray,
    index: int,
) -> bool:
    """判断当前点是否靠近质心链端部。"""

    return index == 1 or index == len(residuals) - 2


def _raw_path_reverses_across_residual(
    points: np.ndarray,
    residuals: np.ndarray,
    index: int,
) -> bool:
    """判断原始路径在当前残差方向上是否一进一出。"""

    current = residuals[index]
    incoming_projection = float((points[index] - points[index - 1]) @ current)
    outgoing_projection = float((points[index + 1] - points[index]) @ current)
    return incoming_projection * outgoing_projection <= 0.0


def _is_dominant_cpr_local_corner(
    points: np.ndarray,
    residuals: np.ndarray,
    valid: np.ndarray,
    index: int,
) -> bool:
    """判断当前点是否为左右趋势冲突中的局部折角主峰。"""

    if index - 1 < 1 or index + 1 > len(residuals) - 2:
        return False
    if not (bool(valid[index - 1]) and bool(valid[index + 1])):
        return False

    previous = residuals[index - 1]
    current = residuals[index]
    following = residuals[index + 1]
    if float(previous @ following) >= 0.0:
        return False

    current_norm = float(np.linalg.norm(current))
    if current_norm <= float(np.linalg.norm(previous)):
        return False
    if current_norm <= float(np.linalg.norm(following)):
        return False

    if _raw_path_reverses_across_residual(points, residuals, index):
        return False

    return (
        float(current @ previous) < 0.0
        or float(current @ following) < 0.0
    )


def _smooth_cpr_anchor_path(
    anchor_points: List[Point3D],
    spacing_mm: float,
    longitudinal_axis: Point3D | None = None,
) -> List[Point3D]:
    """在所有端面/质心锚点之间生成受控圆滑曲线。

    锚点是解剖定位结果，必须原样保留；圆滑只改变锚点之间的连接方式。
    longitudinal_axis 仅用于保证路径按头脚方向单调前进，避免 CPR 发生折返。
    """

    if len(anchor_points) < 3:
        return list(anchor_points)

    target_spacing = max(float(spacing_mm), 1e-6)
    arrays = np.array([
        [point.x, point.y, point.z]
        for point in anchor_points
    ], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(arrays, axis=0), axis=1)
    if not np.any(segment_lengths > 1e-6):
        return list(anchor_points)

    if longitudinal_axis is not None:
        longitudinal = np.array([
            longitudinal_axis.x,
            longitudinal_axis.y,
            longitudinal_axis.z,
        ], dtype=float)
        longitudinal_norm = float(np.linalg.norm(longitudinal))
        if longitudinal_norm > 1e-6:
            longitudinal /= longitudinal_norm
            if float((arrays[-1] - arrays[0]) @ longitudinal) < 0.0:
                longitudinal = -longitudinal
            # A malformed endpoint/centroid order should not be hidden by a
            # spline; retaining anchors lets the caller surface the raw path.
            if np.any(np.diff(arrays @ longitudinal) <= 1e-6):
                return list(anchor_points)

    # Chord-length parameterization keeps the tangent scale stable when adjacent
    # vertebrae have different physical spacing.
    parameters = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    tangents = np.zeros_like(arrays)
    tangents[0] = (arrays[1] - arrays[0]) / max(parameters[1], 1e-6)
    tangents[-1] = (arrays[-1] - arrays[-2]) / max(
        parameters[-1] - parameters[-2],
        1e-6,
    )
    for index in range(1, len(arrays) - 1):
        delta = parameters[index + 1] - parameters[index - 1]
        tangents[index] = (arrays[index + 1] - arrays[index - 1]) / max(delta, 1e-6)

    # Keep cubic control points inside the neighboring chord scale. This rounds
    # each corner without pulling a nearby centroid toward a global arc.
    for index in range(len(arrays)):
        tangent_length = float(np.linalg.norm(tangents[index]))
        if tangent_length <= 1e-6:
            continue
        if index == 0:
            limit = 3.0 * segment_lengths[0] / max(parameters[1], 1e-6)
        elif index == len(arrays) - 1:
            limit = 3.0 * segment_lengths[-1] / max(
                parameters[-1] - parameters[-2],
                1e-6,
            )
        else:
            limit = 3.0 * min(
                segment_lengths[index - 1],
                segment_lengths[index],
            ) / max(
                parameters[index + 1] - parameters[index - 1],
                1e-6,
            )
        if tangent_length > limit:
            tangents[index] *= limit / tangent_length

    curve_points: List[Point3D] = [anchor_points[0]]
    for index, segment_length in enumerate(segment_lengths):
        if segment_length <= 1e-6:
            continue
        steps = max(1, int(math.ceil(segment_length / (target_spacing * 0.5))))
        delta_t = parameters[index + 1] - parameters[index]
        for step in range(1, steps + 1):
            if step == steps:
                curve_points.append(anchor_points[index + 1])
                continue
            u = step / steps
            h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
            h10 = u ** 3 - 2.0 * u ** 2 + u
            h01 = -2.0 * u ** 3 + 3.0 * u ** 2
            h11 = u ** 3 - u ** 2
            point = (
                h00 * arrays[index]
                + h10 * delta_t * tangents[index]
                + h01 * arrays[index + 1]
                + h11 * delta_t * tangents[index + 1]
            )
            curve_points.append(Point3D(
                float(point[0]),
                float(point[1]),
                float(point[2]),
            ))
    return curve_points


def _smooth_cpr_centroid_path_with_straight_endpoints(
    anchor_points: List[Point3D],
    centroid_points: List[Point3D],
    spacing_mm: float,
    longitudinal_axis: Point3D,
) -> List[Point3D]:
    """平滑质心链，并把首末端面作为不会过冲的直线端段拼接。"""

    corrected_centroid_points = _correct_reversing_cpr_centroids(
        centroid_points,
        spacing_mm,
        longitudinal_axis,
    )
    curve_points = _smooth_cpr_anchor_path(
        corrected_centroid_points,
        spacing_mm,
        longitudinal_axis,
    )
    if not corrected_centroid_points:
        return curve_points

    terminal_spacing = max(float(spacing_mm) * 0.5, 1e-6)
    if (
        anchor_points
        and vector_length(
            subtract_points(anchor_points[0], corrected_centroid_points[0])
        ) > 1e-6
    ):
        prefix = _resample_straight_segment(
            anchor_points[0],
            corrected_centroid_points[0],
            terminal_spacing,
        )
        curve_points = prefix[:-1] + curve_points

    if (
        anchor_points
        and vector_length(
            subtract_points(anchor_points[-1], corrected_centroid_points[-1])
        ) > 1e-6
    ):
        suffix = _resample_straight_segment(
            corrected_centroid_points[-1],
            anchor_points[-1],
            terminal_spacing,
        )
        curve_points = curve_points + suffix[1:]
    return curve_points


def _resample_straight_segment(
    start: Point3D,
    end: Point3D,
    spacing_mm: float,
) -> List[Point3D]:
    """退化情况下按首末点生成加密直线，保持阶段五采样稳定。"""

    delta = subtract_points(end, start)
    length = vector_length(delta)
    if length <= 1e-6:
        return [start]
    steps = max(1, int(math.ceil(length / max(float(spacing_mm), 1e-6))))
    return [
        Point3D(
            start.x + delta.x * step / steps,
            start.y + delta.y * step / steps,
            start.z + delta.z * step / steps,
        )
        for step in range(steps + 1)
    ]


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
