#!/src/planning/spine_common.py
# -*- coding: utf-8 -*-

"""脊柱类 planner 的公共方法。"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.models import Point3D
from src.planning.geometry import voxel_index_to_physical
from src.planning.straight import CutSegment


def voxel_centroid_to_physical(
    centroid: Dict[str, float],
    image_info: Dict[str, Any],
) -> Point3D:
    """
    把第三步输出的 z/y/x 体素质心转成 x/y/z 物理坐标。

    这里必须同时使用 spacing、origin 和 direction，避免带方向矩阵的
    NIfTI 在阶段四生成错误的物理坐标。
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

        upper_bbox = _label_bbox(localization_result, upper_label)
        lower_bbox = _label_bbox(localization_result, lower_label)
        if upper_bbox is not None and lower_bbox is not None:
            target_region_bbox = union_bboxes(upper_bbox, lower_bbox)
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
        else:
            if fallback_bbox is None:
                warnings.append(f"跳过脊柱区间 {group_name}: 缺少可用 bbox")
                continue
            target_region_bbox = dict(fallback_bbox)
            target_start, target_end = upper_center, lower_center
            warnings.append(
                f"脊柱区间 {group_name} 缺少椎体 bbox，已回退到质心范围"
            )

        if target_start.z <= target_end.z:
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
        ))

    return segments, warnings
