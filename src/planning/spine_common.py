#!/src/planning/spine_common.py
# -*- coding: utf-8 -*-

"""脊柱类 planner 的公共方法。"""

from typing import Any, Dict, List, Optional

from src.models import Point3D, SlicePlan
from src.planning.geometry import (
    build_plane_stack_slices,
    image_axis_direction,
    voxel_index_to_physical,
)
from src.planning.whole_range import plane_extent_mm


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


def build_axial_spine_plan(
    request: Any,
    plane_config: Any,
    group_name: str,
    range_start: Point3D,
    range_end: Point3D,
    localization_result: Dict[str, Any],
    source_labels: List[str],
) -> SlicePlan:
    """构建脊柱横状位 plane_stack 计划，并在这里展开每一张 slice。"""

    bbox = localization_result.get("correction_scope_bbox")
    if not bbox:
        raise ValueError("定位结果缺少 correction_scope_bbox")

    path_points = [range_start, range_end]
    slices = build_plane_stack_slices(
        path_points=path_points,
        slice_spacing=plane_config.slice_spacing,
        normal=image_axis_direction(localization_result["image"], 2),
        row_dir=image_axis_direction(localization_result["image"], 0),
        col_dir=image_axis_direction(localization_result["image"], 1),
    )

    plan_key = request.cut_mode.value
    if group_name != "whole_range":
        plan_key = f"{plan_key}_{group_name}"

    spacing = localization_result["image"]["spacing_xyz"]
    return SlicePlan(
        plan_id=f"{request.body_part.value}_{plan_key}_{plane_config.plane.value}",
        body_part=request.body_part.value,
        cut_mode=request.cut_mode.value,
        plane=plane_config.plane.value,
        group_name=group_name,
        geometry_type="plane_stack",
        range_start=range_start,
        range_end=range_end,
        path_points=path_points,
        slices=slices,
        extent_mm=plane_extent_mm(
            bbox,
            localization_result["image"],
            plane_config.plane,
            plane_config.line_extent_mode,
        ),
        output_spacing_mm=Point3D(spacing[0], spacing[1], 0.0),
        slice_thickness=plane_config.slice_thickness,
        slice_spacing=plane_config.slice_spacing,
        window_presets=[
            window.value
            for window in plane_config.window_presets
        ],
        source_labels=source_labels,
    )
