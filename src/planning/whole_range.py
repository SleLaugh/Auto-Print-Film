#!/src/planning/whole_range.py
# -*- coding: utf-8 -*-

"""整切 planner 的公共方法。"""

from typing import Any, Dict, List, Optional

from src.models import ExtentMM, Point3D, SlicePlan
from src.models.enums import PlaneTypeEnum, CutLineExtentModeEnum
from src.planning.geometry import (
    build_plane_stack_slices,
    image_axis_direction,
    normalize_vector,
    subtract_points,
    vector_length,
    voxel_index_to_physical,
)


def bbox_axial_target_points(
    localization_result: Dict[str, Any],
) -> tuple[Point3D, Point3D]:
    """把包含端点的 bbox z 体素索引转换成目标物理外边界。"""

    bbox = localization_result.get("correction_scope_bbox")
    if not bbox:
        raise ValueError("定位结果缺少 correction_scope_bbox")

    image_info = localization_result["image"]
    x_center = (bbox["x_min"] + bbox["x_max"]) / 2.0
    y_center = (bbox["y_min"] + bbox["y_max"]) / 2.0
    target_start = voxel_index_to_physical(
        Point3D(x_center, y_center, bbox["z_min"] - 0.5),
        image_info,
    )
    target_end = voxel_index_to_physical(
        Point3D(x_center, y_center, bbox["z_max"] + 0.5),
        image_info,
    )
    return target_start, target_end


def bbox_axial_range_points(
    localization_result: Dict[str, Any],
    slice_thickness: float,
) -> tuple[Point3D, Point3D]:
    """
    用整体目标 bbox 的最外侧 z 范围生成横状位整切有效中心范围。

    第一张和最后一张中心分别从真实目标边界向内缩半个层厚，使厚层覆盖
    边界与目标边界对齐。目标短于层厚时，两端都取目标中点。
    """

    target_start, target_end = bbox_axial_target_points(localization_result)
    target_vector = subtract_points(target_end, target_start)
    target_length = vector_length(target_vector)
    if target_length <= slice_thickness:
        return (
            Point3D(
                (target_start.x + target_end.x) / 2.0,
                (target_start.y + target_end.y) / 2.0,
                (target_start.z + target_end.z) / 2.0,
            ),
        ) * 2

    direction = normalize_vector(target_vector)
    half_thickness = slice_thickness / 2.0
    return (
        Point3D(
            target_start.x + direction.x * half_thickness,
            target_start.y + direction.y * half_thickness,
            target_start.z + direction.z * half_thickness,
        ),
        Point3D(
            target_end.x - direction.x * half_thickness,
            target_end.y - direction.y * half_thickness,
            target_end.z - direction.z * half_thickness,
        ),
    )


def bbox_extent_mm(bbox: Dict[str, float], image_info: Dict[str, Any]) -> ExtentMM:
    """从整体 bbox 估算单张重建图像的覆盖范围。"""

    spacing = image_info["spacing_xyz"]
    return ExtentMM(
        width=(bbox["x_max"] - bbox["x_min"]) * spacing[0],
        height=(bbox["y_max"] - bbox["y_min"]) * spacing[1],
    )


def plane_extent_mm(
    bbox: Dict[str, float],
    image_info: Dict[str, Any],
    plane: PlaneTypeEnum,
    mode: CutLineExtentModeEnum = CutLineExtentModeEnum.整平面,
) -> ExtentMM:
    """计算切割线所在平面的横向/纵向物理范围。"""

    if mode == CutLineExtentModeEnum.部位范围:
        return bbox_extent_mm(bbox, image_info)

    size = image_info.get("size_xyz")
    if not size:
        return bbox_extent_mm(bbox, image_info)
    spacing = image_info["spacing_xyz"]
    if plane == PlaneTypeEnum.横状位:
        axes = (0, 1)
    elif plane == PlaneTypeEnum.矢状位:
        axes = (1, 2)
    elif plane == PlaneTypeEnum.冠状位:
        axes = (0, 2)
    else:
        raise ValueError(f"不支持的切割平面: {plane}")
    return ExtentMM(
        width=max(size[axes[0]] - 1, 1) * spacing[axes[0]],
        height=max(size[axes[1]] - 1, 1) * spacing[axes[1]],
    )


def build_axial_whole_range_plan_from_bbox(
    request: Any,
    plane_config: Any,
    localization_result: Dict[str, Any],
    source_labels: Optional[List[str]] = None,
) -> SlicePlan:
    """
    根据整体 bbox 构建横状位整切计划，并在第四步展开每一张切片。

    这是“横状位整切”的统一入口：不管目标是腰椎这种多个椎体组成的整体，
    还是头骨、肺部这种单个目标，只要第三步已经给出 correction_scope_bbox，
    这里都按 bbox 的最外侧 z_min/z_max 来生成切片。
    """

    if plane_config.plane != PlaneTypeEnum.横状位:
        raise ValueError("横状位整切公共方法只能处理横状位")

    bbox = localization_result.get("correction_scope_bbox")
    if not bbox:
        raise ValueError("定位结果缺少 correction_scope_bbox")

    range_start, range_end = bbox_axial_range_points(
        localization_result,
        plane_config.slice_thickness,
    )
    target_start, target_end = bbox_axial_target_points(localization_result)
    path_points = [range_start, range_end]
    slices = build_plane_stack_slices(
        path_points=path_points,
        slice_spacing=plane_config.slice_spacing,
        normal=image_axis_direction(localization_result["image"], 2),
        row_dir=image_axis_direction(localization_result["image"], 0),
        col_dir=image_axis_direction(localization_result["image"], 1),
    )

    spacing = localization_result["image"]["spacing_xyz"]
    return SlicePlan(
        plan_id=f"{request.body_part.value}_{request.cut_mode.value}_{plane_config.plane.value}",
        body_part=request.body_part.value,
        cut_mode=request.cut_mode.value,
        plane=plane_config.plane.value,
        group_name="whole_range",
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
        source_labels=source_labels or ["correction_scope_bbox"],
        target_start=target_start,
        target_end=target_end,
    )
