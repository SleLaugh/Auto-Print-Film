#!/src/planning/whole_range.py
# -*- coding: utf-8 -*-

"""整切 planner 的公共方法。"""

from typing import Any, Dict, List, Optional

from src.models import ExtentMM, Point3D, SlicePlan
from src.models.enums import PlaneTypeEnum
from src.planning.geometry import (
    build_plane_stack_slices,
    image_axis_direction,
    voxel_index_to_physical,
)


def bbox_axial_range_points(
    localization_result: Dict[str, Any],
    slice_thickness: float,
) -> tuple[Point3D, Point3D]:
    """
    用整体目标 bbox 的最外侧 z 范围生成横状位整切起止点。

    横状位整切只关心“目标整体块”的上下界，不关心这个目标是单个器官，
    还是由多个椎体合成的大块。x/y 取整体 bbox 中心，用来代表这一组
    横状位切片的中心路径。

    这里生成的是输出图像的中心范围。厚层图像会在第五步按中心点上下
    各覆盖半个层厚，所以整切中心范围要向上下各外扩 half thickness，
    否则调试线和后续输出图像都会显得没有包住目标最外侧。
    """

    bbox = localization_result.get("correction_scope_bbox")
    if not bbox:
        raise ValueError("定位结果缺少 correction_scope_bbox")

    image_info = localization_result["image"]
    x_center = (bbox["x_min"] + bbox["x_max"]) / 2.0
    y_center = (bbox["y_min"] + bbox["y_max"]) / 2.0
    z_spacing = image_info["spacing_xyz"][2]
    z_margin_voxels = (slice_thickness / 2.0) / z_spacing
    return (
        voxel_index_to_physical(
            Point3D(x_center, y_center, bbox["z_min"] - z_margin_voxels),
            image_info,
        ),
        voxel_index_to_physical(
            Point3D(x_center, y_center, bbox["z_max"] + z_margin_voxels),
            image_info,
        ),
    )


def bbox_extent_mm(bbox: Dict[str, float], image_info: Dict[str, Any]) -> ExtentMM:
    """从整体 bbox 估算单张重建图像的覆盖范围。"""

    spacing = image_info["spacing_xyz"]
    return ExtentMM(
        width=(bbox["x_max"] - bbox["x_min"]) * spacing[0],
        height=(bbox["y_max"] - bbox["y_min"]) * spacing[1],
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
        extent_mm=bbox_extent_mm(bbox, localization_result["image"]),
        output_spacing_mm=Point3D(spacing[0], spacing[1], 0.0),
        slice_thickness=plane_config.slice_thickness,
        slice_spacing=plane_config.slice_spacing,
        window_presets=[
            window.value
            for window in plane_config.window_presets
        ],
        source_labels=source_labels or ["correction_scope_bbox"],
    )
