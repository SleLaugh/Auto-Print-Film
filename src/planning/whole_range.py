#!/src/planning/whole_range.py
# -*- coding: utf-8 -*-

"""整切 planner 的公共方法。"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.models import ExtentMM, Point3D, SlicePlan
from src.models.enums import PlaneTypeEnum, CutLineExtentModeEnum
from src.planning.geometry import (
    voxel_index_to_physical,
)
from src.planning.straight import (
    CutSegment,
    build_straight_slice_plan,
    default_straight_frame,
    inset_segment_range,
    resolve_projected_extent,
)


@dataclass(frozen=True)
class _PlaneAxes:
    """描述一个默认直线状位对应的输入图像堆叠轴及路径方向。"""

    stack_axis: int
    """切片中心依次移动的输入图像轴，0/1/2 对应 x/y/z。"""

    path_sign: float
    """切片序列在对应患者物理坐标轴上的方向，只允许 1 或 -1。"""

_AXIS_NAMES = ("x", "y", "z")
"""体素轴序号到 bbox 字段前缀的固定映射。"""

_PLANE_AXES = {
    # 横状位从患者物理 z 高端向低端输出。
    PlaneTypeEnum.横状位: _PlaneAxes(2, -1.0),
    # 矢状位沿患者物理 x 高端向低端输出。
    PlaneTypeEnum.矢状位: _PlaneAxes(0, -1.0),
    # 冠状位沿患者物理 y 低端向高端输出。
    PlaneTypeEnum.冠状位: _PlaneAxes(1, 1.0),
}
"""三个默认直线状位的堆叠轴和路径顺序定义。"""


def _plane_axes(plane: PlaneTypeEnum) -> _PlaneAxes:
    """返回指定状位的默认直线轴定义。"""

    try:
        return _PLANE_AXES[plane]
    except KeyError as error:
        raise ValueError(f"不支持的切割平面: {plane}") from error


def bbox_target_points(
    localization_result: Dict[str, Any],
    plane: PlaneTypeEnum,
) -> tuple[Point3D, Point3D]:
    """
    把目标 bbox 在指定堆叠轴上的体素外边界转换成物理坐标。

    另外两个轴固定在 bbox 中心。每个状位的起止顺序由 path_sign 独立控制，
    不会因为调整序列顺序而改变单张图像的法向或行列方向。
    """

    bbox = localization_result.get("correction_scope_bbox")
    if not bbox:
        raise ValueError("定位结果缺少 correction_scope_bbox")

    axes = _plane_axes(plane)
    index_values = [
        (float(bbox[f"{axis_name}_min"]) + float(bbox[f"{axis_name}_max"]))
        / 2.0
        for axis_name in _AXIS_NAMES
    ]
    stack_name = _AXIS_NAMES[axes.stack_axis]
    minimum_boundary = float(bbox[f"{stack_name}_min"]) - 0.5
    maximum_boundary = float(bbox[f"{stack_name}_max"]) + 0.5

    image_info = localization_result["image"]
    minimum_index = list(index_values)
    maximum_index = list(index_values)
    minimum_index[axes.stack_axis] = minimum_boundary
    maximum_index[axes.stack_axis] = maximum_boundary
    minimum_point = voxel_index_to_physical(Point3D(*minimum_index), image_info)
    maximum_point = voxel_index_to_physical(Point3D(*maximum_index), image_info)

    # 路径顺序按患者物理坐标判断，避免输入图像轴反向时再次输出反序。
    physical_axis_name = _AXIS_NAMES[axes.stack_axis]
    minimum_coordinate = getattr(minimum_point, physical_axis_name)
    maximum_coordinate = getattr(maximum_point, physical_axis_name)
    lower_point, upper_point = (
        (minimum_point, maximum_point)
        if minimum_coordinate <= maximum_coordinate
        else (maximum_point, minimum_point)
    )
    if axes.path_sign > 0:
        return lower_point, upper_point
    return upper_point, lower_point


def bbox_axial_target_points(
    localization_result: Dict[str, Any],
) -> tuple[Point3D, Point3D]:
    """兼容旧调用：返回横状位 bbox 的物理外边界。"""

    return bbox_target_points(localization_result, PlaneTypeEnum.横状位)


def bbox_range_points(
    localization_result: Dict[str, Any],
    slice_thickness: float,
    plane: PlaneTypeEnum,
) -> tuple[Point3D, Point3D]:
    """
    在指定堆叠轴上生成整切的有效切片中心范围。

    第一张和最后一张中心从目标外边界向内缩半个层厚。目标短于层厚时，
    两端都使用目标中点，从而只生成一张居中的厚层图像。
    """

    target_start, target_end = bbox_target_points(localization_result, plane)
    return inset_segment_range(
        target_start,
        target_end,
        slice_thickness,
    )


def bbox_axial_range_points(
    localization_result: Dict[str, Any],
    slice_thickness: float,
) -> tuple[Point3D, Point3D]:
    """
    用整体目标 bbox 的最外侧 z 范围生成横状位整切有效中心范围。

    第一张和最后一张中心分别从真实目标边界向内缩半个层厚，使厚层覆盖
    边界与目标边界对齐。目标短于层厚时，两端都取目标中点。
    """

    return bbox_range_points(
        localization_result,
        slice_thickness,
        PlaneTypeEnum.横状位,
    )


def bbox_extent_mm(bbox: Dict[str, float], image_info: Dict[str, Any]) -> ExtentMM:
    """从整体 bbox 估算单张重建图像的覆盖范围。"""

    spacing = image_info["spacing_xyz"]
    return ExtentMM(
        width=(bbox["x_max"] - bbox["x_min"]) * spacing[0],
        height=(bbox["y_max"] - bbox["y_min"]) * spacing[1],
    )


def plane_bbox_extent_mm(
    bbox: Dict[str, float],
    image_info: Dict[str, Any],
    plane: PlaneTypeEnum,
) -> ExtentMM:
    """从目标 bbox 计算指定状位平面内的物理宽度和高度。"""

    frame = default_straight_frame(plane, image_info)
    return resolve_projected_extent(
        CutLineExtentModeEnum.部位范围,
        {"image": image_info},
        bbox,
        frame,
    ).extent_mm


def plane_extent_mm(
    bbox: Dict[str, float],
    image_info: Dict[str, Any],
    plane: PlaneTypeEnum,
    mode: CutLineExtentModeEnum = CutLineExtentModeEnum.整平面,
) -> ExtentMM:
    """计算切割线所在平面的横向/纵向物理范围。"""

    frame = default_straight_frame(plane, image_info)
    return resolve_projected_extent(
        mode,
        {"image": image_info},
        bbox,
        frame,
    ).extent_mm


def build_whole_range_plan_from_bbox(
    request: Any,
    plane_config: Any,
    localization_result: Dict[str, Any],
    source_labels: Optional[List[str]] = None,
) -> SlicePlan:
    """
    根据整体 bbox 构建指定状位的默认直线整切计划。

    不管目标是多个椎体组成的整体，还是头骨、肺部这种单个目标，只要第三步
    已经给出 correction_scope_bbox，这里都会按当前状位的堆叠轴生成同一种
    plane_stack 协议。部位特殊角度不属于本方法职责。
    """

    bbox = localization_result.get("correction_scope_bbox")
    if not bbox:
        raise ValueError("定位结果缺少 correction_scope_bbox")

    target_start, target_end = bbox_target_points(
        localization_result,
        plane_config.plane,
    )
    segment = CutSegment(
        group_name="whole_range",
        target_start=target_start,
        target_end=target_end,
        target_region_bbox=bbox,
        source_labels=tuple(source_labels or ["correction_scope_bbox"]),
    )
    frame = default_straight_frame(
        plane_config.plane,
        localization_result["image"],
    )
    return build_straight_slice_plan(
        request=request,
        cut_mode="whole_range",
        plane_config=plane_config,
        localization_result=localization_result,
        segment=segment,
        frame=frame,
    )


def build_axial_whole_range_plan_from_bbox(
    request: Any,
    plane_config: Any,
    localization_result: Dict[str, Any],
    source_labels: Optional[List[str]] = None,
) -> SlicePlan:
    """
    兼容旧函数名并转发到通用整切构造器。

    旧名称包含 axial，但迁移期允许传入三种状位，避免现有调用方在统一 planner
    接入完成前出现断裂。新代码应直接调用 build_whole_range_plan_from_bbox。
    """

    return build_whole_range_plan_from_bbox(
        request=request,
        plane_config=plane_config,
        localization_result=localization_result,
        source_labels=source_labels,
    )
