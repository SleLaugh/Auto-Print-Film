#!/src/planning/whole_range.py
# -*- coding: utf-8 -*-

"""整切 planner 的公共方法。"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _PlaneAxes:
    """描述一个默认直线状位对应的输入图像轴及各自方向符号。"""

    stack_axis: int
    """切片中心依次移动的输入图像轴，0/1/2 对应 x/y/z。"""

    path_sign: float
    """切片序列在对应患者物理坐标轴上的方向，只允许 1 或 -1。"""

    normal_sign: float
    """切片法向相对 stack_axis 的方向符号，只允许 1 或 -1。"""

    row_axis: int
    """输出图像横向对应的输入图像轴。"""

    row_sign: float
    """输出图像横向相对 row_axis 的方向符号。"""

    col_axis: int
    """输出图像纵向对应的输入图像轴。"""

    col_sign: float
    """输出图像纵向相对 col_axis 的方向符号。"""


_AXIS_NAMES = ("x", "y", "z")
"""体素轴序号到 bbox 字段前缀的固定映射。"""

_PLANE_AXES = {
    # 横状位从患者物理 z 高端向低端输出，行列方向和图像 z 轴法向保持不变。
    PlaneTypeEnum.横状位: _PlaneAxes(2, -1.0, 1.0, 0, 1.0, 1, 1.0),
    # 矢状位沿 x 反向堆叠，z 反向显示可让头侧位于图像上方。
    PlaneTypeEnum.矢状位: _PlaneAxes(0, -1.0, -1.0, 1, 1.0, 2, -1.0),
    # 冠状位沿 y 正向堆叠，z 同样反向显示；三组方向均保持右手系。
    PlaneTypeEnum.冠状位: _PlaneAxes(1, 1.0, 1.0, 0, 1.0, 2, -1.0),
}
"""三个默认直线状位的路径、法向和平面内方向定义。"""


def _plane_axes(plane: PlaneTypeEnum) -> _PlaneAxes:
    """返回指定状位的默认直线轴定义。"""

    try:
        return _PLANE_AXES[plane]
    except KeyError as error:
        raise ValueError(f"不支持的切割平面: {plane}") from error


def _scale_direction(direction: Point3D, sign: float) -> Point3D:
    """按方向符号翻转输入图像轴向量。"""

    return Point3D(
        direction.x * sign,
        direction.y * sign,
        direction.z * sign,
    )


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
    target_vector = subtract_points(target_end, target_start)
    target_length = vector_length(target_vector)
    if target_length <= slice_thickness:
        midpoint = Point3D(
            (target_start.x + target_end.x) / 2.0,
            (target_start.y + target_end.y) / 2.0,
            (target_start.z + target_end.z) / 2.0,
        )
        return midpoint, midpoint

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

    axes = _plane_axes(plane)
    spacing = image_info["spacing_xyz"]
    row_name = _AXIS_NAMES[axes.row_axis]
    col_name = _AXIS_NAMES[axes.col_axis]
    return ExtentMM(
        width=(bbox[f"{row_name}_max"] - bbox[f"{row_name}_min"])
        * spacing[axes.row_axis],
        height=(bbox[f"{col_name}_max"] - bbox[f"{col_name}_min"])
        * spacing[axes.col_axis],
    )


def plane_extent_mm(
    bbox: Dict[str, float],
    image_info: Dict[str, Any],
    plane: PlaneTypeEnum,
    mode: CutLineExtentModeEnum = CutLineExtentModeEnum.整平面,
) -> ExtentMM:
    """计算切割线所在平面的横向/纵向物理范围。"""

    if mode == CutLineExtentModeEnum.部位范围:
        return plane_bbox_extent_mm(bbox, image_info, plane)

    size = image_info.get("size_xyz")
    if not size:
        return plane_bbox_extent_mm(bbox, image_info, plane)
    spacing = image_info["spacing_xyz"]
    plane_axes = _plane_axes(plane)
    axes = (plane_axes.row_axis, plane_axes.col_axis)
    return ExtentMM(
        width=max(size[axes[0]] - 1, 1) * spacing[axes[0]],
        height=max(size[axes[1]] - 1, 1) * spacing[axes[1]],
    )


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

    axes = _plane_axes(plane_config.plane)
    range_start, range_end = bbox_range_points(
        localization_result,
        plane_config.slice_thickness,
        plane_config.plane,
    )
    target_start, target_end = bbox_target_points(
        localization_result,
        plane_config.plane,
    )
    path_points = [range_start, range_end]
    image_info = localization_result["image"]
    slices = build_plane_stack_slices(
        path_points=path_points,
        slice_spacing=plane_config.slice_spacing,
        normal=_scale_direction(
            image_axis_direction(image_info, axes.stack_axis),
            axes.normal_sign,
        ),
        row_dir=_scale_direction(
            image_axis_direction(image_info, axes.row_axis),
            axes.row_sign,
        ),
        col_dir=_scale_direction(
            image_axis_direction(image_info, axes.col_axis),
            axes.col_sign,
        ),
    )

    spacing = image_info["spacing_xyz"]
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
            image_info,
            plane_config.plane,
            plane_config.line_extent_mode,
        ),
        output_spacing_mm=Point3D(
            spacing[axes.row_axis],
            spacing[axes.col_axis],
            0.0,
        ),
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
