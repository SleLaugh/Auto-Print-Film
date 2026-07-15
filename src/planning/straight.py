#!/src/planning/straight.py
# -*- coding: utf-8 -*-

"""第四阶段通用直线切割的方向、范围和计划构造工具。"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List

from src.models import ExtentMM, Point3D, SlicePlan
from src.models.enums import CutLineExtentModeEnum, PlaneTypeEnum
from src.planning.geometry import (
    build_plane_stack_slices,
    image_axis_direction,
    normalize_vector,
    subtract_points,
    vector_length,
    voxel_index_to_physical,
)


@dataclass(frozen=True)
class StraightPlaneFrame:
    """一组普通直线切片共同使用的物理坐标系。"""

    row_dir: Point3D
    """输出图像从左向右的物理单位方向。"""

    col_dir: Point3D
    """输出图像从上向下的物理单位方向。"""

    normal: Point3D
    """单张切片的物理法向，后续带角度直线仍通过本字段表达。"""


@dataclass(frozen=True)
class ProjectedExtent:
    """三维范围投影到切片平面后得到的宽高和中心坐标。"""

    extent_mm: ExtentMM
    """输出图像在 row_dir、col_dir 上覆盖的物理宽度和高度。"""

    center_u: float
    """范围中心在 row_dir 上相对物理原点的投影坐标。"""

    center_v: float
    """范围中心在 col_dir 上相对物理原点的投影坐标。"""


@dataclass(frozen=True)
class CutSegment:
    """第四阶段准备交给通用直线构造器的一个待切区间。"""

    group_name: str
    """当前区间的稳定名称，例如 whole_range 或 L1-L2。"""

    target_start: Point3D
    """目标覆盖范围的物理起始边界点。"""

    target_end: Point3D
    """目标覆盖范围的物理结束边界点。"""

    target_region_bbox: Dict[str, float]
    """当前区间执行 target_bbox 范围切时使用的体素包围框。"""

    source_labels: tuple[str, ...]
    """产生当前区间的定位 label，用于调试、报告和问题追踪。"""


def _scale_direction(direction: Point3D, sign: float) -> Point3D:
    """按 1 或 -1 翻转一个输入图像轴方向。"""

    return Point3D(
        direction.x * sign,
        direction.y * sign,
        direction.z * sign,
    )


def default_straight_frame(
    plane: PlaneTypeEnum,
    image_info: Dict[str, Any],
) -> StraightPlaneFrame:
    """根据输入图像方向矩阵返回横、矢、冠状位的零角度直线坐标系。"""

    if plane == PlaneTypeEnum.横状位:
        row_axis, row_sign = 0, 1.0
        col_axis, col_sign = 1, 1.0
        normal_axis, normal_sign = 2, 1.0
    elif plane == PlaneTypeEnum.矢状位:
        row_axis, row_sign = 1, 1.0
        col_axis, col_sign = 2, -1.0
        normal_axis, normal_sign = 0, -1.0
    elif plane == PlaneTypeEnum.冠状位:
        row_axis, row_sign = 0, 1.0
        col_axis, col_sign = 2, -1.0
        normal_axis, normal_sign = 1, 1.0
    else:
        raise ValueError(f"不支持的切割平面: {plane}")

    return StraightPlaneFrame(
        row_dir=normalize_vector(_scale_direction(
            image_axis_direction(image_info, row_axis),
            row_sign,
        )),
        col_dir=normalize_vector(_scale_direction(
            image_axis_direction(image_info, col_axis),
            col_sign,
        )),
        normal=normalize_vector(_scale_direction(
            image_axis_direction(image_info, normal_axis),
            normal_sign,
        )),
    )


def bbox_physical_corners(
    bbox: Dict[str, float],
    image_info: Dict[str, Any],
) -> List[Point3D]:
    """把体素 bbox 的八个中心角点转换成物理坐标点。"""

    return [
        voxel_index_to_physical(Point3D(x, y, z), image_info)
        for x, y, z in product(
            (float(bbox["x_min"]), float(bbox["x_max"])),
            (float(bbox["y_min"]), float(bbox["y_max"])),
            (float(bbox["z_min"]), float(bbox["z_max"])),
        )
    ]


def image_physical_corners(image_info: Dict[str, Any]) -> List[Point3D]:
    """返回原图体素中心范围的八个物理角点。"""

    size = image_info.get("size_xyz")
    if not size or len(size) != 3:
        raise ValueError("图像信息缺少有效的 size_xyz，无法计算整平面范围")
    if any(int(value) <= 0 for value in size):
        raise ValueError("图像 size_xyz 的三个维度必须大于 0")

    return [
        voxel_index_to_physical(Point3D(x, y, z), image_info)
        for x, y, z in product(
            (0.0, float(size[0]) - 1.0),
            (0.0, float(size[1]) - 1.0),
            (0.0, float(size[2]) - 1.0),
        )
    ]


def _dot(point: Point3D, direction: Point3D) -> float:
    """计算物理点在指定单位方向上的投影坐标。"""

    return (
        point.x * direction.x
        + point.y * direction.y
        + point.z * direction.z
    )


def project_region_extent(
    physical_points: Iterable[Point3D],
    frame: StraightPlaneFrame,
) -> ProjectedExtent:
    """把三维范围点投影到直线切片坐标系并计算平面内包围范围。"""

    points = list(physical_points)
    if not points:
        raise ValueError("计算直线切割范围时至少需要一个物理点")

    u_values = [_dot(point, frame.row_dir) for point in points]
    v_values = [_dot(point, frame.col_dir) for point in points]
    minimum_u, maximum_u = min(u_values), max(u_values)
    minimum_v, maximum_v = min(v_values), max(v_values)
    return ProjectedExtent(
        extent_mm=ExtentMM(
            width=maximum_u - minimum_u,
            height=maximum_v - minimum_v,
        ),
        center_u=(minimum_u + maximum_u) / 2.0,
        center_v=(minimum_v + maximum_v) / 2.0,
    )


def resolve_projected_extent(
    mode: CutLineExtentModeEnum,
    localization_result: Dict[str, Any],
    target_region_bbox: Dict[str, float],
    frame: StraightPlaneFrame,
) -> ProjectedExtent:
    """按整平面或当前目标 bbox 选择三维范围并完成统一投影。"""

    image_info = localization_result["image"]
    if mode == CutLineExtentModeEnum.部位范围:
        points = bbox_physical_corners(target_region_bbox, image_info)
    elif mode == CutLineExtentModeEnum.整平面:
        # 旧测试数据可能没有 size_xyz，此时保持原有回退到目标 bbox 的行为。
        if image_info.get("size_xyz"):
            points = image_physical_corners(image_info)
        else:
            points = bbox_physical_corners(target_region_bbox, image_info)
    else:
        raise ValueError(f"不支持的切割线范围模式: {mode}")
    return project_region_extent(points, frame)


def inset_segment_range(
    target_start: Point3D,
    target_end: Point3D,
    slice_thickness: float,
) -> tuple[Point3D, Point3D]:
    """把区间两端向内缩半层厚，得到首末切片的有效中心范围。"""

    if slice_thickness <= 0:
        raise ValueError("层厚必须大于 0")
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


def _recenter_point_in_plane(
    point: Point3D,
    projected_extent: ProjectedExtent,
    frame: StraightPlaneFrame,
) -> Point3D:
    """只沿切片行列方向平移一点，使其落到选定范围的平面内中心。"""

    delta_u = projected_extent.center_u - _dot(point, frame.row_dir)
    delta_v = projected_extent.center_v - _dot(point, frame.col_dir)
    return Point3D(
        point.x + frame.row_dir.x * delta_u + frame.col_dir.x * delta_v,
        point.y + frame.row_dir.y * delta_u + frame.col_dir.y * delta_v,
        point.z + frame.row_dir.z * delta_u + frame.col_dir.z * delta_v,
    )


def _default_output_spacing(
    plane: PlaneTypeEnum,
    image_info: Dict[str, Any],
) -> Point3D:
    """返回三个零角度默认状位对应的输出行列像素间距。"""

    spacing = image_info["spacing_xyz"]
    if plane == PlaneTypeEnum.横状位:
        row_axis, col_axis = 0, 1
    elif plane == PlaneTypeEnum.矢状位:
        row_axis, col_axis = 1, 2
    elif plane == PlaneTypeEnum.冠状位:
        row_axis, col_axis = 0, 2
    else:
        raise ValueError(f"不支持的切割平面: {plane}")
    return Point3D(
        float(spacing[row_axis]),
        float(spacing[col_axis]),
        0.0,
    )


def build_straight_slice_plan(
    request: Any,
    cut_mode: str,
    plane_config: Any,
    localization_result: Dict[str, Any],
    segment: CutSegment,
    frame: StraightPlaneFrame,
) -> SlicePlan:
    """把一个待切区间、直线方向和图像范围组装成第五阶段可执行计划。"""

    projected_extent = resolve_projected_extent(
        plane_config.line_extent_mode,
        localization_result,
        segment.target_region_bbox,
        frame,
    )
    centered_target_start = _recenter_point_in_plane(
        segment.target_start,
        projected_extent,
        frame,
    )
    centered_target_end = _recenter_point_in_plane(
        segment.target_end,
        projected_extent,
        frame,
    )
    # 必须先让两个目标边界落到最终切片坐标系的同一平面内中心，再沿二者
    # 形成的最终法向内缩。否则以后加入倾角时，半层厚只会在旧轴上内缩，
    # 投影到新法向后的实际内缩量会不足。
    centered_range_start, centered_range_end = inset_segment_range(
        centered_target_start,
        centered_target_end,
        plane_config.slice_thickness,
    )
    path_points = [centered_range_start, centered_range_end]
    slices = build_plane_stack_slices(
        path_points=path_points,
        slice_spacing=plane_config.slice_spacing,
        normal=frame.normal,
        row_dir=frame.row_dir,
        col_dir=frame.col_dir,
    )

    plan_key = cut_mode
    if segment.group_name != "whole_range":
        plan_key = f"{plan_key}_{segment.group_name}"
    return SlicePlan(
        plan_id=(
            f"{request.body_part.value}_{plan_key}_{plane_config.plane.value}"
        ),
        body_part=request.body_part.value,
        cut_mode=cut_mode,
        plane=plane_config.plane.value,
        group_name=segment.group_name,
        geometry_type="plane_stack",
        range_start=centered_range_start,
        range_end=centered_range_end,
        path_points=path_points,
        slices=slices,
        extent_mm=projected_extent.extent_mm,
        output_spacing_mm=_default_output_spacing(
            plane_config.plane,
            localization_result["image"],
        ),
        slice_thickness=plane_config.slice_thickness,
        slice_spacing=plane_config.slice_spacing,
        window_presets=list(plane_config.window_presets),
        source_labels=list(segment.source_labels),
        target_start=centered_target_start,
        target_end=centered_target_end,
    )
