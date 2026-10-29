#!/src/planning/straight.py
# -*- coding: utf-8 -*-

"""第六阶段通用直线切割的方向、范围和计划构造工具。"""

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Dict, Iterable, List

from src.anatomy.orientation import patient_reference_axes
from src.models import AnatomicalFrame, ExtentMM, Point3D, SlicePlan
from src.models.enums import PlaneTypeEnum
from src.planning.sampling import ResolvedSampling, resolve_sampling
from src.planning.face_extension import expand_physical_points, resolve_face_extension
from src.planning.geometry import (
    build_plane_stack_slices,
    image_axis_direction,
    image_direction_matrix,
    normalize_vector,
    subtract_points,
    vector_length,
    voxel_index_to_physical,
)


# DICOM 方向余弦和物理坐标通常经过有限小数位存储。斜向区间在还原后可能
# 出现数微米的边界误差；1e-3mm 以内按同一物理边界处理。该常量同时
# 供计划计算与诊断输出使用，保证两者对“已覆盖”的判定一致。
CENTER_COVERAGE_TOLERANCE_MM = 1e-3


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
    """第六阶段交给通用直线构造器的一个待切区间。"""

    group_name: str
    """当前区间的稳定名称，例如 whole_range 或 L1_L2。"""

    target_start: Point3D
    """目标覆盖范围的物理起始边界点。"""

    target_end: Point3D
    """目标覆盖范围的物理结束边界点。"""

    target_region_bbox: Dict[str, float]
    """当前区间执行 target_bbox 范围切时使用的体素包围框。"""

    source_labels: tuple[str, ...]
    """产生当前区间的定位 label，用于调试、报告和问题追踪。"""

    frame: StraightPlaneFrame | None = None
    """当前区间可选的专属直线坐标系；为空时由 planner 使用默认方向。"""

    recenter_target_in_plane: bool = True
    """是否把目标点搬到所选范围的平面中心；椎间最近点必须保留原位置。"""


def resolve_localizer_plane(
    localization_result: Dict[str, Any],
    slice_type: str,
    plane: PlaneTypeEnum,
) -> str | None:
    """读取当前状位与切法显式配置的定位图状位。

    缺少 ``slice_rules`` 时返回 ``None``，兼容不经过统一配置构造的旧阶段结果；
    正式配置会在应用启动时保证该字段存在且与重建状位不同。
    """

    slice_rules = localization_result.get("slice_rules")
    if not isinstance(slice_rules, dict):
        return None
    plane_rules = slice_rules.get(plane.value)
    if not isinstance(plane_rules, dict):
        return None
    slice_type_value = str(getattr(slice_type, "value", slice_type))
    rule = plane_rules.get(slice_type_value)
    if not isinstance(rule, dict):
        return None
    localizer_plane = rule.get("localizer_plane")
    if localizer_plane is None:
        return None
    if (
        not isinstance(localizer_plane, str)
        or localizer_plane not in {item.value for item in PlaneTypeEnum}
    ):
        raise ValueError(f"不支持的定位图参考状位: {localizer_plane}")
    if localizer_plane == plane.value:
        raise ValueError(f"定位图参考状位不能与重建状位相同: {localizer_plane}")
    return localizer_plane


def _non_negative_sampling_extension(value: Any, endpoint: str) -> float:
    """校验内部采样接口的首末额外覆盖距离。"""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"采样范围 {endpoint} 扩展必须是有限数字")
    value_mm = float(value)
    if value_mm < 0:
        raise ValueError(f"采样范围 {endpoint} 扩展不能小于 0")
    return value_mm


def anatomical_straight_frame(
    plane: PlaneTypeEnum,
    anatomical_frame: AnatomicalFrame,
    image_info: Dict[str, Any],
) -> StraightPlaneFrame:
    """
    把统一解剖坐标系映射成横、矢、冠状位切片坐标系。

    row_dir、col_dir 决定切片画面的左右和上下，normal 决定切片堆叠方向。
    三状位的画面和法向全部使用解剖轴，避免在矢状位重新引入原图倾斜。
    """

    lr = anatomical_frame.left_right_axis
    ap = anatomical_frame.anterior_posterior_axis
    si = anatomical_frame.superior_inferior_axis
    # 横状位画面使用左右/前后轴，沿头脚方向逐层推进。
    if plane == PlaneTypeEnum.横状位:
        return StraightPlaneFrame(row_dir=lr, col_dir=ap, normal=si)
    # 矢状位画面使用前后/脚头轴，并按右到左方向推进。
    if plane == PlaneTypeEnum.矢状位:
        return StraightPlaneFrame(
            row_dir=ap,
            col_dir=_scale_direction(si, -1.0),
            normal=_scale_direction(lr, -1.0),
        )
    # 冠状位画面使用左右/脚头轴，沿前后方向推进。
    if plane == PlaneTypeEnum.冠状位:
        return StraightPlaneFrame(
            row_dir=lr,
            col_dir=_scale_direction(si, -1.0),
            normal=ap,
        )
    raise ValueError(f"不支持的切割平面: {plane}")


def _scale_direction(direction: Point3D, sign: float) -> Point3D:
    """按 1 或 -1 翻转一个输入图像轴方向。"""

    return Point3D(
        direction.x * sign,
        direction.y * sign,
        direction.z * sign,
    )


def _add_direction(point: Point3D, direction: Point3D, distance: float) -> Point3D:
    """沿单位方向移动一个物理点。"""

    return Point3D(
        point.x + direction.x * distance,
        point.y + direction.y * distance,
        point.z + direction.z * distance,
    )


def _cross_direction(first: Point3D, second: Point3D) -> Point3D:
    """计算两个物理方向的叉积，用于构造右手切片显示坐标系。"""

    return Point3D(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
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
        # 兼容没有解剖坐标的旧结果时，同样按患者前→后、头→脚显示。
        # 仅校正存储轴符号，保留原始斜扫描角度。
        axes = patient_reference_axes(image_direction_matrix(image_info))
        return StraightPlaneFrame(
            row_dir=Point3D(*(float(value) for value in axes[:, 1])),
            col_dir=Point3D(*(float(value) for value in -axes[:, 2])),
            normal=Point3D(*(float(value) for value in -axes[:, 0])),
        )
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
    localization_result: Dict[str, Any],
    target_region_bbox: Dict[str, float],
    frame: StraightPlaneFrame,
) -> ProjectedExtent:
    """计算原始采样视野；最终整组居中和占比由重建阶段处理。"""

    image_info = localization_result["image"]
    if image_info.get("size_xyz"):
        points = image_physical_corners(image_info)
    else:
        points = bbox_physical_corners(target_region_bbox, image_info)
    return project_region_extent(points, frame)


def expand_projected_extent_around_anchor(
    projected_extent: ProjectedExtent,
    anchor: Point3D,
    frame: StraightPlaneFrame,
) -> ProjectedExtent:
    """以指定平面内锚点为中心扩展范围，且不裁掉原投影覆盖区。"""

    half_width = projected_extent.extent_mm.width / 2.0
    half_height = projected_extent.extent_mm.height / 2.0
    minimum_u = projected_extent.center_u - half_width
    maximum_u = projected_extent.center_u + half_width
    minimum_v = projected_extent.center_v - half_height
    maximum_v = projected_extent.center_v + half_height
    anchor_u = _dot(anchor, frame.row_dir)
    anchor_v = _dot(anchor, frame.col_dir)
    return ProjectedExtent(
        extent_mm=ExtentMM(
            width=2.0 * max(
                abs(anchor_u - minimum_u),
                abs(maximum_u - anchor_u),
            ),
            height=2.0 * max(
                abs(anchor_v - minimum_v),
                abs(maximum_v - anchor_v),
            ),
        ),
        center_u=anchor_u,
        center_v=anchor_v,
    )


def cover_segment_range(
    target_start: Point3D,
    target_end: Point3D,
    slice_thickness: float | None,
    slice_spacing: float | None,
    start_extension_mm: float = 0.0,
    end_extension_mm: float = 0.0,
    slice_image_count: int | None = None,
) -> tuple[Point3D, Point3D]:
    """计算完整覆盖有效范围的首末切片中心。"""

    range_start, range_end, _ = resolve_segment_sampling(
        target_start=target_start,
        target_end=target_end,
        slice_thickness=slice_thickness,
        slice_spacing=slice_spacing,
        slice_image_count=slice_image_count,
        start_extension_mm=start_extension_mm,
        end_extension_mm=end_extension_mm,
    )
    return range_start, range_end


def resolve_segment_sampling(
    target_start: Point3D,
    target_end: Point3D,
    slice_thickness: float | None,
    slice_spacing: float | None,
    slice_image_count: int | None = None,
    start_extension_mm: float = 0.0,
    end_extension_mm: float = 0.0,
) -> tuple[Point3D, Point3D, ResolvedSampling]:
    """结合直线目标和既有首尾扩展计算实际参数及切片中心范围。"""

    start_extension_mm = _non_negative_sampling_extension(
        start_extension_mm,
        endpoint="start",
    )
    end_extension_mm = _non_negative_sampling_extension(
        end_extension_mm,
        endpoint="end",
    )

    target_vector = subtract_points(target_end, target_start)
    target_length = vector_length(target_vector)
    effective_length = target_length + start_extension_mm + end_extension_mm
    sampling = resolve_sampling(
        effective_range_mm=effective_length,
        slice_thickness=slice_thickness,
        slice_spacing=slice_spacing,
        slice_image_count=slice_image_count,
    )
    if target_length == 0.0 and start_extension_mm == 0.0 and end_extension_mm == 0.0:
        if sampling.slice_image_count != 1:
            raise ValueError("零长度有效范围无法生成多张切片")
        midpoint = Point3D(
            (target_start.x + target_end.x) / 2.0,
            (target_start.y + target_end.y) / 2.0,
            (target_start.z + target_end.z) / 2.0,
        )
        return midpoint, midpoint, sampling
    if target_length == 0.0:
        raise ValueError("零长度目标无法确定切片范围扩展方向")

    direction = normalize_vector(target_vector)
    effective_start = Point3D(
        target_start.x - direction.x * start_extension_mm,
        target_start.y - direction.y * start_extension_mm,
        target_start.z - direction.z * start_extension_mm,
    )
    # 第一张中心向内半个层厚，若总范围超过有效范围，再向外移动均分后的距离。
    symmetric_range_difference = (
        sampling.total_range_mm - effective_length
    ) / 2.0
    start_center_offset = (
        sampling.slice_thickness / 2.0
        - symmetric_range_difference
    )
    range_start = Point3D(
        effective_start.x + direction.x * start_center_offset,
        effective_start.y + direction.y * start_center_offset,
        effective_start.z + direction.z * start_center_offset,
    )
    center_range = (
        (sampling.slice_image_count - 1)
        * sampling.slice_spacing
    )
    range_end = Point3D(
        range_start.x + direction.x * center_range,
        range_start.y + direction.y * center_range,
        range_start.z + direction.z * center_range,
    )
    return range_start, range_end, sampling


def recenter_point_in_plane(
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
    slice_type: str,
    plane_config: Any,
    localization_result: Dict[str, Any],
    segment: CutSegment,
    frame: StraightPlaneFrame,
) -> SlicePlan:
    """把待切区间、直线方向和图像范围组装成第七阶段可执行计划。"""

    face_extension = resolve_face_extension(
        localization_result,
        slice_type,
        plane_config.plane,
    )
    expanded_target_start = segment.target_start
    expanded_target_end = segment.target_end
    if face_extension is None:
        projected_extent = resolve_projected_extent(
            localization_result,
            segment.target_region_bbox,
            frame,
        )
    else:
        target_points = expand_physical_points(
            bbox_physical_corners(
                segment.target_region_bbox,
                localization_result["image"],
            ),
            localization_result["image"],
            face_extension,
        )
        extent_points = list(target_points)
        if localization_result["image"].get("size_xyz"):
            extent_points = image_physical_corners(localization_result["image"]) + extent_points
        projected_extent = project_region_extent(extent_points, frame)
        minimum_normal = min(_dot(point, frame.normal) for point in target_points)
        maximum_normal = max(_dot(point, frame.normal) for point in target_points)
        start_projection = _dot(segment.target_start, frame.normal)
        end_projection = _dot(segment.target_end, frame.normal)
        if end_projection >= start_projection:
            start_target_projection, end_target_projection = minimum_normal, maximum_normal
        else:
            start_target_projection, end_target_projection = maximum_normal, minimum_normal
        expanded_target_start = _add_direction(
            segment.target_start,
            frame.normal,
            start_target_projection - start_projection,
        )
        expanded_target_end = _add_direction(
            segment.target_end,
            frame.normal,
            end_target_projection - end_projection,
        )
    if segment.recenter_target_in_plane:
        centered_target_start = recenter_point_in_plane(
            expanded_target_start,
            projected_extent,
            frame,
        )
        centered_target_end = recenter_point_in_plane(
            expanded_target_end,
            projected_extent,
            frame,
        )
    else:
        centered_target_start = expanded_target_start
        centered_target_end = expanded_target_end
        target_anchor = Point3D(
            (expanded_target_start.x + expanded_target_end.x) / 2.0,
            (expanded_target_start.y + expanded_target_end.y) / 2.0,
            (expanded_target_start.z + expanded_target_end.z) / 2.0,
        )
        # 最近点决定真实椎间位置，不能移到整幅图像中心。输出平面会以该
        # 位置为中心，因此同步对称扩展宽高，确保原选定范围仍完整覆盖。
        projected_extent = expand_projected_extent_around_anchor(
            projected_extent,
            target_anchor,
            frame,
        )
    # 重定位后的端点或已共用平面坐标的四棱柱最近点对，都沿二者形成的
    # 最终法向计算有效范围。这样带角度切片也能保持固定层间距，并让
    # 首末厚层外缘覆盖目标边界。
    centered_range_start, centered_range_end, sampling = resolve_segment_sampling(
        centered_target_start,
        centered_target_end,
        plane_config.slice_thickness,
        plane_config.slice_spacing,
        slice_image_count=plane_config.slice_image_count,
    )
    path_points = [centered_range_start, centered_range_end]
    slices = build_plane_stack_slices(
        path_points=path_points,
        slice_spacing=sampling.slice_spacing,
        normal=frame.normal,
        row_dir=frame.row_dir,
        col_dir=frame.col_dir,
        slice_image_count=sampling.slice_image_count,
    )

    plan_key = slice_type
    if segment.group_name != "whole_range":
        plan_key = f"{plan_key}_{segment.group_name}"
    return SlicePlan(
        plan_id=(
            f"{request.slice_part.value}_{plan_key}_{plane_config.plane.value}"
        ),
        slice_part=request.slice_part.value,
        slice_type=slice_type,
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
        slice_thickness=sampling.slice_thickness,
        slice_spacing=sampling.slice_spacing,
        window_presets=list(plane_config.window_presets),
        source_labels=list(segment.source_labels),
        fill_percent=plane_config.fill_percent,
        target_start=centered_target_start,
        target_end=centered_target_end,
        localizer_plane=resolve_localizer_plane(
            localization_result,
            slice_type,
            plane_config.plane,
        ),
        config_id=plane_config.config_id,
        output_plane=plane_config.output_plane,
        sampling_details=sampling.to_dict(),
    )
