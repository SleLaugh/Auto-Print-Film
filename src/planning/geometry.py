#!/src/planning/geometry.py
# -*- coding: utf-8 -*-

"""
阶段四几何工具

这里不写任何部位业务，只处理点、方向和切片清单展开。
"""

import math
from typing import Any, Dict, List

from src.models import PlannedSlice, Point3D


def vector_length(vector: Point3D) -> float:
    """计算三维向量长度。"""

    return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def normalize_vector(vector: Point3D) -> Point3D:
    """把方向向量转成单位向量。"""

    length = vector_length(vector)
    if length == 0:
        raise ValueError("方向向量长度不能为 0")
    return Point3D(vector.x / length, vector.y / length, vector.z / length)


def subtract_points(end: Point3D, start: Point3D) -> Point3D:
    """计算从 start 指向 end 的向量。"""

    return Point3D(end.x - start.x, end.y - start.y, end.z - start.z)


def image_direction_matrix(image_info: Dict[str, Any]) -> List[List[float]]:
    """
    读取 SimpleITK 的 direction，并整理成 3x3 矩阵。

    SimpleITK 的 direction 是按行展开的 3x3 矩阵。物理坐标换算不能只用
    origin + index * spacing；真实 CT/NIfTI 常会带方向矩阵，忽略它会让
    第四步生成的切割线在 ITK-SNAP 中偏到错误层面。
    """

    direction = image_info.get("direction") or [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    if len(direction) != 9:
        raise ValueError("图像 direction 必须是 9 个数")
    return [
        [float(direction[0]), float(direction[1]), float(direction[2])],
        [float(direction[3]), float(direction[4]), float(direction[5])],
        [float(direction[6]), float(direction[7]), float(direction[8])],
    ]


def voxel_index_to_physical(
    index_xyz: Point3D,
    image_info: Dict[str, Any],
) -> Point3D:
    """把 x/y/z 体素索引按 spacing、origin、direction 转成物理坐标。"""

    spacing = image_info["spacing_xyz"]
    origin = image_info["origin_xyz"]
    matrix = image_direction_matrix(image_info)
    scaled_index = [
        index_xyz.x * spacing[0],
        index_xyz.y * spacing[1],
        index_xyz.z * spacing[2],
    ]
    return Point3D(
        x=origin[0] + sum(matrix[0][axis] * scaled_index[axis] for axis in range(3)),
        y=origin[1] + sum(matrix[1][axis] * scaled_index[axis] for axis in range(3)),
        z=origin[2] + sum(matrix[2][axis] * scaled_index[axis] for axis in range(3)),
    )


def image_axis_direction(image_info: Dict[str, Any], axis_index: int) -> Point3D:
    """
    读取某个体素轴在物理空间中的方向。

    axis_index 为 0/1/2，分别表示图像 x/y/z 轴。direction 矩阵的列向量
    才是对应体素轴的物理方向。
    """

    if axis_index not in (0, 1, 2):
        raise ValueError("axis_index 只能是 0、1、2")
    matrix = image_direction_matrix(image_info)
    return Point3D(
        x=matrix[0][axis_index],
        y=matrix[1][axis_index],
        z=matrix[2][axis_index],
    )


def interpolate_points(start: Point3D, end: Point3D, spacing: float) -> List[Point3D]:
    """按层间距从起点到终点生成切片中心点。"""

    if spacing <= 0:
        raise ValueError("层间距必须大于 0")

    vector = subtract_points(end, start)
    distance = vector_length(vector)
    if distance == 0:
        return [start]

    direction = normalize_vector(vector)
    tolerance = max(distance, spacing, 1.0) * 1e-9
    step_count = int(math.floor((distance + tolerance) / spacing))
    points = [
        Point3D(
            start.x + direction.x * spacing * index,
            start.y + direction.y * spacing * index,
            start.z + direction.z * spacing * index,
        )
        for index in range(step_count + 1)
    ]

    return points


def build_plane_stack_slices(
    path_points: List[Point3D],
    slice_spacing: float,
    normal: Point3D,
    row_dir: Point3D,
    col_dir: Point3D,
) -> List[PlannedSlice]:
    """把普通平面切割路径展开成第五步可执行的切片清单。"""

    if len(path_points) < 2:
        raise ValueError("普通平面切割路径至少需要 2 个点")

    centers = interpolate_points(path_points[0], path_points[-1], slice_spacing)
    normalized_normal = normalize_vector(normal)
    normalized_row = normalize_vector(row_dir)
    normalized_col = normalize_vector(col_dir)

    return [
        PlannedSlice(
            index=index + 1,
            center=center,
            normal=normalized_normal,
            row_dir=normalized_row,
            col_dir=normalized_col,
        )
        for index, center in enumerate(centers)
    ]
