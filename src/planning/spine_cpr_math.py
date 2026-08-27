"""CPR 规划使用的基础三维向量运算。"""

from typing import List

import numpy as np

from src.models import Point3D


def midpoint(points: List[Point3D]) -> Point3D:
    """返回曲线采样点的物理中心。"""

    middle = np.mean([[point.x, point.y, point.z] for point in points], axis=0)
    return Point3D(float(middle[0]), float(middle[1]), float(middle[2]))


def dot(left: Point3D, right: Point3D) -> float:
    """计算两个 Point3D 的点积。"""

    return left.x * right.x + left.y * right.y + left.z * right.z


def scale(vector: Point3D, factor: float) -> Point3D:
    """缩放三维向量。"""

    return Point3D(vector.x * factor, vector.y * factor, vector.z * factor)


def add(left: Point3D, right: Point3D) -> Point3D:
    """相加三维点或向量。"""

    return Point3D(left.x + right.x, left.y + right.y, left.z + right.z)


def cross(left: Point3D, right: Point3D) -> Point3D:
    """计算三维叉积。"""

    return Point3D(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )
