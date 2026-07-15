#!/src/models/anatomy.py
# -*- coding: utf-8 -*-

"""第三步输出给第四步使用的统一解剖坐标实体。"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict

from .reconstruction import Point3D


_VECTOR_TOLERANCE = 1e-6


def _point_from_dict(value: Dict[str, Any]) -> Point3D:
    """把 JSON 字典中的 x/y/z 数值还原为物理三维点。"""

    return Point3D(float(value["x"]), float(value["y"]), float(value["z"]))


def _components(point: Point3D) -> tuple[float, float, float]:
    """按 x、y、z 顺序读取三维点或方向向量的浮点分量。"""

    return float(point.x), float(point.y), float(point.z)


def _dot(left: Point3D, right: Point3D) -> float:
    """计算两个三维方向的点积，用于长度及正交性校验。"""

    return sum(a * b for a, b in zip(_components(left), _components(right)))


def _cross(left: Point3D, right: Point3D) -> tuple[float, float, float]:
    """计算两个三维方向的叉积，用于校验解剖轴的右手关系。"""

    lx, ly, lz = _components(left)
    rx, ry, rz = _components(right)
    return ly * rz - lz * ry, lz * rx - lx * rz, lx * ry - ly * rx


@dataclass(frozen=True)
class AxisBounds:
    """目标物理点在一个解剖单位轴上的毫米投影范围。"""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """校验投影边界为有限数值，且最小值不大于最大值。"""

        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("解剖轴范围必须是有限数值")
        if self.minimum > self.maximum:
            raise ValueError("解剖轴范围必须满足 minimum <= maximum")

    def to_dict(self) -> Dict[str, float]:
        """转换为可直接写入 JSON 的轴范围字典。"""

        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AxisBounds":
        """从轴范围字典恢复实体，并复用实体自身的合法性校验。"""

        return cls(float(value["minimum"]), float(value["maximum"]))


@dataclass(frozen=True)
class AnatomicalBounds:
    """目标在左右、前后、头脚三个解剖轴上的投影范围。"""

    left_right: AxisBounds
    anterior_posterior: AxisBounds
    superior_inferior: AxisBounds

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """序列化左右、前后、头脚三个方向的目标投影范围。"""

        return {
            "left_right": self.left_right.to_dict(),
            "anterior_posterior": self.anterior_posterior.to_dict(),
            "superior_inferior": self.superior_inferior.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnatomicalBounds":
        """从统一解剖边界字典恢复三个方向的投影范围。"""

        return cls(
            left_right=AxisBounds.from_dict(value["left_right"]),
            anterior_posterior=AxisBounds.from_dict(value["anterior_posterior"]),
            superior_inferior=AxisBounds.from_dict(value["superior_inferior"]),
        )


@dataclass(frozen=True)
class AnatomicalFrame:
    """一个目标整体或局部区间的物理解剖坐标系。"""

    origin: Point3D
    left_right_axis: Point3D
    anterior_posterior_axis: Point3D
    superior_inferior_axis: Point3D
    bounds: AnatomicalBounds
    method: str
    confidence: float
    applied: bool
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """校验坐标有限、三轴单位正交且满足右手系，并检查元数据。"""

        points = (
            self.origin,
            self.left_right_axis,
            self.anterior_posterior_axis,
            self.superior_inferior_axis,
        )
        if any(not math.isfinite(value) for point in points for value in _components(point)):
            raise ValueError("解剖坐标必须是有限数值")
        axes = (
            self.left_right_axis,
            self.anterior_posterior_axis,
            self.superior_inferior_axis,
        )
        if any(abs(math.sqrt(_dot(axis, axis)) - 1.0) > _VECTOR_TOLERANCE for axis in axes):
            raise ValueError("解剖方向必须是单位向量")
        if any(
            abs(_dot(axes[left], axes[right])) > _VECTOR_TOLERANCE
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("三个解剖方向必须两两正交")
        cross = _cross(self.left_right_axis, self.anterior_posterior_axis)
        if sum(a * b for a, b in zip(cross, _components(self.superior_inferior_axis))) < 1.0 - _VECTOR_TOLERANCE:
            raise ValueError("三个解剖方向必须组成右手坐标系")
        if not self.method.strip():
            raise ValueError("解剖坐标 method 不能为空")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("解剖坐标 confidence 必须位于 0.0 到 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """把单个解剖坐标系序列化为阶段间传递使用的字典。"""

        return {
            "origin": self.origin.to_dict(),
            "left_right_axis": self.left_right_axis.to_dict(),
            "anterior_posterior_axis": self.anterior_posterior_axis.to_dict(),
            "superior_inferior_axis": self.superior_inferior_axis.to_dict(),
            "bounds": self.bounds.to_dict(),
            "method": self.method,
            "confidence": self.confidence,
            "applied": self.applied,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnatomicalFrame":
        """从阶段结果字典恢复单个解剖坐标系。"""

        return cls(
            origin=_point_from_dict(value["origin"]),
            left_right_axis=_point_from_dict(value["left_right_axis"]),
            anterior_posterior_axis=_point_from_dict(value["anterior_posterior_axis"]),
            superior_inferior_axis=_point_from_dict(value["superior_inferior_axis"]),
            bounds=AnatomicalBounds.from_dict(value["bounds"]),
            method=str(value["method"]),
            confidence=float(value["confidence"]),
            applied=bool(value["applied"]),
            warnings=[str(item) for item in value.get("warnings", [])],
        )


@dataclass(frozen=True)
class AnatomicalGeometryResult:
    """阶段三统一解剖几何输出；整切使用全局坐标，区间切使用局部坐标。"""

    global_frame: AnatomicalFrame
    local_frames: dict[str, AnatomicalFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """确保每个局部坐标系都有可追踪的非空区间名称。"""

        if any(not str(name).strip() for name in self.local_frames):
            raise ValueError("局部解剖坐标名称不能为空")

    def to_dict(self) -> Dict[str, Any]:
        """序列化全局坐标系及按区间命名的局部坐标系。"""

        return {
            "global_frame": self.global_frame.to_dict(),
            "local_frames": {
                name: frame.to_dict()
                for name, frame in self.local_frames.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnatomicalGeometryResult":
        """从第三步输出字典恢复完整解剖几何结果。"""

        return cls(
            global_frame=AnatomicalFrame.from_dict(value["global_frame"]),
            local_frames={
                str(name): AnatomicalFrame.from_dict(frame)
                for name, frame in value.get("local_frames", {}).items()
            },
        )
