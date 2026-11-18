"""把已求解的最终区域传递给取景，防止再次使用完整标签扩大组合范围。"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from src.models import AnatomicalFrame


@dataclass(frozen=True)
class RegionFramingBounds:
    axes: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    corners: np.ndarray

    @classmethod
    def from_plan(cls, plan: dict) -> "RegionFramingBounds | None":
        geometry = plan.get("region_geometry")
        if not geometry or not geometry.get("constrained"):
            return None
        frame = AnatomicalFrame.from_dict(geometry["frame"])
        axes = np.column_stack([
            [axis.x, axis.y, axis.z] for axis in (
                frame.left_right_axis, frame.anterior_posterior_axis, frame.superior_inferior_axis,
            )
        ])
        bounds = (
            frame.bounds.left_right, frame.bounds.anterior_posterior, frame.bounds.superior_inferior,
        )
        lower = np.asarray([bound.minimum for bound in bounds])
        upper = np.asarray([bound.maximum for bound in bounds])
        corners = np.asarray(list(product(*zip(lower, upper)))) @ axes.T
        return cls(axes, lower, upper, corners)

    def select_points(self, points: np.ndarray | None, voxel_axes: np.ndarray) -> np.ndarray | None:
        """只保留与区域有正体积交集的体素；使用物理方向，不按数组z裁剪。"""
        if points is None:
            return None
        projected = points @ self.axes
        half_voxel = np.abs(voxel_axes.T @ self.axes).sum(axis=0) / 2
        selected = np.all(
            (projected + half_voxel > self.lower + 1e-7)
            & (projected - half_voxel < self.upper - 1e-7), axis=1,
        )
        return points[selected]

    def projected_bounds(self, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected = self.corners @ axes[:, :2]
        return projected.min(axis=0), projected.max(axis=0)

    def clip_bounds(self, bounds, axes: np.ndarray):
        """限制单层取景证据，体素半宽也不能把已确定的区域边界扩回去。"""
        if bounds is None:
            return None
        lower, upper = self.projected_bounds(axes)
        clipped = np.maximum(bounds[0], lower), np.minimum(bounds[1], upper)
        return clipped if np.all(clipped[1] > clipped[0]) else None
