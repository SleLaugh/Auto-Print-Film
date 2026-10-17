"""用已通过质量检查的物理对称面约束取景中心，不改变切面方向。"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SymmetryFramingPlane:
    """第五阶段明确给出的镜像面，位置不能由一般目标质心替代。"""

    normal_xyz: tuple[float, float, float]
    point_xyz: tuple[float, float, float]
    source_labels: tuple[str, ...] = ()

    def centered_projection(
        self,
        axes: np.ndarray,
        normal_depth: float,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        """沿切面内 LR 投影移动 bbox 中心，其余分量和法向深度不变。

        矢状面等 LR 接近切面法向的情况不能通过面内平移回到对称面。
        对称中心落到有效目标范围之外时也保留 bbox，避免错误面点造成空白。
        """

        normal = np.asarray(self.normal_xyz, dtype=float)
        point = np.asarray(self.point_xyz, dtype=float)
        if (
            normal.shape != (3,) or point.shape != (3,)
            or not np.all(np.isfinite(normal)) or not np.all(np.isfinite(point))
            or float(np.linalg.norm(normal)) < 1e-8
        ):
            return None, "invalid_symmetry_plane"
        normal = normal / np.linalg.norm(normal)
        projected_lr = normal @ axes[:, :2]
        length_squared = float(projected_lr @ projected_lr)
        # 只接受 LR 主要位于当前切面内的取景，排除近矢位的病态大平移。
        if length_squared < 0.5:
            return None, "left_right_out_of_plane"
        midpoint = (lower + upper) / 2
        physical_midpoint = axes[:, :2] @ midpoint + axes[:, 2] * normal_depth
        shift = float(normal @ (point - physical_midpoint)) / length_squared * projected_lr
        target = midpoint + shift
        tolerance = 1e-7
        if (
            not np.all(np.isfinite(target))
            or np.any(target < lower - tolerance) or np.any(target > upper + tolerance)
        ):
            return None, "symmetry_plane_outside_target"
        return target, "applied"


def _physical_vector(value) -> np.ndarray | None:
    if not isinstance(value, dict):
        return None
    try:
        vector = np.array([value[axis] for axis in "xyz"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return vector if np.all(np.isfinite(vector)) else None


def resolve_symmetry_framing_plane(
    plan: dict,
    anatomical_analysis: dict,
) -> SymmetryFramingPlane | None:
    """为可靠椎间双面计划提供镜像中心。

    不从 global_frame.origin 推断镜像面位置；它通常只是完整目标的体素质心。
    其他部位和证据不完整的计划继续使用既有 bbox 取景契约。
    """

    if plan.get("geometry_type") != "plane_stack":
        return None
    labels = plan.get("source_labels") or []
    if plan.get("slice_type") != "椎间隙":
        return None
    if len(labels) != 2 or len(set(labels)) != 2:
        return None
    annotations = (anatomical_analysis.get("spine_geometry") or {}).get(
        "filtered_annotation_centroids", {}
    )
    axes, centers = [], []
    for label in labels:
        annotation = annotations.get(label)
        if not isinstance(annotation, dict):
            return None
        diagnostic = annotation.get("symmetry_diagnostics")
        if not isinstance(diagnostic, dict) or diagnostic.get("status") not in {
            "accepted", "unchanged",
        }:
            return None
        normal = _physical_vector(annotation.get("symmetry_left_right_axis"))
        center = _physical_vector(annotation.get("symmetry_plane_center"))
        if normal is None or center is None or np.linalg.norm(normal) < 1e-8:
            return None
        normal /= np.linalg.norm(normal)
        if axes and float(normal @ axes[0]) < 0:
            normal = -normal
        axes.append(normal)
        centers.append(center)
    if len(axes) == 2 and float(axes[0] @ axes[1]) < math.cos(math.radians(25.0)):
        return None
    normal = np.sum(axes, axis=0)
    normal /= np.linalg.norm(normal)
    return SymmetryFramingPlane(
        normal_xyz=tuple(map(float, normal)),
        point_xyz=tuple(map(float, np.mean(centers, axis=0))),
        source_labels=tuple(labels),
    )
