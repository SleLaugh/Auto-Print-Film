"""从逐椎体的物理点云估计左右对称面及横断面旋转。

完整椎体包含椎弓、棘突等附件，其最长 PCA 轴会随椎体层级改变；姿态
因此用镜像匹配判断，不能把不同层级的横向和纵向 PCA 长轴直接平均。
"""

import math
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion

from src.anatomy.symmetry import estimate_reflection_symmetry_plane


def _image_geometry(image: sitk.Image | Mapping[str, Any]):
    """接受运行时图像或阶段 JSON，返回毫米坐标转换所需的几何。"""

    if isinstance(image, sitk.Image):
        spacing, origin, direction = image.GetSpacing(), image.GetOrigin(), image.GetDirection()
    else:
        spacing = image["spacing_xyz"]
        origin = image["origin_xyz"]
        direction = image["direction"]
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float).reshape(3, 3)
    if (
        spacing.shape != (3,)
        or origin.shape != (3,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(direction))
        or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6)
    ):
        raise ValueError("脊柱姿态估计需要有效的三维图像几何")
    # 输入网格允许反射或头脚反序；镜像匹配的内部参考基必须是右手系。
    axes = np.column_stack((direction[:, 0], direction[:, 1],
                            np.cross(direction[:, 0], direction[:, 1])))
    return spacing, origin, direction, axes


def _point_dict(point: np.ndarray) -> dict[str, float]:
    return {axis: float(value) for axis, value in zip("xyz", point)}


def estimate_spine_label_symmetry(
    indices_zyx: np.ndarray,
    image: sitk.Image | Mapping[str, Any],
    *,
    max_points: int = 3000,
) -> dict[str, Any]:
    """给一个完整椎体生成可序列化的可靠左右轴。

    输入为该标签的 z/y/x 稀疏体素索引，不分配全尺寸 Mask。采样在转换为
    物理坐标前完成；每个椎体最多 3000 点，结果不依赖进程数或随机状态。
    ``reliable`` 只在匹配通过质量检查或确认为近正位时为真。拒绝的结果仍
    带输入左右轴供诊断，调用方不能把这个回退轴当成解剖测量。
    """

    spacing, origin, direction, axes = _image_geometry(image)
    indices = np.asarray(indices_zyx)
    if indices.ndim != 2 or indices.shape[1:] != (3,):
        raise ValueError("椎体索引必须为 N×3 的 z/y/x 坐标")
    if not np.all(np.isfinite(indices)):
        raise ValueError("椎体索引包含非有限坐标")
    if not isinstance(max_points, (int, np.integer)) or max_points < 64:
        raise ValueError("逐椎体姿态采样上限至少为 64")
    point_count = len(indices)
    # 体积内部的近邻误差主要反映随机抽样密度，会掩盖 1–3 度的小角度。
    # 使用标签局部边界提供形状证据；只分配局部 bbox，禁止全 CT 尺寸 Mask。
    if point_count and np.issubdtype(indices.dtype, np.integer):
        minimum, maximum = indices.min(axis=0), indices.max(axis=0)
        shape = maximum - minimum + 1
        if int(np.prod(shape)) <= 8_000_000:
            mask = np.zeros(tuple(shape), dtype=bool)
            local_indices = indices - minimum
            mask[tuple(local_indices.T)] = True
            boundary = mask & ~binary_erosion(mask, border_value=0)
            indices = np.argwhere(boundary) + minimum
    if len(indices) > max_points:
        selection = np.sort(np.random.default_rng(0).choice(len(indices), max_points, replace=False))
        indices = indices[selection]
    points = origin + (indices[:, [2, 1, 0]].astype(float) * spacing) @ direction.T
    normal, plane_center, diagnostics = estimate_reflection_symmetry_plane(
        points, axes, max_points=max_points,
    )
    diagnostics = {**diagnostics, "source_point_count": int(point_count)}
    reliable = diagnostics.get("status") in {"accepted", "unchanged"}
    # 候选绕图像 z 轴的分量必须返回图像 XY 基，而非直接取患者物理 X/Y。
    axial_angle = math.degrees(math.atan2(float(normal @ direction[:, 1]),
                                         float(normal @ direction[:, 0])))
    return {
        "symmetry_left_right_axis": _point_dict(normal),
        "symmetry_plane_center": _point_dict(plane_center),
        "symmetry_diagnostics": diagnostics,
        "axial_angle_degrees": float(axial_angle),
        "point_count": int(point_count),
        "reliable": bool(reliable),
    }


def estimate_spine_axial_symmetry(
    label_indices_zyx: Mapping[str, np.ndarray],
    image: sitk.Image | Mapping[str, Any],
    *,
    threshold_degrees: float = 3.0,
    min_labels: int = 2,
    max_angle_spread_degrees: float = 15.0,
    max_points_per_label: int = 3000,
) -> dict[str, Any]:
    """稳健聚合可靠的逐椎体镜像角，供第三阶段生成 XY 回正候选。

    先在物理空间拟合，再将左右轴投回图像 XY；不会受层厚、斜扫描或
    PCA 横纵主轴交换影响。离群椎体只退出聚合，不能否决其余一致标签。
    """

    if min_labels < 1 or threshold_degrees < 0.0 or max_angle_spread_degrees <= 0.0:
        raise ValueError("脊柱旋转聚合阈值无效")
    labels = []
    skipped = []
    for label_name, indices in label_indices_zyx.items():
        pose = estimate_spine_label_symmetry(indices, image, max_points=max_points_per_label)
        item = {
            "label_name": label_name,
            "point_count": pose["point_count"],
            "angle_degrees": pose["axial_angle_degrees"],
            "symmetry_diagnostics": pose["symmetry_diagnostics"],
        }
        if pose["reliable"]:
            labels.append(item)
        else:
            skipped.append(label_name)
    if labels:
        median = float(np.median([item["angle_degrees"] for item in labels]))
        inliers = [item for item in labels
                   if abs(item["angle_degrees"] - median) <= max_angle_spread_degrees]
        skipped.extend(item["label_name"] for item in labels if item not in inliers)
    else:
        median, inliers = 0.0, []
    enough = len(inliers) >= min_labels
    # 平方根权重限制大椎体对结果的支配，近正位标签也须参与整体姿态判断。
    angle = (float(np.average(
        [item["angle_degrees"] for item in inliers],
        weights=[math.sqrt(item["point_count"]) for item in inliers],
    )) if enough else 0.0)
    spread = max((abs(item["angle_degrees"] - angle) for item in inliers), default=0.0)
    should_correct = enough and abs(angle) > threshold_degrees
    reason = ("可用于镜像聚合的可靠椎体数量不足" if not enough else
              "候选角未超过回正阈值" if not should_correct else "")
    return {
        "method": "vertebrae_reflection_symmetry_aggregation",
        "point_count": sum(item["point_count"] for item in inliers),
        "min_points": 64,
        "min_vertebrae": int(min_labels),
        "threshold_degrees": float(threshold_degrees),
        "max_label_angle_spread_degrees": float(max_angle_spread_degrees),
        "angle_spread_degrees": float(spread),
        "median_label_angle_degrees": median,
        "angle_degrees": angle,
        "abs_angle_degrees": abs(angle),
        "enough_points": enough,
        "should_correct": should_correct,
        "used_label_names": [item["label_name"] for item in inliers],
        "skipped_label_names": skipped,
        "label_angles": labels,
        "reason": reason,
    }
