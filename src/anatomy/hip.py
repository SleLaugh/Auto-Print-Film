#!/src/anatomy/hip.py
# -*- coding: utf-8 -*-

"""根据双侧髋骨的镜像对称平面建立三状位共用的解剖坐标。"""

from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion

from src.anatomy.symmetry import (
    build_symmetry_frame_axes,
    estimate_reflection_symmetry_plane,
)
from src.anatomy.orientation import patient_reference_axes
from src.models import (
    AnatomicalBounds,
    AnatomicalFrame,
    AnatomicalGeometryResult,
    AxisBounds,
    Point3D,
)


_SUPERIOR_CAP_THICKNESS_MM = 5.0
_ROBUST_SUPERIOR_PERCENTILE = 99.0
_MIN_COMPONENT_POINTS = 8
_MIN_BILATERAL_VOLUME_RATIO = 0.25


def _point(vector: np.ndarray) -> Point3D:
    """把 NumPy 三维物理坐标转换为阶段间统一使用的点实体。"""

    return Point3D(float(vector[0]), float(vector[1]), float(vector[2]))


def _point_dict(vector: np.ndarray) -> dict[str, float]:
    """把三维物理坐标转换为可写入调试 JSON 的字典。"""

    return _point(vector).to_dict()


def _dicom_axes(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从图像方向矩阵读取左右、前后和头脚三个 DICOM 物理轴。"""

    axes = patient_reference_axes(np.asarray(image.GetDirection()))
    return tuple(axes[:, axis] for axis in range(3))


def _physical_points(mask: np.ndarray, image: sitk.Image, side_name: str) -> np.ndarray:
    """把指定侧 z/y/x Mask 的体素中心批量转换为患者物理坐标。"""

    if mask.ndim != 3:
        raise ValueError(f"{side_name}髋骨 Mask 必须是三维数组")
    if tuple(mask.shape[::-1]) != tuple(image.GetSize()):
        raise ValueError(f"{side_name}髋骨 Mask 尺寸与原图不一致")
    indices_zyx = np.argwhere(mask)
    if not len(indices_zyx):
        raise ValueError(f"{side_name}髋骨 Mask 为空")
    indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return origin + (indices_xyz * spacing) @ direction.T


def _retain_largest_component(
    mask: np.ndarray,
    side_name: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """每侧只保留最大连通主体，避免孤立误分割改变姿态和切片范围。"""

    binary = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
    components = sitk.GetArrayFromImage(sitk.ConnectedComponent(binary))
    sizes = np.bincount(components.ravel())
    if len(sizes) <= 1:
        raise ValueError(f"{side_name}髋骨 Mask 为空")
    component_id = int(np.argmax(sizes[1:]) + 1)
    retained_mask = components == component_id
    retained_count = int(np.count_nonzero(retained_mask))
    if retained_count < _MIN_COMPONENT_POINTS:
        raise ValueError(f"{side_name}髋骨主体体素太少，无法估计姿态")
    original_count = int(np.count_nonzero(mask))
    return retained_mask, {
        "component_count": int(len(sizes) - 1),
        "retained_point_count": retained_count,
        "discarded_point_count": original_count - retained_count,
    }


def _outer_superior_landmark(
    points: np.ndarray,
    *,
    lr_axis: np.ndarray,
    si_axis: np.ndarray,
    combined_lr_center: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """在稳健顶部 5 mm 内取远离身体中线的外侧半区中心。"""

    si_projection = points @ si_axis
    robust_top = float(np.percentile(si_projection, _ROBUST_SUPERIOR_PERCENTILE))
    cap = points[si_projection >= robust_top - _SUPERIOR_CAP_THICKNESS_MM]
    if not len(cap):
        raise ValueError("髋骨顶部 5mm 区域为空")

    lr_projection = cap @ lr_axis
    cap_center = float(np.mean(lr_projection))
    cap_median = float(np.median(lr_projection))
    # 不依赖输入 label 的左右符号：哪一侧位于整体中线高值方向，就选择该侧
    # LR 投影较大的外半区；另一侧选择较小的外半区。
    if cap_center >= combined_lr_center:
        outer = cap[lr_projection >= cap_median]
    else:
        outer = cap[lr_projection <= cap_median]
    if not len(outer):
        raise ValueError("髋骨外侧顶部区域为空")
    return np.mean(outer, axis=0), {
        "robust_superior_projection": robust_top,
        "cap_point_count": int(len(cap)),
        "outer_cap_point_count": int(len(outer)),
    }


def _axis_bounds(points: np.ndarray, axis: np.ndarray) -> AxisBounds:
    """计算目标物理点在指定单位轴上的投影边界。"""

    projection = points @ axis
    return AxisBounds(float(np.min(projection)), float(np.max(projection)))


def _bilateral_quality_reason(
    left_component: np.ndarray,
    right_component: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    spacing: tuple[float, ...],
) -> str:
    """拒绝重叠、严重单侧缺损和两侧质心无法区分的姿态估计。"""

    if np.any(left_component & right_component):
        return "左右髋骨主体存在重叠"
    counts = (len(left_points), len(right_points))
    if min(counts) / max(counts) < _MIN_BILATERAL_VOLUME_RATIO:
        return "左右髋骨主体体积差异过大，可能存在单侧缺损"
    center_distance = float(np.linalg.norm(
        np.mean(left_points, axis=0) - np.mean(right_points, axis=0)
    ))
    if center_distance < 2.0 * min(spacing):
        return "左右髋骨质心距离过小，无法确定双侧方向"
    return ""


def _boundary_physical_points(mask: np.ndarray, image: sitk.Image) -> np.ndarray:
    """用六邻域主体边界拟合镜像面，避免内部密集体素稀释轮廓误差。"""

    boundary = mask & ~binary_erosion(mask, border_value=0)
    return _physical_points(boundary, image, "双侧主体边界")


def estimate_hip_frame_with_diagnostics(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    image: sitk.Image,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """由双侧主体的镜像对称性估计左右轴，生成统一的三维刚性坐标。

    左右轴保留横状面旋转和左右高低倾斜；前后轴由输入 AP 轴在左右轴的
    正交平面内投影得到，不根据骨盆形态额外估计俯仰。顶部 AB 仅供诊断，
    不再决定切片角度。质量不足时保留输入方向并发出警告；最终三个方向的
    覆盖范围始终重新投影全部保留主体，不能用采样点或 AB 代替完整范围。
    """

    left_component, left_stats = _retain_largest_component(left_mask, "左侧")
    right_component, right_stats = _retain_largest_component(right_mask, "右侧")
    left_points = _physical_points(left_component, image, "左侧")
    right_points = _physical_points(right_component, image, "右侧")
    all_points = np.concatenate((left_points, right_points), axis=0)
    dicom_lr, dicom_ap, dicom_si = _dicom_axes(image)
    reference_axes = np.column_stack((dicom_lr, dicom_ap, dicom_si))
    combined_lr_center = float(np.mean(all_points @ dicom_lr))

    left_landmark, left_cap_stats = _outer_superior_landmark(
        left_points,
        lr_axis=dicom_lr,
        si_axis=dicom_si,
        combined_lr_center=combined_lr_center,
    )
    right_landmark, right_cap_stats = _outer_superior_landmark(
        right_points,
        lr_axis=dicom_lr,
        si_axis=dicom_si,
        combined_lr_center=combined_lr_center,
    )

    quality_reason = _bilateral_quality_reason(
        left_component,
        right_component,
        left_points,
        right_points,
        image.GetSpacing(),
    )
    if quality_reason:
        lr_axis = dicom_lr
        symmetry_center = np.mean(all_points, axis=0)
        symmetry_diagnostics = {
            "accepted": False,
            "status": "rejected",
            "reason": quality_reason,
            "warnings": [quality_reason],
            "point_count": int(len(all_points)),
            "sample_count": 0,
            "angle_degrees": 0.0,
        }
    else:
        boundary_points = np.concatenate((
            _boundary_physical_points(left_component, image),
            _boundary_physical_points(right_component, image),
        ))
        lr_axis, symmetry_center, symmetry_diagnostics = (
            estimate_reflection_symmetry_plane(
                boundary_points,
                reference_axes,
            )
        )
    accepted = bool(symmetry_diagnostics["accepted"])
    if not accepted:
        lr_axis = dicom_lr
    axes = build_symmetry_frame_axes(
        np.asarray(lr_axis),
        reference_axes,
    )
    lr_axis, ap_axis, si_axis = axes.T

    warnings = list(symmetry_diagnostics.get("warnings", []))
    rejected = symmetry_diagnostics.get("status") == "rejected"
    if rejected and not warnings:
        warnings.append(str(symmetry_diagnostics.get("reason") or "镜像对称估计质量不足"))
    warnings = [f"髋骨姿态回正未采用，保留输入方向: {warning}" for warning in warnings]

    ab_plane_projection = float(np.mean([
        np.dot(left_landmark, si_axis),
        np.dot(right_landmark, si_axis),
    ]))
    # 对称性拟合只负责方向；覆盖边界必须来自完整主体，避免优化采样或顶部
    # 诊断点遗漏远端，使三状位中的任意一组切片裁掉真实髋骨结构。
    superior_inferior_bounds = _axis_bounds(all_points, si_axis)
    if (
        superior_inferior_bounds.maximum
        <= superior_inferior_bounds.minimum
    ):
        raise ValueError("髋骨头脚投影范围无效")

    frame = AnatomicalFrame(
        origin=_point(np.mean(all_points, axis=0)),
        left_right_axis=_point(lr_axis),
        anterior_posterior_axis=_point(ap_axis),
        superior_inferior_axis=_point(si_axis),
        bounds=AnatomicalBounds(
            left_right=_axis_bounds(all_points, lr_axis),
            anterior_posterior=_axis_bounds(all_points, ap_axis),
            superior_inferior=superior_inferior_bounds,
        ),
        method="hip_reflection_symmetry" if accepted else "hip_dicom_axes",
        confidence=(
            float(symmetry_diagnostics.get("confidence", 1.0))
            if accepted
            else (0.0 if rejected else 1.0)
        ),
        applied=accepted,
        warnings=warnings,
    )
    diagnostics: dict[str, Any] = {
        "top_cap_thickness_mm": _SUPERIOR_CAP_THICKNESS_MM,
        "robust_superior_percentile": _ROBUST_SUPERIOR_PERCENTILE,
        "left_landmark": _point_dict(left_landmark),
        "right_landmark": _point_dict(right_landmark),
        "left_component_count": left_stats["component_count"],
        "right_component_count": right_stats["component_count"],
        "left_retained_point_count": left_stats["retained_point_count"],
        "right_retained_point_count": right_stats["retained_point_count"],
        "left_discarded_point_count": left_stats["discarded_point_count"],
        "right_discarded_point_count": right_stats["discarded_point_count"],
        "left_cap": left_cap_stats,
        "right_cap": right_cap_stats,
        "symmetry": symmetry_diagnostics,
        "symmetry_plane_center": _point_dict(np.asarray(symmetry_center)),
        "pitch_source": "projected_input_anterior_posterior_axis",
        "ab_role": "diagnostic_only",
        "ab_plane_projection_mm": ab_plane_projection,
        "coverage_superior_projection_mm": superior_inferior_bounds.maximum,
        "coverage_inferior_projection_mm": superior_inferior_bounds.minimum,
        "superior_shift_from_ab_mm": (
            superior_inferior_bounds.maximum - ab_plane_projection
        ),
        "left_right_axis": _point_dict(lr_axis),
        "anterior_posterior_axis": _point_dict(ap_axis),
        "superior_inferior_axis": _point_dict(si_axis),
    }
    return AnatomicalGeometryResult(global_frame=frame, local_frames={}), diagnostics
