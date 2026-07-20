#!/src/anatomy/hip.py
# -*- coding: utf-8 -*-

"""根据左右髋骨外侧顶部估计髋骨横状位的解剖坐标。"""

import math
from typing import Any

import numpy as np
import SimpleITK as sitk

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


def _normalize(vector: np.ndarray, error_message: str) -> np.ndarray:
    """把物理方向归一化，并为无效方向给出明确的业务异常。"""

    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1e-8:
        raise ValueError(error_message)
    return vector / length


def _point(vector: np.ndarray) -> Point3D:
    """把 NumPy 三维物理坐标转换为阶段间统一使用的点实体。"""

    return Point3D(float(vector[0]), float(vector[1]), float(vector[2]))


def _point_dict(vector: np.ndarray) -> dict[str, float]:
    """把三维物理坐标转换为可写入调试 JSON 的字典。"""

    return _point(vector).to_dict()


def _dicom_axes(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从图像方向矩阵读取左右、前后和头脚三个 DICOM 物理轴。"""

    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return tuple(
        _normalize(direction[:, axis], "DICOM 图像方向无效")
        for axis in range(3)
    )


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
    """每侧只保留最大连通主体，避免孤立误分割改变顶部和切片范围。"""

    binary = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
    components = sitk.GetArrayFromImage(sitk.ConnectedComponent(binary))
    sizes = np.bincount(components.ravel())
    if len(sizes) <= 1:
        raise ValueError(f"{side_name}髋骨 Mask 为空")
    component_id = int(np.argmax(sizes[1:]) + 1)
    retained_mask = components == component_id
    retained_count = int(np.count_nonzero(retained_mask))
    if retained_count < _MIN_COMPONENT_POINTS:
        raise ValueError(f"{side_name}髋骨主体体素太少，无法计算顶部方向")
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


def estimate_hip_frame_with_diagnostics(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    image: sitk.Image,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """用左右髋骨外侧顶部 AB 线建立倾斜横切坐标及调试信息。

    AB 只保留冠状面中的左右和头脚分量，前后轴沿用 DICOM 方向。这样仅改变
    横状位角度，不会引入额外的前后旋转。AB 只负责确定角度；最终上下界由
    两个髋骨主体全部物理点沿最终法向的投影极值决定，保证整组平行切片完整
    覆盖目标。
    """

    left_component, left_stats = _retain_largest_component(left_mask, "左侧")
    right_component, right_stats = _retain_largest_component(right_mask, "右侧")
    left_points = _physical_points(left_component, image, "左侧")
    right_points = _physical_points(right_component, image, "右侧")
    all_points = np.concatenate((left_points, right_points), axis=0)
    dicom_lr, dicom_ap, dicom_si = _dicom_axes(image)
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

    ab_vector = right_landmark - left_landmark
    coronal_ab = ab_vector - float(np.dot(ab_vector, dicom_ap)) * dicom_ap
    lr_axis = _normalize(coronal_ab, "髋骨左右外侧顶部 AB 长度无效")
    if float(np.dot(lr_axis, dicom_lr)) < 0.0:
        lr_axis = -lr_axis
    ap_axis = dicom_ap
    si_axis = _normalize(
        np.cross(lr_axis, ap_axis),
        "髋骨横切坐标无法正交化",
    )
    if float(np.dot(si_axis, dicom_si)) < 0.0:
        # 同时翻转平面内左右轴，保持右手系且使头脚法向与 DICOM 同向。
        lr_axis = -lr_axis
        si_axis = -si_axis

    ab_plane_projection = float(np.mean([
        np.dot(left_landmark, si_axis),
        np.dot(right_landmark, si_axis),
    ]))
    # AB 区域中心只用于稳健估计角度，不能直接作为头侧覆盖边界。角度确定后
    # 必须重新投影完整主体，使用与 AB 平行的最外侧支撑平面包住全部髋骨点。
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
        method="hip_outer_superior_caps",
        confidence=1.0,
        applied=True,
        warnings=[],
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
