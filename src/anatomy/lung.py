#!/src/anatomy/lung.py
# -*- coding: utf-8 -*-

"""从一个整肺分割 Mask 估计肺部全局解剖坐标系。"""

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


_MIN_TARGET_POINTS = 32
_MAX_DOMINANT_COMPONENTS = 2
_MIN_COMPONENT_SIZE_RATIO = 0.05


def _normalize(vector: np.ndarray) -> np.ndarray:
    """把物理方向归一化为单位向量，并拒绝零长度或非法向量。"""

    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("解剖方向向量长度无效")
    return vector / length


def _point(vector: np.ndarray) -> Point3D:
    """把 NumPy 三维向量转换为阶段实体统一使用的 Point3D。"""

    return Point3D(float(vector[0]), float(vector[1]), float(vector[2]))


def _dicom_axes(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从方向矩阵读取 DICOM 左右、前后、头脚三个物理单位轴。"""

    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return (
        _normalize(direction[:, 0]),
        _normalize(direction[:, 1]),
        _normalize(direction[:, 2]),
    )


def target_mask_physical_points(mask: np.ndarray, image: sitk.Image) -> np.ndarray:
    """把 z/y/x Mask 中的目标体素中心批量转换成患者物理坐标。"""

    if mask.ndim != 3:
        raise ValueError("肺部目标 Mask 必须是三维数组")
    indices_zyx = np.argwhere(mask)
    if not len(indices_zyx):
        raise ValueError("肺部目标 Mask 为空")
    indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return origin + (indices_xyz * spacing) @ direction.T


def retain_dominant_lung_components(
    target_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """保留一个整肺标签中的一至两个主体，排除远端小块误分割。

    整肺模型可能把左右肺输出为两个互不连通的区域，也可能经气道连接成一个
    区域。因此最多保留两个主要连通域；第二个区域只有达到最大区域体积的 5%
    才会保留，避免远离胸部的小误分割拉大头脚范围并带偏中线。
    """

    if target_mask.ndim != 3:
        raise ValueError("肺部目标 Mask 必须是三维数组")
    binary = sitk.GetImageFromArray(target_mask.astype(np.uint8, copy=False))
    components = sitk.GetArrayFromImage(sitk.ConnectedComponent(binary))
    sizes = np.bincount(components.ravel())
    if len(sizes) <= 1:
        raise ValueError("肺部目标 Mask 为空")

    component_ids = np.argsort(sizes[1:])[::-1] + 1
    largest_size = int(sizes[component_ids[0]])
    retained_ids = [
        int(component_id)
        for component_id in component_ids[:_MAX_DOMINANT_COMPONENTS]
        if int(sizes[component_id]) >= largest_size * _MIN_COMPONENT_SIZE_RATIO
    ]
    retained_mask = np.isin(components, retained_ids)
    retained_count = int(np.count_nonzero(retained_mask))
    original_count = int(np.count_nonzero(target_mask))
    return retained_mask, {
        "component_count": int(len(component_ids)),
        "retained_component_count": len(retained_ids),
        "discarded_component_count": int(len(component_ids) - len(retained_ids)),
        "retained_point_count": retained_count,
        "discarded_point_count": original_count - retained_count,
        "retained_component_sizes": [int(sizes[value]) for value in retained_ids],
    }


def _bounds(points: np.ndarray, axis: np.ndarray) -> AxisBounds:
    """计算全部目标物理点在指定解剖单位轴上的最小、最大投影。"""

    projections = points @ axis
    return AxisBounds(float(np.min(projections)), float(np.max(projections)))


def _build_frame(
    points: np.ndarray,
    lr_axis: np.ndarray,
    ap_axis: np.ndarray,
    si_axis: np.ndarray,
    *,
    method: str,
    confidence: float,
    applied: bool,
    warnings: list[str],
) -> AnatomicalFrame:
    """使用同一批目标点组装坐标原点、三轴边界和可信度信息。"""

    return AnatomicalFrame(
        origin=_point(np.mean(points, axis=0)),
        left_right_axis=_point(lr_axis),
        anterior_posterior_axis=_point(ap_axis),
        superior_inferior_axis=_point(si_axis),
        bounds=AnatomicalBounds(
            left_right=_bounds(points, lr_axis),
            anterior_posterior=_bounds(points, ap_axis),
            superior_inferior=_bounds(points, si_axis),
        ),
        method=method,
        confidence=confidence,
        applied=applied,
        warnings=warnings,
    )


def estimate_lung_frame_with_diagnostics(
    target_mask: np.ndarray,
    image: sitk.Image,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """用肺主体确定中心和范围，方向始终保持输入图像的 DICOM 三轴。"""

    dominant_mask, component_diagnostics = retain_dominant_lung_components(target_mask)
    points = target_mask_physical_points(dominant_mask, image)
    dicom_lr, dicom_ap, dicom_si = _dicom_axes(image)
    diagnostics: dict[str, Any] = {
        "point_count": int(len(points)),
        "angle_source": "disabled",
        "correction_degrees": 0.0,
        **component_diagnostics,
    }
    if len(points) < _MIN_TARGET_POINTS:
        diagnostics["fallback_reason"] = "肺部目标体素太少"
        method = "dicom_axes_fallback"
        confidence = 0.0
        warnings = ["肺部目标体素太少，仅保留目标中心和DICOM标准方向"]
    else:
        method = "lung_dicom_axes"
        confidence = 1.0
        warnings = []
    frame = _build_frame(
        points,
        dicom_lr,
        dicom_ap,
        dicom_si,
        method=method,
        confidence=confidence,
        applied=False,
        warnings=warnings,
    )
    return AnatomicalGeometryResult(global_frame=frame, local_frames={}), diagnostics


def estimate_lung_anatomical_geometry(
    target_mask: np.ndarray,
    image: sitk.Image,
) -> AnatomicalGeometryResult:
    """返回不含部位专属诊断的正式肺部解剖几何结果。"""

    result, _ = estimate_lung_frame_with_diagnostics(target_mask, image)
    return result
