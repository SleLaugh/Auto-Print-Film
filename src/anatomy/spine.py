#!/src/anatomy/spine.py
# -*- coding: utf-8 -*-

"""根据首尾椎体质心构造颈椎、胸椎共用的三维直线坐标系。"""

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


_MIN_ENDPOINT_DISTANCE_MM = 1.0
_MIN_AXIS_ALIGNMENT = 1e-6


def _normalize(vector: np.ndarray, error_message: str) -> np.ndarray:
    """返回三维单位向量，长度无效时抛出带业务语义的错误。"""

    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(error_message)
    return vector / length


def _point(vector: np.ndarray) -> Point3D:
    """把 NumPy 三维向量转换成阶段实体使用的物理点。"""

    return Point3D(float(vector[0]), float(vector[1]), float(vector[2]))


def _mask_physical_points(mask: np.ndarray, image: sitk.Image) -> np.ndarray:
    """将 z/y/x 排列的单个椎体 Mask 批量转换为患者物理坐标。"""

    indices_zyx = np.argwhere(mask)
    if not len(indices_zyx):
        return np.empty((0, 3), dtype=float)
    indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return origin + (indices_xyz * spacing) @ direction.T


def _axis_bounds(points: np.ndarray, axis: np.ndarray) -> AxisBounds:
    """计算全部目标点在一个解剖单位轴上的毫米投影范围。"""

    projections = points @ axis
    return AxisBounds(float(np.min(projections)), float(np.max(projections)))


def _dicom_axes(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从输入图像方向矩阵读取患者左右、前后和头脚参考方向。"""

    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return tuple(
        _normalize(direction[:, index], "输入图像方向矩阵包含零长度轴")
        for index in range(3)
    )


def estimate_spine_frame_with_diagnostics(
    label_masks: dict[str, np.ndarray],
    start_label: str,
    end_label: str,
    image: sitk.Image,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """用首尾椎体质心定方向，并用全部有效椎体确定中心和范围。

    首尾质心只决定一条无向三维直线；最终头脚轴的符号与输入 DICOM 头侧
    方向保持一致。中间椎体允许缺失，但会进入警告和诊断信息。
    """

    if not start_label or not end_label:
        raise ValueError("脊柱开始、结束椎体名称不能为空")
    physical_points: dict[str, np.ndarray] = {}
    missing_labels: list[str] = []
    expected_shape = tuple(reversed(image.GetSize()))
    for label, mask in label_masks.items():
        if mask.ndim != 3 or mask.shape != expected_shape:
            raise ValueError(f"椎体 {label} Mask 尺寸与输入图像不一致")
        points = _mask_physical_points(mask, image)
        if len(points):
            physical_points[label] = points
        else:
            missing_labels.append(label)

    if start_label not in physical_points:
        raise ValueError(f"开始椎体 {start_label} 缺失，无法计算脊柱方向")
    if end_label not in physical_points:
        raise ValueError(f"结束椎体 {end_label} 缺失，无法计算脊柱方向")

    start_center = np.mean(physical_points[start_label], axis=0)
    end_center = np.mean(physical_points[end_label], axis=0)
    raw_axis_vector = end_center - start_center
    endpoint_distance = float(np.linalg.norm(raw_axis_vector))
    if not math.isfinite(endpoint_distance) or endpoint_distance < _MIN_ENDPOINT_DISTANCE_MM:
        raise ValueError("首尾椎体质心距离过小，无法计算脊柱方向")
    raw_spine_axis = raw_axis_vector / endpoint_distance

    dicom_lr, dicom_ap, dicom_si = _dicom_axes(image)
    spine_axis = raw_spine_axis.copy()
    if float(np.dot(spine_axis, dicom_si)) < 0.0:
        spine_axis = -spine_axis

    # 去掉 DICOM 左右轴沿脊柱方向的分量，保证左右轴与斜向脊柱轴正交。
    left_right_axis = _normalize(
        dicom_lr - float(np.dot(dicom_lr, spine_axis)) * spine_axis,
        "DICOM左右轴与脊柱轴近似平行，无法构造解剖坐标",
    )
    if float(np.dot(left_right_axis, dicom_lr)) < 0.0:
        left_right_axis = -left_right_axis
    anterior_posterior_axis = _normalize(
        np.cross(spine_axis, left_right_axis),
        "无法由脊柱轴和左右轴构造前后方向",
    )
    if float(np.dot(anterior_posterior_axis, dicom_ap)) <= _MIN_AXIS_ALIGNMENT:
        raise ValueError("脊柱解剖坐标与DICOM患者方向不一致")

    used_labels = [label for label in label_masks if label in physical_points]
    all_points = np.concatenate([physical_points[label] for label in used_labels])
    warnings = []
    if missing_labels:
        warnings.append("部分中间椎体缺失: " + ", ".join(missing_labels))
    frame = AnatomicalFrame(
        origin=_point(np.mean(all_points, axis=0)),
        left_right_axis=_point(left_right_axis),
        anterior_posterior_axis=_point(anterior_posterior_axis),
        superior_inferior_axis=_point(spine_axis),
        bounds=AnatomicalBounds(
            left_right=_axis_bounds(all_points, left_right_axis),
            anterior_posterior=_axis_bounds(all_points, anterior_posterior_axis),
            superior_inferior=_axis_bounds(all_points, spine_axis),
        ),
        method="spine_endpoint_centroids",
        confidence=1.0 if not missing_labels else 0.8,
        applied=True,
        warnings=warnings,
    )
    diagnostics: dict[str, Any] = {
        "start_label": start_label,
        "end_label": end_label,
        "start_center": [float(value) for value in start_center],
        "end_center": [float(value) for value in end_center],
        "endpoint_distance_mm": endpoint_distance,
        "raw_spine_axis": [float(value) for value in raw_spine_axis],
        "spine_axis": [float(value) for value in spine_axis],
        "used_labels": used_labels,
        "missing_labels": missing_labels,
    }
    return AnatomicalGeometryResult(global_frame=frame, local_frames={}), diagnostics
