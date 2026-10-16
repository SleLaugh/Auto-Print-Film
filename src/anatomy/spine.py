#!/src/anatomy/spine.py
# -*- coding: utf-8 -*-

"""根据脊柱首尾质心与可靠椎体左右对称面构造整体三维坐标系。"""

import math
from typing import Any

import numpy as np
import SimpleITK as sitk

from src.anatomy.orientation import patient_reference_axes

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

    return _indices_physical_points(np.argwhere(mask), image)


def _indices_physical_points(indices_zyx: np.ndarray, image: sitk.Image) -> np.ndarray:
    """将已有 N×3 稀疏体素坐标转换为毫米坐标，不构造整卷 Mask。"""

    indices = np.asarray(indices_zyx)
    if indices.ndim != 2 or indices.shape[1:] != (3,):
        raise ValueError("椎体坐标必须为 N×3 的 z/y/x 索引")
    try:
        finite = bool(np.all(np.isfinite(indices)))
    except TypeError as error:
        raise ValueError("椎体坐标必须为有限数字") from error
    if not finite:
        raise ValueError("椎体坐标必须为有限数字")
    if not len(indices):
        return np.empty((0, 3), dtype=float)
    size_zyx = np.asarray(image.GetSize()[::-1])
    if np.any(indices < 0) or np.any(indices > size_zyx - 1):
        raise ValueError("椎体坐标超出输入图像范围")
    indices_xyz = indices[:, [2, 1, 0]].astype(float)
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

    axes = patient_reference_axes(np.asarray(image.GetDirection()))
    return tuple(axes[:, index] for index in range(3))


def estimate_spine_frame_with_diagnostics(
    label_masks: dict[str, np.ndarray] | None,
    start_label: str,
    end_label: str,
    image: sitk.Image,
    label_pose_annotations: dict[str, Any] | None = None,
    *,
    label_indices_zyx: dict[str, np.ndarray] | None = None,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """用首尾椎体质心定方向，并用全部有效椎体确定中心和范围。

    首尾质心确定头脚轴，逐椎体可靠对称面确定左右方向。方向符号按患者
    LPS 头侧及左右/前后参考选择；缺少可靠左右证据时保留参考左右轴
    投影并明确诊断。提供 ``label_indices_zyx`` 时以其为唯一标签来源，
    直接复用已有稀疏坐标；``label_masks`` 可传 None，不读取或重建 Mask。
    中心、首尾质心及范围始终使用全部体素，不以姿态拟合的抽样代替。
    """

    if not start_label or not end_label:
        raise ValueError("脊柱开始、结束椎体名称不能为空")
    if image.GetDimension() != 3:
        raise ValueError("脊柱坐标系只支持三维图像")
    physical_points: dict[str, np.ndarray] = {}
    missing_labels: list[str] = []
    expected_shape = tuple(reversed(image.GetSize()))
    labels = label_indices_zyx if label_indices_zyx is not None else (label_masks or {})
    for label, data in labels.items():
        if label_indices_zyx is not None:
            try:
                points = _indices_physical_points(data, image)
            except ValueError as error:
                raise ValueError(f"椎体 {label}: {error}") from error
        else:
            if data.ndim != 3 or data.shape != expected_shape:
                raise ValueError(f"椎体 {label} Mask 尺寸与输入图像不一致")
            points = _mask_physical_points(data, image)
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
    # 起止点用实际患者毫米坐标判定头脚，不能随存储单轴反向倒置成片。
    if spine_axis[2] < 0.0 or (
        abs(float(spine_axis[2])) < 1e-8 and float(np.dot(spine_axis, dicom_si)) < 0.0
    ):
        spine_axis = -spine_axis

    reliable_lr_axes = []
    for label in physical_points:
        annotation = (label_pose_annotations or {}).get(label, {})
        diagnostic = annotation.get("symmetry_diagnostics") or {}
        raw_lr = annotation.get("symmetry_left_right_axis")
        if diagnostic.get("status") not in {"accepted", "unchanged"} or not raw_lr:
            continue
        candidate = np.array([raw_lr[axis] for axis in ("x", "y", "z")], dtype=float)
        if not np.all(np.isfinite(candidate)) or np.linalg.norm(candidate) < 1e-6:
            continue
        candidate /= np.linalg.norm(candidate)
        if float(np.dot(candidate, dicom_lr)) < 0.0:
            candidate = -candidate
        reliable_lr_axes.append(candidate)
    reference_lr = (
        _normalize(np.median(reliable_lr_axes, axis=0), "脊柱左右对称方向无效")
        if reliable_lr_axes else dicom_lr
    )
    # 与首尾轴正交化，确保实际重采样使用同一个右手单位正交坐标。
    left_right_axis = _normalize(
        reference_lr - float(np.dot(reference_lr, spine_axis)) * spine_axis,
        "左右对称方向与脊柱轴近似平行，无法构造解剖坐标",
    )
    if float(np.dot(left_right_axis, dicom_lr)) < 0.0:
        left_right_axis = -left_right_axis
    anterior_posterior_axis = _normalize(
        np.cross(spine_axis, left_right_axis),
        "无法由脊柱轴和左右轴构造前后方向",
    )
    if float(np.dot(anterior_posterior_axis, dicom_ap)) <= _MIN_AXIS_ALIGNMENT:
        raise ValueError("脊柱解剖坐标与DICOM患者方向不一致")

    used_labels = [label for label in labels if label in physical_points]
    all_points = np.concatenate([physical_points[label] for label in used_labels])
    warnings = []
    if missing_labels:
        warnings.append("部分中间椎体缺失: " + ", ".join(missing_labels))
    if label_pose_annotations is not None and not reliable_lr_axes:
        warnings.append("椎体左右对称证据不足，整体左右轴回退到输入方向投影")
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
        "left_right_source": "vertebral_reflection_symmetry" if reliable_lr_axes else "dicom_projection",
        "reliable_symmetry_label_count": len(reliable_lr_axes),
    }
    return AnatomicalGeometryResult(global_frame=frame, local_frames={}), diagnostics
