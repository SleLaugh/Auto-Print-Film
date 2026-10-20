#!/src/anatomy/spine.py
# -*- coding: utf-8 -*-

"""由可靠椎体左右对称面构造保留输入前后倾角的整体三维坐标系。"""

import math
from typing import Any

import numpy as np
import SimpleITK as sitk

from src.anatomy.orientation import patient_reference_axes
from src.anatomy.symmetry import build_symmetry_frame_axes

from src.models import (
    AnatomicalBounds,
    AnatomicalFrame,
    AnatomicalGeometryResult,
    AxisBounds,
    Point3D,
)


_MIN_ENDPOINT_DISTANCE_MM = 1.0


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
    """用可靠左右对称方向回正，并用全部有效椎体确定中心和范围。

    左右法向采用可靠逐椎体估计的中位方向；将输入 AP 投影到该法向的
    正交平面，保留原始前后倾角。首尾质心仅用于范围有效性和诊断，不能
    将其连线强制摆竖直；证据不足时保持患者参考三轴并明确警告。
    提供 ``label_indices_zyx`` 时以其为唯一标签来源，
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
        raise ValueError(f"开始椎体 {start_label} 缺失，无法计算脊柱范围")
    if end_label not in physical_points:
        raise ValueError(f"结束椎体 {end_label} 缺失，无法计算脊柱范围")

    start_center = np.mean(physical_points[start_label], axis=0)
    end_center = np.mean(physical_points[end_label], axis=0)
    raw_axis_vector = end_center - start_center
    endpoint_distance = float(np.linalg.norm(raw_axis_vector))
    if not math.isfinite(endpoint_distance) or endpoint_distance < _MIN_ENDPOINT_DISTANCE_MM:
        raise ValueError("首尾椎体质心距离过小，无法计算脊柱范围")
    raw_spine_axis = raw_axis_vector / endpoint_distance

    dicom_lr, dicom_ap, dicom_si = _dicom_axes(image)
    reference_axes = np.column_stack((dicom_lr, dicom_ap, dicom_si))

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
    # 与髋、肺使用同一保留俯仰构造；颅脑在该基础上另行估计形态俯仰。
    # 脊柱不能为了把首尾连线立直而额外改变俯仰，
    # 也不能把可靠左右法向再次投影到首尾连线的法面上。
    axes = build_symmetry_frame_axes(reference_lr, reference_axes)
    left_right_axis, anterior_posterior_axis, spine_axis = axes.T
    correction_degrees = math.degrees(math.atan2(
        float(np.linalg.norm(np.cross(left_right_axis, dicom_lr))),
        float(np.dot(left_right_axis, dicom_lr)),
    ))

    used_labels = [label for label in labels if label in physical_points]
    all_points = np.concatenate([physical_points[label] for label in used_labels])
    warnings = []
    if missing_labels:
        warnings.append("部分中间椎体缺失: " + ", ".join(missing_labels))
    if not reliable_lr_axes:
        warnings.append("椎体左右对称证据不足，整体姿态保留患者参考方向和原始前后倾角")
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
        method="spine_reflection_symmetry" if reliable_lr_axes else "spine_dicom_axes",
        confidence=(1.0 if not missing_labels else 0.8) if reliable_lr_axes else 0.0,
        applied=bool(reliable_lr_axes) and correction_degrees > 1e-6,
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
        "endpoint_axis_role": "diagnostic_only",
        "pitch_policy": "project_input_ap_onto_symmetry_plane",
        "correction_degrees": correction_degrees,
        "used_labels": used_labels,
        "missing_labels": missing_labels,
        "left_right_source": "vertebral_reflection_symmetry" if reliable_lr_axes else "dicom_reference",
        "reliable_symmetry_label_count": len(reliable_lr_axes),
    }
    return AnatomicalGeometryResult(global_frame=frame, local_frames={}), diagnostics
