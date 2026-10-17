#!/src/anatomy/lung.py
# -*- coding: utf-8 -*-

"""从一个整肺分割 Mask 估计肺部全局解剖坐标系。"""

from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from src.models import (
    AnatomicalBounds,
    AnatomicalFrame,
    AnatomicalGeometryResult,
    AxisBounds,
    Point3D,
)
from .symmetry import build_symmetry_frame_axes, estimate_reflection_symmetry_plane
from .orientation import patient_reference_axes


_MIN_TARGET_POINTS = 32
_MAX_DOMINANT_COMPONENTS = 2
_MIN_COMPONENT_SIZE_RATIO = 0.05
_BILATERAL_COMPONENT_SAMPLE_LIMIT = 3000
_MIN_CROSS_COMPONENT_MATCH_FRACTION = 0.80


def _point(vector: np.ndarray) -> Point3D:
    """把 NumPy 三维向量转换为阶段实体统一使用的 Point3D。"""

    return Point3D(float(vector[0]), float(vector[1]), float(vector[2]))


def _dicom_axes(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从方向矩阵读取 DICOM 左右、前后、头脚三个物理单位轴。"""

    axes = patient_reference_axes(np.asarray(image.GetDirection()))
    return tuple(axes[:, axis] for axis in range(3))


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


def _retain_lung_components(
    target_mask: np.ndarray,
    *,
    collect_regions: bool = False,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """保留一个整肺标签中的一至两个主体，排除远端小块误分割。

    整肺模型可能把左右肺输出为两个互不连通的区域，也可能经气道连接成一个
    区域。因此最多保留两个主要连通域；第二个区域只有达到最大区域体积的 5%
    才会保留，避免远离胸部的小误分割拉大头脚范围并带偏中线。
    """

    if target_mask.ndim != 3:
        raise ValueError("肺部目标 Mask 必须是三维数组")
    binary = sitk.GetImageFromArray(target_mask.astype(np.uint8, copy=False))
    component_image = sitk.ConnectedComponent(binary)
    shape_statistics = None
    if collect_regions:
        shape_statistics = sitk.LabelShapeStatisticsImageFilter()
        shape_statistics.Execute(component_image)
    components = sitk.GetArrayFromImage(component_image)
    del component_image
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
    regions = []
    if collect_regions:
        interior = np.zeros_like(retained_mask, dtype=bool)
        interior[1:-1, 1:-1, 1:-1] = (
            retained_mask[1:-1, 1:-1, 1:-1]
            & retained_mask[:-2, 1:-1, 1:-1]
            & retained_mask[2:, 1:-1, 1:-1]
            & retained_mask[1:-1, :-2, 1:-1]
            & retained_mask[1:-1, 2:, 1:-1]
            & retained_mask[1:-1, 1:-1, :-2]
            & retained_mask[1:-1, 1:-1, 2:]
        )
        surface_indices = np.argwhere(retained_mask & ~interior)
        surface_labels = components[tuple(surface_indices.T)]
        for component_id in retained_ids:
            surface_order = np.flatnonzero(surface_labels == component_id)
            regions.append({
                # 连通域在单位索引网格上计算；实际毫米几何在调用方恢复。
                "centroid_index_xyz": shape_statistics.GetCentroid(component_id),
                "surface_indices_zyx": surface_indices[surface_order],
                "surface_order": surface_order,
            })
    return retained_mask, {
        "component_count": int(len(component_ids)),
        "retained_component_count": len(retained_ids),
        "discarded_component_count": int(len(component_ids) - len(retained_ids)),
        "retained_point_count": retained_count,
        "discarded_point_count": original_count - retained_count,
        "retained_component_sizes": [int(sizes[value]) for value in retained_ids],
    }, regions


def retain_dominant_lung_components(
    target_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """保留既有双返回值接口；姿态调用另保留稀疏边界的连通域身份。"""

    retained_mask, diagnostics, _ = _retain_lung_components(target_mask)
    return retained_mask, diagnostics


def _bilateral_component_evidence(
    surfaces: list[np.ndarray],
    centroids: np.ndarray,
    normal: np.ndarray,
    plane_center: np.ndarray,
    minimum_spacing: float,
) -> dict[str, Any]:
    """验证候选面确实交换两个左右主体，不能让同侧碎块各自反射自身。"""

    signed_distances = (centroids - plane_center) @ normal
    separation = abs(float(signed_distances[0] - signed_distances[1]))
    widths = [
        float(np.diff(np.percentile(points @ normal, [2.0, 98.0]))[0])
        for points in surfaces
    ]
    required_separation = max(5.0, 4.0 * minimum_spacing, 0.25 * sum(widths))
    diagnostics: dict[str, Any] = {
        "reliable": False,
        "centroid_signed_distances_mm": signed_distances.tolist(),
        "centroid_lr_separation_mm": separation,
        "required_lr_separation_mm": required_separation,
        "min_cross_component_match_fraction": _MIN_CROSS_COMPONENT_MATCH_FRACTION,
        "reason": "",
    }
    if separation < required_separation:
        diagnostics["reason"] = "两个肺主体沿候选左右轴分离不足，不能证明位于双侧"
        return diagnostics
    side_margin = max(minimum_spacing, 0.10 * separation)
    if (
        float(signed_distances[0] * signed_distances[1]) >= 0.0
        or float(np.min(np.abs(signed_distances))) < side_margin
    ):
        diagnostics["reason"] = "两个肺主体质心未可靠分居候选对称面两侧"
        return diagnostics

    samples = []
    for points in surfaces:
        if len(points) > _BILATERAL_COMPONENT_SAMPLE_LIMIT:
            selection = np.sort(np.random.default_rng(0).choice(
                len(points), _BILATERAL_COMPONENT_SAMPLE_LIMIT, replace=False,
            ))
            points = points[selection]
        samples.append(points)
    joint = np.concatenate(samples)
    component_ids = np.concatenate([
        np.full(len(points), index, dtype=np.uint8)
        for index, points in enumerate(samples)
    ])
    tree = cKDTree(joint)
    cross_fractions = []
    for index, points in enumerate(samples):
        reflected = points - 2.0 * ((points - plane_center) @ normal)[:, None] * normal
        closest = tree.query(reflected, k=1, workers=1)[1]
        cross_fractions.append(float(np.mean(component_ids[closest] != index)))
    diagnostics["cross_component_match_fractions"] = cross_fractions
    diagnostics["component_sample_counts"] = [int(len(points)) for points in samples]
    if min(cross_fractions) < _MIN_CROSS_COMPONENT_MATCH_FRACTION:
        diagnostics["reason"] = "肺主体镜像主要匹配自身，缺少跨左右主体的对应证据"
        return diagnostics
    diagnostics["reliable"] = True
    diagnostics["reason"] = "两个独立主体分居对称面两侧且镜像跨主体对应"
    return diagnostics


def _bounds(points: np.ndarray, axis: np.ndarray) -> AxisBounds:
    """计算全部目标物理点在指定解剖单位轴上的最小、最大投影。"""

    projections = points @ axis
    return AxisBounds(float(np.min(projections)), float(np.max(projections)))


def _fit_bilateral_pose(
    regions: list[dict[str, Any]],
    image: sitk.Image,
    axes: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None]:
    """对两个主体执行相同的严格镜像拟合和跨侧身份验证。"""

    spacing = np.asarray(image.GetSpacing(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    surfaces = [
        np.asarray(image.GetOrigin())
        + (region["surface_indices_zyx"][:, ::-1] * spacing) @ direction.T
        for region in regions
    ]
    centroids = np.asarray([
        image.TransformContinuousIndexToPhysicalPoint(region["centroid_index_xyz"])
        for region in regions
    ])
    # 保留原来的全局 z/y/x 顺序，使既有固定种子候选不受辅助证据重构影响。
    surface_order = np.argsort(np.concatenate([
        region["surface_order"] for region in regions
    ]))
    normal, plane_center, symmetry = estimate_reflection_symmetry_plane(
        np.concatenate(surfaces)[surface_order], axes,
    )
    bilateral = None
    if symmetry["status"] in {"accepted", "unchanged"}:
        bilateral = _bilateral_component_evidence(
            surfaces, centroids, normal, plane_center, float(np.min(spacing)),
        )
        if not bilateral["reliable"]:
            reason = bilateral["reason"] + "，保留输入方向"
            symmetry = {
                **symmetry,
                "accepted": False,
                "source_status": symmetry["status"],
                "status": "rejected",
                "reason": reason,
                "warnings": [reason],
                "confidence": 0.0,
            }
            normal = axes[:, 0]
    return normal, symmetry, bilateral


def _try_pose_support(
    pose_support_mask: np.ndarray,
    dominant_lung_mask: np.ndarray,
    image: sitk.Image,
    axes: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """胸膜腔支持只供姿态使用，不改变肺主体、中心或覆盖范围。"""

    support = np.asarray(pose_support_mask, dtype=bool)
    if support.shape != dominant_lung_mask.shape:
        return None, {"status": "rejected", "reason": "肺姿态支持 Mask 尺寸不匹配"}
    if not np.all(support[dominant_lung_mask]):
        return None, {"status": "rejected", "reason": "肺姿态支持 Mask 未包含全部肺主体"}
    retained, components, regions = _retain_lung_components(support, collect_regions=True)
    diagnostic: dict[str, Any] = {"status": "rejected", **components}
    if components["retained_component_count"] != 2:
        diagnostic["reason"] = "胸膜腔支持缺少两个独立主体"
        return None, diagnostic
    if not np.all(retained[dominant_lung_mask]):
        diagnostic["reason"] = "胸膜腔支持主体过滤后丢失肺主体"
        return None, diagnostic
    normal, symmetry, bilateral = _fit_bilateral_pose(regions, image, axes)
    diagnostic.update({"status": symmetry["status"], "reason": symmetry["reason"],
                       "symmetry": symmetry})
    if bilateral is not None:
        diagnostic["bilateral_evidence"] = bilateral
    if symmetry["status"] not in {"accepted", "unchanged"}:
        return None, diagnostic
    return normal, diagnostic


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
    *,
    pose_support_mask: np.ndarray | None = None,
    pose_support_source: tuple[str, ...] | list[str] = (),
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """以双肺主体的可靠镜像面对齐左右方向，保留未被确定的俯仰基准。

    肺自身存在左右大小和形态差异。原肺拟合仅因整体镜像质量不足而回退时，
    可尝试上游按语义提供的肺加胸膜积液支持 Mask；仍使用相同质量和跨侧
    门槛。支持不改变切割主体、中心或范围，不能用心包/未知标签补全形态。
    两种证据均不足时保留输入方向，不把肺形态高低差直接解释为侧倾。
    """

    dominant_mask, component_diagnostics, regions = _retain_lung_components(
        target_mask, collect_regions=True,
    )
    points = target_mask_physical_points(dominant_mask, image)
    dicom_lr, dicom_ap, dicom_si = _dicom_axes(image)
    diagnostics: dict[str, Any] = {
        "point_count": int(len(points)),
        "angle_source": "bilateral_reflection_symmetry",
        "correction_degrees": 0.0,
        **component_diagnostics,
    }
    if pose_support_mask is not None:
        diagnostics["pose_support"] = {
            "status": "not_needed",
            "source_labels": list(pose_support_source),
        }
    axes = np.column_stack((dicom_lr, dicom_ap, dicom_si))
    applied = False
    if len(points) < _MIN_TARGET_POINTS:
        diagnostics["fallback_reason"] = "肺部目标体素太少"
        method = "dicom_axes_fallback"
        confidence = 0.0
        warnings = ["肺部目标体素太少，仅保留目标中心和DICOM标准方向"]
    elif component_diagnostics["retained_component_count"] < 2:
        method = "lung_dicom_axes"
        confidence = 0.0
        warnings = ["缺少两个独立肺主体，左右姿态证据不足，保留输入方向"]
        diagnostics["fallback_reason"] = warnings[0]
    else:
        normal, symmetry, bilateral = _fit_bilateral_pose(regions, image, axes)
        if bilateral is not None:
            diagnostics["bilateral_evidence"] = bilateral
        # 只补救形态质量不足，不绕过不收敛、边界、方向歧义或跨侧证据拒绝。
        if (
            pose_support_mask is not None
            and symmetry["status"] == "rejected"
            and symmetry.get("rejection_code") == "reflection_quality"
        ):
            diagnostics["lung_symmetry"] = symmetry
            support_normal, support = _try_pose_support(
                pose_support_mask, dominant_mask, image, axes,
            )
            diagnostics["pose_support"] = {
                **support,
                "source_labels": list(pose_support_source),
                "fallback_source": "lung_reflection_quality",
            }
            if support_normal is not None:
                normal = support_normal
                symmetry = support["symmetry"]
                diagnostics["bilateral_evidence"] = support["bilateral_evidence"]
                diagnostics["angle_source"] = "pleural_envelope_reflection_symmetry"
        diagnostics["symmetry"] = symmetry
        axes = build_symmetry_frame_axes(normal, axes)
        applied = bool(symmetry["accepted"])
        method = "lung_reflection_symmetry" if applied else "lung_dicom_axes"
        if applied and diagnostics["angle_source"] == "pleural_envelope_reflection_symmetry":
            method = "lung_pleural_envelope_symmetry"
        confidence = float(symmetry["confidence"])
        warnings = list(symmetry["warnings"])
        diagnostics["correction_degrees"] = float(symmetry["angle_degrees"]) if applied else 0.0
        if symmetry["status"] == "rejected":
            diagnostics["fallback_reason"] = symmetry["reason"]
    frame = _build_frame(
        points,
        axes[:, 0],
        axes[:, 1],
        axes[:, 2],
        method=method,
        confidence=confidence,
        applied=applied,
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
