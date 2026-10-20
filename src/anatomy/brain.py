"""以脑区左右对称面和形态主轴建立三维重建坐标及脑区切片范围。"""

from typing import Any

import numpy as np
import SimpleITK as sitk

from src.anatomy.orientation import patient_reference_axes
from src.anatomy.symmetry import (
    build_symmetry_frame_axes,
    estimate_reflection_symmetry_plane,
)
from src.models import (
    AnatomicalBounds,
    AnatomicalFrame,
    AnatomicalGeometryResult,
    AxisBounds,
    Point3D,
)


_MIN_PITCH_VOXELS = 64
_MIN_PITCH_RATIO = 1.15
_MAX_PITCH_DEGREES = 35.0
_MIN_PITCH_DEGREES = 0.75
_MIDLINE_HALF_WIDTH_MM = 10.0
_MAX_MIDLINE_DISAGREEMENT_DEGREES = 5.0
_MAX_SYMMETRY_POINTS = 6000


def _point(vector: np.ndarray) -> Point3D:
    """将物理位置或单位方向转换成阶段契约实体。"""

    return Point3D(*(float(value) for value in vector))


def _boundary_contacts(mask: np.ndarray) -> dict[str, bool]:
    """记录输入体素范围的六个面是否接触目标，不能据此补造扫描外结构。"""

    return {
        "x_min": bool(np.any(mask[:, :, 0])),
        "x_max": bool(np.any(mask[:, :, -1])),
        "y_min": bool(np.any(mask[:, 0, :])),
        "y_max": bool(np.any(mask[:, -1, :])),
        "z_min": bool(np.any(mask[0])),
        "z_max": bool(np.any(mask[-1])),
    }


def _projected_layers(mask: np.ndarray, transform: np.ndarray):
    """逐层产生物理投影，临时点数组大小仅取决于单层前景数量。"""

    for z_index, section in enumerate(mask):
        yx = np.argwhere(section)
        if not len(yx):
            continue
        xyz = np.column_stack((yx[:, 1], yx[:, 0], np.full(len(yx), z_index)))
        yield xyz @ transform


def _moments(
    mask: np.ndarray,
    transform: np.ndarray,
    *,
    midline_center: float | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """合并逐层中心矩，避免常驻全卷物理点或大坐标相减造成数值损失。"""

    count = 0
    mean = np.zeros(3, dtype=float)
    scatter = np.zeros((3, 3), dtype=float)
    for points in _projected_layers(mask, transform):
        if midline_center is not None:
            points = points[np.abs(points[:, 0] - midline_center) <= _MIDLINE_HALF_WIDTH_MM]
        size = len(points)
        if not size:
            continue
        layer_mean = np.mean(points, axis=0)
        centered = points - layer_mean
        delta = layer_mean - mean
        total = count + size
        scatter += centered.T @ centered + np.outer(delta, delta) * (count * size / total)
        mean += delta * (size / total)
        count = total
    return count, mean, scatter


def _principal_angle(scatter: np.ndarray) -> tuple[float | None, float | None]:
    """返回 AP/SI 最大特征向量朝向 AP 的角度与特征值比。"""

    eigenvalues, eigenvectors = np.linalg.eigh(scatter[1:, 1:])
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 1e-8:
        return None, None
    principal = eigenvectors[:, -1]
    if principal[0] < 0.0:
        principal = -principal
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return angle, float(eigenvalues[-1] / eigenvalues[0])


def _estimate_pitch(
    mask: np.ndarray,
    transform: np.ndarray,
    moments: tuple[int, np.ndarray, np.ndarray],
    contacts: dict[str, bool],
    enabled: bool,
) -> dict[str, Any]:
    """只估计脑形态俯仰；主轴不是听眦线，也不是 AC-PC 解剖标志线。"""

    count, mean, scatter = moments
    diagnostics: dict[str, Any] = {
        "method": "brain_shape_principal_axis",
        "status": "rejected",
        "accepted": False,
        "angle_degrees": 0.0,
        "candidate_angle_degrees": None,
        "ratio": None,
        "midline_angle_degrees": None,
        "midline_ratio": None,
        "midline_point_count": 0,
        "midline_disagreement_degrees": None,
        "point_count": count,
        "min_point_count": _MIN_PITCH_VOXELS,
        "min_ratio": _MIN_PITCH_RATIO,
        "max_angle_degrees": _MAX_PITCH_DEGREES,
        "min_correction_degrees": _MIN_PITCH_DEGREES,
        "midline_half_width_mm": _MIDLINE_HALF_WIDTH_MM,
        "max_midline_disagreement_degrees": _MAX_MIDLINE_DISAGREEMENT_DEGREES,
        "confidence": 0.0,
        "reason": "",
        "rejection_code": "",
    }

    def reject(code: str, reason: str, *, status: str = "rejected"):
        diagnostics.update(status=status, rejection_code=code, reason=reason)
        return diagnostics

    if not enabled:
        return reject("disabled", "已关闭脑形态俯仰，保留对称面内的输入方向", status="disabled")
    if count < _MIN_PITCH_VOXELS:
        return reject("insufficient_voxels", "脑区体素不足，无法估计形态俯仰")
    angle, ratio = _principal_angle(scatter)
    diagnostics.update(candidate_angle_degrees=angle, ratio=ratio)
    if any(contacts.values()):
        return reject("brain_touches_image_boundary", "脑区接触输入图像边界，形态可能被截断")
    if angle is None or ratio is None:
        return reject("insufficient_spread", "脑区缺少稳定的 AP/SI 二维范围")
    if ratio < _MIN_PITCH_RATIO:
        return reject("low_anisotropy", "脑区 AP/SI 形态近圆，主轴方向不稳定")
    if abs(angle) > _MAX_PITCH_DEGREES:
        return reject("angle_out_of_bounds", "脑形态主轴倾角超出允许范围")
    midline_count, _, midline_scatter = _moments(mask, transform, midline_center=float(mean[0]))
    diagnostics["midline_point_count"] = midline_count
    if midline_count < _MIN_PITCH_VOXELS:
        return reject("insufficient_midline_voxels", "中央脑区体素不足，无法验证形态俯仰")
    midline_angle, midline_ratio = _principal_angle(midline_scatter)
    diagnostics.update(midline_angle_degrees=midline_angle, midline_ratio=midline_ratio)
    if midline_angle is None or midline_ratio is None or midline_ratio < _MIN_PITCH_RATIO:
        return reject("midline_low_anisotropy", "中央脑区主轴方向不稳定")
    # 主轴是一条无向直线，比较时按 180 度周期取最短角差。
    disagreement = abs((angle - midline_angle + 90.0) % 180.0 - 90.0)
    diagnostics["midline_disagreement_degrees"] = disagreement
    if disagreement > _MAX_MIDLINE_DISAGREEMENT_DEGREES:
        return reject("midline_disagreement", "全脑与中央脑区主轴倾角不一致")
    diagnostics["confidence"] = float(
        min(1.0, (min(ratio, midline_ratio) - 1.0) / (2.0 * (_MIN_PITCH_RATIO - 1.0)))
        * (1.0 - disagreement / (2.0 * _MAX_MIDLINE_DISAGREEMENT_DEGREES))
    )
    if abs(angle) < _MIN_PITCH_DEGREES:
        return reject("below_minimum_correction", "脑形态主轴已接近输入 AP，保持原俯仰", status="unchanged")
    diagnostics.update(status="accepted", accepted=True, angle_degrees=angle, reason="脑形态主轴及中央脑区一致性检查通过")
    return diagnostics


def _sample_boundary(mask: np.ndarray, voxel_axes: np.ndarray, origin: np.ndarray):
    """按固定种子抽取有上限的表面点，不构造全卷表面物理点数组。"""

    contour_image = sitk.BinaryContour(sitk.GetImageFromArray(mask.astype(np.uint8)))
    contour = sitk.GetArrayFromImage(contour_image) != 0
    count = int(np.count_nonzero(contour))
    selected = (
        np.sort(np.random.default_rng(0).choice(count, _MAX_SYMMETRY_POINTS, replace=False))
        if count > _MAX_SYMMETRY_POINTS
        else np.arange(count)
    )
    points = np.empty((len(selected), 3), dtype=float)
    offset = written = 0
    for z_index, section in enumerate(contour):
        yx = np.argwhere(section)
        stop = int(np.searchsorted(selected, offset + len(yx)))
        ranks = selected[written:stop] - offset
        if len(ranks):
            picked = yx[ranks]
            xyz = np.column_stack((picked[:, 1], picked[:, 0], np.full(len(picked), z_index)))
            points[written:stop] = origin + xyz @ voxel_axes.T
        offset += len(yx)
        written = stop
    return points, count


def estimate_brain_frame_with_diagnostics(
    target_mask: np.ndarray,
    image: sitk.Image,
    *,
    pitch_enabled: bool = True,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """先定左右对称面，再用全脑 AP/SI 形态主轴估计受限的前后倾角。

    全脑与中央左右各 10 mm 脑区的主轴必须一致，近圆、截断或大倾角时保留
    对称面内的输入方向。此方向不代表听眦线或 AC-PC 线。三轴切片范围始终
    仅包含全部脑区体素的立方体支撑；颅骨不参与基础范围。颅顶余量由规划
    阶段显式应用，周边骨轮廓由有界取景处理。物理坐标按原始存储方向换算。
    """

    mask = np.asarray(target_mask, dtype=bool)
    expected_shape = tuple(reversed(image.GetSize()))
    if mask.ndim != 3 or mask.shape != expected_shape:
        raise ValueError("颅脑 Mask 尺寸与三维输入图像不一致")
    if not np.any(mask):
        raise ValueError("颅脑 Mask 为空")
    image_direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    reference_axes = patient_reference_axes(image_direction)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    voxel_axes = image_direction @ np.diag(spacing)
    boundary_points, boundary_count = _sample_boundary(mask, voxel_axes, origin)
    lr, plane_center, symmetry = estimate_reflection_symmetry_plane(boundary_points, reference_axes)
    base_axes = build_symmetry_frame_axes(lr, reference_axes)
    transform = voxel_axes.T @ base_axes
    moments = _moments(mask, transform)
    count, mean, _ = moments
    brain_contacts = _boundary_contacts(mask)
    pitch = _estimate_pitch(mask, transform, moments, brain_contacts, pitch_enabled)
    final_axes = base_axes.copy()
    if pitch["accepted"]:
        theta = np.radians(pitch["angle_degrees"])
        ap, si = base_axes[:, 1], base_axes[:, 2]
        candidate_si = -np.sin(theta) * ap + np.cos(theta) * si
        # 输入已有较大斜角时，受限的相对俯仰仍可能使 SI 越过患者头侧。
        # 不能把足向轴当作 superior 写入规划；只拒绝本次俯仰，保留原 LR。
        if candidate_si[2] <= 1e-8:
            pitch.update(
                status="rejected", accepted=False, angle_degrees=0.0, confidence=0.0,
                reason="形态俯仰与患者头脚方向冲突",
                rejection_code="patient_head_direction_conflict",
            )
        else:
            final_axes[:, 1] = np.cos(theta) * ap + np.sin(theta) * si
            final_axes[:, 2] = candidate_si

    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for projection in _projected_layers(mask, voxel_axes.T @ final_axes):
        lower = np.minimum(lower, np.min(projection, axis=0))
        upper = np.maximum(upper, np.max(projection, axis=0))
    # 任一体素立方体沿单位轴的半宽等于各存储轴投影半宽之和。
    support = 0.5 * np.sum(np.abs(voxel_axes.T @ final_axes), axis=0)
    lower += origin @ final_axes - support
    upper += origin @ final_axes + support
    warnings = list(symmetry["warnings"])
    if pitch["status"] == "rejected":
        warnings.append(f"脑形态俯仰未采用：{pitch['reason']}，保留对称面内输入方向")
    if any(brain_contacts.values()):
        warnings.append("脑区接触输入图像边界，仅覆盖已采集脑区，无法保证扫描外脑区完整")
    confidence = float(symmetry["confidence"])
    if pitch["confidence"] > 0.0:
        confidence = (
            min(confidence, pitch["confidence"])
            if symmetry["accepted"]
            else float(pitch["confidence"])
        )
    frame = AnatomicalFrame(
        origin=_point(origin + base_axes @ mean),
        left_right_axis=_point(final_axes[:, 0]),
        anterior_posterior_axis=_point(final_axes[:, 1]),
        superior_inferior_axis=_point(final_axes[:, 2]),
        bounds=AnatomicalBounds(*(AxisBounds(float(lower[i]), float(upper[i])) for i in range(3))),
        method="brain_shape_principal_axis" if pitch["accepted"] else "brain_reflection_symmetry",
        confidence=confidence,
        applied=not bool(np.allclose(final_axes, reference_axes, atol=1e-8, rtol=0.0)),
        warnings=warnings,
    )
    diagnostics: dict[str, Any] = {
        "symmetry": symmetry,
        "pitch": pitch,
        "target_point_count": int(count),
        "boundary_point_count": boundary_count,
        "plane_center_lps": plane_center.tolist(),
        "pitch_policy": "brain_shape_principal_axis" if pitch_enabled else "project_input_ap_onto_symmetry_plane",
        "coverage": {
            "policy": "brain_only",
            "labels": ["brain"],
            "brain_voxel_count": int(count),
            "voxel_count": int(count),
            "source_boundary_contacts": brain_contacts,
            "truncated": any(brain_contacts.values()),
            "voxel_support_halfwidth_mm": support.tolist(),
        },
    }
    return AnatomicalGeometryResult(global_frame=frame), diagnostics
