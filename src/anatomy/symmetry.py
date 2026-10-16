"""用有界、稳健的镜像匹配估计物理空间中的左右对称面。"""

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree


_MIN_POINTS = 64
_TRIM_FRACTION = 0.90
_MIN_CORRECTION_DEGREES = 0.75
_MIN_IMPROVEMENT_FRACTION = 0.08


def build_symmetry_frame_axes(
    normal_lps: np.ndarray,
    reference_axes: np.ndarray,
) -> np.ndarray:
    """以左右法向定 LR，将输入 AP 投影到对称面，SI 取 LR×AP。

    此策略只纠正左右旋转和侧倾，不从左右对称性推断面内的前后倾角。
    返回列向量依次为 LR/AP/SI 的单位正交右手基。
    """

    normal = np.asarray(normal_lps, dtype=float)
    axes = np.asarray(reference_axes, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-8:
        raise ValueError("左右对称面法向必须为有限非零三维向量")
    if (
        axes.shape != (3, 3)
        or not np.all(np.isfinite(axes))
        or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6)
        or float(np.linalg.det(axes)) < 1.0 - 1e-6
    ):
        raise ValueError("对称面参考轴必须组成单位正交右手系")
    normal = normal / np.linalg.norm(normal)
    if float(np.dot(normal, axes[:, 0])) < 0.0:
        normal = -normal
    ap = axes[:, 1] - float(np.dot(axes[:, 1], normal)) * normal
    if float(np.linalg.norm(ap)) < 1e-6:
        raise ValueError("左右法向与输入 AP 近似平行，无法保留前后倾角")
    ap /= np.linalg.norm(ap)
    return np.column_stack((normal, ap, np.cross(normal, ap)))


def _lr_rotation_ambiguities(
    local: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    max_angle_degrees: float,
) -> list[dict[str, Any]]:
    """检查与许可 LR 方向相交的近重根子空间，而非一律拒绝重复谱。

    LR 特征值独立、AP/SI 重复时只有面内俯仰不确定，不影响左右法向。
    若 LR 本身位于近重根平面，则以低阶角向分布及径向轮廓判断有无真实
    方位信息。有限圆柱的高阶采样相位不能作为患者姿态依据。
    """

    centered = local - np.mean(local, axis=0)
    results = []
    for first, second, other in ((0, 1, 2), (1, 2, 0)):
        relative_gap = float((eigenvalues[second] - eigenvalues[first]) / eigenvalues[-1])
        axis = eigenvectors[:, other]
        if relative_gap >= 0.05 or abs(float(axis[0])) > math.sin(math.radians(max_angle_degrees)):
            continue
        planar = centered @ eigenvectors[:, [first, second]]
        radius = np.linalg.norm(planar, axis=1)
        valid = radius > max(float(np.max(radius)) * 1e-6, 1e-8)
        angular = (planar[valid, 0] + 1j * planar[valid, 1]) / radius[valid]
        axial = centered[valid] @ axis
        radius = radius[valid]
        information = 0.0
        quantiles = np.percentile(axial, [0.0, 33.0, 67.0, 100.0])
        axial_tolerance = max(float(np.ptp(axial)) * 1e-9, 1e-9)
        regions = [np.ones(len(angular), dtype=bool)]
        regions.extend(
            (axial >= low - axial_tolerance) & (axial <= high + axial_tolerance)
            for low, high in zip(quantiles[:-1], quantiles[1:])
        )
        for region in regions:
            if np.count_nonzero(region) < 16:
                continue
            for weight in (np.ones(np.count_nonzero(region)), radius[region] ** 2):
                denominator = float(np.sum(weight))
                for harmonic in range(1, 7):
                    coefficient = abs(np.sum(weight * angular[region] ** harmonic)) / denominator
                    information = max(information, float(coefficient))
        results.append({
            "axis": axis,
            "relative_eigenvalue_gap": relative_gap,
            "angular_information": information,
        })
    return results


def _equivalent_reflection_direction(
    normal: np.ndarray,
    offset: float,
    ambiguities: list[dict[str, Any]],
    objective,
    *,
    max_angle_degrees: float,
    best_error_mm: float,
    baseline_error_mm: float,
) -> dict[str, Any] | None:
    """对近重根方向作有界角扫描，拒绝相离的等价解或平坦方位。

    每个候选旋转轴最多 61 个粗查点（默认 30 度界限）和四次局部细化，
    不增加点云数量或工作线程。此检查仅针对可能改变 LR 的近重根方向；
    方差接近但轮廓有独特左右特征的目标仍可通过。
    """

    tolerance = max(0.05, 0.03 * baseline_error_mm, 0.03 * best_error_mm)
    minimum_separation = min(3.0, max_angle_degrees / 2.0)
    cosine_limit = math.cos(math.radians(max_angle_degrees))
    for ambiguity in ambiguities:
        axis = ambiguity["axis"]

        def rotated(theta_degrees):
            theta = math.radians(float(theta_degrees))
            candidate = (
                normal * math.cos(theta)
                + np.cross(axis, normal) * math.sin(theta)
                + axis * float(axis @ normal) * (1.0 - math.cos(theta))
            )
            return candidate / np.linalg.norm(candidate)

        def error(theta_degrees):
            candidate = rotated(theta_degrees)
            separation = math.degrees(math.acos(float(np.clip(candidate @ normal, -1.0, 1.0))))
            if candidate[0] < cosine_limit or separation < minimum_separation:
                return math.inf
            # 保持过同一中心点的候选面，避免将优化位移误解释为角向信息。
            candidate_offset = offset * float(candidate @ normal)
            parameters = np.array([candidate[1] / candidate[0], candidate[2] / candidate[0], candidate_offset])
            return math.sqrt(max(0.0, float(objective(parameters))))

        steps = np.linspace(-2.0 * max_angle_degrees, 2.0 * max_angle_degrees, 61)
        errors = [error(theta) for theta in steps]
        eligible = [index for index, value in enumerate(errors) if math.isfinite(value)]
        if not eligible:
            continue
        best_index = min(eligible, key=lambda index: errors[index])
        alternatives = [(float(steps[best_index]), errors[best_index])]
        minima = [
            index for index in eligible
            if 0 < index < len(steps) - 1
            and errors[index] <= errors[index - 1]
            and errors[index] <= errors[index + 1]
        ]
        # 非法域在粗查中用 inf 标记，但不能把 inf 传进 Brent 的抛物线运算。
        # 用高于所有粗查残差的有限惩罚做细化，最终仍回到严格域检查，非法
        # 角度不会因惩罚值有限而进入等价解判定。
        invalid_penalty = 4.0 * (max(errors[index] for index in eligible) + 1.0)

        def refinement_error(theta_degrees):
            value = error(theta_degrees)
            return value if math.isfinite(value) else invalid_penalty

        for index in sorted(minima, key=lambda item: errors[item])[:4]:
            refined = minimize_scalar(
                refinement_error,
                bounds=(float(steps[index - 1]), float(steps[index + 1])),
                method="bounded",
                options={"xatol": 0.05, "maxiter": 20},
            )
            strict_error = error(refined.x)
            if refined.success and math.isfinite(strict_error):
                alternatives.append((float(refined.x), strict_error))
        theta, alternative_error = min(alternatives, key=lambda item: item[1])
        if alternative_error <= best_error_mm + tolerance:
            alternative = rotated(theta)
            return {
                "alternative_normal_local": alternative.tolist(),
                "separation_degrees": math.degrees(math.acos(float(np.clip(alternative @ normal, -1.0, 1.0)))),
                "alternative_error_mm": alternative_error,
                "equivalence_tolerance_mm": tolerance,
            }
    return None


def estimate_reflection_symmetry_plane(
    points_physical: np.ndarray,
    reference_axes: np.ndarray,
    *,
    max_points: int = 6000,
    max_angle_degrees: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """拟合接近输入左右轴的三维镜像面，并在证据不足时保留原方向。

    ``points_physical`` 是患者物理毫米坐标，宜使用目标边界或左右同源目标
    的联合点云。``reference_axes`` 的列依次为 LR/AP/SI，必须单位正交且
    为右手系。返回左右法向、面上一点和可序列化的质量诊断。法向始终与
    输入 LR 同向；只有 diagnostics["accepted"] 为真时才应用新方向。

    优化只确定左右对称面，不能确定面内的俯仰角。采样固定种子且有数量
    上限；90% 截尾近邻误差降低局灶缺损和附件对姿态的影响，但不能把真实
    解剖差异变成完全对称。触及搜索边界、缺少改善或拟合质量不足会回退。
    """

    points = np.asarray(points_physical, dtype=float)
    axes = np.asarray(reference_axes, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("对称面点云必须为 N×3 物理坐标")
    if not np.all(np.isfinite(points)):
        raise ValueError("对称面点云包含非有限坐标")
    if (
        axes.shape != (3, 3)
        or not np.all(np.isfinite(axes))
        or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6)
        or float(np.linalg.det(axes)) < 1.0 - 1e-6
    ):
        raise ValueError("对称面参考轴必须组成单位正交右手系")
    if not isinstance(max_points, (int, np.integer)) or max_points < _MIN_POINTS:
        raise ValueError(f"对称面采样上限至少为 {_MIN_POINTS}")
    if not math.isfinite(max_angle_degrees) or not 1.0 <= max_angle_degrees <= 45.0:
        raise ValueError("对称面最大偏角必须位于 1 到 45 度")

    center = np.mean(points, axis=0) if len(points) else np.zeros(3, dtype=float)
    reference_lr = axes[:, 0].copy()
    diagnostics: dict[str, Any] = {
        "method": "bounded_reflection_symmetry",
        "accepted": False,
        "status": "rejected",
        "reason": "",
        "warnings": [],
        "point_count": int(len(points)),
        "sample_count": int(min(len(points), max_points)),
        "max_points": int(max_points),
        "max_angle_degrees": float(max_angle_degrees),
        "trim_fraction": _TRIM_FRACTION,
        "angle_degrees": 0.0,
        "before_error_mm": None,
        "after_error_mm": None,
        "improvement_fraction": 0.0,
        "confidence": 0.0,
        "direction_uniqueness": {"status": "not_checked"},
    }

    def fallback(reason: str, *, unchanged: bool = False):
        diagnostics["status"] = "unchanged" if unchanged else "rejected"
        diagnostics["reason"] = reason
        if not unchanged:
            diagnostics["warnings"] = [reason]
        return reference_lr, center, diagnostics

    if len(points) < _MIN_POINTS:
        return fallback("目标点数不足，保留输入方向")
    if len(points) > max_points:
        selection = np.sort(np.random.default_rng(0).choice(len(points), max_points, replace=False))
        sample = points[selection]
    else:
        sample = points
    local = (sample - center) @ axes
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(local, rowvar=False))
    if eigenvalues[0] <= 1e-6 or eigenvalues[-1] <= 1e-6:
        return fallback("目标点云缺少三维范围，无法确定左右对称面")
    # 球形目标的任意过心面都近似对称，点云抽样噪声不能当成姿态证据。
    spectral_gaps = np.diff(eigenvalues) / eigenvalues[-1]
    if float(np.max(spectral_gaps)) < 0.05:
        diagnostics["direction_uniqueness"] = {"status": "ambiguous", "reason": "near_spherical_shape"}
        return fallback("目标近似球形，左右对称方向不唯一")

    ambiguities = _lr_rotation_ambiguities(local, eigenvalues, eigenvectors, max_angle_degrees)
    diagnostics["direction_uniqueness"] = {
        "status": "pending" if ambiguities else "passed",
        "lr_degenerate_subspaces": [
            {"axis_local": item["axis"].tolist(),
             "relative_eigenvalue_gap": item["relative_eigenvalue_gap"],
             "angular_information": item["angular_information"]}
            for item in ambiguities
        ],
    }
    if any(item["angular_information"] < 0.05 for item in ambiguities):
        diagnostics["direction_uniqueness"]["status"] = "ambiguous"
        return fallback("目标绕轴的方位信息不足，左右对称面不唯一，保留输入方向")

    tree = cKDTree(local)
    neighbour_distances = tree.query(local, k=2, workers=1)[0][:, 1]
    sample_spacing = float(np.median(neighbour_distances))
    robust_span = np.percentile(local, 99.0, axis=0) - np.percentile(local, 1.0, axis=0)
    if float(np.min(robust_span)) < 1.0:
        return fallback("目标物理范围过小，保留输入方向")
    offset_limit = min(15.0, max(2.0, float(robust_span[0]) * 0.15))
    slope_limit = math.tan(math.radians(max_angle_degrees))
    bounds = [(-slope_limit, slope_limit), (-slope_limit, slope_limit), (-offset_limit, offset_limit)]
    retained_count = max(1, int(math.ceil(len(local) * _TRIM_FRACTION)))

    def residuals(parameters: np.ndarray) -> np.ndarray:
        normal = np.array([1.0, parameters[0], parameters[1]], dtype=float)
        normal /= np.linalg.norm(normal)
        reflected = local - 2.0 * (local @ normal - parameters[2])[:, None] * normal
        return tree.query(reflected, k=1, workers=1)[0]

    def objective(parameters: np.ndarray) -> float:
        deviations = residuals(parameters)
        retained = np.partition(deviations, retained_count - 1)[:retained_count]
        error = float(np.mean(retained * retained))
        slope = math.hypot(float(parameters[0]), float(parameters[1]))
        # 两个斜率各自有界之外，再限制总偏角，避免对角方向突破最大角度。
        if slope > slope_limit:
            error += (slope - slope_limit) ** 2 * float(np.max(robust_span)) ** 2
        return error

    baseline = objective(np.zeros(3, dtype=float))
    starts = [np.zeros(3, dtype=float)]
    pca_index = int(np.argmax(np.abs(eigenvectors[0, :])))
    pca_normal = eigenvectors[:, pca_index]
    if pca_normal[0] < 0.0:
        pca_normal = -pca_normal
    if pca_normal[0] > math.cos(math.radians(max_angle_degrees)):
        starts.append(np.array([pca_normal[1] / pca_normal[0], pca_normal[2] / pca_normal[0], 0.0]))
    candidates = [
        minimize(
            objective,
            start,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 45, "xtol": 1e-4, "ftol": 1e-5},
        )
        for start in starts
    ]
    finite_candidates = [result for result in candidates if np.isfinite(result.fun) and np.all(np.isfinite(result.x))]
    if not finite_candidates:
        return fallback("镜像面优化未得到有限结果，保留输入方向")
    best = min(finite_candidates, key=lambda result: float(result.fun))
    normal_local = np.array([1.0, best.x[0], best.x[1]], dtype=float)
    normal_local /= np.linalg.norm(normal_local)
    angle = math.degrees(math.acos(float(np.clip(normal_local[0], -1.0, 1.0))))
    after = float(np.sqrt(max(0.0, best.fun)))
    before = float(np.sqrt(max(0.0, baseline)))
    improvement = (before - after) / max(before, 1e-6)
    inlier_tolerance = max(2.5 * sample_spacing, float(robust_span[0]) * 0.04, 1.0)
    distances = residuals(best.x)
    inlier_fraction = float(np.mean(distances <= inlier_tolerance))
    quality_limit = max(1.5 * sample_spacing, float(robust_span[0]) * 0.04, 1.0)
    diagnostics.update({
        "before_error_mm": before,
        "after_error_mm": after,
        "improvement_fraction": float(improvement),
        "angle_degrees": float(angle),
        "candidate_normal_lps": (axes @ normal_local).tolist(),
        "offset_mm": float(best.x[2]),
        "offset_limit_mm": float(offset_limit),
        "sample_spacing_mm": sample_spacing,
        "inlier_tolerance_mm": inlier_tolerance,
        "inlier_fraction": inlier_fraction,
        "quality_limit_mm": quality_limit,
        "optimizer_success": bool(best.success),
    })
    if not best.success:
        return fallback("镜像面优化未收敛，保留输入方向")
    if angle >= max_angle_degrees - 0.5 or abs(float(best.x[2])) >= offset_limit * 0.98:
        return fallback("镜像面候选触及角度或位移边界，保留输入方向")
    if after > quality_limit or inlier_fraction < 0.70:
        return fallback("目标镜像匹配质量不足，保留输入方向")
    equivalent = _equivalent_reflection_direction(
        normal_local, float(best.x[2]), ambiguities, objective,
        max_angle_degrees=max_angle_degrees,
        best_error_mm=after, baseline_error_mm=before,
    )
    if equivalent is not None:
        diagnostics["direction_uniqueness"].update({"status": "ambiguous", **equivalent})
        return fallback("搜索范围存在等价或平坦的镜像方向，左右对称面不唯一，保留输入方向")
    diagnostics["direction_uniqueness"]["status"] = "passed"
    if angle < _MIN_CORRECTION_DEGREES:
        return fallback("目标已接近输入方向，无需姿态调整", unchanged=True)
    if improvement < _MIN_IMPROVEMENT_FRACTION or before - after < max(0.05, sample_spacing * 0.05):
        return fallback("镜像匹配改善不足，保留输入方向")
    normal = axes @ normal_local
    plane_center = center + float(best.x[2]) * normal
    diagnostics.update({
        "accepted": True,
        "status": "accepted",
        "reason": "镜像匹配在搜索范围内稳定改善",
        "confidence": float(min(1.0, max(0.0, improvement)) * inlier_fraction),
        "normal_lps": normal.tolist(),
        "plane_center_lps": plane_center.tolist(),
    })
    return normal, plane_center, diagnostics
