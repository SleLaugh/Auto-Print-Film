"""从物理尺度的三维占据支撑中核对普通椎体主体。

薄前方结构及左右椎弓连接不应取代椎体内部。这里只恢复原 label 中
已有的体素；缺少稳定内部支撑时不构造新的椎体。
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DenseBodyCore:
    mask: np.ndarray
    restored_mask: np.ndarray
    kernel_zyx: tuple[int, int, int]
    reference_coverage: float
    center_displacement_fraction: float


def find_dense_body_core(
    label_mask: np.ndarray,
    reference_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> DenseBodyCore | None:
    """找靠近既有主体的密实核心，限制恢复距离以避开附件。

    核大小以已筛选主体的 LR 宽度定义，再转换为物理等大的三维窗口。
    核心限于中央 LR 带，避免通过两侧椎弓根连接到后方附件；恢复范围
    最多为该核的半宽，不沿整个 label 无界生长。
    """

    raw = np.asarray(label_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if (raw.ndim != 3 or reference.shape != raw.shape
            or spacing.shape != (3,) or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0.0)):
        return None
    reference_indices = np.argwhere(reference)
    if len(reference_indices) < 8:
        return None
    extent_mm = (np.ptp(reference_indices, axis=0) + 1) * spacing
    width_mm = float(extent_mm[2])
    if width_mm <= 0.0:
        return None
    kernel = np.maximum(3, np.rint(width_mm * 0.6 / spacing).astype(int) // 2 * 2 + 1)
    # 一个方向不足以容纳稳定内部窗口，不能凭过小残块推断完整主体。
    if np.any(np.asarray(raw.shape) < kernel):
        return None
    density = ndimage.uniform_filter(
        raw.astype(np.float32), size=tuple(kernel), mode="constant",
    )
    core = (density >= 0.85) & raw
    raw_indices = np.argwhere(raw)
    center_x = float(np.median(raw_indices[:, 2]))
    half_width = max(1, int(round(width_mm * 0.2 / spacing[2])))
    central_lr = np.abs(np.arange(raw.shape[2]) - center_x) <= half_width
    core &= central_lr[None, None, :]
    components, count = ndimage.label(core, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return None
    reference_center = reference_indices.mean(axis=0)
    candidates = []
    for component_id, region in enumerate(ndimage.find_objects(components), start=1):
        if region is None:
            continue
        local_indices = np.argwhere(components[region] == component_id)
        indices = local_indices + np.array([item.start for item in region])
        physical_extent = (np.ptp(indices, axis=0) + 1) * spacing
        if (physical_extent[0] < extent_mm[0] * 0.25
                or physical_extent[1] < width_mm * 0.2
                or len(indices) < 8):
            continue
        center_distance = float(np.linalg.norm((indices.mean(axis=0) - reference_center) * spacing))
        candidates.append((center_distance, -len(indices), component_id, indices))
    if not candidates:
        return None
    distance, _, selected_id, indices = min(candidates, key=lambda item: item[:2])
    if distance > width_mm * 0.75:
        return None
    selected = components == selected_id
    restored = ndimage.maximum_filter(selected, size=tuple(kernel), mode="constant") & raw
    return DenseBodyCore(
        mask=selected,
        restored_mask=restored,
        kernel_zyx=tuple(int(value) for value in kernel),
        reference_coverage=float(np.mean(reference[tuple(indices.T)])),
        center_displacement_fraction=distance / width_mm,
    )


def recover_body_from_dense_core(
    label_mask: np.ndarray,
    reference_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    diagnostics: dict | None = None,
) -> np.ndarray | None:
    """仅在既有主体漏掉稳定内部核心时，返回受原 label 约束的恢复主体。"""

    evidence = find_dense_body_core(label_mask, reference_mask, spacing_zyx)
    if (evidence is None or evidence.reference_coverage >= 0.9
            or evidence.center_displacement_fraction < 0.08):
        return None
    restored = evidence.restored_mask
    if int(restored.sum()) < int(np.count_nonzero(reference_mask)) * 0.8:
        return None
    if diagnostics is not None:
        indices = np.argwhere(evidence.mask)
        diagnostics.update(
            method="density_core_recovery",
            core_bbox_zyx={"min": indices.min(axis=0).tolist(),
                           "max": indices.max(axis=0).tolist()},
            core_kernel_zyx=list(evidence.kernel_zyx),
            core_reference_coverage=evidence.reference_coverage,
            core_center_displacement_fraction=evidence.center_displacement_fraction,
        )
    return restored


def _consistent_boundary_profiles(
    source: np.ndarray,
    body_indices: np.ndarray,
    spacing: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    raw_array: np.ndarray | None = None,
    required_sides: tuple[str, ...] = ("min", "max"),
) -> dict[str, tuple[np.ndarray, dict]] | None:
    """分别拟合 LR 切面的自然边界，再取稳定切面群的中位边界。"""
    from scipy.stats import theilslopes

    x_minimum, x_maximum = np.quantile(body_indices[:, 2], (0.05, 0.95))
    source = source[(source[:, 1] >= minimum[1]) & (source[:, 1] <= maximum[1])
                    & (source[:, 2] >= x_minimum) & (source[:, 2] <= x_maximum)]
    profiles = []
    expected_ap_span = (maximum[1] - minimum[1]) * spacing[1]
    for x in np.unique(source[:, 2]):
        section = source[source[:, 2] == x]
        rows = []
        for y in np.unique(section[:, 1]):
            z = np.sort(section[section[:, 1] == y, 0])
            groups = np.split(z, np.flatnonzero(np.diff(z) > 2) + 1)
            selected = max(groups, key=lambda group: (
                max(0, min(group[-1], maximum[0]) - max(group[0], minimum[0]) + 1),
                len(group),
            ))
            if len(selected) >= 2:
                rows.append((y, selected[0], selected[-1]))
        if len(rows) < 5:
            continue
        values = np.asarray(rows, dtype=float)
        if np.ptp(values[:, 0]) * spacing[1] < expected_ap_span * 0.8:
            continue
        profile = {"x": float(x), "values": values}
        for column, side in ((1, "min"), (2, "max")):
            if side not in required_sides:
                continue
            y_mm = values[:, 0] * spacing[1]
            z_mm = values[:, column] * spacing[0]
            slope, intercept, _, _ = theilslopes(z_mm, y_mm)
            residuals = np.abs(z_mm - (slope * y_mm + intercept))
            if (np.median(residuals) <= spacing[0] * 0.5
                    and np.percentile(residuals, 90) <= spacing[0] * 1.5):
                ct_evidence = None
                if raw_array is not None:
                    points = np.column_stack((values[:, column], values[:, 0],
                                              np.full(len(values), x)))
                    ct_evidence = _boundary_density_drop(
                        raw_array, points, spacing, float(slope), -1 if column == 1 else 1,
                    )
                    if ct_evidence is None:
                        continue
                profile[side] = float(np.degrees(np.arctan(slope)))
                profile[f"{side}_slope"] = float(slope)
                profile[f"{side}_intercept"] = float(intercept)
                profile[f"{side}_residual"] = float(np.median(residuals))
                profile[f"{side}_ct"] = ct_evidence
        profiles.append(profile)
    if len(profiles) < 3:
        return None
    results = {}
    for column, side in ((1, "min"), (2, "max")):
        if side not in required_sides:
            continue
        reliable = [profile for profile in profiles if side in profile]
        if len(reliable) < 3:
            return None
        angles = np.array([profile[side] for profile in reliable])
        # 这是 LR 切面间一致性的容差，不限制最终端面与 SI 的夹角。
        groups = [np.flatnonzero(np.abs(angles - angle) <= 5.0) for angle in angles]
        selected_group = max(groups, key=lambda group: (len(group), np.ptp([
            reliable[int(index)]["x"] for index in group
        ])))
        selected = [reliable[int(index)] for index in selected_group]
        x_span_mm = np.ptp([profile["x"] for profile in selected]) * spacing[2]
        body_lr_mm = (np.ptp(body_indices[:, 2]) + 1) * spacing[2]
        if (len(selected) < max(3, len(profiles) * 0.2)
                or x_span_mm < body_lr_mm * 0.15):
            return None
        rows = []
        common_y_mm = (minimum[1] + maximum[1]) * spacing[1] * 0.5
        consensus_slope = float(np.median([profile[f"{side}_slope"] for profile in selected]))
        consensus_center_z_mm = float(np.median([
            profile[f"{side}_slope"] * common_y_mm + profile[f"{side}_intercept"]
            for profile in selected
        ]))
        for y in range(int(minimum[1]), int(maximum[1]) + 1):
            z_values = [profile["values"][profile["values"][:, 0] == y, column]
                        for profile in selected]
            z_values = [float(value[0]) for value in z_values if len(value)]
            if len(z_values) >= len(selected) * 0.8:
                # 直接合并边界点会因各 LR 切面的 AP 覆盖不同而切换中位
                # 截距，甚至把一组平行水平端面合成为斜线。合并各自拟合的
                # 斜率及共同 AP 位置高度，保留 LR 群的一致方向。
                z_mm = consensus_center_z_mm + consensus_slope * (y * spacing[1] - common_y_mm)
                rows.append((y, z_mm / spacing[0]))
        if len(rows) < 3:
            return None
        results[side] = (np.asarray(rows), {
            "lr_profile_count": len(selected),
            "lr_profile_available": len(profiles),
            "lr_profile_extent_voxels": [min(profile["x"] for profile in selected),
                                          max(profile["x"] for profile in selected)],
            "median_profile_residual_mm": float(np.median([
                profile[f"{side}_residual"] for profile in selected
            ])),
            "ct_validation": "density_drop" if raw_array is not None else "unavailable",
            **({} if raw_array is None else {
                "ct_min_positive_fraction": min(
                    evidence["positive_fraction"] for profile in selected
                    for evidence in profile[f"{side}_ct"]
                ),
                "ct_min_normalized_drop": min(
                    evidence["normalized_drop"] for profile in selected
                    for evidence in profile[f"{side}_ct"]
                ),
            }),
        })
    return results


def _boundary_density_drop(
    raw_array: np.ndarray,
    points_zyx: np.ndarray,
    spacing: np.ndarray,
    slope: float,
    outward_sign: int,
) -> list[dict] | None:
    """比较自然边界内外的局部相对密度下降，不依赖绝对 HU 阈值。"""
    normal = np.array([float(outward_sign), -float(outward_sign) * slope, 0.0])
    normal /= np.linalg.norm(normal)

    def sample(distances: np.ndarray) -> np.ndarray:
        coordinates = []
        for distance in distances:
            for lr_offset in (-0.4, 0.0, 0.4):
                displaced = points_zyx + normal[None, :] * (distance / spacing)[None, :]
                displaced[:, 2] += lr_offset / spacing[2]
                coordinates.append(displaced)
        sampled = ndimage.map_coordinates(
            raw_array, np.concatenate(coordinates).T, output=np.float32,
            order=1, mode="constant", cval=np.nan, prefilter=False,
        ).reshape(len(coordinates), len(points_zyx))
        return np.median(sampled, axis=0)

    evidence = []
    for near, far in ((0.75, 1.25), (1.0, 2.0), (2.0, 3.0)):
        distances = np.linspace(near, far, 5)
        inside = sample(-distances)
        outside = sample(distances)
        if not np.all(np.isfinite(inside)) or not np.all(np.isfinite(outside)):
            return None
        difference = inside - outside
        scale = max(1.0, float(np.ptp(np.quantile(np.concatenate((inside, outside)), (0.1, 0.9)))))
        positive_fraction = float(np.mean(difference > 0.0))
        normalized_drop = float(np.median(difference / scale))
        if positive_fraction < 0.8 or normalized_drop < 0.2:
            return None
        evidence.append({"window_mm": [near, far],
                         "positive_fraction": positive_fraction,
                         "normalized_drop": normalized_drop})
    return evidence


def fit_body_endplates_from_core(
    body_indices_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    core_bbox_zyx: dict,
    *,
    source_indices_zyx: np.ndarray | None = None,
    raw_array: np.ndarray | None = None,
    required_sides: tuple[str, ...] = ("min", "max"),
) -> list[dict] | None:
    """仅在密实核心的 AP 支撑内拟合上下边界，不把覆盖包络侧壁当端板。

    返回全局 voxel 端点及物理 ZYX 单位外法向；调用者负责用图像 direction
    和 origin 转为物理坐标。原 label 的外包络不作为拟合方向证据。
    """
    from scipy.stats import theilslopes

    indices = np.asarray(body_indices_zyx)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if indices.ndim != 2 or indices.shape[1] != 3 or len(indices) < 8:
        return None
    if not required_sides or any(side not in {"min", "max"} for side in required_sides):
        return None
    try:
        minimum = np.asarray(core_bbox_zyx["min"], dtype=float)
        maximum = np.asarray(core_bbox_zyx["max"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if (minimum.shape != (3,) or maximum.shape != (3,) or spacing.shape != (3,)
            or np.any(spacing <= 0) or not np.all(np.isfinite(spacing))):
        return None
    source = indices if source_indices_zyx is None else np.asarray(source_indices_zyx)
    if source.ndim != 2 or source.shape[1] != 3 or not len(source):
        return None
    if raw_array is not None and (np.ndim(raw_array) != 3 or not np.size(raw_array)):
        return None
    profile_values = None
    if source_indices_zyx is not None:
        profile_values = _consistent_boundary_profiles(
            source, indices, spacing, minimum, maximum, raw_array=raw_array,
            required_sides=required_sides,
        )
        if profile_values is None:
            return None
    # 无原 label 的独立几何调用仍可验证主体的中央边界；生产恢复路径
    # 只使用上面多张 LR 原始切面形成共识的自然边界。
    source = source[(source[:, 2] >= minimum[2]) & (source[:, 2] <= maximum[2])]
    if not len(source):
        return None
    projections, counts = np.unique(source[:, :2], axis=0, return_counts=True)
    ap_rows = np.unique(projections[:, 1])
    ap_rows = ap_rows[(ap_rows >= minimum[1]) & (ap_rows <= maximum[1])]
    if len(ap_rows) < 3:
        return None
    rows = []
    for y in ap_rows:
        row = projections[:, 1] == y
        required_width = max(2.0, float(counts[row].max()) * 0.35)
        reliable_z = projections[row & (counts >= required_width), 0]
        if len(reliable_z) >= 2:
            # 小分割缺口允许跨一层；远离核心的另一个 SI 块不撑大边界。
            groups = np.split(reliable_z, np.flatnonzero(np.diff(reliable_z) > 2) + 1)
            selected = max(groups, key=lambda group: (
                max(0, min(group[-1], maximum[0]) - max(group[0], minimum[0]) + 1),
                len(group),
            ))
            if len(selected) >= 2:
                rows.append((y, selected.min(), selected.max()))
    if len(rows) < 3:
        return None
    values = np.asarray(rows, dtype=float)
    y_physical = values[:, 0] * spacing[1]
    body_extent = (np.ptp(indices, axis=0) + 1) * spacing
    if np.ptp(y_physical) < body_extent[2] * 0.2:
        return None
    center_x = float(np.median(indices[:, 2]))
    result = []
    for column, sign in ((1, -1), (2, 1)):
        side = "min" if sign < 0 else "max"
        if side not in required_sides:
            continue
        fit_values = values[:, (0, column)] if profile_values is None else profile_values[side][0]
        fit_y_physical = fit_values[:, 0] * spacing[1]
        z_physical = fit_values[:, 1] * spacing[0]
        slope, intercept, _, _ = theilslopes(z_physical, fit_y_physical)
        residuals = np.abs(z_physical - (slope * fit_y_physical + intercept))
        if (np.median(residuals) > max(spacing[0], body_extent[0] * 0.1)
                or np.percentile(residuals, 90) > max(spacing[0] * 2, body_extent[0] * 0.2)):
            return None
        endpoints = [[float((slope * y * spacing[1] + intercept) / spacing[0]),
                      float(y), center_x] for y in (fit_values[0, 0], fit_values[-1, 0])]
        normal = np.array([float(sign), -float(sign) * slope, 0.0])
        normal /= np.linalg.norm(normal)
        result.append({
            "side": side,
            "endpoints_zyx": endpoints,
            "normal_zyx": normal.tolist(),
            "point_count": len(fit_values),
            "median_residual_mm": float(np.median(residuals)),
            "method": "dense_core_supported_boundary",
            "source": "original_label" if source_indices_zyx is not None else "body_mask",
            **({} if profile_values is None else profile_values[side][1]),
        })
    if len(result) == 2 and any(
        result[1]["endpoints_zyx"][i][0] - result[0]["endpoints_zyx"][i][0] < 1.0
        for i in (0, 1)
    ):
        return None
    return result
