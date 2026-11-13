"""只为椎间隙提出保守的主体候选，不改编号、MPR 和 CPR 共用的主体提取。

只有后方附件与可靠前方实体冲突时才提出局部替换，仍须由调用方核验端板。
数组采用 z/y/x 轴序，前后 y 约定与原主体候选算法一致。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _supported_layers(counts: np.ndarray) -> int:
    occupied = np.flatnonzero(counts >= max(1.0, float(counts.max()) * 0.1))
    if not len(occupied):
        return 0
    groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
    return max(len(group) for group in groups)


def _shape_evidence(mask: np.ndarray, spacing: np.ndarray) -> dict:
    points = np.argwhere(mask)
    # 局部 mask 投影等价于逐 (z, y) 统计 LR 体素数，避开批量处理中
    # 已禁用的二维结构化排序；只统计有占据的行与层，不把空隙当作支撑。
    lr_counts = mask.sum(axis=2)
    occupied_rows = lr_counts > 0
    widths = lr_counts[occupied_rows]
    depths = occupied_rows.sum(axis=1)
    depths = depths[depths > 0]
    return {
        "volume_voxels": len(points),
        "center_zyx": points.mean(axis=0).tolist(),
        "si_support_mm": _supported_layers(mask.sum(axis=(1, 2))) * float(spacing[0]),
        "ap_depth_mm": float(np.median(depths) * spacing[1]),
        "lr_width_mm": float(np.median(widths) * spacing[2]),
    }


def _core_radius(mask: np.ndarray, spacing: np.ndarray) -> float:
    """仅在连通块包围盒内计算内接半径，边界显式补一层背景。"""
    points = np.argwhere(mask)
    region = tuple(slice(int(a), int(b) + 1) for a, b in
                   zip(points.min(axis=0), points.max(axis=0)))
    distance = ndimage.distance_transform_edt(np.pad(mask[region], 1), sampling=spacing)
    return float(distance.max())


def validate_interspace_body(
    label_mask: np.ndarray,
    body_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    vertebra_name: str,
    diagnostics: dict | None = None,
) -> np.ndarray | None:
    """返回同尺寸局部候选 mask；返回 None 表示保留原主体，不修改输入。

    正常前方主体仅检查投影；位置后移和 SI 支撑不足同时触发时才重算候选
    及距离场。C1、C2、S1 保留专用提取规则，候选仍须通过独立端板核验。
    """
    raw = np.asarray(label_mask, dtype=bool)
    body = np.asarray(body_mask, dtype=bool)
    spacing = np.asarray(spacing_zyx, dtype=float)
    evidence = diagnostics if diagnostics is not None else {}
    evidence.update(method="interspace_body_validation", replaced=False,
                    expensive_validation=False)
    key = str(vertebra_name).rsplit("_", 1)[-1].upper()
    if key in {"C1", "C2", "S1", "SACRUM"}:
        evidence["reason"] = "specialized_anatomy"
        return None
    if (raw.ndim != 3 or body.shape != raw.shape or spacing.shape != (3,)
            or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)):
        evidence["reason"] = "invalid_input"
        return None

    raw_y = raw.sum(axis=(0, 2))
    body_y = body.sum(axis=(0, 2))
    raw_count, body_count = int(raw_y.sum()), int(body_y.sum())
    if raw_count < 8 or body_count < 8:
        evidence["reason"] = "insufficient_existing_body"
        return None
    occupied_y = np.flatnonzero(raw_y)
    y = np.arange(raw.shape[1], dtype=float)
    posterior_shift_mm = float((np.dot(y, body_y) / body_count
                                - np.dot(y, raw_y) / raw_count) * spacing[1])
    minimum_shift_mm = max(3.0, float((occupied_y[-1] - occupied_y[0] + 1)
                                    * spacing[1] * 0.05))
    if posterior_shift_mm <= minimum_shift_mm:
        evidence["reason"] = "anterior_body_unchanged"
        return None
    raw_z = raw.sum(axis=(1, 2))
    body_z = body.sum(axis=(1, 2))
    if _supported_layers(body_z) >= _supported_layers(raw_z) * 0.6:
        evidence["reason"] = "stable_existing_si_support"
        return None

    current = _shape_evidence(body, spacing)
    # 真实矮椎体不能仅因 SI 短就被前方大薄片替换；原主体还须呈现后附件
    # 的 AP 长、LR 窄形态。跨轴比较使用物理长度，而非切片数量。
    if (current["ap_depth_mm"] <= current["si_support_mm"] * 2.0
            or current["ap_depth_mm"] <= current["lr_width_mm"] * 1.8):
        evidence["reason"] = "no_posterior_appendage_shape"
        return None

    from .stage_localization import LocalizationStage

    evidence.update(expensive_validation=True,
                    posterior_shift_mm=posterior_shift_mm,
                    previous_body=current)
    candidates = LocalizationStage._vertebral_body_support_candidates(
        raw, spacing_zyx=tuple(float(value) for value in spacing),
    )
    components, count = ndimage.label(candidates, structure=np.ones((3, 3, 3), dtype=bool))
    body_center = np.asarray(current["center_zyx"])
    current_radius = _core_radius(body, spacing)
    current["core_radius_mm"] = current_radius
    acceptable = []
    for component_id, region in enumerate(ndimage.find_objects(components), start=1):
        if region is None:
            continue
        local = components[region] == component_id
        volume = int(local.sum())
        if volume < body_count * 1.5:
            continue
        candidate = np.zeros_like(body)
        candidate[region] = local
        candidate_evidence = _shape_evidence(candidate, spacing)
        candidate_center = np.asarray(candidate_evidence["center_zyx"])
        anterior_separation_mm = float((body_center[1] - candidate_center[1]) * spacing[1])
        if (anterior_separation_mm < max(5.0, current["lr_width_mm"] * 0.5)
                or candidate_evidence["si_support_mm"] < current["si_support_mm"] * 2.0
                or abs(candidate_center[2] - body_center[2]) * spacing[2]
                > max(current["lr_width_mm"], candidate_evidence["lr_width_mm"]) * 0.75):
            continue
        radius = _core_radius(candidate, spacing)
        # 物理内接核独立于旧 AP/LR 比例门槛；高、宽薄片不能仅凭体量通过。
        if radius < max(2.5, current_radius * 1.5,
                        min(candidate_evidence["si_support_mm"],
                            candidate_evidence["lr_width_mm"]) * 0.25):
            continue
        candidate_evidence.update(core_radius_mm=radius,
                                  anterior_separation_mm=anterior_separation_mm)
        acceptable.append((radius, volume, candidate, candidate_evidence))
    if not acceptable:
        evidence["reason"] = "no_verified_anterior_core"
        return None
    # 多个前体同时符合条件仍有歧义，不按微小评分差异强行选一个。
    if len(acceptable) != 1:
        evidence["reason"] = "ambiguous_anterior_cores"
        return None
    _, _, replacement, candidate_evidence = acceptable[0]
    for trim in (
        LocalizationStage._retain_main_3d_component_with_terminal_extensions,
        LocalizationStage._trim_thin_terminal_si_slices,
        LocalizationStage._trim_lr_edge_columns_by_si_height,
        LocalizationStage._trim_thin_terminal_si_slices,
        LocalizationStage._trim_transverse_edges_by_density,
    ):
        replacement = trim(replacement)
    # 原裁剪步骤可能削掉候选内核，发布前复核实际输出的体量和物理半径。
    if (int(replacement.sum()) < body_count * 1.5
            or _core_radius(replacement, spacing) < current_radius * 1.5):
        evidence["reason"] = "candidate_lost_core_after_trim"
        return None
    evidence.update(replaced=True, reason="posterior_appendage_conflicts_with_anterior_core",
                    replacement_body=candidate_evidence,
                    replacement_voxels=int(replacement.sum()),
                    candidate_count=count)
    return replacement
