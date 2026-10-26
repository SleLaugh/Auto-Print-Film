"""Recover split vertebral body labels from independently separated foreground.

This repair does not create anatomical foreground. Two pure semantic anchors must
bound a complete sequence of anterior bodies, and CT must support each boundary
that would be affected. Only the independently extracted body support is mutable;
posterior elements keep the model's labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from scipy.ndimage import label as connected_components

from src.models import LabelTarget


_BODY_NAMES = (
    "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3",
    "vertebrae_L2", "vertebrae_L1",
) + tuple(f"vertebrae_T{number}" for number in range(12, 0, -1)) + tuple(
    f"vertebrae_C{number}" for number in range(7, 1, -1)
)
_BODY_INDEX = {name: index for index, name in enumerate(_BODY_NAMES)}
_MINIMUM_MIXED_FRACTION = 0.15
_ANCHOR_PURITY = 0.85


def _ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, LabelTarget):
        value = value.label_ids
    elif isinstance(value, Mapping):
        value = value.get("label_ids", value.get("ids", ()))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple, set, np.ndarray)):
        return ()
    return tuple(sorted({int(item) for item in value
                         if isinstance(item, (int, np.integer))
                         and not isinstance(item, bool) and int(item) > 0}))


def _requested_names(label_targets: Any) -> set[str]:
    if isinstance(label_targets, Mapping):
        return set(label_targets).intersection(_BODY_INDEX)
    if isinstance(label_targets, (list, tuple, set)):
        return {value.label_name if isinstance(value, LabelTarget) else str(value)
                for value in label_targets}.intersection(_BODY_INDEX)
    return set()


def _image_geometry(image: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(image, "GetSpacing"):
        spacing = image.GetSpacing()
        direction = image.GetDirection()
    elif isinstance(image, Mapping):
        spacing = image.get("spacing_xyz", image.get("spacing"))
        direction = image.get("direction", np.eye(3).ravel())
    else:
        raise ValueError("缺少图像物理几何")
    spacing = np.asarray(spacing, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if (spacing.shape != (3,) or np.any(~np.isfinite(spacing))
            or np.any(spacing <= 0) or direction.size != 9):
        raise ValueError("无效的图像间距或方向")
    direction = direction.reshape(3, 3)
    if (np.any(~np.isfinite(direction))
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5)
            or abs(direction[2, 2]) < 0.5):
        raise ValueError("图像头脚方向不适合前体序列验证")
    # The support extractor identifies anterior bands at low array Y. A source
    # reflected in AP must be normalized by the caller before using that rule.
    if direction[1, 1] < 0.5:
        raise ValueError("图像前后方向不适合前体支撑提取")
    return spacing[::-1], direction


def _boundary_ct_evidence(raw, lower, upper, spacing, polarity):
    """Verify a CT depression near facing support, excluding body interiors.

    Counting all minima between body centers confuses ordinary cancellous bone
    with another disc. Here only the actual boundary neighborhood provides the
    valley, while separate interior shoulders must both be appreciably denser.
    """
    lower_face = float(np.max(lower["indices"][:, 0] * polarity))
    upper_face = float(np.min(upper["indices"][:, 0] * polarity))
    gap_center = (lower_face + upper_face) / 2.0
    gap_mm = (upper_face - lower_face - 1.0) * spacing[0]
    smaller_height = min(lower["height_mm"], upper["height_mm"])
    if gap_mm < -smaller_height * 0.35 or gap_mm > smaller_height * 0.75:
        return {"accepted": False, "reason": "相向前体边界距离不可靠"}
    lower_center = float(lower["center"][0] * polarity)
    upper_center = float(upper["center"][0] * polarity)
    if not lower_center < gap_center < upper_center:
        return {"accepted": False, "reason": "前体边界未落在相邻中心之间"}
    half_y = max(1, int(round(3.5 / spacing[1])))
    half_x = max(1, int(round(8.0 / spacing[2])))
    radius = max(1.0, 3.0 / spacing[0])
    shoulder_radius = max(radius + 1.0, 10.0 / spacing[0])
    first = max(lower_center, gap_center - shoulder_radius)
    last = min(upper_center, gap_center + shoulder_radius)
    medians, positions = [], []
    for oriented_z in range(int(np.floor(first)), int(np.ceil(last)) + 1):
        z = int(round(oriented_z * polarity))
        if z < 0 or z >= raw.shape[0]:
            continue
        fraction = (oriented_z - lower_center) / (upper_center - lower_center)
        y, x = np.rint(lower["center"][1:] * (1.0 - fraction)
                       + upper["center"][1:] * fraction).astype(int)
        region = raw[z, max(0, y-half_y):min(raw.shape[1], y+half_y+1),
                     max(0, x-half_x):min(raw.shape[2], x+half_x+1)]
        finite = region[np.isfinite(region)]
        if not len(finite):
            return {"accepted": False, "reason": "CT 中央走廊缺少有效密度"}
        positions.append(float(oriented_z))
        medians.append(float(np.median(finite)))
    positions, medians = np.asarray(positions), np.asarray(medians)
    near = np.abs(positions - gap_center) <= radius
    lower_shoulder = (positions < gap_center - radius) & (positions >= first)
    upper_shoulder = (positions > gap_center + radius) & (positions <= last)
    if not all(np.any(mask) for mask in (near, lower_shoulder, upper_shoulder)):
        return {"accepted": False, "reason": "CT 边界两侧缺少独立主体支撑"}
    valley = float(np.min(medians[near]))
    shoulders = [float(np.max(medians[mask]))
                 for mask in (lower_shoulder, upper_shoulder)]
    accepted = min(shoulders) - valley >= 25.0
    return {"accepted": bool(accepted), "gap_center_z": gap_center * polarity,
            "gap_mm": gap_mm, "valley_median_hu": valley,
            "shoulder_median_hu": shoulders,
            "reason": "CT 相向边界存在独立低密度谷" if accepted
                      else "CT 相向边界缺少低密度谷"}


def repair_mixed_spine_body_labels(
    seg_array: np.ndarray,
    raw_array_loader_or_array: np.ndarray | Callable[[], np.ndarray],
    image: Any,
    label_map: Mapping[str, Any],
    *,
    label_targets: Any,
    support_extractor: Callable[..., np.ndarray],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Reassociate multiple mixed anterior bodies within pure semantic anchors.

    ``None`` means the original segmentation must remain in use. The caller is
    responsible for validating the raw image's physical alignment in its loader;
    this function additionally checks its array shape and numeric CT support.
    """
    diagnostic: dict[str, Any] = {"status": "unchanged", "reason": "未发现连续前体标签混合"}

    def skip(reason):
        diagnostic.update(status="skipped", reason=reason)
        return None, diagnostic

    if (not isinstance(seg_array, np.ndarray) or seg_array.ndim != 3
            or not seg_array.size or not np.issubdtype(seg_array.dtype, np.integer)
            or np.any(seg_array < 0)):
        return skip("分割数组必须是非空三维非负整数数组")
    if not isinstance(label_map, Mapping) or not callable(support_extractor):
        return skip("缺少独立脊柱语义标签或前体支撑提取器")
    labels = {name: ids for name, value in label_map.items()
              if name in _BODY_INDEX and (ids := _ids(value))}
    flattened = [value for ids in labels.values() for value in ids]
    if len(flattened) != len(set(flattened)):
        return skip("脊柱语义标签 ID 重复，无法独立验证编号")
    if any(value > np.iinfo(seg_array.dtype).max for value in flattened):
        return skip("语义标签 ID 超出分割数据类型范围")
    requested = _requested_names(label_targets)
    if not requested or not requested.issubset(labels):
        return skip("请求范围缺少独立脊柱语义标签")
    first = max(0, min(_BODY_INDEX[name] for name in requested) - 1)
    last = min(len(_BODY_NAMES) - 1, max(_BODY_INDEX[name] for name in requested) + 1)
    selected_names = set(_BODY_NAMES[first:last+1]).intersection(labels)
    selected_ids = [value for name in selected_names for value in labels[name]]
    try:
        spacing, direction = _image_geometry(image)
        union = np.isin(seg_array, selected_ids)
        occupied_axes = [np.flatnonzero(np.any(union, axis=tuple(
            other for other in range(3) if other != axis))) for axis in range(3)]
        if not len(occupied_axes[0]):
            return skip("请求范围缺少前体前景")
        minimum = np.asarray([positions[0] for positions in occupied_axes])
        maximum = np.asarray([positions[-1] for positions in occupied_axes])
        region = tuple(slice(int(low), int(high)+1)
                       for low, high in zip(minimum, maximum))
        union = union[region].copy()
        support = np.asarray(support_extractor(union, spacing_zyx=tuple(spacing)))
        if support.shape != union.shape:
            return skip("前体支撑数组形状不一致")
        # An extractor can identify existing foreground, never add foreground.
        components, _ = connected_components(support.astype(bool) & union,
                                              structure=np.ones((3, 3, 3), dtype=bool))
        del support, union
        local_points = np.argwhere(components)
        if not len(local_points):
            return skip("缺少独立前体支撑")
        component_ids = components[tuple(local_points.T)]
        order = np.argsort(component_ids)
        splits = np.flatnonzero(np.diff(component_ids[order])) + 1
        id_names = {value: name for name in selected_names for value in labels[name]}
        records = []
        for members in np.split(order, splits):
            indices = local_points[members] + minimum
            physical_spans = (np.ptp(indices, axis=0) + 1) * spacing
            if np.any(physical_spans < (5.0, 6.0, 6.0)):
                continue
            ids, counts = np.unique(seg_array[tuple(indices.T)], return_counts=True)
            fractions: dict[str, float] = {}
            for value, count in zip(ids, counts):
                name = id_names[int(value)]
                fractions[name] = fractions.get(name, 0.0) + float(count) / len(indices)
            dominant = max(fractions, key=fractions.get)
            center = indices.mean(axis=0)
            physical_z = float((center[::-1] * spacing[::-1]) @ direction[2])
            records.append({"indices": indices, "center": center,
                            "physical_z": physical_z, "count": len(indices),
                            "height_mm": float(physical_spans[0]),
                            "fractions": fractions, "dominant": dominant,
                            "anchor": fractions[dominant] >= _ANCHOR_PURITY})
        del components, local_points, component_ids, order
        if not records:
            return skip("缺少足够大的独立前体")
        maximum_count = max(record["count"] for record in records)
        records = [record for record in records if record["count"] >= maximum_count * 0.12]
        records.sort(key=lambda record: record["physical_z"])
        mixed_indices = {index for index, record in enumerate(records) if sum(
            fraction >= _MINIMUM_MIXED_FRACTION
            for fraction in record["fractions"].values()) >= 2}
        if len(mixed_indices) < 2:
            return None, diagnostic
        anchors = [(index, _BODY_INDEX[record["dominant"]])
                   for index, record in enumerate(records) if record["anchor"]]
        if len(anchors) < 2 or len({label for _, label in anchors}) != len(anchors):
            return skip("缺少唯一的上下纯标签锚点")
        assignments = []
        affected_boundaries: set[int] = set()
        for (lower_index, lower_label), (upper_index, upper_label) in zip(anchors, anchors[1:]):
            if upper_label - lower_label != upper_index - lower_index:
                return skip("纯标签锚点之间存在缺失、合并或额外前体")
            interior = list(range(lower_index+1, upper_index))
            if not interior:
                continue
            if sum(index in mixed_indices for index in interior) < 2:
                continue
            for index in interior:
                expected_index = lower_label + index - lower_index
                expected = _BODY_NAMES[expected_index]
                fractions = records[index]["fractions"]
                if (expected not in labels or fractions.get(expected, 0.0) < _MINIMUM_MIXED_FRACTION
                        or any(abs(_BODY_INDEX[name] - expected_index) > 1
                               and fraction >= _MINIMUM_MIXED_FRACTION
                               for name, fraction in fractions.items())):
                    return skip("混合前体缺少自身标签支撑或含有远隔编号")
                assignments.append((index, expected))
                affected_boundaries.update((index-1, index))
        if len(assignments) < 2:
            return skip("连续混合前体没有被可靠纯标签锚点包围")
        centers = np.asarray([record["center"] for record in records])
        steps = np.linalg.norm(np.diff(centers, axis=0) * spacing, axis=1)
        reference_step = float(np.median(steps))
        if (reference_step <= 0 or any(not 0.55 <= steps[index]/reference_step <= 1.6
                                      for index in affected_boundaries)):
            return skip("混合前体之间的物理步长不可靠")
        raw = (raw_array_loader_or_array() if callable(raw_array_loader_or_array)
               else raw_array_loader_or_array)
        if (not isinstance(raw, np.ndarray) or raw.shape != seg_array.shape
                or not np.issubdtype(raw.dtype, np.number)):
            return skip("原图和分割数组形状或密度类型不一致")
        polarity = 1.0 if direction[2, 2] > 0 else -1.0
        evidence = []
        for index in sorted(affected_boundaries):
            boundary = _boundary_ct_evidence(raw, records[index], records[index+1],
                                             spacing, polarity)
            evidence.append({"lower_component": index, "upper_component": index+1,
                             **boundary})
        diagnostic["ct_evidence"] = evidence
        if not all(boundary["accepted"] for boundary in evidence):
            return skip("CT 未能确认全部受影响的独立椎间盘边界")
        result = seg_array.copy()
        changed = 0
        mappings = []
        for index, expected in assignments:
            record = records[index]
            allowed_ids = [value for name, ids in labels.items()
                           if abs(_BODY_INDEX[name] - _BODY_INDEX[expected]) <= 1
                           for value in ids]
            indices = record["indices"]
            original = seg_array[tuple(indices.T)]
            indices = indices[np.isin(original, allowed_ids)]
            target_id = labels[expected][0]
            component_changed = int(np.count_nonzero(seg_array[tuple(indices.T)] != target_id))
            result[tuple(indices.T)] = target_id
            changed += component_changed
            mappings.append({"target_name": expected, "body_voxel_count": record["count"],
                             "source_label_fractions": record["fractions"],
                             "changed_voxels": component_changed})
        if not changed:
            return None, diagnostic
        diagnostic.update(status="repaired", reason="独立前体与 CT 边界确认连续标签混合，已恢复主体编号",
                          changed_voxels=changed, body_mappings=mappings,
                          foreground_preserved=True, pure_anchor_labels_preserved=True,
                          posterior_labels_outside_body_support_preserved=True)
        return result, diagnostic
    except (ValueError, OSError, RuntimeError) as error:
        return skip(f"前体标签恢复缺少可靠数据: {error}")
