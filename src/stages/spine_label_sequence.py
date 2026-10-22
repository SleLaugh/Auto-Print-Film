"""Conservatively recover an empty lumbar anterior-body label slot.

The segmentation remains the source of foreground. CT is used only to verify that
two reliable neighboring bodies surround one disc, rather than a missing body.
Only existing vertebral foreground near the recovered anterior bodies is relabeled;
posterior elements retain the model's labels.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d, label as connected_components
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from src.models import LabelTarget


_BODY_NAMES = (
    "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3",
    "vertebrae_L2", "vertebrae_L1",
) + tuple(f"vertebrae_T{number}" for number in range(12, 0, -1))


def _label_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, LabelTarget):
        value = value.label_ids
    if isinstance(value, Mapping):
        value = value.get("label_ids", value.get("ids", ()))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple, set, np.ndarray)):
        return ()
    return tuple(sorted({int(item) for item in value
                         if isinstance(item, (int, np.integer))
                         and not isinstance(item, bool) and int(item) > 0}))


def _semantic_labels(label_map: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    result = {}
    for name, value in label_map.items():
        match = re.fullmatch(r"vertebrae_([LTS])(\d+)", str(name), re.IGNORECASE)
        if match:
            canonical = f"vertebrae_{match[1].upper()}{int(match[2])}"
            ids = _label_ids(value)
            if ids and canonical in _BODY_NAMES:
                result[canonical] = ids
    # Shared numeric labels do not provide independent anatomical evidence.
    flattened = [item for ids in result.values() for item in ids]
    if len(flattened) != len(set(flattened)):
        return {}
    return result


def _spacing_zyx(image: Any) -> np.ndarray:
    values = (image.GetSpacing() if hasattr(image, "GetSpacing")
              else image.get("spacing_xyz", image.get("spacing")))
    spacing = np.asarray(values, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("无效的脊柱序列图像间距")
    return spacing[::-1]


def _counts(array: np.ndarray) -> dict[int, int]:
    maximum = int(np.max(array))
    if maximum > 65535:
        # Other segmentation tasks may share this volume with sparse large IDs;
        # do not allocate an array indexed by an unrelated label's numeric value.
        sparse: dict[int, int] = {}
        for section in array:
            ids, frequencies = np.unique(section, return_counts=True)
            for value, count in zip(ids, frequencies):
                sparse[int(value)] = sparse.get(int(value), 0) + int(count)
        return sparse
    counts = np.zeros(maximum + 1, dtype=np.int64)
    for section in array:
        values = np.bincount(section.ravel().astype(np.int64, copy=False), minlength=maximum + 1)
        counts += values
    return {int(value): int(counts[value]) for value in np.flatnonzero(counts)}


def _extract_record(array, name, ids, extractor, spacing):
    points = np.argwhere(np.isin(array, ids))
    if not len(points):
        return None
    minimum, maximum = points.min(axis=0), points.max(axis=0) + 1
    region = tuple(slice(int(low), int(high)) for low, high in zip(minimum, maximum))
    mask = extractor(array[region], LabelTarget(label_name=name, label_ids=list(ids)))
    indices = (np.argwhere(mask) + minimum if mask is not None
               else np.empty((0, 3), dtype=np.int64))
    if not len(indices):
        return None
    low, high = indices.min(axis=0), indices.max(axis=0)
    return {
        "name": name, "ids": ids, "indices": indices,
        "raw_indices": points,
        "center": indices.mean(axis=0), "minimum": low, "maximum": high,
        "count": len(indices), "height_mm": float((high[0] - low[0] + 1) * spacing[0]),
    }


def _anterior_slot_support(array, ids, lower, upper, spacing):
    neighbors = np.concatenate((lower["indices"], upper["indices"]))
    # Use a central 24 mm LR strip and the neighbors' anterior-body AP range.
    center_x = float(np.median(neighbors[:, 2]))
    half_x = max(2, int(round(12.0 / spacing[2])))
    x0, x1 = max(0, int(center_x) - half_x), min(array.shape[2], int(center_x) + half_x + 1)
    y0, y1 = int(neighbors[:, 1].min()), int(neighbors[:, 1].max()) + 1
    counts = np.count_nonzero(np.isin(array[:, y0:y1, x0:x1], ids), axis=2)
    peak = int(counts.max()) if counts.size else 0
    supported = counts >= max(3, int(np.ceil(peak * 0.35)))
    layers = np.flatnonzero(supported.any(axis=1)) if peak else np.array([], dtype=int)
    height = float((layers[-1] - layers[0] + 1) * spacing[0]) if len(layers) else 0.0
    projection = np.argwhere(supported)
    projection[:, 1] += y0
    attachment_fraction = 0.0
    overlap_fraction = 0.0
    if len(projection):
        neighbor_projection = np.unique(neighbors[:, :2], axis=0)
        distances, _ = cKDTree(neighbor_projection * spacing[:2]).query(
            projection * spacing[:2], k=1, workers=1,
        )
        # A mislabeled cap must lie on the neighbor itself or its immediately
        # adjacent slice. A detached thin body between two discs is independent
        # anatomical evidence, even when the two CT valleys are close together.
        tolerance = max(float(spacing[0])*1.05, 1.0)
        attachment_fraction = float(np.mean(distances <= tolerance))
        overlap_fraction = float(np.mean(distances <= min(float(spacing[1])*.5, .5)))
    return {"supported_height_mm": height, "supported_z_indices": layers.tolist(),
            "neighbor_attachment_fraction": attachment_fraction,
            "neighbor_projection_overlap_fraction": overlap_fraction,
            "corridor_yx": [[y0, y1], [x0, x1]]}


def _disc_evidence(raw, lower, upper, spacing, reference_height):
    first, last = sorted((lower["center"], upper["center"]), key=lambda p: p[0])
    z_values = np.arange(int(np.floor(first[0])), int(np.ceil(last[0])) + 1)
    if len(z_values) < 5:
        return {"valley_count": 0, "reason": "相邻主体层数不足"}
    y_values = np.interp(z_values, [first[0], last[0]], [first[1], last[1]])
    x_values = np.interp(z_values, [first[0], last[0]], [first[2], last[2]])
    half_y, half_x = max(1, int(round(3.5 / spacing[1]))), max(1, int(round(8.0 / spacing[2])))
    medians = []
    for z, y, x in zip(z_values, y_values, x_values):
        y, x = int(round(y)), int(round(x))
        region = raw[z, max(0, y-half_y):min(raw.shape[1], y+half_y+1),
                     max(0, x-half_x):min(raw.shape[2], x+half_x+1)]
        finite = region[np.isfinite(region)]
        if not len(finite):
            return {"valley_count": 0, "reason": "CT 中央走廊缺少有效密度"}
        medians.append(float(np.median(finite)))
    smoothed = gaussian_filter1d(np.asarray(medians), 0.65)
    prominence = max(25.0, float(np.ptp(smoothed)) * 0.25)
    # Do not merge nearby disc minima: a compressed, genuinely thin vertebra can
    # still have two independent discs. The attachment gate above adds a separate
    # geometric requirement rather than relying on the peak count alone.
    peaks, _ = find_peaks(-smoothed, prominence=prominence, distance=2)
    span = last[0] - first[0]
    peaks = np.asarray([index for index in peaks
                        if first[0] + 0.15*span < z_values[index] < last[0] - 0.15*span], dtype=int)
    return {"valley_count": len(peaks), "valley_z_indices": z_values[peaks].tolist(),
            "valley_median_hu": [medians[index] for index in peaks],
            "minimum_prominence_hu": prominence}


def _reassociate_anterior_foreground(array, records, spacing, labels, eligible_ids, *, diagnostics=None):
    clouds, owners = [], []
    record_trees = {}
    source_owners = {}
    result = array.copy()
    protected_core = np.zeros(array.shape, dtype=bool)
    changed = 0
    for index, record in enumerate(records):
        indices = record["indices"]
        cloud = indices * spacing
        clouds.append(cloud)
        owners.append(np.full(len(indices), index, dtype=np.int32))
        protected_core[tuple(indices.T)] = True
        mutable_ids = [value for value in record["ids"] if value in eligible_ids]
        if not mutable_ids:
            continue
        # The extraction evidence already identifies these voxels' physical body.
        # Preserve that owner even if a tilted neighbor overlaps in sagittal view.
        target_id = int(labels[record["canonical_name"]][0])
        original = array[tuple(indices.T)]
        selected = np.isin(original, mutable_ids)
        selected_indices = indices[selected]
        changed += int(np.count_nonzero(original[selected] != target_id))
        result[tuple(selected_indices.T)] = target_id
        record_trees[index] = cKDTree(cloud, balanced_tree=True, compact_nodes=True)
        source_owners.update({value: index for value in mutable_ids})
    tree = cKDTree(np.concatenate(clouds), balanced_tree=True, compact_nodes=True)
    cloud_owner = np.concatenate(owners)
    # A small physical margin includes mislabeled endplate caps, while excluding
    # remote posterior fragments. Distance includes LR: sagittal projection alone
    # cannot distinguish neighboring bodies tilted through the same YZ location.
    margin_mm = 2.0

    def within_body_corridor(points, record, voxel_padding=0):
        height = float((record["maximum"][0] - record["minimum"][0] + 1) * spacing[0])
        extent = np.array((max(margin_mm, height*.35), margin_mm, margin_mm)) / spacing
        extent += voxel_padding
        return np.all((points >= record["minimum"] - extent)
                      & (points <= record["maximum"] + extent), axis=1)

    # A verified false-slot cap can contain an outer corner farther than 2 mm
    # from the filtered body. Preserve the cap's connected 3D ownership instead
    # of splitting a confirmed neighbor endplate into two anatomical labels.
    # This propagation applies only to the unowned slot, within an anterior-body
    # corridor, with actual 3D near-body seeds and one unambiguous seed owner.
    unowned_ids = [value for value in eligible_ids if value not in source_owners]
    cap_points = np.argwhere(np.isin(array, unowned_ids))
    component_diagnostics = []
    if len(cap_points):
        cap_clouds, cap_owners = [], []
        for owner, record in enumerate(records):
            # Filtered core bounds can cut a genuine endplate corner. For cap
            # contact only, retain the neighbor's original label voxels inside
            # that body's trusted anterior corridor. Never use remote processes.
            reference = record.get("raw_indices", record["indices"])
            reference = reference[within_body_corridor(reference, record)]
            cap_clouds.append(reference * spacing)
            cap_owners.append(np.full(len(reference), owner, dtype=np.int32))
        cap_tree = cKDTree(np.concatenate(cap_clouds), balanced_tree=True, compact_nodes=True)
        cap_owner = np.concatenate(cap_owners)
        minimum = cap_points.min(axis=0)
        local_points = cap_points - minimum
        local = np.zeros(tuple(local_points.max(axis=0) + 1), dtype=bool)
        local[tuple(local_points.T)] = True
        components, _ = connected_components(local, structure=np.ones((3, 3, 3), dtype=bool))
        component_ids = components[tuple(local_points.T)]
        order = np.argsort(component_ids)
        splits = np.flatnonzero(np.diff(component_ids[order])) + 1
        for members in np.split(order, splits):
            points = cap_points[members]
            # The entire connected component is either supported or preserved;
            # do not later relabel isolated pieces of an unsupported component.
            protected_core[tuple(points.T)] = True
            distances, nearest = cap_tree.query(points * spacing, k=1, workers=1)
            close = distances <= margin_mm
            component_info = {
                "voxel_count": len(points), "bbox_zyx": {
                    "min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
                "seed_voxel_count": int(np.count_nonzero(close)),
                "status": "preserved", "reason": "insufficient_3d_seeds",
            }
            component_diagnostics.append(component_info)
            if np.count_nonzero(close) < max(1, int(np.ceil(len(points)*.1))):
                continue
            owner_counts = np.bincount(cap_owner[nearest[close]], minlength=len(records))
            owner = int(np.argmax(owner_counts))
            agreement = float(owner_counts[owner] / np.count_nonzero(close))
            in_corridor = within_body_corridor(points, records[owner])
            component_info.update(seed_owner_fraction=agreement,
                                  candidate_target_name=records[owner]["canonical_name"],
                                  corridor_fraction=float(np.mean(in_corridor)))
            if (owner_counts[owner] < np.count_nonzero(close)*.9
                    or not np.any(
                        np.all(np.abs(points - cap_tree.data[nearest] / spacing) <= 1.000001, axis=1)
                        & close & (cap_owner[nearest] == owner)
                    )):
                component_info["reason"] = "ambiguous_3d_attachment"
                continue
            # A connected cap may exceed a sampled corridor by one boundary voxel;
            # require almost all of it inside, and every point within that single
            # voxel rounding margin. This does not enlarge the 2 mm seed distance.
            if (np.mean(in_corridor) < .95
                    or not np.all(within_body_corridor(points, records[owner], voxel_padding=1))):
                component_info["reason"] = "outside_anterior_body_corridor"
                continue
            target_id = int(labels[records[owner]["canonical_name"]][0])
            old = array[tuple(points.T)]
            component_changed = int(np.count_nonzero(old != target_id))
            changed += component_changed
            result[tuple(points.T)] = target_id
            component_info.update(status="reassociated", reason="connected_3d_cap",
                                  target_name=records[owner]["canonical_name"],
                                  changed_voxels=component_changed)
    if diagnostics is not None:
        diagnostics["cap_components"] = component_diagnostics

    for z in range(array.shape[0]):
        yx = np.argwhere(np.isin(array[z], eligible_ids) & ~protected_core[z])
        if not len(yx):
            continue
        points = np.column_stack((np.full(len(yx), z), yx)) * spacing
        source_ids = array[z, yx[:, 0], yx[:, 1]]
        distances = np.full(len(yx), np.inf)
        assigned = np.full(len(yx), -1, dtype=np.int32)
        unowned = np.ones(len(yx), dtype=bool)
        for source_id, index in source_owners.items():
            source = source_ids == source_id
            if not np.any(source):
                continue
            # Edges of an independently identified label may extend that same
            # body, but cannot steal another body's canonical number.
            distances[source], _ = record_trees[index].query(points[source], k=1, workers=1)
            assigned[source] = index
            unowned[source] = False
        if np.any(unowned):
            distances[unowned], nearest = tree.query(points[unowned], k=1, workers=1)
            assigned[unowned] = cloud_owner[nearest]
        for index, record in enumerate(records):
            selected = (assigned == index) & (distances <= margin_mm)
            selected &= yx[:, 0] <= record["maximum"][1] + margin_mm / spacing[1]
            selected &= yx[:, 0] >= record["minimum"][1] - margin_mm / spacing[1]
            selected &= yx[:, 1] >= record["minimum"][2] - margin_mm / spacing[2]
            selected &= yx[:, 1] <= record["maximum"][2] + margin_mm / spacing[2]
            selected_yx = yx[selected]
            if not len(selected_yx):
                continue
            target_id = int(labels[record["canonical_name"]][0])
            old = result[z, selected_yx[:, 0], selected_yx[:, 1]]
            changed += int(np.count_nonzero(old != target_id))
            result[z, selected_yx[:, 0], selected_yx[:, 1]] = target_id
    return result, changed


def repair_lumbar_body_sequence(
    seg_array: np.ndarray,
    raw_array: np.ndarray | Callable[[], np.ndarray],
    image: Any,
    label_map: Mapping[str, Any],
    label_targets: Any = None,
    body_extractor: Callable[[np.ndarray, LabelTarget], np.ndarray | None] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Recover one false anterior-body slot above reliable S1/L5/L4 anchors.

    ``raw_array`` may be a lazy loader. It is called only after label-only gates
    identify a genuinely small, unsupported slot. ``None`` means no mutation.
    Labels are resolved semantically; numeric model IDs are never assumed.
    """
    diagnostic: dict[str, Any] = {"status": "unchanged", "reason": "未发现腰椎前体空槽"}
    if (not isinstance(seg_array, np.ndarray) or seg_array.ndim != 3 or seg_array.size == 0
            or not np.issubdtype(seg_array.dtype, np.integer)):
        return None, {"status": "skipped", "reason": "分割数组必须是三维整数数组"}
    if np.any(seg_array < 0):
        return None, {"status": "skipped", "reason": "分割标签不能为负数"}
    labels = _semantic_labels(label_map)
    if not all(name in labels for name in _BODY_NAMES[:6]) or body_extractor is None:
        return None, {"status": "skipped", "reason": "缺少独立腰椎语义标签或主体提取器"}
    if any(value > np.iinfo(seg_array.dtype).max for ids in labels.values() for value in ids):
        return None, {"status": "skipped", "reason": "语义标签 ID 超出分割数据类型范围"}
    counts = _counts(seg_array)
    totals = {name: sum(counts.get(value, 0) for value in ids)
              for name, ids in labels.items()}
    anchor_volume = float(np.median([totals[_BODY_NAMES[index]] for index in (1, 2)]))
    if anchor_volume <= 0 or not totals[_BODY_NAMES[0]]:
        return None, {"status": "skipped", "reason": "缺少 S1/L5/L4 序列锚点"}
    suspicious = [index for index in (3, 4)
                  if totals[_BODY_NAMES[index]] < anchor_volume * 0.35]
    if not suspicious:
        return None, diagnostic
    try:
        spacing = _spacing_zyx(image)
        records = {name: _extract_record(seg_array, name, ids, body_extractor, spacing)
                   for name, ids in labels.items() if totals[name]}
        anchors = [records.get(name) for name in _BODY_NAMES[:3]]
        if any(record is None for record in anchors):
            return None, {"status": "skipped", "reason": "S1/L5/L4 缺少可靠前体锚点"}
        s1, l5, l4 = anchors
        reference_height = float(np.median([l5["height_mm"], l4["height_mm"]]))
        reference_volume = float(np.median([l5["count"], l4["count"]]))
        polarity = 1.0 if l5["center"][0] > s1["center"][0] else -1.0
        if (l4["center"][0] - l5["center"][0])*polarity <= 0:
            return None, {"status": "skipped", "reason": "S1/L5/L4 前体顺序不可靠"}
        reference_steps = [float(np.linalg.norm((l4["center"]-l5["center"])*spacing))]
        for first_name, second_name in zip(_BODY_NAMES[2:], _BODY_NAMES[3:]):
            first, second = records.get(first_name), records.get(second_name)
            if (first and second and min(first["height_mm"], second["height_mm"]) >= reference_height*.6
                    and min(first["count"], second["count"]) >= reference_volume*.35):
                step = float(np.linalg.norm((second["center"]-first["center"])*spacing))
                if (second["center"][0]-first["center"][0])*polarity > 0:
                    reference_steps.append(step)
        reference_step = float(np.median(reference_steps))
        candidates = []
        for index in suspicious:
            name = _BODY_NAMES[index]
            lower, upper = records.get(_BODY_NAMES[index-1]), records.get(_BODY_NAMES[index+1])
            if lower is None or upper is None:
                continue
            support = _anterior_slot_support(seg_array, labels[name], lower, upper, spacing)
            distance = float(np.linalg.norm((upper["center"]-lower["center"])*spacing))
            ratio = distance/reference_step if reference_step > 0 else float("inf")
            if (support["supported_height_mm"] >= reference_height*.35 or not .7 <= ratio <= 1.35
                    or support["neighbor_attachment_fraction"] < .65
                    or support["neighbor_projection_overlap_fraction"] < .1
                    or (upper["center"][0]-lower["center"][0])*polarity <= 0):
                continue
            # Inferior canonical bodies must all be present and substantial.
            if any(records.get(anchor) is None or records[anchor]["count"] < reference_volume*.35
                   for anchor in _BODY_NAMES[1:index]):
                continue
            candidates.append((index, lower, upper, support, ratio))
        if len(candidates) != 1:
            return None, {"status": "skipped", "reason": "前体空槽证据不足或存在多个歧义槽位"}
        index, lower, upper, support, ratio = candidates[0]
        raw = raw_array() if callable(raw_array) else raw_array
        if not isinstance(raw, np.ndarray) or raw.shape != seg_array.shape:
            raise ValueError("原图和分割数组形状不一致")
        evidence = _disc_evidence(raw, lower, upper, spacing, reference_height)
        diagnostic.update(slot_name=_BODY_NAMES[index], slot_support=support,
                          neighbor_step_ratio=ratio, ct_evidence=evidence)
        if evidence["valley_count"] != 1:
            diagnostic.update(status="skipped", reason="相邻前体之间无法确认唯一 CT 椎间盘谷")
            return None, diagnostic
        ordered = []
        required_names = set(_BODY_NAMES[:6])
        if isinstance(label_targets, Mapping):
            required_names.update(name for name in label_targets if name in _BODY_NAMES)
        highest_required = max(_BODY_NAMES.index(name) for name in required_names)
        for original_index, name in enumerate(_BODY_NAMES):
            if original_index == index:
                continue
            record = records.get(name)
            if record is None:
                # Do not jump across another missing label or synthesize a body.
                break
            if original_index > index and (record["height_mm"] < reference_height*.35
                                          or record["count"] < reference_volume*.2):
                break
            canonical_index = original_index - (1 if original_index > index else 0)
            if canonical_index > highest_required:
                break
            copy = dict(record, canonical_name=_BODY_NAMES[canonical_index])
            if ordered and (copy["center"][0]-ordered[-1]["center"][0])*polarity <= 0:
                diagnostic.update(status="skipped", reason="前体序列存在错序，停止自动恢复")
                return None, diagnostic
            ordered.append(copy)
        if not required_names.issubset({record["canonical_name"] for record in ordered}):
            diagnostic.update(status="skipped", reason="上方独立前体不足，无法恢复请求范围")
            return None, diagnostic
        # Inferior anchors are references, never relabeling candidates. The false
        # slot's cap may be returned to its lower anchor, while the anchor's own
        # foreground remains byte-for-byte unchanged. Stop at requested anatomy.
        eligible_ids = list(labels[_BODY_NAMES[index]])
        for record in ordered:
            if _BODY_NAMES.index(record["name"]) > index:
                eligible_ids.extend(record["ids"])
        reassociation_diagnostic = {}
        result, changed = _reassociate_anterior_foreground(
            seg_array, ordered, spacing, labels, eligible_ids, diagnostics=reassociation_diagnostic,
        )
        diagnostic["cap_components"] = reassociation_diagnostic["cap_components"]
        neighboring_targets = {_BODY_NAMES[index-1], _BODY_NAMES[index]}
        if not any(component.get("status") == "reassociated"
                   and component.get("target_name") in neighboring_targets
                   for component in diagnostic["cap_components"]):
            diagnostic.update(status="skipped", reason="缺少与相邻前体真实三维相连的可信薄帽")
            return None, diagnostic
        if not changed:
            return None, diagnostic
        if not np.array_equal(result != 0, seg_array != 0):
            raise ValueError("前体重关联改变了分割前景")
        diagnostic.update(status="repaired", reason="CT 确认前体空槽，已按骶腰锚点恢复前体编号",
                          changed_voxels=changed, foreground_preserved=True,
                          body_mappings=[{"source_name": record["name"], "target_name": record["canonical_name"],
                                          "body_voxel_count": record["count"]} for record in ordered],
                          cap_components=reassociation_diagnostic["cap_components"],
                          posterior_labels_outside_body_corridor_preserved=True,
                          inferior_anchor_labels_preserved=True)
        return result, diagnostic
    except (ValueError, OSError) as error:
        diagnostic.update(status="skipped", reason=f"前体序列恢复缺少可靠数据: {error}")
        return None, diagnostic
