"""椎间隙专用端板核验；正常四棱柱路径保持原样。

包络侧边的中心连线评分不能证明它是端板。仅当选边与自然边界冲突，
或主体复核提出替换时，才在原标签的局部 LR 切面拟合相向端板并核对 CT。
S1 只拟合面向 L5 的上端板，不要求其下端面具有普通椎体形状。
"""

from copy import deepcopy
from typing import Callable

import numpy as np

from .spine_body_core import fit_body_endplates_from_core, _boundary_density_drop


def _xyz(point):
    return dict(zip("xyz", map(float, point)))


def _quad_edges(annotation, image, toward, sign):
    quad = annotation.get("sagittal_body_quadrilateral")
    if not isinstance(quad, dict):
        return None
    try:
        vertices = np.array([[p[k] for k in "xyz"] for p in quad["vertices"]], float)
    except (KeyError, TypeError, ValueError):
        return None
    if vertices.shape != (4, 3) or not np.all(np.isfinite(vertices)):
        return None
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    row, si = direction[:, 0], direction[:, 2]
    center = vertices.mean(axis=0)
    toward = np.asarray(toward, float)
    toward /= max(np.linalg.norm(toward), 1e-9)
    edges = []
    for i, first in enumerate(vertices):
        last = vertices[(i + 1) % 4]
        normal = np.cross(row, last - first)
        length = np.linalg.norm(normal)
        if length <= 1e-6:
            continue
        normal /= length
        if np.dot(normal, (first + last) * 0.5 - center) < 0:
            normal = -normal
        edges.append({"endpoints": np.array([first, last]), "normal": normal,
                      "alignment": float(np.dot(normal, toward)),
                      "side_alignment": float(sign * np.dot(normal, si))})
    if not edges:
        return None
    chosen = max(edges, key=lambda edge: edge["alignment"])
    # 这是深入复核的触发条件，不把真实端板限制在固定角度。
    anatomical = [edge for edge in edges
                  if edge["side_alignment"] > 0.2 and edge["alignment"] > 0]
    desired = max(anatomical, key=lambda edge: edge["alignment"]) if anatomical else chosen
    return chosen, desired


def _rows(source, x, y_min, y_max, body_min_z, body_max_z):
    section = source[(source[:, 2] == x) & (source[:, 1] >= y_min)
                     & (source[:, 1] <= y_max)]
    rows = []
    for y in np.unique(section[:, 1]):
        z = np.sort(section[section[:, 1] == y, 0])
        groups = np.split(z, np.flatnonzero(np.diff(z) > 2) + 1)
        selected = max(groups, key=lambda g: (
            max(0, min(g[-1], body_max_z) - max(g[0], body_min_z) + 1), len(g),
        ))
        if len(selected) >= 2:
            rows.append((y, selected[0], selected[-1]))
    return np.asarray(rows, float).reshape(-1, 3)


def _quick_boundary_normal(body, source, spacing, direction, side):
    """最多九个 LR 截面，只作冲突检测；不能作为替换端板的证据。"""

    minimum, maximum = np.quantile(body, (0.25, 0.75), axis=0)
    sample_x = np.unique(np.rint(np.linspace(minimum[2], maximum[2], 9)).astype(int))
    slopes = []
    column = 1 if side == "min" else 2
    for x in sample_x:
        values = _rows(source, x, minimum[1], maximum[1], minimum[0], maximum[0])
        if len(values) < 5:
            continue
        y, z = values[:, 0] * spacing[1], values[:, column] * spacing[0]
        slope, intercept = np.polyfit(y, z, 1)
        residuals = np.abs(z - (slope * y + intercept))
        if np.median(residuals) <= spacing[0] * 0.5 and np.quantile(residuals, 0.9) <= spacing[0]:
            slopes.append(slope)
    if len(slopes) < 3:
        return None
    angles = np.degrees(np.arctan(slopes))
    consensus = np.abs(angles - np.median(angles)) <= 5
    if np.count_nonzero(consensus) < max(3, len(sample_x) * 0.5):
        return None
    slope = float(np.median(np.asarray(slopes)[consensus]))
    sign = -1 if side == "min" else 1
    normal_zyx = np.array([sign, -sign * slope, 0.0])
    return direction @ (normal_zyx / np.linalg.norm(normal_zyx))[::-1]


def _ct_confirm_face(face, body, source, raw, spacing):
    """原边界允许一个采样层的皮质定位误差；密度下降门槛不放宽。"""

    side = face["side"]
    column = 1 if side == "min" else 2
    sign = -1 if side == "min" else 1
    endpoints = np.asarray(face["endpoints_zyx"], float)
    y_min, y_max = sorted(endpoints[:, 1])
    lr_min, lr_max = face["lr_profile_extent_voxels"]
    sample_x = np.unique(np.rint(np.linspace(lr_min, lr_max, 13)).astype(int))
    normal = np.asarray(face["normal_zyx"], float)
    slope = -normal[1] / normal[0]
    accepted = []
    available = 0
    # 原标签中心采样是体素中心，CT 皮质可落在中心以外一层范围内。
    max_offset = min(2.0, float(spacing[0]))
    for x in sample_x:
        rows = _rows(source, x, y_min, y_max, body[:, 0].min(), body[:, 0].max())
        if len(rows) < 5 or np.ptp(rows[:, 0]) < (y_max - y_min) * 0.8:
            continue
        values = rows[:, column] * spacing[0]
        prediction = slope * rows[:, 0] * spacing[1]
        residuals = np.abs((values - prediction) - np.median(values - prediction))
        if np.quantile(residuals, 0.9) > spacing[0] * 1.5:
            continue
        available += 1
        points = np.column_stack((rows[:, column], rows[:, 0], np.full(len(rows), x)))
        for offset in (0.0, max_offset * 0.5, max_offset):
            evidence = _boundary_density_drop(
                raw, points + normal * (offset / spacing), spacing, slope, sign,
            )
            if evidence is not None:
                accepted.append((x, offset, evidence))
                break
    if (len(accepted) < max(3, available * 0.6)
            or np.ptp([item[0] for item in accepted]) < (lr_max - lr_min) * 0.25):
        return None
    offset = float(np.median([item[1] for item in accepted]))
    return {**face,
            "endpoints_zyx": (endpoints + normal * (offset / spacing)).tolist(),
            "ct_validation": "density_drop",
            "ct_profile_count": len(accepted), "ct_profile_available": available,
            "ct_boundary_offset_mm": offset,
            "ct_min_positive_fraction": min(ev["positive_fraction"] for _, _, evidence in accepted for ev in evidence),
            "ct_min_normalized_drop": min(ev["normalized_drop"] for _, _, evidence in accepted for ev in evidence)}


def _fit_facing_face(image, body, source, edge, side, raw):
    spacing = np.asarray(image.GetSpacing()[::-1], float)
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    endpoints = np.array([image.TransformPhysicalPointToContinuousIndex(tuple(p))
                          for p in edge["endpoints"]])
    y_min, y_max = sorted(endpoints[:, 1])
    minimum, maximum = np.quantile(body, (0.25, 0.75), axis=0)
    # 端板前后缘可能弯曲；在同一条候选端板范围寻找有足够 LR 共识的直段。
    windows = ((0.05, 0.95), (0.25, 0.95), (0.5, 1.0), (0.0, 0.5), (0.15, 0.75))
    tried = set()
    for low, high in windows:
        lo, hi = minimum.copy(), maximum.copy()
        lo[1], hi[1] = y_min + low * (y_max - y_min), y_min + high * (y_max - y_min)
        box = {"min": np.ceil(lo).astype(int).tolist(), "max": np.floor(hi).astype(int).tolist()}
        key = (box["min"][1], box["max"][1])
        if key in tried or key[1] - key[0] < 4:
            continue
        tried.add(key)
        faces = fit_body_endplates_from_core(
            body, tuple(spacing), box, source_indices_zyx=source, required_sides=(side,),
        )
        if not faces:
            continue
        face = faces[0]
        normal = direction @ np.asarray(face["normal_zyx"])[::-1]
        if np.dot(normal, edge["normal"]) < np.cos(np.radians(30.0)):
            continue
        face = _ct_confirm_face(face, body, source, raw, spacing)
        if face is None:
            continue
        physical = [image.TransformContinuousIndexToPhysicalPoint(tuple(p[::-1]))
                    for p in face.pop("endpoints_zyx")]
        face.pop("normal_zyx")
        return {**face, "endpoints": [_xyz(p) for p in physical], "normal": _xyz(normal)}
    return None


def _positive_facing_gap(image, faces):
    """独立端板的共同中段须形成正间隙，交叉端板不能进入规划器。"""

    normals = [np.array([face["normal"][k] for k in "xyz"]) for face in faces]
    normal = normals[0] - normals[1]
    length = np.linalg.norm(normal)
    if length <= 1e-6:
        return False
    normal /= length
    row = np.asarray(image.GetDirection()).reshape(3, 3)[:, 0]
    tangent = np.cross(normal, row)
    endpoints = [np.array([[p[k] for k in "xyz"] for p in face["endpoints"]]) for face in faces]
    projections = [points @ tangent for points in endpoints]
    lower = max(float(p.min()) for p in projections)
    upper = min(float(p.max()) for p in projections)
    if upper - lower <= 1e-6:
        return False
    for fraction in (0.25, 0.5, 0.75):
        position = lower + fraction * (upper - lower)
        points = []
        for pair, values in zip(endpoints, projections):
            span = values[1] - values[0]
            if abs(span) <= 1e-6:
                return False
            points.append(pair[0] + (position - values[0]) / span * (pair[1] - pair[0]))
        if np.dot(points[1] - points[0], normal) <= 1e-6:
            return False
    return True


def validate_interspace_endplates(
    image, body_indices, source_indices_by_label, annotations, pair_definitions,
    *, raw_array_loader: Callable[[], np.ndarray] | None = None,
):
    """返回 {组名: {标签: 验证标注}} 和诊断；不写入任何输入容器。"""

    replacements, diagnostics = {}, {}
    spacing = np.asarray(image.GetSpacing()[::-1], float)
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    raw = None
    load_attempted = False
    quick_cache, fit_cache = {}, {}
    for group, upper, lower in pair_definitions:
        labels = (upper, lower)
        if any(not isinstance(annotations.get(n), dict) or not annotations[n].get("prism_found")
               or body_indices.get(n) is None or len(body_indices[n]) < 8
               or source_indices_by_label.get(n) is None for n in labels):
            continue
        # C2 已有独立的下端板恢复流程，保留其专用判定。
        if any(n.rsplit("_", 1)[-1] in {"C1", "C2"} for n in labels):
            continue
        centers = [np.mean(body_indices[n], axis=0) for n in labels]
        if abs(centers[1][0] - centers[0][0]) <= 1e-6:
            continue
        signs = (int(np.sign(centers[1][0] - centers[0][0])), int(np.sign(centers[0][0] - centers[1][0])))
        sides = tuple("min" if sign < 0 else "max" for sign in signs)
        physical_centers = [np.mean([[p[k] for k in "xyz"]
                                    for p in annotations[n]["sagittal_body_quadrilateral"]["vertices"]], axis=0)
                            for n in labels]
        edges, reasons = [], []
        for i, n in enumerate(labels):
            pair = _quad_edges(annotations[n], image, physical_centers[1 - i] - physical_centers[i], signs[i])
            if pair is None:
                break
            chosen, desired = pair
            edges.append(desired)
            if annotations[n].get("interspace_body_recheck"):
                reasons.append(n + ":body_candidate_changed")
            if chosen["side_alignment"] <= 0.2 and desired["side_alignment"] > 0.2:
                reasons.append(n + ":sidewall_selected")
            key = (n, sides[i])
            if key not in quick_cache:
                quick_cache[key] = _quick_boundary_normal(body_indices[n], source_indices_by_label[n], spacing, direction, sides[i])
            normal = quick_cache[key]
            if normal is not None and np.dot(normal, chosen["normal"]) < np.cos(np.radians(25.0)):
                reasons.append(n + ":original_boundary_conflict")
        if len(edges) != 2 or not reasons:
            continue
        diagnostic = {"reasons": reasons, "status": "kept_original"}
        diagnostics[group] = diagnostic
        if not load_attempted:
            load_attempted = True
            if raw_array_loader is not None:
                try:
                    raw = raw_array_loader()
                except (OSError, RuntimeError, ValueError) as error:
                    diagnostic["raw_error"] = str(error)
        if raw is None:
            diagnostic["reason"] = "ct_unavailable"
            continue
        faces = []
        for n, side, edge in zip(labels, sides, edges):
            key = (n, side, tuple(np.round(edge["normal"], 6)))
            if key not in fit_cache:
                fit_cache[key] = _fit_facing_face(image, body_indices[n], source_indices_by_label[n], edge, side, raw)
            faces.append(fit_cache[key])
        if any(face is None for face in faces):
            diagnostic["reason"] = "natural_endplate_not_confirmed"
            diagnostic["confirmed_labels"] = [n for n, f in zip(labels, faces) if f is not None]
            continue
        # 两端面必须相对，不能将侧壁和另一侧端板拼成一个间隙。
        normals = [np.array([face["normal"][k] for k in "xyz"]) for face in faces]
        if np.dot(normals[0], -normals[1]) < np.cos(np.radians(35.0)):
            diagnostic["reason"] = "facing_endplates_disagree"
            continue
        if not _positive_facing_gap(image, faces):
            diagnostic["reason"] = "facing_endplates_cross_or_miss"
            continue
        result = {}
        for n, side, face in zip(labels, sides, faces):
            annotation = deepcopy(annotations[n])
            annotation["sagittal_body_endplates"] = [face]
            annotation["sagittal_endplate_support"] = {
                "method": "interspace_validated_endplate", "required_endplate_side": side,
                "group_name": group,
            }
            result[n] = annotation
        replacements[group] = result
        diagnostic.update(status="validated", confirmed_labels=list(labels))
    return replacements, diagnostics
