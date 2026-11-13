"""只为椎间隙核验建立局部几何视图，不改写全局主体或工作分割。"""

from copy import deepcopy
from time import perf_counter

import numpy as np


def _replacement_annotation(operations, image, indices, previous, diagnostic):
    """将候选主体转换为独立标注；端板核验通过前不交给规划器。"""

    geometry = operations._sagittal_body_quadrilateral(indices, image)
    if geometry is None:
        return None
    prism = operations._vertebral_body_prism_region(
        indices, image, sagittal_geometry=geometry,
    )
    if prism is None:
        return None
    center, axes, quad = geometry
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    physical = (np.asarray(image.GetOrigin())
                + (indices[:, ::-1] * np.asarray(image.GetSpacing())) @ direction.T)
    lr = (physical - center) @ direction[:, 0]
    centroid = indices.mean(axis=0)
    prism_minimum, prism_mask = prism
    prism_indices = np.argwhere(prism_mask) + prism_minimum
    prism_centroid = prism_indices.mean(axis=0)

    def xyz(point):
        return dict(zip("xyz", map(float, point)))

    annotation = deepcopy(previous)
    # 旧主体的端面/局部轴不能跟随替换主体进入新间隙。
    for key in list(annotation):
        if (key.startswith("local_superior_inferior_")
                or key in {"sagittal_body_endplates", "sagittal_endplate_support",
                           "body_extraction"}):
            annotation.pop(key)
    annotation.update(
        found=True,
        voxel_count=int(len(indices)),
        body_bbox=operations._bbox_from_indices(indices),
        centroid_voxel=xyz(centroid[::-1]),
        centroid_physical=xyz(image.TransformContinuousIndexToPhysicalPoint(
            tuple(map(float, centroid[::-1])),
        )),
        sagittal_body_quadrilateral={
            "center": xyz(center),
            "vertices": [xyz(point) for point in center + quad @ axes.T],
            "left_right_extent_mm": {"min": float(lr.min()), "max": float(lr.max())},
        },
        sagittal_body_faces=operations._estimate_sagittal_body_faces(
            indices, image, sagittal_geometry=geometry,
        ),
        prism_found=True,
        prism_voxel_count=int(len(prism_indices)),
        prism_bbox=operations._bbox_from_indices(prism_indices),
        prism_centroid_voxel=xyz(prism_centroid[::-1]),
        prism_centroid_physical=xyz(image.TransformContinuousIndexToPhysicalPoint(
            tuple(map(float, prism_centroid[::-1])),
        )),
        interspace_body_recheck=diagnostic,
    )
    return annotation


def validate_spine_interspace_geometry(
    operations, image, seg_array, label_targets, label_regions,
    body_indices, annotations, pair_definitions, *, raw_array_loader=None,
):
    """共用已定位的体素索引；只发布经端板验证接受的每组替换。"""

    from .interspace_body import validate_interspace_body
    from .interspace_endplates import validate_interspace_endplates

    started = perf_counter()
    requested_labels = {name for _, upper, lower in pair_definitions
                        for name in (upper, lower)}
    source_indices = {}
    candidate_indices = dict(body_indices)
    candidate_annotations = dict(annotations)
    body_diagnostics = {}
    replaced_labels = set()
    for name in sorted(requested_labels):
        target = label_targets.get(name)
        previous = annotations.get(name)
        indices = body_indices.get(name)
        if target is None:
            continue
        region = (label_regions or {}).get(name)
        source = (region.indices_zyx if region is not None
                  else np.argwhere(np.isin(seg_array, target.label_ids)))
        source_indices[name] = source
        if (not len(source) or indices is None or not len(indices)
                or not isinstance(previous, dict) or not previous.get("found")):
            continue
        minimum, maximum = source.min(axis=0), source.max(axis=0)
        shape = tuple((maximum - minimum + 1).astype(int))
        label_mask = np.zeros(shape, dtype=bool)
        label_mask[tuple((source - minimum).T)] = True
        body_mask = np.zeros(shape, dtype=bool)
        body_mask[tuple((indices - minimum).T)] = True
        diagnostic = {}
        replacement = validate_interspace_body(
            label_mask, body_mask, tuple(reversed(image.GetSpacing())),
            vertebra_name=name, diagnostics=diagnostic,
        )
        if diagnostic:
            body_diagnostics[name] = diagnostic
        if replacement is None:
            continue
        new_indices = np.argwhere(replacement) + minimum
        annotation = _replacement_annotation(
            operations, image, new_indices, previous, diagnostic,
        )
        if annotation is not None:
            candidate_indices[name] = new_indices
            candidate_annotations[name] = annotation
            replaced_labels.add(name)

    validated, diagnostics = validate_interspace_endplates(
        image, candidate_indices, source_indices, candidate_annotations,
        pair_definitions, raw_array_loader=raw_array_loader,
    )
    pairs = {}
    for group, upper, lower in pair_definitions:
        if group not in validated or not replaced_labels.intersection((upper, lower)):
            continue
        pair = operations._nearest_body_interspace_geometry(
            image, candidate_indices[upper], candidate_indices[lower],
        )
        if pair is not None:
            pairs[group] = {"upper_label": upper, "lower_label": lower, **pair}
    return validated, pairs, {
        "bodies": body_diagnostics,
        "endplates": diagnostics,
        "elapsed_seconds": perf_counter() - started,
        "body_candidate_count": len(replaced_labels),
        "validated_group_count": len(validated),
    }
