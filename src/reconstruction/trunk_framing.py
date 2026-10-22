"""保守地从身体取景支持中分离可识别的上臂，不代替器官分割。"""

import numpy as np
from scipy import ndimage
import SimpleITK as sitk


_BONE_THRESHOLD_HU = 150.0
_MIN_ARM_BONE_RADIUS_MM = 10.0
_MIN_ARM_BONE_AREA_MM2 = 250.0
_LUNG_PROTECTION_MM = 20.0
_CENTRAL_CORE_HALF_WIDTH_MM = 20.0


def _remove_separate_arms(
    body: np.ndarray,
    target: np.ndarray,
    bounds: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """仅排除与可靠中央躯干完全分开的外围组件，不按最大组件盲选。"""

    labels, count = ndimage.label(body, structure=np.ones((3, 3), dtype=bool))
    if count < 2:
        return body
    x_min, x_max, y_min, y_max = bounds
    center_x = 0.5 * (x_min + x_max)
    yy, xx = np.ogrid[:body.shape[0], :body.shape[1]]
    anchor = ((np.abs(xx - center_x) * spacing[1] <= _CENTRAL_CORE_HALF_WIDTH_MM)
              & (yy >= y_min) & (yy <= y_max) & body)
    areas = np.bincount(labels.ravel())
    target_ids = np.unique(labels[target & body])
    target_ids = target_ids[target_ids > 0]
    if len(target_ids) > 1:
        return body
    if len(target_ids) == 1:
        trunk_id = int(target_ids[0])
        if not np.any(anchor & (labels == trunk_id)):
            return body
    else:
        anchored_ids = np.unique(labels[anchor])
        anchored_ids = anchored_ids[anchored_ids > 0]
        if len(anchored_ids) != 1:
            return body
        trunk_id = int(anchored_ids[0])
    if areas[trunk_id] < 0.5 * np.count_nonzero(body):
        return body
    result = body.copy()
    for index, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None or index == trunk_id:
            continue
        left, right = region[1].start, region[1].stop - 1
        entirely_lateral = (right < center_x - _CENTRAL_CORE_HALF_WIDTH_MM / spacing[1]
                            or left > center_x + _CENTRAL_CORE_HALF_WIDTH_MM / spacing[1])
        reaches_outside_target = left < x_min or right > x_max
        if entirely_lateral and reaches_outside_target:
            result[region] &= labels[region] != index
    return result


def extract_trunk_slice(
    hu: np.ndarray,
    body_mask: np.ndarray,
    target_slice: np.ndarray,
    target_bounds_xy: tuple[float, float, float, float],
    spacing_yx: np.ndarray,
) -> np.ndarray:
    """返回用于肺部取景的躯干支持，证据不足时保留完整身体。

    ``target_bounds_xy`` 是整组肺在源 CT 中的固定索引边界，顺序为
    ``(x_min, x_max, y_min, y_max)``。边界只定位中央主干与外围候选骨，
    不用它剪裁身体；末层肺变小或消失也不会缩小躯干。

    与可靠中央躯干完全分离、且不含肺的外围组件可直接排除。连着躯干时，
    只把外围厚实、近圆的骨断面作为上臂候选，避免将肋骨和薄肩胛骨
    当成上臂。填孔身体距离场的标记分水岭在上臂与主干之间的连接处分离，
    肺周围 20 mm 及中央主干作为保护种子。没有可靠拓扑/骨芯证据或分离结果不可靠时
    原样返回；肩背肌仍属于躯干。这里不承诺识别所有姿势或已截断的手臂。
    """

    body = np.asarray(body_mask, dtype=bool)
    target = np.asarray(target_slice, dtype=bool)
    spacing = np.asarray(spacing_yx, dtype=float)
    if hu.ndim != 2 or body.shape != hu.shape or target.shape != hu.shape:
        raise ValueError("躯干取景需要形状一致的二维 CT、身体和目标掩膜")
    if spacing.shape != (2,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("躯干取景需要正的二维物理像素间距")
    bounds = np.asarray(target_bounds_xy, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("躯干取景需要有限的整组目标边界")
    x_min, x_max, y_min, y_max = bounds
    if x_min > x_max or y_min > y_max:
        raise ValueError("躯干取景目标边界顺序错误")
    if not np.any(body):
        return body

    body = _remove_separate_arms(body, target, bounds, spacing)
    center_x = 0.5 * (x_min + x_max)
    width_x = x_max - x_min + 1.0
    body_x = np.flatnonzero(np.any(body, axis=0))
    body_center_x = 0.5 * (body_x[0] + body_x[-1])
    body_width_x = body_x[-1] - body_x[0] + 1.0
    bone_labels, count = ndimage.label((hu > _BONE_THRESHOLD_HU) & body,
                                      structure=np.ones((3, 3), dtype=bool))
    if not count:
        return body
    protected = target & body
    if np.any(protected):
        protected = (ndimage.distance_transform_edt(~protected, sampling=spacing)
                     <= _LUNG_PROTECTION_MM) & body
    arm_markers = np.zeros_like(body)
    pixel_area = float(np.prod(spacing))
    for index, region in enumerate(ndimage.find_objects(bone_labels), start=1):
        if region is None:
            continue
        component = bone_labels[region] == index
        # 对局部骨断面填孔保留骨髓腔；不扩张、连接原本分开的骨组件。
        filled = ndimage.binary_fill_holes(component)
        if filled.sum() * pixel_area < _MIN_ARM_BONE_AREA_MM2:
            continue
        physical_spans = np.asarray(filled.shape) * spacing
        if (max(physical_spans) / min(physical_spans) > 1.8
                or float(filled.mean()) < 0.55):
            continue
        radius = ndimage.distance_transform_edt(np.pad(filled, 1), sampling=spacing).max()
        if radius < _MIN_ARM_BONE_RADIUS_MM:
            continue
        _, xx = np.nonzero(component)
        candidate_x = float(xx.mean()) + region[1].start
        # 只在整组肺投影的外围寻找上臂；这是候选门槛，不是身体裁切线。
        if abs(candidate_x - center_x) < 0.4 * width_x:
            continue
        # 单侧肺或不对称肺不能把肺外的脊柱误当上臂。骨芯还须位于
        # 实际身体的外侧区域；身体中央的厚实骨始终作为主干保护。
        if abs(candidate_x - body_center_x) < 0.3 * body_width_x:
            continue
        if np.any(filled & protected[region]):
            continue
        arm_markers[region] |= filled
    if not np.any(arm_markers):
        return body

    distance = ndimage.distance_transform_edt(body, sampling=spacing)
    xx = np.arange(body.shape[1], dtype=float)[None, :]
    core = (np.abs(xx - center_x) * spacing[1] <= _CENTRAL_CORE_HALF_WIDTH_MM)
    core = core & body & (distance >= 8.0)
    # 未被认作上臂的骨仍属于主干支持，尤其保护贴近肱骨头的肩胛骨，
    # 不能让距离场把肩胛骨及其后方软组织随上臂一起分掉。
    trunk_markers = core | protected | ((bone_labels > 0) & ~arm_markers)
    if not np.any(trunk_markers) or np.any(trunk_markers & arm_markers):
        return body
    markers = trunk_markers.astype(np.uint32)
    markers[arm_markers] = 2
    terrain = sitk.GetImageFromArray((-distance).astype(np.float32))
    marker_image = sitk.GetImageFromArray(markers)
    partition = sitk.GetArrayFromImage(sitk.MorphologicalWatershedFromMarkers(
        terrain, marker_image, markWatershedLine=False, fullyConnected=True))
    removed = body & (partition == 2)
    # 内部局部组织不能被冒充成上臂；候选区域必须连到原体表外缘。
    surface = body & ~ndimage.binary_erosion(body, structure=np.ones((3, 3)), border_value=0)
    removal_labels, _ = ndimage.label(removed, structure=np.ones((3, 3), dtype=bool))
    accepted = np.zeros_like(body)
    for index, region in enumerate(ndimage.find_objects(removal_labels), start=1):
        if region is None:
            continue
        component = removal_labels[region] == index
        if (np.any(component & arm_markers[region])
                and np.any(component & surface[region])):
            accepted[region] |= component
    trunk = body & ~accepted
    if (not np.any(accepted) or np.count_nonzero(trunk) < 0.5 * np.count_nonzero(body)
            or np.any(protected & ~trunk)):
        return body
    return trunk
