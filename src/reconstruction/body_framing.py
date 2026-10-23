"""从原始 CT 提取身体外轮廓，只为显示取景提供物理支撑。"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from src.anatomy.orientation import patient_reference_axes
from src.reconstruction.trunk_framing import extract_trunk_slice


@dataclass(frozen=True)
class BodyFramingSupport:
    """逐源层的外边界及真实物理体素轴，不改变切割目标。

    region=trunk 表示其中已有源层按可靠证据排臂，其他源层仍可能保留全体表。
    """

    points_xyz: np.ndarray
    voxel_axes: np.ndarray
    longitudinal_axis_xyz: tuple[float, float, float]
    region: str = "body"


_BODY_THRESHOLD_HU = -500.0
_CLOSING_RADIUS_MM = 1.5
# 用于过滤细床板、噪声和孤立细骨；被拒绝的外层不能由内部组织冒充。
_MIN_BODY_AREA_MM2 = 500.0
_MIN_BODY_SPAN_MM = 25.0
_MIN_BODY_RADIUS_MM = 8.0
_MIN_PERIPHERAL_AREA_MM2 = 4.0
_BODY_NEIGHBOURHOOD_MM = 3.0
_MAX_SEPARATE_TABLE_RADIUS_MM = 4.0
_MIN_SEPARATE_TABLE_WIDTH_MM = 100.0


def _closing_footprint(spacing_yx: np.ndarray) -> np.ndarray:
    radius = np.ceil(_CLOSING_RADIUS_MM / spacing_yx).astype(int)
    y, x = np.ogrid[-radius[0]:radius[0] + 1, -radius[1]:radius[1] + 1]
    return (y * spacing_yx[0]) ** 2 + (x * spacing_yx[1]) ** 2 <= _CLOSING_RADIUS_MM ** 2


def _separate_posterior_table(
    component: np.ndarray,
    bounds: tuple[slice, slice],
    spacing_yx: np.ndarray,
    patient_in_plane_axes: np.ndarray | None,
    accepted_posterior: float | None,
    accepted_lateral: tuple[float, float] | None,
) -> bool:
    """只排除独立薄床；中央床面在人体后方，允许外侧床翼向前翘起。"""

    if patient_in_plane_axes is None or accepted_posterior is None or accepted_lateral is None:
        return False
    radius = ndimage.distance_transform_edt(np.pad(component, 1), sampling=spacing_yx).max()
    if radius > _MAX_SEPARATE_TABLE_RADIUS_MM:
        return False
    yx = np.argwhere(component) + np.asarray([item.start for item in bounds])
    # 同一源层共有的原点和层方向位移不影响前后关系，故只投影面内索引。
    projected = yx[:, ::-1] @ patient_in_plane_axes.T
    half_voxel = np.abs(patient_in_plane_axes).sum(axis=1) / 2
    width = float(np.ptp(projected[:, 0]) + 2 * half_voxel[0])
    if width < _MIN_SEPARATE_TABLE_WIDTH_MM:
        return False
    if float(projected[:, 1].min() - half_voxel[1]) > accepted_posterior + 1e-7:
        # 源 FOV 可把床圈截成左右片段：整个组件都在后方时，无需横跨中央。
        return True
    midpoint = sum(accepted_lateral) / 2
    core_half_width = min(30.0, (accepted_lateral[1] - accepted_lateral[0]) / 4)
    central = projected[np.abs(projected[:, 0] - midpoint) <= core_half_width]
    # 必须有横跨中央带的后方床面证据；单侧细长肢体不能冒充扫描床。
    if not len(central) or np.ptp(central[:, 0]) < min(10.0, core_half_width):
        return False
    anterior = float(central[:, 1].min() - half_voxel[1])
    return anterior > accepted_posterior + 1e-7


def _body_slice(
    hu: np.ndarray,
    spacing_yx: np.ndarray,
    footprint: np.ndarray,
    *,
    preserve_small_components: bool = False,
    patient_in_plane_axes: np.ndarray | None = None,
) -> np.ndarray:
    """先分离身体和扫描床，再仅对身体闭合、填补肺等内部空腔。"""

    components, count = ndimage.label(hu > _BODY_THRESHOLD_HU, np.ones((3, 3)))
    result = np.zeros(hu.shape, dtype=bool)
    if not count:
        return result
    areas = np.bincount(components.ravel()) * float(np.prod(spacing_yx))
    padding = np.asarray(footprint.shape) // 2 + 1
    accepted_bounds = []
    rejected_components = []
    for index, bounds in enumerate(ndimage.find_objects(components), start=1):
        if bounds is None:
            continue
        span = np.array([item.stop - item.start for item in bounds]) * spacing_yx
        if areas[index] < _MIN_BODY_AREA_MM2 or np.any(span < _MIN_BODY_SPAN_MM):
            rejected_components.append((index, bounds))
            continue
        component = components[bounds] == index
        # 局部 bbox 外补零，避免贴图像边缘的身体在闭合时被侵蚀。
        padded = np.pad(component, tuple((int(value), int(value)) for value in padding))
        # 薄而弯曲的扫描床也可能有很大的 bbox，须按实心物理厚度排除。
        # 检查必须在填孔前进行，不能把中空的床架填成厚实的“人体”。
        if ndimage.distance_transform_edt(padded, sampling=spacing_yx).max() < _MIN_BODY_RADIUS_MM:
            rejected_components.append((index, bounds))
            continue
        filled = ndimage.binary_fill_holes(padded)
        closed = ndimage.binary_closing(filled, structure=footprint)
        closed = ndimage.binary_fill_holes(closed)
        unpad = tuple(slice(int(value), -int(value)) for value in padding)
        result[bounds] |= closed[unpad]
        accepted_bounds.append(bounds)
    for index, bounds in rejected_components:
        # 只有 bbox 能包围已接受组件的外层才需要进一步检查。普通薄床板、
        # 独立细环和体内小碎块不应导致可靠的身体轮廓被丢弃。
        if not any(all(outer.start <= inner.start and outer.stop >= inner.stop
                       for outer, inner in zip(bounds, accepted))
                   for accepted in accepted_bounds):
            continue
        padded = np.pad(components[bounds] == index,
                        tuple((int(value), int(value)) for value in padding))
        # 仅用闭合/填孔检查嵌套关系，不把被筛掉的薄层重新认作身体或床。
        # 先闭合可封住小缺口；若内部组织被这种不可靠外层包围，整层回退，
        # 不能把内部器官当作完整身体边界来移动画面中心。
        closed = ndimage.binary_closing(padded, structure=footprint)
        enclosed = ndimage.binary_fill_holes(closed) & ~closed
        unpad = tuple(slice(int(value), -int(value)) for value in padding)
        if np.any(enclosed[unpad] & result[bounds]):
            return np.zeros_like(result)
    if preserve_small_components and accepted_bounds:
        # 低剂量 CT 的体外空气可出现孤立阈上噪点，不能让单像素撑满画布。
        # 靠近已确认体表的小组件仍保留；较大的独立小肢体也不受此过滤。
        body_distance = (
            ndimage.distance_transform_edt(~result, sampling=spacing_yx)
            if any(areas[index] < _MIN_PERIPHERAL_AREA_MM2 for index, _ in rejected_components)
            else None
        )
        accepted_posterior = None
        accepted_lateral = None
        if patient_in_plane_axes is not None:
            # 用已确认人体的 bbox 外界作保守后缘。斜扫 bbox 可能比真实人体
            # 更靠后，只会让难判断的设备被保留，不会多删人体。
            ap_steps = patient_in_plane_axes[1]
            accepted_posterior = max(
                sum(max(step * (bounds[axis].start - 0.5),
                        step * (bounds[axis].stop - 0.5))
                    for step, axis in zip(ap_steps, (1, 0)))
                for bounds in accepted_bounds
            )
            lr_steps = patient_in_plane_axes[0]
            accepted_lateral = (
                min(sum(min(step * (bounds[axis].start - 0.5),
                            step * (bounds[axis].stop - 0.5))
                        for step, axis in zip(lr_steps, (1, 0))) for bounds in accepted_bounds),
                max(sum(max(step * (bounds[axis].start - 0.5),
                            step * (bounds[axis].stop - 0.5))
                        for step, axis in zip(lr_steps, (1, 0))) for bounds in accepted_bounds),
            )
        for index, bounds in rejected_components:
            component = components[bounds] == index
            if areas[index] < _MIN_PERIPHERAL_AREA_MM2:
                if not np.any(body_distance[bounds][component] <= _BODY_NEIGHBOURHOOD_MM):
                    continue
            if not _separate_posterior_table(
                component, bounds, spacing_yx, patient_in_plane_axes, accepted_posterior, accepted_lateral,
            ):
                # 小肢体和碎体表不满足主躯干门槛，也不能因此被删除；仅保留
                # 原组件，不闭合或填孔，避免与床板或其他未知组件相连。
                result[bounds] |= component
    # 多个符合身体厚度的连通域都保留，避免髋部下端的双腿或断开的手臂丢失一侧。
    return result


def build_body_framing_support(
    image: sitk.Image,
    *,
    target_mask: sitk.Image | None = None,
    exclude_arms: bool = False,
    preserve_small_components: bool = False,
) -> BodyFramingSupport | None:
    """提取身体外轮廓；无可识别身体时返回 None，由调用方保留原取景。

    原始 HU 的每一存储层分别处理，不旋转或重采样 CT。闭合仅限已与扫描床
    分离的身体连通域，排除外界空气及可分离的薄床板，填补内部肺空腔。仅保存
    稠密的一体素厚外壳，保留任意 MPR 厚层与轮廓相交的支撑，避免整卷点云。
    被拒绝的外层若包围内部组织，该层不提供支持，不能用内部器官代替身体边界。
    肺部可传入目标 Mask 以保守排除可识别上臂；范围参考整组肺，不能随末层小肺收缩。
    不能可靠分离上臂时保留完整体表；髋骨等调用方默认不排除肢体。
    preserve_small_components 为真时，在已有可靠主躯干的源层保留小肢体和
    碎体表；排除远离已确认人体超过 3 mm 且面积不足 4 mm² 的孤立噪点，
    以及独立、薄、沿患者左右宽且中央床面位于人体后方的明确床板。
    外侧床翼可以向前翘起；不拆开与人体相连的设备。
    方向不足以判断前后关系时保留组件；无可靠主躯干时仍不提供支持。
    与身体厚度接近或已和身体连通的垫子/设备不能仅凭 HU 可靠区分，可能被包含。
    """

    if image.GetDimension() != 3:
        raise ValueError("身体取景支持需要三维 CT")
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    reference = patient_reference_axes(direction)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0) or not np.all(np.isfinite(origin)):
        raise ValueError("身体取景支持需要有限原点与正像素间距")
    voxel_axes = direction * spacing
    spacing_yx = spacing[:2][::-1]
    footprint = _closing_footprint(spacing_yx)
    patient_in_plane_axes = (
        voxel_axes[:2, :2] if abs(direction[2, 2]) >= 0.8 else None
    )
    target_array = None
    target_bounds_xy = None
    # 排臂必须沿患者左右方向，不能把转存后的 AP 方向误认作双臂。
    # 面内轴互换时转置二维工作数组；非轴向或方向过于倾斜时保守保留身体。
    lateral_axis = int(np.argmax(np.abs(direction[0])))
    arm_orientation_supported = (
        lateral_axis in (0, 1)
        and abs(direction[0, lateral_axis]) >= 0.8
        and abs(direction[2, 2]) >= 0.8
    )
    if exclude_arms and target_mask is not None and arm_orientation_supported:
        if target_mask.GetDimension() != 3:
            raise ValueError("躯干取景的目标 Mask 需要三维图像")
        if (target_mask.GetSize() != image.GetSize()
                or target_mask.GetSpacing() != image.GetSpacing()
                or target_mask.GetOrigin() != image.GetOrigin()
                or target_mask.GetDirection() != image.GetDirection()):
            target_mask = sitk.Resample(
                target_mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )
        target_array = sitk.GetArrayViewFromImage(target_mask) > 0
        xs = np.flatnonzero(np.any(target_array, axis=(0, 1)))
        ys = np.flatnonzero(np.any(target_array, axis=(0, 2)))
        if len(xs) and len(ys):
            target_bounds_xy = (int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1]))
    points = []
    separated_arms = False
    for z, hu in enumerate(sitk.GetArrayViewFromImage(image)):
        body = _body_slice(
            hu, spacing_yx, footprint,
            preserve_small_components=preserve_small_components,
            patient_in_plane_axes=patient_in_plane_axes,
        )
        if not np.any(body):
            continue
        if target_bounds_xy is not None:
            if lateral_axis == 1:
                x_min, x_max, y_min, y_max = target_bounds_xy
                trunk = extract_trunk_slice(
                    hu.T, body.T, target_array[z].T,
                    (y_min, y_max, x_min, x_max), spacing_yx[::-1],
                ).T
            else:
                trunk = extract_trunk_slice(hu, body, target_array[z], target_bounds_xy, spacing_yx)
            separated_arms |= bool(np.any(body & ~trunk))
            body = trunk
        boundary = body & ~ndimage.binary_erosion(body, structure=np.ones((3, 3)), border_value=0)
        yx = np.argwhere(boundary)
        xyz = np.column_stack((yx[:, 1], yx[:, 0], np.full(len(yx), z)))
        points.append(xyz @ voxel_axes.T + origin)
    if not points:
        return None
    points_xyz = np.concatenate(points)
    points_xyz.setflags(write=False)
    voxel_axes.setflags(write=False)
    return BodyFramingSupport(
        points_xyz, voxel_axes, tuple(map(float, reference[:, 2])),
        region="trunk" if separated_arms else "body",
    )
