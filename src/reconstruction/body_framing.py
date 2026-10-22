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


def _closing_footprint(spacing_yx: np.ndarray) -> np.ndarray:
    radius = np.ceil(_CLOSING_RADIUS_MM / spacing_yx).astype(int)
    y, x = np.ogrid[-radius[0]:radius[0] + 1, -radius[1]:radius[1] + 1]
    return (y * spacing_yx[0]) ** 2 + (x * spacing_yx[1]) ** 2 <= _CLOSING_RADIUS_MM ** 2


def _body_slice(hu: np.ndarray, spacing_yx: np.ndarray, footprint: np.ndarray) -> np.ndarray:
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
    # 多个符合身体厚度的连通域都保留，避免髋部下端的双腿或断开的手臂丢失一侧。
    return result


def build_body_framing_support(
    image: sitk.Image,
    *,
    target_mask: sitk.Image | None = None,
    exclude_arms: bool = False,
) -> BodyFramingSupport | None:
    """提取身体外轮廓；无可识别身体时返回 None，由调用方保留原取景。

    原始 HU 的每一存储层分别处理，不旋转或重采样 CT。闭合仅限已与扫描床
    分离的身体连通域，排除外界空气及可分离的薄床板，填补内部肺空腔。仅保存
    稠密的一体素厚外壳，保留任意 MPR 厚层与轮廓相交的支撑，避免整卷点云。
    被拒绝的外层若包围内部组织，该层不提供支持，不能用内部器官代替身体边界。
    肺部可传入目标 Mask 以保守排除可识别上臂；范围参考整组肺，不能随末层小肺收缩。
    不能可靠分离上臂时保留完整体表；髋骨等调用方默认不排除肢体。
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
        body = _body_slice(hu, spacing_yx, footprint)
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
