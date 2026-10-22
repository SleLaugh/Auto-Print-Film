"""仅供脊柱定位图使用的冠状曲面背景和切片平面交线。

曲面只改变定位图在每个头脚位置读取的前后深度，保留原图的左右、头脚
网格。这里不创建或修改正式 MPR / CPR 的采样计划。
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk


_EPSILON = 1e-8
_JOIN_TOLERANCE = 1e-5
_SPINE_LABEL_PATTERN = re.compile(
    r"(?:(?:vertebrae_)?(?:C[1-7]|T(?:[1-9]|1[0-2])|L[1-5]|S1)|sacrum)\Z"
)


def _point(value: Any) -> np.ndarray | None:
    """只接收有限三维坐标，避免坏标注使整个重建流程失败。"""

    try:
        if isinstance(value, dict):
            result = np.asarray([value[axis] for axis in ("x", "y", "z")], dtype=float)
        else:
            result = np.asarray(value, dtype=float)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        return None
    return result


def _direction(value: Any) -> np.ndarray | None:
    vector = _point(value)
    if vector is None:
        return None
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= _EPSILON:
        return None
    return vector / length


def _image_affine(image: sitk.Image) -> tuple[np.ndarray, np.ndarray] | None:
    """物理点 = origin + affine @ 连续体素 XYZ；不能假设原图是单位方向。"""

    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        return None
    origin = np.asarray(image.GetOrigin(), dtype=float)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    affine = direction @ np.diag(spacing)
    if (
        not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(affine))
        or np.any(spacing <= 0.0)
        or abs(float(np.linalg.det(direction))) <= _EPSILON
    ):
        return None
    return origin, affine


def _filtered_center(result: Any, label: str) -> np.ndarray | None:
    if not isinstance(result, dict):
        return None
    geometry = result.get("spine_geometry")
    if not isinstance(geometry, dict):
        return None
    annotations = geometry.get("filtered_annotation_centroids")
    if not isinstance(annotations, dict):
        return None
    if label in {"S1", "vertebrae_S1", "sacrum"}:
        candidates = ("vertebrae_S1", "S1", "sacrum")
    else:
        alternate = (
            label.removeprefix("vertebrae_")
            if label.startswith("vertebrae_")
            else f"vertebrae_{label}"
        )
        candidates = (label, alternate)
    annotation = next((annotations[name] for name in candidates if name in annotations), None)
    if not isinstance(annotation, dict):
        return None
    if annotation.get("prism_found"):
        center = _point(annotation.get("prism_centroid_physical"))
        if center is not None:
            return center
    if annotation.get("found"):
        return _point(annotation.get("centroid_physical"))
    return None


def _surface_points(
    plan: dict,
    localization_result: dict | None,
    anatomical_analysis: dict | None,
) -> list[np.ndarray]:
    """CPR 跟随实际中心线；MPR 只取当前范围内经过过滤的椎体主体。"""

    if plan.get("slice_type") == "CPR":
        raw_curve = plan.get("curve_points")
        if isinstance(raw_curve, (list, tuple)) and len(raw_curve) >= 2:
            curve = [_point(value) for value in raw_curve]
            if all(point is not None for point in curve):
                return curve

    source_labels = plan.get("source_labels")
    source_labels = source_labels if isinstance(source_labels, (list, tuple)) else []
    source_labels = [
        label for label in source_labels
        if isinstance(label, str) and _SPINE_LABEL_PATTERN.fullmatch(label)
    ]
    if not source_labels:
        # 生产整切计划常只记录 global_frame 等几何来源；这时从本次已解析
        # 的合并定位目标取得范围，不能把同一卷 CT 的全部椎体标注都读进来。
        targets = (localization_result or {}).get("label_targets")
        if isinstance(targets, dict):
            source_labels = [
                label for label in targets
                if isinstance(label, str) and _SPINE_LABEL_PATTERN.fullmatch(label)
            ]
    points = []
    for label in dict.fromkeys(source_labels):
        center = _filtered_center(anatomical_analysis, label)
        if center is None:
            center = _filtered_center(localization_result, label)
        if center is not None:
            points.append(center)
    # 全量标签可能包含棘突等后方附件，不能作为这张定位图的主体深度回退。
    return points


@dataclass(frozen=True)
class SpineLocalizerSurface:
    """按原图 z 索引记录 y 深度的曲面，reference_index 仅用于显示几何。"""

    depth_indices: np.ndarray
    reference_index: int

    def sample(self, image: sitk.Image) -> sitk.Image:
        """逐行线性插值生成 X×1×Z 背景，不复制或浮点化整卷 CT。"""

        size = image.GetSize()
        depths = np.asarray(self.depth_indices, dtype=float)
        if (
            image.GetDimension() != 3
            or image.GetNumberOfComponentsPerPixel() != 1
            or depths.shape != (size[2],)
            or not np.all(np.isfinite(depths))
        ):
            raise ValueError("脊柱定位曲面深度与原图不匹配")
        source = sitk.GetArrayViewFromImage(image)
        sampled = np.empty((size[2], 1, size[0]), dtype=np.float32)
        for z_index, raw_depth in enumerate(depths):
            depth = float(np.clip(raw_depth, 0.0, size[1] - 1.0))
            lower = int(np.floor(depth))
            upper = min(lower + 1, size[1] - 1)
            weight = depth - lower
            # 先将单行转为 float32，避免整数 CT 的加减溢出。
            lower_row = np.asarray(source[z_index, lower, :], dtype=np.float32)
            if weight <= _EPSILON or lower == upper:
                sampled[z_index, 0, :] = lower_row
            else:
                upper_row = np.asarray(source[z_index, upper, :], dtype=np.float32)
                sampled[z_index, 0, :] = lower_row * (1.0 - weight) + upper_row * weight

        result = sitk.GetImageFromArray(sampled)
        reference = min(max(int(self.reference_index), 0), size[1] - 1)
        result.SetSpacing(image.GetSpacing())
        result.SetDirection(image.GetDirection())
        result.SetOrigin(image.TransformIndexToPhysicalPoint((0, reference, 0)))
        return result

    def plane_intersections(
        self,
        image: sitk.Image,
        planned_slice: dict,
        extent_mm: dict | None = None,
    ) -> list[np.ndarray]:
        """求切片平面与分段线性曲面的交线，返回连续体素 XYZ。

        每段曲面写成 P(x,t)=B+Dx*x+V*t，在 0≤x≤sizeX−1、0≤t≤1
        内求真实交线。定位图省略 extent_mm，仅按显示网格裁线，避免 Fill
        缩短定位线；显式提供范围的几何查询仍可按有限矩形裁剪。
        法向 y 分量参与求交，不把固定参考层的旧线搬到曲面上。
        """

        geometry = _image_affine(image)
        center = _point(planned_slice.get("center"))
        normal = _direction(planned_slice.get("normal"))
        row = _direction(planned_slice.get("row_dir"))
        col = _direction(planned_slice.get("col_dir"))
        half_width = half_height = None
        if extent_mm is not None:
            try:
                half_width = float(extent_mm["width"]) / 2.0
                half_height = float(extent_mm["height"]) / 2.0
            except (KeyError, TypeError, ValueError, OverflowError):
                return []
            if (
                not np.isfinite(half_width)
                or not np.isfinite(half_height)
                or min(half_width, half_height) < 0.0
            ):
                return []
        size = image.GetSize()
        depths = np.asarray(self.depth_indices, dtype=float)
        if (
            geometry is None
            or center is None
            or normal is None
            or row is None
            or col is None
            or depths.shape != (size[2],)
            or not np.all(np.isfinite(depths))
        ):
            return []
        origin, affine = geometry
        dx, dy, dz = affine.T
        depths = np.clip(depths, 0.0, size[1] - 1.0)
        segments = []
        for z_index in range(size[2] - 1):
            depth = depths[z_index]
            depth_delta = depths[z_index + 1] - depth
            base = origin + dy * depth + dz * z_index
            along_z = dz + dy * depth_delta
            a = float(np.dot(normal, dx))
            b = float(np.dot(normal, along_z))
            c = float(np.dot(normal, base - center))
            scale = max(float(np.linalg.norm(dx)), float(np.linalg.norm(along_z)), 1.0)
            tolerance = _EPSILON * scale
            if abs(a) <= tolerance and abs(b) <= tolerance:
                # 完全平行时没有交线；完全共面则为二维重叠，也没有唯一定位线。
                continue
            if abs(a) >= abs(b):
                parameter_origin = np.asarray([-c / a, 0.0])
                parameter_direction = np.asarray([-b / a, 1.0])
            else:
                # n·Dx 接近零时仍可求解，交线可能是固定 z 的横向线。
                parameter_origin = np.asarray([0.0, -c / b])
                parameter_direction = np.asarray([1.0, -a / b])
            physical_origin = base + dx * parameter_origin[0] + along_z * parameter_origin[1]
            physical_direction = dx * parameter_direction[0] + along_z * parameter_direction[1]
            constraints = [
                (parameter_origin[0], parameter_direction[0], 0.0, size[0] - 1.0),
                (parameter_origin[1], parameter_direction[1], 0.0, 1.0),
            ]
            if extent_mm is not None:
                constraints.extend([
                    (
                        float(np.dot(physical_origin - center, row)),
                        float(np.dot(physical_direction, row)),
                        -half_width,
                        half_width,
                    ),
                    (
                        float(np.dot(physical_origin - center, col)),
                        float(np.dot(physical_direction, col)),
                        -half_height,
                        half_height,
                    ),
                ])
            interval = _clip_parameter(constraints)
            if interval is None:
                continue
            endpoints = []
            for value in interval:
                x_index, t_index = parameter_origin + parameter_direction * value
                endpoints.append([x_index, depth + depth_delta * t_index, z_index + t_index])
            segment = np.asarray(endpoints, dtype=float)
            if np.linalg.norm(segment[1] - segment[0]) > _JOIN_TOLERANCE:
                segments.append(segment)
        return _merge_segments(segments)


def _clip_parameter(
    constraints: list[tuple[float, float, float, float]],
) -> tuple[float, float] | None:
    """对 low≤origin+direction*s≤high 的所有约束求共同参数区间。"""

    start, end = -np.inf, np.inf
    for origin, direction, lower, upper in constraints:
        if abs(direction) <= _EPSILON:
            if origin < lower - _EPSILON or origin > upper + _EPSILON:
                return None
            continue
        bound_a, bound_b = (lower - origin) / direction, (upper - origin) / direction
        start = max(start, min(bound_a, bound_b))
        end = min(end, max(bound_a, bound_b))
        if end < start - _EPSILON:
            return None
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        return None
    return float(start), float(end)


def _merge_segments(segments: list[np.ndarray]) -> list[np.ndarray]:
    """合并相邻 z 段，既减少逐段抗锯齿开销，也去掉边界上的重复线。"""

    paths: list[np.ndarray] = []
    for segment in segments:
        if (segment[0, 2], segment[0, 0]) > (segment[-1, 2], segment[-1, 0]):
            segment = segment[::-1]
        merged = False
        for index, path in enumerate(paths):
            if np.allclose(path[-1], segment[0], rtol=0.0, atol=_JOIN_TOLERANCE):
                paths[index] = np.concatenate((path, segment[1:]), axis=0)
                merged = True
                break
            if np.allclose(path[-1], segment[-1], rtol=0.0, atol=_JOIN_TOLERANCE):
                # 横向交线可能同时属于前后两段曲面，不重复描画。
                if np.allclose(path[0], segment[0], rtol=0.0, atol=_JOIN_TOLERANCE):
                    merged = True
                    break
        if not merged:
            paths.append(segment)
    return paths


def build_spine_localizer_surface(
    image: sitk.Image,
    plan: dict,
    localization_result: dict | None = None,
    anatomical_analysis: dict | None = None,
) -> SpineLocalizerSurface | None:
    """为脊柱整段及颈胸腰部位的矢状位 MPR / CPR 构建冠状定位图曲面。

    缺少可靠主体锚点时返回 None，调用方继续使用原有固定层定位图。
    该回退只涉及定位图，不改变正式切片，也不借用后方附件的全量质心。
    """

    if (
        plan.get("slice_part") not in {"颈椎", "胸椎", "腰椎", "脊柱"}
        or plan.get("plane") != "矢状位"
        or plan.get("localizer_plane") != "冠状位"
        or plan.get("slice_type") not in {"MPR", "CPR"}
    ):
        return None
    geometry = _image_affine(image)
    if geometry is None or min(image.GetSize()) <= 0 or image.GetSize()[2] < 2:
        return None
    points = _surface_points(plan, localization_result, anatomical_analysis)
    if len(points) < 2:
        return None
    origin, affine = geometry
    indices = np.linalg.solve(affine, (np.asarray(points) - origin).T).T
    if not np.all(np.isfinite(indices)):
        return None
    indices = indices[np.argsort(indices[:, 2], kind="stable")]
    z_values: list[float] = []
    y_values: list[float] = []
    group_start = 0
    for group_end in range(1, len(indices) + 1):
        if group_end < len(indices) and abs(indices[group_end, 2] - indices[group_start, 2]) <= 1e-6:
            continue
        group = indices[group_start:group_end]
        z_values.append(float(np.mean(group[:, 2])))
        y_values.append(float(np.mean(group[:, 1])))
        group_start = group_end
    if len(z_values) < 2:
        return None
    depths = np.interp(np.arange(image.GetSize()[2], dtype=float), z_values, y_values)
    depths = np.clip(depths, 0.0, image.GetSize()[1] - 1.0)

    reference = _point(plan.get("localizer_reference"))
    if reference is not None:
        reference_y = float(np.linalg.solve(affine, reference - origin)[1])
        if not np.isfinite(reference_y):
            reference_y = float(np.median(indices[:, 1]))
    else:
        slice_centers = [
            center
            for planned_slice in plan.get("slices", [])
            if isinstance(planned_slice, dict)
            and (center := _point(planned_slice.get("center"))) is not None
        ]
        if slice_centers:
            center_indices = np.linalg.solve(affine, (np.asarray(slice_centers) - origin).T).T
            reference_y = float(np.median(center_indices[:, 1]))
        else:
            reference_y = float(np.median(indices[:, 1]))
    reference_index = min(max(int(round(reference_y)), 0), image.GetSize()[1] - 1)
    return SpineLocalizerSurface(depth_indices=depths, reference_index=reference_index)
