"""在已确定的上下界内，用 CT 体表解析区域的左右、前后范围。"""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import SimpleITK as sitk

from src.models import AnatomicalBounds, AnatomicalFrame, AxisBounds
from src.reconstruction.body_framing import build_body_framing_support


_BOX_EDGES = tuple(
    (first, last) for first in range(8) for last in range(first + 1, 8)
    if (first ^ last) in (1, 2, 4)
)


def _box_bounds(corners: np.ndarray, axes: np.ndarray, lower: float, upper: float):
    """物理源盒与 SI 区间相交后的 LR/AP 包围范围，不将整卷 FOV 投影进来。"""

    projected = corners @ axes
    depths = projected[:, 2]
    tolerance = max(1e-7, np.finfo(float).eps * max(abs(lower), abs(upper), 1.0) * 32)
    if upper <= depths.min() + tolerance or lower >= depths.max() - tolerance:
        return None
    vertices = list(projected[(depths >= lower) & (depths <= upper), :2])
    for first, last in _BOX_EDGES:
        span = depths[last] - depths[first]
        if abs(span) <= 1e-12:
            continue
        for boundary in (lower, upper):
            fraction = (boundary - depths[first]) / span
            if 0 <= fraction <= 1:
                vertices.append(projected[first, :2] + fraction * (
                    projected[last, :2] - projected[first, :2]
                ))
    if not vertices:
        return None
    vertices = np.asarray(vertices)
    return vertices.min(axis=0), vertices.max(axis=0)


class _BodyRegionProjector:
    """仅存外壳；各区域复用一次体表提取和同方向投影。"""

    def __init__(self, image: sitk.Image, target_mask: sitk.Image | None):
        self.image = image
        self.origin = np.asarray(image.GetOrigin(), dtype=float)
        self.voxel_axes = np.asarray(image.GetDirection()).reshape(3, 3) * image.GetSpacing()
        support = build_body_framing_support(
            image, target_mask=target_mask, exclude_arms=True,
            # 区域范围使用已确认的身体主体；不能再纳入被厚度门槛拒绝的
            # 独立外围薄层，否则稍厚弧床会重新撑大冠、矢状切片范围。
            preserve_small_components=False, separate_arms_only=True,
        )
        self.points = np.empty((0, 3)) if support is None else support.points_xyz
        self.support_region = "unavailable" if support is None else support.region
        self.known = np.zeros(image.GetSize()[2], dtype=bool)
        if len(self.points):
            z = np.rint((self.points - self.origin) @ np.linalg.inv(self.voxel_axes.T)[:, 2]).astype(int)
            self.known[z] = True
        transitions = np.diff(np.r_[False, ~self.known, False].astype(np.int8))
        self.fallback_boxes = [
            self._source_box(start, stop)
            for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))
        ]
        self.cache: dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _source_box(self, start: int, stop: int):
        size = self.image.GetSize()
        indices = np.asarray(list(product(
            (-0.5, size[0] - 0.5), (-0.5, size[1] - 0.5), (start - 0.5, stop - 0.5),
        )))
        return indices @ self.voxel_axes.T + self.origin

    def resolve(self, frame: AnatomicalFrame):
        axes = np.column_stack([
            (axis.x, axis.y, axis.z) for axis in (
                frame.left_right_axis, frame.anterior_posterior_axis, frame.superior_inferior_axis,
            )
        ])
        key = axes.tobytes()
        if key not in self.cache:
            projected = self.points @ axes
            half_voxel = np.abs(self.voxel_axes.T @ axes).sum(axis=0) / 2
            first_slice = self._source_box(0, 1) @ axes[:, 2]
            z_steps = np.arange(self.image.GetSize()[2]) * (self.voxel_axes[:, 2] @ axes[:, 2])
            source_ranges = np.column_stack((first_slice.min() + z_steps, first_slice.max() + z_steps))
            self.cache[key] = projected, half_voxel, source_ranges
        projected, half_voxel, source_ranges = self.cache[key]
        si = frame.bounds.superior_inferior
        lower, upper = si.minimum, si.maximum
        tolerance = max(1e-7, np.finfo(float).eps * max(abs(lower), abs(upper), 1.0) * 32)
        selected = ((projected[:, 2] + half_voxel[2] > lower + tolerance)
                    & (projected[:, 2] - half_voxel[2] < upper - tolerance))
        bounds = []
        if np.any(selected):
            points = projected[selected, :2]
            bounds.append((points.min(axis=0) - half_voxel[:2], points.max(axis=0) + half_voxel[:2]))
        fallback_count = int(np.count_nonzero(
            ~self.known & (source_ranges[:, 1] > lower + tolerance)
            & (source_ranges[:, 0] < upper - tolerance)
        ))
        if fallback_count:
            for corners in self.fallback_boxes:
                fallback = _box_bounds(corners, axes, lower, upper)
                if fallback is not None:
                    bounds.append(fallback)
        if not bounds:
            raise ValueError("所选上下范围与原 CT 没有可用交集，无法确定身体横向范围")
        transverse_lower = np.min([item[0] for item in bounds], axis=0)
        transverse_upper = np.max([item[1] for item in bounds], axis=0)
        warnings = []
        if fallback_count:
            warnings.append(
                f"所选区域内 {fallback_count} 个源层的体表不可识别，已使用这些源层的原 CT 视野补足横向范围"
            )
        resolved = replace(frame, bounds=AnatomicalBounds(
            AxisBounds(float(transverse_lower[0]), float(transverse_upper[0])),
            AxisBounds(float(transverse_lower[1]), float(transverse_upper[1])),
            si,
        ), warnings=[*frame.warnings, *warnings])
        diagnostic = {
            "status": (
                "body_with_source_fallback" if np.any(selected) else "source_fallback"
            ) if fallback_count else "body",
            "fallback_source_slice_count": fallback_count,
            "body_point_count": int(np.count_nonzero(selected)),
            "warnings": warnings,
        }
        return resolved.to_dict(), diagnostic


def apply_body_transverse_regions(
    raw_image: sitk.Image,
    target_mask: sitk.Image | None,
    anatomical_regions: Mapping[str, dict[str, Any]],
    regions_config: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """返回补足身体 LR/AP 的区域，不改变已解出的 SI、回正方向或输入对象。

    仅 ``transverse_source=body`` 触发 CT 提取。``target_mask`` 为区域目标标签
    的并集，只锚定可靠中央躯干；完全分离的外围组件可排除，连接手臂保守保留。
    体表未知的源层回退该层源 FOV 与 SI 区间的交集，不能以器官边界冒充体表。
    每个修改后的区域附 ``transverse_support`` 诊断，回退警告同时写入 frame。
    """

    names = [name for name, config in regions_config.items() if config.get("transverse_source") == "body"]
    result = dict(anatomical_regions)
    if not names:
        return result
    if raw_image.GetDimension() != 3:
        raise ValueError("身体区域需要三维原 CT")
    if target_mask is not None and (
        target_mask.GetDimension() != 3 or target_mask.GetSize() != raw_image.GetSize()
        or any(not np.allclose(actual, expected, rtol=0, atol=1e-5) for actual, expected in (
            (target_mask.GetSpacing(), raw_image.GetSpacing()),
            (target_mask.GetOrigin(), raw_image.GetOrigin()),
            (target_mask.GetDirection(), raw_image.GetDirection()),
        ))
    ):
        raise ValueError("身体区域的目标 Mask 与原 CT 空间不一致")
    projector = _BodyRegionProjector(raw_image, target_mask)
    for name in names:
        if name not in result:
            raise ValueError(f"身体区域 {name} 尚未解析上下边界")
        region = dict(result[name])
        frames = {"global_frame": region["global_frame"], **region.get("local_frames", {})}
        resolved_frames, diagnostics = {}, {}
        for frame_name, serialized in frames.items():
            try:
                resolved_frames[frame_name], diagnostics[frame_name] = projector.resolve(
                    AnatomicalFrame.from_dict(serialized),
                )
            except ValueError as error:
                raise ValueError(f"身体区域 {name}，方向 {frame_name}: {error}") from error
        region.update(
            global_frame=resolved_frames.pop("global_frame"), local_frames=resolved_frames,
            constrained=True, transverse_source="body",
            transverse_support={
                "source": "ct_body_contour", "arm_policy": "separate_components_only",
                "support_region": projector.support_region, "frames": diagnostics,
            },
        )
        result[name] = region
    return result
