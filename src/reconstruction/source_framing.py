"""原 CT 的切面保护：完整源画布，或排除空气/床板后的完整人体。"""

from itertools import product

import numpy as np
import SimpleITK as sitk

from src.reconstruction.vertical_support import _sampling_offsets
from src.reconstruction.body_framing import build_body_framing_support


class SourceImageSupport:
    """原图半体素外边界与厚层的交集；仅保存八个角点，不展开源体素。"""

    def __init__(self, image: sitk.Image):
        self.image = image
        self.origin = np.asarray(image.GetOrigin(), dtype=float)
        voxel_axes = np.asarray(image.GetDirection()).reshape(3, 3) * image.GetSpacing()
        indices = np.asarray(list(product(*[(-0.5, size - 0.5) for size in image.GetSize()])))
        self.corners = indices @ voxel_axes.T
        self.edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                      if (i ^ j) in (1, 2, 4)]

    def slab_bounds(
        self, plan: dict, center: np.ndarray, axes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """返回源盒与实际采样厚层相交部分在行、列方向的完整外边界。"""

        interval = self._slab_interval(plan, center, axes[:, 2])
        if interval is None:
            return None
        return self._box_bounds(self.corners, axes, *interval)

    def _slab_interval(self, plan: dict, center: np.ndarray, normal: np.ndarray):
        """名义厚层和采样器实际使用的源层中心的联合范围，相对原点。"""

        offsets = _sampling_offsets(self.image, plan, center, normal)
        if not len(offsets):
            return None
        half_thickness = float(plan.get("slice_thickness", 1.0)) / 2
        # 轴对齐路径按源层真实重叠厚度采样；接边源层中心可能位于 slab 外。
        # 连续厚层与实际采样中心都纳入保护，也覆盖近轴对齐时的微小面内变化。
        depth = float((center - self.origin) @ normal)
        lower = depth + min(-half_thickness, float(np.min(offsets)))
        upper = depth + max(half_thickness, float(np.max(offsets)))
        return lower, upper

    def _box_bounds(self, corners, axes, lower, upper):
        normal = axes[:, 2]
        depths = corners @ normal
        depth = (lower + upper) / 2
        tolerance = max(1e-7, np.finfo(float).eps * max(abs(depth), 1.0) * 32)
        if upper <= float(depths.min()) + tolerance or lower >= float(depths.max()) - tolerance:
            return None
        vertices = list(corners[(depths >= lower) & (depths <= upper)])
        for first, last in self.edges:
            span = float(depths[last] - depths[first])
            if abs(span) <= 1e-12:
                continue
            for boundary in (lower, upper):
                fraction = float((boundary - depths[first]) / span)
                if 0 <= fraction <= 1:
                    vertices.append(corners[first] + fraction * (corners[last] - corners[first]))
        if not vertices:
            return None
        projected = np.asarray(vertices) @ axes[:, :2] + self.origin @ axes[:, :2]
        return projected.min(axis=0), projected.max(axis=0)


class BodyImageSupport(SourceImageSupport):
    """只限制倍率，不改变目标中心；未知源层局部回退完整源画布。

    身体外壳按原 CT 的每一存储层提取，保留分开的肩臂和双腿，不排臂。
    稠密外壳及其半体素支撑保护任意斜切厚层，另留 2 mm 的体表余量。
    空气及可分离的薄床板不再撑大视野；连着人体的设备保守保留。
    """

    margin_mm = 2.0

    def __init__(self, image: sitk.Image):
        super().__init__(image)
        body = build_body_framing_support(image, exclude_arms=False, preserve_small_components=True)
        self._points = np.empty((0, 3)) if body is None else body.points_xyz - self.origin
        self._voxel_axes = np.asarray(image.GetDirection()).reshape(3, 3) * image.GetSpacing()
        known = np.zeros(image.GetSize()[2], dtype=bool)
        if len(self._points):
            indices_z = np.rint(self._points @ np.linalg.inv(self._voxel_axes.T)[:, 2]).astype(int)
            known[indices_z] = True
        self.fallback_slice_count = int(np.count_nonzero(~known))
        # 不把失败层静默略过；按连续源层区间保存局部源盒。
        transitions = np.diff(np.r_[False, ~known, False].astype(np.int8))
        self._fallback_boxes = []
        for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            indices = np.asarray(list(product(
                (-0.5, image.GetSize()[0] - 0.5),
                (-0.5, image.GetSize()[1] - 0.5),
                (start - 0.5, stop - 0.5),
            )))
            self._fallback_boxes.append(indices @ self._voxel_axes.T)
        self._projection_key = None

    def slab_bounds(self, plan, center, axes):
        interval = self._slab_interval(plan, center, axes[:, 2])
        if interval is None:
            return None
        lower_depth, upper_depth = interval
        source_bounds = self._box_bounds(self.corners, axes, *interval)
        if source_bounds is None:
            return None
        key = tuple(axes.ravel())
        if key != self._projection_key:
            self._projected = self._points @ axes[:, :2] + self.origin @ axes[:, :2]
            self._depths = self._points @ axes[:, 2]
            self._half_voxel = np.abs(self._voxel_axes.T @ axes).sum(axis=0) / 2
            self._projection_key = key
        half = self._half_voxel
        tolerance = max(1e-7, np.finfo(float).eps * max(abs(lower_depth), abs(upper_depth), 1.0) * 32)
        selected = ((self._depths + half[2] > lower_depth + tolerance)
                    & (self._depths - half[2] < upper_depth - tolerance))
        bounds = []
        if np.any(selected):
            points = self._projected[selected]
            bounds.append((
                np.maximum(points.min(axis=0) - half[:2] - self.margin_mm, source_bounds[0]),
                np.minimum(points.max(axis=0) + half[:2] + self.margin_mm, source_bounds[1]),
            ))
        for corners in self._fallback_boxes:
            fallback = self._box_bounds(corners, axes, *interval)
            if fallback is not None:
                bounds.append(fallback)
        if not bounds:
            return None
        return np.min([item[0] for item in bounds], axis=0), np.max([item[1] for item in bounds], axis=0)
