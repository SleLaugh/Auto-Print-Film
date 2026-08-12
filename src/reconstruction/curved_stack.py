"""曲面堆叠（curved_stack）CPR 重采样。"""

import math
import time
from typing import Any, Dict

import numpy as np
import SimpleITK as sitk

from src.utils import DebugLog


class CurvedStackResampler:
    """沿中心锚点折线和计划堆叠偏移生成单张 CPR 图像。"""

    _PARALLEL_OFFSET_SUPPORT_MM = 80.0
    _MAX_DIRECTION_CHANGE_RATIO = 0.9
    _DIRECTION_SMOOTHING_SEARCH_STEPS = 24

    def sample_slice(
        self,
        image: sitk.Image,
        plan: Dict[str, Any],
        planned_slice: Dict[str, Any],
    ) -> sitk.Image:
        """采样一张 CPR HU 图，返回 size=(cols, rows, 1) 的图像。"""

        perf_start = time.perf_counter()
        curve_points = self._curve_points(plan)
        curve_spacing = float(plan["curve_spacing_mm"])
        if curve_spacing <= 0:
            raise ValueError("curve_spacing_mm 必须大于 0")

        extent = plan["extent_mm"]
        output_spacing = plan["output_spacing_mm"]
        spacing_x = float(output_spacing["x"])
        spacing_y = float(output_spacing["y"])
        if spacing_x <= 0 or spacing_y <= 0:
            raise ValueError("output_spacing_mm 的 x/y 必须大于 0")

        columns = max(1, int(round(float(extent["width"]) / spacing_x)) + 1)
        rows = max(1, int(round(float(extent["height"]) / spacing_y)) + 1)
        curve_samples = self._resample_curve(curve_points, rows)
        offset_axis = self._normalize(self._point(plan["offset_axis"]))
        offset_mm = float(planned_slice.get("offset_mm", 0.0))
        if not math.isfinite(offset_mm):
            raise ValueError("offset_mm 必须是有限数值")
        slice_thickness = float(plan.get("slice_thickness", 1.0))
        if not math.isfinite(slice_thickness) or slice_thickness <= 0.0:
            raise ValueError("slice_thickness 必须是大于 0 的有限数值")
        maximum_offset_mm = max(
            self._maximum_offset_extent_mm(plan),
            abs(offset_mm) + slice_thickness / 2.0,
        )
        offset_curves, preferred_normals = self._offset_curve_geometry(
            curve_samples,
            offset_axis,
            [offset_mm],
            support_mm=maximum_offset_mm,
        )
        offset_curve = offset_curves[offset_mm]
        tangents, in_plane_directions, slab_directions = self._surface_frame(
            offset_curve,
            preferred_normals[offset_mm],
        )
        in_plane_offsets = (
            np.arange(columns, dtype=float) - (columns - 1) / 2.0
        ) * spacing_x

        source = sitk.GetArrayViewFromImage(image)
        slab_offsets = self._slab_offsets(image, slice_thickness)
        sampled = self._sample_curved_surface_slab(
            image=image,
            source=source,
            curve_samples=offset_curve,
            in_plane_directions=in_plane_directions,
            offset_directions=slab_directions,
            in_plane_offsets=in_plane_offsets,
            slab_offsets=slab_offsets,
        )

        output = sitk.GetImageFromArray(sampled[np.newaxis, ...])
        output.SetSpacing((spacing_x, spacing_y, 1.0))
        origin = offset_curve[0]
        first_in_plane = in_plane_directions[0]
        if columns > 1:
            origin = origin + first_in_plane * in_plane_offsets[0]
        output.SetOrigin(tuple(float(value) for value in origin))
        output.SetDirection(
            self._direction(
                first_in_plane,
                tangents[0],
                slab_directions[0],
            )
        )
        DebugLog(
            "    [CPR采样性能] "
            f"plan={plan.get('plan_id', '')}, "
            f"slice_index={planned_slice.get('index', '')}, "
            f"offset_mm={offset_mm:.3f}, "
            f"rows={rows}, cols={columns}, "
            f"slab_count={len(slab_offsets)}, "
            f"slice_thickness={slice_thickness:.3f}mm, "
            "offset_geometry=shared_parallel_field, "
            f"max_offset_extent={maximum_offset_mm:.3f}mm, "
            f"耗时={time.perf_counter() - perf_start:.3f}s"
        )
        return output

    @staticmethod
    def _curve_points(plan: Dict[str, Any]) -> np.ndarray:
        """读取并校验 CPR 中心曲线点。"""

        points = [
            CurvedStackResampler._point(point)
            for point in plan.get("curve_points", [])
        ]
        if len(points) < 2:
            raise ValueError("curved_stack 至少需要两个 curve_points")
        curve = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(curve)):
            raise ValueError("curve_points 必须是有限数值")
        return curve

    @staticmethod
    def _resample_curve(curve: np.ndarray, rows: int) -> np.ndarray:
        """按弧长把折线曲线重采样成输出行数。"""

        deltas = np.diff(curve, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        total_length = float(np.sum(lengths))
        if total_length <= 0.0 or not math.isfinite(total_length):
            raise ValueError("curve_points 曲线长度必须大于 0")
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        distances = np.linspace(0.0, total_length, rows)
        samples = np.empty((rows, 3), dtype=float)
        for index, distance in enumerate(distances):
            segment = min(
                int(np.searchsorted(cumulative, distance, side="right") - 1),
                len(lengths) - 1,
            )
            start_distance = cumulative[segment]
            segment_length = lengths[segment]
            ratio = 0.0 if segment_length <= 0.0 else (
                distance - start_distance
            ) / segment_length
            samples[index] = curve[segment] + deltas[segment] * ratio
        return samples

    @staticmethod
    def _curve_tangents(samples: np.ndarray) -> np.ndarray:
        """为每个曲线采样点计算稳定切向。"""

        tangents = np.empty_like(samples)
        for index in range(len(samples)):
            if index == 0:
                tangent = samples[1] - samples[0]
            elif index == len(samples) - 1:
                tangent = samples[-1] - samples[-2]
            else:
                tangent = samples[index + 1] - samples[index - 1]
            tangents[index] = CurvedStackResampler._normalize(tangent)
        return tangents

    @classmethod
    def _curve_frame(
        cls,
        samples: np.ndarray,
        preferred_offset_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成连续的曲线切向、横向和局部法向。

        ``preferred_offset_axis`` 只决定首行法向朝向。后续法向把上一行法向
        平行输运到当前切向的正交平面，避免局部坐标系翻转。
        """

        tangents = cls._curve_tangents(samples)
        preferred = cls._normalize(np.asarray(preferred_offset_axis, dtype=float))
        offset_directions = np.empty_like(tangents)
        previous: np.ndarray | None = None

        for index, tangent in enumerate(tangents):
            if previous is None:
                candidate = preferred - np.dot(preferred, tangent) * tangent
            else:
                candidate = previous - np.dot(previous, tangent) * tangent

            candidate_length = float(np.linalg.norm(candidate))
            if not math.isfinite(candidate_length) or candidate_length <= 1e-8:
                candidate = preferred - np.dot(preferred, tangent) * tangent
                candidate_length = float(np.linalg.norm(candidate))
            if not math.isfinite(candidate_length) or candidate_length <= 1e-8:
                candidate = cls._stable_perpendicular(tangent)
            else:
                candidate = candidate / candidate_length

            if previous is not None and float(np.dot(candidate, previous)) < 0.0:
                candidate = -candidate
            offset_directions[index] = candidate
            previous = candidate

        _, in_plane_directions, _ = cls._surface_frame(
            samples,
            offset_directions,
        )

        return tangents, in_plane_directions, offset_directions

    @classmethod
    def _offset_curve_samples(
        cls,
        curve_samples: np.ndarray,
        preferred_offset_axis: np.ndarray,
        offsets_mm: list[float],
        *,
        support_mm: float | None = None,
    ) -> Dict[float, np.ndarray]:
        """返回与实际 CPR 采样一致的每个 offset 中心曲线。"""

        curves, _ = cls._offset_curve_geometry(
            curve_samples,
            preferred_offset_axis,
            offsets_mm,
            support_mm=support_mm,
        )
        return curves

    @classmethod
    def _offset_curve_geometry(
        cls,
        curve_samples: np.ndarray,
        preferred_offset_axis: np.ndarray,
        offsets_mm: list[float],
        *,
        support_mm: float | None = None,
    ) -> tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
        """用同一单位方向场生成所有 offset 曲线和首选曲面法向。

        方向场从中心曲线局部法向得到，并只在其沿弧长转动过快、无法支撑标准
        大范围堆叠时做最小物理尺度平滑。所有层共享该方向场，因此相邻中心面
        对应行距离始终等于 offset 差值；远层不会各自变直、形成扇形或留下厚层
        间隙。80 mm 设计范围内，增加更远层也不会改变已经存在的曲面。
        """

        normalized_offsets: list[float] = []
        for raw_offset_mm in offsets_mm:
            offset_mm = float(raw_offset_mm)
            if not math.isfinite(offset_mm):
                raise ValueError("offset_mm 必须是有限数值")
            normalized_offsets.append(offset_mm)

        requested_support = max(
            (abs(value) for value in normalized_offsets),
            default=0.0,
        )
        if support_mm is not None:
            support = float(support_mm)
            if not math.isfinite(support) or support < 0.0:
                raise ValueError("support_mm 必须是非负有限数值")
            requested_support = max(requested_support, support)
        parallel_support = max(
            cls._PARALLEL_OFFSET_SUPPORT_MM,
            requested_support,
        )
        offset_directions = cls._parallel_offset_directions(
            curve_samples,
            preferred_offset_axis,
            parallel_support,
        )
        curves: Dict[float, np.ndarray] = {}
        preferred_normals: Dict[float, np.ndarray] = {}
        for offset_mm in normalized_offsets:
            curves[offset_mm] = curve_samples + offset_directions * offset_mm
            preferred_normals[offset_mm] = offset_directions
        return curves, preferred_normals

    @classmethod
    def _parallel_offset_directions(
        cls,
        samples: np.ndarray,
        preferred_offset_axis: np.ndarray,
        support_mm: float,
    ) -> np.ndarray:
        """求能在给定范围内保持曲线前进方向的共享单位偏移场。"""

        _, _, base_directions = cls._curve_frame(
            samples,
            preferred_offset_axis,
        )
        if len(samples) < 3:
            return base_directions
        arc_steps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        valid_steps = arc_steps[np.isfinite(arc_steps) & (arc_steps > 1e-8)]
        if len(valid_steps) == 0:
            return base_directions

        support = float(support_mm)
        if not math.isfinite(support) or support < 0.0:
            raise ValueError("support_mm 必须是非负有限数值")

        def smoothed_directions(
            sigma_mm: float,
        ) -> tuple[np.ndarray, float]:
            guide_curve = (
                samples
                if sigma_mm <= 1e-8
                else cls._smooth_curve_coordinates(samples, sigma_mm)
            )
            _, _, directions = cls._curve_frame(
                guide_curve,
                preferred_offset_axis,
            )
            direction_steps = np.linalg.norm(
                np.diff(directions, axis=0),
                axis=1,
            )
            change_ratios = np.divide(
                support * direction_steps,
                arc_steps,
                out=np.zeros_like(direction_steps),
                where=arc_steps > 1e-8,
            )
            finite_ratios = change_ratios[np.isfinite(change_ratios)]
            maximum_ratio = (
                float(np.max(finite_ratios))
                if len(finite_ratios)
                else 0.0
            )
            return directions, maximum_ratio

        directions, maximum_ratio = smoothed_directions(0.0)
        if maximum_ratio <= cls._MAX_DIRECTION_CHANGE_RATIO:
            return directions

        lower_sigma = 0.0
        upper_sigma = max(1.0, float(np.median(valid_steps)))
        curve_length = float(np.sum(valid_steps))
        while maximum_ratio > cls._MAX_DIRECTION_CHANGE_RATIO:
            directions, maximum_ratio = smoothed_directions(upper_sigma)
            if maximum_ratio <= cls._MAX_DIRECTION_CHANGE_RATIO:
                break
            lower_sigma = upper_sigma
            upper_sigma *= 2.0
            if upper_sigma > max(curve_length * 2.0, 1.0):
                raise ValueError("CPR 中心曲线无法生成稳定的等距偏移方向场")

        for _ in range(cls._DIRECTION_SMOOTHING_SEARCH_STEPS):
            middle_sigma = (lower_sigma + upper_sigma) / 2.0
            candidate, candidate_ratio = smoothed_directions(middle_sigma)
            if candidate_ratio <= cls._MAX_DIRECTION_CHANGE_RATIO:
                upper_sigma = middle_sigma
                directions = candidate
            else:
                lower_sigma = middle_sigma
        return directions

    @staticmethod
    def _smooth_curve_coordinates(
        samples: np.ndarray,
        sigma_mm: float,
    ) -> np.ndarray:
        """按物理尺度平滑曲线，同时精确保留首末端点。"""

        sigma = float(sigma_mm)
        if not math.isfinite(sigma) or sigma <= 1e-8:
            return samples.copy()
        arc_steps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        valid_steps = arc_steps[np.isfinite(arc_steps) & (arc_steps > 1e-8)]
        if len(valid_steps) == 0:
            return samples.copy()

        sample_step = float(np.median(valid_steps))
        half_width = max(1, int(math.ceil(3.0 * sigma / sample_step)))
        positions = np.arange(-half_width, half_width + 1, dtype=float)
        positions *= sample_step
        kernel = np.exp(-0.5 * np.square(positions / sigma))
        kernel /= float(np.sum(kernel))
        padded = np.pad(
            samples,
            ((half_width, half_width), (0, 0)),
            mode="edge",
        )
        smoothed = np.stack([
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(samples.shape[1])
        ], axis=1)

        cumulative = np.concatenate(([0.0], np.cumsum(arc_steps)))
        if cumulative[-1] <= 1e-8:
            return samples.copy()
        ratios = (cumulative / cumulative[-1])[:, np.newaxis]
        smoothed += (
            (1.0 - ratios) * (samples[0] - smoothed[0])
            + ratios * (samples[-1] - smoothed[-1])
        )
        return smoothed

    @classmethod
    def _surface_frame(
        cls,
        samples: np.ndarray,
        preferred_normals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """为一张 offset 曲面生成逐行正交采样坐标系。"""

        tangents = cls._curve_tangents(samples)
        rows = np.empty_like(tangents)
        normals = np.empty_like(tangents)
        previous_row: np.ndarray | None = None
        for index, (tangent, preferred) in enumerate(
            zip(tangents, preferred_normals)
        ):
            row = np.cross(tangent, preferred)
            row_length = float(np.linalg.norm(row))
            if not math.isfinite(row_length) or row_length <= 1e-8:
                if previous_row is None:
                    row = cls._stable_perpendicular(tangent)
                else:
                    row = previous_row - np.dot(previous_row, tangent) * tangent
                    row_length = float(np.linalg.norm(row))
                    if not math.isfinite(row_length) or row_length <= 1e-8:
                        row = cls._stable_perpendicular(tangent)
            row = cls._normalize(row)
            normal = cls._normalize(np.cross(row, tangent))
            if float(np.dot(normal, preferred)) < 0.0:
                row = -row
                normal = -normal
            rows[index] = row
            normals[index] = normal
            previous_row = row
        return tangents, rows, normals

    @staticmethod
    def _maximum_offset_extent_mm(plan: Dict[str, Any]) -> float:
        """返回全部层中心和半层厚占用的最大绝对偏移。"""

        offsets: list[float] = []
        for raw_value in plan.get("offsets_mm") or []:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                offsets.append(abs(value))
        half_thickness = float(plan.get("slice_thickness", 0.0))
        if not math.isfinite(half_thickness) or half_thickness < 0.0:
            raise ValueError("slice_thickness 必须是非负有限数值")
        half_thickness /= 2.0
        return max(offsets, default=0.0) + half_thickness

    @staticmethod
    def _stable_perpendicular(direction: np.ndarray) -> np.ndarray:
        """方向与首选轴平行时，选取数值最稳定的正交单位向量。"""

        basis = np.eye(3, dtype=float)[int(np.argmin(np.abs(direction)))]
        candidate = basis - np.dot(basis, direction) * direction
        return CurvedStackResampler._normalize(candidate)

    def _sample_curved_surface_slab(
        self,
        image: sitk.Image,
        source: np.ndarray,
        curve_samples: np.ndarray,
        in_plane_directions: np.ndarray,
        offset_directions: np.ndarray,
        in_plane_offsets: np.ndarray,
        slab_offsets: list[float],
    ) -> np.ndarray:
        """采样一个 CPR 厚层 slab，并对多个子曲面做算术平均。"""

        rows = len(curve_samples)
        columns = len(in_plane_offsets)
        accumulated = np.zeros((rows, columns), dtype=np.float32)
        surface_points = (
            curve_samples[:, np.newaxis, :]
            + in_plane_directions[:, np.newaxis, :]
            * in_plane_offsets[np.newaxis, :, np.newaxis]
        )
        for slab_offset in slab_offsets:
            points = (
                surface_points
                + offset_directions[:, np.newaxis, :] * slab_offset
            )
            accumulated += self._sample_linear_grid(image, source, points)
        return accumulated / float(len(slab_offsets))

    @staticmethod
    def _slab_offsets(image: sitk.Image, thickness: float) -> list[float]:
        """按输入最小体素间距在 CPR 层厚内生成居中的子曲面偏移。"""

        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("slice_thickness 必须是大于 0 的有限数值")
        source_spacing = min(float(value) for value in image.GetSpacing())
        sample_step = min(source_spacing / 2.0, 0.25)
        sample_count = max(1, int(math.ceil(thickness / sample_step)))
        sample_step = thickness / sample_count
        return [
            -thickness / 2.0 + sample_step / 2.0 + index * sample_step
            for index in range(sample_count)
        ]

    @staticmethod
    def _sample_linear(
        image: sitk.Image,
        source: np.ndarray,
        point: np.ndarray,
    ) -> float:
        """用输入图像物理坐标执行三线性插值，越界返回空气值。"""

        index_xyz = image.TransformPhysicalPointToContinuousIndex(
            tuple(float(value) for value in point)
        )
        x, y, z = (float(value) for value in index_xyz)
        size_x, size_y, size_z = image.GetSize()
        if x < 0 or y < 0 or z < 0 or x > size_x - 1 or y > size_y - 1 or z > size_z - 1:
            return -1024.0

        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        z0 = int(math.floor(z))
        x1 = min(x0 + 1, size_x - 1)
        y1 = min(y0 + 1, size_y - 1)
        z1 = min(z0 + 1, size_z - 1)
        dx = x - x0
        dy = y - y0
        dz = z - z0

        c000 = float(source[z0, y0, x0])
        c100 = float(source[z0, y0, x1])
        c010 = float(source[z0, y1, x0])
        c110 = float(source[z0, y1, x1])
        c001 = float(source[z1, y0, x0])
        c101 = float(source[z1, y0, x1])
        c011 = float(source[z1, y1, x0])
        c111 = float(source[z1, y1, x1])

        c00 = c000 * (1.0 - dx) + c100 * dx
        c10 = c010 * (1.0 - dx) + c110 * dx
        c01 = c001 * (1.0 - dx) + c101 * dx
        c11 = c011 * (1.0 - dx) + c111 * dx
        c0 = c00 * (1.0 - dy) + c10 * dy
        c1 = c01 * (1.0 - dy) + c11 * dy
        return c0 * (1.0 - dz) + c1 * dz

    @staticmethod
    def _sample_linear_grid(
        image: sitk.Image,
        source: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """对整张 CPR 曲面执行批量三线性插值，越界点返回空气值。"""

        origin = np.asarray(image.GetOrigin(), dtype=float)
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        inverse_direction = np.linalg.inv(direction)
        continuous = ((points - origin) @ inverse_direction.T) / spacing
        x = continuous[..., 0]
        y = continuous[..., 1]
        z = continuous[..., 2]
        size_x, size_y, size_z = image.GetSize()
        # 只吸收矩阵求逆和物理坐标换算的机器尾差，不接纳真实越界坐标。
        coordinate_scale = max(
            1.0,
            float(max(size_x, size_y, size_z)),
            float(np.max(np.abs(origin / spacing))),
        )
        coordinate_tolerance = (
            np.finfo(np.float64).eps
            * 64.0
            * max(1.0, float(np.linalg.cond(direction)))
            * coordinate_scale
        )
        valid = (
            (x >= -coordinate_tolerance)
            & (y >= -coordinate_tolerance)
            & (z >= -coordinate_tolerance)
            & (x <= size_x - 1 + coordinate_tolerance)
            & (y <= size_y - 1 + coordinate_tolerance)
            & (z <= size_z - 1 + coordinate_tolerance)
        )

        x0 = np.floor(np.clip(x, 0.0, size_x - 1)).astype(np.intp)
        y0 = np.floor(np.clip(y, 0.0, size_y - 1)).astype(np.intp)
        z0 = np.floor(np.clip(z, 0.0, size_z - 1)).astype(np.intp)
        x1 = np.minimum(x0 + 1, size_x - 1)
        y1 = np.minimum(y0 + 1, size_y - 1)
        z1 = np.minimum(z0 + 1, size_z - 1)
        dx = np.clip(x - x0, 0.0, 1.0)
        dy = np.clip(y - y0, 0.0, 1.0)
        dz = np.clip(z - z0, 0.0, 1.0)

        c000 = source[z0, y0, x0].astype(np.float32, copy=False)
        c100 = source[z0, y0, x1].astype(np.float32, copy=False)
        c010 = source[z0, y1, x0].astype(np.float32, copy=False)
        c110 = source[z0, y1, x1].astype(np.float32, copy=False)
        c001 = source[z1, y0, x0].astype(np.float32, copy=False)
        c101 = source[z1, y0, x1].astype(np.float32, copy=False)
        c011 = source[z1, y1, x0].astype(np.float32, copy=False)
        c111 = source[z1, y1, x1].astype(np.float32, copy=False)

        c00 = c000 * (1.0 - dx) + c100 * dx
        c10 = c010 * (1.0 - dx) + c110 * dx
        c01 = c001 * (1.0 - dx) + c101 * dx
        c11 = c011 * (1.0 - dx) + c111 * dx
        c0 = c00 * (1.0 - dy) + c10 * dy
        c1 = c01 * (1.0 - dy) + c11 * dy
        sampled = c0 * (1.0 - dz) + c1 * dz
        return np.where(valid, sampled, -1024.0).astype(np.float32, copy=False)

    @staticmethod
    def _point(value: Dict[str, float]) -> np.ndarray:
        """按 x/y/z 顺序读取三维点或方向。"""

        return np.asarray([
            float(value["x"]),
            float(value["y"]),
            float(value["z"]),
        ])

    @staticmethod
    def _normalize(value: np.ndarray) -> np.ndarray:
        """归一化三维方向并拒绝零长度方向。"""

        length = float(np.linalg.norm(value))
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("CPR 方向向量长度无效")
        return value / length

    @staticmethod
    def _direction(
        row_dir: np.ndarray,
        col_dir: np.ndarray,
        normal: np.ndarray,
    ) -> tuple[float, ...]:
        """按 SimpleITK 矩阵布局组装输出图像方向。"""

        return (
            float(row_dir[0]), float(col_dir[0]), float(normal[0]),
            float(row_dir[1]), float(col_dir[1]), float(normal[1]),
            float(row_dir[2]), float(col_dir[2]), float(normal[2]),
        )
