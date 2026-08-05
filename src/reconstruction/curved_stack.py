"""曲面堆叠（curved_stack）CPR 重采样。"""

import math
import time
from typing import Any, Dict

import numpy as np
import SimpleITK as sitk

from src.utils import DebugLog


class CurvedStackResampler:
    """沿中心锚点折线和计划堆叠偏移生成单张 CPR 图像。"""

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
        tangents = self._curve_tangents(curve_samples)

        offset_axis = self._normalize(self._point(plan["offset_axis"]))
        in_plane_directions = self._in_plane_directions(
            tangents,
            offset_axis,
        )
        offset_mm = float(planned_slice.get("offset_mm", 0.0))
        in_plane_offsets = (
            np.arange(columns, dtype=float) - (columns - 1) / 2.0
        ) * spacing_x

        source = sitk.GetArrayViewFromImage(image)
        slice_thickness = float(plan.get("slice_thickness", 1.0))
        slab_offsets = self._slab_offsets(image, slice_thickness)
        sampled = self._sample_curved_surface_slab(
            image=image,
            source=source,
            curve_samples=curve_samples,
            in_plane_directions=in_plane_directions,
            offset_axis=offset_axis,
            base_offset_mm=offset_mm,
            in_plane_offsets=in_plane_offsets,
            slab_offsets=slab_offsets,
        )

        output = sitk.GetImageFromArray(sampled[np.newaxis, ...])
        output.SetSpacing((spacing_x, spacing_y, 1.0))
        origin = curve_samples[0] + offset_axis * offset_mm
        first_in_plane = in_plane_directions[0]
        if columns > 1:
            origin = origin + first_in_plane * in_plane_offsets[0]
        output.SetOrigin(tuple(float(value) for value in origin))
        output_tangent = self._normalize(
            np.cross(offset_axis, first_in_plane)
        )
        output.SetDirection(
            self._direction(
                first_in_plane,
                output_tangent,
                offset_axis,
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

    @staticmethod
    def _in_plane_directions(
        tangents: np.ndarray,
        offset_axis: np.ndarray,
    ) -> np.ndarray:
        """计算每行平面内方向，局部退化时借用最近的有效方向。"""

        candidates = np.cross(tangents, offset_axis)
        lengths = np.linalg.norm(candidates, axis=1)
        valid_indices = np.flatnonzero(np.isfinite(lengths) & (lengths > 1e-8))
        if len(valid_indices) == 0:
            raise ValueError("CPR 方向向量长度无效")

        directions = np.empty_like(candidates)
        for index in range(len(candidates)):
            if lengths[index] > 1e-8 and math.isfinite(float(lengths[index])):
                directions[index] = candidates[index] / lengths[index]
                continue
            nearest = int(valid_indices[np.argmin(np.abs(valid_indices - index))])
            directions[index] = candidates[nearest] / lengths[nearest]
        return directions

    def _sample_curved_surface_slab(
        self,
        image: sitk.Image,
        source: np.ndarray,
        curve_samples: np.ndarray,
        in_plane_directions: np.ndarray,
        offset_axis: np.ndarray,
        base_offset_mm: float,
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
            offset_mm = base_offset_mm + slab_offset
            points = surface_points + offset_axis * offset_mm
            accumulated += self._sample_linear_grid(image, source, points)
        return accumulated / float(len(slab_offsets))

    @staticmethod
    def _slab_offsets(image: sitk.Image, thickness: float) -> list[float]:
        """按输入最小体素间距在 CPR 层厚内生成居中的子曲面偏移。"""

        if thickness <= 0.0:
            raise ValueError("slice_thickness 必须大于 0")
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
