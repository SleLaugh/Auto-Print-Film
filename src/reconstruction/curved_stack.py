"""曲面堆叠（curved_stack）CPR 重采样。"""

import math
from typing import Any, Dict

import numpy as np
import SimpleITK as sitk


class CurvedStackResampler:
    """沿中心曲线和左右偏移生成单张 CPR 图像。"""

    def sample_slice(
        self,
        image: sitk.Image,
        plan: Dict[str, Any],
        planned_slice: Dict[str, Any],
    ) -> sitk.Image:
        """采样一张 CPR HU 图，返回 size=(cols, rows, 1) 的图像。"""

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
        offset_mm = float(planned_slice.get("offset_mm", 0.0))
        ap_offsets = (
            np.arange(columns, dtype=float) - (columns - 1) / 2.0
        ) * spacing_x

        source = sitk.GetArrayViewFromImage(image)
        sampled = np.empty((rows, columns), dtype=np.float32)
        for row in range(rows):
            tangent = tangents[row]
            ap_dir = self._normalize(np.cross(tangent, offset_axis))
            row_center = curve_samples[row] + offset_axis * offset_mm
            for column, ap_offset in enumerate(ap_offsets):
                point = row_center + ap_dir * ap_offset
                sampled[row, column] = self._sample_linear(image, source, point)

        output = sitk.GetImageFromArray(sampled[np.newaxis, ...])
        output.SetSpacing((spacing_x, spacing_y, 1.0))
        origin = curve_samples[0] + offset_axis * offset_mm
        if columns > 1:
            first_ap = self._normalize(np.cross(tangents[0], offset_axis))
            origin = origin + first_ap * ap_offsets[0]
        output.SetOrigin(tuple(float(value) for value in origin))
        output.SetDirection(
            self._direction(
                self._normalize(np.cross(tangents[0], offset_axis)),
                tangents[0],
                offset_axis,
            )
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
