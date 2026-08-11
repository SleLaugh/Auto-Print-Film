"""普通平面堆叠（plane_stack）重采样。"""

import math
from typing import Any, Dict, Tuple

import numpy as np
import SimpleITK as sitk


class PlaneStackResampler:
    """按照第六阶段给出的平面中心和方向采样 2D MPR 图像。"""

    def sample_slice(
        self,
        image: sitk.Image,
        plan: Dict[str, Any],
        planned_slice: Dict[str, Any],
    ) -> sitk.Image:
        """采样一张带层厚的 HU 平面，返回 size=(cols, rows, 1) 的图像。"""

        extent = plan["extent_mm"]
        output_spacing = plan["output_spacing_mm"]
        spacing_x = float(output_spacing["x"])
        spacing_y = float(output_spacing["y"])
        if spacing_x <= 0 or spacing_y <= 0:
            raise ValueError("output_spacing_mm 的 x/y 必须大于 0")

        columns = max(1, int(round(float(extent["width"]) / spacing_x)) + 1)
        rows = max(1, int(round(float(extent["height"]) / spacing_y)) + 1)

        center = self._point(planned_slice["center"])
        normal = self._normalize(self._point(planned_slice["normal"]))
        row_dir = self._normalize(self._point(planned_slice["row_dir"]))
        col_dir = self._normalize(self._point(planned_slice["col_dir"]))
        first_pixel = self._sub(
            self._sub(
                center,
                self._scale(row_dir, (columns - 1) * spacing_x / 2.0),
            ),
            self._scale(col_dir, (rows - 1) * spacing_y / 2.0),
        )

        thickness = float(plan.get("slice_thickness", 1.0))
        if thickness <= 0:
            raise ValueError("slice_thickness 必须大于 0")
        axis_info = self._aligned_normal_axis(image, normal)
        if axis_info is not None:
            averaged = self._sample_axis_aligned_slab(
                image=image,
                center=center,
                first_pixel=first_pixel,
                normal=normal,
                row_dir=row_dir,
                col_dir=col_dir,
                columns=columns,
                rows=rows,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                thickness=thickness,
                axis_info=axis_info,
            )
        else:
            offsets = self._slab_offsets(image, thickness)
            plane_arrays = [
                self._sample_plane(
                    image=image,
                    first_pixel=first_pixel,
                    normal=normal,
                    row_dir=row_dir,
                    col_dir=col_dir,
                    columns=columns,
                    rows=rows,
                    spacing_x=spacing_x,
                    spacing_y=spacing_y,
                    normal_offset=offset,
                )
                for offset in offsets
            ]
            averaged = np.mean(plane_arrays, axis=0)
        output = sitk.GetImageFromArray(
            np.asarray(averaged, dtype=np.float32)[np.newaxis, ...]
        )
        output.SetSpacing((spacing_x, spacing_y, 1.0))
        output.SetOrigin(tuple(first_pixel))
        output.SetDirection(self._direction(row_dir, col_dir, normal))
        return output

    @staticmethod
    def _slab_offsets(image: sitk.Image, thickness: float) -> list[float]:
        """按输入最小体素间距在层厚内生成居中的采样平面。"""

        source_spacing = min(float(value) for value in image.GetSpacing())
        sample_step = min(source_spacing / 2.0, 0.25)
        sample_count = max(1, int(math.ceil(thickness / sample_step)))
        sample_step = thickness / sample_count
        return [
            -thickness / 2.0 + sample_step / 2.0 + index * sample_step
            for index in range(sample_count)
        ]

    @staticmethod
    def _aligned_normal_axis(
        image: sitk.Image,
        normal: Tuple[float, float, float],
    ) -> Tuple[int, float] | None:
        """识别切片法向是否与某个输入图像轴平行，并返回轴号及方向符号。"""

        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        normal_array = np.asarray(normal, dtype=float)
        for axis in range(3):
            axis_direction = direction[:, axis]
            dot = float(np.dot(axis_direction, normal_array))
            if abs(abs(dot) - 1.0) <= 1e-6:
                return axis, dot
        return None

    def _sample_axis_aligned_slab(
        self,
        image: sitk.Image,
        center: Tuple[float, float, float],
        first_pixel: Tuple[float, float, float],
        normal: Tuple[float, float, float],
        row_dir: Tuple[float, float, float],
        col_dir: Tuple[float, float, float],
        columns: int,
        rows: int,
        spacing_x: float,
        spacing_y: float,
        thickness: float,
        axis_info: Tuple[int, float],
    ) -> np.ndarray:
        """按原始层与目标 slab 的实际重叠厚度做精确轴向加权。"""

        axis, _ = axis_info
        source_spacing = float(image.GetSpacing()[axis])
        half_thickness = thickness / 2.0
        default_value = -1024.0
        weighted_sum = np.full(
            (rows, columns),
            default_value * thickness,
            dtype=np.float32,
        )

        for index in range(image.GetSize()[axis]):
            voxel_index = [0, 0, 0]
            voxel_index[axis] = index
            source_center = image.TransformIndexToPhysicalPoint(voxel_index)
            relative = tuple(
                source_center[component] - center[component]
                for component in range(3)
            )
            center_offset = sum(
                relative[component] * normal[component]
                for component in range(3)
            )
            source_start = center_offset - source_spacing / 2.0
            source_end = center_offset + source_spacing / 2.0
            overlap = max(
                0.0,
                min(source_end, half_thickness)
                - max(source_start, -half_thickness),
            )
            if overlap <= 0:
                continue

            plane = self._sample_plane(
                image=image,
                first_pixel=first_pixel,
                normal=normal,
                row_dir=row_dir,
                col_dir=col_dir,
                columns=columns,
                rows=rows,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                normal_offset=center_offset,
            )
            weighted_sum += (plane - default_value) * overlap

        return weighted_sum / thickness

    @staticmethod
    def _sample_plane(
        image: sitk.Image,
        first_pixel: Tuple[float, float, float],
        normal: Tuple[float, float, float],
        row_dir: Tuple[float, float, float],
        col_dir: Tuple[float, float, float],
        columns: int,
        rows: int,
        spacing_x: float,
        spacing_y: float,
        normal_offset: float,
    ) -> np.ndarray:
        """在给定法向偏移处重采样一个二维物理平面并返回浮点像素数组。"""

        origin = tuple(
            first_pixel[axis] + normal[axis] * normal_offset
            for axis in range(3)
        )
        reference = sitk.Image(columns, rows, 1, sitk.sitkFloat32)
        reference.SetSpacing((spacing_x, spacing_y, 1.0))
        reference.SetOrigin(origin)
        reference.SetDirection(PlaneStackResampler._direction(row_dir, col_dir, normal))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024.0)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        sampled = resampler.Execute(image)
        return sitk.GetArrayFromImage(sampled)[0].astype(np.float32)

    @staticmethod
    def _point(value: Dict[str, float]) -> Tuple[float, float, float]:
        """按 x/y/z 顺序把方向或物理点字典转换为三元组。"""

        return float(value["x"]), float(value["y"]), float(value["z"])

    @staticmethod
    def _normalize(value: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """把切片方向归一化为单位向量，并拒绝零长度方向。"""

        length = math.sqrt(sum(component * component for component in value))
        if length <= 0:
            raise ValueError("切片方向向量长度不能为 0")
        return tuple(component / length for component in value)

    @staticmethod
    def _scale(value: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
        """将三维向量的每个分量乘以指定缩放系数。"""

        return tuple(component * factor for component in value)

    @staticmethod
    def _sub(
        left: Tuple[float, float, float],
        right: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """计算两个三维物理点或向量的逐分量差。"""

        return tuple(left[index] - right[index] for index in range(3))

    @staticmethod
    def _direction(
        row_dir: Tuple[float, float, float],
        col_dir: Tuple[float, float, float],
        normal: Tuple[float, float, float],
    ) -> Tuple[float, ...]:
        """按 SimpleITK 矩阵布局组装行、列、法向三个物理方向。"""

        return (
            row_dir[0], col_dir[0], normal[0],
            row_dir[1], col_dir[1], normal[1],
            row_dir[2], col_dir[2], normal[2],
        )
