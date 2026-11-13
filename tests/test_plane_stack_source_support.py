"""厚层端点只能平均真实源采样，避免越界补空气遮黑有效组织。"""

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pydicom
import SimpleITK as sitk

from src.reconstruction.plane_stack import PlaneStackResampler
from src.stages.stage_reconstruction import ReconstructionStage


def _point(values):
    return dict(zip("xyz", map(float, values)))


def _geometry(*, center=(4, 2, 3), angle=0, thickness=4, size=(9, 5)):
    radians = math.radians(angle)
    row = (math.cos(radians), 0, math.sin(radians))
    col = (0, 1, 0)
    normal = (-math.sin(radians), 0, math.cos(radians))
    plan = {
        "extent_mm": {"width": size[0] - 1, "height": size[1] - 1},
        "output_spacing_mm": {"x": 1.0, "y": 1.0},
        "slice_thickness": thickness,
    }
    item = {
        "center": _point(center), "row_dir": _point(row),
        "col_dir": _point(col), "normal": _point(normal),
    }
    first = np.asarray(center) - np.asarray(row) * (size[0] - 1) / 2
    first -= np.asarray(col) * (size[1] - 1) / 2
    sample = {
        "first_pixel": tuple(first), "normal": normal,
        "row_dir": row, "col_dir": col, "columns": size[0], "rows": size[1],
        "spacing_x": 1.0, "spacing_y": 1.0,
    }
    return plan, item, sample


def _support_image(image):
    """小型测试 oracle：独立常量源图表示 ITK 实际插值支持。"""
    support = sitk.GetImageFromArray(np.ones(tuple(reversed(image.GetSize())), np.uint8))
    support.CopyInformation(image)
    return support


class PlaneStackSourceSupportTest(unittest.TestCase):
    def setUp(self):
        self.sampler = PlaneStackResampler()

    def test_axis_endpoint_preserves_tissue_and_zero_support_keeps_air(self):
        image = sitk.GetImageFromArray(np.full((4, 5, 9), 100, np.int16))
        for center_z, expected in ((-0.25, 100), (4.25, 100), (-3, -1024), (7, -1024)):
            with self.subTest(center_z=center_z):
                plan, item, _ = _geometry(center=(4, 2, center_z), thickness=2)
                actual = sitk.GetArrayFromImage(self.sampler.sample_slice(image, plan, item))
                np.testing.assert_array_equal(actual, expected)

    def test_axis_partial_slab_includes_real_air_with_its_actual_thickness(self):
        values = np.full((3, 5, 9), 300, np.int16)
        values[0] = -1024
        image = sitk.GetImageFromArray(values)
        plan, item, _ = _geometry(center=(4, 2, -0.25), thickness=2)
        actual = sitk.GetArrayFromImage(self.sampler.sample_slice(image, plan, item))
        # [-1.25,.75] 覆盖源第零层1mm、第一层.25mm，不能丢弃真实空气。
        np.testing.assert_allclose(actual, (-1024 + 300 * 0.25) / 1.25, atol=1e-4)

    def test_oblique_scan_endpoint_keeps_constant_tissue_at_partly_supported_pixels(self):
        image = sitk.GetImageFromArray(np.full((5, 5, 9), 100, np.int16))
        plan, item, geometry = _geometry(angle=30, center=(4, 2, 5), thickness=4)
        support_image = _support_image(image)
        support_count = np.zeros((5, 9), np.int32)
        offsets = self.sampler._slab_offsets(image, 4)
        for offset in offsets:
            support_count += self.sampler._sample_plane(
                support_image, **geometry, normal_offset=offset, default_value=0,
            ).astype(bool)
        self.assertTrue(np.any((support_count > 0) & (support_count < len(offsets))))
        self.assertTrue(np.any(support_count == 0))
        actual = sitk.GetArrayFromImage(self.sampler.sample_slice(image, plan, item))[0]
        np.testing.assert_array_equal(actual[support_count > 0], 100)
        np.testing.assert_array_equal(actual[support_count == 0], -1024)

    def test_oblique_partial_slab_matches_independent_supported_mean_including_air(self):
        values = np.indices((5, 5, 9)).sum(axis=0).astype(np.int16) * 17
        values[0:2] = -1024
        image = sitk.GetImageFromArray(values)
        plan, item, geometry = _geometry(angle=30, center=(4, 2, 0), thickness=4)
        support_image = _support_image(image)
        numerator = np.zeros((5, 9), np.float32)
        denominator = np.zeros((5, 9), np.int32)
        for offset in self.sampler._slab_offsets(image, 4):
            supported = self.sampler._sample_plane(
                support_image, **geometry, normal_offset=offset, default_value=0,
            ) > 0
            plane = self.sampler._sample_plane(
                image, **geometry, normal_offset=offset, default_value=0,
            )
            numerator += np.where(supported, plane, 0)
            denominator += supported
        expected = np.full((5, 9), -1024, np.float32)
        np.divide(numerator, denominator, out=expected, where=denominator > 0)
        actual = sitk.GetArrayFromImage(self.sampler.sample_slice(image, plan, item))[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)

    def test_fully_supported_oblique_pixels_keep_original_float32_mean_exactly(self):
        image = sitk.GetImageFromArray(
            np.random.default_rng(34).integers(-1024, 2500, (15, 15, 15), dtype=np.int16),
        )
        plan, item, geometry = _geometry(
            angle=20, center=(7, 7, 7), thickness=3.7, size=(5, 5),
        )
        expected = np.mean([
            self.sampler._sample_plane(image, **geometry, normal_offset=offset)
            for offset in self.sampler._slab_offsets(image, 3.7)
        ], axis=0)
        actual = sitk.GetArrayFromImage(self.sampler.sample_slice(image, plan, item))[0]
        np.testing.assert_array_equal(actual, expected)

    def test_fully_supported_axis_pixels_keep_original_float32_weighting_exactly(self):
        values = np.array([43, 157, -532, 1397, -87], np.int16)
        image = sitk.GetImageFromArray(np.broadcast_to(values[:, None, None], (5, 5, 9)).copy())
        plan, item, _ = _geometry(center=(4, 2, 2.1), thickness=3.4)
        expected = np.full((5, 9), -1024.0 * 3.4, np.float32)
        # slab[.4,3.8]：源层0/1/2/3/4分别重叠.1/1/1/1/.3毫米。
        for value, overlap in zip(values, (0.1, 1, 1, 1, 0.3)):
            expected += (np.full_like(expected, value) + 1024) * overlap
        expected /= 3.4
        actual = sitk.GetArrayFromImage(self.sampler.sample_slice(image, plan, item))[0]
        np.testing.assert_array_equal(actual, expected)

    def test_fully_supported_near_axis_slab_preserves_overlap_roundoff(self):
        image = sitk.GetImageFromArray(
            np.random.default_rng(23).integers(-1024, 2500, (13, 13, 13), dtype=np.int16),
        )
        plan, item, geometry = _geometry(
            angle=0.06, center=(6, 6, 6), thickness=3.4, size=(5, 5),
        )
        normal = np.array(geometry['normal'])
        self.assertIsNotNone(self.sampler._aligned_normal_axis(image, tuple(normal)))
        expected = np.full((5, 5), -1024 * 3.4, np.float32)
        total_overlap = 0.0
        for z in range(13):
            depth = float((np.array((0, 0, z)) - (6, 6, 6)) @ normal)
            overlap = min(depth + 0.5, 1.7) - max(depth - 0.5, -1.7)
            if overlap > 0:
                plane = self.sampler._sample_plane(image, **geometry, normal_offset=depth)
                expected += (plane + 1024) * overlap
                total_overlap += overlap
        self.assertGreater(total_overlap, 3.4)
        expected /= 3.4
        actual = sitk.GetArrayFromImage(self.sampler.sample_slice(image, plan, item))[0]
        np.testing.assert_array_equal(actual, expected)

    def test_float_nan_and_infinity_are_supported_source_values(self):
        for value in (float("nan"), float("inf"), -float("inf"), -1024.0):
            with self.subTest(value=value):
                image = sitk.GetImageFromArray(np.full((3, 5, 9), value, np.float32))
                _, _, geometry = _geometry(center=(4, 2, -0.25), thickness=2)
                geometry["normal_offset"] = 0.0
                plane, supported = self.sampler._sample_supported_plane(image, **geometry)
                expected = self.sampler._sample_plane(image, **geometry)
                self.assertTrue(np.all(supported))
                np.testing.assert_array_equal(plane, expected)
                geometry["normal_offset"] = -1.0
                outside, supported = self.sampler._sample_supported_plane(image, **geometry)
                self.assertFalse(np.any(supported))
                np.testing.assert_array_equal(outside, -1024)

    def test_integer_support_requires_one_plane_sample_and_no_source_size_mask(self):
        image = sitk.GetImageFromArray(np.full((7, 9, 11), -1024, np.int16))
        _, _, geometry = _geometry(center=(4, 2, -0.75), angle=30)
        with mock.patch.object(
            self.sampler, "_sample_plane", wraps=self.sampler._sample_plane,
        ) as sample, mock.patch("src.reconstruction.plane_stack.sitk.Image", wraps=sitk.Image) as factory:
            self.sampler._sample_supported_plane(image, **geometry, normal_offset=0)
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(factory.call_count, 0)

    def test_stage7_preserves_partial_slab_tissue_in_all_planes_and_windows(self):
        """真实采样、上下裁空、并行双窗和 DICOM 解码不能再次把边缘组织压黑。"""
        yaw = math.radians(17)
        direction = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]) @ np.diag([-1.0, 1.0, -1.0])
        for axis, plane in enumerate(('矢状位', '冠状位', '横状位')):
            for angle in (0, 30):
                for dtype in (np.int16, np.float32):
                    with self.subTest(plane=plane, angle=angle, dtype=dtype):
                        image = sitk.GetImageFromArray(np.full((5, 7, 9), 100, dtype))
                        image.SetSpacing((0.7, 1.1, 1.4))
                        image.SetDirection(tuple(direction.ravel()))
                        image.SetOrigin((-11, 20, -33))
                        radians = math.radians(angle)
                        base_row = direction[:, (axis + 1) % 3]
                        col = direction[:, (axis + 2) % 3]
                        base_normal = direction[:, axis]
                        row = math.cos(radians) * base_row + math.sin(radians) * base_normal
                        normal = math.cos(radians) * base_normal - math.sin(radians) * base_row
                        index = (np.array(image.GetSize()) - 1) / 2
                        index[axis] = image.GetSize()[axis] - 1 + 0.6 / image.GetSpacing()[axis]
                        center = image.TransformContinuousIndexToPhysicalPoint(tuple(index))
                        item = {
                            'index': 1, 'center': _point(center), 'row_dir': _point(row),
                            'col_dir': _point(col), 'normal': _point(normal),
                        }
                        plan = {
                            'plan_id': 'partial_source_support', 'plane': plane,
                            'slice_part': '颈椎', 'slice_type': 'MPR',
                            'geometry_type': 'plane_stack', 'slices': [item],
                            'slice_thickness': 4, 'slice_spacing': 4,
                            'extent_mm': {'width': 10, 'height': 12},
                            'output_spacing_mm': {'x': 1.0, 'y': 1.0},
                            'window_presets': ['骨窗', '软组织窗'],
                        }
                        with tempfile.TemporaryDirectory() as folder:
                            ReconstructionStage({'parallel': True})._execute_plan(
                                image, plan, Path(folder), '1.2.3', {},
                            )
                            datasets = [
                                pydicom.dcmread(Path(folder) / plane / window / '1.dcm')
                                for window in plan['window_presets']
                            ]
                        data = datasets[0]
                        yy, xx = np.indices((data.Rows, data.Columns))
                        spacing_y, spacing_x = map(float, data.PixelSpacing)
                        pixels = (
                            np.array(data.ImagePositionPatient, float)
                            + xx[..., None] * row * spacing_x
                            + yy[..., None] * col * spacing_y
                        )
                        # 独立按源连续索引检查16个子平面的支持，不使用被测采样器
                        # 或裁空结果作为期望 HU；源内恒为100HU，部分支持也应保持100。
                        source_steps = direction * image.GetSpacing()
                        support = np.zeros(xx.shape, np.uint8)
                        for offset in np.linspace(-1.875, 1.875, 16):
                            indices = (pixels + normal * offset - image.GetOrigin()) @ np.linalg.inv(source_steps).T
                            support += np.all(
                                (indices >= -0.5) & (indices < np.array(image.GetSize()) - 0.5),
                                axis=-1,
                            )
                        self.assertTrue(np.any((support > 0) & (support < 16)))
                        self.assertTrue(np.any(support == 0))
                        for dataset in datasets:
                            actual = dataset.pixel_array * float(dataset.RescaleSlope)
                            actual += float(dataset.RescaleIntercept)
                            # 各窗量化步长不同，允许半个 uint16 灰度步长的 HU 误差。
                            np.testing.assert_allclose(
                                actual[support > 0], 100, rtol=0,
                                atol=float(dataset.RescaleSlope) / 2 + 1e-3,
                            )
                            np.testing.assert_array_equal(dataset.pixel_array[support == 0], 0)
                            self.assertEqual(float(dataset.SliceThickness), 4)
                            self.assertEqual(float(dataset.SpacingBetweenSlices), 4)
                            self.assertEqual(int(dataset.InstanceNumber), 1)
                            for field in (
                                'Rows', 'Columns', 'PixelSpacing',
                                'ImagePositionPatient', 'ImageOrientationPatient',
                            ):
                                self.assertEqual(getattr(dataset, field), getattr(data, field))


if __name__ == "__main__":
    unittest.main()
