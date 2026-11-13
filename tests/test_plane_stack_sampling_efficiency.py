"""普通 MPR 优化须保留逐层重叠、源支持、像素和图像几何。"""

import math
from unittest import mock

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.plane_stack import PlaneStackResampler


class _FullAxisReferenceSampler(PlaneStackResampler):
    """冻结优化前的全轴扫描与 reference image 重采样方式。"""

    @staticmethod
    def _axis_slab_indices(image, center, normal, thickness, axis_info):
        return range(image.GetSize()[axis_info[0]])

    @staticmethod
    def _sample_plane(
        image, first_pixel, normal, row_dir, col_dir, columns, rows,
        spacing_x, spacing_y, normal_offset, default_value=-1024.0,
    ):
        reference = sitk.Image(columns, rows, 1, sitk.sitkFloat32)
        reference.SetSpacing((spacing_x, spacing_y, 1.0))
        reference.SetOrigin(tuple(first_pixel[i] + normal[i] * normal_offset for i in range(3)))
        reference.SetDirection(PlaneStackResampler._direction(row_dir, col_dir, normal))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        return sitk.GetArrayFromImage(resampler.Execute(image))[0].astype(np.float32)


def _rotated_direction():
    angle = math.radians(23)
    return np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, -1],
    ])


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (-1, 1))
def test_axis_candidate_range_includes_every_positive_original_overlap(axis, sign):
    """独立遍历 ITK 物理点作 oracle，覆盖近轴、反向与大原点消减。"""
    image = sitk.Image(37, 41, 43, sitk.sitkInt16)
    image.SetSpacing((0.17, 1.3, 3.7))
    direction = _rotated_direction()
    image.SetDirection(tuple(direction.ravel()))
    for origin_size in (0.0, 1e9, 1e15, 1e18):
        image.SetOrigin((origin_size, -origin_size, origin_size))
        for angle in (0.0, math.acos(1 - 0.999e-6)):
            normal = sign * (math.cos(angle) * direction[:, axis]
                             + math.sin(angle) * direction[:, (axis + 1) % 3])
            axis_info = PlaneStackResampler._aligned_normal_axis(image, tuple(normal))
            assert axis_info is not None
            for depth, thickness in ((-8, 0.02), (-0.5, 0.02), (0.0, 2.6),
                                     (18.5, 0.17), (42.5, 3.7), (65, 0.4)):
                index = [7.5, 12.5, 22.5]
                index[axis] = depth
                center = image.TransformContinuousIndexToPhysicalPoint(index)
                candidates = PlaneStackResampler._axis_slab_indices(
                    image, center, tuple(normal), thickness, axis_info,
                )
                overlapping = []
                for source_index in range(image.GetSize()[axis]):
                    voxel_index = [0, 0, 0]
                    voxel_index[axis] = source_index
                    point = image.TransformIndexToPhysicalPoint(voxel_index)
                    offset = sum((point[i] - center[i]) * normal[i] for i in range(3))
                    overlap = min(offset + image.GetSpacing()[axis] / 2, thickness / 2)
                    overlap -= max(offset - image.GetSpacing()[axis] / 2, -thickness / 2)
                    if overlap > 0:
                        overlapping.append(source_index)
                assert all(index in candidates for index in overlapping)
                assert list(candidates) == sorted(candidates)


def test_axis_candidate_range_is_small_for_thin_slab_in_deep_volume():
    image = sitk.Image(1, 1, 4096, sitk.sitkInt16)
    image.SetSpacing((0.7, 0.7, 0.8))
    indices = PlaneStackResampler._axis_slab_indices(
        image, (0, 0, 1600.1), (0, 0, 1), 2.0, (2, 1.0),
    )
    assert 2000 in indices
    assert len(indices) <= 8


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("angle,sign", [(0, 1), (0.06, -1), (27, 1)])
@pytest.mark.parametrize("floating", [False, True])
def test_plane_stack_pixels_and_geometry_match_original_sampler(axis, angle, sign, floating):
    values = np.random.default_rng(104).integers(-1024, 2500, (11, 9, 13), dtype=np.int16)
    if floating:
        values = values.astype(np.float32)
        values[4, 3, 4:7] = (np.nan, np.inf, -np.inf)
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.7, 1.1, 1.6))
    image.SetOrigin((1e9 + 0.2, -1e9 - 0.6, 1e9 + 0.1))
    direction = _rotated_direction()
    image.SetDirection(tuple(direction.ravel()))
    radians = math.radians(angle)
    row = math.cos(radians) * direction[:, (axis + 1) % 3] + math.sin(radians) * direction[:, axis]
    normal = sign * (math.cos(radians) * direction[:, axis]
                     - math.sin(radians) * direction[:, (axis + 1) % 3])
    col = direction[:, (axis + 2) % 3]
    # 整数例跨越源端点；浮点例穿过 NaN/Inf，画布还覆盖面内源边界。
    center_index = np.array((6.0, 4.0, 5.0))
    center_index[axis] = 4.25 if floating else 0.25
    center = image.TransformContinuousIndexToPhysicalPoint(tuple(center_index))
    item = {name: dict(zip("xyz", vector)) for name, vector in (
        ("center", center), ("row_dir", row), ("col_dir", col), ("normal", normal),
    )}
    plan = {"extent_mm": {"width": 10, "height": 8},
            "output_spacing_mm": {"x": 0.8, "y": 1.2}, "slice_thickness": 2.7}
    with np.errstate(invalid="ignore"):
        expected = _FullAxisReferenceSampler().sample_slice(image, plan, item)
        actual = PlaneStackResampler().sample_slice(image, plan, item)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(expected))
    assert actual.GetSize() == expected.GetSize()
    assert actual.GetSpacing() == expected.GetSpacing()
    assert actual.GetOrigin() == expected.GetOrigin()
    assert actual.GetDirection() == expected.GetDirection()


def test_static_plane_sampler_has_no_reference_allocation_and_returns_independent_writable_float32():
    image = sitk.GetImageFromArray(np.full((3, 5, 7), 108, np.int16))
    geometry = dict(first_pixel=(0, 0, 1), normal=(0, 0, 1), row_dir=(1, 0, 0),
                    col_dir=(0, 1, 0), columns=7, rows=5, spacing_x=1, spacing_y=1,
                    normal_offset=0)
    with mock.patch("src.reconstruction.plane_stack.sitk.Image", wraps=sitk.Image) as factory:
        first = PlaneStackResampler._sample_plane(image, **geometry)
    assert factory.call_count == 0
    assert first.dtype == np.float32
    assert first.flags.writeable
    first[:] = -17
    np.testing.assert_array_equal(PlaneStackResampler._sample_plane(image, **geometry), 108)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), 108)
