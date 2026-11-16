"""复用单个 slab 的 filter 必须保持二维采样、源支持及扩展调用行为。"""

from concurrent.futures import ThreadPoolExecutor
import math
from unittest import mock

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.plane_stack import PlaneStackResampler


class _FreshFilterSampler(PlaneStackResampler):
    """经原静态 API 为每个偏移创建 filter，作为逐平面精确对照。"""

    @staticmethod
    def _sample_plane(*args, **kwargs):
        return PlaneStackResampler._sample_plane(*args, **kwargs)


def _case(axis=2, angle=27, dtype=np.int16, origin_scale=0.0):
    values = np.random.default_rng(247).integers(-1024, 2500, (11, 9, 13), dtype=np.int16)
    values = values.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        values[3:7, 3:6, 4:7] = np.nan
        values[5, 4, 7:9] = (np.inf, -np.inf)
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.7, 1.1, 1.6))
    yaw = math.radians(23)
    direction = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, -1],
    ])
    image.SetDirection(tuple(direction.ravel()))
    image.SetOrigin((origin_scale + 0.2, -origin_scale - 0.6, origin_scale + 0.1))
    angle = math.radians(angle)
    row = math.cos(angle) * direction[:, (axis + 1) % 3] + math.sin(angle) * direction[:, axis]
    normal = math.cos(angle) * direction[:, axis] - math.sin(angle) * direction[:, (axis + 1) % 3]
    col = direction[:, (axis + 2) % 3]
    center = image.TransformContinuousIndexToPhysicalPoint((5.75, 3.5, 4.25))
    item = {name: dict(zip("xyz", vector)) for name, vector in (
        ("center", center), ("row_dir", row), ("col_dir", col), ("normal", normal),
    )}
    plan = {
        "extent_mm": {"width": 12, "height": 15},
        "output_spacing_mm": {"x": 0.8, "y": 1.2},
        "slice_thickness": 3.7,
    }
    return image, plan, item


def _assert_identical(actual, expected):
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(expected))
    assert actual.GetSize() == expected.GetSize()
    assert actual.GetSpacing() == expected.GetSpacing()
    assert actual.GetOrigin() == expected.GetOrigin()
    assert actual.GetDirection() == expected.GetDirection()
    assert actual.GetPixelID() == expected.GetPixelID()


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("angle", (0, 0.06, 27))
@pytest.mark.parametrize("dtype", (np.int16, np.float32, np.float64))
@pytest.mark.parametrize("origin_scale", (0.0, 1e9))
def test_filter_reuse_preserves_pixels_and_geometry(axis, angle, dtype, origin_scale):
    args = _case(axis, angle, dtype, origin_scale)
    with np.errstate(invalid="ignore"):
        expected = _FreshFilterSampler().sample_slice(*args)
        actual = PlaneStackResampler().sample_slice(*args)
    _assert_identical(actual, expected)


@pytest.mark.parametrize("angle", (0, 27))
def test_each_slice_uses_a_private_filter(angle):
    args = _case(angle=angle)
    sampler = PlaneStackResampler()
    constructor = sitk.ResampleImageFilter
    with mock.patch("src.reconstruction.plane_stack.sitk.ResampleImageFilter", wraps=constructor) as factory:
        first = sampler.sample_slice(*args)
        assert factory.call_count == 1
        second = sampler.sample_slice(*args)
        assert factory.call_count == 2
    _assert_identical(first, second)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_verification_default_and_source_support_do_not_leak_to_later_offsets(dtype):
    values = np.full((7, 9, 11), 100, dtype)
    values[2:5, 3:6, 3:6] = np.nan
    values[3, 4, 6:8] = (np.inf, -np.inf)
    image = sitk.GetImageFromArray(values)
    geometry = dict(first_pixel=(-1.0, -1.0, 0.0), normal=(0.0, 0.0, 1.0),
                    row_dir=(1.0, 0.0, 0.0), col_dir=(0.0, 1.0, 0.0),
                    columns=13, rows=11, spacing_x=1.0, spacing_y=1.0)
    sampler = PlaneStackResampler()
    sample = sampler._prepare_supported_plane_sampler(image, **geometry)
    for offset in (-2.0, 3.0, 1.0, 12.0, 4.25, 0.0):
        expected, expected_support = sampler._sample_supported_plane(
            image, **geometry, normal_offset=offset,
        )
        actual, actual_support = sample(offset)
        np.testing.assert_array_equal(actual_support, expected_support)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("angle", (0, 27))
@pytest.mark.parametrize("name", ("_sample_plane", "_sample_supported_plane"))
@pytest.mark.parametrize("class_patch", (False, True))
def test_patched_sampling_methods_are_honored(angle, name, class_patch):
    args = _case(angle=angle)
    sampler = PlaneStackResampler()
    expected = sampler.sample_slice(*args)
    target = PlaneStackResampler if class_patch else sampler
    if class_patch and name == "_sample_supported_plane":
        # autospec 保留普通实例方法绑定；静态方法和实例 patch 可直接 wraps。
        original = getattr(PlaneStackResampler, name)
        patcher = mock.patch.object(target, name, autospec=True, side_effect=original)
    else:
        patcher = mock.patch.object(target, name, wraps=getattr(sampler, name))
    with patcher as method:
        actual = sampler.sample_slice(*args)
    assert method.call_count > 1
    _assert_identical(actual, expected)


def test_subclass_supported_plane_override_is_honored():
    class CustomSampler(PlaneStackResampler):
        def _sample_supported_plane(self, image, **geometry):
            plane, supported = super()._sample_supported_plane(image, **geometry)
            plane[supported] = 321.0
            return plane, supported

    image, plan, item = _case(angle=27)
    actual = sitk.GetArrayFromImage(CustomSampler().sample_slice(image, plan, item))
    assert np.any(actual == 321.0)
    assert np.all(np.isin(actual, (-1024.0, 321.0)))


def test_concurrent_slices_on_one_sampler_keep_filters_and_geometry_private():
    cases = [_case(axis=axis, angle=angle, dtype=dtype) for axis, angle, dtype in (
        (0, 0, np.int16), (1, 27, np.float32), (2, 0.06, np.float64), (0, 31, np.int16),
    )]
    with np.errstate(invalid="ignore"):
        expected = [_FreshFilterSampler().sample_slice(*case) for case in cases]
    sampler = PlaneStackResampler()

    def sample(case):
        with np.errstate(invalid="ignore"):
            return sampler.sample_slice(*case)

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(sample, cases * 2))
    for index, result in enumerate(actual):
        _assert_identical(result, expected[index % len(expected)])
