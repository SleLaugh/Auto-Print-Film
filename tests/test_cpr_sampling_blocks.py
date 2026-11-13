"""CPR 分块只限制临时数组大小，保留厚层逐点采样和累计结果。"""

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.curved_stack import CurvedStackResampler


@pytest.mark.parametrize("is_mask", (False, True))
@pytest.mark.parametrize("direction", (
    np.eye(3),
    np.array(((0.8, -0.6, 0.0), (0.6, 0.8, 0.0), (0.0, 0.0, -1.0))),
    np.array(((1.0, 0.002, 0.0), (0.0, 1.0, 0.001), (0.0, 0.0, 1.0))),
))
def test_blocked_slab_matches_full_grid_including_boundaries(is_mask, direction):
    rng = np.random.default_rng(106)
    pixels = (
        rng.integers(0, 2, size=(13, 15, 17), dtype=np.uint8)
        if is_mask else rng.uniform(-1024.0, 1600.0, size=(13, 15, 17)).astype(np.float32)
    )
    image = sitk.GetImageFromArray(pixels)
    image.SetOrigin((123.4, -231.6, 45.7))
    image.SetSpacing((0.7, 0.9, 1.3))
    image.SetDirection(tuple(direction.ravel()))
    source = sitk.GetArrayViewFromImage(image)
    # 包含真正越界点、边界点与非整数位置；行数非分块大小的整数倍。
    indices = np.column_stack((np.full(19, 8.0), np.full(19, 7.0), np.linspace(-1, 13, 19)))
    curve = np.asarray([
        image.TransformContinuousIndexToPhysicalPoint(tuple(point)) for point in indices
    ])
    horizontal = np.tile(direction[:, 0], (19, 1))
    normals = np.tile(direction[:, 1], (19, 1))
    in_plane_offsets = np.linspace(-7.0, 7.0, 23)
    slabs = [-1.4, -0.7, 0.0, 0.7, 1.4]
    kwargs = dict(
        image=image, source=source, curve_samples=curve,
        in_plane_directions=horizontal, offset_directions=normals,
        in_plane_offsets=in_plane_offsets, slab_offsets=slabs, is_mask=is_mask,
    )
    full = CurvedStackResampler()
    full._SAMPLING_BLOCK_POINTS = 10000
    expected = full._sample_curved_surface_slab(**kwargs)
    blocked = CurvedStackResampler()
    blocked._SAMPLING_BLOCK_POINTS = 69
    actual = blocked._sample_curved_surface_slab(**kwargs)
    np.testing.assert_array_equal(actual, expected)

    if is_mask:
        assert set(np.unique(actual)) <= {0.0, 1.0}
    else:
        # 独立的单点插值接口核验厚层语义，包含每一步 float32 累计。
        for row, column in ((0, 0), (4, 6), (9, 11), (18, 22)):
            point = curve[row] + horizontal[row] * in_plane_offsets[column]
            value = np.float32(0)
            for offset in slabs:
                value += np.float32(full._sample_linear(image, source, point + normals[row] * offset))
            assert actual[row, column] == np.float32(value / len(slabs))


def test_large_slab_limits_linear_sampling_grid_size():
    class CapturingSampler(CurvedStackResampler):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def _sample_linear_grid(self, image, source, points):
            self.shapes.append(points.shape[:2])
            return np.full(points.shape[:2], 17.0, dtype=np.float32)

    sampler = CapturingSampler()
    sampler._SAMPLING_BLOCK_POINTS = 120
    rows, columns = 31, 23
    output = sampler._sample_curved_surface_slab(
        image=sitk.Image(2, 2, 2, sitk.sitkFloat32), source=np.zeros((2, 2, 2)),
        curve_samples=np.zeros((rows, 3)),
        in_plane_directions=np.tile((1.0, 0.0, 0.0), (rows, 1)),
        offset_directions=np.tile((0.0, 1.0, 0.0), (rows, 1)),
        in_plane_offsets=np.arange(columns), slab_offsets=[-0.5, 0.0, 0.5],
    )
    assert all(height * width <= sampler._SAMPLING_BLOCK_POINTS for height, width in sampler.shapes)
    assert sum(height for height, _ in sampler.shapes) == rows * 3
    np.testing.assert_array_equal(output, np.full((rows, columns), 17.0, dtype=np.float32))
