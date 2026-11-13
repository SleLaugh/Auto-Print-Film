"""共享稀疏索引必须保持逐标签定位输入与原全卷扫描完全一致。"""

import numpy as np
import pytest

from src.models import LabelTarget
from src.stages.stage_localization import LocalizationOperations


@pytest.mark.parametrize("layout", ["contiguous", "reversed", "transposed"])
def test_regions_preserve_order_and_bounds(layout):
    source = np.random.default_rng(7341).choice(
        np.array([-3, 0, 5, 11, 70001], dtype=np.int64), size=(7, 9, 11),
    )
    if layout == "reversed":
        source = source[::-1, :, ::-1]
    elif layout == "transposed":
        source = source.transpose(2, 0, 1)
    original = source.copy()
    groups = {
        "multi": [11, 5, 11],
        "overlap": [5, -3],
        "background": [0],
        "large": [70001],
        "missing": [70002],
        "empty": [],
    }
    targets = {name: LabelTarget(name, ids) for name, ids in groups.items()}
    regions = LocalizationOperations._build_label_voxel_regions(source, targets)
    assert list(regions) == list(targets)
    for name, ids in groups.items():
        expected = np.argwhere(np.isin(source, ids))
        actual = regions[name]
        np.testing.assert_array_equal(actual.indices_zyx, expected)
        assert actual.indices_zyx.dtype == expected.dtype
        assert actual.indices_zyx.flags.f_contiguous
        if not len(expected):
            assert actual.bbox is None
        else:
            minimum, maximum = expected.min(axis=0), expected.max(axis=0)
            assert actual.bbox.to_dict() == {
                "z_min": int(minimum[0]), "z_max": int(maximum[0]),
                "y_min": int(minimum[1]), "y_max": int(maximum[1]),
                "x_min": int(minimum[2]), "x_max": int(maximum[2]),
            }
    np.testing.assert_array_equal(source, original)


def test_rebuilt_regions_follow_replaced_segmentation():
    source = np.zeros((4, 5, 6), dtype=np.uint8)
    source[1, 2, 3] = 7
    targets = {"label": LabelTarget("label", [7])}
    first = LocalizationOperations._build_label_voxel_regions(source, targets)
    replacement = np.roll(source, 1, axis=0)
    second = LocalizationOperations._build_label_voxel_regions(replacement, targets)
    np.testing.assert_array_equal(first["label"].indices_zyx, [[1, 2, 3]])
    np.testing.assert_array_equal(second["label"].indices_zyx, [[2, 2, 3]])


def test_no_targets_produce_no_regions():
    assert LocalizationOperations._build_label_voxel_regions(np.zeros((1, 2, 3)), {}) == {}
