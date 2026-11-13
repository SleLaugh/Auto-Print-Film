"""共享标签索引必须保持原标签查找的坐标、顺序和内存布局。"""

import gc
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from src.core.label_index import SparseLabelIndex


@pytest.mark.parametrize("layout", ["c", "fortran", "transpose", "reverse", "stride"])
@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float32])
def test_sparse_groups_match_full_scan_exactly(layout, dtype):
    random = np.random.default_rng(701)
    source = random.choice([-4, 0, 1, 3, 9], size=(9, 11, 13)).astype(dtype)
    if layout == "fortran":
        source = np.asfortranarray(source)
    elif layout == "transpose":
        source = source.transpose(2, 0, 1)
    elif layout == "reverse":
        source = source[::-1, :, ::-1]
    elif layout == "stride":
        source = source[::2, 1::2, ::3]
    groups = [[], [-4], [0], [9, 1], [1, 1, 3], [17], [17, 3, 0], [0, -4, 1, 3, 9, 17]]
    index = SparseLabelIndex(source, [-4, 0, 1, 3, 9, 17, 1])
    for ids in groups:
        expected = np.argwhere(np.isin(source, ids))
        actual = index.indices(ids)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype == np.dtype(np.intp)
        assert actual.flags.f_contiguous
        assert actual.strides == expected.strides or actual.size == 0


@pytest.mark.parametrize(
    "source,ids,groups",
    [
        (np.array([-2**50, 0, 2**40, 2**50], dtype=np.int64), [-2**50, 0, 2**40, 2**50], [[-2**50], [2**40, 0], [2**50]]),
        (np.array([0, 2**63 + 7, 2**64 - 1, 1], dtype=np.uint64), [0, 2**63 + 7, 2**64 - 1], [[0], [2**63 + 7], [2**64 - 1], [0, 2**63 + 7]]),
        (np.array([True, False, False, True]), [0, 1], [[0], [1], [0, 1]]),
    ],
)
def test_large_and_unsigned_labels_preserve_numpy_membership(source, ids, groups):
    source = source.reshape((1, 2, 2))
    index = SparseLabelIndex(source, ids)
    for group in groups:
        np.testing.assert_array_equal(index.indices(group), np.argwhere(np.isin(source, group)))


@pytest.mark.parametrize("shape", [(0, 3, 4), (3, 0, 4), (3, 4, 0), (1, 1, 1)])
def test_empty_volumes_and_empty_target_sets(shape):
    source = np.zeros(shape, dtype=np.uint8)
    index = SparseLabelIndex(source, [1, 2])
    assert index.indices([1]).shape == (0, 3)
    empty = SparseLabelIndex(source, [])
    assert empty.contains_ids([])
    assert not empty.contains_ids([0])
    assert empty.indices([]).shape == (0, 3)


def test_source_identity_coverage_and_iterable_ids_are_explicit():
    source = np.zeros((2, 3, 4), dtype=np.uint8)
    index = SparseLabelIndex(source, (value for value in [0, 1, 1, 4]))
    assert index.matches_source(source)
    assert not index.matches_source(source.copy())
    assert not index.matches_source(source.view())
    assert index.label_ids == frozenset({0, 1, 4})
    assert index.contains_ids(iter([4, 1, 1]))  # 存在于索引范围不代表存在体素。
    assert not index.contains_ids([0, 7])
    with pytest.raises(ValueError, match="未收录标签.*7"):
        index.indices([0, 7])
    np.testing.assert_array_equal(index.indices(iter([0, 1])), np.argwhere(source == 0))


def test_only_one_full_volume_scan_and_no_group_clouds_are_retained():
    source = np.zeros((37, 31, 29), dtype=np.int16)
    source[1, 2, 3] = 5
    source[7, 8, 9] = 6
    source[10, 11, 12] = 5
    isin = np.isin
    calls = []

    def record(array, ids, *args, **kwargs):
        calls.append(array)
        return isin(array, ids, *args, **kwargs)

    with patch("src.core.label_index.np.isin", side_effect=record):
        index = SparseLabelIndex(source, [5, 6, 8])
        first = index.indices([5])
        first_reference = weakref.ref(first)
        np.testing.assert_array_equal(first, [[1, 2, 3], [10, 11, 12]])
        del first
        gc.collect()
        assert first_reference() is None
        index.indices([6, 8])
        index.indices([5, 6])
    assert sum(array is source for array in calls) == 1
    assert all(array.shape == (3,) for array in calls[1:])
    retained = [value for value in vars(index).values() if isinstance(value, np.ndarray) and value is not source]
    assert len(retained) == 2
    assert all(value.ndim == 1 and value.size == 3 for value in retained)
    assert sum(value.nbytes for value in retained) == 3 * (np.dtype(np.intp).itemsize + source.dtype.itemsize)


def test_returned_coordinates_can_be_changed_without_corrupting_the_index():
    source = np.ones((2, 3, 4), dtype=np.uint8)
    index = SparseLabelIndex(source, [1])
    coordinates = index.indices([1])
    coordinates[:] = -1
    np.testing.assert_array_equal(index.indices([1]), np.argwhere(source == 1))


def test_source_is_retained_without_a_full_copy():
    source = np.zeros((3, 4, 5), dtype=np.int16)[:, :, ::-1]
    source[1, 2, 3] = 1
    reference = weakref.ref(source)
    index = SparseLabelIndex(source, [1])
    del source
    gc.collect()
    assert reference() is not None
    assert index.matches_source(reference())
    del index
    gc.collect()
    assert reference() is None


def test_invalid_shapes_and_noninteger_ids_are_rejected():
    with pytest.raises(TypeError, match="numpy.ndarray"):
        SparseLabelIndex([[[1]]], [1])
    with pytest.raises(ValueError, match="三维"):
        SparseLabelIndex(np.zeros((3, 4)), [0])
    with pytest.raises(TypeError):
        SparseLabelIndex(np.zeros((3, 4, 5)), [1.5])
