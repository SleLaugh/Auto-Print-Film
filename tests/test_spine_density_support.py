"""连续 SI 支撑必须区分内部短缺口和彼此遥远的薄片。"""

import numpy as np
import pytest

from src.stages.localization_spine_density import si_pillar_continuous_support


def test_terminal_thin_plates_do_not_gain_full_height_from_their_span():
    mask = np.zeros((20, 2, 3), dtype=bool)
    mask[:2, 0, 0] = True
    mask[-2:, 0, 0] = True
    mask[:, 1, 2] = True

    support = si_pillar_continuous_support(mask)

    assert support[0, 0] == 2
    assert support[1, 2] == 20
    assert np.count_nonzero(support) == 2


def test_one_slice_hole_connects_real_support_without_filling_the_mask():
    mask = np.ones((20, 2, 3), dtype=np.uint8)
    mask[9] = 0
    original = mask.copy()

    support = si_pillar_continuous_support(mask)

    np.testing.assert_array_equal(support, np.full((2, 3), 19))
    np.testing.assert_array_equal(mask, original)


def test_repeated_one_slice_holes_only_count_occupied_layers():
    mask = np.zeros((20, 1, 1), dtype=bool)
    mask[::2] = True

    assert si_pillar_continuous_support(mask)[0, 0] == 10


@pytest.mark.parametrize("maximum_gap_slices,expected", [(0, 3), (1, 4), (2, 7)])
def test_gap_tolerance_is_inclusive_and_longest_chain_counts_real_layers(
    maximum_gap_slices, expected,
):
    # 2 + one empty + 2 + two empty + 3: progressively join the three blocks.
    mask = np.asarray([1, 1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=bool)[:, None, None]

    assert si_pillar_continuous_support(mask, maximum_gap_slices)[0, 0] == expected


def test_columns_track_independent_chains():
    mask = np.zeros((8, 2, 2), dtype=bool)
    mask[[0, 1, 3, 4], 0, 0] = True
    mask[[0, 1, 4, 5, 6], 0, 1] = True
    mask[[2], 1, 0] = True
    mask[:, 1, 1] = True

    np.testing.assert_array_equal(
        si_pillar_continuous_support(mask), np.asarray([[4, 3], [1, 8]]),
    )


@pytest.mark.parametrize("shape", [(0, 3, 4), (5, 0, 4), (5, 3, 0)])
def test_empty_dimensions_preserve_transverse_shape(shape):
    support = si_pillar_continuous_support(np.zeros(shape, dtype=bool))

    assert support.shape == shape[1:]
    assert support.dtype == np.dtype(np.intp)
    assert not np.any(support)


@pytest.mark.parametrize("shape", [(), (3,), (3, 4), (2, 3, 4, 5)])
def test_requires_three_dimensions(shape):
    with pytest.raises(ValueError, match="三维 mask"):
        si_pillar_continuous_support(np.zeros(shape, dtype=bool))


@pytest.mark.parametrize("gap", [-1, 0.5, True, None, "1"])
def test_gap_must_be_a_nonnegative_integer(gap):
    with pytest.raises(ValueError, match="非负整数"):
        si_pillar_continuous_support(np.zeros((3, 2, 2), dtype=bool), gap)


def test_accepts_numpy_integer_gap_and_noncontiguous_mask():
    mask = np.zeros((10, 3, 4), dtype=np.uint8)
    mask[::2] = 1

    support = si_pillar_continuous_support(mask[:, ::-1, ::2], np.int64(1))

    np.testing.assert_array_equal(support, np.full((3, 2), 5))
