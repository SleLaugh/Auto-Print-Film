"""AP/LR 形态比较须独立于同一物理体素结构的轴向采样密度。"""

from unittest import mock

import numpy as np
import pytest

from src.models import LabelTarget
from src.stages.stage_localization import LocalizationStage


def _resample(array, factors):
    sampled = array
    for axis, factor in enumerate(factors):
        sampled = np.repeat(sampled, factor, axis=axis)
    return sampled


def _body(array, factors=(1, 1, 1)):
    return LocalizationStage._vertebral_body_candidate_mask(
        _resample(array, factors), LabelTarget("vertebrae_L2", [1]),
        spacing_zyx=tuple(1.0 / factor for factor in factors),
    )


@pytest.mark.parametrize("factors", [(1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (2, 2, 3)])
def test_resampling_cannot_move_body_to_posterior_component(factors):
    array = np.zeros((40, 75, 100), dtype=np.uint8)
    array[10:25, 16:32, 30:62] = 1
    array[10:25, 42:55, 40:52] = 1

    baseline = _body(array)
    actual = _body(array, factors)

    assert np.all(baseline[10:25, 16:32, 30:62])
    assert not np.any(baseline[:, 42:55])
    np.testing.assert_array_equal(actual, _resample(baseline, factors))


@pytest.mark.parametrize("factors", [(1, 2, 1), (1, 3, 1), (1, 1, 2), (1, 2, 3)])
def test_resampling_keeps_thin_anterior_protrusion_separate_from_body(factors):
    array = np.zeros((24, 48, 84), dtype=np.uint8)
    array[4:20, 16:32, 30:54] = 1
    array[4:20, 6:12, 32:52] = 1
    array[4:20, 12:16, 41:43] = 1

    baseline = _body(array)
    actual = _body(array, factors)

    assert np.all(baseline[4:20, 16:32, 30:54])
    assert not np.any(baseline[:, :16])
    np.testing.assert_array_equal(actual, _resample(baseline, factors))


def test_default_spacing_preserves_existing_isotropic_candidate_selection():
    array = np.zeros((24, 48, 84), dtype=bool)
    array[4:20, 16:32, 30:54] = True
    array[4:20, 6:12, 32:52] = True
    array[4:20, 12:16, 41:43] = True

    default = LocalizationStage._vertebral_body_support_candidates(array)
    explicit = LocalizationStage._vertebral_body_support_candidates(array, spacing_zyx=(1, 1, 1))
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(
        LocalizationStage._select_vertebral_body_support(default),
        LocalizationStage._select_vertebral_body_support(explicit, spacing_zyx=(1, 1, 1)),
    )


@pytest.mark.parametrize("spacing_zyx", [(1.7, 0.85, 0.55), (2.0, 1.1, 0.8)])
def test_front_protrusion_is_smaller_than_connected_body_in_anisotropic_space(spacing_zyx):
    array = np.zeros((40, 70, 90), dtype=np.uint8)
    array[10:25, 16:32, 30:54] = 1
    array[10:25, 6:12, 32:52] = 1
    array[10:25, 12:16, 41:43] = 1
    body = LocalizationStage._vertebral_body_candidate_mask(
        array, LabelTarget("vertebrae_L2", [1]), spacing_zyx=spacing_zyx,
    )

    assert np.all(body[10:25, 16:32, 30:54])
    assert not np.any(body[:, :16])
    assert int(body.sum()) == 15 * 16 * 24


@pytest.mark.parametrize("bridge_layers", [1, 2])
def test_relative_front_rule_keeps_a_complete_body_before_larger_posterior_band(bridge_layers):
    array = np.zeros((40, 70, 90), dtype=np.uint8)
    array[10:25, 16:32, 30:54] = 1
    array[6:29, 40:60, 30:54] = 1
    array[17:17 + bridge_layers, 32:40, 37:47] = 1
    body = LocalizationStage._vertebral_body_candidate_mask(
        array, LabelTarget("vertebrae_L2", [1]), spacing_zyx=(1.7, .85, .55),
    )

    assert np.all(body[10:25, 16:32, 30:54])
    assert not np.any(body[:, 32:])
    assert int(body.sum()) == 15 * 16 * 24


@pytest.mark.parametrize("posterior_layers", [1, 2])
def test_larger_posterior_band_in_few_slices_cannot_replace_stable_front_body(posterior_layers):
    array = np.zeros((40, 70, 90), dtype=np.uint8)
    array[8:30, 16:28, 30:54] = 1
    array[17:17 + posterior_layers, 40:66, 30:54] = 1
    array[17:17 + posterior_layers, 28:40, 41:43] = 1
    body = LocalizationStage._vertebral_body_candidate_mask(
        array, LabelTarget("vertebrae_L2", [1]), spacing_zyx=(1.7, .85, .55),
    )

    assert np.all(body[8:30, 16:28, 30:54])
    assert not np.any(body[:, 28:])
    assert int(body.sum()) == 22 * 12 * 24


def test_relative_protrusion_rule_preserves_body_only_terminal_slices():
    array = np.zeros((40, 70, 90), dtype=np.uint8)
    array[6:29, 16:32, 30:54] = 1
    array[10:25, 6:12, 32:52] = 1
    array[10:25, 12:16, 41:43] = 1
    body = LocalizationStage._vertebral_body_candidate_mask(
        array, LabelTarget("vertebrae_L2", [1]), spacing_zyx=(1.7, .85, .55),
    )

    assert np.all(body[6:29, 16:32, 30:54])
    assert not np.any(body[:, :16])
    assert int(body.sum()) == 23 * 16 * 24


@pytest.mark.parametrize("stable_rear", [False, True])
@pytest.mark.parametrize("spacing_zyx", [(1, 1, 1), (1.7, .85, .55), (2, 1.1, .8)])
def test_real_short_ap_endplate_slices_cannot_switch_to_posterior_band(stable_rear, spacing_zyx):
    short_layers = 2
    truth = np.zeros((36, 76, 80), dtype=bool)
    truth[4:28, 16:32, 28:52] = True
    truth[4:4 + short_layers, 22:32, 28:52] = False
    array = truth.astype(np.uint8)
    array[4:28 if stable_rear else 4 + short_layers, 40:64, 28:52] = 1
    array[4:4 + short_layers, 22:40, 39:41] = 1

    body = LocalizationStage._vertebral_body_candidate_mask(
        array, LabelTarget("vertebrae_L3", [1]), spacing_zyx=spacing_zyx,
    )

    np.testing.assert_array_equal(body, truth)


@pytest.mark.parametrize("stable_rear", [False, True])
@pytest.mark.parametrize("spacing_zyx", [(1, 1, 1), (1.7, .85, .55), (2, 1.1, .8)])
def test_stronger_front_body_layers_still_count_when_half_the_layers_are_short(stable_rear, spacing_zyx):
    truth = np.zeros((36, 76, 80), dtype=bool)
    truth[4:28, 16:32, 28:52] = True
    truth[4:16, 22:32, 28:52] = False
    array = truth.astype(np.uint8)
    array[4:28 if stable_rear else 16, 40:64, 28:52] = 1
    array[4:16, 22:40, 39:41] = 1
    target = LabelTarget("vertebrae_L3", [1])

    candidates = LocalizationStage._vertebral_body_support_candidates(array > 0, spacing_zyx=spacing_zyx)
    np.testing.assert_array_equal(candidates, truth)
    body = LocalizationStage._vertebral_body_candidate_mask(array, target, spacing_zyx=spacing_zyx)
    # 此极端构造的半段 AP 变深，原 SI 柱高裁边会另行裁剪。这里明确
    # 验证选择器完整保留前体，以及后续处理与从正确主体出发完全一致，
    # 不把已有裁边行为误归入本次前后分支选择修复。
    with mock.patch.object(LocalizationStage, "_vertebral_body_support_candidates", return_value=truth):
        expected = LocalizationStage._vertebral_body_candidate_mask(array, target, spacing_zyx=spacing_zyx)
    np.testing.assert_array_equal(body, expected)
    assert not np.any(body[:, 40:])
