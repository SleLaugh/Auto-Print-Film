"""定位线灰度统计只在同次重建的同一原图内复用，背景和几何不变。"""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import SlicePartEnum
from src.stages.stage_reconstruction import ReconstructionStage, _LocalizerIntensityCache
from tests.test_interspace_scale_integration import _make_context
from tests.test_thick_slice_localizer import _case, _geometry
from tests.test_thick_cpr_localizer import _image, _cpr_plan


@pytest.mark.parametrize("value", [0.0, 17.25, float("nan"), float("inf"), -float("inf")])
def test_cached_statistic_preserves_constant_and_nonfinite_semantics(value):
    image = sitk.GetImageFromArray(np.full((2, 3, 4), value, dtype=np.float32))
    source = sitk.GetArrayViewFromImage(image)
    lower, upper = float(np.min(source)), float(np.max(source))
    expected = upper + max(upper - lower, 1.0)
    cache = _LocalizerIntensityCache(image)
    with patch.object(cache, "compute", wraps=cache.compute) as compute:
        actual = [cache.get(image), cache.get(image)]
    np.testing.assert_array_equal(actual, [expected, expected])
    assert compute.call_count == 1


def test_cache_does_not_reuse_another_image_even_when_geometry_matches():
    first = sitk.GetImageFromArray(np.arange(24, dtype=np.int16).reshape(2, 3, 4))
    other = first + 100
    cache = _LocalizerIntensityCache(first)
    assert cache.get(first) == 46.0
    assert cache.get(other) == 146.0
    assert cache.get(first) == 46.0


@pytest.mark.parametrize("direction", ["identity", "reverse_z", "rotated"])
def test_multi_plan_localizers_keep_background_alpha_geometry_and_legacy_values(direction):
    image, plan = _case(direction=direction, slice_count=3)
    other = deepcopy(plan)
    other["localizer_reference"]["x"] += 0.75
    cache = _LocalizerIntensityCache(image)
    baseline = [ReconstructionStage._build_plan_localizer_render(image, item) for item in (plan, other)]
    with patch.object(cache, "compute", wraps=cache.compute) as compute:
        actual = [ReconstructionStage._build_plan_localizer_render(image, item, intensity_cache=cache)
                  for item in (plan, other)]
    assert compute.call_count == 1
    for old, new in zip(baseline, actual):
        assert new.legacy_line_value == old.legacy_line_value
        assert _geometry(new.background_image) == _geometry(old.background_image)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(new.background_image),
                                      sitk.GetArrayFromImage(old.background_image))
        np.testing.assert_array_equal(new.line_alpha, old.line_alpha)


def test_stage_shares_one_lazy_cache_between_plans_and_replaces_it_next_request(tmp_path):
    stage = ReconstructionStage({"parallel": False})
    captured = []

    def execute_plan(**kwargs):
        captured.append(kwargs["localizer_intensity_cache"])
        return {}

    with patch.object(stage, "_execute_plan", execute_plan), patch.object(stage, "_build_plan_summary", return_value={}):
        for number in range(2):
            case_dir = tmp_path / str(number)
            case_dir.mkdir()
            context, _ = _make_context(case_dir, SlicePartEnum.腰椎, 80)
            first = len(captured)
            stage.execute(context)
            current = captured[first:]
            assert len(current) > 1
            assert all(cache is current[0] for cache in current)
            assert current[0]._value is None  # 没有定位图消费时不扫描原图。
    assert captured[0] is not captured[-1]


@pytest.mark.parametrize("spine", [False, True])
def test_curved_localizer_reuses_intensity_without_changing_render(spine):
    image, axes = _image(10.0, True)
    plan = _cpr_plan(image, axes, spine=spine)
    expected = ReconstructionStage._build_plan_localizer_render(image, plan)
    cache = _LocalizerIntensityCache(image)
    for _ in range(2):
        actual = ReconstructionStage._build_plan_localizer_render(image, plan, intensity_cache=cache)
        assert actual.legacy_line_value == expected.legacy_line_value
        assert _geometry(actual.background_image) == _geometry(expected.background_image)
        np.testing.assert_array_equal(actual.line_alpha, expected.line_alpha)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(actual.background_image),
                                      sitk.GetArrayFromImage(expected.background_image))
