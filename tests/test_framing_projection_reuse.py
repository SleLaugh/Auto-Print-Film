"""取景摘要复用保持逐层中心、共同 FOV 及单点云生命周期。"""

import copy
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from src.models import SlicePartEnum
from src.reconstruction.framing import TargetFramer, prepare_interspace_framing
from src.reconstruction.framing_context import FramingContext
from src.reconstruction.symmetry_framing import resolve_symmetry_framing_plane
from src.stages.stage_reconstruction import ReconstructionStage
from tests.test_interspace_scale_integration import _make_context
from tests.test_symmetry_framing import _case


@pytest.mark.parametrize("fill", [None, 0, 80, 100])
@pytest.mark.parametrize("oblique,flipped", [(False, False), (True, False), (True, True)])
def test_prepared_interspace_projections_match_full_framer_exactly(fill, oblique, flipped):
    seg, _, coords, plan, analysis, _, _ = _case(
        oblique=oblique, flipped=flipped, context=True,
    )
    plan["fill_percent"] = fill
    plan["framing_in_plane_extension_mm"] = {
        "row_min": 1.3, "row_max": 3.7, "col_min": 2.4, "col_max": 0.0,
    }
    empty = copy.deepcopy(plan["slices"][0])
    empty["index"] = 91
    empty["center"]["z"] += 1000
    plan["slices"].append(empty)
    # 同一 plan 有不同方向时，不能误用上一层的组边界或厚层边界。
    alternate = copy.deepcopy(plan["slices"][0])
    alternate["index"] = 92
    alternate["row_dir"], alternate["col_dir"] = alternate["col_dir"], alternate["row_dir"]
    alternate["normal"] = {key: -value for key, value in alternate["normal"].items()}
    plan["slices"].append(alternate)
    second = {**copy.deepcopy(plan), "plan_id": "same_name", "group_name": "other"}
    plan["plan_id"] = "same_name"  # 显示名称不是缓存身份。
    plans = [plan, second]
    originals = copy.deepcopy(plans)
    context = FramingContext((1, 2, 3))
    symmetry = resolve_symmetry_framing_plane(plan, analysis)
    prepared = {}
    geometry = prepare_interspace_framing(
        seg, coords, plans, context=context,
        symmetry_planes={"same_name": symmetry}, prepared_framers=prepared,
    )
    for candidate in plans:
        baseline = TargetFramer(
            seg, coords, candidate, context=context, symmetry_plane=symmetry,
            shared_geometry=geometry,
        )
        cached = prepared[id(candidate)]
        assert not hasattr(cached, "points")
        assert cached.mask is None
        for item in candidate["slices"]:
            assert cached.plane_geometry(candidate, item) == baseline.plane_geometry(candidate, item)
    assert plans == originals
    assert prepared[id(plan)] is not prepared[id(second)]
    with pytest.raises(ValueError, match="原计划"):
        prepared[id(plan)].plane_geometry(second, second["slices"][0])


def test_prepared_summaries_release_every_large_point_cloud():
    seg, _, coords, plan, _, _, _ = _case(context=True)
    plans = [dict(plan, plan_id=str(index)) for index in range(4)]
    references = []
    original_init = TargetFramer.__init__

    def track(framer, *args, **kwargs):
        assert all(reference() is None for reference in references)
        original_init(framer, *args, **kwargs)
        references.append(weakref.ref(framer.points))

    prepared = {}
    with patch.object(TargetFramer, "__init__", track):
        prepare_interspace_framing(seg, coords, plans, prepared_framers=prepared)
    assert len(prepared) == len(plans)
    assert len(references) == len(plans)
    assert all(reference() is None for reference in references)


class _CountingPoints(np.ndarray):
    def __new__(cls, values, counter):
        result = np.asarray(values).view(cls)
        result.counter = counter
        return result

    def __array_finalize__(self, source):
        self.counter = getattr(source, "counter", None)

    def __matmul__(self, other):
        self.counter[0] += 1
        return np.asarray(self) @ other


@pytest.mark.parametrize("part", ["腰椎", "颈椎", "胸椎", "肺部", "髋骨", "颅脑"])
def test_slab_bounds_reuse_exact_projection_depth_and_thickness(part):
    seg, _, coords, plan, _, _, _ = _case()
    plan.update(slice_part=part, slice_type="MPR")
    framer = TargetFramer(seg, coords, plan)
    counter = [0]
    framer.points = _CountingPoints(framer.points, counter)
    item = plan["slices"][0]
    framer.plane_target_bounds(item)
    center = framer._vector(item["center"])
    bounds = framer._slab_target_bounds(plan, center)
    assert bounds is not None
    first_count = counter[0]
    actual = framer._slab_target_bounds(plan, center.copy())
    np.testing.assert_array_equal(actual, bounds)
    assert counter[0] == first_count
    changed = dict(plan, slice_thickness=plan["slice_thickness"] + 0.25)
    framer._slab_target_bounds(changed, center)
    assert counter[0] > first_count
    count = counter[0]
    # 亚毫米浮点偏移不能被 round 成同一厚层。
    shifted = center + framer._projection_axes[:, 2] * 1e-10
    framer._slab_target_bounds(plan, shifted)
    assert counter[0] > count
    alternate = copy.deepcopy(item)
    alternate["row_dir"], alternate["col_dir"] = alternate["col_dir"], alternate["row_dir"]
    alternate["normal"] = {key: -value for key, value in alternate["normal"].items()}
    framer.plane_target_bounds(alternate)
    expected = TargetFramer(seg, coords, plan)
    expected.plane_target_bounds(alternate)
    np.testing.assert_array_equal(
        framer._slab_target_bounds(plan, center), expected._slab_target_bounds(plan, center),
    )
    empty_center = center + framer._projection_axes[:, 2] * 1000
    assert framer._slab_target_bounds(plan, empty_center) is None
    assert framer._slab_target_bounds(plan, empty_center) is None


def test_reconstruction_uses_prepared_interspace_summaries_without_rebuilding_targets(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, SlicePartEnum.腰椎, 80)
    stage = ReconstructionStage({"parallel": False})
    original_init = TargetFramer.__init__
    constructed = []
    executed = []

    def track(framer, segmentation, coords, plan, **kwargs):
        constructed.append(plan["plan_id"])
        original_init(framer, segmentation, coords, plan, **kwargs)

    def execute_plan(*, plan, target_framer, **kwargs):
        assert not hasattr(target_framer, "points")
        executed.append(plan["plan_id"])
        for item in plan["slices"]:
            target_framer.plane_geometry(plan, item)
        return {}

    monkeypatch.setattr(stage, "_execute_plan", execute_plan)
    monkeypatch.setattr(stage, "_build_plan_summary", lambda plans, output_root: {})
    with patch.object(TargetFramer, "__init__", track):
        stage.execute(context)
    assert constructed == executed
    assert len(constructed) == len(context.get_result("planning")["plans"])
