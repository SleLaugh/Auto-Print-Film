"""同分割多计划共享稀疏标签索引，保持旧全卷扫描的精确取景结果。"""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import SlicePartEnum
from src.reconstruction.framing import (
    FramingLabelIndex,
    TargetFramer,
    prepare_interspace_framing,
)
from src.reconstruction.framing_context import FramingContext
from src.reconstruction.symmetry_framing import resolve_symmetry_framing_plane
from src.stages.stage_reconstruction import ReconstructionStage
from tests.test_interspace_scale_integration import _make_context
from tests.test_symmetry_framing import _case


def _legacy_points(segmentation, ids, context, *, merge, fill):
    """保留旧 mask 合并与 argwhere 顺序作为独立的坐标精确性基准。"""

    array = sitk.GetArrayViewFromImage(segmentation)
    selected = np.isin(array, list(ids))
    voxel_axes = np.asarray(segmentation.GetDirection()).reshape(3, 3) * segmentation.GetSpacing()
    context_points = None
    context_ids = ()
    if context is not None and (merge or (fill or 0) > 0):
        axis = np.asarray(context.axis_xyz, dtype=float)
        axis /= np.linalg.norm(axis)
        axis_steps = voxel_axes.T @ axis
        target_depths = np.argwhere(selected)[:, ::-1] @ axis_steps
        half_support = float(np.abs(axis_steps).sum() / 2)
        lower = float(target_depths.min() - half_support)
        upper = float(target_depths.max() + half_support + context.upper_extension_mm)
        extra = np.argwhere(np.isin(array, list(set(context.label_ids) - ids - {0})))
        depths = extra[:, ::-1] @ axis_steps
        extra = extra[(depths + half_support > lower + 1e-7) & (depths - half_support < upper - 1e-7)]
        if len(extra):
            context_ids = tuple(sorted(map(int, np.unique(array[tuple(extra.T)]))))
            if merge:
                selected[tuple(extra.T)] = True
            else:
                context_points = extra[:, ::-1] @ voxel_axes.T + segmentation.GetOrigin()
    points = np.argwhere(selected)[:, ::-1] @ voxel_axes.T + segmentation.GetOrigin()
    return points, context_points, context_ids, selected


@pytest.mark.parametrize("oblique,flipped", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fill", [None, 0, 80, 100])
@pytest.mark.parametrize("merge", [False, True])
def test_shared_index_keeps_target_context_points_and_geometry_exact(oblique, flipped, fill, merge):
    seg, _, coords, plan, analysis, _, direction = _case(
        oblique=oblique, flipped=flipped, context=True,
    )
    plan["fill_percent"] = fill
    context = FramingContext((1, 2, 3, 99), tuple(direction[:, 2]), upper_extension_mm=0.75)
    index = FramingLabelIndex(seg, {1, 2, 3, 99})
    symmetry = resolve_symmetry_framing_plane(plan, analysis)
    expected_points, expected_context, expected_ids, _ = _legacy_points(
        seg, {1, 2}, context, merge=merge, fill=fill,
    )
    baseline = TargetFramer(
        seg, coords, plan, context=context, merge_context_into_target=merge, symmetry_plane=symmetry,
    )
    actual = TargetFramer(
        seg, coords, plan, context=context, merge_context_into_target=merge,
        symmetry_plane=symmetry, label_index=index,
    )
    np.testing.assert_array_equal(actual.points, expected_points)
    if expected_context is None:
        assert actual._context_points is None
    else:
        np.testing.assert_array_equal(actual._context_points, expected_context)
    assert actual.context_label_ids == expected_ids
    assert actual.mask is None
    for item in plan["slices"]:
        assert actual.plane_geometry(plan, item) == baseline.plane_geometry(plan, item)


@pytest.mark.parametrize("part", ["肺部", "髋骨", "颅脑"])
@pytest.mark.parametrize("selected_region", [False, True])
def test_index_preserves_selected_region_and_source_support(part, selected_region):
    seg, raw, coords, plan, _, _, _ = _case(oblique=True, context=True)
    plan.update(slice_part=part, slice_type="MPR")
    if selected_region:
        plan["framing_policy"] = {"basis": "selected_region", "margin_mm": 2.5}
    context = FramingContext((1, 2, 3))
    index = FramingLabelIndex(seg, {1, 2, 3})
    baseline = TargetFramer(seg, coords, plan, context=context, source_image=raw)
    actual = TargetFramer(seg, coords, plan, context=context, source_image=raw, label_index=index)
    np.testing.assert_array_equal(actual.points, baseline.points)
    assert actual.context_label_ids == baseline.context_label_ids
    for item in plan["slices"]:
        assert actual.plane_geometry(plan, item) == baseline.plane_geometry(plan, item)


def test_multiple_target_sets_use_shared_index_without_full_volume_scans():
    seg, _, coords, plan, _, _, _ = _case(oblique=True, context=True)
    plans = [dict(plan, source_labels=names) for names in (["C2"], ["C3"], ["C2", "C3"])]
    context = FramingContext((1, 2, 3))
    index = FramingLabelIndex.for_plans(seg, coords, plans, context=context)
    assert index is not None
    original_isin = np.isin
    shape = tuple(reversed(seg.GetSize()))

    def sparse_only(array, ids, *args, **kwargs):
        assert np.shape(array) != shape, "已建立共享索引后不应重复扫描全卷标签"
        return original_isin(array, ids, *args, **kwargs)

    for candidate in plans:
        baseline = TargetFramer(seg, coords, candidate, context=context)
        with patch("src.reconstruction.framing.np.isin", sparse_only):
            actual = TargetFramer(seg, coords, candidate, context=context, label_index=index)
        np.testing.assert_array_equal(actual.points, baseline.points)
        np.testing.assert_array_equal(actual._context_points, baseline._context_points)


def test_index_is_bound_to_original_image_and_uncovered_ids_fall_back():
    seg, _, coords, plan, _, _, _ = _case()
    only_first = FramingLabelIndex(seg, {1})
    assert only_first.indices(seg, {2}) is None
    other = sitk.Image(seg)
    assert only_first.indices(other, {1}) is None
    labels = sitk.GetArrayFromImage(seg)
    labels[labels == 1] = 0
    changed = sitk.GetImageFromArray(labels)
    changed.CopyInformation(seg)
    for image in (seg, other, changed):
        baseline = TargetFramer(image, coords, plan)
        actual = TargetFramer(image, coords, plan, label_index=only_first)
        np.testing.assert_array_equal(actual.points, baseline.points)


@pytest.mark.parametrize("source_labels,coords_update,message", [
    (["missing"], {}, "缺少用于居中的目标 label_ids"),
    (["absent"], {"absent": {"label_ids": [99]}}, "分割图中未找到目标标签 \\[99\\]"),
])
def test_missing_target_errors_survive_indexing(source_labels, coords_update, message):
    seg, _, coords, plan, _, _, _ = _case()
    coords.update(coords_update)
    plan["source_labels"] = source_labels
    index = FramingLabelIndex(seg, {1, 2, 99})
    with pytest.raises(ValueError, match=message):
        TargetFramer(seg, coords, plan, label_index=index)


def test_curved_masks_bypass_point_index_and_only_mpr_plans_create_union():
    seg, _, coords, plan, _, _, _ = _case(oblique=True, context=True)
    curved = dict(plan, geometry_type="curved_stack")
    context = FramingContext((1, 2, 3))
    assert FramingLabelIndex.for_plans(seg, coords, [curved, curved], context=context) is None
    assert FramingLabelIndex.for_plans(seg, coords, [plan], context=context) is None
    baseline = TargetFramer(seg, coords, curved, context=context)
    index = FramingLabelIndex(seg, {1, 2, 3})
    with patch.object(index, "indices", side_effect=AssertionError("CPR 不展开共享点索引")):
        actual = TargetFramer(seg, coords, curved, context=context, label_index=index)
    assert not hasattr(actual, "points")
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual.mask), sitk.GetArrayFromImage(baseline.mask))
    assert actual.context_label_ids == baseline.context_label_ids


def test_prepared_interspace_summaries_keep_exact_geometry_with_shared_label_index():
    seg, _, coords, plan, _, _, _ = _case(oblique=True, context=True)
    plans = [plan, dict(copy.deepcopy(plan), plan_id="second", source_labels=["C3"])]
    context = FramingContext((1, 2, 3))
    index = FramingLabelIndex.for_plans(seg, coords, plans, context=context)
    baseline_prepared, actual_prepared = {}, {}
    baseline = prepare_interspace_framing(seg, coords, plans, context=context, prepared_framers=baseline_prepared)
    actual = prepare_interspace_framing(
        seg, coords, plans, context=context, prepared_framers=actual_prepared, label_index=index,
    )
    assert actual == baseline
    for candidate in plans:
        for item in candidate["slices"]:
            assert actual_prepared[id(candidate)].plane_geometry(candidate, item) == (
                baseline_prepared[id(candidate)].plane_geometry(candidate, item)
            )


def test_reconstruction_builds_one_index_and_shares_it_across_plans(tmp_path, monkeypatch):
    context, _ = _make_context(tmp_path, SlicePartEnum.腰椎, 80)
    stage = ReconstructionStage({"parallel": False})
    constructed, received = [], []
    original_index_init = FramingLabelIndex.__init__
    original_framer_init = TargetFramer.__init__

    def index_init(index, *args, **kwargs):
        original_index_init(index, *args, **kwargs)
        constructed.append(index)

    def framer_init(framer, *args, **kwargs):
        received.append(kwargs.get("label_index"))
        original_framer_init(framer, *args, **kwargs)

    monkeypatch.setattr(stage, "_execute_plan", lambda **kwargs: {})
    monkeypatch.setattr(stage, "_build_plan_summary", lambda *args: {})
    with patch.object(FramingLabelIndex, "__init__", index_init), patch.object(TargetFramer, "__init__", framer_init):
        stage.execute(context)
    assert len(constructed) == 1
    assert len(received) == len(context.get_result("planning")["plans"])
    assert all(index is constructed[0] for index in received)
