"""脑区决定层组，脑区 SI 范围内的骨仅补充面内取景。"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.config import load_slice_part_config
from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import (
    LocalizationOperations,
    LocalizationStage,
    _LOCALIZATION_WORKSPACE_KEY,
)
from src.stages.stage_planning import PlanningStage
from src.stages.stage_reconstruction import ReconstructionStage
from src.stages.stage_spatial_correction import SpatialCorrectionStage


def _write_dataset(directory, labels):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset.json"
    path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ("skull", " SKULL "))
def test_framing_support_uses_actual_semantic_labels_and_excludes_other_foreground(tmp_path, name):
    dataset = _write_dataset(tmp_path, {
        "background": 0, "brain": 7, name: [0, 13, 17],
        "skull_extra": 23, "other": 29,
    })
    array = np.array([[[7, 7, 13, 0], [17, 23, 29, 0]]], dtype=np.uint8)
    label_ids, source_labels, diagnostic = AnatomicalAnalysisStage._brain_framing_support(
        array, str(dataset),
    )
    assert label_ids == (13, 17)
    assert source_labels == (name,)
    assert diagnostic["status"] not in {"missing", "empty"}
    assert diagnostic["source_labels"] == [name]
    assert diagnostic["voxel_count"] == 2


@pytest.mark.parametrize(("bone_label", "status"), (
    (None, "missing"), (0, "missing"), (13, "empty"),
))
def test_missing_or_empty_bone_is_optional_and_does_not_select_other_foreground(
    tmp_path, bone_label, status,
):
    labels = {"background": 0, "brain": 7, "other": 19}
    if bone_label is not None:
        labels["skull"] = bone_label
    dataset = _write_dataset(tmp_path, labels)
    array = np.array([[[0, 7, 19]]], dtype=np.uint8)
    label_ids, source_labels, diagnostic = AnatomicalAnalysisStage._brain_framing_support(
        array, str(dataset),
    )
    assert label_ids == source_labels == ()
    assert diagnostic["status"] == status
    assert diagnostic["voxel_count"] == 0


def test_geometry_bridge_uses_only_brain_for_pose_and_stack_bounds():
    brain = np.zeros((3, 3, 4), dtype=bool)
    brain[1, 1, 1] = True
    image = object()
    expected = (object(), {"target": "brain"})
    with patch(
        "src.stages.stage_localization.estimate_brain_frame_with_diagnostics",
        return_value=expected,
    ) as estimate:
        result = LocalizationOperations._build_anatomical_geometry("颅脑", brain, image)
    assert result is expected
    estimate.assert_called_once_with(brain, image)


def _synthetic_labels(remote_bone=None, superior_cap=False):
    z, y, x = np.indices((58, 40, 42), dtype=float)
    relative = np.stack(((x - 20) * 0.8, (y - 19) * 0.9, (z - 18) * 1.2), axis=-1)
    angle = np.radians(-18.0)
    rotation = np.array([
        [1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])
    local = relative @ rotation
    brain = np.sum((local / [7.0, 11.0, 6.0]) ** 2, axis=-1) <= 1
    outer = np.sum((local / [11.0, 15.0, 11.0]) ** 2, axis=-1) <= 1
    inner = np.sum((local / [9.5, 13.5, 9.5]) ** 2, axis=-1) < 1
    array = np.zeros(brain.shape, dtype=np.uint8)
    array[outer & ~inner] = 19
    if superior_cap:
        # The cap is 9--13 mm above the brain's 6 mm superior radius.
        cap = ((local[..., 2] >= 15) & (local[..., 2] <= 18.5)
               & (np.abs(local[..., 0]) <= 5) & (np.abs(local[..., 1]) <= 4))
        array[cap] = 19
    if remote_bone == "inferior":
        array[1:4, 30:36, 32:38] = 19
    elif remote_bone == "superior":
        # Keep this control well outside even the configured superior margin.
        array[52:55, 4:8, 3:7] = 19
    array[brain] = 7
    # Unrelated foreground is deliberately farther away than either target.
    array[:2, :2, :2] = 77
    return array


def _run_to_planning(directory, monkeypatch, *, mode="available", fill_percent=0,
                     remote_bone=None, all_planes=False, superior_cap=False,
                     superior_extension=None):
    directory.mkdir(parents=True, exist_ok=True)
    array = _synthetic_labels(remote_bone, superior_cap)
    labels = {"background": 0, "brain": 7, "other": 77}
    if mode != "missing":
        labels["skull"] = 19
    if mode == "empty":
        array[array == 19] = 0
    _write_dataset(directory, labels)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 0.9, 1.2))
    image.SetOrigin((-24.0, -18.0, -30.0))
    seg_path, raw_path = directory / "seg.nii.gz", directory / "raw.nii.gz"
    sitk.WriteImage(image, str(seg_path))
    # NIfTI stores spacing as float32; verify against the same saved geometry
    # the stages read rather than the pre-serialization Python doubles.
    image = sitk.ReadImage(str(seg_path))
    raw = sitk.GetImageFromArray(np.where(array == 19, 900., np.where(array == 7, 40., -1000.)).astype(np.float32))
    raw.CopyInformation(image)
    sitk.WriteImage(raw, str(raw_path))
    request = Request(
        dicom_dir=str(directory / "unused_dicom"), output_dir=str(directory / "output"),
        slice_part=SlicePartEnum.颅脑,
        plane_configs=[PlaneConfig(
            plane=plane, slice_thickness=5., slice_spacing=5.,
            window_presets=["骨窗"], fill_percent=fill_percent,
        ) for plane in (tuple(PlaneTypeEnum) if all_planes else (PlaneTypeEnum.横状位,))],
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {
        "nifti_path": str(raw_path), "source_dicom_metadata": {"0020|000d": "1.2.3.4"},
    })
    context.set_result("segmentation", {"seg_path": str(seg_path), "model": {"dataset_id": 1}})
    monkeypatch.setattr("src.stages.localization_execution.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_planning.is_debug_file_enabled", lambda: False)
    config = {"correction_requested": False}
    if superior_extension is not None:
        parts = load_slice_part_config()
        parts["slice_parts"]["颅脑"]["slice_rules"]["横状位"]["MPR"]["range_extension"]["start_mm"] = superior_extension
        config["slice_part_config"] = parts
    SpatialCorrectionStage(config).execute(context)
    LocalizationStage(config).execute(context)
    original_localization = deepcopy(context.get_result("localization"))
    AnatomicalAnalysisStage(config).execute(context)
    assert context.get_result("localization") == original_localization
    assert set(original_localization["coords"]) == {"brain"}
    PlanningStage().execute(context)
    return context, image, array


def _axes(frame):
    return np.column_stack([
        [frame[name][axis] for axis in "xyz"]
        for name in ("left_right_axis", "anterior_posterior_axis", "superior_inferior_axis")
    ])


def _points(mask, image):
    xyz = np.argwhere(mask)[:, ::-1]
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    return image.GetOrigin() + (xyz * image.GetSpacing()) @ direction.T


@pytest.mark.parametrize("mode", ("available", "missing", "empty"))
def test_stage5_publishes_optional_bone_support_without_expanding_brain_targets(
    tmp_path, monkeypatch, mode,
):
    context, image, array = _run_to_planning(tmp_path, monkeypatch, mode=mode)
    anatomical = context.get_result("anatomical_analysis")
    effective = {**context.get_result("localization"), **anatomical}
    assert anatomical["framing_source_labels"] == (["skull"] if mode == "available" else [])
    assert anatomical["framing_label_ids"] == ([19] if mode == "available" else [])
    assert set(effective["coords"]) == set(effective["label_targets"]) == {"brain"}
    plan = context.get_result("planning")["plans"][0]
    assert plan["source_labels"] == ["brain"]
    indices = np.argwhere(array == 7)
    bbox = effective["correction_scope_bbox"]
    for index, axis in enumerate("zyx"):
        assert bbox[f"{axis}_min"] == indices[:, index].min()
        assert bbox[f"{axis}_max"] == indices[:, index].max()
    points = _points(array == 7, image)
    frame = anatomical["anatomical_geometry"]["global_frame"]
    axes = _axes(frame)
    support = np.abs(np.diag(image.GetSpacing()) @ axes).sum(axis=0) / 2
    projected = points @ axes
    for index, axis in enumerate(("left_right", "anterior_posterior", "superior_inferior")):
        assert frame["bounds"][axis]["minimum"] <= projected[:, index].min() - support[index] + 1e-7
        assert frame["bounds"][axis]["maximum"] >= projected[:, index].max() + support[index] - 1e-7


def test_bone_cannot_extend_brain_pose_or_any_of_the_three_plane_stacks(tmp_path, monkeypatch):
    contexts = [
        _run_to_planning(tmp_path / name, monkeypatch, remote_bone=remote, all_planes=True)[0]
        for name, remote in (("base", None), ("inferior", "inferior"), ("superior", "superior"))
    ]
    first = contexts[0].get_result("anatomical_analysis")["anatomical_geometry"]["global_frame"]
    reference_plans = contexts[0].get_result("planning")["plans"]
    assert len(reference_plans) == 3
    pitch = np.degrees(np.arctan2(_axes(first)[2, 1], _axes(first)[1, 1]))
    assert pitch == pytest.approx(-18., abs=2.)
    for context in contexts[1:]:
        frame = context.get_result("anatomical_analysis")["anatomical_geometry"]["global_frame"]
        assert frame == first
        for plan, reference in zip(context.get_result("planning")["plans"], reference_plans):
            assert plan["source_labels"] == ["brain"]
            for name in ("target_start", "target_end", "range_start", "range_end", "slices"):
                assert plan[name] == reference[name]
            assert len(plan["slices"]) == len(reference["slices"])
        assert _LOCALIZATION_WORKSPACE_KEY not in context.metadata


def _framing_points(context, image, array, plan):
    """Independent physical SI clipping oracle, including cuboid overlap."""
    frame = context.get_result("anatomical_analysis")["anatomical_geometry"]["global_frame"]
    brain_points = _points(array == 7, image)
    bone_points = _points(array == 19, image)
    si = _axes(frame)[:, 2]
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    half_support = np.abs((direction * image.GetSpacing()).T @ si).sum() / 2
    lower = (brain_points @ si).min() - half_support
    upper = ((brain_points @ si).max() + half_support
             + float(plan.get("framing_superior_extension_mm", 0.)))
    depths = bone_points @ si
    keep = (depths + half_support > lower + 1e-7) & (depths - half_support < upper - 1e-7)
    return brain_points, np.concatenate((brain_points, bone_points[keep])), int(np.count_nonzero(~keep))


@pytest.mark.parametrize("extension", (None, 0., 7.5, 15.))
def test_superior_framing_margin_serializes_the_configured_axial_snapshot_only(
    tmp_path, monkeypatch, extension,
):
    context, _, _ = _run_to_planning(
        tmp_path, monkeypatch, all_planes=True, superior_extension=extension,
    )
    # PlanningStage serializes SlicePlan; a JSON round trip must retain the
    # request-independent snapshot consumed later by ReconstructionStage.
    plans = json.loads(json.dumps(context.get_result("planning")["plans"]))
    expected_extension = extension
    if expected_extension is None:
        expected_extension = load_slice_part_config()["slice_parts"]["颅脑"][
            "slice_rules"
        ]["横状位"]["MPR"]["range_extension"]["start_mm"]
    for plan in plans:
        if plan["plane"] == "横状位":
            assert plan["framing_superior_extension_mm"] == expected_extension
        else:
            assert plan.get("framing_superior_extension_mm", 0.) == 0.


def test_configured_superior_margin_covers_cranial_cap_without_changing_brain_or_inferior_end(
    tmp_path, monkeypatch,
):
    baseline, image, array = _run_to_planning(
        tmp_path / "zero", monkeypatch, all_planes=True,
        superior_cap=True, superior_extension=0.,
    )
    extended, _, _ = _run_to_planning(
        tmp_path / "extended", monkeypatch, all_planes=True, superior_cap=True,
        superior_extension=15.,
    )
    first_frame = baseline.get_result("anatomical_analysis")["anatomical_geometry"]["global_frame"]
    assert extended.get_result("anatomical_analysis")["anatomical_geometry"]["global_frame"] == first_frame
    reference = {p["plane"]: p for p in baseline.get_result("planning")["plans"]}
    for plan in extended.get_result("planning")["plans"]:
        before = reference[plan["plane"]]
        assert plan["target_start"] == before["target_start"]
        assert plan["target_end"] == before["target_end"]
        if plan["plane"] != "横状位":
            assert plan["slices"] == before["slices"]
            assert plan.get("framing_superior_extension_mm", 0.) == 0.
            continue
        assert plan["framing_superior_extension_mm"] == 15.
        normal = np.array([plan["slices"][0]["normal"][a] for a in "xyz"])
        half_voxel = np.abs(np.asarray(image.GetSpacing()) * normal).sum() / 2
        brain_top = (_points(array == 7, image) @ normal).max() + half_voxel
        bone_top = (_points(array == 19, image) @ normal).max() + half_voxel
        assert 9. <= bone_top - brain_top <= 13.
        old_top = np.dot([before["range_start"][a] for a in "xyz"], normal) + 2.5
        new_top = np.dot([plan["range_start"][a] for a in "xyz"], normal) + 2.5
        assert old_top < bone_top - 8.
        assert new_top >= bone_top
        assert new_top >= brain_top + 15. - 1e-6
        old_bottom = np.dot([before["range_end"][a] for a in "xyz"], normal) - 2.5
        new_bottom = np.dot([plan["range_end"][a] for a in "xyz"], normal) - 2.5
        # The lower target is unchanged; only symmetric rounding slack may move
        # its outer slab edge by at most half a configured spacing.
        assert abs(new_bottom - old_bottom) <= plan["slice_spacing"] / 2 + 1e-6
        assert len(plan["slices"]) == len(before["slices"]) + 3


@pytest.mark.parametrize("fill_percent", (50., 100.))
def test_superior_cap_participates_in_new_first_layer_framing_without_inferior_jaw(
    tmp_path, monkeypatch, fill_percent,
):
    context, image, array = _run_to_planning(
        tmp_path, monkeypatch, fill_percent=fill_percent,
        superior_cap=True, remote_bone="inferior", superior_extension=15.,
    )
    plan = context.get_result("planning")["plans"][0]
    assert plan["framing_superior_extension_mm"] == 15.
    brain_points, points, ignored = _framing_points(context, image, array, plan)
    assert ignored > 0
    normal = np.array([plan["slices"][0]["normal"][a] for a in "xyz"])
    half_voxel = np.abs(np.asarray(image.GetSpacing()) * normal).sum() / 2
    first_depth = np.dot([plan["slices"][0]["center"][a] for a in "xyz"], normal)
    assert not np.any(np.abs(brain_points @ normal - first_depth) < 2.5 + half_voxel)
    assert np.any(np.abs(points @ normal - first_depth) < 2.5 + half_voxel)
    ReconstructionStage({"parallel": False}).execute(context)
    output = context.get_result("reconstruction")["outputs"][0]
    first_framing = output["framing"][0]
    assert first_framing.get("target_center_source", first_framing["center_source"]) == "slice_target"
    assert output["framing"][0]["context_label_ids"] == [19]
    _verify_output(plan, output, image, brain_points, points, fill_percent)


@pytest.mark.parametrize("fill_percent", (0., 50., 100.))
def test_remote_bone_cannot_change_dicom_framing_in_any_plane_or_fill_mode(
    tmp_path, monkeypatch, fill_percent,
):
    reference = None
    for name, remote in (("base", None), ("inferior", "inferior"), ("superior", "superior")):
        context, image, array = _run_to_planning(
            tmp_path / name, monkeypatch, fill_percent=fill_percent,
            remote_bone=remote, all_planes=True,
        )
        ReconstructionStage({"parallel": False}).execute(context)
        outputs = context.get_result("reconstruction")["outputs"]
        actual = {}
        for plan, output in zip(context.get_result("planning")["plans"], outputs):
            brain_points, framing_points, ignored = _framing_points(context, image, array, plan)
            assert ignored > 0  # The inferior shell is not display support.
            actual[plan["plane"]] = _verify_output(
                plan, output, image, brain_points, framing_points, fill_percent,
            )
        if reference is None:
            reference = actual
        else:
            assert actual.keys() == reference.keys()
            for plane in actual:
                for key in actual[plane]:
                    np.testing.assert_allclose(actual[plane][key], reference[plane][key], atol=1e-7)
        assert set(context.get_result("localization")["coords"]) == {"brain"}


def _verify_output(plan, output, image, brain_points, points, fill_percent):
    directory = Path(output["window_outputs"]["骨窗"])
    files = sorted(directory.glob("[0-9]*.dcm"), key=lambda path: int(path.stem))
    assert len(files) == len(plan["slices"]) == output["slice_count"]
    row, col, normal = (
        np.array([plan["slices"][0][name][axis] for axis in "xyz"])
        for name in ("row_dir", "col_dir", "normal")
    )
    axes = np.column_stack((row, col, normal))
    projected = points @ axes
    support = np.abs(np.diag(image.GetSpacing()) @ axes).sum(axis=0) / 2
    group_lower = projected[:, :2].min(axis=0) - support[:2]
    group_upper = projected[:, :2].max(axis=0) + support[:2]
    group_center = (group_lower + group_upper) / 2
    # Valid bone must still enlarge the displayed support, especially laterally
    # in axial/coronal images, even though it cannot lengthen the stack.
    brain_projection = brain_points @ axes
    brain_size = np.ptp(brain_projection[:, :2], axis=0) + 2 * support[:2]
    if plan["plane"] in {"横状位", "冠状位"}:
        assert group_upper[0] - group_lower[0] > brain_size[0] + 2.
    scales, depths, centers, spacings, sizes = [], [], [], [], []
    base_spacing = np.array([plan["output_spacing_mm"][axis] for axis in "xy"])
    original_size = np.maximum(1, np.rint(np.array([
        plan["extent_mm"][axis] for axis in ("width", "height")
    ]) / base_spacing).astype(int) + 1)
    for path, planned, framing in zip(files, plan["slices"], output["framing"]):
        dataset = pydicom.dcmread(str(path))
        iop = np.array(dataset.ImageOrientationPatient, dtype=float)
        np.testing.assert_allclose(iop, np.r_[row, col], atol=1e-7)
        np.testing.assert_allclose(np.cross(iop[:3], iop[3:]), normal, atol=1e-7)
        assert float(dataset.SliceThickness) == float(dataset.SpacingBetweenSlices) == 5.
        spacing = np.array(dataset.PixelSpacing, dtype=float)[::-1]
        size = np.array([dataset.Columns, dataset.Rows])
        # Source-support cropping can change Rows only. The common sampling
        # scale and original column grid must still be exactly preserved.
        assert dataset.Columns == original_size[0]
        np.testing.assert_allclose(spacing, base_spacing / framing["scale"], atol=1e-7)
        np.testing.assert_array_equal(size, framing["output_size_xy"])
        crop = framing.get("vertical_source_support", {})
        top, bottom = crop.get("crop_top_rows", 0), crop.get("crop_bottom_rows", 0)
        assert isinstance(top, int) and top >= 0
        assert isinstance(bottom, int) and bottom >= 0
        assert dataset.Rows == original_size[1] - top - bottom
        if top or bottom:
            assert crop["original_rows"] == original_size[1]
        center = np.array(dataset.ImagePositionPatient, dtype=float)
        center += row * ((size[0] - 1) * spacing[0] / 2)
        center += col * ((size[1] - 1) * spacing[1] / 2)
        planned_center = np.array([planned["center"][axis] for axis in "xyz"])
        assert float((center - planned_center) @ normal) == pytest.approx(0., abs=1e-6)
        depths.append(float(center @ normal))
        centers.append(center)
        spacings.append(spacing)
        sizes.append(size)
        intersects = np.abs(projected[:, 2] - depths[-1]) < 2.5 + support[2] - 1e-7
        expected_center = group_center
        if np.any(intersects):
            lower = projected[intersects, :2].min(axis=0) - support[:2]
            upper = projected[intersects, :2].max(axis=0) + support[:2]
            if fill_percent > 0:
                expected_center = (lower + upper) / 2
            actual_lower = center @ axes[:, :2] - size * spacing / 2
            actual_upper = center @ axes[:, :2] + size * spacing / 2
            assert np.all(actual_lower <= lower + 1e-6)
            assert np.all(actual_upper >= upper - 1e-6)
        # The old target center remains the independent framing oracle. An
        # asymmetric integer crop moves the displayed center by this exact
        # col-axis distance; neither the row axis nor the slice normal may move.
        original_target_center = (
            axes[:, :2] @ expected_center + normal * depths[-1]
        )
        crop_shift_mm = (top - bottom) * spacing[1] / 2
        np.testing.assert_allclose(
            center, original_target_center + col * crop_shift_mm, atol=1e-6,
        )
        np.testing.assert_allclose(
            (center - original_target_center) @ axes,
            [0., crop_shift_mm, 0.], atol=1e-6,
        )
        if top or bottom:
            np.testing.assert_allclose(
                [crop["original_center"][axis] for axis in "xyz"],
                original_target_center, atol=1e-6,
            )
        np.testing.assert_allclose(
            center, [framing["center"][axis] for axis in "xyz"], atol=1e-6,
        )
        scales.append(framing["scale"])
        np.testing.assert_allclose(
            framing["target_size_mm"], group_upper - group_lower, atol=1e-6,
        )
        assert dataset.pixel_array.shape == (dataset.Rows, dataset.Columns)
    step = -5. if plan["plane"] == "横状位" else 5.
    np.testing.assert_allclose(np.diff(depths), step, atol=1e-6)
    assert max(depths) + 2.5 >= brain_projection[:, 2].max() + support[2] - 1e-6
    assert min(depths) - 2.5 <= brain_projection[:, 2].min() - support[2] + 1e-6
    np.testing.assert_allclose(scales, scales[0], atol=1e-12)
    if fill_percent == 0:
        assert scales[0] == 1.
    return {"centers": centers, "spacings": spacings, "sizes": sizes, "scales": scales}
