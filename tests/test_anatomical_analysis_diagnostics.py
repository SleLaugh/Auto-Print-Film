"""Small synthetic checks for stage-five checkpoints and unchanged geometry."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import LabelTarget, Request, SlicePartEnum
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import _LOCALIZATION_WORKSPACE_KEY


@contextmanager
def _without_checkpoints(*_args, **details):
    yield details


def _small_context(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    stage = AnatomicalAnalysisStage()
    segmentation = np.zeros((28, 24, 24), dtype=np.uint8)
    segmentation[3:11, 6:16, 6:18] = 2
    segmentation[16:24, 6:16, 6:18] = 1
    image = sitk.GetImageFromArray(segmentation)
    image.SetSpacing((0.8, 1.1, 1.4))
    image.SetOrigin((12.0, -8.0, 4.0))
    targets = {
        name: LabelTarget(name, [value])
        for name, value in (("vertebrae_L1", 1), ("vertebrae_L2", 2))
    }
    regions = stage._build_label_voxel_regions(segmentation, targets)
    target_mask = stage._build_target_mask(segmentation, targets)
    seg_path = str(tmp_path / "seg.nii.gz")
    raw_path = tmp_path / "raw.nii.gz"
    raw = (np.arange(segmentation.size).reshape(segmentation.shape) % 1400 - 800).astype(np.int16)
    raw_image = sitk.GetImageFromArray(raw)
    raw_image.CopyInformation(image)
    sitk.WriteImage(raw_image, str(raw_path))
    cut_config = {"start_labels": ["vertebrae_L1"], "end_labels": ["vertebrae_L2"]}
    context = ProcessingContext(Request(
        dicom_dir=str(tmp_path / "dicom"), output_dir=str(tmp_path / "output"),
        slice_part=SlicePartEnum.腰椎,
    ))
    context.set_result("localization", {
        "seg_for_reconstruction_path": seg_path,
        "raw_for_reconstruction_path": str(raw_path),
        "resolved_cut_config": cut_config,
        "resolved_cut_configs": {
            "whole_range": cut_config,
            "interspace": {"intervals": [{
                "region": "L1_L2", **cut_config,
            }]},
        },
        "localization_result_path": "",
    })
    context.metadata[_LOCALIZATION_WORKSPACE_KEY] = {
        "seg_path": seg_path, "image": image, "seg_array": segmentation,
        "label_targets": targets, "label_regions": regions,
        "label_coordinates": stage._extract_label_coordinates(
            segmentation, targets, label_regions=regions,
        ),
        "target_mask": target_mask,
        "correction_scope_bbox": stage._build_target_bbox(target_mask),
    }
    return stage, context, segmentation, raw, image


@pytest.fixture
def diagnostic_messages(monkeypatch):
    messages = []
    monkeypatch.setattr("src.utils.DebugLog", messages.append)
    monkeypatch.setattr("src.utils.is_debug_log_file_enabled", lambda: True)
    monkeypatch.setattr("src.core.resources.resource_usage_summary", lambda: "pid=123 rss_mib=32.0")
    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    return messages


def _checkpoint(messages, step, phase, label=None):
    matches = [message for message in messages
               if f"stage=anatomical_analysis step={step} phase={phase}" in message]
    if label is not None:
        matches = [message for message in matches if f'label="{label}"' in message]
    return matches


def test_checkpoints_preserve_real_geometry_and_input_image_values(tmp_path, diagnostic_messages):
    baseline_stage, baseline, *_ = _small_context(tmp_path / "baseline")
    with patch("src.stages.stage_localization.resource_operation", _without_checkpoints), patch(
        "src.stages.stage_anatomical_analysis.resource_operation", _without_checkpoints,
    ):
        baseline_stage.execute(baseline)

    stage, context, segmentation, raw, image = _small_context(tmp_path / "observed")
    original_segmentation, original_raw = segmentation.copy(), raw.copy()
    stage.execute(context)

    result = context.get_result("anatomical_analysis")
    assert result == baseline.get_result("anatomical_analysis")
    assert result["spine_geometry"]["interspace_body_pairs"]["L1_L2"]
    np.testing.assert_array_equal(segmentation, original_segmentation)
    np.testing.assert_array_equal(sitk.GetArrayViewFromImage(image), original_segmentation)
    raw_image = sitk.ReadImage(str(tmp_path / "observed" / "raw.nii.gz"))
    np.testing.assert_array_equal(sitk.GetArrayViewFromImage(raw_image), original_raw)
    assert _LOCALIZATION_WORKSPACE_KEY not in context.metadata
    for step in ("build_spine_body_geometry", "extract_spine_bodies_and_prisms",
                 "resolve_prism_overlaps", "measure_body_interspace", "build_anatomical_frame"):
        assert _checkpoint(diagnostic_messages, step, "start")
        assert _checkpoint(diagnostic_messages, step, "complete")
    for label in ("vertebrae_L1", "vertebrae_L2"):
        for step in ("extract_vertebral_body", "build_sagittal_quadrilateral",
                     "rasterize_vertebral_prism", "estimate_vertebral_faces",
                     "estimate_vertebral_symmetry"):
            assert len(_checkpoint(diagnostic_messages, step, "start", label)) == 1
            assert len(_checkpoint(diagnostic_messages, step, "complete", label)) == 1
    assert not _checkpoint(diagnostic_messages, "read_spine_raw", "start")


@pytest.mark.parametrize("failing_method, failing_step, label", [
    ("_vertebral_body_prism_region", "rasterize_vertebral_prism", "vertebrae_L1"),
    ("_build_anatomical_geometry", "build_anatomical_frame", None),
])
def test_native_step_failure_identifies_operation_and_releases_workspace(
    tmp_path, diagnostic_messages, failing_method, failing_step, label,
):
    stage, context, *_ = _small_context(tmp_path)
    failure = RuntimeError("synthetic native operation failure")
    with patch.object(AnatomicalAnalysisStage, failing_method, side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            stage.execute(context)

    assert caught.value is failure
    assert _checkpoint(diagnostic_messages, failing_step, "failed", label)
    assert not _checkpoint(diagnostic_messages, failing_step, "complete", label)
    assert _LOCALIZATION_WORKSPACE_KEY not in context.metadata


def test_prism_allocation_failure_logs_grid_shape_before_allocating(tmp_path, diagnostic_messages):
    stage, context, *_ = _small_context(tmp_path)
    failure = MemoryError("synthetic grid allocation failure")
    with patch("src.stages.stage_localization.np.meshgrid", side_effect=failure):
        with pytest.raises(MemoryError) as caught:
            stage.execute(context)

    assert caught.value is failure
    allocation = _checkpoint(diagnostic_messages, "allocate_prism_grid", "start")
    assert len(allocation) == 1
    assert "shape_zyx=" in allocation[0]
    assert _checkpoint(diagnostic_messages, "allocate_prism_grid", "failed")
    assert _checkpoint(diagnostic_messages, "rasterize_vertebral_prism", "failed", "vertebrae_L1")
    assert _LOCALIZATION_WORKSPACE_KEY not in context.metadata


@pytest.mark.parametrize("raw_matches", [True, False])
def test_optional_raw_verification_logs_success_or_existing_geometry_fallback(
    tmp_path, diagnostic_messages, raw_matches,
):
    stage, context, segmentation, raw, _ = _small_context(tmp_path)
    original_extract = stage._build_spine_body_prism_regions

    def mark_recovered_body(*args, **kwargs):
        result = original_extract(*args, **kwargs)
        # Exercise the existing lazy CT verification path with a small synthetic
        # recovery diagnostic; do not change either extracted body or prism.
        kwargs["body_diagnostics"]["vertebrae_L1"] = {
            "method": "density_core_recovery",
            "core_bbox_zyx": {"min": [18, 8, 8], "max": [21, 13, 15]},
        }
        return result

    raw_path = Path(context.get_result("localization")["raw_for_reconstruction_path"])
    expected_raw = raw
    if not raw_matches:
        expected_raw = np.zeros((2, 2, 2), dtype=np.int16)
        sitk.WriteImage(sitk.GetImageFromArray(expected_raw), str(raw_path))
    before = segmentation.copy()
    with patch.object(AnatomicalAnalysisStage, "_build_spine_body_prism_regions",
                      side_effect=mark_recovered_body):
        stage.execute(context)

    expected_phase = "complete" if raw_matches else "failed"
    assert _checkpoint(diagnostic_messages, "read_spine_raw", expected_phase)
    assert _checkpoint(diagnostic_messages, "load_spine_raw_for_endplates", expected_phase)
    assert _checkpoint(diagnostic_messages, "estimate_supported_endplates", "complete", "vertebrae_L1")
    assert _checkpoint(diagnostic_messages, "build_spine_body_geometry", "complete")
    assert _checkpoint(diagnostic_messages, "build_anatomical_frame", "complete")
    assert context.get_result("anatomical_analysis")["spine_geometry"] is not None
    np.testing.assert_array_equal(segmentation, before)
    raw_image = sitk.ReadImage(str(raw_path))
    np.testing.assert_array_equal(sitk.GetArrayViewFromImage(raw_image), expected_raw)
    assert _LOCALIZATION_WORKSPACE_KEY not in context.metadata
