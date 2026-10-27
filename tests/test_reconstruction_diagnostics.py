import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import Request, SlicePartEnum
from src.stages import stage_reconstruction as reconstruction_module
from src.stages.stage_reconstruction import ReconstructionStage


def _make_context(tmp_path, *, curved=False, windows=None):
    raw_path = tmp_path / "raw.nii.gz"
    seg_path = tmp_path / "seg.nii.gz"
    values = np.arange(8 * 8 * 8, dtype=np.int16).reshape(8, 8, 8)
    sitk.WriteImage(sitk.GetImageFromArray(values), str(raw_path))
    labels = np.zeros_like(values, dtype=np.uint8)
    labels[:, 1:6, 1:6] = 1
    sitk.WriteImage(sitk.GetImageFromArray(labels), str(seg_path))
    plan = {
        "plan_id": "diagnostic_CPR" if curved else "diagnostic_MPR",
        "slice_part": "腰椎", "slice_type": "CPR" if curved else "MPR",
        "plane": "冠状位" if curved else "横状位",
        "group_name": "CPR" if curved else "whole_range",
        "geometry_type": "curved_stack" if curved else "plane_stack",
        "extent_mm": {"width": 4.0, "height": 4.0},
        "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
        "slice_thickness": 1.0, "slice_spacing": 2.0,
        "window_presets": windows or ["骨窗"], "source_labels": ["target"],
        "slices": [
            {
                "index": index, "center": {"x": 3.0, "y": 3.0, "z": z},
                "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
                "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
            }
            for index, z in enumerate((2.0, 4.0), start=1)
        ],
    }
    if curved:
        plan.update(
            curve_points=[{"x": 3.0, "y": 3.0, "z": z} for z in (1.0, 5.0)],
            curve_spacing_mm=1.0, offset_axis={"x": 1.0, "y": 0.0, "z": 0.0},
            offsets_mm=[-0.5, 0.5], fill_percent=80,
            localizer_plane="矢状位", localizer_reference={"x": 3.0, "y": 3.0, "z": 3.0},
        )
        for planned_slice, offset in zip(plan["slices"], plan["offsets_mm"]):
            planned_slice.update(
                offset_mm=offset, center={"x": 3.0 + offset, "y": 3.0, "z": 3.0},
                normal={"x": 1.0, "y": 0.0, "z": 0.0},
                row_dir={"x": 0.0, "y": 1.0, "z": 0.0},
                col_dir={"x": 0.0, "y": 0.0, "z": 1.0},
            )
    context = ProcessingContext(Request(
        dicom_dir=str(tmp_path), output_dir=str(tmp_path / "output"),
        slice_part=SlicePartEnum.腰椎,
    ))
    context.set_result("localization", {
        "raw_for_reconstruction_path": str(raw_path),
        "seg_for_reconstruction_path": str(seg_path),
        "coords": {"target": {"label_ids": [1], "found": True}},
    })
    context.set_result("preprocessing", {"source_dicom_metadata": {"0020|000d": "1.2.3"}})
    context.set_result("planning", {"plans": [plan]})
    return context, plan


@pytest.fixture
def messages(monkeypatch):
    captured = []
    monkeypatch.setattr("src.utils.DebugLog", captured.append)
    monkeypatch.setattr("src.utils.is_debug_log_file_enabled", lambda: True)
    monkeypatch.setattr(reconstruction_module, "DebugLog", lambda *args: None)
    monkeypatch.setattr(reconstruction_module, "is_debug_file_enabled", lambda: False)
    return captured


def _boundary(messages, step, phase, *, slice_index=None, window=None):
    return [
        message for message in messages
        if f"step={step} phase={phase} " in message
        and (slice_index is None or f"slice_index={slice_index} " in message)
        and (window is None or f'window="{window}"' in message)
    ]


@pytest.mark.parametrize("curved", [False, True], ids=["plane_stack", "curved_stack"])
def test_diagnostics_identify_each_slice_without_changing_pixels_or_geometry(
    tmp_path, monkeypatch, messages, curved,
):
    context, plan = _make_context(tmp_path, curved=curved)
    original_plan = copy.deepcopy(plan)
    original_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    stage = ReconstructionStage({"parallel": False})
    stage.execute(context)
    diagnostics = list(messages)
    actual = {
        path.relative_to(context.request.output_dir): pydicom.dcmread(path)
        for path in Path(context.request.output_dir).rglob("*.dcm")
    }
    assert len(actual) == 3  # Two reconstructed slices and one localizer.

    # Re-run the same tiny plan with diagnostic scopes disabled, retaining real sampling/writes.
    quiet_output = tmp_path / "quiet_output"
    context.request.output_dir = str(quiet_output)
    with monkeypatch.context() as quiet:
        quiet.setattr(reconstruction_module, "resource_operation", lambda *args, **kwargs: nullcontext({}))
        stage.execute(context)
    for relative, dataset in actual.items():
        reference = pydicom.dcmread(quiet_output / relative)
        np.testing.assert_array_equal(dataset.pixel_array, reference.pixel_array)
        for tag in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
            assert dataset.get(tag) == reference.get(tag)
    assert plan == original_plan
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == original_threads

    for step in ("execute_plan", "prepare_target_framing", "build_localizer",
                 "freeze_localizer_geometry", "write_window_localizer", "write_plan_windows"):
        assert _boundary(diagnostics, step, "start")
        assert _boundary(diagnostics, step, "complete")
    for index in (1, 2):
        for step in ("sample_hu_slice", "write_window_slice"):
            assert _boundary(diagnostics, step, "start", slice_index=index)
            completed = _boundary(diagnostics, step, "complete", slice_index=index)
            assert completed
            assert all(f'plan_id="{plan["plan_id"]}"' in message for message in completed)
            assert all("elapsed_seconds=" in message and "rss_mib=" in message for message in completed)
        framing_step = "frame_cpr_slice" if curved else "frame_plane_slice"
        assert _boundary(diagnostics, framing_step, "complete", slice_index=index)
        if curved:
            assert _boundary(diagnostics, "sample_cpr_mask", "complete", slice_index=index)
    if curved:
        assert _boundary(diagnostics, "cpr_group_bounds", "complete")


@pytest.mark.parametrize("failed_mask", [False, True], ids=["hu", "mask"])
def test_cpr_sampling_failure_keeps_original_exception_and_slice_identity(
    tmp_path, monkeypatch, messages, failed_mask,
):
    context, plan = _make_context(tmp_path, curved=True)
    error = RuntimeError("synthetic sampling failure")
    original = reconstruction_module.CurvedStackResampler.sample_slice

    def fail_second_slice(self, image, plan, planned_slice, *, is_mask=False):
        if planned_slice["index"] == 2 and is_mask == failed_mask:
            raise error
        return original(self, image, plan, planned_slice, is_mask=is_mask)

    monkeypatch.setattr(reconstruction_module.CurvedStackResampler, "sample_slice", fail_second_slice)
    with pytest.raises(RuntimeError) as caught:
        ReconstructionStage({"parallel": False}).execute(context)

    assert caught.value is error
    step = "sample_cpr_mask" if failed_mask else "sample_hu_slice"
    failed = _boundary(messages, step, "failed", slice_index=2)
    assert len(failed) == 1
    assert f'plan_id="{plan["plan_id"]}"' in failed[0]
    assert 'error_type="RuntimeError"' in failed[0]
    assert not _boundary(messages, step, "complete", slice_index=2)
    assert _boundary(messages, "execute_plan", "failed")
    assert not _boundary(messages, "write_plan_windows", "start")
    assert not list(Path(context.request.output_dir).rglob("*.dcm"))


def test_parallel_write_failure_logs_window_and_slice_and_reaches_caller(
    tmp_path, monkeypatch, messages,
):
    context, plan = _make_context(tmp_path, windows=["骨窗", "脑窗"])
    error = OSError("synthetic DICOM write failure")

    def write_slice(**kwargs):
        if kwargs["window"].name == "脑窗" and kwargs["planned_slice"]["index"] == 2:
            raise error

    monkeypatch.setattr(reconstruction_module, "write_ct_dicom", write_slice)
    with pytest.raises(OSError) as caught:
        ReconstructionStage({"parallel": True, "max_window_workers": 2}).execute(context)

    assert caught.value is error
    failed = _boundary(messages, "write_window_slice", "failed", slice_index=2, window="脑窗")
    assert len(failed) == 1
    assert f'plan_id="{plan["plan_id"]}"' in failed[0]
    assert _boundary(messages, "write_window_series", "failed", window="脑窗")
    assert _boundary(messages, "write_plan_windows", "failed")
    assert _boundary(messages, "execute_plan", "failed")
    assert not _boundary(messages, "write_window_localizer", "start")
