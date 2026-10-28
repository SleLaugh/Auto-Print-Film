"""19 标签脊柱三 MPR、两 CPR 的真实 CPU 缓存与重算对照。"""

import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from src.config import AppConfig, LoggingConfig, PipelineConfig, load_slice_part_config
from src.core.pipeline import Pipeline
from src.models import PlaneConfig, Request, WindowPreset
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.stages.stage_segmentation import SegmentationStage

from test_series_reuse_end_to_end import _forbid_middle_stages, _prepare_series


def _spine_source_and_segmentation():
    # Same 19-label fixture as test_spine_part_pipeline: C7 through S1,
    # distinct vertebral-body and spinous-process HU values.
    names = (["vertebrae_C7"] + [f"vertebrae_T{i}" for i in range(1, 13)]
             + [f"vertebrae_L{i}" for i in range(1, 6)] + ["vertebrae_S1"])
    segmentation = np.zeros((240, 40, 40), dtype=np.uint8)
    source = np.full(segmentation.shape, -1000, dtype=np.int16)
    labels = {"background": 0, "vertebrae_C6": 99}
    for index, name in enumerate(names):
        label_id, center_z = index + 1, 228 - index * 12
        labels[name] = label_id
        body = np.s_[center_z - 3:center_z + 4, 12:20, 10:27]
        posterior = np.s_[center_z - 2:center_z + 3, 20:29, 17:20]
        segmentation[body] = label_id
        segmentation[posterior] = label_id
        source[body] = 100
        source[posterior] = 900
    segmentation[234:239, 12:20, 10:27] = 99
    raw = sitk.GetImageFromArray(source)
    raw.SetSpacing((0.8, 0.8, 1.2))
    return raw, segmentation, labels


def _spine_request(parent, name, *, width=1600, center=450):
    requested = [
        (PlaneTypeEnum.冠状位, SliceType.MPR),
        (PlaneTypeEnum.矢状位, SliceType.MPR),
        (PlaneTypeEnum.横状位, SliceType.MPR),
        (PlaneTypeEnum.矢状位, SliceType.CPR),
        (PlaneTypeEnum.冠状位, SliceType.CPR),
    ]
    root = parent / name
    return Request(
        dicom_dir=str(root / "dicom"), output_dir=str(root / "胸腰椎"),
        cache_dir=str(root / "cache"), series_cache_parent=str(parent),
        slice_part=SlicePartEnum.胸腰椎, scan_thickness=1.2,
        plane_configs=[PlaneConfig(
            plane=plane, slice_type=slice_type, slice_thickness=1,
            slice_spacing=None, slice_image_count=3, window_presets=["骨窗"], fill_percent=80,
        ) for plane, slice_type in requested],
        window_presets=[WindowPreset("骨窗", width, center)],
    )


def _pipeline(*, enabled):
    return Pipeline(config=AppConfig(
        logging=LoggingConfig(console_enabled=False),
        pipeline=PipelineConfig(run_preflight=False, series_reuse_enabled=enabled),
        slice_part_config=load_slice_part_config(),
    ))


def _datasets(response):
    plans = response.context.get_result("planning")["plans"]
    outputs = response.context.get_result("reconstruction")["outputs"]
    assert len(plans) == len(outputs) == 5
    result = {}
    for plan, output in zip(plans, outputs):
        root = Path(output["window_outputs"]["骨窗"])
        series = {
            path.name: pydicom.dcmread(path)
            for path in sorted(root.glob("*.dcm"))
        }
        assert set(series) == {"1.dcm", "2.dcm", "3.dcm", "Localizer.dcm"}
        result[(plan["plane"], plan["slice_type"])] = series
    return result


def test_three_mpr_and_two_cpr_cache_hit_matches_full_reconstruction(
    tmp_path, monkeypatch, record_property,
):
    raw, labels_array, labels = _spine_source_and_segmentation()
    for name in ("A", "B", "C"):
        _prepare_series(tmp_path, name, raw)
    inference_calls = []

    def fake_inference(_stage, **kwargs):
        inference_calls.append(kwargs)
        input_image = sitk.ReadImage(str(kwargs["input_path"]))
        seg = sitk.GetImageFromArray(labels_array)
        seg.CopyInformation(input_image)
        destination = Path(kwargs["output_path"])
        sitk.WriteImage(seg, str(destination))
        (destination.parent / "dataset.json").write_text(
            json.dumps({"labels": labels}), encoding="utf-8",
        )

    monkeypatch.setattr(SegmentationStage, "_run_nnunet_inference", fake_inference)
    monkeypatch.setattr("src.core.series_reuse._model_identity", lambda *args: {"test-model": "spine-v1"})
    monkeypatch.setattr("src.core.series_reuse._engine_identity", lambda *args: {"test-engine": "v1"})
    monkeypatch.setattr("src.utils.Logger_utils._logger_manager.debug_file_enabled", False)
    original = _pipeline(enabled=True).run(_spine_request(tmp_path, "A"))
    assert original.success, original.errors
    assert len(inference_calls) == 1
    adopted_pipeline = _pipeline(enabled=True)
    _forbid_middle_stages(adopted_pipeline, monkeypatch)
    adopted = adopted_pipeline.run(_spine_request(tmp_path, "B", width=2200, center=350))
    assert adopted.success, adopted.errors
    assert len(inference_calls) == 1
    baseline = _pipeline(enabled=False).run(_spine_request(tmp_path, "C", width=2200, center=350))
    assert baseline.success, baseline.errors
    assert len(inference_calls) == 2
    actual_groups, expected_groups = _datasets(adopted), _datasets(baseline)
    assert actual_groups.keys() == expected_groups.keys()
    for key in actual_groups:
        for filename in actual_groups[key]:
            actual, expected = actual_groups[key][filename], expected_groups[key][filename]
            assert actual.PatientID == "patient-B"
            assert str(actual.FrameOfReferenceUID).endswith(".66.2")
            assert actual.SOPInstanceUID != expected.SOPInstanceUID
            assert actual.SeriesInstanceUID != expected.SeriesInstanceUID
            np.testing.assert_array_equal(actual.pixel_array, expected.pixel_array)
            for field in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
                np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
            assert (actual.Rows, actual.Columns) == (expected.Rows, expected.Columns)
            assert float(actual.WindowWidth) == 2200
            assert float(actual.WindowCenter) == 350
    metrics = {
        "shape_zyx": [240, 40, 40], "plans": "3 MPR + 2 CPR, 3 slices each",
        "external_inference": "synthetic; timings do not predict production speedup",
        "original_wall_seconds": original.total_time,
        "adopted_wall_seconds": adopted.total_time,
        "baseline_wall_seconds": baseline.total_time,
        "original_stage_seconds": original.timings,
        "adopted_stage_seconds": adopted.timings,
        "baseline_stage_seconds": baseline.timings,
        "cache_bytes": {
            name: sum(path.stat().st_size for path in (tmp_path / name / "cache").rglob("*")
                      if path.is_file()) for name in ("A", "B")
        },
    }
    (tmp_path / "measurements.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    record_property("series_reuse_spine_measurements", json.dumps(metrics))
    print("SERIES_REUSE_SMALL_SPINE=" + json.dumps(metrics))
