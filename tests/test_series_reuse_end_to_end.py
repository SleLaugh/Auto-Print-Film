"""真实 CPU 规划、采样、DICOM 与报告链路的跨序列复用回归。"""

import json
from pathlib import Path
import shutil
from unittest.mock import Mock

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
import pytest
import SimpleITK as sitk

from src.config import AppConfig, LoggingConfig, PipelineConfig, load_slice_part_config
from src.core.pipeline import Pipeline
from src.models import PlaneConfig, Request, WindowPreset
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.stages.stage_segmentation import SegmentationStage


def _lung_image():
    z, y, x = np.mgrid[:40, :64, :64]
    body = ((x - 32) / 29) ** 2 + ((y - 32) / 26) ** 2 <= 1
    lungs = np.zeros(body.shape, dtype=bool)
    for center_x in (21, 43):
        lungs |= (
            ((x - center_x) / 9) ** 2
            + ((y - 32) / 16) ** 2
            + ((z - 20) / 17) ** 2
        ) <= 1
    values = np.full(body.shape, -1000, dtype=np.int16)
    values[body] = 30
    values[lungs] = -700
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1, 1, 1.5))
    image.SetOrigin((-32, -32, -30))
    return image


def _prepare_series(parent: Path, name: str, image: sitk.Image):
    root = parent / name
    source = root / "dicom"
    source.mkdir(parents=True, exist_ok=True)
    nifti = root / "cache" / "nifti" / "raw_0000.nii.gz"
    nifti.parent.mkdir(parents=True)
    sitk.WriteImage(image, str(nifti))
    _write_source_dicom_volume(source, name, image)
    return root


def _write_source_dicom_volume(directory, name, image):
    """Use genuine signed CT slices so opted-in provenance can rebuild NIfTI."""
    values = sitk.GetArrayFromImage(image)
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    for index, pixels in enumerate(values):
        uid = f"1.2.826.0.1.3680043.10.999.{ord(name)}.1.{index + 1}"
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = directory / f"{index + 1:04d}.dcm"
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = uid
        dataset.Modality = "CT"
        dataset.PatientName = f"PATIENT^{name}"
        dataset.PatientID = f"patient-{name}"
        dataset.StudyInstanceUID = "1.2.826.0.1.3680043.10.999.100"
        dataset.SeriesInstanceUID = f"1.2.826.0.1.3680043.10.999.{ord(name)}"
        dataset.FrameOfReferenceUID = f"1.2.826.0.1.3680043.10.999.{ord(name)}.2"
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = list(image.TransformIndexToPhysicalPoint((0, 0, index)))
        dataset.ImageOrientationPatient = [*direction[:, 0], *direction[:, 1]]
        dataset.PixelSpacing = [image.GetSpacing()[1], image.GetSpacing()[0]]
        dataset.SliceThickness = image.GetSpacing()[2]
        dataset.SpacingBetweenSlices = image.GetSpacing()[2]
        dataset.Rows, dataset.Columns = pixels.shape
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.RescaleSlope = 1
        dataset.RescaleIntercept = 0
        dataset.PixelData = pixels.astype("<i2", copy=False).tobytes()
        dataset.save_as(path, enforce_file_format=True)


def _request(parent: Path, name: str, *, width=1500, center=-600, count=3):
    root = parent / name
    return Request(
        dicom_dir=str(root / "dicom"),
        output_dir=str(root / "肺部"),
        cache_dir=str(root / "cache"),
        series_cache_parent=str(parent),
        slice_part=SlicePartEnum.肺部,
        scan_thickness=1.5,
        plane_configs=[PlaneConfig(
            plane=PlaneTypeEnum.横状位,
            slice_thickness=1.5,
            slice_spacing=None,
            slice_image_count=count,
            window_presets=["肺窗"],
        )],
        window_presets=[WindowPreset("肺窗", width, center)],
    )


def _pipeline(*, enabled=True):
    return Pipeline(config=AppConfig(
        logging=LoggingConfig(console_enabled=False),
        pipeline=PipelineConfig(run_preflight=False, series_reuse_enabled=enabled),
        slice_part_config=load_slice_part_config(),
    ))


@pytest.fixture
def lung_reuse_case(tmp_path, monkeypatch):
    image = _lung_image()
    _prepare_series(tmp_path, "A", image)
    _prepare_series(tmp_path, "B", image)
    _prepare_series(tmp_path, "C", image)
    inference_calls = []

    def fake_inference(_stage, **kwargs):
        inference_calls.append(kwargs)
        raw = sitk.ReadImage(str(kwargs["input_path"]))
        segmentation = sitk.Cast((raw > -950) & (raw < -300), sitk.sitkUInt8)
        destination = Path(kwargs["output_path"])
        sitk.WriteImage(segmentation, str(destination))
        (destination.parent / "dataset.json").write_text(
            json.dumps({"labels": {"background": 0, "lung": 1}}), encoding="utf-8",
        )

    monkeypatch.setattr(SegmentationStage, "_run_nnunet_inference", fake_inference)
    monkeypatch.setattr("src.core.series_reuse._model_identity", lambda *args: {"test-model": "lung-v1"})
    monkeypatch.setattr("src.core.series_reuse._engine_identity", lambda *args: {"test-engine": "v1"})
    # These flags describe persisted debug artifacts, not algorithm behavior.
    monkeypatch.setattr("src.utils.Logger_utils._logger_manager.debug_file_enabled", False)
    return tmp_path, inference_calls


def _slices(response):
    result = response.context.get_result("reconstruction")
    window = Path(result["outputs"][0]["window_outputs"]["肺窗"])
    return [pydicom.dcmread(path) for path in sorted(window.glob("[0-9]*.dcm"))]


def _forbid_middle_stages(pipeline, monkeypatch):
    for stage in pipeline.stages:
        if stage.name in {
            "segmentation", "spatial_correction", "localization", "anatomical_analysis", "planning",
        }:
            monkeypatch.setattr(stage, "execute", Mock(side_effect=AssertionError(
                f"cache hit must bypass {stage.name}",
            )))


def test_lung_cache_hit_rewindows_and_uses_current_source_metadata(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    assert len(inference_calls) == 1
    first_slices = _slices(first)
    pipeline = _pipeline()
    _forbid_middle_stages(pipeline, monkeypatch)
    second = pipeline.run(_request(parent, "B", width=1000, center=-500))
    assert second.success, second.errors
    assert len(inference_calls) == 1
    second_slices = _slices(second)
    assert len(first_slices) == len(second_slices) == 3
    for old, current in zip(first_slices, second_slices):
        assert str(current.PatientID) == "patient-B"
        assert str(current.PatientName) == "PATIENT^B"
        assert str(current.FrameOfReferenceUID).endswith(".66.2")
        assert current.SOPInstanceUID != old.SOPInstanceUID
        assert current.SeriesInstanceUID != old.SeriesInstanceUID
        assert float(current.WindowWidth) == 1000
        assert float(current.WindowCenter) == -500
        np.testing.assert_array_equal(current.ImageOrientationPatient, old.ImageOrientationPatient)
        np.testing.assert_array_equal(current.ImagePositionPatient, old.ImagePositionPatient)
        assert (current.Rows, current.Columns) == (old.Rows, old.Columns)
    assert any(not np.array_equal(old.pixel_array, new.pixel_array)
               for old, new in zip(first_slices, second_slices))
    baseline = _pipeline(enabled=False).run(_request(parent, "C", width=1000, center=-500))
    assert baseline.success, baseline.errors
    for current, expected in zip(second_slices, _slices(baseline)):
        np.testing.assert_array_equal(current.pixel_array, expected.pixel_array)
        np.testing.assert_array_equal(current.ImageOrientationPatient, expected.ImageOrientationPatient)
        np.testing.assert_array_equal(current.ImagePositionPatient, expected.ImagePositionPatient)
    report = json.loads((parent / "B" / "肺部" / "report.json").read_text(encoding="utf-8"))
    assert "patient-A" not in json.dumps(report)


def test_reused_series_remains_a_donor_after_original_cache_is_deleted(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    second = _pipeline().run(_request(parent, "B"))
    assert second.success, second.errors
    assert len(inference_calls) == 1
    assert (parent / "B" / "cache" / "315" / "segmentation" / "seg.nii.gz").is_file()
    shutil.rmtree(parent / "A" / "cache")
    pipeline = _pipeline()
    _forbid_middle_stages(pipeline, monkeypatch)
    third = pipeline.run(_request(parent, "C"))
    assert third.success, third.errors
    assert len(inference_calls) == 1
    assert len(_slices(third)) == 3


def test_changed_plan_does_not_reuse_previous_slice_count(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    pipeline = _pipeline()
    planning = next(stage for stage in pipeline.stages if stage.name == "planning")
    plan_spy = Mock(wraps=planning.execute)
    monkeypatch.setattr(planning, "execute", plan_spy)
    second = pipeline.run(_request(parent, "B", count=5))
    assert second.success, second.errors
    assert plan_spy.call_count == 1
    assert len(_slices(second)) == 5
    # Full-cache reuse must miss; compatible segmentation reuse may still avoid inference.
    assert 1 <= len(inference_calls) <= 2


@pytest.mark.parametrize("difference", ("voxel", "origin"))
def test_image_or_geometry_difference_does_not_reuse_full_plan(
    lung_reuse_case, monkeypatch, difference,
):
    parent, inference_calls = lung_reuse_case
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    raw_path = parent / "B" / "cache" / "nifti" / "raw_0000.nii.gz"
    raw = sitk.ReadImage(str(raw_path))
    if difference == "voxel":
        raw[21, 32, 20] = -699
    else:
        origin = raw.GetOrigin()
        raw.SetOrigin((origin[0] + 1, origin[1], origin[2]))
    sitk.WriteImage(raw, str(raw_path))
    _write_source_dicom_volume(parent / "B" / "dicom", "B", raw)
    pipeline = _pipeline()
    planning = next(stage for stage in pipeline.stages if stage.name == "planning")
    plan_spy = Mock(wraps=planning.execute)
    monkeypatch.setattr(planning, "execute", plan_spy)
    second = pipeline.run(_request(parent, "B"))
    assert second.success, second.errors
    assert plan_spy.call_count == 1
    assert len(inference_calls) == 2


def test_disabled_switch_executes_original_pipeline(lung_reuse_case):
    parent, inference_calls = lung_reuse_case
    first = _pipeline(enabled=False).run(_request(parent, "A"))
    second = _pipeline(enabled=False).run(_request(parent, "B"))
    assert first.success, first.errors
    assert second.success, second.errors
    assert len(inference_calls) == 2
    assert len(_slices(second)) == 3


def _ready_marker(parent, series="A"):
    markers = list((parent / series / "cache" / "series_reuse" / "v1").glob("*/*/*/ready.json"))
    assert len(markers) == 1
    return markers[0]


def _planning_spy(pipeline, monkeypatch):
    planning = next(stage for stage in pipeline.stages if stage.name == "planning")
    spy = Mock(wraps=planning.execute)
    monkeypatch.setattr(planning, "execute", spy)
    return spy


@pytest.mark.parametrize("damage", ("corrupt", "missing", "unfinished"))
def test_incomplete_or_damaged_cache_falls_back_to_real_planning(
    lung_reuse_case, monkeypatch, damage,
):
    parent, inference_calls = lung_reuse_case
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    marker = _ready_marker(parent)
    ready = json.loads(marker.read_text(encoding="utf-8"))
    if damage == "unfinished":
        marker.rename(marker.with_name("unfinished.json"))
    else:
        artifact = next(name for name in ready["files"]
                        if name.startswith("hu/") and name.endswith(".npz"))
        hu_path = marker.parent / artifact
        if damage == "corrupt":
            hu_path.write_bytes(b"incomplete HU artifact")
        else:
            hu_path.unlink()
    pipeline = _pipeline()
    plan_spy = _planning_spy(pipeline, monkeypatch)
    second = pipeline.run(_request(parent, "B"))
    assert second.success, second.errors
    assert plan_spy.call_count == 1
    assert len(_slices(second)) == 3
    assert len(inference_calls) == 2


def test_plan_round_trip_retains_b_original_plan_and_label_sidecar(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    original = _pipeline().run(_request(parent, "A"))
    assert original.success, original.errors
    adopted = _pipeline().run(_request(parent, "B"))
    assert adopted.success, adopted.errors
    assert len(inference_calls) == 1
    label_map = parent / "B" / "cache" / "315" / "segmentation" / "dataset.json"
    assert json.loads(label_map.read_text(encoding="utf-8"))["labels"]["lung"] == 1
    # No model directory exists: a different plan must use the retained sidecar.
    changed_pipeline = _pipeline()
    plan_spy = _planning_spy(changed_pipeline, monkeypatch)
    changed = changed_pipeline.run(_request(parent, "A", count=5))
    assert changed.success, changed.errors
    assert plan_spy.call_count == 1
    assert len(_slices(changed)) == 5
    assert len(inference_calls) == 1
    # Force the final request to discover B rather than A's own old snapshot.
    shutil.rmtree(parent / "A" / "cache" / "series_reuse" / "v1")
    restored_pipeline = _pipeline()
    _forbid_middle_stages(restored_pipeline, monkeypatch)
    restored = restored_pipeline.run(_request(parent, "A"))
    assert restored.success, restored.errors
    assert len(_slices(restored)) == 3
    assert len(inference_calls) == 1
    for actual, expected in zip(_slices(restored), _slices(adopted)):
        np.testing.assert_array_equal(actual.pixel_array, expected.pixel_array)
        np.testing.assert_array_equal(actual.ImagePositionPatient, expected.ImagePositionPatient)


def test_published_typed_results_keep_existing_paths(lung_reuse_case):
    parent, _inference_calls = lung_reuse_case
    original = _pipeline().run(_request(parent, "A"))
    adopted = _pipeline().run(_request(parent, "B"))
    assert original.success, original.errors
    assert adopted.success, adopted.errors
    context = adopted.context
    typed_segmentation = context.stage_results["segmentation"].to_dict()
    assert Path(typed_segmentation["seg_path"]).is_file()
    assert typed_segmentation["seg_path"] == context.get_result("segmentation")["seg_path"]
    typed_spatial = context.stage_results["spatial_correction"].to_dict()
    for field in ("raw_for_reconstruction_path", "seg_for_reconstruction_path", "dataset_json_path"):
        assert Path(typed_spatial[field]).is_file(), (field, typed_spatial[field])
        assert typed_spatial[field] == context.get_result("spatial_correction")[field]


def test_small_lung_measurement_records_stage_times_and_cache_bytes(lung_reuse_case, record_property):
    parent, inference_calls = lung_reuse_case
    original = _pipeline().run(_request(parent, "A"))
    adopted = _pipeline().run(_request(parent, "B"))
    assert original.success, original.errors
    assert adopted.success, adopted.errors
    assert len(inference_calls) == 1
    metrics = {
        "shape_zyx": [40, 64, 64],
        "external_inference": "synthetic; timings do not predict production speedup",
        "original_wall_seconds": original.total_time,
        "adopted_wall_seconds": adopted.total_time,
        "original_stage_seconds": original.timings,
        "adopted_stage_seconds": adopted.timings,
        "cache_bytes": {
            name: sum(path.stat().st_size for path in (parent / name / "cache").rglob("*")
                      if path.is_file()) for name in ("A", "B")
        },
    }
    (parent / "measurements.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    record_property("series_reuse_measurements", json.dumps(metrics))
    print("SERIES_REUSE_SMALL_LUNG=" + json.dumps(metrics))
