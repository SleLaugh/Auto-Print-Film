import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, PlaneTypeEnum, Request, WindowPreset
from src.reconstruction.hu_cache import (
    capture_plan,
    initialize_hu_cache,
    iter_cached_plans,
    validate_hu_cache,
)
from src.reconstruction.windowing import apply_window_to_image
from src.stages.stage_reconstruction import ReconstructionStage


def _payload():
    image = sitk.GetImageFromArray(
        np.linspace(-1200.125, 2400.875, 42, dtype=np.float32).reshape(1, 6, 7),
    )
    image.SetSpacing((0.7, 0.8, 1.0))
    image.SetOrigin((11.25, -20.5, 30.75))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    image.SetMetaData("0020|0032", "11.25\\-20.5\\30.75")
    image.SetMetaData("0020|0037", "0\\1\\0\\-1\\0\\0")
    image.SetMetaData("0008|2111", "原始HU背景")
    planned_slice = {
        "index": 1,
        "center": {"x": 11.25, "y": -20.5, "z": 30.75},
        "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
        "col_dir": {"x": -1.0, "y": 0.0, "z": 0.0},
        "framing": {"output_spacing_mm": {"x": 0.7, "y": 0.8}},
    }
    plan = {
        "plan_id": "测试计划", "config_id": "配置01",
        "plane": "横状位", "output_plane": "横状位",
        "group_name": "whole_range", "slice_type": "MPR",
        "geometry_type": "plane_stack", "slices": [planned_slice],
        "extent_mm": {"width": 4.9, "height": 4.8},
        "output_spacing_mm": {"x": 0.7, "y": 0.8, "z": 0.0},
        "slice_thickness": 4.0, "slice_spacing": 3.0,
        "window_presets": ["骨窗"],
    }
    background = sitk.GetImageFromArray(sitk.GetArrayFromImage(image)[0])
    background.SetSpacing(image.GetSpacing()[:2])
    background.SetOrigin(image.GetOrigin()[:2])
    background.SetDirection((0.0, -1.0, 1.0, 0.0))
    for key in image.GetMetaDataKeys():
        background.SetMetaData(key, image.GetMetaData(key))
    localizer = {
        "background_image": background,
        "line_alpha": np.full((6, 7), 0.25, dtype=np.float32),
        "legacy_line_value": 2400.875, "background_kind": "orthogonal_plane",
    }
    return plan, [(planned_slice, image)], localizer


def _capture(path):
    plan, slices, localizer = _payload()
    initialize_hu_cache(path, [plan])
    capture_plan(path, plan, slices, localizer)
    return plan, slices, localizer


def test_hu_and_localizer_roundtrip_preserves_values_geometry_metadata(tmp_path):
    plan, slices, localizer = _capture(tmp_path)
    assert validate_hu_cache(tmp_path)
    cached, = iter_cached_plans(tmp_path)
    assert cached["plan"] == plan
    assert cached["sampled_slices"][0][0] == slices[0][0]
    for original, restored in (
        (slices[0][1], cached["sampled_slices"][0][1]),
        (localizer["background_image"], cached["localizer"]["background_image"]),
    ):
        assert original.GetPixelID() == restored.GetPixelID()
        np.testing.assert_array_equal(sitk.GetArrayFromImage(original), sitk.GetArrayFromImage(restored))
        assert original.GetSpacing() == restored.GetSpacing()
        assert original.GetOrigin() == restored.GetOrigin()
        assert original.GetDirection() == restored.GetDirection()
        assert {key: original.GetMetaData(key) for key in original.GetMetaDataKeys()} == {
            key: restored.GetMetaData(key) for key in restored.GetMetaDataKeys()
        }
    np.testing.assert_array_equal(localizer["line_alpha"], cached["localizer"]["line_alpha"])


@pytest.mark.parametrize("failure", ["missing", "corrupt", "incomplete", "escape"])
def test_complete_validation_rejects_missing_corrupt_and_partial_cache(tmp_path, failure):
    _capture(tmp_path)
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if failure == "missing":
        next(tmp_path.glob("*-slice-*.npz")).unlink()
    elif failure == "corrupt":
        with next(tmp_path.glob("*-slice-*.npz")).open("ab") as stream:
            stream.write(b"changed")
    elif failure == "incomplete":
        index["expected_plan_ids"].append("never captured")
        index_path.write_text(json.dumps(index))
    else:
        descriptor = index["plans"]["测试计划"]
        descriptor["file"] = "../outside.json"
        index_path.write_text(json.dumps(index))
    assert not validate_hu_cache(tmp_path)


def test_empty_plan_set_is_a_valid_cache(tmp_path):
    initialize_hu_cache(tmp_path, [])
    assert validate_hu_cache(tmp_path)
    assert list(iter_cached_plans(tmp_path)) == []


def test_array_decode_is_validated_even_when_checksum_matches(tmp_path):
    _capture(tmp_path)
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record_path = tmp_path / index["plans"]["测试计划"]["file"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    image = record["slices"][0]["image"]
    array_path = tmp_path / image["file"]
    array_path.write_bytes(b"not an npz")
    image["sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    index["plans"]["测试计划"]["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index))
    assert not validate_hu_cache(tmp_path)


def test_cached_render_rewindows_hu_and_uses_current_identity_without_resampling(tmp_path):
    hu = tmp_path / "bundle" / "hu"
    plan, slices, _ = _capture(hu)
    window = WindowPreset("新窗", window_width=4200.5, window_center=-50.25)
    stage = ReconstructionStage({"parallel": False})
    uids = []
    for run in (1, 2):
        request = Request(
            dicom_dir="/missing/source", output_dir=str(tmp_path / f"output{run}"),
            plane_configs=[PlaneConfig(PlaneTypeEnum.横状位, 4.0, 3.0, ["新窗"])],
            window_presets=[window],
        )
        context = ProcessingContext(request)
        context.metadata["series_reuse_session"] = SimpleNamespace(hit=True, cache_dir=hu.parent)
        context.set_result("preprocessing", {"source_dicom_metadata": {
            "0020|000d": "1.2.826.0.1.100", "0010|0010": "Current^Patient",
            "0010|0020": f"CURRENT-{run}",
        }})
        # No original raw, seg, localization or framer is needed after a hit.
        with mock.patch("src.stages.stage_reconstruction.sitk.ReadImage", side_effect=AssertionError("raw read")), mock.patch.object(
            stage, "_resampler_for_plan", side_effect=AssertionError("resampling"),
        ):
            stage.execute(context)
        result = context.get_result("reconstruction")
        assert result["image_count"] == 1
        assert result["outputs"][0]["framing"][0]["output_spacing_mm"] == {"x": 0.7, "y": 0.8}
        output_dir = Path(result["outputs"][0]["window_outputs"]["新窗"])
        assert output_dir.is_relative_to(Path(request.output_dir))
        dicom = pydicom.dcmread(output_dir / "1.dcm")
        expected = sitk.GetArrayFromImage(apply_window_to_image(slices[0][1], window))[0]
        np.testing.assert_array_equal(dicom.pixel_array, expected)
        assert dicom.PatientID == f"CURRENT-{run}"
        assert str(dicom.PatientName) == "Current^Patient"
        assert str(dicom.StudyInstanceUID) == "1.2.826.0.1.100"
        assert float(dicom.WindowWidth) == window.window_width
        assert float(dicom.SliceThickness) == plan["slice_thickness"]
        assert list(map(float, dicom.ImagePositionPatient)) == list(slices[0][1].GetOrigin())
        localizer = pydicom.dcmread(output_dir / "Localizer.dcm")
        assert localizer.PatientID == f"CURRENT-{run}"
        assert float(localizer.WindowCenter) == window.window_center
        uids.append((str(dicom.SeriesInstanceUID), str(dicom.SOPInstanceUID)))
    assert uids[0][0] != uids[1][0]
    assert uids[0][1] != uids[1][1]


def test_capture_failure_disables_reuse_but_preserves_normal_output(tmp_path):
    plan, slices, localizer = _payload()
    stage = ReconstructionStage({"parallel": False})
    session = SimpleNamespace(disable=mock.Mock())
    writes = []
    with mock.patch.object(stage, "_resampler_for_plan") as resampler, mock.patch.object(
        stage, "_build_plan_localizer_render", return_value=None,
    ), mock.patch("src.stages.stage_reconstruction.capture_plan", side_effect=OSError("full disk")), mock.patch.object(
        stage, "_write_plan_windows", side_effect=lambda **kwargs: writes.append(kwargs),
    ):
        resampler.return_value.sample_slice.return_value = slices[0][1]
        result = stage._execute_plan(
            image=object(), plan=copy.deepcopy(plan), output_root=tmp_path / "output",
            study_uid="1.2.3", source_metadata={},
            hu_cache_dir=tmp_path / "hu", hu_cache_session=session,
        )
    session.disable.assert_called_once()
    assert len(writes) == 1
    assert result["image_count"] == 1


def test_normal_stage_capture_and_replay_preserve_all_output_pixels_and_geometry(tmp_path):
    from tests.test_stage7_reconstruction import Stage7ReconstructionTest

    context = Stage7ReconstructionTest._make_context_with_one_plan(str(tmp_path))
    bundle = tmp_path / "bundle"
    session = SimpleNamespace(hit=False, cache_dir=bundle, disabled=False, disable=mock.Mock())
    context.metadata["series_reuse_session"] = session
    context.set_result("preprocessing", {"source_dicom_metadata": {
        "0020|000d": "1.2.826.0.1.101", "0010|0020": "CURRENT",
    }})
    stage = ReconstructionStage({"parallel": False})
    stage.execute(context)
    session.disable.assert_not_called()
    assert validate_hu_cache(bundle / "hu")
    original_output = Path(context.request.output_dir)
    replay = ProcessingContext(copy.deepcopy(context.request))
    replay.request.output_dir = str(tmp_path / "replay")
    replay.metadata["series_reuse_session"] = SimpleNamespace(hit=True, cache_dir=bundle)
    replay.set_result("preprocessing", copy.deepcopy(context.get_result("preprocessing")))
    with mock.patch("src.stages.stage_reconstruction.sitk.ReadImage", side_effect=AssertionError("raw read")):
        stage.execute(replay)
    originals = {path.relative_to(original_output): path for path in original_output.rglob("*.dcm")}
    restored = {path.relative_to(Path(replay.request.output_dir)): path for path in Path(replay.request.output_dir).rglob("*.dcm")}
    assert originals.keys() == restored.keys()
    assert any(path.name == "Localizer.dcm" for path in originals)
    for name, original_path in originals.items():
        original = pydicom.dcmread(original_path)
        cached = pydicom.dcmread(restored[name])
        np.testing.assert_array_equal(original.pixel_array, cached.pixel_array)
        for tag in (
            "Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient",
            "SliceThickness", "SpacingBetweenSlices", "WindowWidth", "WindowCenter",
            "RescaleSlope", "RescaleIntercept",
        ):
            assert getattr(original, tag, None) == getattr(cached, tag, None)
