"""701 标签组合必须贯通回正、体表区域、三状位规划与实际 DICOM。"""

import itertools
import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import LocalizationStage, _LOCALIZATION_WORKSPACE_KEY
from src.stages.stage_planning import PlanningStage
from src.stages.stage_reconstruction import ReconstructionStage
from src.stages.stage_spatial_correction import SpatialCorrectionStage


LABELS = {
    "background": 0, "liver": 1, "hip_left": 2, "hip_right": 3,
    "kidney_left": 4, "kidney_right": 5, "femur_left": 6, "femur_right": 7,
}


def _fixture(directory, part, fill_percent, monkeypatch, *, missing=None):
    z, y, x = np.ogrid[:150, :80, :100]
    body = np.broadcast_to(((x - 50) / 44) ** 2 + ((y - 40) / 32) ** 2 <= 1, (150, 80, 100))
    labels = np.zeros(body.shape, dtype=np.uint8)
    labels[95:121, 23:53, 36:67] = 1
    labels[20:61, 38:61, 25:34] = 2
    labels[20:63, 38:61, 66:75] = 3
    labels[75:101, 40:52, 34:44] = 4
    labels[70:96, 40:52, 56:66] = 5
    labels[8:31, 40:53, 29:39] = 6
    labels[8:29, 40:53, 61:71] = 7
    if missing:
        for name in ([missing] if isinstance(missing, str) else missing):
            labels[labels == LABELS[name]] = 0
    image = sitk.GetImageFromArray(labels)
    image.SetSpacing((2.0, 2.0, 2.0))
    image.SetOrigin((-100.0, -80.0, -40.0))
    angle = np.deg2rad(11.0)
    image.SetDirection((np.cos(angle), -np.sin(angle), 0., np.sin(angle), np.cos(angle), 0., 0., 0., 1.))
    raw = sitk.GetImageFromArray(np.where(body, 40, -1000).astype(np.int16))
    raw.CopyInformation(image)
    raw_path, seg_path = directory / "raw.nii.gz", directory / "seg.nii.gz"
    sitk.WriteImage(raw, str(raw_path))
    sitk.WriteImage(image, str(seg_path))
    (directory / "dataset.json").write_text(json.dumps({"labels": LABELS}), encoding="utf-8")
    request = Request(
        dicom_dir=str(directory / "unused_dicom"), output_dir=str(directory / "output"),
        slice_part=part,
        plane_configs=[PlaneConfig(
            plane=plane, slice_thickness=2., slice_spacing=None, slice_image_count=3,
            window_presets=["腹部窗"], fill_percent=fill_percent,
        ) for plane in PlaneTypeEnum],
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {
        "nifti_path": str(raw_path), "source_dicom_metadata": {"0020|000d": "1.2.3.4"},
    })
    context.set_result("segmentation", {"seg_path": str(seg_path), "model": {"dataset_id": 701}})
    for module in ("localization_execution", "stage_anatomical_analysis", "stage_planning"):
        monkeypatch.setattr(f"src.stages.{module}.is_debug_file_enabled", lambda: False)
    return context, sitk.ReadImage(str(seg_path)), labels, body


def _axes(frame):
    return np.column_stack([
        [frame[key][axis] for axis in "xyz"]
        for key in ("left_right_axis", "anterior_posterior_axis", "superior_inferior_axis")
    ])


def _points(mask, image):
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    return np.asarray(image.GetOrigin()) + (np.argwhere(mask)[:, ::-1] * image.GetSpacing()) @ direction.T


@pytest.mark.parametrize("part", (SlicePartEnum.上腹部, SlicePartEnum.下腹部, SlicePartEnum.全腹部))
@pytest.mark.parametrize("fill_percent", (0., 50., 100.))
def test_abdomen_bounds_and_body_survive_all_three_dicom_series(tmp_path, monkeypatch, part, fill_percent):
    context, image, labels, body = _fixture(tmp_path, part, fill_percent, monkeypatch)
    config = {"correction_requested": False}
    for stage in (
        SpatialCorrectionStage(config), LocalizationStage(config),
        AnatomicalAnalysisStage(config), PlanningStage(),
    ):
        stage.execute(context)

    anatomical = context.get_result("anatomical_analysis")
    pose = anatomical["anatomical_geometry"]["global_frame"]
    region = anatomical["anatomical_regions"]["default"]
    frame = region["global_frame"]
    axes = _axes(frame)
    assert region["transverse_source"] == "body"
    assert region["constrained"]
    assert _LOCALIZATION_WORKSPACE_KEY not in context.metadata
    np.testing.assert_allclose(_axes(pose), axes)
    # Oracle reads actual label physical points, independent of region resolver.
    upper_edge = {name: float((_points(labels == value, image) @ axes[:, 2]).max())
                  for name, value in LABELS.items() if value}
    expected_upper = (upper_edge["liver"] if part is not SlicePartEnum.下腹部 else
                      max(upper_edge["kidney_left"], upper_edge["kidney_right"]) + 30.)
    expected_lower = max(upper_edge[name] for name in (
        ("hip_left", "hip_right") if part is SlicePartEnum.上腹部 else ("femur_left", "femur_right")
    ))
    assert frame["bounds"]["superior_inferior"] == pytest.approx({
        "minimum": expected_lower, "maximum": expected_upper,
    })
    # The body is substantially wider/deeper than every organ/bone union.
    body_points = _points(body, image) @ axes
    selected = body_points[(body_points[:, 2] >= expected_lower) & (body_points[:, 2] <= expected_upper)]
    for index, axis in enumerate(("left_right", "anterior_posterior")):
        assert frame["bounds"][axis]["minimum"] <= selected[:, index].min() + 1e-5
        assert frame["bounds"][axis]["maximum"] >= selected[:, index].max() - 1e-5

    plans = context.get_result("planning")["plans"]
    assert len(plans) == 3
    for plan in plans:
        assert plan["region_geometry"]["frame"] == frame
        assert len(plan["slices"]) == 3
        assert plan["slice_part"] == part.value
        assert plan["localizer_plane"] == ("冠状位" if plan["plane"] == "矢状位" else "矢状位")

    ReconstructionStage({"parallel": False}).execute(context)
    outputs = context.get_result("reconstruction")["outputs"]
    assert len(outputs) == 3
    bounds = frame["bounds"]
    corners = np.asarray(list(itertools.product(*[
        [bounds[axis]["minimum"], bounds[axis]["maximum"]]
        for axis in ("left_right", "anterior_posterior", "superior_inferior")
    ]))) @ axes.T
    for output in outputs:
        directory = Path(output["window_outputs"]["腹部窗"])
        series = [pydicom.dcmread(directory / f"{index}.dcm") for index in range(1, 4)]
        assert (directory / "Localizer.dcm").is_file()
        assert len({str(ds.SeriesInstanceUID) for ds in series}) == 1
        assert len({(ds.Rows, ds.Columns, tuple(ds.PixelSpacing)) for ds in series}) == 1
        for ds in series:
            assert ds.pixel_array.size > 0
            origin = np.asarray(ds.ImagePositionPatient, dtype=float)
            right, down = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
            columns = (corners - origin) @ right / float(ds.PixelSpacing[1])
            rows = (corners - origin) @ down / float(ds.PixelSpacing[0])
            assert columns.min() >= -0.5 - 1e-4
            assert columns.max() <= ds.Columns - 0.5 + 1e-4
            assert rows.min() >= -0.5 - 1e-4
            assert rows.max() <= ds.Rows - 0.5 + 1e-4


@pytest.mark.parametrize("missing", ("kidney_left", "hip_right", "femur_left"))
def test_missing_required_lower_abdomen_landmark_fails_before_planning(tmp_path, monkeypatch, missing):
    context, _, _, _ = _fixture(tmp_path, SlicePartEnum.下腹部, 100., monkeypatch, missing=missing)
    with pytest.raises(ValueError, match="缺失|为空|未找到|不存在"):
        SpatialCorrectionStage({"correction_requested": False}).execute(context)
    assert context.get_result("planning") is None


def test_upper_abdomen_does_not_require_kidneys_for_its_liver_and_hip_boundaries(tmp_path, monkeypatch):
    context, _, _, _ = _fixture(tmp_path, SlicePartEnum.上腹部, 100., monkeypatch, missing="kidney_left")
    config = {"correction_requested": False}
    for stage in (
        SpatialCorrectionStage(config), LocalizationStage(config),
        AnatomicalAnalysisStage(config), PlanningStage(),
    ):
        stage.execute(context)
    assert set(context.get_result("localization")["label_targets"]) == {"liver", "hip_left", "hip_right"}
    assert len(context.get_result("planning")["plans"]) == 3


@pytest.mark.parametrize("missing", ("liver", "hip_left", "hip_right", "femur_left", "femur_right"))
def test_missing_required_whole_abdomen_landmark_fails_before_planning(tmp_path, monkeypatch, missing):
    context, _, _, _ = _fixture(tmp_path, SlicePartEnum.全腹部, 100., monkeypatch, missing=missing)
    with pytest.raises(ValueError, match="缺失|为空|未找到|不存在"):
        SpatialCorrectionStage({"correction_requested": False}).execute(context)
    assert context.get_result("planning") is None


def test_whole_abdomen_does_not_require_kidneys_or_add_their_thirty_mm_offset(tmp_path, monkeypatch):
    context, image, labels, _ = _fixture(
        tmp_path, SlicePartEnum.全腹部, 100., monkeypatch,
        missing=("kidney_left", "kidney_right"),
    )
    config = {"correction_requested": False}
    for stage in (
        SpatialCorrectionStage(config), LocalizationStage(config),
        AnatomicalAnalysisStage(config), PlanningStage(),
    ):
        stage.execute(context)
    assert set(context.get_result("localization")["label_targets"]) == {
        "liver", "hip_left", "hip_right", "femur_left", "femur_right",
    }
    region = context.get_result("anatomical_analysis")["anatomical_regions"]["default"]
    frame = region["global_frame"]
    si_axis = _axes(frame)[:, 2]
    assert frame["bounds"]["superior_inferior"] == pytest.approx({
        "maximum": float((_points(labels == LABELS["liver"], image) @ si_axis).max()),
        "minimum": max(float((_points(labels == LABELS[name], image) @ si_axis).max())
                       for name in ("femur_left", "femur_right")),
    })
    assert len(context.get_result("planning")["plans"]) == 3
