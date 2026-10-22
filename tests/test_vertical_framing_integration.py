"""去除上下源图外填充后，真实第七阶段成片和定位图仍保持物理几何。"""

import copy
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.reconstruction.framing import TargetFramer, prepare_interspace_framing
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image
from src.stages.stage_reconstruction import ReconstructionStage


def _point(values):
    return dict(zip("xyz", map(float, values)))


def _fixture(fill_percent, slice_type):
    # 所有源体素均高于填充值，独立通过采样 HU 识别整行是否完全超出源图。
    # 世界坐标倾斜且源 x/z 反向存储，防止实现误用世界 z 或数组首尾。
    z, y, x = np.indices((20, 34, 30))
    raw = sitk.GetImageFromArray((40 + 2 * x + 3 * y + 7 * z).astype(np.float32))
    raw.SetSpacing((0.7, 1.1, 1.4))
    raw.SetOrigin((-18.0, 22.0, -50.0))
    yaw = np.deg2rad(17.0)
    direction = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ]) @ np.diag([-1.0, 1.0, -1.0])
    raw.SetDirection(tuple(direction.ravel()))
    labels = np.zeros((20, 34, 30), dtype=np.uint8)
    labels[1:5, 5:16, 4:26] = 1
    labels[10:17, 18:28, 4:26] = 1
    seg = sitk.GetImageFromArray(labels)
    seg.CopyInformation(raw)

    pitch = np.deg2rad(30.0)
    row = direction[:, 0]
    col = direction @ np.array([0.0, np.cos(pitch), np.sin(pitch)])
    normal = direction @ np.array([0.0, -np.sin(pitch), np.cos(pitch)])
    origin = np.array(raw.GetOrigin())
    source_center = origin + direction @ np.array([10.15, 18.15, 13.3])
    source_depth = float((source_center - origin) @ normal)
    plan = {
        "plan_id": "vertical_padding", "slice_part": "颈椎",
        "slice_type": slice_type, "plane": "横状位", "geometry_type": "plane_stack",
        "group_name": "whole_range" if slice_type == "MPR" else "C3_C4",
        "source_labels": ["target"], "fill_percent": fill_percent,
        "window_presets": ["骨窗", "软组织窗"],
        "slice_thickness": 0.2, "slice_spacing": 15.0,
        "extent_mm": {"width": 41.3, "height": 75.9},
        "output_spacing_mm": {"x": 0.7, "y": 1.1, "z": 0.0},
        "slices": [{
            "index": index,
            "center": _point(source_center + normal * (depth - source_depth)),
            "row_dir": _point(row), "col_dir": _point(col), "normal": _point(normal),
        } for index, depth in enumerate((-12.0, 3.0, 18.0), start=1)],
    }
    plan["range_start"] = plan["slices"][0]["center"]
    plan["range_end"] = plan["slices"][-1]["center"]
    plan["path_points"] = [plan["range_start"], plan["range_end"]]
    return raw, seg, plan, {"target": {"label_ids": [1]}}


def _original_samples(raw, seg, plan, coords):
    shared = (
        prepare_interspace_framing(seg, coords, [plan])
        if plan["slice_type"] == "椎间隙" else None
    )
    framer = TargetFramer(seg, coords, plan, shared_geometry=shared)
    originals = []
    for item in plan["slices"]:
        sampling_plan, framed = framer.plane_geometry(plan, item)
        sampled = PlaneStackResampler().sample_slice(raw, sampling_plan, framed)
        array = sitk.GetArrayFromImage(sampled)[0]
        supported = np.flatnonzero(np.any(array > -1024.0, axis=1))
        assert len(supported) > 0
        first, last = map(int, (supported[0], supported[-1]))
        # 在此单偏移厚层 fixture 中，有效支持行应连续，且源数据都可见。
        np.testing.assert_array_equal(supported, np.arange(first, last + 1))
        originals.append((sampled, framed, first, last))
    return originals


def _series_directory(root, plan, window):
    path = Path(root) / plan["plane"] / window
    return path if plan["slice_type"] == "MPR" else path / plan["group_name"]


@pytest.mark.parametrize("fill_percent", (0, 80))
@pytest.mark.parametrize("slice_type", ("MPR", "椎间隙"))
def test_stage7_crops_only_unsupported_rows_and_preserves_dicom_geometry(
    tmp_path, fill_percent, slice_type,
):
    raw, seg, plan, coords = _fixture(fill_percent, slice_type)
    original_plan = copy.deepcopy(plan)
    originals = _original_samples(raw, seg, plan, coords)
    expected_heights = [last - first + 1 for _, _, first, last in originals]
    assert len(set(expected_heights)) > 1
    assert any(first > 0 for _, _, first, _ in originals)
    assert any(last < image.GetSize()[1] - 1 for image, _, _, last in originals)

    raw_path, seg_path = tmp_path / "raw.mha", tmp_path / "seg.mha"
    sitk.WriteImage(raw, str(raw_path))
    sitk.WriteImage(seg, str(seg_path))
    request = Request(
        str(tmp_path / "missing_source_dicoms"), str(tmp_path / "actual"),
        slice_part=SlicePartEnum.颈椎,
        plane_configs=[PlaneConfig(
            PlaneTypeEnum.横状位, 0.2, 15.0, plan["window_presets"],
            fill_percent=fill_percent, slice_type=SliceType(slice_type),
        )],
    )
    context = ProcessingContext(request)
    context.set_result("localization", {
        "raw_for_reconstruction_path": str(raw_path),
        "seg_for_reconstruction_path": str(seg_path), "coords": coords,
    })
    context.set_result("planning", {"plans": [plan]})
    ReconstructionStage(config={"parallel": True}).execute(context)
    result = context.get_result("reconstruction")["outputs"][0]

    # 不传目标取景器构建同一原计划的定位图，验证裁成片不改变其全幅定位线。
    baseline_root = tmp_path / "baseline"
    ReconstructionStage()._execute_plan(raw, plan, baseline_root, "1.2.3", {})

    plane_locations = []
    for index, (original, framed, first, last) in enumerate(originals, start=1):
        columns = original.GetSize()[0]
        height = last - first + 1
        expected_origin = original.TransformIndexToPhysicalPoint((0, first, 0))
        axes = np.array(original.GetDirection()).reshape(3, 3)
        normal = axes[:, 2]
        expected_center = np.array(expected_origin) + axes[:, 0] * (
            (columns - 1) * original.GetSpacing()[0] / 2
        ) + axes[:, 1] * ((height - 1) * original.GetSpacing()[1] / 2)
        expected_location = float(np.array(original.GetOrigin()) @ normal)
        datasets = []
        for window in plan["window_presets"]:
            directory = _series_directory(request.output_dir, plan, window)
            dataset = pydicom.dcmread(directory / f"{index}.dcm")
            datasets.append(dataset)
            assert (dataset.Columns, dataset.Rows) == (columns, height)
            np.testing.assert_allclose(dataset.ImagePositionPatient, expected_origin, atol=1e-7)
            np.testing.assert_allclose(
                dataset.ImageOrientationPatient, axes[:, :2].T.ravel(), atol=1e-8,
            )
            np.testing.assert_allclose(
                dataset.PixelSpacing, original.GetSpacing()[1::-1], atol=1e-8,
            )
            assert float(dataset.SliceLocation) == pytest.approx(expected_location, abs=1e-7)
            assert float(dataset.SliceThickness) == plan["slice_thickness"]
            assert float(dataset.SpacingBetweenSlices) == plan["slice_spacing"]
            assert int(dataset.InstanceNumber) == index
            windowed = sitk.GetArrayFromImage(apply_window_to_image(original, window))[0]
            np.testing.assert_array_equal(dataset.pixel_array, windowed[first:last + 1])
            assert np.any(dataset.pixel_array[0] > 0)
            assert np.any(dataset.pixel_array[-1] > 0)
        for field in (
            "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
            "Rows", "Columns", "SliceLocation",
        ):
            assert getattr(datasets[0], field) == getattr(datasets[1], field)
        plane_locations.append(float(datasets[0].SliceLocation))

        framing = result["framing"][index - 1]
        assert framing["output_size_xy"] == [columns, height]
        np.testing.assert_allclose(
            [framing["center"][axis] for axis in "xyz"], expected_center, atol=1e-7,
        )
        np.testing.assert_allclose(
            [framing["output_spacing_mm"][axis] for axis in "xy"],
            original.GetSpacing()[:2], atol=1e-10,
        )
        assert framing["extent_mm"]["height"] == pytest.approx(
            (height - 1) * original.GetSpacing()[1], abs=1e-8,
        )
        # 不对称裁剪只移动面内中心，不能挪动切层的法向位置。
        np.testing.assert_allclose(
            (expected_center - np.array([framed["center"][axis] for axis in "xyz"])) @ normal,
            0.0, atol=1e-8,
        )

    np.testing.assert_allclose(np.diff(plane_locations), plan["slice_spacing"], atol=1e-7)
    for window in plan["window_presets"]:
        actual = pydicom.dcmread(_series_directory(request.output_dir, plan, window) / "Localizer.dcm")
        baseline = pydicom.dcmread(_series_directory(baseline_root, plan, window) / "Localizer.dcm")
        for field in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"):
            assert getattr(actual, field) == getattr(baseline, field)
        np.testing.assert_array_equal(actual.pixel_array, baseline.pixel_array)
    assert plan == original_plan
