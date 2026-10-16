"""椎间隙统一取景的 DICOM 端到端回归，包含实际像素标尺。"""

import copy
import math
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, PlaneTypeEnum, Request, SlicePartEnum
from src.models.enums import SliceType
from src.stages.stage_reconstruction import ReconstructionStage
from src.stages.stage_reporting import ReportingStage


def _point(values):
    return dict(zip("xyz", map(float, values)))


def _make_context(tmp_path, part, fill_percent):
    """前四组目标较小，末组明显更大，各组放置同毫米大小的 L 形标尺。"""

    shape = (56, 112, 112)
    raw = np.full(shape, -1000, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.indices(shape)
    plans = []
    coords = {}
    anchors = {}
    prefix = "L" if part == SlicePartEnum.腰椎 else "C"
    for index in range(5):
        name = f"{prefix}{index + 1}_{prefix}{index + 2}"
        anchor = np.array([46 + index * 2, 46 + index * 2, 6 + index * 10], dtype=float)
        angle = [-0.06, 0.02, 0.04, -0.02, 0.1][index]
        row = np.array([1.0, 0.0, 0.0])
        col = np.array([0.0, math.cos(angle), math.sin(angle)])
        normal = np.cross(row, col)
        radius_x, radius_y = (9, 10) if index < 4 else (22, 26)
        names = []
        for side, dz in enumerate((-3, 3)):
            label_id = index * 2 + side + 1
            label_name = f"target_{label_id}"
            names.append(label_name)
            labels[
                int(anchor[2] + dz),
                int(anchor[1] - radius_y):int(anchor[1] + radius_y + 1),
                int(anchor[0] - radius_x):int(anchor[0] + radius_x + 1),
            ] = label_id
            coords[label_name] = {"label_ids": [label_id], "found": True}
        u = x - anchor[0]
        v = (y - anchor[1]) * col[1] + (z - anchor[2]) * col[2]
        w = (y - anchor[1]) * normal[1] + (z - anchor[2]) * normal[2]
        marker = (
            (np.abs(w) <= 3.0)
            & (u >= -4) & (u < 4)
            & (v >= -5) & (v < 5)
            & ((u < -1) | (v < -2))
        )
        raw[marker] = 1600
        slices = [
            {
                "index": slice_index + 1,
                # 有意偏离标签中心，确保各间隙仍单独居中。
                "center": _point(anchor + normal * offset + row * 7 + col * 5),
                "row_dir": _point(row), "col_dir": _point(col),
                "normal": _point(normal),
            }
            for slice_index, offset in enumerate((-1.0, 1.0))
        ]
        plan = {
            "plan_id": f"{part.value}_{name}",
            "config_id": "配置01", "slice_part": part.value,
            "slice_type": "椎间隙", "plane": "横状位",
            "output_plane": "横状位", "group_name": name,
            "geometry_type": "plane_stack", "source_labels": names,
            "slice_thickness": 1.0, "slice_spacing": 2.0,
            "slice_image_count": 2,
            "extent_mm": {"width": 48 + index * 4, "height": 44 + index * 6},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "fill_percent": fill_percent, "window_presets": ["骨窗", "软组织窗"],
            "range_start": slices[0]["center"], "range_end": slices[-1]["center"],
            "path_points": [item["center"] for item in slices], "slices": slices,
        }
        plans.append(plan)
        anchors[name] = anchor

    raw_image = sitk.GetImageFromArray(raw)
    raw_path, seg_path = tmp_path / "raw.nii.gz", tmp_path / "seg.nii.gz"
    sitk.WriteImage(raw_image, str(raw_path))
    segmentation = sitk.GetImageFromArray(labels)
    segmentation.CopyInformation(raw_image)
    sitk.WriteImage(segmentation, str(seg_path))
    request = Request(
        dicom_dir=str(tmp_path / "source"), output_dir=str(tmp_path / "output"),
        slice_part=part,
        plane_configs=[PlaneConfig(
            plane=PlaneTypeEnum.横状位, slice_type=SliceType.椎间隙,
            slice_thickness=1.0, slice_spacing=2.0, slice_image_count=2,
            window_presets=["骨窗", "软组织窗"], fill_percent=fill_percent,
            config_id="配置01", output_plane="横状位",
        )],
    )
    context = ProcessingContext(request)
    context.set_result("localization", {
        "raw_for_reconstruction_path": str(raw_path),
        "seg_for_reconstruction_path": str(seg_path), "coords": coords,
    })
    context.set_result("planning", {"plans": plans})
    context.set_result("preprocessing", {"source_dicom_metadata": {
        "0020|000d": "1.2.826.0.1.3680043.2.1125.800",
    }})
    return context, anchors


def _dicom_center(dataset):
    directions = np.asarray(dataset.ImageOrientationPatient, dtype=float).reshape(2, 3).T
    spacing = np.asarray(dataset.PixelSpacing, dtype=float)[::-1]
    half_extent = (np.array([dataset.Columns, dataset.Rows]) - 1) * spacing / 2
    return np.asarray(dataset.ImagePositionPatient, dtype=float) + directions @ half_extent


@pytest.mark.parametrize("part", [SlicePartEnum.腰椎, SlicePartEnum.颈椎])
@pytest.mark.parametrize("fill_percent", [0, 80, 100])
def test_interspace_config_preserves_physical_scale_across_all_gaps_and_windows(
    tmp_path, part, fill_percent,
):
    context, anchors = _make_context(tmp_path, part, fill_percent)
    original_plans = copy.deepcopy(context.get_result("planning")["plans"])

    ReconstructionStage({"parallel": False}).execute(context)

    assert context.get_result("planning")["plans"] == original_plans
    outputs = context.get_result("reconstruction")["outputs"]
    assert len(outputs) == 5
    geometries = []
    marker_sizes = []
    for plan, output in zip(original_plans, outputs):
        name = plan["group_name"]
        assert output["slice_count"] == 2
        assert output["slice_thickness"] == 1.0
        assert output["slice_spacing"] == 2.0
        for window, directory in output["window_outputs"].items():
            assert {path.name for path in Path(directory).glob("*.dcm")} == {
                "1.dcm", "2.dcm", "Localizer.dcm",
            }
            for planned_slice in plan["slices"]:
                dataset = pydicom.dcmread(Path(directory) / f"{planned_slice['index']}.dcm")
                spacing = tuple(map(float, dataset.PixelSpacing))
                geometries.append((dataset.Rows, dataset.Columns, *spacing))
                # 共用前四组矩阵中位数；末组的大画布不再把前面几组一起撑大。
                assert (dataset.Rows, dataset.Columns) == (54, 55)
                framing = output["framing"][planned_slice["index"] - 1]
                assert framing["output_size_xy"] == [dataset.Columns, dataset.Rows]
                np.testing.assert_allclose(
                    spacing[::-1],
                    [framing["output_spacing_mm"][axis] for axis in "xy"],
                )
                np.testing.assert_allclose(
                    [framing["extent_mm"][axis] for axis in ("width", "height")],
                    (np.array([dataset.Columns, dataset.Rows]) - 1) * spacing[::-1],
                )
                assert int(dataset.InstanceNumber) == planned_slice["index"]
                assert float(dataset.SliceThickness) == 1.0
                assert float(dataset.SpacingBetweenSlices) == 2.0
                normal = np.array(list(planned_slice["normal"].values()))
                offset = -1.0 if planned_slice["index"] == 1 else 1.0
                np.testing.assert_allclose(
                    _dicom_center(dataset), anchors[name] + normal * offset, atol=1e-5,
                )
                expected_direction = [
                    *planned_slice["row_dir"].values(), *planned_slice["col_dir"].values(),
                ]
                np.testing.assert_allclose(dataset.ImageOrientationPatient, expected_direction, atol=1e-8)
                occupied = np.argwhere(dataset.pixel_array > 0.8 * 65535)
                assert len(occupied) > 0
                marker_sizes.append(np.ptp(occupied, axis=0) + 1)
                # 同一标尺在各组仍保持 8mm × 10mm 的物理比例，不能为填满画面拉伸。
                physical_size = (np.ptp(occupied, axis=0) + 1) * spacing
                np.testing.assert_allclose(physical_size, [10, 8], atol=2.0)
                if window == "软组织窗":
                    assert float(dataset.WindowWidth) == 400.0
                    assert float(dataset.WindowCenter) == 50.0
    np.testing.assert_allclose(geometries, np.tile(geometries[0], (len(geometries), 1)))
    assert np.all(np.ptp(marker_sizes, axis=0) <= 2)
    if fill_percent == 0:
        assert geometries[0][2:] == (1.0, 1.0)

    validation = ReportingStage()._validate_reconstruction(context.get_result("reconstruction"))
    assert validation["valid"], validation["errors"]
    report = ReportingStage()._build_delivery_report(context)
    plane = report["PlaneList"][0]
    assert plane["FillPercent"] == fill_percent
    for window in plane["WindowPresetList"]:
        assert [region["Region"] for region in window["RegionList"]] == [
            plan["group_name"] for plan in original_plans
        ]
        assert all(region["ActualSliceImageCount"] == 2 for region in window["RegionList"])


def test_same_plane_configs_keep_independent_geometry_and_optional_fill_percent(tmp_path):
    context, _ = _make_context(tmp_path, SlicePartEnum.腰椎, None)
    first_plans = context.get_result("planning")["plans"]
    second_plans = copy.deepcopy(first_plans)
    for index, plan in enumerate(second_plans):
        plan["config_id"] = "配置02"
        plan["plan_id"] += "_配置02"
        plan["output_plane"] = "横状位1"
        plan.pop("fill_percent")  # 省略与 null 都不缩放，但仍需统一矩阵。
        plan["output_spacing_mm"] = {"x": 2.0, "y": 2.0, "z": 0.0}
        plan["extent_mm"] = {"width": (26 + index * 2) * 2, "height": (22 + index * 3) * 2}
        plan["window_presets"] = ["骨窗"]
    context.get_result("planning")["plans"] = first_plans + second_plans
    original_plans = copy.deepcopy(context.get_result("planning")["plans"])

    ReconstructionStage({"parallel": False}).execute(context)

    assert context.get_result("planning")["plans"] == original_plans
    by_config = {"配置01": [], "配置02": []}
    for output in context.get_result("reconstruction")["outputs"]:
        dataset = pydicom.dcmread(Path(output["window_outputs"]["骨窗"]) / "1.dcm")
        by_config[output["config_id"]].append((
            dataset.Rows, dataset.Columns, *map(float, dataset.PixelSpacing),
        ))
        assert Path(output["window_outputs"]["骨窗"]).parent.parent.name == output["plane"]
    assert by_config["配置01"] == [(54, 55, 1.0, 1.0)] * 5
    assert by_config["配置02"] == [(28, 30, 2.0, 2.0)] * 5
