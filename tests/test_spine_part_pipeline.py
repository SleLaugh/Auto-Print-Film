"""胸腰椎 C7–S1 与全脊柱 C2–S1 贯通定位、解剖、MPR / CPR 和 DICOM 链路。"""

import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut
import pytest
import SimpleITK as sitk

from src.config import load_slice_part_config
from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.reconstruction.spine_localizer import build_spine_localizer_surface
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import LocalizationStage
from src.stages.stage_planning import PlanningStage
from src.stages.stage_reconstruction import ReconstructionStage
from src.stages.stage_spatial_correction import SpatialCorrectionStage


@pytest.mark.parametrize("sacral_label", ("vertebrae_S1", "sacrum"))
@pytest.mark.parametrize("fill_percent", (0.0, 80.0, 100.0))
@pytest.mark.parametrize("part,cervical_start", (
    (SlicePartEnum.胸腰椎, 7), (SlicePartEnum.全脊柱, 2),
))
def test_whole_spine_reaches_all_five_series_with_real_sacral_label(
    tmp_path, monkeypatch, sacral_label, fill_percent, part, cervical_start,
):
    """同次请求完整定位 19 / 24 标签，兼容 S1 别名并实际写出五套切片。"""

    label_names = (
        [f"vertebrae_C{index}" for index in range(cervical_start, 8)]
        + [f"vertebrae_T{index}" for index in range(1, 13)]
        + [f"vertebrae_L{index}" for index in range(1, 6)]
        + [sacral_label]
    )
    segmentation = np.zeros((12 * (len(label_names) + 1), 40, 40), dtype=np.uint8)
    source = np.full(segmentation.shape, -1000.0, dtype=np.float32)
    excluded_label = "vertebrae_C6" if part is SlicePartEnum.胸腰椎 else "vertebrae_S2"
    labels = {"background": 0, excluded_label: 99}
    for index, name in enumerate(label_names):
        label_id, center_z = index + 1, segmentation.shape[0] - 12 - index * 12
        labels[name] = label_id
        # 前部椎体与后部棘突具有不同灰度，防止取景仅靠标签数量蒙混过关。
        body = np.s_[center_z - 3:center_z + 4, 12:20, 10:27]
        posterior = np.s_[center_z - 2:center_z + 3, 20:29, 17:20]
        segmentation[body] = label_id
        segmentation[posterior] = label_id
        source[body] = 100.0
        source[posterior] = 900.0
    # 同一模型可见的范围外标注不应进入本次定位或切割。
    segmentation[-6:-1, 12:20, 10:27] = 99
    image = sitk.GetImageFromArray(segmentation)
    image.SetSpacing((0.8, 0.8, 1.2))
    raw = sitk.GetImageFromArray(source)
    raw.CopyInformation(image)
    seg_path, raw_path = tmp_path / "seg.nii.gz", tmp_path / "raw.nii.gz"
    sitk.WriteImage(image, str(seg_path))
    sitk.WriteImage(raw, str(raw_path))
    (tmp_path / "dataset.json").write_text(
        json.dumps({"labels": labels}), encoding="utf-8",
    )
    requested = [
        (PlaneTypeEnum.冠状位, SliceType.MPR),
        (PlaneTypeEnum.矢状位, SliceType.MPR),
        (PlaneTypeEnum.横状位, SliceType.MPR),
        (PlaneTypeEnum.矢状位, SliceType.CPR),
        (PlaneTypeEnum.冠状位, SliceType.CPR),
    ]
    request = Request(
        dicom_dir=str(tmp_path / "unused_dicom"),
        output_dir=str(tmp_path / "output"),
        slice_part=part,
        plane_configs=[
            PlaneConfig(
                plane=plane, slice_type=slice_type,
                slice_thickness=1.0, slice_spacing=None, slice_image_count=3,
                window_presets=["骨窗"], fill_percent=fill_percent,
            )
            for plane, slice_type in requested
        ],
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {
        "nifti_path": str(raw_path),
        "source_dicom_metadata": {"0020|000d": "1.2.3.4"},
    })
    context.set_result("segmentation", {
        "seg_path": str(seg_path), "model": {"dataset_id": 1},
    })
    monkeypatch.setattr("src.stages.localization_execution.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_planning.is_debug_file_enabled", lambda: False)
    stage_config = {
        "slice_part_config": load_slice_part_config(), "correction_requested": False,
    }
    for stage in (
        SpatialCorrectionStage(stage_config), LocalizationStage(stage_config),
        AnatomicalAnalysisStage(stage_config), PlanningStage(),
    ):
        stage.execute(context)

    localization = context.get_result("localization")
    anatomy = context.get_result("anatomical_analysis")
    assert len(label_names) == (19 if part is SlicePartEnum.胸腰椎 else 24)
    assert list(localization["label_targets"]) == label_names
    assert excluded_label not in localization["coords"]
    diagnostic = anatomy["anatomical_geometry_diagnostics"]
    assert diagnostic["start_label"] == label_names[0]
    assert diagnostic["end_label"] == sacral_label
    assert diagnostic["used_labels"] == label_names
    annotations = anatomy["spine_geometry"]["filtered_annotation_centroids"]
    assert set(annotations) == set(label_names)
    assert all(annotation["found"] for annotation in annotations.values())
    assert "localization_workspace_v1" not in context.metadata

    plans = context.get_result("planning")["plans"]
    assert len(plans) == 5
    assert {(plan["plane"], plan["slice_type"]) for plan in plans} == {
        (plane.value, slice_type.value) for plane, slice_type in requested
    }
    assert len({plan["plan_id"] for plan in plans}) == 5
    assert len({plan["output_plane"] for plan in plans}) == 5
    for plan in plans:
        assert plan["slice_part"] == part.value
        assert len(plan["slices"]) == 3
        if plan["slice_type"] == "CPR":
            assert plan["geometry_type"] == "curved_stack"
            # CPR 保留当前配置范围内的全部标签，全脊柱从 C2 开始。
            assert plan["source_labels"] == label_names
            assert len(plan["curve_points"]) >= len(label_names)
        else:
            assert plan["geometry_type"] == "plane_stack"
            if plan["plane"] == "冠状位":
                assert plan["source_labels"] == label_names
        if plan["plane"] == "矢状位":
            assert build_spine_localizer_surface(
                raw, plan, localization, anatomy,
            ) is not None

    ReconstructionStage({"parallel": False}).execute(context)
    outputs = context.get_result("reconstruction")["outputs"]
    assert len(outputs) == 5
    for plan, output in zip(plans, outputs):
        output_dir = Path(output["window_outputs"]["骨窗"])
        for index in range(1, 4):
            assert (output_dir / f"{index}.dcm").is_file()
        dataset = pydicom.dcmread(output_dir / "2.dcm")
        assert dataset.pixel_array.size > 0
        if plan["plane"] == "冠状位" and plan["slice_type"] == "MPR":
            # 实际成片必须覆盖首末椎体，不能只检查计划中的标签。
            origin = np.asarray(dataset.ImagePositionPatient, dtype=float)
            right, down = np.asarray(dataset.ImageOrientationPatient, dtype=float).reshape(2, 3)
            for endpoint in (label_names[0], sacral_label):
                bbox = localization["coords"][endpoint]["bbox"]
                for z in (bbox["z_min"], bbox["z_max"]):
                    for x in (bbox["x_min"], bbox["x_max"]):
                        physical = np.asarray(image.TransformContinuousIndexToPhysicalPoint((
                            float(x), float(bbox["y_min"]), float(z),
                        )))
                        column_index = np.dot(physical - origin, right) / float(dataset.PixelSpacing[1])
                        row_index = np.dot(physical - origin, down) / float(dataset.PixelSpacing[0])
                        assert -0.5 - 1e-6 <= column_index <= dataset.Columns - 0.5 + 1e-6
                        assert -0.5 - 1e-6 <= row_index <= dataset.Rows - 0.5 + 1e-6
        if plan["plane"] == "矢状位":
            # LPS: 屏幕向右为后方，屏幕向下为足侧，与原片一致。
            row, column = np.asarray(dataset.ImageOrientationPatient).reshape(2, 3)
            assert row[1] > 0.0
            assert column[2] < 0.0
            pixels = apply_modality_lut(dataset.pixel_array, dataset)
            body_columns = np.nonzero((pixels >= 90.0) & (pixels <= 110.0))[1]
            posterior_columns = np.nonzero(pixels >= 850.0)[1]
            assert body_columns.size > 0
            assert posterior_columns.size > 0
            assert body_columns.mean() < posterior_columns.mean()
