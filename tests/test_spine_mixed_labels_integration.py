"""混合椎体标签修复必须进入定位、解剖和椎间切割的同一工作分割。"""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.config import load_slice_part_config
from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import LocalizationStage
from src.stages.stage_planning import PlanningStage
from src.stages.stage_spatial_correction import SpatialCorrectionStage


@pytest.mark.parametrize("part, names, expected", [
    (SlicePartEnum.腰椎, ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11"],
     ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]),
    (SlicePartEnum.颈椎, ["T1", "C7", "C6", "C5", "C4", "C3", "C2", "C1"],
     ["C2_C3", "C3_C4", "C4_C5", "C5_C6", "C6_C7"]),
])
def test_mixed_body_labels_reach_all_configured_interspaces(tmp_path, monkeypatch,
                                                          part, names, expected):
    seg = np.zeros((240, 86, 86), dtype=np.uint16)
    raw = np.full(seg.shape, 45.0, dtype=np.float32)
    labels = {"background": 0}
    for index, name in enumerate(names):
        label_id = 10 + index * 3
        labels[f"vertebrae_{name}"] = label_id
        z = 8 + index * 26
        seg[z:z + 20, 18:46, 22:64] = label_id
        raw[z:z + 20, 18:46, 22:64] = 165
        raw[z:z + 2, 18:46, 22:64] = 390
        raw[z + 18:z + 20, 18:46, 22:64] = 390
        seg[z + 2:z + 18, 46:62, 40:44] = label_id
        seg[z:z + 18, 61:74, 34:52] = label_id
    # 同一真实主体被相邻语义标签分开；两端保留独立且纯净的编号锚点。
    for index, count in ((4, 5), (5, 12), (6, 8)) if part == SlicePartEnum.腰椎 else (
        (3, 5), (4, 12), (5, 8),
    ):
        z = 8 + index * 26
        seg[z:z + count, 18:46, 22:64] = labels[f"vertebrae_{names[index - 1]}"]
    original = seg.copy()
    image = sitk.GetImageFromArray(seg)
    image.SetSpacing((0.75, 0.8, 1.2))
    image.SetOrigin((-71, -165, -530))
    raw_image = sitk.GetImageFromArray(raw)
    raw_image.CopyInformation(image)
    seg_path, raw_path = tmp_path / "seg.nii.gz", tmp_path / "raw.nii.gz"
    sitk.WriteImage(image, str(seg_path))
    sitk.WriteImage(raw_image, str(raw_path))
    (tmp_path / "dataset.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    request = Request(
        dicom_dir=str(tmp_path / "unused"), output_dir=str(tmp_path / "output"),
        slice_part=part, plane_configs=[PlaneConfig(
            plane=PlaneTypeEnum.横状位, slice_type=SliceType.椎间隙,
            slice_thickness=5.0, slice_spacing=5.0, slice_image_count=4,
            window_presets=["软组织窗"], fill_percent=90,
        )],
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {"nifti_path": str(raw_path)})
    context.set_result("segmentation", {"seg_path": str(seg_path), "model": {"dataset_id": 1}})
    monkeypatch.setattr("src.stages.localization_execution.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_planning.is_debug_file_enabled", lambda: False)
    config = {"slice_part_config": load_slice_part_config(), "correction_requested": False}
    for stage in (SpatialCorrectionStage(config), LocalizationStage(config),
                  AnatomicalAnalysisStage(config), PlanningStage()):
        stage.execute(context)

    correction = context.get_result("spatial_correction")
    assert correction["spine_label_sequence"]["status"] == "repaired"
    working_path = correction["seg_for_reconstruction_path"]
    assert Path(working_path) != seg_path
    repaired = sitk.GetArrayFromImage(sitk.ReadImage(working_path))
    np.testing.assert_array_equal(repaired != 0, original != 0)
    np.testing.assert_array_equal(repaired[:, 60:, :], original[:, 60:, :])
    np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))), original)
    plans = context.get_result("planning")
    assert [plan["group_name"] for plan in plans["plans"]] == expected
    assert all(len(plan["slices"]) == 4 for plan in plans["plans"])
    assert not any(w.startswith("跳过脊柱区间 ") for w in plans["warnings"])
