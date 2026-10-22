"""S1 标准配置必须在运行时匹配实际模型标签，并贯穿定位、规划和输出。"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.anatomy.spine_labels import resolve_spine_label_name
from src.config import load_slice_part_config, validate_slice_part_config
from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.spine_interspace import configured_interspace_definitions
from src.planning.spine_cpr import spine_target_bbox
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import LocalizationStage
from src.stages.stage_planning import PlanningStage
from src.stages.stage_reconstruction import ReconstructionStage
from src.stages.stage_reporting import ReportingStage
from src.stages.stage_spatial_correction import SpatialCorrectionStage


def _sacral_config():
    config = load_slice_part_config()
    lumbar = config["slice_parts"]["腰椎"]
    lumbar["localization"]["regions"] = {
        "default": {"type": "whole_range", "start_label": "L5", "end_label": "S1"},
        "L5_S1": {"type": "interspace", "start_label": "L5", "end_label": "S1"},
    }
    for rules in lumbar["slice_rules"].values():
        for slice_type, rule in rules.items():
            rule["regions"] = ["L5_S1"] if slice_type == "椎间隙" else ["default"]
    validate_slice_part_config(config)
    return config


@pytest.mark.parametrize("requested", ("S1", "vertebrae_S1", "sacrum"))
def test_s1_alias_resolution_uses_available_targets_and_preserves_missing_policy(requested):
    assert resolve_spine_label_name(requested, {"sacrum"}) == "sacrum"
    assert resolve_spine_label_name(requested, {"sacrum", "vertebrae_S1"}) == "vertebrae_S1"
    assert resolve_spine_label_name(requested, {"vertebrae_L5"}) == requested
    assert resolve_spine_label_name("vertebrae_L4", {"sacrum"}) == "vertebrae_L4"


def test_interspace_alias_uses_real_targets_without_rewriting_configuration():
    localization = {
        "label_targets": {"vertebrae_L5": {}, "sacrum": {}},
        # 旧定位快照中的占位符不能优先于本次实际 targets。
        "coords": {"vertebrae_L5": {}, "vertebrae_S1": {}, "sacrum": {}},
        "resolved_cut_configs": {"interspace": {"intervals": [{
            "region": "L5_S1", "start_labels": ["vertebrae_L5"],
            "end_labels": ["vertebrae_S1"],
        }]}},
    }
    before = deepcopy(localization)
    assert configured_interspace_definitions(localization) == [
        ("L5_S1", "vertebrae_L5", "sacrum"),
    ]
    assert localization == before


@pytest.mark.parametrize("has_first_sacral_label", (False, True))
def test_spine_target_bbox_resolves_s1_against_actual_targets(has_first_sacral_label):
    lumbar_bbox = dict(x_min=10, x_max=20, y_min=10, y_max=20, z_min=20, z_max=30)
    sacral_bbox = dict(x_min=8, x_max=22, y_min=8, y_max=22, z_min=5, z_max=10)
    localization = {
        "label_targets": {"vertebrae_L5": {}, "sacrum": {}},
        "coords": {
            "vertebrae_L5": {"found": True, "bbox": lumbar_bbox},
            "vertebrae_S1": {"found": False, "bbox": None},
            "sacrum": {"found": True, "bbox": sacral_bbox},
        },
    }
    expected_label = "sacrum"
    if has_first_sacral_label:
        expected_label = "vertebrae_S1"
        localization["label_targets"][expected_label] = {}
        localization["coords"][expected_label] = {"found": True, "bbox": sacral_bbox}
        localization["coords"]["sacrum"]["bbox"] = {**sacral_bbox, "z_min": 0}
    before = deepcopy(localization)

    bbox, source_labels = spine_target_bbox(localization, ["vertebrae_L5", "vertebrae_S1"])

    assert source_labels == ["vertebrae_L5", expected_label]
    assert bbox == {**sacral_bbox, "z_max": 30}
    assert localization == before


@pytest.mark.parametrize("has_first_sacral_label", (False, True))
def test_sacral_endpoint_reaches_anatomical_frame_interspace_and_dicom_output(
    tmp_path, monkeypatch, has_first_sacral_label,
):
    config = _sacral_config()
    raw_path, seg_path = tmp_path / "raw.nii.gz", tmp_path / "seg.nii.gz"
    segmentation = np.zeros((64, 48, 48), dtype=np.uint8)
    segmentation[42:50, 18:32, 16:32] = 3
    segmentation[8:22, 16:34, 15:33] = 9
    labels = {"background": 0, "vertebrae_L5": 3, "sacrum": 9}
    if has_first_sacral_label:
        labels["vertebrae_S1"] = 2
        segmentation[26:34, 18:32, 16:32] = 2
    expected_sacral = "vertebrae_S1" if has_first_sacral_label else "sacrum"
    image = sitk.GetImageFromArray(segmentation)
    image.SetSpacing((0.8, 0.8, 1.2))
    sitk.WriteImage(image, str(seg_path))
    raw = sitk.GetImageFromArray(segmentation.astype(np.float32) * 50)
    raw.CopyInformation(image)
    sitk.WriteImage(raw, str(raw_path))
    (tmp_path / "dataset.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    request = Request(
        dicom_dir=str(tmp_path / "unused_dicom"), output_dir=str(tmp_path / "output"),
        slice_part=SlicePartEnum.腰椎,
        plane_configs=[PlaneConfig(
            plane=PlaneTypeEnum.横状位, slice_thickness=3., slice_spacing=3.,
            slice_image_count=2, slice_type=SliceType.椎间隙,
            window_presets=["骨窗"], fill_percent=80.,
        )],
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {
        "nifti_path": str(raw_path), "source_dicom_metadata": {"0020|000d": "1.2.3.4"},
    })
    context.set_result("segmentation", {"seg_path": str(seg_path), "model": {"dataset_id": 1}})
    # 本例覆盖标签契约；极小人工标签不依赖姿态优化或调试图写出。
    monkeypatch.setattr("src.stages.localization_execution.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    stage_config = {"slice_part_config": config, "correction_requested": False}
    for stage in (SpatialCorrectionStage(stage_config), LocalizationStage(stage_config),
                  AnatomicalAnalysisStage(stage_config), PlanningStage()):
        stage.execute(context)
    localization = context.get_result("localization")
    anatomical = context.get_result("anatomical_analysis")
    assert anatomical["anatomical_geometry_diagnostics"]["end_label"] == expected_sacral
    assert anatomical["anatomical_geometry_diagnostics"]["used_labels"] == [
        "vertebrae_L5", expected_sacral,
    ]
    assert localization["resolved_cut_configs"]["whole_range"]["end_labels"] == ["vertebrae_S1"]
    assert anatomical["spine_geometry"]["interspace_body_pairs"]["L5_S1"]["lower_label"] == expected_sacral
    plans = context.get_result("planning")["plans"]
    assert len(plans) == 1
    assert plans[0]["group_name"] == "L5_S1"
    assert plans[0]["source_labels"] == ["vertebrae_L5", expected_sacral]
    # 第七阶段必须能按已解析的 source_labels 找到目标并输出这个间隙。
    ReconstructionStage({"parallel": False}).execute(context)
    outputs = context.get_result("reconstruction")["outputs"]
    assert len(outputs) == 1
    assert outputs[0]["group_name"] == "L5_S1"
    output_dir = outputs[0]["window_outputs"]["骨窗"]
    assert (Path(output_dir) / "1.dcm").is_file()
    assert (Path(output_dir) / "2.dcm").is_file()
    ReportingStage().execute(context)
    reporting = context.get_result("reporting")
    assert reporting["validation"]["valid"]
    report = json.loads(Path(reporting["report_path"]).read_text(encoding="utf-8"))
    assert report["Success"]
    assert report["PlaneList"][0]["WindowPresetList"][0]["RegionList"][0]["Region"] == "L5_S1"
