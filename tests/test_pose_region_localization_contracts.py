"""回正、边界和取景标签各自生效，并复用第三至第五阶段的体素快照。"""

import json
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.config import load_slice_part_config
from src.core.context import ProcessingContext
from src.models import LabelTarget, Request
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import LocalizationStage, _LOCALIZATION_WORKSPACE_KEY
from src.stages.stage_spatial_correction import SpatialCorrectionStage


def _bounds(labels):
    return {
        "upper": {"labels": labels, "edge": "upper", "reduce": "max", "offset_mm": 0.0},
        "lower": {"labels": labels, "edge": "lower", "reduce": "min", "offset_mm": 0.0},
    }


def test_resolved_named_regions_and_pose_use_dataset_s1_label():
    stage = LocalizationStage()
    pose = {"method": "spine", "labels": ["L5", "S1"], "roles": {"start": "L5", "end": "S1"}}
    config = {"regions": {
        "default": {"type": "whole_range", "target_labels": ["L5", "S1"], "bounds": _bounds(["L5", "S1"])},
        "sacral_top": {"type": "whole_range", "target_labels": ["S1"], "bounds": _bounds(["S1"])},
    }}
    labels = {"vertebrae_L5": [2], "sacrum": [1]}

    resolved = stage._resolve_label_aliases(config, labels)
    resolved_pose = stage._resolve_pose_aliases(pose, labels)
    targets = stage._resolve_combined_label_targets({"whole_range": resolved}, labels, resolved_pose)

    assert list(resolved["regions"]) == ["default", "sacral_top"]
    assert resolved["regions"]["default"]["target_labels"] == ["vertebrae_L5", "sacrum"]
    assert resolved["regions"]["sacral_top"]["bounds"]["upper"]["labels"] == ["sacrum"]
    assert resolved_pose["roles"]["end"] == ["sacrum"]
    assert set(targets) == {"vertebrae_L5", "sacrum"}
    assert pose["roles"]["end"] == "S1"
    assert config["regions"]["sacral_top"]["target_labels"] == ["S1"]


def test_spatial_fit_uses_pose_exclusions_independently_from_cpr():
    stage = SpatialCorrectionStage()
    array = np.zeros((40, 24, 24), dtype=np.uint8)
    targets = {}
    for index, name in enumerate(("vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "region_only"), 1):
        array[index * 6:index * 6 + 3, 7:16, 7:16] = index
        targets[name] = LabelTarget(name, [index])
    image = sitk.GetImageFromArray(array)
    regions = stage._build_label_voxel_regions(array, targets)
    pose = {
        "method": "spine", "labels": ["vertebrae_C1", "vertebrae_C2", "vertebrae_C3"],
        "roles": {"start": ["vertebrae_C1"], "end": ["vertebrae_C3"]},
        "fit_excluded_labels": ["vertebrae_C2"],
    }
    assert stage._resolve_spine_cpr_config("颈椎")["excluded_labels"] == ["vertebrae_C1"]
    with patch("src.stages.stage_localization.estimate_spine_axial_symmetry", return_value={
        "angle_degrees": 0.0, "should_correct": False,
    }) as estimate:
        result = stage._build_xy_correction_candidate(
            array, targets, slice_part="颈椎", image=image,
            label_regions=regions, resolved_pose_config=pose,
        )

    assert set(estimate.call_args.args[0]) == {"vertebrae_C1", "vertebrae_C3"}
    assert result["skipped_label_names"] == ["vertebrae_C2"]


@pytest.mark.parametrize("rebuild_workspace", [False, True])
def test_stage_pipeline_keeps_boundary_labels_out_of_pose_mask(tmp_path, monkeypatch, rebuild_workspace):
    config = load_slice_part_config()
    config["label_aliases"]["lung"] = "lung"
    config["label_aliases"]["boundary_only"] = "boundary_only"
    config["slice_parts"]["肺部"]["localization"]["pose"] = {"method": "lung", "labels": ["lung"]}
    region = config["slice_parts"]["肺部"]["localization"]["regions"]["default"]
    region["target_labels"] = ["lung"]
    region["bounds"] = _bounds(["lung"])
    region["bounds"]["upper"] = {
        "labels": ["boundary_only"], "edge": "upper", "reduce": "max", "offset_mm": 3.0,
    }
    array = np.zeros((32, 24, 40), dtype=np.uint8)
    array[5:16, 6:18, 5:15] = 1
    array[5:16, 6:18, 25:35] = 1
    array[25:28, 8:16, 17:23] = 2
    image = sitk.GetImageFromArray(array)
    seg_path, raw_path = tmp_path / "seg.nii.gz", tmp_path / "raw.nii.gz"
    sitk.WriteImage(image, str(seg_path))
    sitk.WriteImage(image, str(raw_path))
    (tmp_path / "dataset.json").write_text(json.dumps({
        "labels": {"background": 0, "lung": 1, "boundary_only": 2},
    }), encoding="utf-8")
    request = Request.from_dict({
        "dicom_dir": str(tmp_path / "dicom"), "output_dir": str(tmp_path),
        "SlicePart": "肺部", "ScanThickness": 1.0,
    })
    context = ProcessingContext(request)
    context.set_result("preprocessing", {"nifti_path": str(raw_path)})
    context.set_result("segmentation", {"seg_path": str(seg_path), "model": {"model_id": 903}})
    stage_config = {"slice_part_config": config}
    correction, localization, anatomy = (
        SpatialCorrectionStage(stage_config), LocalizationStage(stage_config), AnatomicalAnalysisStage(stage_config),
    )
    monkeypatch.setattr("src.stages.localization_execution.is_debug_file_enabled", lambda: False)
    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    correction.execute(context)
    assert set(context.get_result("spatial_correction")["label_targets"]) == {"lung", "boundary_only"}
    np.testing.assert_array_equal(context.metadata[_LOCALIZATION_WORKSPACE_KEY]["target_mask"], array == 1)
    if rebuild_workspace:
        context.metadata.pop(_LOCALIZATION_WORKSPACE_KEY)
    localization.execute(context)
    with patch.object(anatomy, "_build_anatomical_geometry", wraps=anatomy._build_anatomical_geometry) as estimate:
        anatomy.execute(context)

    np.testing.assert_array_equal(estimate.call_args.kwargs["target_mask"], array == 1)
    result = context.get_result("anatomical_analysis")
    assert result["resolved_pose_config"]["labels"] == ["lung"]
    final_region = result["anatomical_regions"]["default"]
    assert final_region["target_labels"] == ["lung"]
    assert final_region["constrained"] is True
    assert final_region["global_frame"]["bounds"]["superior_inferior"]["maximum"] == pytest.approx(30.0)
    assert result["anatomical_geometry"]["global_frame"]["bounds"]["superior_inferior"]["maximum"] == pytest.approx(15.0)
    assert _LOCALIZATION_WORKSPACE_KEY not in context.metadata
