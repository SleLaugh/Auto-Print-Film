"""C2 的宽基底不能因齿突长轴和关节翼而被全部丢弃。"""

from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import LabelTarget, Request, SlicePartEnum
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_localization import _LOCALIZATION_WORKSPACE_KEY
from src.stages.stage_localization import LocalizationStage


def _c2_with_dens_and_articular_wings():
    array = np.zeros((52, 100, 200), dtype=np.uint8)
    array[4:24, 10:40, 70:130] = 25  # 下方主体
    array[24:47, 18:33, 88:112] = 25  # 上方细长齿突
    array[10:26, 20:40, 20:180] = 25  # 两侧宽关节翼
    array[4:30, 60:85, 20:180] = 25  # 后方附件
    array[15, 39:61, 99:101] = 25  # 原标签内的细桥
    return array


def test_failed_c2_selection_recovers_existing_inferior_body_only():
    array = _c2_with_dens_and_articular_wings()
    original = array.copy()
    diagnostics = {}

    body = LocalizationStage._vertebral_body_candidate_mask(
        array, LabelTarget("vertebrae_C2", [25]),
        spacing_zyx=(1.0, 0.5, 0.5), diagnostics=diagnostics,
    )

    assert diagnostics["method"] == "c2_inferior_body_recovery"
    assert body[8, 24, 100]  # 恢复下方主体，不能只留下上方齿突。
    assert not np.any(body[26:])
    assert not np.any(body[:, 60:])
    assert np.all(~body | (array == 25))
    np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("label", ("vertebrae_C3", "vertebrae_L1"))
def test_same_failed_shape_does_not_relax_other_vertebrae(label):
    with patch.object(LocalizationStage, "_recover_c2_inferior_body", side_effect=AssertionError):
        body = LocalizationStage._vertebral_body_candidate_mask(
            _c2_with_dens_and_articular_wings(), LabelTarget(label, [25]),
            spacing_zyx=(1.0, 0.5, 0.5),
        )
    assert not np.any(body)


def test_successful_existing_c2_selection_never_enters_recovery():
    array = np.zeros((40, 50, 60), dtype=np.uint8)
    array[4:24, 8:30, 18:42] = 25
    with patch.object(LocalizationStage, "_recover_c2_inferior_body", side_effect=AssertionError):
        body = LocalizationStage._vertebral_body_candidate_mask(
            array, LabelTarget("vertebrae_C2", [25]), spacing_zyx=(1.0, 0.5, 0.5),
        )
    np.testing.assert_array_equal(body, array == 25)


@pytest.mark.parametrize("shape", ("body_only", "dens_only", "detached_superior_fragment"))
def test_missing_connected_body_and_dens_evidence_does_not_recover(shape):
    mask = np.zeros((60, 50, 80), dtype=bool)
    if shape != "dens_only":
        mask[4:24, 8:30, 20:60] = True
    if shape == "dens_only":
        mask[4:46, 14:24, 34:46] = True
    if shape == "detached_superior_fragment":
        mask[50:56, 14:24, 34:46] = True
    original = mask.copy()
    recovered = LocalizationStage._recover_c2_inferior_body(mask, spacing_zyx=(1.0, 0.5, 0.5))
    assert recovered is None
    np.testing.assert_array_equal(mask, original)


@pytest.mark.parametrize("axis,spacing", (
    (0, (0.5, 0.5, 0.5)), (1, (1.0, 0.25, 0.5)), (2, (1.0, 0.5, 0.25)),
))
def test_recovery_follows_physical_support_when_one_axis_is_resampled(axis, spacing):
    mask = _c2_with_dens_and_articular_wings() == 25
    original = LocalizationStage._recover_c2_inferior_body(mask, spacing_zyx=(1.0, 0.5, 0.5))
    resampled = LocalizationStage._recover_c2_inferior_body(np.repeat(mask, 2, axis=axis), spacing_zyx=spacing)
    assert original is not None
    np.testing.assert_array_equal(resampled, np.repeat(original, 2, axis=axis))


@pytest.mark.parametrize("spacing", ((1.0, 0.5, 0.5), (0.5, 0.5, 0.5)))
def test_recovery_maps_identical_body_back_when_si_index_direction_is_reversed(spacing):
    mask = _c2_with_dens_and_articular_wings() == 25
    if spacing[0] == 0.5:
        mask = np.repeat(mask, 2, axis=0)
    forward_diagnostics = {}
    reversed_diagnostics = {}
    forward = LocalizationStage._recover_c2_inferior_body(
        mask, spacing_zyx=spacing, diagnostics=forward_diagnostics,
    )
    reversed_mask = mask[::-1].copy()
    original = reversed_mask.copy()
    reverse = LocalizationStage._recover_c2_inferior_body(
        reversed_mask, spacing_zyx=spacing, diagnostics=reversed_diagnostics,
    )

    assert forward is not None
    np.testing.assert_array_equal(reverse, forward[::-1])
    np.testing.assert_array_equal(reversed_mask, original)
    assert reversed_diagnostics["superior_body_limit_voxel"] == (
        len(mask) - 1 - forward_diagnostics["superior_body_limit_voxel"]
    )


@pytest.mark.parametrize("cpr_excluded", ([], ["vertebrae_C2"]))
@pytest.mark.parametrize("fit_excluded", ([], ["vertebrae_C2"]))
def test_stage5_keeps_existing_annotations_when_c2_is_recovered(tmp_path, monkeypatch, cpr_excluded, fit_excluded):
    array = np.zeros((120, 100, 200), dtype=np.uint8)
    array[60:112] = _c2_with_dens_and_articular_wings()
    array[32:52, 10:40, 70:130] = 24
    array[4:24, 10:40, 70:130] = 23
    original = array.copy()
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))
    targets = {f"vertebrae_C{i}": LabelTarget(f"vertebrae_C{i}", [27 - i])
               for i in (2, 3, 4)}
    with patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        with patch.object(LocalizationStage, "_recover_c2_inferior_body", return_value=None):
            baseline = LocalizationStage._build_spine_body_geometry("颈椎", image, array, targets)
        actual = LocalizationStage._build_spine_body_geometry("颈椎", image, array, targets)

    assert actual["filtered_annotation_centroids"] == baseline["filtered_annotation_centroids"]
    assert actual["filtered_annotation_centroids"]["vertebrae_C2"]["found"] is False
    recovered = actual["interspace_recovered_annotations"]["vertebrae_C2"]
    assert recovered["found"] and recovered["prism_found"]
    assert actual["interspace_body_pairs"]["C3_C4"] == baseline["interspace_body_pairs"]["C3_C4"]
    assert "C2_C3" in actual["interspace_body_pairs"]
    np.testing.assert_array_equal(array, original)

    # 改 CPR 选用标签不能把仅供椎间隙的恢复 C2 重新纳入整体回正；即便
    # 回正也排除 C2，C2_C3 仍需要同一份主体证据。
    stage = AnatomicalAnalysisStage()
    labels = list(targets)
    pose = {
        "method": "spine", "labels": ["C2", "C3", "C4"],
        "roles": {"start": "C2", "end": "C4"},
        "fit_excluded_labels": [name.removeprefix("vertebrae_") for name in fit_excluded],
    }
    cut = {"regions": {"default": {
        "type": "whole_range", "target_labels": labels,
        "bounds": {
            "upper": {"labels": labels, "edge": "upper", "reduce": "max", "offset_mm": 0.0},
            "lower": {"labels": labels, "edge": "lower", "reduce": "min", "offset_mm": 0.0},
        },
    }}}
    seg_path, raw_path = str(tmp_path / "seg.nii.gz"), str(tmp_path / "raw.nii.gz")
    raw_image = sitk.GetImageFromArray(np.zeros(array.shape, dtype=np.int16))
    raw_image.CopyInformation(image)
    sitk.WriteImage(raw_image, raw_path)
    context = ProcessingContext(Request(
        dicom_dir=str(tmp_path), output_dir=str(tmp_path / "out"), slice_part=SlicePartEnum.颈椎,
    ))
    context.set_result("localization", {
        "pose_config": pose, "resolved_pose_config": stage._resolve_pose_aliases(pose),
        "seg_for_reconstruction_path": seg_path, "raw_for_reconstruction_path": raw_path,
        "resolved_cut_config": cut,
        "resolved_cut_configs": {
            "whole_range": cut,
            "interspace": {"intervals": [{
                "region": "C2_C3", "start_labels": ["vertebrae_C2"], "end_labels": ["vertebrae_C3"],
            }]},
        },
        "spine_cpr_config": {"full_label": True, "excluded_labels": cpr_excluded},
        "localization_result_path": "",
    })
    label_regions = stage._build_label_voxel_regions(array, targets)
    target_mask = stage._build_target_mask(array, targets)
    context.metadata[_LOCALIZATION_WORKSPACE_KEY] = {
        "seg_path": seg_path, "image": image, "seg_array": array,
        "label_targets": targets, "label_regions": label_regions,
        "label_coordinates": stage._extract_label_coordinates(array, targets, label_regions=label_regions),
        "target_mask": target_mask, "correction_scope_bbox": stage._build_target_bbox(target_mask),
    }
    expected_pose, _ = stage._build_anatomical_geometry(
        "颈椎", target_mask, image,
        label_indices_zyx={name: region.indices_zyx for name, region in label_regions.items()},
        start_label="vertebrae_C2", end_label="vertebrae_C4",
        label_pose_annotations=baseline["filtered_annotation_centroids"],
    )
    monkeypatch.setattr("src.stages.stage_anatomical_analysis.is_debug_file_enabled", lambda: False)
    with patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}), \
         patch.object(stage, "_build_anatomical_geometry", wraps=stage._build_anatomical_geometry) as estimate:
        stage.execute(context)
    result = context.get_result("anatomical_analysis")
    assert result["anatomical_geometry"] == expected_pose.to_dict()
    assert result["spine_geometry"]["filtered_annotation_centroids"]["vertebrae_C2"]["found"] is False
    assert "vertebrae_C2" in result["spine_geometry"]["interspace_recovered_annotations"]
    assert result["spine_geometry"]["excluded_from_annotation_centroids"] == []
    pose_annotations = estimate.call_args.kwargs["label_pose_annotations"]
    if fit_excluded:
        assert "vertebrae_C2" not in pose_annotations
    else:
        assert pose_annotations["vertebrae_C2"] == baseline["filtered_annotation_centroids"]["vertebrae_C2"]
