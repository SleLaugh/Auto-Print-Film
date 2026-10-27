"""椎体序列修正与第三阶段工作分割路径的集成约束。"""

import json
import numpy as np
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import LabelTarget, Request
from src.models.enums import SlicePartEnum
from src.stages import stage_spatial_correction as spatial_module
from src.stages.stage_localization import (
    LocalizationStage,
    _LOCALIZATION_WORKSPACE_KEY,
)
from src.stages.stage_spatial_correction import SpatialCorrectionStage
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.spine_label_sequence import repair_lumbar_body_sequence


_NAMES = {
    2: "vertebrae_S1", 3: "vertebrae_L5", 4: "vertebrae_L4",
    5: "vertebrae_L3", 6: "vertebrae_L2", 7: "vertebrae_L1",
    8: "vertebrae_T12", 9: "vertebrae_T11", 10: "vertebrae_T10",
}


def _sequence_fixture(*, shifted=False, true_missing=False):
    """前体等间距，异常 L3 只剩前一椎体的端板薄帽。"""

    seg = np.zeros((255, 86, 86), dtype=np.uint16)
    raw = np.full(seg.shape, 45, dtype=np.float32)
    masks = {}
    sequence = (2, 3, 4, 6, 7, 8, 9, 10) if shifted or true_missing else tuple(_NAMES)
    for index, label in enumerate(sequence):
        z = 8 + index * 26
        if true_missing and label >= 6:
            z += 26
        mask = np.zeros(seg.shape, dtype=bool)
        mask[z:z + 20, 18:46, 22:64] = True
        seg[mask] = label
        raw[mask] = 165
        raw[z:z + 2, 18:46, 22:64] = 390
        raw[z + 18:z + 20, 18:46, 22:64] = 390
        masks[label] = mask
        # 后方附件保持原语义，若全局换整个 label 的编号便会破坏这里。
        seg[z:z + 18, 61:74, 34:52] = label
        raw[z:z + 18, 61:74, 34:52] = 350
    cap = np.zeros(seg.shape, dtype=bool)
    if shifted or true_missing:
        upper_z = int(np.argwhere(masks[4])[:, 0].max())
        # 薄帽与邻体同层部分错分，余下 LR 支撑仍显示同一连续端板。
        cap[upper_z - 1:upper_z + 1, 18:46, 22:39] = True
        seg[cap] = 5
        seg[70:94, 61:74, 34:52] = 5
        raw[70:94, 61:74, 34:52] = 350
    if true_missing:
        # CT 中确有一个完整、独立的未分割椎体，不能把上方编号整体压低。
        raw[86:106, 18:46, 22:64] = 165
        raw[86:88, 18:46, 22:64] = 390
        raw[104:106, 18:46, 22:64] = 390
    image = sitk.GetImageFromArray(seg)
    image.SetSpacing((0.75, 0.8, 1.2))
    image.SetOrigin((-71.0, -165.0, -530.0))
    label_map = {"background": [0], "sacrum": [1]}
    label_map.update({name: [label] for label, name in _NAMES.items()})
    # 腰椎原请求只读到 T12；修正后 T12 必须还能读取额外的原 T11 主体。
    targets = {
        name: LabelTarget(name, [label]) for label, name in _NAMES.items()
        if label <= 8
    }
    return seg, raw, image, label_map, targets, masks, cap


def _repair(seg, raw, image, labels, targets):
    return repair_lumbar_body_sequence(
        seg, raw, image, labels, label_targets=targets,
        body_extractor=LocalizationStage._vertebral_body_candidate_mask,
    )


def test_false_slot_repair_preserves_foreground_source_and_posterior():
    seg, raw, image, labels, targets, masks, cap = _sequence_fixture(shifted=True)
    original = seg.copy()

    repaired, diagnostic = _repair(seg, raw, image, labels, targets)

    assert repaired is not None, diagnostic
    assert diagnostic["status"] == "repaired"
    assert repaired.dtype == seg.dtype
    np.testing.assert_array_equal(seg, original)
    np.testing.assert_array_equal(repaired != 0, original != 0)
    np.testing.assert_array_equal(repaired[:, 58:, :], original[:, 58:, :])
    assert np.all(repaired[cap] == 4), diagnostic
    for source, target in ((6, 5), (7, 6), (8, 7), (9, 8)):
        assert np.all(repaired[masks[source]] == target), (source, diagnostic)
    # 请求最高到 T12；它需要额外读取 T11 前体，范围以外更高层不重编号。
    np.testing.assert_array_equal(repaired[masks[10]], original[masks[10]])


def test_false_slot_repair_does_not_reassign_inferior_anchor_foreground():
    seg, raw, image, labels, targets, _, _ = _sequence_fixture(shifted=True)
    # 骶骨端板或相邻附件的近接不能让最近主体云顺手重分 S1/L5。
    seg[35, 20, 24] = 2
    seg[25, 20, 24] = 3
    original = seg.copy()
    inferior = np.isin(original, (2, 3, 4))

    repaired, diagnostic = _repair(seg, raw, image, labels, targets)

    assert repaired is not None, diagnostic
    np.testing.assert_array_equal(repaired[inferior], original[inferior])


def test_lumbar_boundary_reads_extra_body_outside_selected_targets():
    seg, raw, image, labels, targets, masks, _ = _sequence_fixture(shifted=True)
    assert "vertebrae_T11" not in targets

    repaired, diagnostic = _repair(seg, raw, image, labels, targets)

    assert repaired is not None, diagnostic
    assert np.all(repaired[masks[9]] == 8), diagnostic
    assert np.all(repaired[masks[8]] == 7), diagnostic


def test_actual_unsegmented_body_does_not_compress_sequence():
    seg, raw, image, labels, targets, _, _ = _sequence_fixture(true_missing=True)
    original = seg.copy()

    repaired, diagnostic = _repair(seg, raw, image, labels, targets)

    assert repaired is None, diagnostic
    assert diagnostic["status"] in ("unchanged", "skipped")
    np.testing.assert_array_equal(seg, original)


def test_independent_thin_body_between_two_discs_does_not_reindex():
    seg, raw, image, labels, targets, _, cap = _sequence_fixture(shifted=True)
    # 同样短小的标签若位于两侧间盘之间，就是独立前体证据。不能仅因
    # 高度很薄且两个 CT 谷较近，就把它当作邻体的错分端板。
    seg[cap] = 4
    seg[82:85, 18:46, 22:64] = 5
    raw[82:85, 18:46, 22:64] = 390
    original = seg.copy()

    repaired, diagnostic = _repair(seg, raw, image, labels, targets)

    assert repaired is None, diagnostic
    np.testing.assert_array_equal(seg, original)


def test_normal_sequence_does_not_read_ct_or_change_labels():
    seg, _, image, labels, targets, _, _ = _sequence_fixture()
    original = seg.copy()

    def unexpected_read():
        raise AssertionError("正常序列不应加载原始 CT")

    repaired, diagnostic = _repair(seg, unexpected_read, image, labels, targets)

    assert repaired is None, diagnostic
    assert diagnostic["status"] == "unchanged"
    np.testing.assert_array_equal(seg, original)


@pytest.mark.parametrize("removed", (2, 3, 4))
def test_missing_inferior_anchor_does_not_reindex_upper_bodies(removed):
    seg, raw, image, labels, targets, _, _ = _sequence_fixture(shifted=True)
    seg[seg == removed] = 0
    original = seg.copy()

    repaired, diagnostic = _repair(seg, raw, image, labels, targets)

    assert repaired is None, diagnostic
    assert diagnostic["status"] in ("unchanged", "skipped")
    np.testing.assert_array_equal(seg, original)


def _small_image():
    array = np.zeros((12, 24, 24), dtype=np.uint8)
    array[2:5, 5:10, 6:13] = 2
    array[7:10, 6:12, 7:14] = 7
    array[10:12, 6:12, 7:14] = 8
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.65, 0.8, 1.5))
    image.SetOrigin((-21.0, -54.0, -360.0))
    return image, array


def _helper(stage, image, array, tmp_path, **overrides):
    values = dict(
        image=image, seg_array=array, raw_path=tmp_path / "missing-raw.nii.gz",
        seg_path=tmp_path / "original-seg.nii.gz", output_dir=tmp_path / "output",
        slice_part="腰椎", label_map={name: [label] for label, name in _NAMES.items()},
        label_targets={"vertebrae_L1": LabelTarget("vertebrae_L1", [7])},
    )
    values.update(overrides)
    return stage._repair_spine_body_sequence(**values)


@pytest.mark.parametrize("part", ("颈椎", "胸椎", "肺部", "颅脑"))
def test_non_lumbar_scope_never_calls_sequence_repair(monkeypatch, tmp_path, part):
    image, array = _small_image()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("不相关部位不应调用腰椎序列修正")

    monkeypatch.setattr(spatial_module, "repair_lumbar_body_sequence", unexpected_call)
    _, result, path, diagnostic = _helper(
        SpatialCorrectionStage(), image, array, tmp_path, slice_part=part,
    )

    assert result is array
    assert path == str(tmp_path / "original-seg.nii.gz")
    assert diagnostic is None
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("part", ("腰椎", "胸腰椎", "全脊柱"))
def test_unchanged_helper_does_not_read_ct_or_write_working_seg(monkeypatch, tmp_path, part):
    image, array = _small_image()

    def unchanged(seg, lazy_raw, *args, **kwargs):
        assert callable(lazy_raw)
        return None, {"status": "unchanged", "reason": "complete_sequence"}

    monkeypatch.setattr(spatial_module, "repair_lumbar_body_sequence", unchanged)
    _, result, path, diagnostic = _helper(
        SpatialCorrectionStage(), image, array, tmp_path, slice_part=part,
    )

    assert result is array
    assert path == str(tmp_path / "original-seg.nii.gz")
    assert diagnostic["working_seg_path"] == path
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("part", ("腰椎", "胸腰椎", "全脊柱"))
def test_helper_preserves_geometry_and_original_cache(monkeypatch, tmp_path, part):
    image, array = _small_image()
    source = tmp_path / "original-seg.nii.gz"
    sitk.WriteImage(image, str(source))
    source_bytes = source.read_bytes()
    repaired = array.copy()
    repaired[2, 5, 6] = 7

    monkeypatch.setattr(
        spatial_module, "repair_lumbar_body_sequence",
        lambda *args, **kwargs: (repaired, {"status": "repaired", "changed_voxels": 1}),
    )
    result_image, result, path, diagnostic = _helper(
        SpatialCorrectionStage(), image, array, tmp_path, slice_part=part,
    )

    assert path == str(tmp_path / "output/nifti/seg_spine_sequence_corrected.nii.gz")
    disk_image = sitk.ReadImage(path)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(disk_image), repaired)
    for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        assert getattr(result_image, getter)() == getattr(image, getter)()
        assert getattr(disk_image, getter)() == pytest.approx(getattr(image, getter)())
    np.testing.assert_array_equal(result, repaired)
    np.testing.assert_array_equal(array, sitk.GetArrayFromImage(image))
    assert source.read_bytes() == source_bytes
    assert diagnostic["source_seg_path"] == str(source)
    assert diagnostic["working_seg_path"] == path


@pytest.mark.parametrize("part", ("腰椎", "胸腰椎", "全脊柱"))
def test_helper_rejects_repair_that_changes_foreground(monkeypatch, tmp_path, part):
    image, array = _small_image()
    malformed = array.copy()
    malformed[0, 0, 0] = 7
    monkeypatch.setattr(
        spatial_module, "repair_lumbar_body_sequence",
        lambda *args, **kwargs: (malformed, {"status": "repaired"}),
    )

    with pytest.raises(ValueError, match="前景"):
        _helper(SpatialCorrectionStage(), image, array, tmp_path, slice_part=part)

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("part", ("腰椎", "胸腰椎", "全脊柱"))
def test_ct_loader_checks_alignment_only_when_requested(monkeypatch, tmp_path, part):
    image, array = _small_image()
    raw_image = sitk.GetImageFromArray(array.astype(np.float32))
    raw_image.CopyInformation(image)
    raw_image.SetOrigin((99.0, 22.0, 10.0))
    raw_path = tmp_path / "misaligned-raw.nii.gz"
    sitk.WriteImage(raw_image, str(raw_path))

    def request_ct(seg, lazy_raw, *args, **kwargs):
        lazy_raw()
        raise AssertionError("错位原图不应通过空间检查")

    monkeypatch.setattr(spatial_module, "repair_lumbar_body_sequence", request_ct)
    with pytest.raises(ValueError, match="空间不一致"):
        _helper(SpatialCorrectionStage(), image, array, tmp_path, raw_path=raw_path, slice_part=part)


@pytest.mark.parametrize("xy_applied", (False, True))
def test_stage3_uses_repaired_seg_for_coords_xy_and_workspace(monkeypatch, tmp_path, xy_applied):
    image, array = _small_image()
    raw_path = tmp_path / "raw.nii.gz"
    seg_path = tmp_path / "seg.nii.gz"
    raw_image = sitk.GetImageFromArray(array.astype(np.float32))
    raw_image.CopyInformation(image)
    sitk.WriteImage(raw_image, str(raw_path))
    sitk.WriteImage(image, str(seg_path))
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"labels": {
        "background": 0, **{name: label for label, name in _NAMES.items()},
    }}), encoding="utf-8")
    repaired = array.copy()
    repaired[2, 5, 6] = 7
    monkeypatch.setattr(
        spatial_module, "repair_lumbar_body_sequence",
        lambda *args, **kwargs: (repaired, {"status": "repaired", "changed_voxels": 1}),
    )
    stage = SpatialCorrectionStage()
    monkeypatch.setattr(stage, "_find_dataset_json", lambda *a, **k: dataset_path)
    monkeypatch.setattr(stage, "_resolve_combined_label_targets", lambda *a, **k: {
        "vertebrae_L1": LabelTarget("vertebrae_L1", [7]),
        "vertebrae_S1": LabelTarget("vertebrae_S1", [2]),
        "vertebrae_T12": LabelTarget("vertebrae_T12", [8]),
    })
    monkeypatch.setattr(stage, "_build_xy_correction_candidate", lambda *a, **k: {
        "method": "test", "angle_degrees": 0, "correction_status": "not_needed",
        "should_correct": False,
    })
    working = tmp_path / "output/nifti/seg_spine_sequence_corrected.nii.gz"
    final_seg = tmp_path / "output/nifti/seg_corrected_candidate.nii.gz"
    final_raw = tmp_path / "output/nifti/raw_corrected_candidate.nii.gz"

    def write_xy(**kwargs):
        assert kwargs["seg_path"] == str(working)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(sitk.ReadImage(kwargs["seg_path"])), repaired,
        )
        if xy_applied:
            sitk.WriteImage(sitk.ReadImage(str(working)), str(final_seg))
            sitk.WriteImage(raw_image, str(final_raw))
        return {
            "raw_for_reconstruction_path": str(raw_path),
            "seg_for_reconstruction_path": str(working),
            "corrected_raw_path": str(raw_path), "corrected_seg_path": str(working),
            "candidate_raw_path": str(final_raw) if xy_applied else "",
            "candidate_seg_path": str(final_seg) if xy_applied else "",
            "correction_applied": False,
            "correction_params": {
                "correction_status": "candidate" if xy_applied else "not_needed",
            },
        }

    monkeypatch.setattr(stage, "_write_xy_correction_candidate_images", write_xy)
    context = ProcessingContext(Request(
        dicom_dir=str(tmp_path / "dicom"), output_dir=str(tmp_path / "output"),
        slice_part=SlicePartEnum.腰椎,
    ))
    context.set_result("preprocessing", {"nifti_path": str(raw_path)})
    context.set_result("segmentation", {"seg_path": str(seg_path), "model": {"dataset_id": 292}})

    stage.execute(context)

    payload = context.get_result("spatial_correction")
    expected_path = str(final_seg if xy_applied else working)
    assert payload["seg_input_path"] == str(seg_path)
    assert payload["seg_for_reconstruction_path"] == expected_path
    assert payload["corrected_seg_path"] == expected_path
    assert payload["spine_label_sequence"]["sequence_seg_path"] == str(working)
    assert payload["spine_label_sequence"]["working_seg_path"] == expected_path
    workspace = context.metadata[_LOCALIZATION_WORKSPACE_KEY]
    assert workspace["seg_path"] == expected_path
    np.testing.assert_array_equal(workspace["seg_array"], repaired)
    assert workspace["label_coordinates"]["vertebrae_L1"].voxel_count == int(
        np.count_nonzero(repaired == 7)
    )
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))), array,
    )
    LocalizationStage().execute(context)
    localization = context.get_result("localization")
    assert localization["seg_for_reconstruction_path"] == expected_path
    assert localization["spine_label_sequence"] == payload["spine_label_sequence"]
    assert localization["coords"]["vertebrae_L1"]["voxel_count"] == int(
        np.count_nonzero(repaired == 7)
    )
    AnatomicalAnalysisStage().execute(context)
    geometry = context.get_result("anatomical_analysis")["spine_geometry"]
    assert geometry["label_sequence"] == payload["spine_label_sequence"]
