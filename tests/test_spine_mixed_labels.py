"""Mixed body labels require independent foreground, anchors, and CT boundaries."""

import numpy as np
import pytest

from src.stages.spine_mixed_labels import repair_mixed_spine_body_labels


def _support(mask, *, spacing_zyx):
    # The caller crops at the anterior foreground row. Preserve complete anterior
    # bodies and remove posterior elements without consulting semantic labels.
    support = mask.copy()
    support[:, 20:, :] = False
    return support


def _case(part="lumbar", mixed=True):
    names = (["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11"]
             if part == "lumbar" else ["C7", "C6", "C5", "C4", "C3", "C2"])
    # Nonconsecutive IDs prevent accidental assumptions about model numbering.
    labels = {"vertebrae_" + name: [value]
              for name, value in zip(names, (3, 17, 41, 11, 29, 53, 7, 61))}
    seg = np.zeros((20 * len(names) + 10, 50, 44), dtype=np.uint8)
    raw = np.full(seg.shape, 30.0, dtype=np.float32)
    centers = [12 + 20 * index for index in range(len(names))]
    for index, (name, center) in enumerate(zip(names, centers)):
        value = labels["vertebrae_" + name][0]
        seg[center-6:center+6, 10:24, 10:32] = value
        raw[center-6:center+6, 10:24, 10:32] = 200.0
        seg[center-4:center+5, 38:43, 18:24] = value
    seg[1:4, 2:4, 2:4] = 99  # Unrelated anatomy shares the segmentation volume.
    affected = [4, 5, 6] if part == "lumbar" else [2, 3]
    if mixed:
        for index in affected:
            center = centers[index]
            inferior_id = labels["vertebrae_" + names[index-1]][0]
            seg[center-6:center+1, 10:24, 10:32] = inferior_id
    targets = {name: {} for name in labels
               if part != "lumbar" or name != "vertebrae_T11"}
    image = {"spacing_xyz": [1.0, 1.0, 1.0], "direction": np.eye(3).ravel().tolist()}
    return seg, raw, image, labels, targets, centers, affected


def _repair(case, *, raw=None, support=_support):
    seg, ct, image, labels, targets, _, _ = case
    return repair_mixed_spine_body_labels(
        seg, ct if raw is None else raw, image, labels,
        label_targets=targets, support_extractor=support,
    )


@pytest.mark.parametrize("part", ("lumbar", "cervical"))
def test_multiple_split_bodies_restore_semantic_cores_and_preserve_other_foreground(part):
    case = _case(part)
    seg, _, _, labels, _, centers, affected = case
    original = seg.copy()
    result, diagnostic = _repair(case)

    assert result is not None, diagnostic
    assert diagnostic["status"] == "repaired"
    np.testing.assert_array_equal(seg, original)
    np.testing.assert_array_equal(result != 0, seg != 0)
    np.testing.assert_array_equal(result == 99, seg == 99)
    np.testing.assert_array_equal(result[:, 35:, :], seg[:, 35:, :])
    names = list(labels)
    for index, center in enumerate(centers):
        body = result[center-6:center+6, 10:24, 10:32]
        assert np.all(body == labels[names[index]][0])
        if index not in affected:
            np.testing.assert_array_equal(result[center-6:center+6], seg[center-6:center+6])
    assert all(boundary["accepted"] for boundary in diagnostic["ct_evidence"])


@pytest.mark.parametrize("part", ("lumbar", "cervical"))
def test_clean_bodies_return_before_loading_raw(part):
    case = _case(part, mixed=False)

    def unexpected_loader():
        pytest.fail("clean component labels must not load raw CT")

    result, diagnostic = _repair(case, raw=unexpected_loader)
    assert result is None
    assert diagnostic["status"] == "unchanged"


@pytest.mark.parametrize("orientation", ("rotated", "reflected_lr", "reversed_si"))
def test_physical_order_handles_oblique_reflected_and_reversed_source_axes(orientation):
    case = list(_case())
    if orientation == "reversed_si":
        case[0] = case[0][::-1].copy()
        case[1] = case[1][::-1].copy()
        direction = np.diag((1.0, 1.0, -1.0))
    elif orientation == "reflected_lr":
        direction = np.diag((-1.0, 1.0, 1.0))
    else:
        angle = np.deg2rad(23.0)
        direction = np.array(((1.0, 0.0, 0.0),
                              (0.0, np.cos(angle), -np.sin(angle)),
                              (0.0, np.sin(angle), np.cos(angle))))
    case[2] = {**case[2], "direction": direction.ravel().tolist()}
    result, diagnostic = _repair(case)
    assert result is not None, diagnostic
    if orientation == "reversed_si":
        result = result[::-1]
    reference, reference_diagnostic = _repair(_case())
    np.testing.assert_array_equal(result, reference)


def test_missing_anatomical_body_cannot_compress_numbering():
    case = _case()
    seg, _, _, _, _, centers, _ = case
    seg[centers[2]-6:centers[2]+6, 10:24, 10:32] = 0
    original = seg.copy()
    result, diagnostic = _repair(case)
    assert result is None
    assert "缺失、合并或额外前体" in diagnostic["reason"]
    np.testing.assert_array_equal(seg, original)


def test_merged_anatomical_bodies_are_not_split_by_semantic_priors():
    case = _case()
    seg, _, _, labels, _, centers, _ = case
    seg[centers[4]+5:centers[5]-5, 15:19, 18:23] = labels["vertebrae_L2"][0]
    result, diagnostic = _repair(case)
    assert result is None
    assert "缺失、合并或额外前体" in diagnostic["reason"]


def test_distant_semantic_label_inside_body_rejects_ambiguous_chain():
    case = _case()
    seg, _, _, labels, _, centers, _ = case
    seg[centers[4]-6:centers[4]-3, 10:24, 10:32] = labels["vertebrae_T11"][0]
    result, diagnostic = _repair(case)
    assert result is None
    assert "远隔编号" in diagnostic["reason"]


def test_duplicate_pure_semantic_anchors_reject_sequence():
    case = _case()
    seg, _, _, labels, _, centers, _ = case
    seg[centers[-1]-6:centers[-1]+6, 10:24, 10:32] = labels["vertebrae_L3"][0]
    result, diagnostic = _repair(case)
    assert result is None
    assert "唯一" in diagnostic["reason"]


@pytest.mark.parametrize("raw_kind", ("none", "shape", "loader_error", "flat", "nan"))
def test_unavailable_misaligned_or_unsupported_ct_cannot_repair(raw_kind):
    case = _case()
    raw = case[1]
    if raw_kind == "none":
        source = lambda: None
    elif raw_kind == "shape":
        source = raw[:-1]
    elif raw_kind == "loader_error":
        def source():
            raise ValueError("原图与分割物理坐标不匹配")
    else:
        source = np.full_like(raw, 200.0 if raw_kind == "flat" else np.nan)
    result, diagnostic = _repair(case, raw=source)
    assert result is None
    assert diagnostic["status"] == "skipped"


def test_inside_body_ct_dip_does_not_count_as_another_disc():
    case = _case()
    _, raw, _, _, _, centers, _ = case
    for center in centers:
        raw[center+1:center+3, 10:24, 10:32] = 70.0
    result, diagnostic = _repair(case)
    assert result is not None, diagnostic
    assert all(boundary["accepted"] for boundary in diagnostic["ct_evidence"])


def test_single_mixed_body_lacks_independent_repeated_mixing_evidence():
    case = _case(mixed=False)
    seg, _, _, labels, _, centers, _ = case
    seg[centers[4]-6:centers[4]+1, 10:24, 10:32] = labels["vertebrae_L3"][0]
    result, diagnostic = _repair(case)
    assert result is None
    assert diagnostic["status"] == "unchanged"


def test_small_semantic_speckles_do_not_trigger_reassociation():
    case = _case(mixed=False)
    seg, _, _, labels, _, centers, _ = case
    seg[centers[4], 15, 20] = labels["vertebrae_L3"][0]
    seg[centers[5], 15, 20] = labels["vertebrae_L2"][0]
    result, diagnostic = _repair(case)
    assert result is None
    assert diagnostic["status"] == "unchanged"


def test_shared_numeric_label_ids_cannot_provide_semantic_anchors():
    case = _case()
    case[3]["vertebrae_L1"] = case[3]["vertebrae_L2"]
    result, diagnostic = _repair(case)
    assert result is None
    assert "重复" in diagnostic["reason"]


def test_invalid_extractor_output_is_rejected_without_mutation():
    result, diagnostic = _repair(_case(), support=lambda mask, **kwargs: mask[:-1])
    assert result is None
    assert "形状" in diagnostic["reason"]


def test_ap_reflected_source_requires_normalization_before_anterior_extraction():
    case = _case()
    case[2]["direction"] = np.diag((1.0, -1.0, 1.0)).ravel().tolist()

    def unsupported_extractor(*args, **kwargs):
        pytest.fail("AP-reflected source must be rejected before support extraction")

    result, diagnostic = _repair(case, support=unsupported_extractor)
    assert result is None
    assert "前后方向" in diagnostic["reason"]
