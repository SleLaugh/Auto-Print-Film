"""主体选择、四棱柱端面和椎间方向必须使用同一段真实主体。"""

import copy
from unittest import mock

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import LabelTarget
from src.planning.spine_common import resolve_spine_interspace_segments
from src.stages.stage_localization import LocalizationStage


def _xyz(values):
    return dict(zip("xyz", map(float, values)))


def _bbox(indices):
    return {
        f"{axis}_{side}": int(value)
        for side, coordinates in (("min", indices.min(axis=0)), ("max", indices.max(axis=0)))
        for axis, value in zip("zyx", coordinates)
    }


def _single_vertebra(kind):
    label = np.zeros((40, 70, 90), np.uint8)
    core = np.zeros_like(label, dtype=bool)
    core[10:25, 16:32, 30:54] = True
    label[core] = 1
    if kind == "larger_posterior_component":
        # 真实前方主体不到最大后方连通块的20%，不能在评分前被删除。
        label[8:30, 40:65, 8:80] = 1
    elif kind == "anterior_protrusion":
        # 前突起与主体经窄桥相连；第一段宽行不能直接成为整个椎体主体。
        label[10:25, 6:12, 32:52] = 1
        label[10:25, 12:16, 41:43] = 1
    elif kind == "posterior_wings":
        # 后翼造成的全层LR跨度远大于主体，不能据此提高主体行宽门槛。
        label[12:23, 32:40, 39:45] = 1
        label[12:23, 38:44, 3:9] = 1
        label[12:23, 38:44, 75:81] = 1
        label[12:23, 40:42, 3:81] = 1
    elif kind in {"one_slice_posterior_bridge", "two_slice_posterior_bridge"}:
        # 大部分层上前后AP带分离，仅1/2层经窄桥成为同一个3D连通块。
        # 若先union所有AP候选，后块会沿桥重新混入并拉长主体端面。
        label[6:29, 40:60, 30:54] = 1
        bridge_layers = 1 if kind == "one_slice_posterior_bridge" else 2
        label[17:17 + bridge_layers, 32:40, 37:47] = 1
    return label, core


def _image_spaces():
    yaw, tilt = np.deg2rad((19.0, 12.0))
    rotate_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    rotate_x = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)],
    ])
    return (
        pytest.param(np.eye(3), (1.0, 1.0, 1.0), id="identity"),
        pytest.param(rotate_z @ rotate_x, (0.55, 0.85, 1.7), id="oblique-anisotropic"),
        pytest.param(np.diag((-1.0, 1.0, 1.0)), (0.8, 1.1, 2.0), id="reflected-lr-anisotropic"),
    )


@pytest.mark.parametrize("slice_part,upper,lower", (
    ("颈椎", "C3", "C4"), ("胸椎", "T7", "T8"), ("腰椎", "L2", "L3"),
))
@pytest.mark.parametrize("kind", (
    "clean", "larger_posterior_component", "anterior_protrusion", "posterior_wings",
    "one_slice_posterior_bridge", "two_slice_posterior_bridge",
))
@pytest.mark.parametrize("direction,spacing", _image_spaces())
def test_body_selection_keeps_true_endplates_and_full_label_coverage(
    slice_part, upper, lower, kind, direction, spacing,
):
    single, core = _single_vertebra(kind)
    segmentation = np.zeros((80, 70, 90), np.uint8)
    labels = (f"vertebrae_{upper}", f"vertebrae_{lower}")
    offsets = (40, 0)
    coords, targets = {}, {}
    for label_id, label_name, offset in zip((1, 2), labels, offsets):
        segmentation[offset:offset + 40][single > 0] = label_id
        indices = np.argwhere(segmentation == label_id)
        coords[label_name] = {
            "found": True,
            "label_ids": [label_id],
            "voxel_count": int(len(indices)),
            "centroid": _xyz(indices.mean(axis=0)[::-1]),
            "bbox": _bbox(indices),
        }
        targets[label_name] = LabelTarget(label_name, [label_id])
    original_segmentation = segmentation.copy()
    original_coords = copy.deepcopy(coords)
    image = sitk.GetImageFromArray(segmentation)
    image.SetSpacing(spacing)
    image.SetDirection(tuple(direction.ravel()))
    image.SetOrigin((-37.0, 12.0, -97.0))
    image_info = {
        "size_xyz": list(image.GetSize()), "spacing_xyz": list(spacing),
        "direction": list(image.GetDirection()), "origin_xyz": list(image.GetOrigin()),
    }

    # 完整label对称性是独立算法；此处专门贯穿主体提取、真实最近间隙和端面。
    with mock.patch("src.stages.stage_localization.estimate_spine_label_symmetry", return_value={}):
        geometry = LocalizationStage._build_spine_body_geometry(
            slice_part, image, segmentation, targets,
        )
    localization = {"image": image_info, "coords": coords, "spine_geometry": geometry}
    group = f"{upper}_{lower}"
    segments, warnings = resolve_spine_interspace_segments(
        localization, [(group, labels[0], labels[1])],
    )

    assert len(segments) == 1, warnings
    assert not any(warning.startswith("跳过脊柱区间 ") for warning in warnings)
    source_si = direction[:, 2]
    core_indices = np.argwhere(core)
    for label_name, offset in zip(labels, offsets):
        annotation = geometry["filtered_annotation_centroids"][label_name]
        assert annotation["found"] and annotation["prism_found"]
        assert annotation["voxel_count"] >= np.count_nonzero(core) * 0.95
        if kind in {"one_slice_posterior_bridge", "two_slice_posterior_bridge"}:
            assert annotation["body_bbox"]["y_max"] <= 31
            assert annotation["body_bbox"]["z_min"] >= 10 + offset
            assert annotation["body_bbox"]["z_max"] <= 24 + offset
            assert annotation["voxel_count"] == np.count_nonzero(core)
        expected_center = core_indices.mean(axis=0)[::-1] + (0, 0, offset)
        actual_center = np.array([annotation["centroid_voxel"][axis] for axis in "xyz"])
        np.testing.assert_allclose(actual_center, expected_center, atol=1.0, rtol=0)
        np.testing.assert_allclose(
            list(annotation["centroid_physical"].values()),
            image.TransformContinuousIndexToPhysicalPoint(tuple(actual_center)), atol=1e-6,
        )
        # 上下端面必须落在主体真实首末层附近，不能来自前突起或后翼。
        end_faces = [face for face in annotation["sagittal_body_faces"]
                     if abs(np.array([face["normal"][axis] for axis in "xyz"]) @ source_si) > 0.98]
        assert len(end_faces) == 2
        end_z = sorted(image.TransformPhysicalPointToContinuousIndex(
            tuple(face["center"][axis] for axis in "xyz"),
        )[2] for face in end_faces)
        np.testing.assert_allclose(end_z, [10 + offset, 24 + offset], atol=1.0, rtol=0)

    segment = segments[0]
    normal = np.array([getattr(segment.frame.normal, axis) for axis in "xyz"])
    assert normal @ source_si < -0.98
    start = image.TransformPhysicalPointToContinuousIndex(tuple(
        getattr(segment.target_start, axis) for axis in "xyz"
    ))
    end = image.TransformPhysicalPointToContinuousIndex(tuple(
        getattr(segment.target_end, axis) for axis in "xyz"
    ))
    np.testing.assert_allclose(start[2], 50.0, atol=1.0, rtol=0)
    np.testing.assert_allclose(end[2], 24.0, atol=1.0, rtol=0)
    np.testing.assert_allclose((start[1], end[1]), (23.5, 23.5), atol=1.0, rtol=0)
    # 主体筛选不改全label覆盖：大后块、左右翼和前突起仍属于取景bbox。
    assert segment.target_region_bbox == _bbox(np.argwhere(segmentation > 0))
    assert coords == original_coords
    np.testing.assert_array_equal(segmentation, original_segmentation)
