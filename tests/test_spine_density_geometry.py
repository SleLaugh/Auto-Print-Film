"""断续外缘不能靠首末 SI 跨度撑大主体四棱柱。"""

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import LabelTarget
from src.stages.stage_localization import LocalizationStage


def _core_with_terminal_shell(axis):
    core = np.zeros((28, 40, 40), dtype=bool)
    core[4:24, 10:30, 10:30] = True
    shell = core.copy()
    for layers in (slice(4, 6), slice(22, 24)):
        if axis == 1:
            shell[layers, 5:10, 10:30] = True
        else:
            shell[layers, 10:30, 5:10] = True
    return core, shell


def _image_spaces():
    yaw, tilt = np.deg2rad((17.0, 11.0))
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
        pytest.param(rotate_z @ rotate_x, (0.45, 0.7, 1.5), id="oblique-anisotropic"),
        pytest.param(np.diag((-1.0, 1.0, 1.0)), (0.8, 0.8, 2.0), id="reflected-lr"),
    )


def _image(mask, direction, spacing):
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetDirection(tuple(direction.ravel()))
    image.SetSpacing(spacing)
    image.SetOrigin((-37.0, 19.0, -115.0))
    return image


def _full_prism(body, image):
    indices = np.argwhere(body)
    geometry = LocalizationStage._sagittal_body_quadrilateral(indices, image)
    assert geometry is not None
    region = LocalizationStage._vertebral_body_prism_region(
        indices, image, sagittal_geometry=geometry,
    )
    assert region is not None
    minimum, local_mask = region
    full_mask = np.zeros_like(body)
    slices = tuple(slice(int(start), int(start + size))
                   for start, size in zip(minimum, local_mask.shape))
    full_mask[slices] = local_mask
    return geometry, full_mask


@pytest.mark.parametrize("label_name", ("vertebrae_C4", "vertebrae_T8", "vertebrae_L3"))
def test_discontinuous_anterior_shell_cannot_define_complete_body(label_name):
    core, shell = _core_with_terminal_shell(axis=1)
    segmentation = shell.astype(np.uint8) * 5
    original = segmentation.copy()

    body = LocalizationStage._vertebral_body_candidate_mask(
        segmentation, LabelTarget(label_name, [5]),
    )

    # 两片前缘与主体处于同一 3D 连通块，首末高度相同，真实占据仅 20%。
    np.testing.assert_array_equal(body, core)
    np.testing.assert_array_equal(segmentation, original)


@pytest.mark.parametrize("axis", (1, 2), ids=("anterior", "lateral"))
@pytest.mark.parametrize("direction,spacing", _image_spaces())
def test_terminal_shell_is_removed_before_physical_quad_and_prism(axis, direction, spacing):
    core, shell = _core_with_terminal_shell(axis)
    image = _image(shell, direction, spacing)
    original = shell.copy()

    trimmed = LocalizationStage._trim_transverse_edges_by_density(shell)
    geometry, prism = _full_prism(trimmed, image)

    np.testing.assert_array_equal(trimmed, core)
    np.testing.assert_array_equal(prism, core)
    np.testing.assert_array_equal(shell, original)
    center, sagittal_axes, quad = geometry
    corners = center + quad @ sagittal_axes.T
    corner_indices = np.array([
        image.TransformPhysicalPointToContinuousIndex(tuple(point))
        for point in corners
    ])
    expected = np.argwhere(core)
    # 物理坐标中的旋转、层厚和左右反射不改变真实 AP/SI 边界。
    np.testing.assert_allclose(
        corner_indices[:, (1, 2)].min(axis=0),
        expected[:, (1, 0)].min(axis=0), atol=0.01, rtol=0,
    )
    np.testing.assert_allclose(
        corner_indices[:, (1, 2)].max(axis=0),
        expected[:, (1, 0)].max(axis=0), atol=0.01, rtol=0,
    )


@pytest.mark.parametrize("direction,spacing", _image_spaces())
def test_one_slice_segmentation_hole_preserves_body_and_does_not_fill_mask(direction, spacing):
    core, _ = _core_with_terminal_shell(axis=1)
    incomplete = core.copy()
    incomplete[13] = False
    original = incomplete.copy()
    image = _image(incomplete, direction, spacing)

    trimmed = LocalizationStage._trim_transverse_edges_by_density(incomplete)
    _, prism = _full_prism(trimmed, image)

    np.testing.assert_array_equal(trimmed, incomplete)
    np.testing.assert_array_equal(incomplete, original)
    assert not np.any(trimmed[13])
    # 规划包络仍应覆盖完整首末边界；主体 mask 本身保持真实漏分。
    np.testing.assert_array_equal(prism, core)


@pytest.mark.parametrize("axis", (1, 2), ids=("anterior", "lateral"))
def test_frequently_interrupted_but_dense_edge_is_not_a_terminal_shell(axis):
    core, _ = _core_with_terminal_shell(axis)
    body = core.copy()
    for start, stop in ((4, 8), (10, 14), (16, 20), (22, 24)):
        if axis == 1:
            body[start:stop, 5:10, 10:30] = True
        else:
            body[start:stop, 10:30, 5:10] = True
    original = body.copy()

    trimmed = LocalizationStage._trim_transverse_edges_by_density(body)

    # 最长链仅 4/20，但总占据 14/20；不能仅凭链短把有充分体量的边缘删除。
    np.testing.assert_array_equal(trimmed, original)
    np.testing.assert_array_equal(body, original)


def test_two_slice_internal_hole_does_not_shrink_complete_body_edges():
    core, _ = _core_with_terminal_shell(axis=1)
    body = core.copy()
    body[12:14] = False
    original = body.copy()

    trimmed = LocalizationStage._trim_transverse_edges_by_density(body)

    np.testing.assert_array_equal(trimmed, original)
    np.testing.assert_array_equal(body, original)


@pytest.mark.parametrize("edge_runs,remove_edge", (
    pytest.param(((0, 8), (22, 24)), False, id="continuous-exactly-one-third"),
    pytest.param(((0, 6), (18, 24)), False, id="occupied-exactly-one-half"),
    pytest.param(((0, 9), (22, 24)), False, id="only-occupancy-is-low"),
    pytest.param(((0, 5), (8, 12), (20, 24)), False, id="only-continuity-is-low"),
    pytest.param(((0, 7), (22, 24)), True, id="both-strictly-below-threshold"),
))
def test_density_requires_both_strict_fraction_thresholds(edge_runs, remove_edge):
    core = np.zeros((32, 40, 40), dtype=bool)
    core[4:28, 10:30, 10:30] = True
    body = core.copy()
    for start, stop in edge_runs:
        body[4 + start:4 + stop, 5:10, 10:30] = True
    original = body.copy()

    trimmed = LocalizationStage._trim_transverse_edges_by_density(body)

    # 等号以及只满足一个条件均不足以认定空壳；两个条件必须同时严格成立。
    np.testing.assert_array_equal(trimmed, core if remove_edge else original)
    np.testing.assert_array_equal(body, original)
    assert np.all(trimmed <= body)


def test_all_hollow_columns_without_credible_core_preserve_upstream_mask():
    body = np.zeros((28, 40, 40), dtype=bool)
    body[4:6, 5:30, 5:30] = True
    body[22:24, 5:30, 5:30] = True
    original = body.copy()

    trimmed = LocalizationStage._trim_transverse_edges_by_density(body)

    # 两个方向都找不到足够高度的可信核心，不能裁到最后一行/列充当主体。
    np.testing.assert_array_equal(trimmed, original)
    np.testing.assert_array_equal(body, original)


def test_independent_axis_bounds_cannot_remove_every_joint_solid_core():
    body = np.zeros((20, 12, 12), dtype=bool)
    body[:2] = True
    body[-2:] = True
    body[:, :3, 6:12] = True
    body[:, 6:12, :3] = True
    original = body.copy()

    trimmed = LocalizationStage._trim_transverse_edges_by_density(body)

    # AP 与 LR 各有可信边，但它们属于两个错位实心块；独立求界后取交集
    # 会把两个实心核都删除，只剩 36 体素空壳角，不能接受这组联合边界。
    np.testing.assert_array_equal(trimmed, original)
    np.testing.assert_array_equal(body, original)


def test_density_cannot_lower_reference_and_reintroduce_previously_trimmed_edge():
    core = np.zeros((28, 70, 40), dtype=bool)
    core[4:24, 10:20, 10:30] = True
    body = core.copy()
    body[10:14, 5:10, 10:30] = True
    body[4:6, 20:60, 10:30] = True
    body[22:24, 20:60, 10:30] = True
    original = body.copy()
    # 大多数 AP 行是高跨度空壳。若把它们先替换成低链长再算全局门槛，
    # 会降低原有参考高度，导致原应删除的前缘低高度区域重新进入主体。
    upstream = LocalizationStage._trim_lr_edge_columns_by_si_height(body)
    upstream_snapshot = upstream.copy()
    assert np.any(body & ~upstream)
    assert not np.any(upstream[:, :10])

    trimmed = LocalizationStage._trim_transverse_edges_by_density(upstream)

    np.testing.assert_array_equal(trimmed, core)
    assert np.all(trimmed <= upstream)
    assert not np.any(trimmed[:, :10])
    np.testing.assert_array_equal(upstream, upstream_snapshot)
    np.testing.assert_array_equal(body, original)


@pytest.mark.parametrize("direction,spacing", _image_spaces())
def test_slanted_endplates_with_short_holes_keep_all_real_terminal_voxels(direction, spacing):
    body = np.zeros((34, 40, 40), dtype=bool)
    for y in range(10, 30):
        lower = 4 + (y - 10) // 4
        body[lower:lower + 20, y, 10:30] = True
        body[lower + 10, y, 10:30] = False
    original = body.copy()
    image = _image(body, direction, spacing)

    trimmed = LocalizationStage._trim_transverse_edges_by_density(body)
    geometry, prism = _full_prism(trimmed, image)

    np.testing.assert_array_equal(trimmed, original)
    np.testing.assert_array_equal(body, original)
    assert np.all(prism[original])
    center, sagittal_axes, quad = geometry
    corners = center + quad @ sagittal_axes.T
    indices = np.array([image.TransformPhysicalPointToContinuousIndex(tuple(point))
                        for point in corners])
    edges = np.roll(indices, -1, axis=0) - indices
    # 上下端面随主体倾斜；不能因缺口退化成轴对齐的水平包络。
    tilted_endplates = (np.abs(edges[:, 1]) > 10) & (np.abs(edges[:, 2]) > 2)
    assert np.count_nonzero(tilted_endplates) == 2
