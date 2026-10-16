"""脊柱回正必须使用物理镜像轴，不能混合横纵 PCA 长轴。"""

import math

import numpy as np
import pytest
import SimpleITK as sitk

from src.anatomy.spine_pose import (
    estimate_spine_axial_symmetry,
    estimate_spine_label_symmetry,
)


def _rotation(yaw=0.0, roll=0.0):
    yaw, roll = np.deg2rad((yaw, roll))
    rz = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                   (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
    ry = np.array(((np.cos(roll), 0.0, np.sin(roll)), (0.0, 1.0, 0.0),
                   (-np.sin(roll), 0.0, np.cos(roll))))
    return rz @ ry


def _symmetric_vertebra(yaw=12.0, roll=7.0, wider=False):
    generator = np.random.default_rng(7)
    half = generator.normal(size=(450, 3)) * (6.0, 8.0, 4.0)
    half[:, 0] = np.abs(half[:, 0])
    # 后方附件固定前后语义；改变椎体宽度可以交换 XY PCA 主轴。
    posterior = generator.normal(size=(150, 3)) * (1.0, 2.0, 2.0) + (0.0, 18.0, 0.0)
    half = np.concatenate((half, posterior))
    reflected = half * (-1.0, 1.0, 1.0)
    points = np.concatenate((half, reflected))
    if wider:
        points[:, 0] *= 3.0
    return points @ _rotation(yaw, roll).T


def _indices(points, image):
    direction = np.array(image.GetDirection()).reshape(3, 3)
    xyz = (points - np.array(image.GetOrigin())) @ direction / np.array(image.GetSpacing())
    return xyz[:, ::-1]


@pytest.mark.parametrize("direction", (np.eye(3), _rotation(27.0, -14.0), np.diag((-1., 1., 1.))))
def test_label_symmetry_recovers_physical_axis_and_image_xy_angle(direction):
    image = sitk.Image((80, 80, 80), sitk.sitkUInt8)
    image.SetSpacing((0.6, 1.1, 2.5))
    image.SetOrigin((-20.0, -30.0, -40.0))
    image.SetDirection(tuple(direction.ravel()))
    # Pose relative to input axes is fixed even for oblique/reflected acquisitions.
    points = _symmetric_vertebra() @ direction.T + (5., -10., 8.)
    result = estimate_spine_label_symmetry(_indices(points, image), image)
    assert result["reliable"], result["symmetry_diagnostics"]
    normal = np.array(list(result["symmetry_left_right_axis"].values()))
    expected = direction @ _rotation(12.0, 7.0)[:, 0]
    np.testing.assert_allclose(normal, expected, atol=0.01)
    assert result["axial_angle_degrees"] == pytest.approx(12.0, abs=0.5)


def test_aggregate_does_not_reverse_when_vertebra_long_axis_switches():
    image = sitk.Image((80, 80, 80), sitk.sitkUInt8)
    labels = {
        "vertebrae_L1": _symmetric_vertebra(yaw=12.0, roll=0.0)[:, ::-1],
        "vertebrae_L5": _symmetric_vertebra(yaw=12.0, roll=0.0, wider=True)[:, ::-1],
    }
    result = estimate_spine_axial_symmetry(labels, image)
    assert result["should_correct"], result
    assert result["angle_degrees"] == pytest.approx(12.0, abs=0.5)
    assert result["used_label_names"] == list(labels)


def test_small_or_flat_label_is_not_a_reliable_pose():
    image = sitk.Image((20, 20, 20), sitk.sitkUInt8)
    small = estimate_spine_label_symmetry(np.ones((10, 3), dtype=int), image)
    assert not small["reliable"]
    flat = np.array([(0, y, x) for y in range(15) for x in range(15)])
    result = estimate_spine_label_symmetry(flat, image)
    assert not result["reliable"]
    aggregate = estimate_spine_axial_symmetry({"missing": flat}, image)
    assert not aggregate["should_correct"]
    assert not aggregate["enough_points"]


def test_aggregate_bounds_each_label_sampling(monkeypatch):
    import src.anatomy.spine_pose as module

    calls = []

    def fit(points, axes, *, max_points):
        calls.append((len(points), max_points))
        angle = math.radians(9.0)
        return np.array((np.cos(angle), np.sin(angle), 0.0)), np.zeros(3), {
            "status": "accepted", "accepted": True,
        }

    monkeypatch.setattr(module, "estimate_reflection_symmetry_plane", fit)
    indices = np.indices((30, 30, 30)).reshape(3, -1).T
    result = estimate_spine_axial_symmetry(
        {"a": indices, "b": indices + 40}, sitk.Image((80, 80, 80), sitk.sitkUInt8),
        max_points_per_label=512,
    )
    assert result["angle_degrees"] == pytest.approx(9.0)
    assert calls == [(512, 512), (512, 512)]


def test_invalid_image_geometry_is_rejected():
    with pytest.raises(ValueError, match="图像几何"):
        estimate_spine_label_symmetry(np.ones((100, 3)), {
            "spacing_xyz": (1., 0., 1.), "origin_xyz": (0., 0., 0.),
            "direction": np.eye(3).ravel(),
        })
