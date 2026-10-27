"""脊柱整切只按左右对称面回正，端点连线不能额外决定画面俯仰。"""

from itertools import product

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.anatomy.spine import estimate_spine_frame_with_diagnostics
from src.models import PlaneConfig, Request, WindowPreset
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.reconstruction.dicom import write_ct_dicom
from src.reconstruction.plane_stack import PlaneStackResampler


_PART_LABELS = {
    SlicePartEnum.颈椎: ("vertebrae_C1", "vertebrae_C4", "vertebrae_C7"),
    SlicePartEnum.胸椎: ("vertebrae_T1", "vertebrae_T6", "vertebrae_T12"),
    SlicePartEnum.腰椎: ("vertebrae_L1", "vertebrae_L3", "vertebrae_L5"),
    SlicePartEnum.胸腰椎: ("vertebrae_C7", "vertebrae_T6", "vertebrae_S1"),
    SlicePartEnum.全脊柱: ("vertebrae_C1", "vertebrae_T6", "vertebrae_S1"),
}
_PITCH_DEGREES = {
    SlicePartEnum.颈椎: -6.24,
    SlicePartEnum.胸椎: 12.0,
    SlicePartEnum.腰椎: -17.0,
    SlicePartEnum.胸腰椎: 9.0,
    SlicePartEnum.全脊柱: 15.0,
}
_ORIENTATIONS = ("identity", "oblique", "flipped_lr", "reversed_si", "reversed_all")
_HU_GRADIENT = np.array((3.0, 7.0, 11.0))
_HU_OFFSET = 1200.0


def _vector(point):
    return np.array((point.x, point.y, point.z), dtype=float)


def _xyz(vector):
    return dict(zip("xyz", map(float, vector)))


def _source_reference(orientation):
    if orientation == "identity":
        return np.eye(3)
    yaw, pitch = np.deg2rad((17.0, 13.0))
    rz = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                   (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, np.cos(pitch), -np.sin(pitch)),
                   (0.0, np.sin(pitch), np.cos(pitch))))
    return rz @ rx


def _case(part, orientation, *, quality="accepted", physical_ramp=False):
    reference = _source_reference(orientation)
    spacing = np.array((0.7, 1.1, 1.6))
    origin = np.array((-20.0, -30.0, -45.0))
    shape = (65, 61, 57)
    pixels = np.zeros(shape, dtype=np.float32)
    if physical_ramp:
        z, y, x = np.indices(shape, dtype=np.float32)
        grid_gradient = reference.T @ _HU_GRADIENT
        pixels = np.asarray(
            _HU_OFFSET + origin @ _HU_GRADIENT
            + x * spacing[0] * grid_gradient[0]
            + y * spacing[1] * grid_gradient[1]
            + z * spacing[2] * grid_gradient[2],
            dtype=np.float32,
        )
    source = sitk.GetImageFromArray(pixels)
    source.SetSpacing(tuple(spacing))
    source.SetOrigin(tuple(origin))
    source.SetDirection(tuple(reference.ravel()))

    yaw, roll = np.deg2rad((11.0, 8.0))
    lr = reference @ np.array((np.cos(yaw) * np.cos(roll),
                               np.sin(yaw) * np.cos(roll), -np.sin(roll)))
    # 独立给出约定的几何预期，不调用生产代码中的坐标系构造器。
    ap = reference[:, 1] - np.dot(reference[:, 1], lr) * lr
    ap /= np.linalg.norm(ap)
    si = np.cross(lr, ap)
    expected_axes = np.column_stack((lr, ap, si))
    center = np.asarray(source.TransformIndexToPhysicalPoint((28, 30, 32)))
    pitch = np.deg2rad(_PITCH_DEGREES[part])
    # 独立 lateral 分量可检出把 LR 再投影到首尾连线法面的错误。
    endpoint_vector = 6.0 * lr + 42.0 * si + 42.0 * np.tan(pitch) * ap
    centers = (center + endpoint_vector / 2.0, center + 2.0 * ap,
               center - endpoint_vector / 2.0)
    offsets = np.array(tuple(product((-1.0, 0.0, 1.0), repeat=3)))
    labels = _PART_LABELS[part]
    physical_points = {}
    for label, label_center in zip(labels, centers):
        index = np.asarray(source.TransformPhysicalPointToContinuousIndex(tuple(label_center)))
        physical_points[label] = origin + ((index + offsets) * spacing) @ reference.T

    flips = {
        "flipped_lr": (True, False, False),
        "reversed_si": (False, False, True),
        "reversed_all": (True, True, True),
    }.get(orientation, (False, False, False))
    image = sitk.Flip(source, flips)
    indices = {
        label: np.array([
            image.TransformPhysicalPointToContinuousIndex(tuple(point))[::-1]
            for point in points
        ])
        for label, points in physical_points.items()
    }
    annotations = None if quality == "absent" else {
        label: {
            # 对称面法向无向，反号不应改变整切姿态。
            "symmetry_left_right_axis": _xyz(lr * sign),
            **({"symmetry_diagnostics": {"status": quality}}
               if quality != "unverified" else {}),
        }
        for label, sign in zip(labels, (-1.0, 1.0, -1.0))
    }
    geometry, diagnostics = estimate_spine_frame_with_diagnostics(
        None, labels[0], labels[-1], image, annotations, label_indices_zyx=indices,
    )
    all_xyz = np.concatenate(list(indices.values()))[:, ::-1]
    minimum, maximum = all_xyz.min(axis=0), all_xyz.max(axis=0)
    localization = {
        "anatomical_geometry": geometry.to_dict(),
        "correction_scope_bbox": {
            f"{axis}_{side}": float(values[index])
            for side, values in (("min", minimum), ("max", maximum))
            for index, axis in enumerate("xyz")
        },
        "image": {
            "size_xyz": list(image.GetSize()), "spacing_xyz": list(image.GetSpacing()),
            "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection()),
        },
    }
    return image, geometry, diagnostics, localization, expected_axes, reference, endpoint_vector


def _axes(frame):
    return np.column_stack(tuple(map(_vector, (
        frame.left_right_axis, frame.anterior_posterior_axis, frame.superior_inferior_axis,
    ))))


@pytest.mark.parametrize("part", tuple(_PART_LABELS))
@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_all_spine_parts_preserve_pitch_and_measured_lr_for_whole_range(part, orientation):
    _, geometry, diagnostics, localization, expected, reference, endpoint = _case(part, orientation)
    frame = geometry.global_frame
    np.testing.assert_allclose(_axes(frame), expected, atol=1e-12)
    np.testing.assert_allclose(_axes(frame).T @ _axes(frame), np.eye(3), atol=1e-12)
    assert np.linalg.det(_axes(frame)) == pytest.approx(1.0)
    assert frame.method == "spine_reflection_symmetry"
    assert frame.applied
    assert diagnostics["reliable_symmetry_label_count"] == 3
    np.testing.assert_allclose(
        np.asarray(diagnostics["start_center"]) - np.asarray(diagnostics["end_center"]),
        endpoint, atol=1e-12,
    )
    assert abs(endpoint @ reference[:, 1]) > 1.0
    request = Request("/tmp/dicom", "/tmp/output", slice_part=part)
    lr, ap, si = expected.T
    directions = {
        PlaneTypeEnum.横状位: (lr, ap, si),
        PlaneTypeEnum.矢状位: (ap, -si, -lr),
        PlaneTypeEnum.冠状位: (lr, -si, ap),
    }
    for plane, expected_directions in directions.items():
        plan = build_whole_range_plan_from_bbox(
            request, PlaneConfig(plane, 0.7, 1.0, ["骨窗"]), localization,
        )
        for planned_slice in (plan.slices[0], plan.slices[-1]):
            np.testing.assert_allclose(
                tuple(map(_vector, (planned_slice.row_dir, planned_slice.col_dir,
                                    planned_slice.normal))),
                expected_directions, atol=1e-12,
            )
        if plane == PlaneTypeEnum.矢状位:
            row, col = map(_vector, (plan.slices[0].row_dir, plan.slices[0].col_dir))
            pitch = np.rad2deg(np.arctan2(endpoint @ row, -endpoint @ col))
            assert pitch == pytest.approx(_PITCH_DEGREES[part], abs=1e-10)
            assert abs(endpoint @ row) > 4.0  # 颈椎 6.24° 也不能被强制摆竖直。


@pytest.mark.parametrize("part", tuple(_PART_LABELS))
@pytest.mark.parametrize("orientation", ("identity", "oblique", "reversed_all"))
@pytest.mark.parametrize("quality", ("absent", "rejected", "unverified"))
def test_missing_reliable_lr_keeps_source_pitch_with_warning(part, orientation, quality):
    _, geometry, diagnostics, _, _, reference, endpoint = _case(
        part, orientation, quality=quality,
    )
    frame = geometry.global_frame
    np.testing.assert_allclose(_axes(frame), reference, atol=1e-12)
    assert frame.method == "spine_dicom_axes"
    assert not frame.applied
    assert diagnostics["reliable_symmetry_label_count"] == 0
    assert frame.warnings
    # 数据有明确前后位移，证据不足时不能转而使用端点自动消除它。
    assert abs(endpoint @ _vector(frame.anterior_posterior_axis)) > 1.0
    assert abs(endpoint @ _vector(frame.superior_inferior_axis)) < np.linalg.norm(endpoint)


@pytest.mark.parametrize("status", ("accepted", "unchanged"))
def test_verified_lr_matching_reference_does_not_report_applied_correction(status):
    image = sitk.Image((12, 12, 18), sitk.sitkUInt8)
    indices = {"C1": np.array(((13.0, 3.0, 5.0), (15.0, 3.0, 5.0))),
               "C7": np.array(((3.0, 7.0, 5.0), (5.0, 7.0, 5.0)))}
    annotations = {label: {
        "symmetry_left_right_axis": _xyz((1.0, 0.0, 0.0)),
        "symmetry_diagnostics": {"status": status},
    } for label in indices}
    geometry, _ = estimate_spine_frame_with_diagnostics(
        None, "C1", "C7", image, annotations, label_indices_zyx=indices,
    )
    np.testing.assert_allclose(_axes(geometry.global_frame), np.eye(3), atol=1e-12)
    assert geometry.global_frame.method == "spine_reflection_symmetry"
    assert not geometry.global_frame.applied


@pytest.mark.parametrize("part", tuple(_PART_LABELS))
@pytest.mark.parametrize("orientation", ("oblique", "reversed_all"))
@pytest.mark.parametrize("plane", tuple(PlaneTypeEnum))
def test_written_spine_dicom_keeps_pitch_pixels_and_millimeter_geometry(
    tmp_path, part, orientation, plane,
):
    image, _, _, localization, expected, _, _ = _case(part, orientation, physical_ramp=True)
    request = Request("/tmp/dicom", str(tmp_path), slice_part=part)
    plan = build_whole_range_plan_from_bbox(
        request, PlaneConfig(plane, 0.7, 1.0, ["骨窗"]), localization,
    ).to_dict()
    planned_slice = plan["slices"][len(plan["slices"]) // 2]
    # 非方形输出像素能检出 row/col 交换和把像素数误当毫米距离。
    plan["output_spacing_mm"] = {"x": 0.8, "y": 1.3}
    sampled = PlaneStackResampler().sample_slice(image, plan, planned_slice)
    window = WindowPreset("物理方向验证", 4000.0, 1000.0)
    output = tmp_path / "spine.dcm"
    write_ct_dicom(
        image=sampled, output_path=output, plan=plan, planned_slice=planned_slice,
        window=window, series_uid="1.2.3.4.5", study_uid="1.2.3.4",
        total_slices=len(plan["slices"]),
    )
    dataset = pydicom.dcmread(output)
    lr, ap, si = expected.T
    row, col = {
        PlaneTypeEnum.横状位: (lr, ap),
        PlaneTypeEnum.矢状位: (ap, -si),
        PlaneTypeEnum.冠状位: (lr, -si),
    }[plane]
    np.testing.assert_allclose(dataset.ImageOrientationPatient, np.concatenate((row, col)), atol=1e-8)
    np.testing.assert_allclose(dataset.ImagePositionPatient, sampled.GetOrigin(), atol=1e-8)
    np.testing.assert_allclose(dataset.PixelSpacing, (1.3, 0.8), atol=1e-12)
    center_row, center_column = np.array(dataset.pixel_array.shape) // 2
    sampled_pixels = sitk.GetArrayFromImage(sampled)[0]
    physical_points = []
    for dr, dc in ((-3, -2), (0, 0), (3, 2)):
        r, c = int(center_row + dr), int(center_column + dc)
        physical = (np.asarray(dataset.ImagePositionPatient, dtype=float)
                    + c * float(dataset.PixelSpacing[1]) * row
                    + r * float(dataset.PixelSpacing[0]) * col)
        physical_points.append(physical)
        np.testing.assert_allclose(
            physical, sampled.TransformIndexToPhysicalPoint((c, r, 0)), atol=1e-8,
        )
        expected_hu = _HU_OFFSET + physical @ _HU_GRADIENT
        assert sampled_pixels[r, c] == pytest.approx(expected_hu, abs=3e-4)
        expected_gray = int(np.rint(np.clip((expected_hu + 1000.0) / 4000.0, 0.0, 1.0) * 65535))
        assert abs(int(dataset.pixel_array[r, c]) - expected_gray) <= 1
    expected_distance = np.hypot(6.0 * 1.3, 4.0 * 0.8)
    assert np.linalg.norm(physical_points[-1] - physical_points[0]) == pytest.approx(
        expected_distance, abs=1e-8,
    )
