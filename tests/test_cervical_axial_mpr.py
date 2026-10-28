"""颈椎横状位 MPR 沿配置首尾长轴斜切，完整覆盖原始椎体标签。"""

from copy import deepcopy
from itertools import product

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.anatomy.spine import estimate_spine_frame_with_diagnostics
from src.models import PlaneConfig, Request, WindowPreset
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.reconstruction.dicom import write_ct_dicom
from src.reconstruction.plane_stack import PlaneStackResampler
from src.stages.stage_anatomical_analysis import AnatomicalAnalysisStage
from src.stages.stage_reconstruction import ReconstructionStage


def _vector(point):
    return np.array((point.x, point.y, point.z), dtype=float)


def _axes(frame):
    return np.column_stack(tuple(map(_vector, (
        frame.left_right_axis, frame.anterior_posterior_axis,
        frame.superior_inferior_axis,
    ))))


def _case(orientation="identity", end_label="C7", reliable_lr=True, ramp=False):
    """所有椎体均带前后结构，C1 额外突出点用于检出端点质心被抽样替代。"""

    yaw, pitch = np.deg2rad((19.0, 11.0))
    rz = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                   (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, np.cos(pitch), -np.sin(pitch)),
                   (0.0, np.sin(pitch), np.cos(pitch))))
    reference = np.eye(3) if orientation == "identity" else rz @ rx
    spacing = np.array((0.7, 1.1, 1.6))
    origin = np.array((-25.0, -30.0, -45.0))
    shape = (88, 72, 64)
    pixels = np.zeros(shape, dtype=np.float32)
    gradient = np.array((3.0, 7.0, 11.0))
    if ramp:
        z, y, x = np.indices(shape)
        storage_gradient = reference.T @ gradient
        pixels = np.asarray(
            1200.0 + origin @ gradient
            + x * spacing[0] * storage_gradient[0]
            + y * spacing[1] * storage_gradient[1]
            + z * spacing[2] * storage_gradient[2], dtype=np.float32,
        )
    source = sitk.GetImageFromArray(pixels)
    source.SetSpacing(tuple(spacing))
    source.SetOrigin(tuple(origin))
    source.SetDirection(tuple(reference.ravel()))

    labels = [f"C{index}" for index in range(1, 8)]
    if end_label == "T1":
        labels.append("T1")
    offsets = np.array(tuple(product((-1, 0, 1), repeat=3)))
    physical_points = {}
    for index, label in enumerate(labels):
        center = np.array((24 + index, 20 + 3 * index, 74 - 8 * index))
        structures = np.array(((-8, 0, 0), (8, 0, 0), (0, -7, 0),
                               (0, 9, 0), (0, 10, -1), (0, 10, 1)))
        points = center + np.vstack((offsets, structures))
        if label == "C1":
            points = np.vstack((points, center + (10, 12, 3)))
        physical_points[label] = origin + (points * spacing) @ reference.T

    flips = {"flipped_lr": (True, False, False),
             "reversed_all": (True, True, True)}.get(orientation, (False,) * 3)
    image = sitk.Flip(source, flips)
    indices = {label: np.array([
        image.TransformPhysicalPointToContinuousIndex(tuple(point))[::-1]
        for point in points
    ]) for label, points in physical_points.items()}
    measured_lr = reference @ np.array((1.0, 0.22, 0.10))
    measured_lr /= np.linalg.norm(measured_lr)
    annotations = {label: {
        "symmetry_left_right_axis": dict(zip("xyz", measured_lr * (-1) ** index)),
        "symmetry_diagnostics": {"status": "accepted"},
    } for index, label in enumerate(labels)} if reliable_lr else None
    geometry, diagnostics = AnatomicalAnalysisStage._build_anatomical_geometry(
        slice_part="颈椎", target_mask=np.zeros(shape, dtype=bool), image=image,
        start_label="C1", end_label=end_label, label_indices_zyx=indices,
        label_pose_annotations=annotations,
    )
    all_xyz = np.concatenate(list(indices.values()))[:, ::-1]
    localization = {
        "anatomical_geometry": geometry.to_dict(),
        "correction_scope_bbox": {
            f"{axis}_{side}": float(values[index])
            for side, values in (("min", all_xyz.min(axis=0)),
                                 ("max", all_xyz.max(axis=0)))
            for index, axis in enumerate("xyz")
        },
        "image": {
            "size_xyz": list(image.GetSize()),
            "spacing_xyz": list(image.GetSpacing()),
            "origin_xyz": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
        },
    }
    return image, indices, physical_points, annotations, geometry, diagnostics, localization


@pytest.mark.parametrize("orientation", ("identity", "oblique", "flipped_lr", "reversed_all"))
@pytest.mark.parametrize("end_label", ("C7", "T1"))
@pytest.mark.parametrize("reliable_lr", (False, True))
def test_cervical_axial_normal_uses_full_physical_endpoint_centroids(
    orientation, end_label, reliable_lr,
):
    image, indices, points, annotations, geometry, _, _ = _case(
        orientation, end_label, reliable_lr,
    )
    frame = geometry.local_frames["axial_mpr"]
    expected_normal = points["C1"].mean(axis=0) - points[end_label].mean(axis=0)
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(_vector(frame.superior_inferior_axis), expected_normal, atol=1e-12)
    assert abs(expected_normal[1]) > 0.05  # 通用水平横断面无法满足这个病例。
    np.testing.assert_allclose(_axes(frame).T @ _axes(frame), np.eye(3), atol=1e-12)
    assert np.linalg.det(_axes(frame)) == pytest.approx(1.0)
    assert _vector(frame.left_right_axis)[0] > 0.0
    assert _vector(frame.anterior_posterior_axis)[1] > 0.0

    # 新横断面独立于原有整体坐标；冠矢状位仍须读取旧整体坐标。
    baseline, _ = estimate_spine_frame_with_diagnostics(
        None, "C1", end_label, image, annotations, label_indices_zyx=indices,
    )
    assert baseline.local_frames == {}
    assert geometry.global_frame.to_dict() == baseline.global_frame.to_dict()
    candidate_lr = (np.array(list(annotations["C1"]["symmetry_left_right_axis"].values()))
                    if reliable_lr else np.array(image.GetDirection()).reshape(3, 3)[:, 0])
    # 投影后的 LR 应在证据与长轴张成的平面内，不能额外旋转画面。
    assert abs(np.cross(candidate_lr, expected_normal) @ _vector(frame.left_right_axis)) < 1e-12

    all_points = np.concatenate(list(points.values()))
    for axis_name, axis in zip(
        ("left_right", "anterior_posterior", "superior_inferior"), _axes(frame).T,
    ):
        projected = all_points @ axis
        bounds = getattr(frame.bounds, axis_name)
        assert bounds.minimum == pytest.approx(projected.min())
        assert bounds.maximum == pytest.approx(projected.max())


@pytest.mark.parametrize("end_label", ("C7", "T1"))
@pytest.mark.parametrize("extension_pair", ((0.0, 0.0), (0.0, 5.0)))
def test_axial_stack_is_parallel_and_covers_all_vertebrae_and_terminal_slabs(
    end_label, extension_pair,
):
    _, _, points, _, geometry, _, localization = _case("oblique", end_label)
    start_extension, end_extension = extension_pair
    localization["slice_rules"] = {"横状位": {"MPR": {
        "range_extension": {"start_mm": start_extension, "end_mm": end_extension},
    }}}
    request = Request("/tmp/dicom", "/tmp/output", slice_part=SlicePartEnum.颈椎)
    plan = build_whole_range_plan_from_bbox(
        request, PlaneConfig(PlaneTypeEnum.横状位, 5.0, 5.0, ["骨窗"]), localization,
    )
    normal = _vector(geometry.local_frames["axial_mpr"].superior_inferior_axis)
    centers = np.array([_vector(item.center) for item in plan.slices])
    for item in plan.slices:
        np.testing.assert_allclose(_vector(item.normal), normal, atol=1e-12)
        assert abs(_vector(item.row_dir) @ normal) < 1e-12
        assert abs(_vector(item.col_dir) @ normal) < 1e-12
    np.testing.assert_allclose(np.diff(centers, axis=0),
                               np.tile(-5.0 * normal, (len(centers) - 1, 1)), atol=1e-12)
    projected = np.concatenate(list(points.values())) @ normal
    assert _vector(plan.target_start) @ normal == pytest.approx(projected.max())
    assert _vector(plan.target_end) @ normal == pytest.approx(projected.min())
    assert centers[0] @ normal + 2.5 >= projected.max() + start_extension - 1e-9
    assert centers[-1] @ normal - 2.5 <= projected.min() - end_extension + 1e-9
    # 首末余量只来自按整层数取整；不能多漏/多加一层。
    assert centers[0] @ normal + 2.5 < projected.max() + start_extension + 2.5 + 1e-9
    assert centers[-1] @ normal - 2.5 > projected.min() - end_extension - 2.5 - 1e-9
    row, col = map(_vector, (plan.slices[0].row_dir, plan.slices[0].col_dir))
    for label, label_points in points.items():
        relative = label_points - centers[0]
        assert np.max(np.abs(relative @ row)) <= plan.extent_mm.width / 2.0 + 1e-9, label
        assert np.max(np.abs(relative @ col)) <= plan.extent_mm.height / 2.0 + 1e-9, label


@pytest.mark.parametrize("part", (SlicePartEnum.胸椎, SlicePartEnum.腰椎,
                                  SlicePartEnum.胸腰椎, SlicePartEnum.全脊柱))
def test_stage_only_adds_endpoint_axial_frame_for_cervical_spine(part):
    image, indices, _, annotations, cervical_geometry, _, _ = _case("oblique")
    geometry, _ = AnatomicalAnalysisStage._build_anatomical_geometry(
        slice_part=part.value, target_mask=np.zeros(image.GetSize()[::-1], dtype=bool),
        image=image, start_label="C1", end_label="C7", label_indices_zyx=indices,
        label_pose_annotations=annotations,
    )
    assert geometry.local_frames == {}
    assert geometry.global_frame.to_dict() == cervical_geometry.global_frame.to_dict()


@pytest.mark.parametrize("part,plane,slice_type", (
    (SlicePartEnum.颈椎, PlaneTypeEnum.冠状位, SliceType.MPR),
    (SlicePartEnum.颈椎, PlaneTypeEnum.矢状位, SliceType.MPR),
    (SlicePartEnum.颈椎, PlaneTypeEnum.横状位, SliceType.CPR),
    (SlicePartEnum.颈椎, PlaneTypeEnum.横状位, SliceType.椎间隙),
    (SlicePartEnum.胸椎, PlaneTypeEnum.横状位, SliceType.MPR),
    (SlicePartEnum.腰椎, PlaneTypeEnum.横状位, SliceType.MPR),
    (SlicePartEnum.胸腰椎, PlaneTypeEnum.横状位, SliceType.MPR),
    (SlicePartEnum.全脊柱, PlaneTypeEnum.横状位, SliceType.MPR),
))
def test_unrelated_whole_range_plans_ignore_cervical_axial_frame(part, plane, slice_type):
    *_, localization = _case("oblique")
    request = Request("/tmp/dicom", "/tmp/output", slice_part=part)
    config = PlaneConfig(plane, 5.0, 5.0, ["骨窗"], slice_type=slice_type)
    with_local_frame = build_whole_range_plan_from_bbox(request, config, localization)
    old_localization = deepcopy(localization)
    old_localization["anatomical_geometry"]["local_frames"] = {}
    baseline = build_whole_range_plan_from_bbox(request, config, old_localization)
    assert with_local_frame.to_dict() == baseline.to_dict()


@pytest.mark.parametrize("orientation", ("identity", "oblique", "reversed_all"))
def test_resampled_pixels_and_dicom_geometry_follow_cervical_cross_section(tmp_path, orientation):
    image, _, _, _, geometry, _, localization = _case(orientation, ramp=True)
    request = Request("/tmp/dicom", "/tmp/output", slice_part=SlicePartEnum.颈椎)
    plan = build_whole_range_plan_from_bbox(
        request, PlaneConfig(PlaneTypeEnum.横状位, 2.0, 5.0, ["骨窗"]), localization,
    )
    item = plan.slices[len(plan.slices) // 2]
    serialized_plan = plan.to_dict()
    # 不同的行列间距可以检出方向交换和毫米/体素混用。
    serialized_plan["output_spacing_mm"] = {"x": 0.8, "y": 1.3}
    sampled = PlaneStackResampler().sample_slice(image, serialized_plan, item.to_dict())
    expected_axes = _axes(geometry.local_frames["axial_mpr"])
    np.testing.assert_allclose(np.array(sampled.GetDirection()).reshape(3, 3), expected_axes, atol=1e-12)
    midpoint = ((sampled.GetSize()[0] - 1) / 2.0,
                (sampled.GetSize()[1] - 1) / 2.0, 0.0)
    np.testing.assert_allclose(sampled.TransformContinuousIndexToPhysicalPoint(midpoint),
                               _vector(item.center), atol=1e-10)
    pixels = sitk.GetArrayFromImage(sampled)[0]
    output = tmp_path / "cervical_axial.dcm"
    write_ct_dicom(
        image=sampled, output_path=output, plan=serialized_plan,
        planned_slice=item.to_dict(), window=WindowPreset("物理方向验证", 4000.0, 1000.0),
        series_uid="1.2.3.4.5", study_uid="1.2.3.4", total_slices=len(plan.slices),
    )
    dataset = pydicom.dcmread(output)
    row, col, normal = expected_axes.T
    np.testing.assert_allclose(dataset.ImageOrientationPatient, np.concatenate((row, col)), atol=1e-8)
    np.testing.assert_allclose(dataset.ImagePositionPatient, sampled.GetOrigin(), atol=1e-8)
    np.testing.assert_allclose(dataset.PixelSpacing, (1.3, 0.8), atol=1e-12)
    assert float(dataset.SliceThickness) == pytest.approx(2.0)
    assert float(dataset.SpacingBetweenSlices) == pytest.approx(5.0)
    assert float(dataset.SliceLocation) == pytest.approx(_vector(item.center) @ normal)
    written_pixels = dataset.pixel_array
    middle_x, middle_y = (int(value) for value in midpoint[:2])
    for dx, dy in product((-2, 0, 2), repeat=2):
        x, y = middle_x + dx, middle_y + dy
        point = sampled.TransformIndexToPhysicalPoint((x, y, 0))
        dicom_point = (np.asarray(dataset.ImagePositionPatient, dtype=float)
                       + x * float(dataset.PixelSpacing[1]) * row
                       + y * float(dataset.PixelSpacing[0]) * col)
        np.testing.assert_allclose(dicom_point, point, atol=1e-8)
        expected_hu = 1200.0 + np.array(point) @ np.array((3.0, 7.0, 11.0))
        assert pixels[y, x] == pytest.approx(expected_hu, abs=0.002)
        expected_gray = int(np.rint(np.clip((expected_hu + 1000.0) / 4000.0, 0.0, 1.0) * 65535))
        assert abs(int(written_pixels[y, x]) - expected_gray) <= 1


def test_cervical_axial_localizer_lines_are_sloped_and_lie_in_the_planned_planes():
    image, _, _, _, _, _, localization = _case("oblique")
    request = Request("/tmp/dicom", "/tmp/output", slice_part=SlicePartEnum.颈椎)
    plan = build_whole_range_plan_from_bbox(
        request, PlaneConfig(PlaneTypeEnum.横状位, 5.0, 5.0, ["骨窗"]), localization,
    )
    reference_index = image.TransformPhysicalPointToContinuousIndex(tuple(_vector(plan.localizer_reference)))
    fixed_x = int(round(reference_index[0]))
    background = sitk.RegionOfInterest(image, [1, *image.GetSize()[1:]], [fixed_x, 0, 0])
    for item in (plan.slices[0], plan.slices[len(plan.slices) // 2], plan.slices[-1]):
        segment = ReconstructionStage._slice_localizer_intersection(
            planned_slice=item.to_dict(), background_image=background, fixed_axis=0,
        )
        assert segment is not None
        endpoints = np.asarray(segment)
        np.testing.assert_allclose((endpoints - _vector(item.center)) @ _vector(item.normal),
                                   0.0, atol=1e-8)
        indices = np.array([
            background.TransformPhysicalPointToContinuousIndex(tuple(point))
            for point in endpoints
        ])
        np.testing.assert_allclose(indices[:, 0], 0.0, atol=1e-8)
        assert np.ptp(indices[:, 1]) > 20.0
        assert np.ptp(indices[:, 2]) > 2.0  # 矢状定位图上的白线必须倾斜。
