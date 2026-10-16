"""椎间隙采用可靠局部左右轴，并兼容没有姿态标注的旧定位结果。"""

from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.models import PlaneConfig, Request, WindowPreset
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.cervical import (
    CervicalInterspacePlanner,
    resolve_cervical_interspace_segments,
)
from src.planning.lumbar import LumbarInterspacePlanner, resolve_lumbar_interspace_segments
from src.planning.straight import (
    bbox_physical_corners,
    build_straight_slice_plan,
    default_straight_frame,
)
from src.reconstruction.dicom import write_ct_dicom
from src.reconstruction.framing import TargetFramer
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image


def _xyz(values):
    return dict(zip(("x", "y", "z"), map(float, values)))


def _vector(point):
    return np.array((point.x, point.y, point.z))


def _add_symmetry_pose(localization, direction, status="accepted"):
    yaw, roll = np.deg2rad((11.0, 8.0))
    local_axis = np.array((np.cos(yaw) * np.cos(roll),
                           np.sin(yaw) * np.cos(roll), -np.sin(roll)))
    axis = direction @ local_axis
    for annotation in localization["spine_geometry"]["filtered_annotation_centroids"].values():
        annotation["symmetry_left_right_axis"] = _xyz(axis)
        annotation["symmetry_diagnostics"] = {"status": status, "accepted": status == "accepted"}
    return axis


def _source_direction(kind):
    if kind == "rotated":
        # 同时倾斜输入横断面并旋转左右/前后轴，防止实现硬编码患者 X/Y。
        tilt, rotation = np.deg2rad((12.0, 35.0))
        rotate_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt), np.cos(tilt)],
        ])
        rotate_z = np.array([
            [np.cos(rotation), -np.sin(rotation), 0.0],
            [np.sin(rotation), np.cos(rotation), 0.0],
            [0.0, 0.0, 1.0],
        ])
        return rotate_z @ rotate_x
    if kind == "reflected_lr":
        return np.diag((-1.0, 1.0, 1.0))
    if kind == "reversed_si":
        return np.diag((1.0, 1.0, -1.0))
    return np.eye(3)


def _case(slice_part, geometry, orientation):
    direction = _source_direction(orientation)
    # 图像横纵梯度强度不同，既能检出上下翻转，也能检出左右翻转和轴交换。
    z, y, x = np.indices((41, 41, 41), dtype=np.float32)
    image = sitk.GetImageFromArray(10.0 * x + 100.0 * y + z)
    image.SetDirection(tuple(direction.ravel()))
    image.SetOrigin((-11.0, 7.0, -13.0))

    def physical(index):
        return _xyz(image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, index))))

    upper_z, lower_z = (27.0, 13.0) if direction[2, 2] > 0.0 else (13.0, 27.0)
    labels = (
        ("vertebrae_L1", "vertebrae_L2")
        if slice_part == SlicePartEnum.腰椎
        else ("vertebrae_C2", "vertebrae_C3")
    )
    coords, annotations = {}, {}
    for label, center_z in zip(labels, (upper_z, lower_z)):
        coords[label] = {
            "found": True,
            "centroid": {"x": 20.0, "y": 20.0, "z": center_z},
            "bbox": {
                "x_min": 16, "x_max": 24,
                "y_min": 16, "y_max": 24,
                "z_min": center_z - 3, "z_max": center_z + 3,
            },
        }
        annotation = {"found": True, "centroid_physical": physical((20, 20, center_z))}
        if geometry == "prism":
            annotation.update({
                "prism_found": True,
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        physical((20, py, center_z + dz + 0.2 * (py - 20)))
                        for py, dz in ((16, -3), (24, -3), (24, 3), (16, 3))
                    ],
                    "left_right_extent_mm": {"min": -4.0, "max": 4.0},
                },
            })
        elif geometry == "faces":
            annotation["sagittal_body_faces"] = [
                {
                    "axis_index": 0,
                    "side": side,
                    "point_count": 100,
                    "center": physical((20, 20, center_z + offset)),
                    "normal": _xyz(direction @ np.array((0.0, 0.2 * sign, sign))),
                }
                for side, offset, sign in (("min", -3, -1.0), ("max", 3, 1.0))
            ]
        annotations[label] = annotation

    localization = {
        "image": {
            "size_xyz": list(image.GetSize()),
            "spacing_xyz": list(image.GetSpacing()),
            "origin_xyz": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
        },
        "coords": coords,
        "spine_geometry": {"filtered_annotation_centroids": annotations},
        "resolved_cut_configs": {
            "interspace": {"intervals": [{
                "region": "L1_L2" if slice_part == SlicePartEnum.腰椎 else "C2_C3",
                "start_labels": [labels[0]],
                "end_labels": [labels[1]],
            }]},
        },
    }
    request = Request(
        dicom_dir="/tmp/dicom",
        output_dir="/tmp/output",
        slice_part=slice_part,
        plane_configs=[PlaneConfig(
            plane=PlaneTypeEnum.横状位,
            slice_thickness=0.5,
            slice_spacing=None,
            slice_image_count=3,
            slice_type=SliceType.椎间隙,
            window_presets=["骨窗"],
        )],
    )
    return image, localization, request


@pytest.mark.parametrize("slice_part", (SlicePartEnum.腰椎, SlicePartEnum.颈椎))
@pytest.mark.parametrize("geometry", ("prism", "faces", "bbox"))
@pytest.mark.parametrize("orientation", ("identity", "rotated", "reflected_lr", "reversed_si"))
def test_interspace_sampling_keeps_source_display_direction(slice_part, geometry, orientation):
    image, localization, request = _case(slice_part, geometry, orientation)
    planner, resolver = (
        (LumbarInterspacePlanner(), resolve_lumbar_interspace_segments)
        if slice_part == SlicePartEnum.腰椎
        else (CervicalInterspacePlanner(), resolve_cervical_interspace_segments)
    )
    result = planner.build_plans(request, localization)
    assert len(result.plans) == 1, result.warnings
    plan = result.plans[0]
    source_frame = default_straight_frame(PlaneTypeEnum.横状位, localization["image"])
    segment = resolver(localization)[0][0]
    assert (segment.frame is None) == (geometry == "bbox")
    legacy_plan = build_straight_slice_plan(
        request=request,
        slice_type=SliceType.椎间隙.value,
        plane_config=request.plane_configs[0],
        localization_result=localization,
        segment=segment,
        frame=segment.frame or source_frame,
    )

    # 显示朝向修正不应改变椎间切面的倾斜角、覆盖范围或头到脚的切片顺序。
    np.testing.assert_allclose(
        [_vector(item.center) for item in plan.slices],
        [_vector(item.center) for item in legacy_plan.slices],
        atol=1e-6,
    )
    assert plan.extent_mm == legacy_plan.extent_mm
    assert plan.slice_thickness == legacy_plan.slice_thickness
    assert plan.slice_spacing == legacy_plan.slice_spacing
    direction = np.array(image.GetDirection()).reshape(3, 3)
    for planned_slice in plan.slices:
        np.testing.assert_allclose(_vector(planned_slice.row_dir), direction[:, 0], atol=1e-6)
        assert np.dot(_vector(planned_slice.col_dir), direction[:, 1]) > 0.95
        if geometry != "bbox":
            assert abs(np.dot(_vector(planned_slice.normal), direction[:, 2])) < 0.999
            assert abs(np.dot(
                _vector(planned_slice.normal), _vector(segment.frame.normal),
            )) == pytest.approx(1.0)
        sampled = PlaneStackResampler().sample_slice(
            image, plan.to_dict(), planned_slice.to_dict(),
        )
        pixels = sitk.GetArrayFromImage(sampled)[0]
        row, column = np.array(pixels.shape) // 2
        # 中部四个非对称点均在输入有效视野内；前后/左右应与原始数组一致。
        assert pixels[row + 3, column] > pixels[row - 3, column] + 500
        assert pixels[row, column + 3] > pixels[row, column - 3] + 50


@pytest.mark.parametrize("slice_part", (SlicePartEnum.腰椎, SlicePartEnum.颈椎))
@pytest.mark.parametrize("symmetry_correction", (False, True))
def test_oblique_interspace_dicom_geometry_and_pixels_agree(
    tmp_path: Path, slice_part, symmetry_correction,
):
    image, localization, request = _case(slice_part, "prism", "rotated")
    if symmetry_correction:
        _add_symmetry_pose(localization, np.array(image.GetDirection()).reshape(3, 3))
    planner = LumbarInterspacePlanner() if slice_part == SlicePartEnum.腰椎 else CervicalInterspacePlanner()
    plan = planner.build_plans(request, localization).plans[0].to_dict()
    resolver = (
        resolve_lumbar_interspace_segments
        if slice_part == SlicePartEnum.腰椎
        else resolve_cervical_interspace_segments
    )
    segment = resolver(localization)[0][0]
    legacy_plan = build_straight_slice_plan(
        request=request,
        slice_type=SliceType.椎间隙.value,
        plane_config=request.plane_configs[0],
        localization_result=localization,
        segment=segment,
        frame=segment.frame,
    ).to_dict()
    segmentation = np.zeros((41, 41, 41), dtype=np.uint8)
    for label_id, coord in enumerate(localization["coords"].values(), start=1):
        coord["label_ids"] = [label_id]
        bbox = coord["bbox"]
        segmentation[tuple(
            slice(int(bbox[f"{axis}_min"]), int(bbox[f"{axis}_max"]) + 1)
            for axis in "zyx"
        )] = label_id
    segmentation_image = sitk.GetImageFromArray(segmentation)
    segmentation_image.CopyInformation(image)
    plan["fill_percent"] = legacy_plan["fill_percent"] = 80.0
    framer = TargetFramer(segmentation_image, localization["coords"], plan)
    framed_plan, planned_slice = framer.plane_geometry(plan, plan["slices"][1])
    legacy_framed_plan, legacy_slice = framer.plane_geometry(legacy_plan, legacy_plan["slices"][1])
    np.testing.assert_allclose(
        list(planned_slice["center"].values()), list(legacy_slice["center"].values()), atol=1e-6,
    )
    assert framed_plan["extent_mm"] == pytest.approx(legacy_framed_plan["extent_mm"])
    assert framed_plan["output_spacing_mm"] == pytest.approx(legacy_framed_plan["output_spacing_mm"])
    plan = framed_plan
    sampled = PlaneStackResampler().sample_slice(image, plan, planned_slice)
    window = WindowPreset("方向测试", 6000.0, 2500.0)
    output = tmp_path / "interspace.dcm"
    write_ct_dicom(
        image=sampled,
        output_path=output,
        plan=plan,
        planned_slice=planned_slice,
        window=window,
        series_uid="1.2.3.4.5",
        study_uid="1.2.3.4",
        total_slices=3,
    )
    dataset = pydicom.dcmread(output)
    expected_pixels = sitk.GetArrayFromImage(apply_window_to_image(sampled, window))[0]
    np.testing.assert_array_equal(dataset.pixel_array, expected_pixels)
    np.testing.assert_allclose(dataset.ImagePositionPatient, sampled.GetOrigin(), atol=1e-6)
    np.testing.assert_allclose(
        dataset.ImageOrientationPatient,
        [planned_slice[key][axis] for key in ("row_dir", "col_dir") for axis in ("x", "y", "z")],
        atol=1e-6,
    )
    # 用写盘后的 IOP/IPP 反算任一像素位置，必须与重采样器使用的位置相同。
    row, column = np.array(dataset.pixel_array.shape) // 2 + (3, 2)
    physical = (
        np.array(dataset.ImagePositionPatient, dtype=float)
        + np.array(dataset.ImageOrientationPatient[:3], dtype=float) * column * dataset.PixelSpacing[1]
        + np.array(dataset.ImageOrientationPatient[3:], dtype=float) * row * dataset.PixelSpacing[0]
    )
    np.testing.assert_allclose(
        physical,
        sampled.TransformIndexToPhysicalPoint((int(column), int(row), 0)),
        atol=1e-6,
    )


@pytest.mark.parametrize("geometry", ("prism", "faces", "bbox"))
@pytest.mark.parametrize("orientation", ("identity", "rotated", "reflected_lr", "reversed_si"))
def test_interspace_uses_measured_local_symmetry_axis_without_moving_gap(geometry, orientation):
    image, localization, request = _case(SlicePartEnum.腰椎, geometry, orientation)
    old_segments, old_warnings = resolve_lumbar_interspace_segments(localization)
    assert any("缺少可靠左右对称轴" in warning for warning in old_warnings)
    expected_axis = _add_symmetry_pose(localization, np.array(image.GetDirection()).reshape(3, 3))
    segments, warnings = resolve_lumbar_interspace_segments(localization)
    assert len(segments) == len(old_segments) == 1, warnings
    old, new = old_segments[0], segments[0]
    assert new.frame is not None
    np.testing.assert_allclose(_vector(new.frame.row_dir), expected_axis, atol=1e-6)
    assert np.dot(_vector(new.frame.normal), expected_axis) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(
        np.cross(_vector(new.frame.row_dir), _vector(new.frame.col_dir)),
        _vector(new.frame.normal), atol=1e-6,
    )
    np.testing.assert_allclose(
        (_vector(new.target_start) + _vector(new.target_end)) / 2.0,
        (_vector(old.target_start) + _vector(old.target_end)) / 2.0, atol=1e-6,
    )
    assert np.linalg.norm(_vector(new.target_end) - _vector(new.target_start)) == pytest.approx(
        np.linalg.norm(_vector(old.target_end) - _vector(old.target_start)),
    )
    if geometry == "bbox":
        assert any("真实主体质心连线" in warning for warning in warnings)
    plan = LumbarInterspacePlanner().build_plans(request, localization).plans[0]
    assert plan.slice_thickness == pytest.approx(0.5)
    for planned_slice in plan.slices:
        np.testing.assert_allclose(_vector(planned_slice.row_dir), expected_axis, atol=1e-6)
        assert np.dot(_vector(planned_slice.col_dir), _source_direction(orientation)[:, 1]) > 0.9
        corners = np.array([_vector(corner) for corner in bbox_physical_corners(
            new.target_region_bbox, localization["image"],
        )])
        offsets = corners - _vector(planned_slice.center)
        assert np.max(np.abs(offsets @ _vector(planned_slice.row_dir))) <= plan.extent_mm.width / 2 + 1e-6
        assert np.max(np.abs(offsets @ _vector(planned_slice.col_dir))) <= plan.extent_mm.height / 2 + 1e-6


@pytest.mark.parametrize("quality", ("rejected", "missing"))
def test_rejected_or_unverified_symmetry_axis_keeps_legacy_geometry_and_warns(quality):
    image, localization, _ = _case(SlicePartEnum.腰椎, "prism", "identity")
    original = resolve_lumbar_interspace_segments(localization)[0][0]
    _add_symmetry_pose(localization, np.array(image.GetDirection()).reshape(3, 3), status="rejected")
    if quality == "missing":
        for annotation in localization["spine_geometry"]["filtered_annotation_centroids"].values():
            annotation.pop("symmetry_diagnostics")
    segments, warnings = resolve_lumbar_interspace_segments(localization)
    assert any("缺少可靠左右对称轴" in warning for warning in warnings)
    assert segments[0].frame == original.frame
    assert segments[0].target_start == original.target_start
    assert segments[0].target_end == original.target_end


def test_symmetry_pose_cannot_resurrect_overlapping_bbox_interval():
    image, localization, _ = _case(SlicePartEnum.腰椎, "bbox", "identity")
    _add_symmetry_pose(localization, np.array(image.GetDirection()).reshape(3, 3))
    localization["coords"]["vertebrae_L1"]["bbox"].update(z_min=12, z_max=20)
    segments, warnings = resolve_lumbar_interspace_segments(localization)
    assert not segments
    assert any("没有形成有效头脚区间" in warning for warning in warnings)
