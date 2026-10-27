"""存储轴可反向，但患者头脚、物理位置及实际成片方向不能随之翻转。"""

from itertools import product

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.anatomy.hip import estimate_hip_frame_with_diagnostics
from src.anatomy.lung import estimate_lung_frame_with_diagnostics
from src.anatomy.orientation import patient_reference_axes
from src.anatomy.brain import estimate_brain_frame_with_diagnostics
from src.anatomy.spine import estimate_spine_frame_with_diagnostics
from src.models import PlaneConfig, Request, WindowPreset
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.planning import select_planner
from src.planning.straight import default_straight_frame
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.reconstruction.dicom import write_ct_dicom
from src.reconstruction.plane_stack import PlaneStackResampler


PARTS = (SlicePartEnum.颅脑, SlicePartEnum.髋骨, SlicePartEnum.肺部,
         SlicePartEnum.颈椎, SlicePartEnum.胸椎, SlicePartEnum.腰椎, SlicePartEnum.胸腰椎, SlicePartEnum.全脊柱)
FLIPS = tuple(product((False, True), repeat=3))


def _case(part, flips):
    labels = np.zeros((20, 18, 24), dtype=np.uint8)
    if part in {SlicePartEnum.颅脑, SlicePartEnum.肺部, SlicePartEnum.髋骨}:
        labels[6:9, 6:8, 5:9] = 1
        labels[10:13, 7:9, 15:19] = 2
    else:
        labels[12:14, 7:9, 10:12] = 1
        labels[6:8, 7:9, 10:12] = 2
    image = sitk.Flip(sitk.GetImageFromArray(labels), list(flips))
    array = sitk.GetArrayFromImage(image)
    if part == SlicePartEnum.颅脑:
        geometry, _ = estimate_brain_frame_with_diagnostics(array > 0, image)
    elif part == SlicePartEnum.肺部:
        geometry, _ = estimate_lung_frame_with_diagnostics(array > 0, image)
    elif part == SlicePartEnum.髋骨:
        geometry, _ = estimate_hip_frame_with_diagnostics(array == 1, array == 2, image)
    else:
        geometry, _ = estimate_spine_frame_with_diagnostics(
            {"start": array == 1, "end": array == 2}, "start", "end", image,
        )
    indices = np.argwhere(array > 0)[:, ::-1]
    lower, upper = indices.min(axis=0), indices.max(axis=0)
    localization = {
        "anatomical_geometry": geometry.to_dict(),
        "correction_scope_bbox": {
            f"{axis}_{side}": int(values[index])
            for side, values in (("min", lower), ("max", upper))
            for index, axis in enumerate("xyz")
        },
        "image": {"size_xyz": list(image.GetSize()), "spacing_xyz": list(image.GetSpacing()),
                  "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection())},
    }
    return image, geometry.global_frame, localization


@pytest.mark.parametrize("flips", FLIPS)
def test_patient_reference_ignores_storage_axis_signs_and_keeps_obliquity(flips):
    yaw, tilt = np.deg2rad((23.0, 17.0))
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                   [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., np.cos(tilt), -np.sin(tilt)],
                   [0., np.sin(tilt), np.cos(tilt)]])
    reference = rz @ rx
    stored = reference @ np.diag([-1. if flip else 1. for flip in flips])
    np.testing.assert_allclose(patient_reference_axes(stored), reference, atol=1e-12)
    sagittal_frame = default_straight_frame(
        PlaneTypeEnum.矢状位, {"direction": stored.ravel().tolist()},
    )
    for actual, expected in (
        (sagittal_frame.row_dir, reference[:, 1]),
        (sagittal_frame.col_dir, -reference[:, 2]),
        (sagittal_frame.normal, -reference[:, 0]),
    ):
        np.testing.assert_allclose(list(actual.to_dict().values()), expected, atol=1e-12)


@pytest.mark.parametrize("part", PARTS)
@pytest.mark.parametrize("flips", FLIPS)
def test_all_parts_keep_head_up_and_same_physical_bounds_when_storage_flips(part, flips):
    _, baseline, _ = _case(part, (False, False, False))
    _, frame, localization = _case(part, flips)
    np.testing.assert_allclose(list(frame.origin.to_dict().values()), list(baseline.origin.to_dict().values()))
    assert frame.bounds == baseline.bounds
    for actual, expected in zip(
        (frame.left_right_axis, frame.anterior_posterior_axis, frame.superior_inferior_axis),
        ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
    ):
        np.testing.assert_allclose(list(actual.to_dict().values()), expected, atol=1e-12)
    request = Request("/tmp/dicom", "/tmp/output", slice_part=part)
    for plane in (PlaneTypeEnum.横状位, PlaneTypeEnum.冠状位, PlaneTypeEnum.矢状位):
        plan = build_whole_range_plan_from_bbox(request, PlaneConfig(plane, 1., 1., ["骨窗"]), localization)
        if plane == PlaneTypeEnum.横状位:
            assert plan.slices[0].center.z > plan.slices[-1].center.z
        else:
            assert plan.slices[0].col_dir.z < 0.0
        if plane == PlaneTypeEnum.矢状位:
            # 存储轴符号不能改变矢状位的左右：前方在左，后方在右。
            patient_ap = np.asarray(list(frame.anterior_posterior_axis.to_dict().values()))
            for planned_slice in plan.slices:
                display_right = np.asarray(list(planned_slice.row_dir.to_dict().values()))
                assert np.dot(display_right, patient_ap) == pytest.approx(1.0)


@pytest.mark.parametrize("part", PARTS)
@pytest.mark.parametrize("flips", FLIPS)
@pytest.mark.parametrize("use_anatomical_geometry", (True, False))
def test_written_coronal_sagittal_pixels_and_iop_keep_patient_display_direction(
    tmp_path, part, flips, use_anatomical_geometry,
):
    _, _, localization = _case(part, flips)
    if not use_anatomical_geometry:
        localization.pop("anatomical_geometry")
    z, y, x = np.indices((20, 18, 24), dtype=np.float32)
    # 三个方向采用不同梯度；翻转存储数组时保留同一患者物理强度场。
    raw = sitk.Flip(sitk.GetImageFromArray(10 * x + 100 * y + z), list(flips))
    request = Request(
        "/tmp/dicom", str(tmp_path), slice_part=part,
        plane_configs=[
            PlaneConfig(plane, 1., 1., ["骨窗"])
            for plane in (
                (PlaneTypeEnum.冠状位, PlaneTypeEnum.矢状位)
                if use_anatomical_geometry else (PlaneTypeEnum.矢状位,)
            )
        ],
    )
    # 从部位和切法的正式分发入口取计划，包含腰椎的专用 planner。
    result = select_planner(part.value, "MPR").build_plans(request, localization)
    assert len(result.plans) == len(request.plane_configs)
    window = WindowPreset("方向验证", 6000., 2500.)
    for plan in result.plans:
        plan_data, planned_slice = plan.to_dict(), plan.slices[len(plan.slices) // 2].to_dict()
        sampled = PlaneStackResampler().sample_slice(raw, plan_data, planned_slice)
        path = tmp_path / f"{plan.plane}.dcm"
        write_ct_dicom(image=sampled, output_path=path, plan=plan_data, planned_slice=planned_slice,
                       window=window, series_uid="1.2.3.4.5", study_uid="1.2.3.4", total_slices=len(plan.slices))
        dataset = pydicom.dcmread(path)
        row_axis = np.asarray(dataset.ImageOrientationPatient[:3], dtype=float)
        col_axis = np.asarray(dataset.ImageOrientationPatient[3:], dtype=float)
        expected_row_axis = (0., 1., 0.) if plan.plane == "矢状位" else (1., 0., 0.)
        np.testing.assert_allclose(row_axis, expected_row_axis, atol=1e-12)
        np.testing.assert_allclose(col_axis, (0., 0., -1.), atol=1e-12)
        hu = dataset.pixel_array.astype(float) * float(dataset.RescaleSlope) + float(dataset.RescaleIntercept)
        center_row, center_column = np.array(hu.shape) // 2
        probe_pixels = (
            (center_row - 2, center_column), (center_row + 2, center_column),
            (center_row, center_column - 2), (center_row, center_column + 2),
        )
        for row, column in probe_pixels:
            point = np.asarray(dataset.ImagePositionPatient, dtype=float)
            point = point + column * float(dataset.PixelSpacing[1]) * row_axis
            point = point + row * float(dataset.PixelSpacing[0]) * col_axis
            expected_hu = point @ [10., 100., 1.]
            assert hu[row, column] == pytest.approx(expected_hu, abs=0.1)
        assert hu[center_row - 2, center_column] > hu[center_row + 2, center_column]
        # 独立检查真实像素的左右关系，防止仅改 IOP 却仍把图像镜像写盘。
        expected_horizontal_gradient = 100.0 if plan.plane == "矢状位" else 10.0
        assert hu[center_row, center_column + 2] - hu[center_row, center_column - 2] == pytest.approx(
            4.0 * expected_horizontal_gradient, abs=0.1,
        )
