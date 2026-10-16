"""存储轴可反向，但患者头脚、物理位置及实际成片方向不能随之翻转。"""

from itertools import product

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.anatomy.hip import estimate_hip_frame_with_diagnostics
from src.anatomy.lung import estimate_lung_frame_with_diagnostics
from src.anatomy.orientation import patient_reference_axes
from src.anatomy.skull import estimate_skull_frame_with_diagnostics
from src.anatomy.spine import estimate_spine_frame_with_diagnostics
from src.models import PlaneConfig, Request, WindowPreset
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.reconstruction.dicom import write_ct_dicom
from src.reconstruction.plane_stack import PlaneStackResampler


PARTS = (SlicePartEnum.头骨, SlicePartEnum.髋骨, SlicePartEnum.肺部,
         SlicePartEnum.颈椎, SlicePartEnum.胸椎, SlicePartEnum.腰椎)
FLIPS = tuple(product((False, True), repeat=3))


def _case(part, flips):
    labels = np.zeros((20, 18, 24), dtype=np.uint8)
    if part in {SlicePartEnum.头骨, SlicePartEnum.肺部, SlicePartEnum.髋骨}:
        labels[6:9, 6:8, 5:9] = 1
        labels[10:13, 7:9, 15:19] = 2
    else:
        labels[12:14, 7:9, 10:12] = 1
        labels[6:8, 7:9, 10:12] = 2
    image = sitk.Flip(sitk.GetImageFromArray(labels), list(flips))
    array = sitk.GetArrayFromImage(image)
    if part == SlicePartEnum.头骨:
        geometry, _ = estimate_skull_frame_with_diagnostics(array > 0, image)
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


@pytest.mark.parametrize("flips", ((True, False, False), (False, True, False), (False, False, True)))
def test_written_coronal_sagittal_pixels_and_iop_keep_patient_head_up(tmp_path, flips):
    _, _, localization = _case(SlicePartEnum.头骨, flips)
    z, y, x = np.indices((20, 18, 24), dtype=np.float32)
    raw = sitk.Flip(sitk.GetImageFromArray(10 * x + 100 * y + z), list(flips))
    request = Request("/tmp/dicom", str(tmp_path), slice_part=SlicePartEnum.头骨)
    window = WindowPreset("方向验证", 6000., 2500.)
    for plane in (PlaneTypeEnum.冠状位, PlaneTypeEnum.矢状位):
        plan = build_whole_range_plan_from_bbox(request, PlaneConfig(plane, 1., 1., ["骨窗"]), localization)
        plan_data, planned_slice = plan.to_dict(), plan.slices[len(plan.slices) // 2].to_dict()
        sampled = PlaneStackResampler().sample_slice(raw, plan_data, planned_slice)
        path = tmp_path / f"{plane.value}.dcm"
        write_ct_dicom(image=sampled, output_path=path, plan=plan_data, planned_slice=planned_slice,
                       window=window, series_uid="1.2.3.4.5", study_uid="1.2.3.4", total_slices=len(plan.slices))
        dataset = pydicom.dcmread(path)
        row_axis = np.asarray(dataset.ImageOrientationPatient[:3], dtype=float)
        col_axis = np.asarray(dataset.ImageOrientationPatient[3:], dtype=float)
        assert col_axis[2] < 0.0
        hu = dataset.pixel_array.astype(float) * float(dataset.RescaleSlope) + float(dataset.RescaleIntercept)
        center_row, center_column = np.array(hu.shape) // 2
        for row in (center_row - 2, center_row + 2):
            point = np.asarray(dataset.ImagePositionPatient, dtype=float)
            point = point + center_column * float(dataset.PixelSpacing[1]) * row_axis
            point = point + row * float(dataset.PixelSpacing[0]) * col_axis
            expected_hu = point @ [10., 100., 1.]
            assert hu[row, center_column] == pytest.approx(expected_hu, abs=0.1)
        assert hu[center_row - 2, center_column] > hu[center_row + 2, center_column]
