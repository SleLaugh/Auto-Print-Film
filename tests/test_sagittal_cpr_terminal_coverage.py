"""冠状投影后的矢状位 CPR 应覆盖原始首末椎体的真实头足边缘。"""

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.base import select_planner
from src.reconstruction.curved_stack import CurvedStackResampler
from tests import test_stage6_slice_planning as planning_fixtures


_PART_LABELS = {
    SlicePartEnum.颈椎: ("C2", "C3", "C4"),
    SlicePartEnum.胸椎: ("T1", "T2", "T3"),
    SlicePartEnum.腰椎: ("L1", "L2", "L3"),
    SlicePartEnum.胸腰椎: ("C7", "T6", "S1"),
    SlicePartEnum.全脊柱: ("C1", "T6", "S1"),
}
_SUPERIOR_HU = 900.0
_INFERIOR_HU = 1500.0


def _terminal_case(tmp_path, part, ap_slope_sign, oblique):
    """较窄 AP、较长 SI 标签可检出先沿 3D 切线延伸再投影造成的短裁。"""

    labels = _PART_LABELS[part]
    localization = planning_fixtures.Stage6SlicePlanningTest._spine_localization_result(
        labels, part.value,
    )
    segmentation = np.zeros((120, 120, 120), dtype=np.uint8)
    pixels = np.full(segmentation.shape, -1024.0, dtype=np.float32)
    ap_centers = (40, 60, 80) if ap_slope_sign > 0 else (80, 60, 40)
    for label_id, (label, y, z) in enumerate(zip(labels, ap_centers, (80, 60, 40)), 1):
        annotation = localization["coords"][f"vertebrae_{label}"]
        annotation["centroid"] = {"x": 60.0, "y": float(y), "z": float(z)}
        annotation["bbox"] = {
            "x_min": 56, "x_max": 64,
            "y_min": y - 2, "y_max": y + 2,
            "z_min": z - 10, "z_max": z + 10,
        }
        annotation["label_ids"] = [label_id]
        region = np.s_[z - 10:z + 11, y - 2:y + 3, 56:65]
        segmentation[region] = label_id
        pixels[region] = 100.0
    pixels[89:91, ap_centers[0] - 2:ap_centers[0] + 3, 56:65] = _SUPERIOR_HU
    pixels[30:32, ap_centers[-1] - 2:ap_centers[-1] + 3, 56:65] = _INFERIOR_HU
    source, mask = (sitk.GetImageFromArray(array) for array in (pixels, segmentation))
    if oblique:
        yaw, pitch = np.deg2rad((19.0, 13.0))
        rz = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                       (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
        rx = np.array(((1.0, 0.0, 0.0), (0.0, np.cos(pitch), -np.sin(pitch)),
                       (0.0, np.sin(pitch), np.cos(pitch))))
        for image in (source, mask):
            image.SetDirection(tuple((rz @ rx).ravel()))
            image.SetOrigin((-31.0, 27.0, -49.0))
    segmentation_path = tmp_path / "terminal_labels.nii.gz"
    sitk.WriteImage(mask, str(segmentation_path))
    localization["seg_for_reconstruction_path"] = str(segmentation_path)
    localization["image"].update({
        "size_xyz": list(source.GetSize()), "spacing_xyz": list(source.GetSpacing()),
        "origin_xyz": list(source.GetOrigin()), "direction": list(source.GetDirection()),
    })
    localization["correction_scope_bbox"] = {
        "x_min": 56, "x_max": 64,
        "y_min": 38, "y_max": 82,
        "z_min": 30, "z_max": 90,
    }
    return source, localization


@pytest.mark.parametrize("part", tuple(_PART_LABELS))
@pytest.mark.parametrize("ap_slope_sign", (-1, 1))
@pytest.mark.parametrize("oblique", (False, True))
def test_projected_sagittal_cpr_keeps_both_raw_label_terminal_hu_bands(
    tmp_path, part, ap_slope_sign, oblique,
):
    """首末标签各 2 mm 端部标记必须实际采到，不能只检查延伸字段。"""

    source, localization = _terminal_case(tmp_path, part, ap_slope_sign, oblique)
    request = Request(
        "/tmp/dicom", "/tmp/output", slice_part=part,
        plane_configs=[PlaneConfig(
            PlaneTypeEnum.矢状位, 1.0, 5.0, ["骨窗"],
            slice_type=SliceType.CPR, fill_percent=0,
        )],
    )
    plan = select_planner(part.value, SliceType.CPR.value).build_plans(
        request, localization,
    ).plans[0].to_dict()
    layer = min(plan["slices"], key=lambda item: abs(item["offset_mm"]))
    output = CurvedStackResampler().sample_slice(source, plan, layer)
    pixels = sitk.GetArrayFromImage(output)[0]
    superior = np.argwhere(np.abs(pixels - _SUPERIOR_HU) <= 1.0)
    inferior = np.argwhere(np.abs(pixels - _INFERIOR_HU) <= 1.0)
    assert superior.shape[0] >= 3, "冠状投影切线必须覆盖首椎体 z=89..90 的端部标记"
    assert inferior.shape[0] >= 3, "冠状投影切线必须覆盖末椎体 z=30..31 的端部标记"
    assert superior[:, 0].max() < inferior[:, 0].min(), "首末标签顺序应为头侧到足侧"

    direction = np.asarray(source.GetDirection()).reshape(3, 3)
    si = direction[:, 2]
    curve = np.array([[point[axis] for axis in "xyz"] for point in plan["curve_points"]])
    native_si = (curve - source.GetOrigin()) @ si
    assert native_si.max() >= 90.0
    assert native_si.min() <= 30.0
