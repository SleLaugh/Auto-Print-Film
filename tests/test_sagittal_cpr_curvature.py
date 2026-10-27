"""矢状位 CPR 只展开 LR/SI 侧弯，实际 HU 仍保留 AP/SI 生理曲度。"""

import copy

import numpy as np
import pytest
from scipy import ndimage
import SimpleITK as sitk

from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.base import select_planner
from src.reconstruction.curved_stack import CurvedStackResampler
from src.reconstruction.framing import curved_group_bounds, frame_curved_slice
from tests import test_stage6_slice_planning as planning_fixtures


_PART_LABELS = {
    SlicePartEnum.颈椎: ("C2", "C3", "C4", "C5", "C6"),
    SlicePartEnum.胸椎: ("T2", "T4", "T6", "T8", "T10"),
    SlicePartEnum.腰椎: ("L1", "L2", "L3", "L4", "L5"),
    SlicePartEnum.胸腰椎: ("C7", "T5", "T12", "L3", "S1"),
    SlicePartEnum.全脊柱: ("C1", "C7", "T6", "L3", "S1"),
}
_BODY_HU = (200.0, 500.0, 800.0, 1100.0, 1400.0)
_POSTERIOR_HU = 2500.0
_LR_CENTERS = (52, 64, 76, 64, 52)
_AP_CURVATURE_SIGN = {
    SlicePartEnum.颈椎: -1,
    SlicePartEnum.胸椎: 1,
    SlicePartEnum.腰椎: -1,
    SlicePartEnum.胸腰椎: 1,
    SlicePartEnum.全脊柱: 1,
}
_SI_CENTERS = (100, 82, 64, 46, 28)


def _source_direction(orientation):
    if orientation == "identity":
        return np.eye(3)
    yaw, pitch = np.deg2rad((19.0, 11.0))
    rz = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                   (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, np.cos(pitch), -np.sin(pitch)),
                   (0.0, np.sin(pitch), np.cos(pitch))))
    return rz @ rx


def _reflect_storage(localization, image, mask):
    """翻转全部存储轴并更新索引标注，物理中的同一椎体保持不动。"""

    localization = copy.deepcopy(localization)
    lengths = image.GetSize()
    image, mask = (sitk.Flip(item, (True, True, True)) for item in (image, mask))
    boxes = [localization["correction_scope_bbox"]]
    for annotation in localization["coords"].values():
        boxes.append(annotation["bbox"])
        for axis, length in zip("xyz", lengths):
            annotation["centroid"][axis] = length - 1 - annotation["centroid"][axis]
    for box in boxes:
        for axis, length in zip("xyz", lengths):
            low, high = box[f"{axis}_min"], box[f"{axis}_max"]
            box[f"{axis}_min"], box[f"{axis}_max"] = length - 1 - high, length - 1 - low
    localization["image"].update({
        "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection()),
    })
    return localization, image, mask


def _curved_spine(part, orientation, *, lateral_bend=True):
    """同时给出独立 LR 侧弯和 AP 曲度，避免仅检测到标签是否存在。

    中间椎体比首尾向 LR 移动 24 mm、向 AP 移动正或负 16 mm；9 mm 宽
    的主体不会与首尾固定矢状平面相交。后方窄带用于区分前后显示方向。
    """

    labels = _PART_LABELS[part]
    lr_centers = _LR_CENTERS if lateral_bend else (52,) * len(labels)
    ap_centers = tuple(48 + _AP_CURVATURE_SIGN[part] * bend for bend in (0, 9, 16, 10, 0))
    localization = planning_fixtures.Stage6SlicePlanningTest._spine_localization_result(
        labels, part.value,
    )
    hu = np.full((128, 112, 112), -1024.0, dtype=np.float32)
    target = np.zeros_like(hu, dtype=np.uint8)
    for label, value, x, y, z in zip(
        labels, _BODY_HU, lr_centers, ap_centers, _SI_CENTERS,
    ):
        annotation = localization["coords"][f"vertebrae_{label}"]
        annotation["centroid"] = {"x": float(x), "y": float(y), "z": float(z)}
        annotation["bbox"] = {
            "x_min": x - 4, "x_max": x + 4,
            "y_min": y - 4, "y_max": y + 20,
            "z_min": z - 3, "z_max": z + 3,
        }
        target[z - 3:z + 4, y - 4:y + 21, x - 4:x + 5] = 1
        hu[z - 3:z + 4, y - 4:y + 5, x - 4:x + 5] = value
        hu[z - 3:z + 4, y + 5:y + 14, x - 2:x + 3] = 80.0
        hu[z - 3:z + 4, y + 14:y + 21, x - 4:x + 5] = _POSTERIOR_HU

    localization["correction_scope_bbox"] = {
        "x_min": min(lr_centers) - 4, "x_max": max(lr_centers) + 4,
        "y_min": min(ap_centers) - 4, "y_max": max(ap_centers) + 20,
        "z_min": min(_SI_CENTERS) - 3, "z_max": max(_SI_CENTERS) + 3,
    }
    direction = _source_direction(orientation)
    image, mask = (sitk.GetImageFromArray(array) for array in (hu, target))
    for item in (image, mask):
        item.SetOrigin((-31.0, -47.0, -60.0))
        item.SetDirection(tuple(direction.ravel()))
    localization["image"].update({
        "size_xyz": list(image.GetSize()), "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection()),
    })
    if orientation == "oblique_reversed_storage":
        return _reflect_storage(localization, image, mask)
    return localization, image, mask


def _plan(part, localization, fill_percent):
    request = Request(
        "/tmp/dicom", "/tmp/output", slice_part=part,
        plane_configs=[PlaneConfig(
            PlaneTypeEnum.矢状位, 1.0, 5.0, ["骨窗"],
            slice_type=SliceType.CPR, fill_percent=fill_percent,
        )],
    )
    return select_planner(part.value, SliceType.CPR.value).build_plans(
        request, localization,
    ).plans[0].to_dict()


def _marker_core(pixels, value):
    """只取最大 HU 平台，忽略其他标记边缘插值偶然落入相同 HU 的像素。"""

    regions, count = ndimage.label(np.abs(pixels - value) <= 1.0)
    assert count > 0, f"实际采样丢失 HU={value} 的椎体主体"
    sizes = np.bincount(regions.ravel())
    sizes[0] = 0
    return np.argwhere(regions == sizes.argmax())


def _assert_curvature_and_coverage(output, bend_sign):
    pixels = sitk.GetArrayFromImage(output)[0]
    spacing = np.asarray(output.GetSpacing()[:2])
    cores = [_marker_core(pixels, value) for value in _BODY_HU]
    centers = np.array([core.mean(axis=0)[::-1] * spacing for core in cores])
    for value, core in zip(_BODY_HU, cores):
        # 不仅检出一两个边缘像素：每段都必须保留主体的横向和纵向核心。
        assert len(np.unique(core[:, 1])) >= 4, value
        assert len(np.unique(core[:, 0])) >= 2, value
        assert len(core) >= 8, value
        # 两次线性插值会侵蚀严格相等的 HU 平台；保留一个输出像素的边界
        # 量化余量，同时上面的二维连通核心要求仍会拒绝孤立点或细线。
        assert (np.ptp(core[:, 1]) + 2) * spacing[0] >= 6.0, value
        assert (np.ptp(core[:, 0]) + 2) * spacing[1] >= 3.0, value
    assert np.all(np.diff(centers[:, 1]) > 10.0), "成片应保持从头到足的椎体顺序"
    interpolation = (centers[2, 1] - centers[0, 1]) / (centers[-1, 1] - centers[0, 1])
    straight_ap = centers[0, 0] + interpolation * (centers[-1, 0] - centers[0, 0])
    observed_bend = centers[2, 0] - straight_ap
    assert 12.0 <= bend_sign * observed_bend <= 20.0, (
        "矢状位 CPR 应保留 16 mm 的 AP 生理弯，不能将全部椎体重新居中拉直；"
        f"预期弯曲符号 {bend_sign}，当前中段偏移 {observed_bend:.3f} mm"
    )
    posterior = np.argwhere(pixels >= _POSTERIOR_HU - 1.0)
    assert len(posterior) > 0, "后方骨性标记应保留"
    assert posterior[:, 1].mean() * spacing[0] > centers[:, 0].mean() + 10.0, (
        "椎体应在左，棘突应在右"
    )


@pytest.mark.parametrize("part", tuple(_PART_LABELS))
@pytest.mark.parametrize("orientation", ("identity", "oblique", "oblique_reversed_storage"))
@pytest.mark.parametrize("fill_percent", (0, 80))
def test_sagittal_cpr_preserves_ap_curve_while_following_lr_curve(part, orientation, fill_percent):
    localization, image, target = _curved_spine(part, orientation)
    plan = _plan(part, localization, fill_percent)
    sampler = CurvedStackResampler()
    masks = [sampler.sample_slice(target, plan, layer, is_mask=True) for layer in plan["slices"]]
    bounds = curved_group_bounds(masks)
    index = min(range(len(plan["slices"])), key=lambda i: abs(plan["slices"][i]["offset_mm"]))
    raw = sampler.sample_slice(image, plan, plan["slices"][index])
    _assert_curvature_and_coverage(raw, _AP_CURVATURE_SIGN[part])
    framed, _ = frame_curved_slice(raw, masks[index], fill_percent, bounds)
    _assert_curvature_and_coverage(framed, _AP_CURVATURE_SIGN[part])


@pytest.mark.parametrize("part", tuple(_PART_LABELS))
@pytest.mark.parametrize("orientation", ("identity", "oblique", "oblique_reversed_storage"))
@pytest.mark.parametrize("fill_percent", (0, 80))
def test_ap_only_curve_keeps_the_physical_hu_ramp(part, orientation, fill_percent):
    """没有 LR 侧弯时应等价于直矢状面，AP 弯不能造成逐行横向平移。

    用独立 LPS 线性场计算每个输出像素的预期 HU，能够检出按真实三维
    中心线逐行采样导致的 AP 展平，而非仅验证生产代码自己的曲线字段。
    """

    localization, image, target = _curved_spine(part, orientation, lateral_bend=False)
    gradient = np.array((3.0, 7.0, 11.0))
    source_axes = np.asarray(image.GetDirection()).reshape(3, 3)
    source_gradient = (source_axes.T @ gradient) * np.asarray(image.GetSpacing())
    z, y, x = np.indices(sitk.GetArrayViewFromImage(image).shape, dtype=np.float32)
    ramp = np.asarray(
        900.0 + np.asarray(image.GetOrigin()) @ gradient
        + x * source_gradient[0] + y * source_gradient[1] + z * source_gradient[2],
        dtype=np.float32,
    )
    ramp_image = sitk.GetImageFromArray(ramp)
    ramp_image.CopyInformation(image)
    plan = _plan(part, localization, fill_percent)
    sampler = CurvedStackResampler()
    masks = [sampler.sample_slice(target, plan, layer, is_mask=True) for layer in plan["slices"]]
    index = min(range(len(plan["slices"])), key=lambda i: abs(plan["slices"][i]["offset_mm"]))
    raw = sampler.sample_slice(ramp_image, plan, plan["slices"][index])
    framed, _ = frame_curved_slice(raw, masks[index], fill_percent, curved_group_bounds(masks))
    for output in (raw, framed):
        pixels = sitk.GetArrayFromImage(output)[0]
        r, c = np.indices(pixels.shape, dtype=np.float64)
        axes = np.asarray(output.GetDirection()).reshape(3, 3)
        expected = (
            900.0 + np.asarray(output.GetOrigin()) @ gradient
            + c * output.GetSpacing()[0] * (axes[:, 0] @ gradient)
            + r * output.GetSpacing()[1] * (axes[:, 1] @ gradient)
        )
        physical = (
            np.asarray(output.GetOrigin())[None, None, :]
            + c[..., None] * output.GetSpacing()[0] * axes[:, 0]
            + r[..., None] * output.GetSpacing()[1] * axes[:, 1]
        )
        raw_axes = np.asarray(raw.GetDirection()).reshape(3, 3)
        raw_indices = ((physical - raw.GetOrigin()) @ raw_axes) / raw.GetSpacing()
        # 取景会平移并扩大原展开图视野，超出原图的边缘按设计填空气。
        # 按独立物理坐标选原图内的插值支撑区，不能按实际 HU 过滤错误值。
        supported = np.all(
            (raw_indices[..., :2] >= 1.0)
            & (raw_indices[..., :2] <= np.asarray(raw.GetSize()[:2]) - 2.0),
            axis=2,
        )
        assert supported.sum() > pixels.size / 2
        np.testing.assert_allclose(pixels[supported], expected[supported], rtol=0.0, atol=5e-4)
