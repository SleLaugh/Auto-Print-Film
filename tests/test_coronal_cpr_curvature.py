"""冠状 CPR 展开前后弯，同时保留侧弯、双侧同层及末端完整覆盖。"""

import copy

import numpy as np
import pytest
import SimpleITK as sitk

from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.base import select_planner
from src.reconstruction.curved_stack import CurvedStackResampler
from src.reconstruction.framing import curved_group_bounds, frame_curved_slice
from src.stages.stage_reconstruction import ReconstructionStage
from tests import test_sagittal_cpr_curvature as sagittal
from tests import test_sagittal_cpr_localizer_geometry as localizer
from tests import test_sagittal_cpr_terminal_coverage as terminal


_PARTS = (SlicePartEnum.胸腰椎, SlicePartEnum.全脊柱)
_ORIENTATIONS = ("identity", "oblique", "oblique_reversed_storage")


def _plan(part, localization, fill_percent=0):
    request = Request(
        "/tmp/dicom", "/tmp/output", slice_part=part,
        plane_configs=[PlaneConfig(
            PlaneTypeEnum.冠状位, 1.0, 5.0, ["骨窗"],
            slice_type=SliceType.CPR, fill_percent=fill_percent,
        )],
    )
    return select_planner(part.value, SliceType.CPR.value).build_plans(
        request, localization,
    ).plans[0].to_dict()


def _assert_lateral_bend_and_body_coverage(output):
    pixels = sitk.GetArrayFromImage(output)[0]
    spacing = np.asarray(output.GetSpacing()[:2])
    cores = [sagittal._marker_core(pixels, value) for value in sagittal._BODY_HU]
    centers = np.array([core.mean(axis=0)[::-1] * spacing for core in cores])
    for value, core in zip(sagittal._BODY_HU, cores):
        assert len(np.unique(core[:, 1])) >= 4, value
        assert len(np.unique(core[:, 0])) >= 2, value
        assert len(core) >= 8, value
        # 完整 9 mm 椎体的左右核心均应保留，不能仅检出边缘一条线。
        assert (np.ptp(core[:, 1]) + 2) * spacing[0] >= 6.0, value
    assert np.all(np.diff(centers[:, 1]) > 10.0), "冠状成片应保持从头到足顺序"
    ratio = (centers[2, 1] - centers[0, 1]) / (centers[-1, 1] - centers[0, 1])
    baseline = centers[0, 0] + ratio * (centers[-1, 0] - centers[0, 0])
    observed_bend = centers[2, 0] - baseline
    assert 20.0 <= observed_bend <= 28.0, (
        "冠状 CPR 应保留 24 mm 左右侧弯，不能把全部椎体居中拉直；"
        f"当前中段偏移 {observed_bend:.3f} mm"
    )


@pytest.mark.parametrize("part", _PARTS)
@pytest.mark.parametrize("orientation", _ORIENTATIONS)
@pytest.mark.parametrize("fill_percent", (0, 80))
def test_coronal_cpr_keeps_lateral_bend_and_full_body_width(part, orientation, fill_percent):
    """正式规划、HU 采样和取景后的像素均须保持侧弯及首末椎体。"""

    localization, image, target = sagittal._curved_spine(part, orientation)
    before = copy.deepcopy(localization)
    plan = _plan(part, localization, fill_percent)
    sampler = CurvedStackResampler()
    masks = [sampler.sample_slice(target, plan, layer, is_mask=True) for layer in plan["slices"]]
    bounds = curved_group_bounds(masks)
    index = min(range(len(plan["slices"])), key=lambda i: abs(plan["slices"][i]["offset_mm"]))
    raw = sampler.sample_slice(image, plan, plan["slices"][index])
    _assert_lateral_bend_and_body_coverage(raw)
    framed, _ = frame_curved_slice(raw, masks[index], fill_percent, bounds)
    _assert_lateral_bend_and_body_coverage(framed)
    assert localization == before
    # 在同一定位结果上规划冠状位不能改变矢状 CPR 已有的投影语义。
    sagittal_plan = sagittal._plan(part, localization, fill_percent)
    sagittal._assert_curvature_and_coverage(
        sampler.sample_slice(image, sagittal_plan, min(
            sagittal_plan["slices"], key=lambda item: abs(item["offset_mm"]),
        )), sagittal._AP_CURVATURE_SIGN[part],
    )


@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_coronal_rows_sample_both_sides_at_the_same_ap_and_si_depth(orientation):
    """独立 AP/SI HU 线性场揭示左右采样混入不同深度，特别是末端 S1。"""

    part = SlicePartEnum.胸腰椎
    localization, image, _ = sagittal._curved_spine(part, orientation)
    # 线性场在物理 LR 上恒定，AP 和 SI 均变化；横向混入任意一个方向
    # 都会导致左右像素出现可测的差异，不能通过居中或缩小窗口掩盖。
    native_axes = sagittal._source_direction(orientation)
    gradient = native_axes[:, 1] * 5.0 + native_axes[:, 2] * 2.0
    axes = np.asarray(image.GetDirection()).reshape(3, 3)
    source_gradient = (axes.T @ gradient) * np.asarray(image.GetSpacing())
    z, y, x = np.indices(sitk.GetArrayViewFromImage(image).shape, dtype=np.float32)
    ramp = 900.0 + np.asarray(image.GetOrigin()) @ gradient
    ramp = np.asarray(ramp + x * source_gradient[0] + y * source_gradient[1] + z * source_gradient[2],
                      dtype=np.float32)
    source = sitk.GetImageFromArray(ramp)
    source.CopyInformation(image)
    plan = _plan(part, localization)
    layer = min(plan["slices"], key=lambda item: abs(item["offset_mm"]))
    pixels = sitk.GetArrayFromImage(CurvedStackResampler().sample_slice(source, plan, layer))[0]
    # 边缘体素也在 112^2 宽的源图内；检测整个输出宽度，而非只看中心点。
    spread = np.ptp(pixels, axis=1)
    assert float(spread.max()) < 1e-3, "左右两侧应处于相同 AP/SI 深度"
    assert float(spread[-max(2, len(spread) // 6):].max()) < 1e-3, "S1 两侧不应出现累积扭转"


@pytest.mark.parametrize("part", _PARTS)
@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_fixed_lateral_reference_keeps_the_entire_off_center_s1(part, orientation, monkeypatch):
    """平均 LR 参考明显偏向上段时，bbox 总宽不能保证足侧完整覆盖。"""

    # 上段四椎体在 X=44，S1 在 X=88；总体 bbox 为 [40, 92]，但平均
    # 参考 X=52.8。直接居中 52 mm 宽只覆盖到 X=78.8，会裁掉全部 S1。
    monkeypatch.setattr(sagittal, "_LR_CENTERS", (44, 44, 44, 44, 88))
    localization, image, target = sagittal._curved_spine(part, orientation)
    plan = _plan(part, localization, 80)
    sampler = CurvedStackResampler()
    masks = [sampler.sample_slice(target, plan, layer, is_mask=True) for layer in plan["slices"]]
    index = min(range(len(plan["slices"])), key=lambda i: abs(plan["slices"][i]["offset_mm"]))
    raw = sampler.sample_slice(image, plan, plan["slices"][index])
    framed, _ = frame_curved_slice(raw, masks[index], 80, curved_group_bounds(masks))
    for output in (raw, framed):
        core = sagittal._marker_core(sitk.GetArrayFromImage(output)[0], sagittal._BODY_HU[-1])
        # 原始 9 mm 平台经过采样和取景两次插值，严格同 HU 核心可侵蚀约
        # 一个输出体素；仍需保留至少 7 mm，不能只残留一条边缘。
        assert (np.ptp(core[:, 1]) + 2) * output.GetSpacing()[0] >= 7.0, "偏心 S1 的左右主体不能被裁窄"


@pytest.mark.parametrize("part", _PARTS)
@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_coronal_sampling_consumes_the_measured_anatomical_lr_and_ap_together(part, orientation):
    """第五阶段可靠对称轴必须控制实采横向，不能再次回到源图像 X/Y。"""

    localization, image, _ = sagittal._curved_spine(part, orientation)
    original_sagittal = sagittal._plan(part, localization, 0)
    native_axes = sagittal._source_direction(orientation)
    yaw, roll = np.deg2rad((12.0, 7.0))
    lr = native_axes @ np.array((np.cos(yaw) * np.cos(roll),
                                  np.sin(yaw) * np.cos(roll), -np.sin(roll)))
    ap = native_axes[:, 1] - np.dot(native_axes[:, 1], lr) * lr
    ap /= np.linalg.norm(ap)
    si = np.cross(lr, ap)
    localization["anatomical_geometry"] = {
        "global_frame": {
            "origin": localizer._xyz(image.TransformIndexToPhysicalPoint((64, 56, 64))),
            "left_right_axis": localizer._xyz(lr),
            "anterior_posterior_axis": localizer._xyz(ap),
            "superior_inferior_axis": localizer._xyz(si),
            "bounds": {axis: {"minimum": -1000.0, "maximum": 1000.0}
                       for axis in ("left_right", "anterior_posterior", "superior_inferior")},
            "method": "spine_reflection_symmetry", "confidence": 1.0,
            "applied": True, "warnings": [],
        },
        "local_frames": {},
    }
    plan = _plan(part, localization)
    np.testing.assert_allclose([plan["curve_projection_axis"][axis] for axis in "xyz"], lr, atol=1e-10)
    np.testing.assert_allclose([plan["offset_axis"][axis] for axis in "xyz"], ap, atol=1e-10)
    sampler = localizer._CapturingResampler()
    projection_values = []
    for layer in plan["slices"]:
        sampler.sampled_grids.clear()
        sampler.sample_slice(image, plan, layer)
        grids = np.asarray(sampler.sampled_grids)
        across = np.diff(grids, axis=2)
        np.testing.assert_allclose(
            across, np.broadcast_to(lr * plan["output_spacing_mm"]["x"], across.shape), atol=1e-10,
        )
        projection_values.extend((grids[:, :, grids.shape[2] // 2, :] @ lr).ravel())
    assert np.ptp(projection_values) < 1e-10, "层间偏移不能沿解剖左右轴产生漂移"
    assert sagittal._plan(part, localization, 0) == original_sagittal


@pytest.mark.parametrize("part", _PARTS)
@pytest.mark.parametrize("lr_slope_sign", (-1, 1))
@pytest.mark.parametrize("oblique", (False, True))
def test_coronal_cpr_covers_first_vertebra_and_s1_raw_terminal_bands(
    tmp_path, part, lr_slope_sign, oblique,
):
    """左右斜向首尾线投影后，真实原始标签的头足边缘仍须进入成片。"""

    source, localization = terminal._terminal_case(tmp_path, part, lr_slope_sign, oblique)
    labels = sitk.ReadImage(localization["seg_for_reconstruction_path"])
    # 交换现有 fixture 的索引 X/Y，得到 LR 斜向而 AP 恒定的独立病例。
    images = []
    for image in (source, labels):
        exchanged = sitk.GetImageFromArray(sitk.GetArrayFromImage(image).transpose(0, 2, 1))
        exchanged.CopyInformation(image)
        images.append(exchanged)
    source, labels = images
    sitk.WriteImage(labels, localization["seg_for_reconstruction_path"])
    boxes = [localization["correction_scope_bbox"]]
    for annotation in localization["coords"].values():
        center = annotation["centroid"]
        center["x"], center["y"] = center["y"], center["x"]
        boxes.append(annotation["bbox"])
    for box in boxes:
        for bound in ("min", "max"):
            box[f"x_{bound}"], box[f"y_{bound}"] = box[f"y_{bound}"], box[f"x_{bound}"]
    plan = _plan(part, localization)
    layer = min(plan["slices"], key=lambda item: abs(item["offset_mm"]))
    pixels = sitk.GetArrayFromImage(CurvedStackResampler().sample_slice(source, plan, layer))[0]
    superior = np.argwhere(np.abs(pixels - terminal._SUPERIOR_HU) <= 1.0)
    inferior = np.argwhere(np.abs(pixels - terminal._INFERIOR_HU) <= 1.0)
    assert len(superior) >= 3, "首椎体 z=89..90 的原始头侧标记应保留"
    assert len(inferior) >= 3, "S1 z=30..31 的原始足侧标记应保留"
    assert superior[:, 0].max() < inferior[:, 0].min()


@pytest.mark.parametrize("part", tuple(part.value for part in _PARTS))
@pytest.mark.parametrize("oblique", (False, True))
@pytest.mark.parametrize("thickness", (0.1, 4.0))
def test_coronal_projection_validates_and_localizer_matches_actual_sampling(part, oblique, thickness):
    image, plan, _, _, reference = localizer._case(part, oblique=oblique, thickness=thickness)
    axes = np.asarray(image.GetDirection()).reshape(3, 3)
    lr, ap, _ = axes.T
    plan.update({
        "plan_id": f"{part}_CPR_冠状位", "plane": "冠状位", "localizer_plane": "矢状位",
        "curve_projection_axis": localizer._xyz(lr), "offset_axis": localizer._xyz(ap),
    })
    curve = np.array([[point[axis] for axis in "xyz"] for point in plan["curve_points"]])
    projected = curve - np.outer((curve - reference) @ lr, lr)
    plan["extent_mm"]["height"] = float(np.linalg.norm(np.diff(projected, axis=0), axis=1).sum())
    for layer in plan["slices"]:
        layer.update(center=localizer._xyz(reference + layer["offset_mm"] * ap),
                     normal=localizer._xyz(ap), row_dir=localizer._xyz(lr))
    ReconstructionStage._validate_plan(plan)
    before = copy.deepcopy(plan)
    displayed = ReconstructionStage._curved_localizer_projected_curves(plan=plan, fixed_axis=0)
    sampler = localizer._CapturingResampler()
    for layer in plan["slices"]:
        sampler.sampled_grids.clear()
        sampler.sample_slice(image, plan, layer)
        grids = np.asarray(sampler.sampled_grids)
        centers = grids[:, :, grids.shape[2] // 2, :].mean(axis=0)
        np.testing.assert_allclose(centers, displayed[layer["offset_mm"]], atol=1e-10)
        np.testing.assert_allclose(centers @ lr, reference @ lr, atol=1e-10)
        # 厚层可沿 AP/SI 变化，但左右展开方向必须严格平行 LR。
        across = np.diff(grids, axis=2)
        np.testing.assert_allclose(across, np.broadcast_to(lr * plan["output_spacing_mm"]["x"], across.shape),
                                   atol=1e-10)
    assert plan == before
