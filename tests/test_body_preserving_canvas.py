"""正式肺/髋画布排除外部空气与可分离床板，同时保留完整人体。"""

import copy
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from scipy import ndimage

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.reconstruction.framing import TargetFramer
from src.reconstruction.source_framing import BodyImageSupport, SourceImageSupport
from src.stages.stage_reconstruction import ReconstructionStage


def _point(values):
    return dict(zip("xyz", map(float, values)))


def _vector(value):
    return np.asarray([value[axis] for axis in "xyz"], dtype=float)


def _fixture(part, percent=100, *, plane="横状位", tilted=False):
    """体表、床板和目标各有独立真值，骶尾突出不在目标标签里。"""
    shape = (20, 160, 200)
    spacing = np.asarray((1.0, 1.0, 2.0))
    direction = np.eye(3)
    if tilted:
        yaw, pitch = np.deg2rad((23, 17))
        direction = np.array([
            [np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1],
        ]) @ np.array([
            [1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)],
        ]) @ np.diag((-1.0, 1.0, -1.0))
    y, x = np.ogrid[:shape[1], :shape[2]]
    contour = ((x - 99) / 62) ** 2 + ((y - 68) / 35) ** 2 <= 1
    body = np.broadcast_to(contour, shape).copy()
    body[:, 97:116, 94:105] = True  # 后方窄骨性突出，不能因目标较浅而截断。
    labels = np.zeros(shape, dtype=np.uint8)
    labels[3:14, 48:76, 72:125] = 1
    labels[14:17, 72:77, 115:122] = 1
    labels[7:11, 44:79, 68:130] = 1  # 最大目标参考层不在组首。
    raw_array = np.where(body, 40.0, -1024.0).astype(np.float32)
    raw_array[labels > 0] = -800 if part == "肺部" else 700
    raw_array[:, 106:116, 94:105] = 1800
    # 床与身体相距 1mm；先闭合整幅阈值图会把两者粘在一起。
    table = np.zeros(shape, dtype=bool)
    table[:, 117:120, 5:195] = True
    raw_array[table] = 1500
    raw, seg = (sitk.GetImageFromArray(value) for value in (raw_array, labels))
    for image in (raw, seg):
        image.SetSpacing(tuple(spacing))
        image.SetDirection(tuple(direction.ravel()))
        image.SetOrigin((-92.0, 21.0, 11.0))
    row_axis, col_axis, depth_axis, depths = {
        "横状位": (0, 1, 2, (4, 9, 15, 18)),
        "冠状位": (0, 2, 1, (48, 68, 90, 108)),
        "矢状位": (1, 2, 0, (75, 98, 120, 145)),
    }[plane]
    row, col = direction[:, row_axis], direction[:, col_axis]
    if tilted:
        # 源存储已旋转并翻转，再相对源网格斜切，避免仅测坐标共同旋转。
        angle = np.deg2rad(13)
        col = col * np.cos(angle) + np.cross(row, col) * np.sin(angle)
    normal = np.cross(row, col)
    axes = np.column_stack((row, col, normal))
    source_size = np.asarray(raw.GetSize())
    plan = {
        "plan_id": "body_canvas", "slice_part": part, "slice_type": "MPR",
        "plane": plane, "geometry_type": "plane_stack", "source_labels": ["target"],
        "fill_percent": percent, "slice_thickness": 4.0, "slice_spacing": 10.0,
        "window_presets": ["骨窗", "软组织窗"],
        "extent_mm": dict(zip(("width", "height"), (source_size[[row_axis, col_axis]] - 1) * spacing[[row_axis, col_axis]])),
        "output_spacing_mm": dict(zip("xyz", (*spacing[[row_axis, col_axis]], 0.0))),
        "slices": [],
    }
    for number, depth in enumerate(depths, 1):
        index = (source_size - 1) / 2
        index[depth_axis] = depth
        index[row_axis] += 0.7 * number
        index[col_axis] -= 0.3 * number
        plan["slices"].append({
            "index": number, "center": _point(raw.TransformContinuousIndexToPhysicalPoint(tuple(index))),
            "row_dir": _point(row), "col_dir": _point(col), "normal": _point(normal),
        })
    plan["range_start"], plan["range_end"] = plan["slices"][0]["center"], plan["slices"][-1]["center"]
    plan["path_points"] = [plan["range_start"], plan["range_end"]]
    return raw, seg, plan, {"target": {"label_ids": [1]}}, body, axes


def _body_vertices(mask, image, item, axes, thickness):
    """独立裁实体体素盒，取与厚层正厚度相交的角点及棱面交点。"""
    voxel_axes = np.asarray(image.GetDirection()).reshape(3, 3) * image.GetSpacing()
    points = np.argwhere(mask)[:, ::-1] @ voxel_axes.T + image.GetOrigin()
    normal, center = axes[:, 2], _vector(item["center"])
    half = thickness / 2
    # 轴对齐采样会包含在名义厚层外但与厚层相交的真实源层中心。
    for axis in range(3):
        dot = float(np.asarray(image.GetDirection()).reshape(3, 3)[:, axis] @ normal)
        if abs(abs(dot) - 1) <= 1e-6:
            offsets = []
            for layer in range(image.GetSize()[axis]):
                source_index = [0, 0, 0]
                source_index[axis] = layer
                offset = (np.asarray(image.TransformIndexToPhysicalPoint(source_index)) - center) @ normal
                source_half = image.GetSpacing()[axis] / 2
                if min(offset + source_half, half) - max(offset - source_half, -half) > 0:
                    offsets.append(offset)
            lower, upper = min(-half, min(offsets)), max(half, max(offsets))
            break
    else:
        lower, upper = -half, half
    depths = (points - center) @ normal
    half_support = np.abs(voxel_axes.T @ normal).sum() / 2
    points = points[(depths + half_support > lower + 1e-8) & (depths - half_support < upper - 1e-8)]
    if not len(points):
        return np.empty((0, 3))
    signs = np.asarray(list(product((-0.5, 0.5), repeat=3)))
    corners = points[:, None, :] + (signs @ voxel_axes.T)[None, :, :]
    depths = (corners - center) @ normal
    vertices = [corners[(depths >= lower - 1e-9) & (depths <= upper + 1e-9)]]
    for first, second in combinations(range(8), 2):
        if np.count_nonzero(signs[first] != signs[second]) != 1:
            continue
        span = depths[:, second] - depths[:, first]
        if np.max(np.abs(span)) < 1e-12:
            continue
        for boundary in (lower, upper):
            ratio = (boundary - depths[:, first]) / span
            crossing = (ratio >= 0) & (ratio <= 1)
            vertices.append(corners[crossing, first] + ratio[crossing, None] * (corners[crossing, second] - corners[crossing, first]))
    return np.concatenate(vertices)


def _geometry(ds):
    size = np.asarray((ds.Columns, ds.Rows))
    spacing = np.asarray(ds.PixelSpacing, dtype=float)[::-1]
    axes = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3).T
    origin = np.asarray(ds.ImagePositionPatient, dtype=float)
    center = origin + axes @ ((size - 1) * spacing / 2)
    return size, spacing, axes, origin, center


def _stage(tmp_path, raw, seg, plan, *, extra_plans=()):
    raw_path, seg_path = tmp_path / "raw.mha", tmp_path / "seg.mha"
    sitk.WriteImage(raw, str(raw_path))
    sitk.WriteImage(seg, str(seg_path))
    request = Request(str(tmp_path / "source"), str(tmp_path / "out"),
                      slice_part=SlicePartEnum(plan["slice_part"]),
                      plane_configs=[PlaneConfig(PlaneTypeEnum(plan["plane"]), plan["slice_thickness"],
                                                 plan["slice_spacing"], plan["window_presets"],
                                                 fill_percent=plan["fill_percent"])])
    context = ProcessingContext(request)
    context.set_result("localization", {"raw_for_reconstruction_path": str(raw_path),
                                       "seg_for_reconstruction_path": str(seg_path),
                                       "coords": {"target": {"label_ids": [1]}}})
    context.set_result("planning", {"plans": [plan, *extra_plans]})
    ReconstructionStage({"parallel": plan["fill_percent"] == 100}).execute(context)
    return context.get_result("reconstruction")["outputs"][0]


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("percent", (80, 100))
@pytest.mark.parametrize("tilted", (False, True))
def test_final_canvas_magnifies_without_air_or_table_and_preserves_unlabelled_body(tmp_path, part, percent, tilted):
    raw, seg, plan, coords, truth, axes = _fixture(part, percent, tilted=tilted)
    original = copy.deepcopy(plan)
    target = TargetFramer(seg, coords, plan)
    old_source = TargetFramer(seg, coords, plan)
    old_source._source_support = SourceImageSupport(raw)
    output = _stage(tmp_path, raw, seg, plan)
    common_geometry = []
    shifts = []
    visible_landmarks = 0
    for window in plan["window_presets"]:
        for item in plan["slices"]:
            ds = pydicom.dcmread(Path(output["window_outputs"][window]) / f"{item['index']}.dcm")
            size, spacing, output_axes, origin, center = _geometry(ds)
            _, expected = target.plane_geometry(plan, item)
            if percent == 100:
                details = output["framing"][item["index"] - 1]
                np.testing.assert_allclose(_vector(details["target_center_before_skin_edge"]), _vector(expected["center"]), atol=1e-5)
                np.testing.assert_allclose((center - _vector(expected["center"])) @ axes[:, :2],
                                           details["source_preservation"]["center_adjustment_mm"], atol=1e-5)
            else:
                np.testing.assert_allclose(center, _vector(expected["center"]), atol=1e-5)
            vertices = _body_vertices(truth, raw, item, axes, plan["slice_thickness"])
            assert len(vertices) > 0
            assert np.all(np.abs((vertices - center) @ axes[:, :2]) <= size * spacing / 2 + 1e-5)
            previous_plan, previous = old_source.plane_geometry(plan, item)
            previous_fov = np.asarray(previous["framing"]["output_size_xy"]) * np.asarray([
                previous_plan["output_spacing_mm"][axis] for axis in "xy"
            ])
            # 不能只保安全却继续把空气与床板撑成同样大的画布。
            assert np.prod(size * spacing) < 0.8 * np.prod(previous_fov)
            assert np.all(spacing < np.asarray([previous_plan["output_spacing_mm"][axis] for axis in "xy"]))
            # 后方无标签的骨性突出在最终像素里仍可见。
            index = np.asarray(raw.TransformPhysicalPointToContinuousIndex(tuple(_vector(item["center"]))))
            index[:2] = (99, 111)
            voxel_axes = np.asarray(raw.GetDirection()).reshape(3, 3) * raw.GetSpacing()
            landmark = np.asarray(raw.TransformContinuousIndexToPhysicalPoint(tuple(index)))
            index[2] += ((_vector(item["center"]) - landmark) @ axes[:, 2]) / (voxel_axes[:, 2] @ axes[:, 2])
            if 1 <= index[2] < raw.GetSize()[2] - 2:
                landmark = np.asarray(raw.TransformContinuousIndexToPhysicalPoint(tuple(index)))
                pixel = (landmark - origin) @ output_axes / spacing
                assert np.all(pixel >= 0) and np.all(pixel < size)
                x, y = np.rint(pixel).astype(int)
                assert ds.pixel_array[max(0, y - 2):y + 3, max(0, x - 2):x + 3].max() > 45000
                visible_landmarks += 1
            common_geometry.append((*size, *spacing))
            shifts.append((center - _vector(item["center"])) @ axes[:, :2])
    np.testing.assert_allclose(common_geometry, [common_geometry[0]] * len(common_geometry))
    assert visible_landmarks >= 4
    if part == "肺部":
        np.testing.assert_allclose(shifts, [shifts[0]] * len(shifts), atol=1e-5)
        assert output["framing"][0]["reference_slice_index"] != plan["slices"][0]["index"]
    assert all(item["source_preservation"]["mode"] == "complete_body_slab" for item in output["framing"])
    assert all("vertical_source_support" not in item for item in output["framing"])
    assert plan == original


@pytest.mark.parametrize("part", ("肺部", "髋骨"))
@pytest.mark.parametrize("plane", ("横状位", "冠状位", "矢状位"))
@pytest.mark.parametrize("tilted", (False, True))
def test_body_guard_covers_independent_voxel_truth_in_every_mpr_plane(part, plane, tilted):
    raw, seg, plan, coords, truth, axes = _fixture(part, plane=plane, tilted=tilted)
    body = TargetFramer(seg, coords, plan, source_image=raw)
    baseline = TargetFramer(seg, coords, plan)
    geometries = []
    for item in plan["slices"]:
        framed_plan, framed = body.plane_geometry(plan, item)
        _, old = baseline.plane_geometry(plan, item)
        center = _vector(framed["center"])
        np.testing.assert_allclose(_vector(framed["framing"]["target_center_before_skin_edge"]), _vector(old["center"]), atol=1e-8)
        np.testing.assert_allclose((center - _vector(old["center"])) @ axes[:, :2],
                                   framed["framing"]["source_preservation"]["center_adjustment_mm"], atol=1e-8)
        size = np.asarray(framed["framing"]["output_size_xy"])
        spacing = np.asarray([framed_plan["output_spacing_mm"][axis] for axis in "xy"])
        vertices = _body_vertices(truth, raw, item, axes, plan["slice_thickness"])
        if len(vertices):
            assert np.all(np.abs((vertices - center) @ axes[:, :2]) <= size * spacing / 2 + 1e-6)
        assert abs((center - _vector(item["center"])) @ axes[:, 2]) < 1e-7
        geometries.append((*size, *spacing))
    np.testing.assert_allclose(geometries, [geometries[0]] * len(geometries))


def test_unreliable_larger_source_layer_falls_back_locally_instead_of_using_inner_organ():
    raw, _, _, _, truth, _ = _fixture("髋骨")
    array = sitk.GetArrayFromImage(raw)
    y, x = np.ogrid[:array.shape[1], :array.shape[2]]
    large = ((x - 99) / 90) ** 2 + ((y - 76) / 65) ** 2 <= 1
    shell = large & ~ndimage.binary_erosion(large, iterations=2)
    array[6] = -1024
    array[6, large] = -850
    array[6, shell] = 40
    array[6, 65:100, 85:120] = 40  # 内部厚器官不能冒充外轮廓。
    source = sitk.GetImageFromArray(array)
    source.CopyInformation(raw)
    body = BodyImageSupport(source)
    full = SourceImageSupport(source)
    axes = np.eye(3)
    center = np.asarray(source.TransformIndexToPhysicalPoint((99, 76, 6)))
    plan = {"slice_thickness": 0.5}
    bounds = body.slab_bounds(plan, center, axes)
    required = full.slab_bounds(plan, center, axes)
    assert np.all(bounds[0] <= required[0] + 1e-7)
    assert np.all(bounds[1] >= required[1] - 1e-7)
    # 相邻可靠层仍可去除外部空气，不能退化为整卷一律完整源盒。
    neighbor = np.asarray(source.TransformIndexToPhysicalPoint((99, 68, 5)))
    reliable = body.slab_bounds(plan, neighbor, axes)
    assert np.prod(reliable[1] - reliable[0]) < 0.8 * np.prod(required[1] - required[0])
    assert large.sum() > truth[6].sum() * 1.5


@pytest.mark.parametrize("connected", (False, True))
def test_disconnected_legs_and_ambiguous_connected_bed_never_remove_anatomy(connected):
    y, x = np.ogrid[:180, :240]
    left = ((x - 65) / 32) ** 2 + ((y - 75) / 45) ** 2 <= 1
    right = ((x - 175) / 32) ** 2 + ((y - 75) / 45) ** 2 <= 1
    truth = np.broadcast_to(left | right, (3, 180, 240)).copy()
    array = np.where(truth, 40, -1024).astype(np.int16)
    array[:, 122:125, :] = 1200
    if connected:
        array[:, 118:123, 65] = 40
    raw = sitk.GetImageFromArray(array)
    body = BodyImageSupport(raw)
    center, axes = np.asarray((120.0, 75.0, 1.0)), np.eye(3)
    bounds = body.slab_bounds({"slice_thickness": 1.0}, center, axes)
    xy = np.argwhere(left | right)[:, ::-1]
    assert np.all(bounds[0] <= xy.min(axis=0) - 0.5)
    assert np.all(bounds[1] >= xy.max(axis=0) + 0.5)
    if not connected:
        assert bounds[1][1] <= 122.5  # 体表后缘120.5mm加2mm余量，仍不受床板后缘支撑。


@pytest.mark.parametrize("kind", ("thin_arm", "skin_fragment"))
def test_small_unlabelled_anatomy_does_not_disappear_below_main_body_thresholds(kind):
    raw, _, plan, _, truth, axes = _fixture("肺部")
    array = sitk.GetArrayFromImage(raw)
    if kind == "thin_arm":
        y, x = np.ogrid[:array.shape[1], :array.shape[2]]
        fragment = ((x - 22) / 4) ** 2 + ((y - 67) / 17) ** 2 <= 1  # 9mm宽，低于主躯干门槛。
    else:
        fragment = np.zeros(array.shape[1:], dtype=bool)
        fragment[63:71, 14:16] = True
    array[:, fragment] = 50
    truth[:, fragment] = True
    source = sitk.GetImageFromArray(array)
    source.CopyInformation(raw)
    support = BodyImageSupport(source)
    for item in plan["slices"]:
        bounds = support.slab_bounds(plan, _vector(item["center"]), axes)
        vertices = _body_vertices(truth, source, item, axes, plan["slice_thickness"])
        projected = vertices @ axes[:, :2]
        assert np.all(bounds[0] <= projected.min(axis=0) + 1e-7)
        assert np.all(bounds[1] >= projected.max(axis=0) - 1e-7)


def test_stage_reuses_body_extraction_across_plans_from_same_raw_ct(tmp_path, monkeypatch):
    import src.reconstruction.source_framing as source_framing

    raw, seg, plan, _, _, _ = _fixture("肺部")
    second = copy.deepcopy(plan)
    second["plan_id"] = "body_canvas_second"
    second["fill_percent"] = 80
    calls = []
    original = source_framing.build_body_framing_support

    def counted(image, **kwargs):
        calls.append(kwargs)
        return original(image, **kwargs)

    monkeypatch.setattr(source_framing, "build_body_framing_support", counted)
    _stage(tmp_path, raw, seg, plan, extra_plans=(second,))
    assert len(calls) == 1
    assert calls[0].get("exclude_arms") is False
    assert calls[0].get("preserve_small_components") is True


@pytest.mark.parametrize("flips", tuple(product((False, True), repeat=3)))
def test_isolated_low_dose_air_noise_does_not_expand_the_body_canvas(flips):
    raw, _, plan, _, _, axes = _fixture("髋骨")
    noisy_array = sitk.GetArrayFromImage(raw)
    # 与真实病例一致：面积不足 1 mm² 的体外阈上噪点，而非独立小肢体。
    noisy_array[:, 0, 98] = -476
    noisy_array[:, 1, 89] = -397
    noisy_array[:, 6, 42] = -456
    noisy = sitk.GetImageFromArray(noisy_array)
    noisy.CopyInformation(raw)
    noisy = sitk.Flip(noisy, list(flips))
    baseline, protected = BodyImageSupport(raw), BodyImageSupport(noisy)
    for item in plan["slices"]:
        center = _vector(item["center"])
        np.testing.assert_allclose(protected.slab_bounds(plan, center, axes),
                                   baseline.slab_bounds(plan, center, axes), atol=1e-8)


def test_curved_table_wings_do_not_expand_the_canvas_but_small_limbs_remain():
    raw, _, plan, _, _, axes = _fixture("髋骨")
    array = sitk.GetArrayFromImage(raw)
    array[:, 117:120, :] = -1024  # 移除 fixture 原先平床。
    array[:, 127:130, :] = 1500
    array[:, 90:130, :3] = 1500  # 床翼的 AP bbox 与身体重叠，实体仍分开。
    array[:, 90:130, -3:] = 1500
    # 独立的细小肢体位于身体左侧，不能与宽薄曲床一起丢掉。
    array[:, 60:68, 15:20] = 40
    source = sitk.GetImageFromArray(array)
    source.CopyInformation(raw)
    support = BodyImageSupport(source)
    for item in plan["slices"]:
        lower, upper = support.slab_bounds(plan, _vector(item["center"]), axes)
        local_lower, local_upper = lower - np.asarray(raw.GetOrigin())[:2], upper - np.asarray(raw.GetOrigin())[:2]
        assert local_lower[0] <= 14.5  # 细肢体被保护。
        assert local_lower[0] > 5 and local_upper[0] < 190  # 床翼不撑宽。
        assert local_upper[1] < 125  # 后方床面不撑高。


def test_posterior_table_fragment_need_not_cross_the_body_midline():
    array = np.full((3, 200, 400), -1024, dtype=np.int16)
    array[:, 30:130, 150:330] = 40
    array[:, 50:65, 350:360] = 40  # 独立细肢体仍保留。
    array[:, 160:163, :130] = 500  # 源 FOV 截断的左侧床圈，不经过身体中央。
    raw = sitk.GetImageFromArray(array)
    raw.SetSpacing((1.0, 1.0, 2.0))
    center = np.asarray(raw.TransformIndexToPhysicalPoint((200, 80, 1)))
    lower, upper = BodyImageSupport(raw).slab_bounds({"slice_thickness": 2.0}, center, np.eye(3))
    assert lower[0] > 140 and upper[0] >= 359.5
    assert upper[1] < 140
