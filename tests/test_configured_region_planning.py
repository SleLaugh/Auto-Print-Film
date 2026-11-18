"""配置区域必须贯穿当前状位规划和取景，且不改变默认部位的采样几何。"""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from src.anatomy.regions import resolve_anatomical_regions
from src.models import AnatomicalBounds, AnatomicalFrame, AnatomicalGeometryResult, AxisBounds, PlaneConfig, Point3D, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum
from src.planning.base import select_planner
from src.planning.spine_cpr import configured_spine_cpr_labels, configured_whole_range_spine_labels
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.reconstruction.framing import TargetFramer, uses_source_preserving_canvas


def _vector(point):
    if isinstance(point, dict):
        return np.asarray([point[key] for key in "xyz"])
    return np.asarray([point.x, point.y, point.z])


def _fixture(*, constrained=True, local_frame=False):
    array = np.zeros((34, 32, 32), dtype=np.uint8)
    array[2:30, 9:22, 8:21] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 1.2, 2.0))
    image.SetOrigin((20.0, -50.0, 10.0))
    angle = np.deg2rad(19.0)
    axes = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    image.SetDirection(axes.ravel().tolist())
    indices = np.argwhere(array)
    points = indices[:, ::-1] @ (axes * image.GetSpacing()).T + image.GetOrigin()

    def frame_for(frame_axes):
        projected = points @ frame_axes
        bounds = [AxisBounds(float(projected[:, i].min()), float(projected[:, i].max())) for i in range(3)]
        return AnatomicalFrame(
            origin=Point3D(*points.mean(axis=0)),
            left_right_axis=Point3D(*frame_axes[:, 0]),
            anterior_posterior_axis=Point3D(*frame_axes[:, 1]),
            superior_inferior_axis=Point3D(*frame_axes[:, 2]),
            bounds=AnatomicalBounds(*bounds), method="fixture", confidence=1.0, applied=True,
        )

    frame = frame_for(axes)
    local_frames = {}
    if local_frame:
        tilt = np.deg2rad(13.0)
        local_axes = axes @ np.asarray([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        local_frames["axial_mpr"] = frame_for(local_axes)
    geometry = AnatomicalGeometryResult(frame, local_frames)
    region = {
        "type": "whole_range", "target_labels": ["organ"],
        "bounds": {
            "upper": {"labels": ["organ"], "edge": "upper", "reduce": "max", "offset_mm": -8.0 if constrained else 0.0},
            "lower": {"labels": ["organ"], "edge": "lower", "reduce": "min", "offset_mm": 6.0 if constrained else 0.0},
        },
    }
    regions = {"selected": region}
    solved = resolve_anatomical_regions(regions, geometry, image, {"organ": indices}, ["organ"], "spine")
    rule = {
        "regions": ["selected"],
        "face_extension_mm": {key: 0.0 for key in ("top_mm", "bottom_mm", "front_mm", "back_mm", "left_mm", "right_mm")},
    }
    localization = {
        "image": {"size_xyz": list(image.GetSize()), "spacing_xyz": list(image.GetSpacing()),
                  "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection())},
        "correction_scope_bbox": dict(zip(("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), (8, 20, 9, 21, 2, 29))),
        "anatomical_geometry": geometry.to_dict(), "anatomical_regions": solved,
        "resolved_cut_configs": {"whole_range": {"regions": regions}},
        "slice_rules": {
            plane.value: {"MPR": {**deepcopy(rule), "localizer_plane": "冠状位" if plane != PlaneTypeEnum.冠状位 else "矢状位"}}
            for plane in PlaneTypeEnum
        },
        "coords": {"organ": {"label_ids": [1]}},
    }
    return image, localization


@pytest.mark.parametrize("part", [part for part in SlicePartEnum if part.value != "骶骨"])
@pytest.mark.parametrize("plane", list(PlaneTypeEnum))
def test_default_regions_preserve_existing_plane_geometry(part, plane):
    _, localization = _fixture(constrained=False, local_frame=part is SlicePartEnum.颈椎)
    request = Request("/dicom", "/output", slice_part=part)
    config = PlaneConfig(plane, 2.0, 3.0, ["骨窗"])
    direct = deepcopy(localization)
    direct.pop("anatomical_regions")
    direct.pop("resolved_cut_configs")
    before = build_whole_range_plan_from_bbox(request, config, direct, source_labels=["organ"])
    after = build_whole_range_plan_from_bbox(request, config, localization)
    for key in ("range_start", "range_end", "target_start", "target_end", "extent_mm", "slices", "localizer_reference", "sampling_details"):
        assert after.to_dict().get(key) == before.to_dict().get(key)
    assert after.region_id == "selected"
    assert after.source_labels == ["organ"]
    assert not after.region_geometry["constrained"]


def test_explicit_bounds_then_six_face_extension_are_applied_once():
    _, localization = _fixture()
    extension = localization["slice_rules"]["横状位"]["MPR"]["face_extension_mm"]
    extension.update(top_mm=3.0, bottom_mm=2.0, left_mm=4.0, right_mm=1.0)
    request = Request("/dicom", "/output", slice_part=SlicePartEnum.肺部)
    plan = build_whole_range_plan_from_bbox(request, PlaneConfig(PlaneTypeEnum.横状位, 2, 3, ["肺窗"]), localization)
    original = AnatomicalFrame.from_dict(localization["anatomical_geometry"]["global_frame"])
    final = AnatomicalFrame.from_dict(plan.region_geometry["frame"])
    assert final.bounds.superior_inferior.minimum == pytest.approx(original.bounds.superior_inferior.minimum + 6 - 2)
    assert final.bounds.superior_inferior.maximum == pytest.approx(original.bounds.superior_inferior.maximum - 8 + 3)
    assert final.bounds.left_right.minimum == pytest.approx(original.bounds.left_right.minimum - 1)
    assert final.bounds.left_right.maximum == pytest.approx(original.bounds.left_right.maximum + 4)
    np.testing.assert_allclose(_vector(plan.target_start) @ _vector(final.superior_inferior_axis), final.bounds.superior_inferior.maximum)
    np.testing.assert_allclose(_vector(plan.target_end) @ _vector(final.superior_inferior_axis), final.bounds.superior_inferior.minimum)


def test_cervical_axial_uses_region_in_its_own_local_frame():
    _, localization = _fixture(local_frame=True)
    request = Request("/dicom", "/output", slice_part=SlicePartEnum.颈椎)
    for plane in (PlaneTypeEnum.横状位, PlaneTypeEnum.冠状位):
        plan = build_whole_range_plan_from_bbox(request, PlaneConfig(plane, 2, 3, ["骨窗"]), localization)
        region = localization["anatomical_regions"]["selected"]
        expected = region["local_frames"]["axial_mpr"] if plane is PlaneTypeEnum.横状位 else region["global_frame"]
        assert plan.region_geometry["frame"] == expected


def test_each_plane_selects_its_named_region_without_part_whitelist():
    _, localization = _fixture()
    second = deepcopy(localization["anatomical_regions"]["selected"])
    second["global_frame"]["bounds"]["superior_inferior"]["minimum"] += 5
    localization["anatomical_regions"]["second"] = second
    localization["resolved_cut_configs"]["whole_range"]["regions"]["second"] = deepcopy(
        localization["resolved_cut_configs"]["whole_range"]["regions"]["selected"]
    )
    localization["slice_rules"]["冠状位"]["MPR"]["regions"] = ["second"]
    request = Request("/dicom", "/output", slice_part=SlicePartEnum.肺部)
    request.slice_part = SimpleNamespace(value="配置中的新目标")
    for plane, expected in ((PlaneTypeEnum.横状位, "selected"), (PlaneTypeEnum.冠状位, "second")):
        request.plane_configs = [PlaneConfig(plane, 2, 3, ["骨窗"])]
        plan = select_planner(request.slice_part.value, "MPR").build_plans(request, localization).plans[0]
        assert plan.region_id == expected
        assert plan.target_id == "配置中的新目标"
        assert plan.region_geometry["frame"] == localization["anatomical_regions"][expected]["global_frame"]


@pytest.mark.parametrize("plane", list(PlaneTypeEnum))
@pytest.mark.parametrize("percent", [None, 0, 50, 100])
def test_framing_keeps_explicit_range_instead_of_reloading_whole_label(plane, percent):
    image, localization = _fixture()
    localization["slice_rules"][plane.value]["MPR"]["framing_policy"] = {
        "basis": "selected_region", "margin_mm": 0.0, "centering": "group", "canvas": "original",
    }
    request = Request("/dicom", "/output", slice_part=SlicePartEnum.肺部)
    config = PlaneConfig(plane, 2, 3, ["肺窗"], fill_percent=percent)
    plan = build_whole_range_plan_from_bbox(request, config, localization).to_dict()
    assert not uses_source_preserving_canvas(plan)
    framer = TargetFramer(image, localization["coords"], plan, source_image=sitk.Cast(image, sitk.sitkFloat32))
    frame = AnatomicalFrame.from_dict(plan["region_geometry"]["frame"])
    bounds = (frame.bounds.left_right, frame.bounds.anterior_posterior, frame.bounds.superior_inferior)
    axes = np.column_stack([_vector(axis) for axis in (frame.left_right_axis, frame.anterior_posterior_axis, frame.superior_inferior_axis)])
    corners = np.asarray(list(product(*[(bound.minimum, bound.maximum) for bound in bounds]))) @ axes.T
    for item in plan["slices"]:
        plane_axes = np.column_stack([_vector(item[key]) for key in ("row_dir", "col_dir", "normal")])
        projected = corners @ plane_axes[:, :2]
        lower, upper = projected.min(axis=0), projected.max(axis=0)
        np.testing.assert_allclose(framer.plane_target_bounds(item), (lower, upper))
        _, framed = framer.plane_geometry(plan, item)
        np.testing.assert_allclose(_vector(framed["center"]) @ plane_axes[:, :2], (lower + upper) / 2)
        assert "source_preservation" not in framed["framing"]


@pytest.mark.parametrize("centering", ["group", "reference_slice", "slice_target", None])
@pytest.mark.parametrize("canvas", ["original", "target"])
@pytest.mark.parametrize("percent", [50, 100])
def test_explicit_region_keeps_one_scale_when_slice_target_center_moves(centering, canvas, percent):
    image, localization = _fixture()
    # 两段结构在左右方向偏心程度不同，逐层居中会产生不同的区域保护尺寸。
    array = sitk.GetArrayFromImage(image)
    array.fill(0)
    array[2:15, 9:22, 8:13] = 1
    array[15:30, 9:22, 12:17] = 1
    segmentation = sitk.GetImageFromArray(array)
    segmentation.CopyInformation(image)
    if centering is not None:
        localization["slice_rules"]["横状位"]["MPR"]["framing_policy"] = {
            "basis": "selected_region", "margin_mm": 0.0,
            "centering": centering, "canvas": canvas,
        }
    request = Request("/dicom", "/output", slice_part=SlicePartEnum.肺部)
    config = PlaneConfig(PlaneTypeEnum.横状位, 2, 3, ["肺窗"], fill_percent=percent)
    plan = build_whole_range_plan_from_bbox(request, config, localization).to_dict()
    framer = TargetFramer(segmentation, localization["coords"], plan)
    framed_slices = [framer.plane_geometry(plan, item)[1] for item in plan["slices"]]
    first = framed_slices[0]["framing"]
    for item, framed in zip(plan["slices"], framed_slices):
        details = framed["framing"]
        assert details["scale"] == first["scale"]
        assert details["framing_target_size_mm"] == first["framing_target_size_mm"]
        assert details["output_spacing_mm"] == first["output_spacing_mm"]
        assert details["output_size_xy"] == first["output_size_xy"]
        lower, upper = framer.plane_target_bounds(item)
        axes = np.column_stack([_vector(item[key]) for key in ("row_dir", "col_dir")])
        center = _vector(framed["center"]) @ axes
        field_of_view = np.array(details["output_size_xy"]) * np.array([
            details["output_spacing_mm"][key] for key in ("x", "y")
        ])
        assert np.all(field_of_view + 1e-7 >= 2 * np.maximum(upper - center, center - lower))
    if centering in {"slice_target", None}:
        # 必须保留逐层中心变化，不能靠强制改成整组居中来固定倍率。
        axes = np.column_stack([_vector(plan["slices"][0][key]) for key in ("row_dir", "col_dir")])
        centers = np.array([_vector(item["center"]) @ axes for item in framed_slices])
        assert np.max(np.ptp(centers, axis=0)) > 1.0


def test_cpr_interval_is_independent_of_mpr_target_and_boundary_offsets():
    result = {
        "resolved_cut_configs": {
            "whole_range": {"regions": {"short": {"target_labels": ["vertebrae_C5"], "bounds": {"upper": {"offset_mm": 30}}}}},
            "spine_range": {"start_labels": ["vertebrae_C1"], "end_labels": ["vertebrae_T1"]},
        },
        "spine_cpr_config": {"full_label": True, "excluded_labels": ["vertebrae_C1"]},
    }
    assert configured_whole_range_spine_labels(result) == ["vertebrae_C5"]
    assert configured_spine_cpr_labels(result) == [*[f"vertebrae_C{i}" for i in range(2, 8)], "vertebrae_T1"]


def test_configured_region_requires_stage5_solution():
    _, result = _fixture()
    result.pop("anatomical_regions")
    with pytest.raises(ValueError, match="第五阶段"):
        build_whole_range_plan_from_bbox(
            Request("/dicom", "/output", slice_part=SlicePartEnum.肺部),
            PlaneConfig(PlaneTypeEnum.横状位, 2, 3, ["肺窗"]), result,
        )
