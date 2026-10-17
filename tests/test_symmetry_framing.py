"""可靠对称面居中与完整覆盖可以同时成立，不能改变真实切平面。"""

import copy
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.reconstruction.framing import TargetFramer, prepare_interspace_framing
from src.reconstruction.framing_context import FramingContext
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.symmetry_framing import (
    SymmetryFramingPlane,
    resolve_symmetry_framing_plane,
)
from src.stages.stage_reconstruction import ReconstructionStage
from src.stages.stage_reporting import ReportingStage


def _point(values):
    return dict(zip("xyz", map(float, values)))


def _vector(point):
    return np.array([point[key] for key in "xyz"], dtype=float)


def _case(*, oblique=False, flipped=False, context=False):
    labels = np.zeros((9, 64, 64), np.uint8)
    labels[2, 18:26, 14:35] = 1
    labels[2, 20:22, 35:41] = 1  # 单侧附件令 bbox 中心偏离已知主体对称面。
    labels[5, 30:38, 14:35] = 2
    labels[5, 32:34, 35:45] = 2
    if context:
        labels[2, 12:15, 46:51] = 3
    spacing = np.array((0.8, 1.3, 2.0))
    origin = np.array((-10.0, 3.0, -5.0))
    direction = np.eye(3)
    if oblique:
        yaw, pitch = np.deg2rad((21.0, 16.0))
        direction = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                              (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0))) @ np.array((
            (1.0, 0.0, 0.0), (0.0, np.cos(pitch), -np.sin(pitch)),
            (0.0, np.sin(pitch), np.cos(pitch)),
        ))
    segmentation = sitk.GetImageFromArray(labels)
    segmentation.SetSpacing(tuple(spacing))
    segmentation.SetDirection(tuple(direction.ravel()))
    segmentation.SetOrigin(tuple(origin))
    raw = sitk.GetImageFromArray(np.where(labels, 1500.0, -1000.0).astype(np.float32))
    raw.CopyInformation(segmentation)

    def physical(index):
        return origin + direction @ (np.asarray(index) * spacing)

    slices = [{
        "index": i + 1, "center": _point(physical((31.5, 31.5, z))),
        "row_dir": _point(direction[:, 0]), "col_dir": _point(direction[:, 1]),
        "normal": _point(direction[:, 2]),
    } for i, z in enumerate((2, 5))]
    plan = {
        "plan_id": "symmetry", "config_id": "配置01", "slice_part": "颈椎",
        "plane": "横状位", "slice_type": "椎间隙", "group_name": "C2_C3",
        "geometry_type": "plane_stack", "source_labels": ["C2", "C3"],
        "slice_thickness": 0.5, "slice_spacing": 6.0, "slice_image_count": 2,
        "window_presets": ["骨窗", "软组织窗"], "fill_percent": 100,
        "extent_mm": {"width": 63 * spacing[0], "height": 63 * spacing[1]},
        "output_spacing_mm": {"x": spacing[0], "y": spacing[1], "z": 0.0},
        "slices": slices, "range_start": slices[0]["center"],
        "range_end": slices[-1]["center"],
        "path_points": [slices[0]["center"], slices[-1]["center"]],
    }
    annotations = {
        label: {
            "symmetry_diagnostics": {"status": status},
            "symmetry_left_right_axis": _point(direction[:, 0] * sign),
            "symmetry_plane_center": _point(physical((24, y, z))),
        }
        for label, status, sign, y, z in (
            ("C2", "accepted", 1, 21.5, 2), ("C3", "unchanged", -1, 33.5, 5),
        )
    }
    analysis = {"spine_geometry": {"filtered_annotation_centroids": annotations}}
    coords = {"C2": {"label_ids": [1]}, "C3": {"label_ids": [2]}}
    if flipped:
        segmentation = sitk.Flip(segmentation, (True, False, True))
        raw = sitk.Flip(raw, (True, False, True))
    return segmentation, raw, coords, plan, analysis, physical, direction


@pytest.mark.parametrize("oblique,flipped", ((False, False), (True, False), (True, True)))
@pytest.mark.parametrize("fill", (80, 100))
def test_symmetry_center_preserves_plane_ap_and_complete_target_coverage(oblique, flipped, fill):
    seg, raw, coords, plan, analysis, physical, direction = _case(oblique=oblique, flipped=flipped)
    plan["fill_percent"] = fill
    original = copy.deepcopy(plan)
    plane = resolve_symmetry_framing_plane(plan, analysis)
    assert plane is not None
    framer = TargetFramer(seg, coords, plan, symmetry_plane=plane)
    spacings = []
    for item, z, y, max_x in zip(plan["slices"], (2, 5), (21.5, 33.5), (40.5, 44.5)):
        framed_plan, framed = framer.plane_geometry(plan, item)
        np.testing.assert_allclose(_vector(framed["center"]), physical((24, y, z)), atol=1e-10)
        for field in ("normal", "row_dir", "col_dir"):
            assert framed[field] == item[field]
        shift = _vector(framed["center"]) - _vector(item["center"])
        assert shift @ direction[:, 2] == pytest.approx(0.0, abs=1e-10)
        assert framed["framing"]["center_source"] == "slice_symmetry_plane"
        assert framed["framing"]["symmetry_centering"]["status"] == "applied"
        sampled = PlaneStackResampler().sample_slice(raw, framed_plan, framed)
        actual_center = sampled.TransformContinuousIndexToPhysicalPoint((31.5, 31.5, 0.0))
        np.testing.assert_allclose(actual_center, physical((24, y, z)), atol=1e-10)
        fov = np.array(sampled.GetSize()[:2]) * np.array(sampled.GetSpacing()[:2])
        for x in (13.5, max_x):
            # 独立检查目标外边界到真实输出中心的毫米距离，长侧也必须完整可见。
            assert abs((physical((x, y, z)) - actual_center) @ direction[:, 0]) <= fov[0] / 2 + 1e-9
        assert np.count_nonzero(sitk.GetArrayFromImage(sampled) > 0) > 20
        spacings.append(sampled.GetSpacing())
    assert spacings[0] == spacings[1]
    assert spacings[0][0] / spacings[0][1] == pytest.approx(0.8 / 1.3)
    assert plan == original


@pytest.mark.parametrize("fill", (None, 0))
def test_no_fill_keeps_existing_bbox_contract_even_with_reliable_plane(fill):
    seg, _, coords, plan, analysis, _, _ = _case()
    plan["fill_percent"] = fill
    plane = resolve_symmetry_framing_plane(plan, analysis)
    baseline = TargetFramer(seg, coords, plan)
    actual = TargetFramer(seg, coords, plan, symmetry_plane=plane)
    for item in plan["slices"]:
        assert actual.plane_geometry(plan, item) == baseline.plane_geometry(plan, item)


@pytest.mark.parametrize("defect", ("rejected", "missing_status", "missing_point", "invalid_point", "divergent_axes"))
def test_unreliable_or_incomplete_pair_does_not_use_ordinary_centroid_as_plane(defect):
    _, _, _, plan, analysis, _, _ = _case()
    annotation = analysis["spine_geometry"]["filtered_annotation_centroids"]["C3"]
    if defect == "rejected":
        annotation["symmetry_diagnostics"]["status"] = "rejected"
    elif defect == "missing_status":
        annotation.pop("symmetry_diagnostics")
    elif defect == "missing_point":
        annotation.pop("symmetry_plane_center")
        annotation["centroid_physical"] = _point((24, 20, 5))
    elif defect == "invalid_point":
        annotation["symmetry_plane_center"] = _point((float("nan"), 20, 5))
    else:
        annotation["symmetry_left_right_axis"] = _point((0, 1, 0))
    analysis["anatomical_geometry"] = {"global_frame": {
        "origin": _point((24, 20, 5)), "left_right_axis": _point((1, 0, 0)),
        "method": "spine_reflection_symmetry", "applied": True,
    }}
    assert resolve_symmetry_framing_plane(plan, analysis) is None


@pytest.mark.parametrize("kind", ("outside", "out_of_plane", "invalid"))
def test_unusable_plane_falls_back_without_blank_target(kind):
    seg, _, coords, plan, _, _, _ = _case()
    plane = {
        "outside": SymmetryFramingPlane((1, 0, 0), (1000, 0, 0)),
        "out_of_plane": SymmetryFramingPlane((0, 0, 1), (0, 0, 0)),
        "invalid": SymmetryFramingPlane((0, 0, 0), (0, 0, 0)),
    }[kind]
    baseline = TargetFramer(seg, coords, plan)
    actual = TargetFramer(seg, coords, plan, symmetry_plane=plane)
    for item in plan["slices"]:
        expected_plan, expected = baseline.plane_geometry(plan, item)
        actual_plan, framed = actual.plane_geometry(plan, item)
        assert actual_plan == expected_plan
        assert framed["center"] == expected["center"]
        assert framed["framing"]["center_source"] == "slice_target"
        assert framed["framing"]["symmetry_centering"]["status"] != "applied"


def test_empty_slab_keeps_existing_group_fallback_with_symmetry_evidence():
    seg, _, coords, plan, analysis, _, _ = _case()
    plane = resolve_symmetry_framing_plane(plan, analysis)
    item = copy.deepcopy(plan["slices"][0])
    item["center"]["z"] = 1000
    baseline = TargetFramer(seg, coords, plan).plane_geometry(plan, item)[1]
    actual = TargetFramer(seg, coords, plan, symmetry_plane=plane).plane_geometry(plan, item)[1]
    assert actual["center"] == baseline["center"]
    assert actual["framing"]["center_source"] == "group_fallback"


def test_shared_scale_includes_asymmetric_context_around_symmetry_center():
    seg, _, coords, plan, analysis, physical, direction = _case(context=True)
    plane = resolve_symmetry_framing_plane(plan, analysis)
    second = {**copy.deepcopy(plan), "plan_id": "other_gap", "group_name": "C3_C4"}
    context = FramingContext((1, 2, 3))
    shared = prepare_interspace_framing(
        seg, coords, [plan, second], context=context,
        symmetry_planes={p["plan_id"]: plane for p in (plan, second)},
    )
    for candidate in (plan, second):
        framer = TargetFramer(seg, coords, candidate, context=context,
                              symmetry_plane=plane, shared_geometry=shared)
        for item in candidate["slices"]:
            framed_plan, framed = framer.plane_geometry(candidate, item)
            assert tuple(framed_plan["output_spacing_mm"][axis] for axis in "xy") == shared.spacing_xy
            assert framed["framing"]["context_label_ids"] == [3]
            assert framed["framing"]["center_source"] == "slice_symmetry_plane"
            if item["index"] == 1:
                center = _vector(framed["center"])
                fov_x = shared.size_xy[0] * shared.spacing_xy[0]
                assert abs((physical((50.5, 13, 2)) - center) @ direction[:, 0]) <= fov_x / 2 + 1e-9


def test_reconstruction_writes_same_centered_geometry_to_both_windows(tmp_path: Path):
    seg, _, coords, plan, analysis, physical, direction = _case(oblique=True, flipped=True)
    raw_array = np.zeros(tuple(reversed(seg.GetSize())), np.float32)
    z, y, x = np.indices(raw_array.shape)
    xyz = np.stack((x, y, z), axis=-1)
    points = np.asarray(seg.GetOrigin()) + (xyz * seg.GetSpacing()) @ np.array(seg.GetDirection()).reshape(3, 3).T
    gradient = np.array((0.5, 1.0, 1.5))
    raw_array = np.asarray(20.0 + points @ gradient, dtype=np.float32)
    raw = sitk.GetImageFromArray(raw_array)
    raw.CopyInformation(seg)
    raw_path, seg_path = tmp_path / "raw.nii.gz", tmp_path / "seg.nii.gz"
    sitk.WriteImage(raw, str(raw_path))
    sitk.WriteImage(seg, str(seg_path))
    request = Request(str(tmp_path / "source"), str(tmp_path / "output"), slice_part=SlicePartEnum.颈椎,
                      plane_configs=[PlaneConfig(PlaneTypeEnum.横状位, 0.5, 6.0,
                                     ["骨窗", "软组织窗"], fill_percent=100,
                                     slice_type=SliceType.椎间隙, slice_image_count=2)])
    context = ProcessingContext(request)
    context.set_result("localization", {"raw_for_reconstruction_path": str(raw_path),
                       "seg_for_reconstruction_path": str(seg_path), "coords": coords})
    context.set_result("anatomical_analysis", analysis)
    context.set_result("planning", {"plans": [plan]})
    context.set_result("preprocessing", {"source_dicom_metadata": {"0020|000d": "1.2.3.4.5"}})
    ReconstructionStage({"parallel": False}).execute(context)
    ReportingStage().execute(context)
    assert context.get_result("reporting")["validation"]["valid"]
    output = context.get_result("reconstruction")["outputs"][0]
    geometries = {1: [], 2: []}
    for directory in output["window_outputs"].values():
        for index, z, y in ((1, 2, 21.5), (2, 5, 33.5)):
            ds = pydicom.dcmread(Path(directory) / f"{index}.dcm")
            row, col = np.array(ds.ImageOrientationPatient, float).reshape(2, 3)
            spacing = np.array(ds.PixelSpacing, float)
            center = (np.array(ds.ImagePositionPatient, float) + row*(ds.Columns-1)*spacing[1]/2
                      + col*(ds.Rows-1)*spacing[0]/2)
            np.testing.assert_allclose(center, physical((24, y, z)), atol=1e-5)
            np.testing.assert_allclose(np.r_[row, col], direction[:, :2].T.ravel(), atol=1e-8)
            r, c = ds.Rows//2, ds.Columns//2
            position = np.array(ds.ImagePositionPatient, float) + c*spacing[1]*row + r*spacing[0]*col
            hu = 20.0 + position @ gradient
            gray = np.rint(np.clip((hu-float(ds.WindowCenter)+float(ds.WindowWidth)/2)
                                  / float(ds.WindowWidth), 0, 1)*65535)
            assert abs(int(ds.pixel_array[r, c])-gray) <= 1
            geometries[index].append(np.r_[ds.ImagePositionPatient, ds.ImageOrientationPatient, ds.PixelSpacing])
    for pair in geometries.values():
        np.testing.assert_array_equal(pair[0], pair[1])
