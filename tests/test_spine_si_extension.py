from src.config import load_slice_part_config, localization_profiles_for, slice_rules_for
from src.models import (
    AnatomicalBounds, AnatomicalFrame, AnatomicalGeometryResult, AxisBounds,
    PlaneConfig, Point3D, Request,
)
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.base import select_planner
from src.planning.spine_cpr import SPINE_CONFIG_LABEL_ORDER


def _contract(part: str, superior=50.0, inferior=50.0, *, with_anatomy=False):
    cfg = load_slice_part_config()
    part_cfg = cfg["slice_parts"][part]
    profile = localization_profiles_for(cfg, part)["whole_range"]
    aliases = cfg["label_aliases"]

    def labels(name):
        value = aliases.get(name, name)
        return list(value) if isinstance(value, list) else [value]

    start = labels(profile["start"])[0]
    end = labels(profile["end"])[0]
    start_index = SPINE_CONFIG_LABEL_ORDER.index(start)
    end_index = SPINE_CONFIG_LABEL_ORDER.index(end)
    ordered = SPINE_CONFIG_LABEL_ORDER[start_index : end_index + 1]
    coords = {}
    for index, label in enumerate(ordered):
        # A simple vertical synthetic spine: superior C* at z=300, inferior S1 at z=100.
        z = 300.0 - index * (200.0 / max(1, len(ordered) - 1))
        coords[label] = {
            "label_name": label,
            "label_ids": [index + 1],
            "found": True,
            "centroid": {"x": 60.0, "y": 50.0, "z": z},
            "bbox": {"x_min": 55, "x_max": 65, "y_min": 45, "y_max": 55,
                     "z_min": z - 1, "z_max": z + 1},
        }

    # Planning only needs whole_range labels; preserve the same resolved contract as stage 3.
    resolved = {
        "whole_range": {
            "start_labels": labels(profile["start"]),
            "end_labels": labels(profile["end"]),
            "region": profile["region"],
        },
    }
    cpr = part_cfg["localization"]["spine_cpr"]
    result = {
        "slice_part": part,
        "localization_profile": "whole_range",
        "resolved_cut_configs": resolved,
        "slice_rules": slice_rules_for(cfg, part),
        "spine_cpr_config": {
            "full_label": cpr.get("full_label", True),
            "excluded_labels": cpr.get("excluded_labels", []),
            "height_extension_mm": {
                "top_mm": superior,
                "bottom_mm": inferior,
            },
        },
        # Deliberately omit size_xyz to exercise bbox fallback used by legacy/low-detail caches.
        "image": {
            "spacing_xyz": [1.0, 1.0, 1.0],
            "origin_xyz": [0.0, 0.0, 0.0],
            "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "correction_scope_bbox": {
            "x_min": 55, "x_max": 65, "y_min": 45, "y_max": 55,
            "z_min": 100, "z_max": 300,
        },
        "coords": coords,
    }
    if with_anatomy:
        frame = AnatomicalFrame(
            origin=Point3D(60.0, 50.0, 200.0),
            left_right_axis=Point3D(1.0, 0.0, 0.0),
            anterior_posterior_axis=Point3D(0.0, 1.0, 0.0),
            superior_inferior_axis=Point3D(0.0, 0.0, 1.0),
            bounds=AnatomicalBounds(
                left_right=AxisBounds(55.0, 65.0),
                anterior_posterior=AxisBounds(45.0, 55.0),
                superior_inferior=AxisBounds(100.0, 300.0),
            ),
            method="synthetic_test",
            confidence=1.0,
            applied=True,
        )
        result["anatomical_geometry"] = AnatomicalGeometryResult(
            global_frame=frame,
        ).to_dict()
    return result


def _plan(part, plane, slice_type, *, superior=50.0, inferior=50.0, with_anatomy=False):
    localization = _contract(part, superior=superior, inferior=inferior, with_anatomy=with_anatomy)
    request = Request(
        dicom_dir="/dicom",
        output_dir="/output",
        slice_part=SlicePartEnum(part),
        plane_configs=[PlaneConfig(plane, 1.0, 5.0, ["骨窗"], slice_type=slice_type)],
    )
    return select_planner(part, slice_type.value).build_plans(request, localization).plans[0]


def test_config_declares_50mm_superior_inferior_extension_for_both_spine_parts():
    cfg = load_slice_part_config()
    for part in ("胸腰椎", "全脊柱"):
        extension = cfg["slice_parts"][part]["localization"]["spine_cpr"]["height_extension_mm"]
        assert extension == {"top_mm": 50.0, "bottom_mm": 50.0}


def test_fallback_mpr_extends_si_coverage_in_all_three_planes_for_both_parts():
    # SI bounds of the selected labels are z=100..300; each MPR must cover 50mm beyond both.
    for part in ("胸腰椎", "全脊柱"):
        for plane in PlaneTypeEnum:
            plan = _plan(part, plane, SliceType.MPR)
            if plane is PlaneTypeEnum.横状位:
                z = [float(plan.range_start.z), float(plan.range_end.z)]
                assert min(z) <= 50.0 + 1e-6
                assert max(z) >= 350.0 - 1e-6
            else:
                assert plan.extent_mm.height >= 300.0 - 1e-6


def test_cpr_extends_curve_only_along_si_and_keeps_offset_sampling():
    for part in ("胸腰椎", "全脊柱"):
        for plane in (PlaneTypeEnum.冠状位, PlaneTypeEnum.矢状位):
            plan = _plan(part, plane, SliceType.CPR)
            baseline = _plan(part, plane, SliceType.CPR, superior=0.0, inferior=0.0)
            z = [float(point.z) for point in plan.curve_points]
            assert max(z) >= 350.0 - 1e-6
            assert min(z) <= 50.0 + 1e-6
            # The extension is along the vertical curve, not along CPR offset slabs.
            assert plan.offsets_mm == baseline.offsets_mm
            assert plan.extent_mm.height >= 300.0 - 1e-6


def test_anatomical_frame_mpr_does_not_double_apply_si_extension():
    # The coronal whole-range planner supplies a target bbox while sagittal uses the
    # correction bbox. Both must end at the same 200+50+50mm SI height.
    for part in ("胸腰椎", "全脊柱"):
        for plane in (PlaneTypeEnum.冠状位, PlaneTypeEnum.矢状位):
            plan = _plan(part, plane, SliceType.MPR, with_anatomy=True)
            assert 299.0 <= plan.extent_mm.height <= 303.0
