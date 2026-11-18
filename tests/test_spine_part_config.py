from copy import deepcopy
import unittest

from src.config import (
    capability_entries,
    load_slice_part_config,
    localization_for,
    localization_profiles_for,
    pose_for,
    slice_rules_for,
    validate_slice_part_config,
)
from src.models import Request, SlicePartEnum
from src.stages.stage_segmentation import SegmentationStage


class SpinePartConfigTest(unittest.TestCase):
    """验证胸腰椎和全脊柱从外部请求到模型、定位和能力查询的配置接入。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_slice_part_config()
        cls.stage = SegmentationStage({"slice_part_config": cls.config})

    def test_spine_request_accepts_and_preserves_all_five_combinations(self):
        combinations = (
            ("横状位", "MPR"),
            ("冠状位", "MPR"),
            ("矢状位", "MPR"),
            ("矢状位", "CPR"),
            ("冠状位", "CPR"),
        )
        base_payload = {
            "dicom_dir": "/dicom",
            "output_dir": "/output",
            "StudyType": "CT",
            "ScanThickness": 1.0,
            "plane_configs": [
                {
                    "Plane": plane,
                    "SliceType": slice_type,
                    "SliceThickness": 1.0,
                    "SliceSpacing": 3.0,
                    "WindowPresets": ["骨窗"],
                }
                for plane, slice_type in combinations
            ],
        }

        for part in (SlicePartEnum.胸腰椎, SlicePartEnum.全脊柱):
            with self.subTest(slice_part=part.value):
                payload = {**base_payload, "SlicePart": part.value}
                request = Request.from_dict(payload)

                self.assertIs(request.slice_part, part)
                self.assertEqual(request.validate(), (True, None))
                self.assertEqual(request.to_dict(), payload)

    def test_spine_localization_separates_mpr_bounds_and_cpr_endpoints(self):
        for slice_part, start_label in (("胸腰椎", "C7"), ("全脊柱", "C2")):
            with self.subTest(slice_part=slice_part):
                localization = localization_for(self.config, slice_part)
                self.assertEqual(
                    localization_profiles_for(self.config, slice_part),
                    {
                        "whole_range": {"regions": {"default": localization["regions"]["default"]}},
                        "spine_range": {"region": "cpr_range", "start": start_label, "end": "S1"},
                    },
                )
                cpr = localization["spine_cpr"]
                self.assertTrue(cpr["full_label"])
                self.assertEqual(cpr.get("excluded_labels", []), [])

    def test_whole_spine_matches_thoracolumbar_config_except_start_label(self):
        expected = deepcopy(self.config["slice_parts"]["胸腰椎"])
        localization = expected["localization"]
        extra_labels = ["C2", "C3", "C4", "C5", "C6"]
        localization["pose"]["labels"][:0] = extra_labels
        localization["pose"]["roles"]["start"] = "C2"
        localization["regions"]["cpr_range"]["start_label"] = "C2"
        default = localization["regions"]["default"]
        default["target_labels"][:0] = extra_labels
        for bound in default["bounds"].values():
            bound["labels"][:0] = extra_labels

        self.assertEqual(self.config["slice_parts"]["全脊柱"], expected)

    def test_removed_spine_name_is_not_an_enum_or_config_alias(self):
        with self.assertRaises(ValueError):
            SlicePartEnum("脊柱")
        self.assertNotIn("脊柱", self.config["slice_parts"])
        with self.assertRaises(ValueError):
            Request.from_dict({
                "dicom_dir": "/dicom",
                "output_dir": "/output",
                "SlicePart": "脊柱",
                "ScanThickness": 1.0,
            })

    def test_cervical_default_stops_at_t1_with_axial_mpr_bottom_extension(self):
        """默认整段为 C1–T1，横状位 MPR 仅在底侧增加 5 mm。"""

        self.assertEqual(
            localization_profiles_for(self.config, "颈椎")["spine_range"],
            {"region": "cpr_range", "start": "C1", "end": "T1"},
        )
        localization = localization_for(self.config, "颈椎")
        self.assertEqual(localization["pose"]["roles"], {"start": "C1", "end": "T1"})
        self.assertEqual(localization["pose"]["fit_excluded_labels"], ["C1"])
        self.assertEqual(localization["spine_cpr"]["excluded_labels"], ["C1"])
        self.assertEqual(
            localization["regions"]["default"]["target_labels"],
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1"],
        )
        rules = slice_rules_for(self.config, "颈椎")
        self.assertEqual(
            rules["横状位"]["MPR"]["face_extension_mm"],
            {
                "top_mm": 0.0,
                "bottom_mm": 5.0,
                "front_mm": 0.0,
                "back_mm": 0.0,
                "left_mm": 0.0,
                "right_mm": 0.0,
            },
        )
        for plane in ("冠状位", "矢状位"):
            for rule in rules[plane].values():
                self.assertEqual(
                    rule.get("face_extension_mm", {}),
                    {
                        "top_mm": 0.0,
                        "bottom_mm": 0.0,
                        "front_mm": 0.0,
                        "back_mm": 0.0,
                        "left_mm": 0.0,
                        "right_mm": 0.0,
                    },
                )

    def test_spine_capabilities_include_exactly_the_requested_five_combinations(self):
        for slice_part in ("胸腰椎", "全脊柱"):
            with self.subTest(slice_part=slice_part):
                self.assertEqual(
                    {
                        (study_type, slice_type, plane)
                        for study_type, part, slice_type, plane in capability_entries(self.config)
                        if part == slice_part
                    },
                    {
                        ("CT", "MPR", "横状位"),
                        ("CT", "MPR", "冠状位"),
                        ("CT", "MPR", "矢状位"),
                        ("CT", "CPR", "冠状位"),
                        ("CT", "CPR", "矢状位"),
                    },
                )

    def test_spine_model_selection_respects_all_interval_boundaries(self):
        for slice_part in ("胸腰椎", "全脊柱"):
            for thickness, expected_model in (
                (0.0, 292),
                (1.5, 292),
                (1.6, 297),
                (4.5, 297),
                (4.6, 298),
                (6.0, 298),
            ):
                with self.subTest(slice_part=slice_part, thickness=thickness):
                    model = self.stage._match_model(slice_part, "CT", thickness)
                    self.assertEqual(model["model_id"], expected_model)

    def test_spine_model_selection_rejects_unconfigured_thicknesses(self):
        for slice_part in ("胸腰椎", "全脊柱"):
            for thickness in (1.55, 4.55, 6.1):
                with self.subTest(slice_part=slice_part, thickness=thickness):
                    with self.assertRaisesRegex(ValueError, "未找到匹配"):
                        self.stage._match_model(slice_part, "CT", thickness)

    def test_default_mpr_boundaries_preserve_full_target_extrema(self):
        """迁移不得把原有目标并集的极值收缩成首尾标签的极值。"""

        self.assertEqual(self.config["schema_version"], 1)
        for name, part in self.config["slice_parts"].items():
            if not part["enabled"]:
                continue
            with self.subTest(slice_part=name):
                region = part["localization"]["regions"]["default"]
                for side, reduction in (("upper", "max"), ("lower", "min")):
                    self.assertEqual(region["bounds"][side], {
                        "labels": region["target_labels"],
                        "edge": side,
                        "reduce": reduction,
                        "offset_mm": 0.0,
                    })

    def test_multiple_mpr_regions_are_preserved_without_changing_pose_or_cpr(self):
        config = deepcopy(self.config)
        part = config["slice_parts"]["颈椎"]
        original_pose = pose_for(config, "颈椎")
        original_cpr = localization_profiles_for(config, "颈椎")["spine_range"]
        part["localization"]["regions"]["lower_cervical"] = {
            "type": "whole_range",
            "target_labels": ["C5", "C6", "C7"],
            "bounds": {
                "upper": {"labels": ["C5"], "edge": "upper", "offset_mm": 2.0},
                "lower": {"labels": ["C7"], "edge": "lower", "offset_mm": -3.0},
            },
        }
        part["slice_rules"]["冠状位"]["MPR"]["regions"] = ["lower_cervical"]
        validate_slice_part_config(config)
        profiles = localization_profiles_for(config, "颈椎")
        self.assertEqual(set(profiles["whole_range"]["regions"]), {"default", "lower_cervical"})
        self.assertEqual(profiles["spine_range"], original_cpr)
        self.assertEqual(pose_for(config, "颈椎"), original_pose)
        profiles["whole_range"]["regions"]["default"]["target_labels"].clear()
        original_pose["labels"].clear()
        self.assertTrue(part["localization"]["regions"]["default"]["target_labels"])
        self.assertTrue(part["localization"]["pose"]["labels"])

    def test_multi_label_boundary_requires_explicit_reduction_after_alias_expansion(self):
        config = deepcopy(self.config)
        config["label_aliases"]["lung"] = ["lung_left", "lung_right"]
        boundary = config["slice_parts"]["肺部"]["localization"]["regions"]["default"]["bounds"]["upper"]
        boundary.pop("reduce")
        with self.assertRaisesRegex(ValueError, "多个实际标签.*reduce"):
            validate_slice_part_config(config)
        boundary["reduce"] = "max"
        validate_slice_part_config(config)

    def test_region_boundaries_reject_nonfinite_offsets_and_unknown_fields(self):
        for value in (float("nan"), float("inf"), True, "30"):
            with self.subTest(offset=value):
                config = deepcopy(self.config)
                boundary = config["slice_parts"]["肺部"]["localization"]["regions"]["default"]["bounds"]["upper"]
                boundary["offset_mm"] = value
                with self.assertRaisesRegex(ValueError, "offset_mm"):
                    validate_slice_part_config(config)
        config = deepcopy(self.config)
        config["slice_parts"]["肺部"]["localization"]["regions"]["default"]["start_label"] = "lung"
        with self.assertRaisesRegex(ValueError, "未知字段.*start_label"):
            validate_slice_part_config(config)

    def test_pose_exclusion_cannot_remove_all_labels_or_reference_other_targets(self):
        for excluded in (["lung"], ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1"]):
            with self.subTest(excluded=excluded):
                config = deepcopy(self.config)
                config["slice_parts"]["颈椎"]["localization"]["pose"]["fit_excluded_labels"] = excluded
                with self.assertRaisesRegex(ValueError, "fit_excluded_labels"):
                    validate_slice_part_config(config)

    def test_spine_roles_reject_aliases_that_expand_to_multiple_labels(self):
        for role, alias in (("start", "C1"), ("end", "T1")):
            with self.subTest(role=role):
                config = deepcopy(self.config)
                config["label_aliases"][alias] = [f"vertebrae_{alias}", f"extra_{alias}"]
                with self.assertRaisesRegex(ValueError, f"roles.{role}.*一个实际标签"):
                    validate_slice_part_config(config)

    def test_hip_roles_allow_aliases_that_group_labels_on_one_side(self):
        config = deepcopy(self.config)
        config["label_aliases"]["hip_left"] = ["hip_left_main", "hip_left_part"]
        config["label_aliases"]["hip_right"] = ["hip_right_main", "hip_right_part"]
        validate_slice_part_config(config)

    def test_cpr_cannot_reference_an_mpr_region(self):
        config = deepcopy(self.config)
        config["slice_parts"]["颈椎"]["slice_rules"]["冠状位"]["CPR"]["regions"] = ["default"]
        with self.assertRaisesRegex(ValueError, "必须是 spine_range"):
            validate_slice_part_config(config)


if __name__ == "__main__":
    unittest.main()
