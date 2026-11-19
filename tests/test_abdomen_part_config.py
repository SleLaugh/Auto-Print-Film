"""上、下、全腹部在协议、能力、模型选择与区域配置中的接入约束。"""

import unittest
from copy import deepcopy

from src.config import (
    capability_entries,
    load_slice_part_config,
    localization_for,
    localization_profiles_for,
    pose_for,
    slice_rules_for,
    validate_slice_part_config,
)
from src.core.capabilities import CapabilityRegistry
from src.core.errors import CapabilityError
from src.models import Request, SlicePartEnum
from src.stages.stage_segmentation import SegmentationStage


ABDOMEN_PARTS = (SlicePartEnum.上腹部, SlicePartEnum.下腹部, SlicePartEnum.全腹部)
PLANES = ("横状位", "冠状位", "矢状位")


def abdomen_request(part, *, slice_type="MPR", plane=None, study_type="CT"):
    """构造使用腹部窗的外部请求，支持能力拒绝场景。"""

    return Request.from_dict({
        "dicom_dir": "/dicom",
        "output_dir": "/output",
        "SlicePart": part.value,
        "StudyType": study_type,
        "ScanThickness": 1.0,
        "plane_configs": [
            {
                "Plane": item,
                "SliceType": slice_type,
                "SliceThickness": 1.0,
                "SliceSpacing": 3.0,
                "WindowPresets": ["腹部窗"],
            }
            for item in ((plane,) if plane else PLANES)
        ],
    })


class AbdomenPartConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_slice_part_config()
        cls.registry = CapabilityRegistry.current(cls.config)
        cls.stage = SegmentationStage({
            "slice_part_config": cls.config,
            "_configuration": "3d_lowres",
            "_fold": "9",
        })

    def test_request_parses_and_round_trips_all_abdomen_parts_with_three_mpr_planes(self):
        for part in ABDOMEN_PARTS:
            with self.subTest(part=part.value):
                request = abdomen_request(part)
                self.assertIs(request.slice_part, part)
                self.assertEqual(request.validate(), (True, None))
                self.assertEqual(request.to_dict()["SlicePart"], part.value)
                self.assertEqual(
                    Request.from_dict(request.to_dict()).to_dict(), request.to_dict()
                )
                self.registry.validate(request)

    def test_only_three_ct_mpr_capabilities_are_registered(self):
        for part in ABDOMEN_PARTS:
            with self.subTest(part=part.value):
                self.assertEqual(
                    {
                        (study, slice_type, plane)
                        for study, name, slice_type, plane in capability_entries(self.config)
                        if name == part.value
                    },
                    {("CT", "MPR", plane) for plane in PLANES},
                )

    def test_cpr_interspace_and_mr_requests_are_rejected(self):
        for part in ABDOMEN_PARTS:
            for slice_type in ("CPR", "椎间隙"):
                for plane in PLANES:
                    with self.subTest(part=part.value, slice_type=slice_type, plane=plane):
                        with self.assertRaises(CapabilityError):
                            self.registry.validate(
                                abdomen_request(part, slice_type=slice_type, plane=plane)
                            )
            with self.subTest(part=part.value, study_type="MR"):
                with self.assertRaises(CapabilityError):
                    self.registry.validate(abdomen_request(part, study_type="MR"))

    def test_model_701_covers_continuous_zero_to_six_mm_and_overrides_global_runtime(self):
        for part in ABDOMEN_PARTS:
            for thickness in (0.0, 1.5, 1.55, 1.6, 4.55, 6.0):
                with self.subTest(part=part.value, thickness=thickness):
                    model = self.stage._match_model(part.value, "CT", thickness)
                    self.assertEqual(model["model_id"], 701)
                    self.assertEqual(model["configuration"], "3d_fullres")
                    self.assertEqual(model["fold"], "0")
                    self.assertEqual(model["input_orientation"], "RAS")

    def test_model_selection_rejects_out_of_range_or_unconfigured_studies(self):
        for part in ABDOMEN_PARTS:
            for thickness in (6.01, 10.0):
                with self.subTest(part=part.value, thickness=thickness):
                    with self.assertRaisesRegex(ValueError, "未找到匹配"):
                        self.stage._match_model(part.value, "CT", thickness)
            with self.subTest(part=part.value, study_type="MR"):
                with self.assertRaises(ValueError):
                    self.stage._match_model(part.value, "MR", 1.0)

    def test_input_orientation_rejects_invalid_types_and_unknown_values(self):
        for value in (None, [], {}, 0, True, "", "ras", "RAS ", "LAS", "canonical"):
            with self.subTest(value=value):
                config = deepcopy(self.config)
                rule = config["slice_parts"]["上腹部"]["model_rules"]["CT"][0]
                rule["input_orientation"] = value
                with self.assertRaisesRegex(ValueError, "input_orientation.*RAS/LPS"):
                    validate_slice_part_config(config)

    def test_both_supported_input_orientations_are_valid(self):
        for value in ("RAS", "LPS"):
            with self.subTest(value=value):
                config = deepcopy(self.config)
                rule = config["slice_parts"]["上腹部"]["model_rules"]["CT"][0]
                rule["input_orientation"] = value
                validate_slice_part_config(config)

    def test_existing_nine_parts_do_not_enable_input_reorientation(self):
        for part in ("颅脑", "颈椎", "胸椎", "腰椎", "胸腰椎", "全脊柱", "肺部", "骶骨", "髋骨"):
            with self.subTest(part=part):
                for rules in self.config["slice_parts"][part]["model_rules"].values():
                    for rule in rules:
                        self.assertNotIn("input_orientation", rule)

    def test_pose_uses_both_hips_independently_of_region_organ_labels(self):
        expected = {
            "method": "hip",
            "roles": {"left": "hip_left", "right": "hip_right"},
        }
        for part in ABDOMEN_PARTS:
            with self.subTest(part=part.value):
                self.assertEqual(pose_for(self.config, part.value), expected)
                profiles = localization_profiles_for(self.config, part.value)
                self.assertEqual(set(profiles), {"whole_range"})
                self.assertEqual(set(profiles["whole_range"]["regions"]), {"default"})

    def test_upper_abdomen_stops_at_higher_hip_upper_edge(self):
        region = localization_for(self.config, "上腹部")["regions"]["default"]
        self.assertEqual(region["type"], "whole_range")
        self.assertEqual(region["target_labels"], [
            "liver", "hip_left", "hip_right"
        ])
        self.assertEqual(region["bounds"], {
            "upper": {
                "labels": ["liver"], "edge": "upper", "reduce": "max", "offset_mm": 0.0,
            },
            "lower": {
                "labels": ["hip_left", "hip_right"],
                "edge": "upper", "reduce": "max", "offset_mm": 0.0,
            },
        })

    def test_lower_abdomen_adds_thirty_mm_to_higher_kidney_upper_edge(self):
        region = localization_for(self.config, "下腹部")["regions"]["default"]
        self.assertEqual(region["type"], "whole_range")
        self.assertEqual(region["target_labels"], [
            "kidney_left", "kidney_right", "hip_left", "hip_right", "femur_left", "femur_right"
        ])
        self.assertEqual(region["bounds"], {
            "upper": {
                "labels": ["kidney_left", "kidney_right"],
                "edge": "upper", "reduce": "max", "offset_mm": 30.0,
            },
            "lower": {
                "labels": ["femur_left", "femur_right"],
                "edge": "upper", "reduce": "max", "offset_mm": 0.0,
            },
        })

    def test_aliases_match_seven_structures_trained_in_701(self):
        aliases = self.config["label_aliases"]
        for label in (
            "liver", "kidney_left", "kidney_right", "hip_left", "hip_right",
            "femur_left", "femur_right",
        ):
            with self.subTest(label=label):
                self.assertEqual(aliases[label], label)

    def test_whole_abdomen_uses_liver_upper_edge_and_existing_femur_boundary(self):
        region = localization_for(self.config, "全腹部")["regions"]["default"]
        self.assertEqual(region["target_labels"], [
            "liver", "hip_left", "hip_right", "femur_left", "femur_right",
        ])
        self.assertEqual(region["bounds"], {
            "upper": {
                "labels": ["liver"], "edge": "upper", "reduce": "max", "offset_mm": 0.0,
            },
            "lower": {
                "labels": ["femur_left", "femur_right"],
                "edge": "upper", "reduce": "max", "offset_mm": 0.0,
            },
        })

    def test_all_planes_use_same_region_without_implicit_face_extensions(self):
        zero_extension = {
            "top_mm": 0.0, "bottom_mm": 0.0, "front_mm": 0.0,
            "back_mm": 0.0, "left_mm": 0.0, "right_mm": 0.0,
        }
        for part in ABDOMEN_PARTS:
            rules = slice_rules_for(self.config, part.value)
            for plane in PLANES:
                with self.subTest(part=part.value, plane=plane):
                    self.assertEqual(set(rules[plane]), {"MPR"})
                    rule = rules[plane]["MPR"]
                    self.assertEqual(rule["regions"], ["default"])
                    self.assertEqual(rule["face_extension_mm"], zero_extension)
                    self.assertEqual(
                        rule["localizer_plane"], "冠状位" if plane == "矢状位" else "矢状位"
                    )

    def test_transverse_coverage_uses_body_and_fixed_group_framing(self):
        for part in ABDOMEN_PARTS:
            with self.subTest(part=part.value):
                region = localization_for(self.config, part.value)["regions"]["default"]
                self.assertEqual(region["transverse_source"], "body")
                for plane in PLANES:
                    rule = slice_rules_for(self.config, part.value)[plane]["MPR"]
                    self.assertEqual(rule["framing_policy"], {
                        "basis": "selected_region",
                        "margin_mm": 0.0,
                        "centering": "group",
                        "canvas": "target",
                    })

    def test_transverse_source_rejects_invalid_types_and_unknown_values(self):
        for value in (None, [], {}, 0, True, "", "image", "Body"):
            with self.subTest(value=value):
                config = deepcopy(self.config)
                region = config["slice_parts"]["上腹部"]["localization"]["regions"]["default"]
                region["transverse_source"] = value
                with self.assertRaisesRegex(ValueError, "transverse_source.*labels/body"):
                    validate_slice_part_config(config)

    def test_existing_parts_keep_implicit_label_based_transverse_coverage(self):
        for part in ("颅脑", "颈椎", "胸椎", "腰椎", "胸腰椎", "全脊柱", "肺部", "髋骨"):
            with self.subTest(part=part):
                region = localization_for(self.config, part)["regions"]["default"]
                self.assertNotIn("transverse_source", region)
                self.assertEqual(region.get("transverse_source", "labels"), "labels")

    def test_explicit_label_based_transverse_source_remains_valid(self):
        config = deepcopy(self.config)
        config["slice_parts"]["肺部"]["localization"]["regions"]["default"][
            "transverse_source"
        ] = "labels"
        validate_slice_part_config(config)


if __name__ == "__main__":
    unittest.main()
