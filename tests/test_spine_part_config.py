from copy import deepcopy
import unittest

from src.config import (
    capability_entries,
    load_slice_part_config,
    localization_for,
    localization_profiles_for,
    slice_rules_for,
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

    def test_spine_localization_uses_one_range_without_interspaces(self):
        for slice_part, start_label in (("胸腰椎", "C7"), ("全脊柱", "C2")):
            with self.subTest(slice_part=slice_part):
                self.assertEqual(
                    localization_profiles_for(self.config, slice_part),
                    {"whole_range": {
                        "region": "default", "start": start_label, "end": "S1",
                    }},
                )
                cpr = localization_for(self.config, slice_part)["spine_cpr"]
                self.assertTrue(cpr["full_label"])
                self.assertEqual(cpr.get("excluded_labels", []), [])

    def test_whole_spine_matches_thoracolumbar_config_except_start_label(self):
        expected = deepcopy(self.config["slice_parts"]["胸腰椎"])
        expected["localization"]["regions"]["default"]["start_label"] = "C2"

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
            localization_profiles_for(self.config, "颈椎")["whole_range"],
            {"region": "default", "start": "C1", "end": "T1"},
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


if __name__ == "__main__":
    unittest.main()
