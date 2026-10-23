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
    """验证独立脊柱部位从外部请求到模型、定位和能力查询的配置接入。"""

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
        payload = {
            "dicom_dir": "/dicom",
            "output_dir": "/output",
            "SlicePart": "脊柱",
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

        request = Request.from_dict(payload)

        self.assertIs(request.slice_part, SlicePartEnum.脊柱)
        self.assertEqual(request.validate(), (True, None))
        self.assertEqual(request.to_dict(), payload)

    def test_spine_localization_uses_one_c7_s1_range_without_interspaces(self):
        self.assertEqual(
            localization_profiles_for(self.config, "脊柱"),
            {"whole_range": {"region": "default", "start": "C7", "end": "S1"}},
        )
        cpr = localization_for(self.config, "脊柱")["spine_cpr"]
        self.assertTrue(cpr["full_label"])
        self.assertEqual(cpr.get("excluded_labels", []), [])

    def test_cervical_default_stops_at_t1_with_axial_mpr_end_extension(self):
        """默认整段为 C1–T1，横状位 MPR 仅在末端增加 5 mm。"""

        self.assertEqual(
            localization_profiles_for(self.config, "颈椎")["whole_range"],
            {"region": "default", "start": "C1", "end": "T1"},
        )
        rules = slice_rules_for(self.config, "颈椎")
        self.assertEqual(
            rules["横状位"]["MPR"]["range_extension"],
            {"start_mm": 0.0, "end_mm": 5.0},
        )
        for plane in ("冠状位", "矢状位"):
            for rule in rules[plane].values():
                self.assertEqual(
                    rule.get("range_extension", {}),
                    {"start_mm": 0.0, "end_mm": 0.0},
                )

    def test_spine_capabilities_include_exactly_the_requested_five_combinations(self):
        self.assertEqual(
            {
                (study_type, slice_type, plane)
                for study_type, part, slice_type, plane in capability_entries(self.config)
                if part == "脊柱"
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
        for thickness, expected_model in (
            (0.0, 292),
            (1.5, 292),
            (1.6, 297),
            (4.5, 297),
            (4.6, 298),
            (6.0, 298),
        ):
            with self.subTest(thickness=thickness):
                model = self.stage._match_model("脊柱", "CT", thickness)
                self.assertEqual(model["model_id"], expected_model)

    def test_spine_model_selection_rejects_unconfigured_thicknesses(self):
        for thickness in (1.55, 4.55, 6.1):
            with self.subTest(thickness=thickness):
                with self.assertRaisesRegex(ValueError, "未找到匹配"):
                    self.stage._match_model("脊柱", "CT", thickness)


if __name__ == "__main__":
    unittest.main()
