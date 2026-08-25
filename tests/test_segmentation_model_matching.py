import json
import unittest
from pathlib import Path

from src.stages.stage_segmentation import SegmentationStage


PART_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "config"
    / "study_part_config.json"
)


class SegmentationModelMatchingTest(unittest.TestCase):
    """验证扫描层厚按配置闭区间选择 nnUNet 模型。"""

    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(PART_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.stage = SegmentationStage({"study_part_config": cls.config})

    def test_lumbar_boundaries_select_expected_models(self):
        cases = (
            (0.0, 292),
            (1.5, 292),
            (1.6, 297),
            (4.5, 297),
            (4.6, 298),
            (6.0, 298),
        )

        for thickness, expected_model in cases:
            with self.subTest(thickness=thickness):
                model = self.stage._match_model("腰椎", "CT", thickness)
                self.assertEqual(model["model_id"], expected_model)
                self.assertGreaterEqual(
                    thickness,
                    model["min_scan_thickness"],
                )
                self.assertLessEqual(
                    thickness,
                    model["max_scan_thickness"],
                )

    def test_unconfigured_thickness_gap_fails_explicitly(self):
        for thickness in (1.55, 4.55, 6.1):
            with self.subTest(thickness=thickness):
                with self.assertRaisesRegex(ValueError, "未找到匹配"):
                    self.stage._match_model("腰椎", "CT", thickness)

    def test_lung_only_supports_existing_thin_slice_model(self):
        self.assertEqual(
            self.stage._match_model("肺部", "CT", 1.5)["model_id"],
            315,
        )
        with self.assertRaisesRegex(ValueError, "未找到匹配"):
            self.stage._match_model("肺部", "CT", 1.6)

    def test_config_no_longer_contains_scan_thickness_categories(self):
        for part in self.config["study_parts"].values():
            for mappings in part["model_rules"].values():
                for mapper in mappings:
                    self.assertNotIn("scan_thickness", mapper)
                    self.assertIn("scan_thickness_range_mm", mapper)

    def test_overlapping_ranges_fail_instead_of_picking_first(self):
        mappings = [
            {
                "scan_thickness_range_mm": {"min": 0, "max": 2},
                "model_id": 1,
            },
            {
                "scan_thickness_range_mm": {"min": 1, "max": 3},
                "model_id": 2,
            },
        ]

        for thickness in (0.5, 1.5):
            with self.subTest(thickness=thickness):
                with self.assertRaisesRegex(ValueError, "区间重叠"):
                    SegmentationStage._match_model_mapping(mappings, thickness)

    def test_removed_category_mapper_is_rejected(self):
        mappings = [{"scan_thickness": "default", "model_id": 292}]

        with self.assertRaisesRegex(ValueError, "scan_thickness_range_mm"):
            SegmentationStage._match_model_mapping(mappings, 1.0)

    def test_invalid_request_thickness_is_rejected(self):
        for value in (-1, float("nan"), float("inf"), True, "default"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ScanThickness"):
                    self.stage._normalize_scan_thickness(value)


if __name__ == "__main__":
    unittest.main()
