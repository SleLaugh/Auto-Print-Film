import unittest

from src.models import PlaneConfig, Request, WindowPresetEnum
from src.models.enums import BodyPartEnum, PlaneTypeEnum


class PlaneConfigRequestTest(unittest.TestCase):
    def test_plane_config_from_dict_converts_strings_to_enums(self):
        config = PlaneConfig.from_dict({
            "plane": "冠状位",
            "slice_thickness": 5,
            "slice_spacing": 10,
            "window_presets": ["骨窗", "肺窗", "纵隔窗"],
        })

        self.assertIs(config.plane, PlaneTypeEnum.冠状位)
        self.assertEqual(config.slice_thickness, 5.0)
        self.assertEqual(config.slice_spacing, 10.0)
        self.assertEqual(config.window_presets, [
            WindowPresetEnum.骨窗,
            WindowPresetEnum.肺窗,
            WindowPresetEnum.纵隔窗,
        ])

    def test_request_from_dict_parses_plane_configs_and_to_dict_serializes_values(self):
        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "body_part": BodyPartEnum.腰椎.value,
            "plane_configs": [
                {
                    "plane": "冠状位",
                    "slice_thickness": 5,
                    "slice_spacing": 10,
                    "window_presets": ["骨窗", "肺窗"],
                }
            ],
        })

        self.assertIs(request.plane_configs[0].plane, PlaneTypeEnum.冠状位)
        self.assertEqual(request.to_dict()["plane_configs"], [
            {
                "plane": "冠状位",
                "slice_thickness": 5.0,
                "slice_spacing": 10.0,
                "window_presets": ["骨窗", "肺窗"],
            }
        ])

    def test_request_uses_three_default_bone_window_plane_configs(self):
        request = Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")

        self.assertEqual(request.to_dict()["plane_configs"], [
            {
                "plane": "横状位",
                "slice_thickness": 1.0,
                "slice_spacing": 1.0,
                "window_presets": ["骨窗"],
            },
            {
                "plane": "矢状位",
                "slice_thickness": 1.0,
                "slice_spacing": 1.0,
                "window_presets": ["骨窗"],
            },
            {
                "plane": "冠状位",
                "slice_thickness": 1.0,
                "slice_spacing": 1.0,
                "window_presets": ["骨窗"],
            },
        ])

    def test_request_validate_rejects_empty_plane_configs(self):
        self._assert_invalid([], "plane_configs 不能为空")

    def test_request_validate_rejects_duplicate_planes(self):
        self._assert_invalid(
            [
                {
                    "plane": "横状位",
                    "slice_thickness": 1,
                    "slice_spacing": 1,
                    "window_presets": ["骨窗"],
                },
                {
                    "plane": "横状位",
                    "slice_thickness": 1,
                    "slice_spacing": 1,
                    "window_presets": ["骨窗"],
                },
            ],
            "plane_configs 中存在重复状位",
        )

    def test_request_validate_rejects_non_positive_slice_thickness(self):
        self._assert_invalid(
            [
                {
                    "plane": "横状位",
                    "slice_thickness": 0,
                    "slice_spacing": 1,
                    "window_presets": ["骨窗"],
                }
            ],
            "层厚必须大于 0",
        )

    def test_request_validate_rejects_non_positive_slice_spacing(self):
        self._assert_invalid(
            [
                {
                    "plane": "横状位",
                    "slice_thickness": 1,
                    "slice_spacing": 0,
                    "window_presets": ["骨窗"],
                }
            ],
            "层间距必须大于 0",
        )

    def test_request_validate_rejects_empty_window_presets(self):
        self._assert_invalid(
            [
                {
                    "plane": "横状位",
                    "slice_thickness": 1,
                    "slice_spacing": 1,
                    "window_presets": [],
                }
            ],
            "窗宽窗位不能为空",
        )

    def _assert_invalid(self, plane_configs, expected_message):
        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "plane_configs": plane_configs,
        })

        is_valid, error_message = request.validate()

        self.assertFalse(is_valid)
        self.assertIn(expected_message, error_message)


if __name__ == "__main__":
    unittest.main()
