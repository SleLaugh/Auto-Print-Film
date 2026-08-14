import unittest

from src.models import PlaneConfig, Request
from src.models.enums import (
    BodyPartEnum,
    CutLineExtentModeEnum,
    PlaneTypeEnum,
)


class PlaneConfigRequestTest(unittest.TestCase):
    """汇总 `PlaneConfigRequestTest` 相关测试场景。"""

    def test_plane_config_supports_all_window_preset_names(self):
        # 外部只传中文预设名称，解析和序列化都不展开内部窗宽窗位数值。
        """验证 `test_plane_config_supports_all_window_preset_names` 场景符合预期。"""

        window_names = [
            "脑窗",
            "骨窗",
            "肺窗",
            "纵隔窗",
            "软组织窗",
            "腹部窗",
            "肝窗",
            "肾窗",
            "胰腺窗",
            "骨盆窗",
            "脊柱窗",
        ]
        config = PlaneConfig.from_dict({
            "plane": "横状位",
            "slice_thickness": 5,
            "slice_spacing": 5,
            "window_presets": window_names,
        })

        self.assertEqual(
            config.window_presets,
            window_names,
        )
        self.assertEqual(config.to_dict()["window_presets"], window_names)

    def test_plane_config_from_dict_keeps_window_names_as_strings(self):
        """验证 `test_plane_config_from_dict_keeps_window_names_as_strings` 场景符合预期。"""

        config = PlaneConfig.from_dict({
            "plane": "冠状位",
            "slice_thickness": 5,
            "slice_spacing": 10,
            "window_presets": ["骨窗", "肺窗", "纵隔窗"],
        })

        self.assertIs(config.plane, PlaneTypeEnum.冠状位)
        self.assertEqual(config.slice_thickness, 5.0)
        self.assertEqual(config.slice_spacing, 10.0)
        self.assertIs(
            config.line_extent_mode,
            CutLineExtentModeEnum.整平面,
        )
        self.assertEqual(config.line_extent_mode.value, "整平面")
        self.assertEqual(config.window_presets, ["骨窗", "肺窗", "纵隔窗"])

    def test_plane_config_parses_chinese_target_line_extent_mode(self):
        """外部请求应直接使用中文部位范围协议值。"""

        config = PlaneConfig.from_dict({
            "plane": "横状位",
            "slice_thickness": 3,
            "slice_spacing": 5,
            "window_presets": ["骨窗"],
            "line_extent_mode": "部位范围",
        })

        self.assertIs(
            config.line_extent_mode,
            CutLineExtentModeEnum.部位范围,
        )
        self.assertEqual(config.line_extent_mode.value, "部位范围")

    def test_plane_config_rejects_removed_english_line_extent_mode(self):
        """一次性迁移后不再接受旧英文范围协议值。"""

        for removed_value in ("full_plane", "target_bbox"):
            with self.subTest(removed_value=removed_value):
                with self.assertRaises(ValueError):
                    PlaneConfig.from_dict({
                        "plane": "横状位",
                        "slice_thickness": 3,
                        "slice_spacing": 5,
                        "window_presets": ["骨窗"],
                        "line_extent_mode": removed_value,
                    })

    def test_request_from_dict_parses_plane_configs_and_to_dict_serializes_values(self):
        # from_dict 负责把外部请求转成强约束对象；to_dict 再转回报告可写的结构。
        """验证 `test_request_from_dict_parses_plane_configs_and_to_dict_serializes_values` 场景符合预期。"""

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
                "line_extent_mode": "整平面",
            }
        ])
        self.assertNotIn("cut_mode", request.to_dict())
        self.assertFalse(hasattr(request, "cut_mode"))


    def test_request_defaults_to_currently_supported_axial_plane(self):
        # 默认请求只能声明当前框架实际支持的能力，其他状位必须显式请求并预检。
        """验证 `test_request_defaults_to_currently_supported_axial_plane` 场景符合预期。"""

        request = Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")

        self.assertEqual(request.to_dict()["plane_configs"], [
            {
                "plane": "横状位",
                "slice_thickness": 1.0,
                "slice_spacing": 1.0,
                "window_presets": ["骨窗"],
                "line_extent_mode": "整平面",
            },
        ])

    def test_request_validate_rejects_empty_plane_configs(self):
        """验证 `test_request_validate_rejects_empty_plane_configs` 场景符合预期。"""

        self._assert_invalid([], "plane_configs 不能为空")

    def test_request_validate_rejects_duplicate_planes(self):
        """验证 `test_request_validate_rejects_duplicate_planes` 场景符合预期。"""

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
        """验证 `test_request_validate_rejects_non_positive_slice_thickness` 场景符合预期。"""

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
        """验证 `test_request_validate_rejects_non_positive_slice_spacing` 场景符合预期。"""

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
        """验证 `test_request_validate_rejects_empty_window_presets` 场景符合预期。"""

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

    def test_request_validate_rejects_unknown_window_preset(self):
        """验证 `test_request_validate_rejects_unknown_window_preset` 场景符合预期。"""

        self._assert_invalid(
            [{
                "plane": "横状位",
                "slice_thickness": 1,
                "slice_spacing": 1,
                "window_presets": ["血管窗"],
            }],
            "不支持的窗宽窗位预设",
        )

    def _assert_invalid(self, plane_configs, expected_message):
        # 统一构造非法请求，避免每个校验用例重复写路径和断言。
        """为测试准备 `_assert_invalid` 所需的数据或记录行为。"""

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
