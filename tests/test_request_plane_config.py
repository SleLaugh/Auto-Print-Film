import unittest

from src.models import PlaneConfig, Request
from src.models.enums import (
    SlicePartEnum,
    PlaneTypeEnum,
    SliceType,
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
            "Plane": "横状位",
            "SliceThickness": 5,
            "SliceSpacing": 5,
            "WindowPresets": window_names,
        })

        self.assertEqual(
            config.window_presets,
            window_names,
        )
        self.assertEqual(config.to_dict()["WindowPresets"], window_names)

    def test_plane_config_from_dict_keeps_window_names_as_strings(self):
        """验证 `test_plane_config_from_dict_keeps_window_names_as_strings` 场景符合预期。"""

        config = PlaneConfig.from_dict({
            "Plane": "冠状位",
            "SliceThickness": 5,
            "SliceSpacing": 10,
            "WindowPresets": ["骨窗", "肺窗", "纵隔窗"],
        })

        self.assertIs(config.plane, PlaneTypeEnum.冠状位)
        self.assertEqual(config.slice_thickness, 5.0)
        self.assertEqual(config.slice_spacing, 10.0)
        self.assertIsNone(config.fill_percent)
        self.assertNotIn("FillPercent", config.to_dict())
        self.assertEqual(config.window_presets, ["骨窗", "肺窗", "纵隔窗"])
        self.assertIs(config.slice_type, SliceType.MPR)

    def test_slice_type_enum_order_matches_external_protocol(self):
        """模板界面按 MPR、CPR、椎间隙的稳定顺序展示。"""

        self.assertEqual(
            [slice_type.value for slice_type in SliceType],
            ["MPR", "CPR", "椎间隙"],
        )

    def test_plane_config_parses_explicit_cpr_slice_type(self):
        """外部 SliceType 字段应转换成强类型枚举并原样返回。"""

        config = PlaneConfig.from_dict({
            "Plane": "冠状位",
            "SliceThickness": 3,
            "SliceSpacing": 5,
            "WindowPresets": ["骨窗"],
            "SliceType": "CPR",
        })

        self.assertIs(config.slice_type, SliceType.CPR)
        self.assertEqual(config.to_dict()["SliceType"], "CPR")

    def test_plane_config_rejects_removed_slice_type_values(self):
        """旧切法值不再作为兼容入口。"""

        for removed_value in ("whole_range", "interspace", "cpr", "整切"):
            with self.subTest(removed_value=removed_value):
                with self.assertRaises(ValueError):
                    PlaneConfig.from_dict({
                        "Plane": "横状位",
                        "SliceThickness": 3,
                        "SliceSpacing": 5,
                        "WindowPresets": ["骨窗"],
                        "SliceType": removed_value,
                    })

    def test_plane_config_parses_fill_percent(self):
        """百分比保持 JSON 数字，省略时不写入报告。"""

        config = PlaneConfig.from_dict({
            "Plane": "横状位",
            "SliceThickness": 3,
            "SliceSpacing": 5,
            "WindowPresets": ["骨窗"],
            "FillPercent": 80,
        })

        self.assertEqual(config.fill_percent, 80)
        self.assertEqual(config.to_dict()["FillPercent"], 80)

    def test_invalid_fill_percent_is_rejected(self):
        for value in (-1, 101, 10 ** 400, True, False, "0", "80", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "FillPercent"):
                    PlaneConfig.from_dict({"Plane": "横状位", "FillPercent": value})
                request = Request("/tmp/dicom", "/tmp/output")
                request.plane_configs[0].fill_percent = value
                self.assertFalse(request.validate()[0])

    def test_zero_fill_percent_is_preserved_and_valid(self):
        for value in (0, 0.0):
            with self.subTest(value=value):
                config = PlaneConfig.from_dict({"Plane": "横状位", "FillPercent": value})
                self.assertEqual(config.fill_percent, 0.0)
                self.assertEqual(config.to_dict()["FillPercent"], 0.0)
                request = Request("/tmp/dicom", "/tmp/output")
                request.plane_configs[0].fill_percent = value
                self.assertTrue(request.validate()[0])

    def test_plane_config_rejects_removed_line_extent_mode(self):
        """旧范围字段已移除，不再接受其中文或英文协议值。"""

        for removed_value in ("full_plane", "target_bbox", "整平面", "部位范围"):
            with self.subTest(removed_value=removed_value):
                with self.assertRaises(ValueError):
                    PlaneConfig.from_dict({
                        "Plane": "横状位",
                        "SliceThickness": 3,
                        "SliceSpacing": 5,
                        "WindowPresets": ["骨窗"],
                        "line_extent_mode": removed_value,
                    })

    def test_request_from_dict_parses_plane_configs_and_to_dict_serializes_values(self):
        # from_dict 负责把外部请求转成强约束对象；to_dict 再转回报告可写的结构。
        """验证 `test_request_from_dict_parses_plane_configs_and_to_dict_serializes_values` 场景符合预期。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "SlicePart": SlicePartEnum.腰椎.value,
            "StudyType": "CT",
            "ScanThickness": 1.5,
            "plane_configs": [
                {
                    "Plane": "冠状位",
                    "SliceThickness": 5,
                    "SliceSpacing": 10,
                    "WindowPresets": ["骨窗", "肺窗"],
                }
            ],
        })

        self.assertIs(request.plane_configs[0].plane, PlaneTypeEnum.冠状位)
        self.assertIs(request.slice_part, SlicePartEnum.腰椎)
        self.assertEqual(request.to_dict()["SlicePart"], "腰椎")
        self.assertEqual(request.to_dict()["StudyType"], "CT")
        self.assertEqual(request.scan_thickness, 1.5)
        self.assertEqual(request.to_dict()["ScanThickness"], 1.5)
        self.assertEqual(request.to_dict()["plane_configs"], [
            {
                "Plane": "冠状位",
                "SliceThickness": 5.0,
                "SliceSpacing": 10.0,
                "WindowPresets": ["骨窗", "肺窗"],
                "SliceType": "MPR",
            }
        ])
        self.assertNotIn("cut_mode", request.to_dict())
        self.assertFalse(hasattr(request, "cut_mode"))

    def test_request_uses_custom_window_presets_for_plane_config(self):
        """plane config 可引用自定义名称，未传入的名称仍回退默认配置。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.5,
            "window_presets": [
                {"Name": "骨窗1", "windowWidth": 50, "windowLevel": 60},
            ],
            "plane_configs": [
                {
                    "Plane": "横状位",
                    "SliceThickness": 1,
                    "SliceSpacing": 1,
                    "WindowPresets": ["骨窗1", "脑窗"],
                }
            ],
        })

        self.assertEqual(request.validate(), (True, None))
        self.assertEqual(request.to_dict()["window_presets"], [
            {"Name": "骨窗1", "windowWidth": 50, "windowLevel": 60},
        ])

    def test_request_rejects_window_name_missing_from_custom_and_default(self):
        """plane config 引用两处都不存在的窗位时请求校验失败。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.5,
            "window_presets": [
                {"Name": "骨窗1", "windowWidth": 50, "windowLevel": 60},
            ],
            "plane_configs": [
                {
                    "Plane": "横状位",
                    "SliceThickness": 1,
                    "SliceSpacing": 1,
                    "WindowPresets": ["骨窗2"],
                }
            ],
        })

        is_valid, message = request.validate()
        self.assertFalse(is_valid)
        self.assertIn("不支持的窗宽窗位预设: 骨窗2", message)

    def test_request_requires_pascal_case_scan_thickness(self):
        """外部扫描厚度只接受必填的 PascalCase 数值字段。"""

        with self.assertRaisesRegex(ValueError, "ScanThickness"):
            Request.from_dict({
                "dicom_dir": "/tmp/dicom",
                "output_dir": "/tmp/output",
                "scan_thickness": 1.0,
            })

    def test_request_rejects_removed_study_field_names(self):
        """检查类型和切割部位只接受 StudyType/SlicePart 协议字段。"""

        for removed_name in (
            "StudyPart",
            "body_part",
            "check_type",
            "slice_part",
            "study_part",
            "study_type",
            "target_part",
        ):
            with self.subTest(removed_name=removed_name):
                with self.assertRaisesRegex(ValueError, "已移除字段"):
                    Request.from_dict({
                        "dicom_dir": "/tmp/dicom",
                        "output_dir": "/tmp/output",
                        removed_name: "颈椎",
                        "ScanThickness": 1.0,
                    })

    def test_plane_config_rejects_removed_lowercase_fields(self):
        """旧状位字段不再作为兼容入口。"""

        with self.assertRaises(KeyError):
            PlaneConfig.from_dict({
                "plane": "横状位",
                "slice_thickness": 1,
                "slice_spacing": 1,
                "window_presets": ["骨窗"],
            })

    def test_request_validate_rejects_invalid_scan_thickness(self):
        """扫描层厚必须是非负有限毫米数值。"""

        for value in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                request = Request(
                    dicom_dir="/tmp/dicom",
                    output_dir="/tmp/output",
                    scan_thickness=value,
                )
                is_valid, error_message = request.validate()
                self.assertFalse(is_valid)
                self.assertIn("ScanThickness", error_message)


    def test_request_defaults_to_currently_supported_axial_plane(self):
        # 默认请求只能声明当前框架实际支持的能力，其他状位必须显式请求并预检。
        """验证 `test_request_defaults_to_currently_supported_axial_plane` 场景符合预期。"""

        request = Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")

        self.assertEqual(request.to_dict()["plane_configs"], [
            {
                "Plane": "横状位",
                "SliceThickness": 1.0,
                "SliceSpacing": 1.0,
                "WindowPresets": ["骨窗"],
                "SliceType": "MPR",
            },
        ])

    def test_request_validate_rejects_empty_plane_configs(self):
        """验证 `test_request_validate_rejects_empty_plane_configs` 场景符合预期。"""

        self._assert_invalid([], "plane_configs 不能为空")

    def test_request_allows_same_plane_with_different_sampling_configs(self):
        """同一状位允许配置不同的层厚、层间距或数量。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.0,
            "plane_configs": [
                {
                    "Plane": "横状位",
                    "SliceThickness": 1,
                    "SliceSpacing": 1,
                    "WindowPresets": ["骨窗"],
                },
                {
                    "Plane": "横状位",
                    "SliceThickness": 1,
                    "SliceImageCount": 20,
                    "WindowPresets": ["骨窗"],
                },
            ],
        })

        is_valid, error_message = request.validate()

        self.assertTrue(is_valid, error_message)

    def test_request_accepts_each_sampling_parameter_pair(self):
        """层厚、层间距、数量允许任意选择两个。"""

        sampling_pairs = (
            {"SliceThickness": 5, "SliceSpacing": 5},
            {"SliceThickness": 5, "SliceImageCount": 20},
            {"SliceSpacing": 5, "SliceImageCount": 20},
        )
        for sampling in sampling_pairs:
            with self.subTest(sampling=sampling):
                request = Request.from_dict({
                    "dicom_dir": "/tmp/dicom",
                    "output_dir": "/tmp/output",
                    "ScanThickness": 1.0,
                    "plane_configs": [{
                        "Plane": "横状位",
                        "WindowPresets": ["骨窗"],
                        **sampling,
                    }],
                })
                self.assertEqual(request.validate(), (True, None))

    def test_request_rejects_only_one_sampling_parameter(self):
        """只传层厚、层间距、数量中的一个时请求无效。"""

        self._assert_invalid(
            [
                {
                    "Plane": "横状位",
                    "SliceThickness": 1,
                    "WindowPresets": ["骨窗"],
                },
            ],
            "必须传入其中两个或三个",
        )

    def test_zero_sampling_parameter_is_treated_as_not_configured(self):
        """层厚、层间距或数量为 0 时，应由其余两个参数计算。"""

        cases = (
            (
                "SliceThickness",
                "slice_thickness",
                "requested_slice_thickness",
            ),
            (
                "SliceSpacing",
                "slice_spacing",
                "requested_slice_spacing",
            ),
            (
                "SliceImageCount",
                "slice_image_count",
                "requested_slice_image_count",
            ),
        )
        for field_name, attribute_name, requested_attribute_name in cases:
            with self.subTest(field_name=field_name):
                config = {
                    "Plane": "横状位",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "SliceImageCount": 20,
                    "WindowPresets": ["骨窗"],
                }
                config[field_name] = 0
                request = Request.from_dict({
                    "dicom_dir": "/tmp/dicom",
                    "output_dir": "/tmp/output",
                    "ScanThickness": 1.0,
                    "plane_configs": [config],
                })

                plane_config = request.plane_configs[0]
                self.assertIsNone(getattr(plane_config, attribute_name))
                self.assertEqual(
                    getattr(plane_config, requested_attribute_name),
                    0,
                )
                self.assertNotIn(field_name, plane_config.to_dict())
                self.assertEqual(request.validate(), (True, None))

    def test_direct_plane_config_construction_normalizes_zero(self):
        """内部代码直接构造 PlaneConfig 时同样把 0 作为未配置。"""

        config = PlaneConfig(
            PlaneTypeEnum.横状位,
            0,
            5,
            ["骨窗"],
            slice_image_count=20,
        )

        self.assertIsNone(config.slice_thickness)
        self.assertEqual(config.requested_slice_thickness, 0)

    def test_two_zero_sampling_parameters_leave_too_few_values(self):
        """两个参数为 0 时只剩一个有效参数，请求应被拒绝。"""

        self._assert_invalid(
            [{
                "Plane": "横状位",
                "SliceThickness": 0,
                "SliceSpacing": 0,
                "SliceImageCount": 20,
                "WindowPresets": ["骨窗"],
            }],
            "必须传入其中两个或三个",
        )

    def test_negative_sampling_parameters_are_rejected(self):
        """0 表示未配置，但负数仍然是非法输入。"""

        cases = (
            ("SliceThickness", -1, "层厚"),
            ("SliceSpacing", -1, "层间距"),
            ("SliceImageCount", -1, "数量"),
        )
        for field_name, value, expected_message in cases:
            with self.subTest(field_name=field_name):
                config = {
                    "Plane": "横状位",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "SliceImageCount": 20,
                    "WindowPresets": ["骨窗"],
                }
                config[field_name] = value
                with self.assertRaisesRegex(ValueError, expected_message):
                    Request.from_dict({
                        "dicom_dir": "/tmp/dicom",
                        "output_dir": "/tmp/output",
                        "ScanThickness": 1.0,
                        "plane_configs": [config],
                    })

    def test_boolean_sampling_parameters_are_rejected(self):
        """布尔值不能借用数值 0 和 1 的语义。"""

        with self.assertRaisesRegex(ValueError, "层间距"):
            Request.from_dict({
                "dicom_dir": "/tmp/dicom",
                "output_dir": "/tmp/output",
                "ScanThickness": 1.0,
                "plane_configs": [{
                    "Plane": "横状位",
                    "SliceThickness": 1,
                    "SliceSpacing": False,
                    "SliceImageCount": 20,
                    "WindowPresets": ["骨窗"],
                }],
            })

    def test_request_validate_rejects_empty_window_presets(self):
        """验证 `test_request_validate_rejects_empty_window_presets` 场景符合预期。"""

        self._assert_invalid(
            [
                {
                    "Plane": "横状位",
                    "SliceThickness": 1,
                    "SliceSpacing": 1,
                    "WindowPresets": [],
                }
            ],
            "窗宽窗位不能为空",
        )

    def test_request_validate_rejects_unknown_window_preset(self):
        """验证 `test_request_validate_rejects_unknown_window_preset` 场景符合预期。"""

        self._assert_invalid(
            [{
                "Plane": "横状位",
                "SliceThickness": 1,
                "SliceSpacing": 1,
                "WindowPresets": ["血管窗"],
            }],
            "不支持的窗宽窗位预设",
        )

    def _assert_invalid(self, plane_configs, expected_message):
        # 统一构造非法请求，避免每个校验用例重复写路径和断言。
        """为测试准备 `_assert_invalid` 所需的数据或记录行为。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.0,
            "plane_configs": plane_configs,
        })

        is_valid, error_message = request.validate()

        self.assertFalse(is_valid)
        self.assertIn(expected_message, error_message)


if __name__ == "__main__":
    unittest.main()
