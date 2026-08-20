import sys
import unittest

import main
from src.models.enums import PlaneTypeEnum


class MainPlaneConfigsTest(unittest.TestCase):
    """汇总 `MainPlaneConfigsTest` 相关测试场景。"""

    def test_parse_arguments_no_longer_accepts_cut_mode(self):
        """验证 `test_parse_arguments_no_longer_accepts_cut_mode` 场景符合预期。"""

        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py", "-i", "/tmp/dicom", "-o", "/tmp/output",
                "--scan_thickness", "1.0",
                "--cut_mode", "whole_range",
            ]
            with self.assertRaises(SystemExit):
                main.parse_arguments()
        finally:
            sys.argv = original_argv

    def test_parse_plane_configs_json_string(self):
        # CLI 收到的是一整段 JSON 字符串，解析后应得到 PlaneConfig 对象。
        """验证 `test_parse_plane_configs_json_string` 场景符合预期。"""

        configs = main.parse_plane_configs(
            '[{"Plane":"冠状位","SliceThickness":5,"SliceSpacing":10,"WindowPresets":["骨窗","肺窗"]}]'
        )

        self.assertIs(configs[0].plane, PlaneTypeEnum.冠状位)
        self.assertEqual(configs[0].slice_thickness, 5.0)
        self.assertEqual(configs[0].slice_spacing, 10.0)
        self.assertEqual(configs[0].window_presets, [
            "骨窗",
            "肺窗",
        ])

    def test_parse_plane_configs_none_uses_request_default(self):
        # None 表示用户没有传 CLI 参数，此时应交给 Request 使用默认值。
        """验证 `test_parse_plane_configs_none_uses_request_default` 场景符合预期。"""

        self.assertIsNone(main.parse_plane_configs(None))

    def test_parse_plane_configs_rejects_non_array_json(self):
        # 顶层必须是数组，因为一次请求可以包含多个状位配置。
        """验证 `test_parse_plane_configs_rejects_non_array_json` 场景符合预期。"""

        with self.assertRaisesRegex(ValueError, "plane_configs 必须是 JSON 数组"):
            main.parse_plane_configs('{"Plane":"冠状位"}')

    def test_parse_arguments_accepts_underscore_protocol(self):
        """CLI 参数统一使用小写下划线命名。"""

        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "-i",
                "/tmp/dicom",
                "-o",
                "/tmp/output",
                "--scan_thickness",
                "1.5",
                "--plane_configs",
                '[{"Plane":"冠状位","SliceThickness":5,"SliceSpacing":10,"WindowPresets":["骨窗"]}]',
            ]

            args = main.parse_arguments()

            self.assertEqual(
                args.plane_configs,
                '[{"Plane":"冠状位","SliceThickness":5,"SliceSpacing":10,"WindowPresets":["骨窗"]}]',
            )
            self.assertEqual(args.scan_thickness, 1.5)
        finally:
            # 测试结束后恢复 argv，避免污染同进程里的其他测试。
            sys.argv = original_argv

    def test_parse_arguments_rejects_removed_hyphen_aliases(self):
        """CLI 不再同时维护连字符参数别名。"""

        original_argv = sys.argv
        try:
            for removed_name in (
                "--study-part",
                "--study-type",
                "--scan-thickness",
                "--plane-configs",
            ):
                with self.subTest(removed_name=removed_name):
                    sys.argv = [
                        "main.py",
                        "-i", "/tmp/dicom",
                        "-o", "/tmp/output",
                        "--scan_thickness", "1.0",
                        removed_name, "1.0",
                    ]
                    with self.assertRaises(SystemExit):
                        main.parse_arguments()
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
