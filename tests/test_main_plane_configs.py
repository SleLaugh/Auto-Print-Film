import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from src.config import AppConfig
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

    def test_main_writes_failure_report_when_plane_configs_json_is_invalid(self):
        """请求对象建立前解析失败时也必须返回 report.json。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "main.py",
                "-i", "/tmp/dicom",
                "-o", temp_dir,
                "--study_part", "颈椎",
                "--scan_thickness", "1.5",
                "--plane_configs", "not-json",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(main, "load_app_config", return_value=AppConfig()),
                patch.object(main, "configure_logging"),
                patch.object(main, "RunLog"),
                patch.object(main, "ErrorLog"),
            ):
                exit_code = main.main()

            report = json.loads(
                (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 1)
        self.assertFalse(report["Success"])
        self.assertIn("plane_configs 不是合法 JSON", report["Message"])
        self.assertEqual(report["StudyPart"], "颈椎")
        self.assertEqual(report["ScanThickness"], 1.5)
        self.assertEqual(report["PlaneList"], [])

    def test_main_writes_failure_report_when_argparse_rejects_arguments(self):
        """argparse 失败时应从原始参数中恢复输出目录并写失败报告。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "main.py",
                "-i", "/tmp/dicom",
                "-o", temp_dir,
                "--scan_thickness", "invalid",
            ]
            with patch.object(sys, "argv", argv):
                exit_code = main.main()

            report = json.loads(
                (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 2)
        self.assertFalse(report["Success"])
        self.assertIn("invalid float value", report["Message"])
        self.assertIn("invalid", report["Message"])
        self.assertIsNone(report["ScanThickness"])

    def test_main_writes_failure_report_when_task_is_interrupted(self):
        """请求建立前收到可捕获中断时也必须写入终态报告。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "main.py",
                "-i", "/tmp/dicom",
                "-o", temp_dir,
                "--study_part", "颈椎",
                "--scan_thickness", "1.0",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(main, "parse_arguments", side_effect=KeyboardInterrupt),
                patch.object(main, "ErrorLog"),
            ):
                exit_code = main.main()

            report = json.loads(
                (Path(temp_dir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 130)
        self.assertFalse(report["Success"])
        self.assertEqual(report["Message"], "任务被中断")
        self.assertEqual(report["StudyPart"], "颈椎")
        self.assertEqual(report["PlaneList"], [])


if __name__ == "__main__":
    unittest.main()
