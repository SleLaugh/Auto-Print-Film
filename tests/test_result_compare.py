"""交付结果报告对比工具测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from src.core.result_compare import compare_result_reports


class ResultCompareTest(unittest.TestCase):
    """验证报告路径规范化和结构化差异输出。"""

    def _write_report(self, directory: Path, count: int = 2) -> None:
        report = {
            "Success": True,
            "Message": "",
            "SlicePart": "腰椎",
            "StudyType": "CT",
            "ScanThickness": 1.0,
            "PlaneList": [{
                "ConfigId": "配置01",
                "Plane": "横状位",
                "SliceType": "MPR",
                "WindowPresetList": [{
                    "Name": "骨窗",
                    "RegionList": [{
                        "Region": "default",
                        "ActualSliceImageCount": count,
                        "Path": str(directory / "横状位" / "骨窗"),
                    }],
                }],
            }],
        }
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_different_result_roots_do_not_create_path_difference(self):
        with (
            tempfile.TemporaryDirectory() as left_root,
            tempfile.TemporaryDirectory() as right_root,
        ):
            left = Path(left_root)
            right = Path(right_root)
            self._write_report(left)
            self._write_report(right)

            self.assertEqual(compare_result_reports(left, right), [])

    def test_reports_return_nested_field_difference(self):
        with (
            tempfile.TemporaryDirectory() as left_root,
            tempfile.TemporaryDirectory() as right_root,
        ):
            left = Path(left_root)
            right = Path(right_root)
            self._write_report(left, count=2)
            self._write_report(right, count=3)

            differences = compare_result_reports(left, right)

            self.assertEqual(
                differences,
                [{
                    "path": (
                        "PlaneList[0].WindowPresetList[0].RegionList[0]."
                        "ActualSliceImageCount"
                    ),
                    "left": 2,
                    "right": 3,
                }],
            )


if __name__ == "__main__":
    unittest.main()
