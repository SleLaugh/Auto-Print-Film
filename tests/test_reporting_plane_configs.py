import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import Request
from src.stages.stage_reporting import ReportingStage


class ReportingPlaneConfigsTest(unittest.TestCase):
    def test_delivery_report_contains_only_simple_plane_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request.from_dict({
                "dicom_dir": "/tmp/dicom",
                "output_dir": temp_dir,
                "plane_configs": [
                    {
                        "plane": "冠状位",
                        "slice_thickness": 5,
                        "slice_spacing": 10,
                        "window_presets": ["骨窗", "纵隔窗"],
                    }
                ],
            })
            context = ProcessingContext(request)

            report_path = ReportingStage()._generate_report(context)

            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertEqual(
                set(report),
                {"body_part", "check_type", "plane_configs"},
            )
            self.assertEqual(report["body_part"], "腰椎")
            self.assertEqual(report["check_type"], "CT")
            self.assertEqual(report["plane_configs"], [
                {
                    "plane": "冠状位",
                    "slice_thickness": 5.0,
                    "slice_spacing": 10.0,
                    "line_extent_mode": "full_plane",
                    "image_count": 0,
                    "window_presets": [
                        {"name": "骨窗", "path": ""},
                        {"name": "纵隔窗", "path": ""},
                    ],
                }
            ])

    def test_delivery_report_adds_plane_image_count_and_window_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=2, series_uid="1.2.3.10")
            context = self._make_context(output_dir, dcm_dir, slice_count=2)

            report_path = ReportingStage()._generate_report(context)

            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            axial = next(
                item for item in report["plane_configs"]
                if item["plane"] == "横状位"
            )
            self.assertEqual(axial["image_count"], 2)
            self.assertEqual(
                axial["window_presets"],
                [{"name": "骨窗", "path": str(dcm_dir)}],
            )

    def test_debug_report_keeps_full_validation_details(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=1, series_uid="1.2.3.11")
            context = self._make_context(output_dir, dcm_dir, slice_count=2)

            with patch(
                "src.stages.stage_reporting.is_debug_log_file_enabled",
                return_value=True,
            ):
                ReportingStage().execute(context)

            result = context.get_result("reporting")
            debug_path = Path(result["debug_report_path"])
            self.assertEqual(debug_path.name, "处理报告_调试.json")
            debug_report = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertFalse(debug_report["validation"]["valid"])
            self.assertTrue(
                any("期望 2 张，实际 1 张" in error
                    for error in debug_report["validation"]["errors"])
            )

    def test_execute_stores_validation_summary_in_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=1, series_uid="1.2.3.12")
            context = self._make_context(output_dir, dcm_dir, slice_count=1)

            with patch(
                "src.stages.stage_reporting.is_debug_log_file_enabled",
                return_value=False,
            ):
                ReportingStage().execute(context)

            result = context.get_result("reporting")
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(result["image_count"], 1)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertEqual(result["debug_report_path"], "")

    @staticmethod
    def _make_context(output_dir: Path, dcm_dir: Path, *, slice_count: int):
        request = Request(dicom_dir="/tmp/dicom", output_dir=str(output_dir))
        context = ProcessingContext(request)
        context.set_result("mapping", {
            "planner": "测试 planner",
            "plan_count": 1,
            "plans": [{
                "plan_id": "测试_plan",
                "plane": "横状位",
                "group_name": "整切",
                "geometry_type": "plane_stack",
                "cut_mode": "whole_range",
                "slices": [{"index": index} for index in range(1, slice_count + 1)],
                "warnings": [],
            }],
            "warnings": [],
            "diagnostics": [],
            "metadata_path": "",
        })
        context.set_result("reconstruction", {
            "plan_count": 1,
            "slice_count": slice_count,
            "image_count": slice_count,
            "planes": {"横状位": {
                "plan_count": 1,
                "slice_count": slice_count,
                "image_count": slice_count,
                "output_dir": str(output_dir / "横状位"),
            }},
            "outputs": [{
                "plan_id": "测试_plan",
                "plane": "横状位",
                "group_name": "整切",
                "slice_count": slice_count,
                "image_count": slice_count,
                "window_outputs": {"骨窗": str(dcm_dir)},
            }],
        })
        return context

    @staticmethod
    def _write_dicom_series(directory: Path, *, count: int, series_uid: str):
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(1, count + 1):
            image = sitk.GetImageFromArray(np.zeros((1, 1), dtype=np.uint16))
            image.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
            image.SetMetaData(
                "0008|0018",
                f"1.2.826.0.1.3680043.2.1125.9{index}",
            )
            image.SetMetaData("0008|0008", "DERIVED\\SECONDARY")
            image.SetMetaData("0008|0060", "CT")
            image.SetMetaData("0020|000d", "1.2.3.1")
            image.SetMetaData("0020|000e", series_uid)
            image.SetMetaData("0020|0013", str(index))
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(str(directory / f"{index}.dcm"))
            writer.Execute(image)


if __name__ == "__main__":
    unittest.main()
