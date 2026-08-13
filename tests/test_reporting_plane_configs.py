import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.core.errors import OutputValidationError
from src.models import Request
from src.stages.stage_reporting import ReportingStage


class ReportingPlaneConfigsTest(unittest.TestCase):
    """汇总 `ReportingPlaneConfigsTest` 相关测试场景。"""

    def test_delivery_report_contains_only_simple_plane_configuration(self):
        """验证 `test_delivery_report_contains_only_simple_plane_configuration` 场景符合预期。"""

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
        """验证 `test_delivery_report_adds_plane_image_count_and_window_path` 场景符合预期。"""

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
        """验证 `test_debug_report_keeps_full_validation_details` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=1, series_uid="1.2.3.11")
            context = self._make_context(output_dir, dcm_dir, slice_count=2)
            (output_dir / "report.json").write_text(
                '{"stale": true}',
                encoding="utf-8",
            )

            with patch(
                "src.stages.stage_reporting.is_debug_log_file_enabled",
                return_value=True,
            ):
                with self.assertRaises(OutputValidationError) as raised:
                    ReportingStage().execute(context)

            self.assertIn("期望 2 张，实际 1 张", str(raised.exception))
            debug_path = output_dir / "metadata" / "处理报告_调试.json"
            self.assertEqual(debug_path.name, "处理报告_调试.json")
            debug_report = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertFalse(debug_report["validation"]["valid"])
            self.assertTrue(
                any("期望 2 张，实际 1 张" in error
                    for error in debug_report["validation"]["errors"])
            )
            self.assertFalse((output_dir / "report.json").exists())

    def test_execute_stores_validation_summary_in_context(self):
        """验证 `test_execute_stores_validation_summary_in_context` 场景符合预期。"""

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

    def test_execute_rejects_missing_localizer(self):
        """正式切片齐全时也必须存在可读且带空间几何的定位图。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=1, series_uid="1.2.3.13")
            (dcm_dir / "Localizer.dcm").unlink()
            context = self._make_context(output_dir, dcm_dir, slice_count=1)

            with patch(
                "src.stages.stage_reporting.is_debug_log_file_enabled",
                return_value=False,
            ):
                with self.assertRaises(OutputValidationError) as raised:
                    ReportingStage().execute(context)

            self.assertTrue(
                any("缺少定位图" in error
                    for error in raised.exception.details["errors"])
            )

    def test_validation_rejects_geometry_difference_between_windows(self):
        """同一计划各窗位只能改变灰度，不得改变切片或定位图空间几何。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            bone_dir = output_dir / "横状位" / "骨窗"
            soft_dir = output_dir / "横状位" / "软组织窗"
            self._write_dicom_series(bone_dir, count=2, series_uid="1.2.3.14")
            self._write_dicom_series(soft_dir, count=2, series_uid="1.2.3.15")
            changed_path = soft_dir / "2.dcm"
            dataset = pydicom.dcmread(changed_path)
            dataset.ImagePositionPatient = [0.0, 0.0, 20.0]
            dataset.save_as(changed_path, enforce_file_format=True)

            context = self._make_context(output_dir, bone_dir, slice_count=2)
            reconstruction = context.get_result("reconstruction")
            reconstruction["outputs"][0]["window_outputs"]["软组织窗"] = str(
                soft_dir
            )

            validation = ReportingStage._validate_reconstruction(reconstruction)

            self.assertFalse(validation["valid"])
            self.assertTrue(any(
                "软组织窗" in error
                and "ImagePositionPatient" in error
                and "2.dcm" in error
                for error in validation["errors"]
            ))

    def test_validation_rejects_slice_metrics_different_from_plan(self):
        """正式切片的层厚和层间距必须与第七阶段计划摘要一致。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=1, series_uid="1.2.3.16")
            changed_path = dcm_dir / "1.dcm"
            dataset = pydicom.dcmread(changed_path)
            dataset.SliceThickness = "2"
            dataset.SpacingBetweenSlices = "3"
            dataset.save_as(changed_path, enforce_file_format=True)
            context = self._make_context(output_dir, dcm_dir, slice_count=1)

            validation = ReportingStage._validate_reconstruction(
                context.get_result("reconstruction")
            )

            self.assertFalse(validation["valid"])
            self.assertTrue(any(
                "SliceThickness 与计划 1mm 不一致" in error
                for error in validation["errors"]
            ))
            self.assertTrue(any(
                "SpacingBetweenSlices 与计划 1mm 不一致" in error
                for error in validation["errors"]
            ))

    def test_validation_rejects_duplicate_uids_between_windows(self):
        """不同窗位必须使用独立 Series UID 和 SOP UID。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            bone_dir = output_dir / "横状位" / "骨窗"
            soft_dir = output_dir / "横状位" / "软组织窗"
            self._write_dicom_series(bone_dir, count=1, series_uid="1.2.3.17")
            self._write_dicom_series(soft_dir, count=1, series_uid="1.2.3.18")
            bone_slice = pydicom.dcmread(bone_dir / "1.dcm")
            soft_slice_path = soft_dir / "1.dcm"
            soft_slice = pydicom.dcmread(soft_slice_path)
            soft_slice.SeriesInstanceUID = bone_slice.SeriesInstanceUID
            soft_slice.SOPInstanceUID = bone_slice.SOPInstanceUID
            soft_slice.save_as(soft_slice_path, enforce_file_format=True)

            context = self._make_context(output_dir, bone_dir, slice_count=1)
            reconstruction = context.get_result("reconstruction")
            reconstruction["outputs"][0]["window_outputs"]["软组织窗"] = str(
                soft_dir
            )
            reconstruction["outputs"][0]["image_count"] = 2
            reconstruction["image_count"] = 2

            validation = ReportingStage._validate_reconstruction(reconstruction)

            self.assertFalse(validation["valid"])
            self.assertTrue(any(
                "重复使用 SeriesInstanceUID" in error
                for error in validation["errors"]
            ))
            self.assertTrue(any(
                "重复使用 SOPInstanceUID" in error
                for error in validation["errors"]
            ))

    def test_validation_rejects_nonorthogonal_dicom_orientation(self):
        """方向余弦不是正交单位向量时不能发布 DICOM。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=1, series_uid="1.2.3.19")
            changed_path = dcm_dir / "1.dcm"
            dataset = pydicom.dcmread(changed_path)
            dataset.ImageOrientationPatient = [1, 0, 0, 1, 0, 0]
            dataset.save_as(changed_path, enforce_file_format=True)
            context = self._make_context(output_dir, dcm_dir, slice_count=1)

            validation = ReportingStage._validate_reconstruction(
                context.get_result("reconstruction")
            )

            self.assertFalse(validation["valid"])
            self.assertTrue(any(
                "正交单位向量" in error for error in validation["errors"]
            ))

    @staticmethod
    def _make_context(output_dir: Path, dcm_dir: Path, *, slice_count: int):
        """为测试准备 `_make_context` 所需的数据或记录行为。"""

        request = Request(dicom_dir="/tmp/dicom", output_dir=str(output_dir))
        context = ProcessingContext(request)
        context.set_result("planning", {
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
                "geometry_type": "plane_stack",
                "slice_thickness": 1.0,
                "slice_spacing": 1.0,
                "slice_count": slice_count,
                "image_count": slice_count,
                "window_outputs": {"骨窗": str(dcm_dir)},
            }],
        })
        return context

    @staticmethod
    def _write_dicom_series(directory: Path, *, count: int, series_uid: str):
        """为测试准备 `_write_dicom_series` 所需的数据或记录行为。"""

        directory.mkdir(parents=True, exist_ok=True)
        for index in range(1, count + 1):
            image = sitk.GetImageFromArray(np.zeros((3, 4), dtype=np.uint16))
            image.SetSpacing((0.8, 0.7))
            image.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
            image.SetMetaData(
                "0008|0018",
                f"{series_uid}.{index}",
            )
            image.SetMetaData("0008|0008", "DERIVED\\SECONDARY")
            image.SetMetaData("0008|0060", "CT")
            image.SetMetaData("0020|000d", "1.2.3.1")
            image.SetMetaData("0020|000e", series_uid)
            image.SetMetaData("0020|0013", str(index))
            image.SetMetaData("0020|0032", f"0\\0\\{index - 1}")
            image.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
            image.SetMetaData("0018|0050", "1")
            image.SetMetaData("0018|0088", "1")
            image.SetMetaData("0028|0030", "0.7\\0.8")
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            output_path = directory / f"{index}.dcm"
            writer.SetFileName(str(output_path))
            writer.Execute(image)
            ReportingPlaneConfigsTest._restore_test_geometry(
                output_path,
                position=[0.0, 0.0, float(index - 1)],
            )

        localizer = sitk.GetImageFromArray(np.zeros((5, 6), dtype=np.uint16))
        localizer.SetSpacing((1.2, 1.1))
        localizer.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
        localizer.SetMetaData("0008|0018", f"{series_uid}.98")
        localizer.SetMetaData("0008|0008", "DERIVED\\SECONDARY\\LOCALIZER")
        localizer.SetMetaData("0008|0060", "CT")
        localizer.SetMetaData("0020|000d", "1.2.3.1")
        localizer.SetMetaData("0020|000e", f"{series_uid}.99")
        localizer.SetMetaData("0020|0013", "1")
        localizer.SetMetaData("0020|0032", "10\\20\\30")
        localizer.SetMetaData("0020|0037", "0\\1\\0\\0\\0\\-1")
        localizer.SetMetaData("0028|0030", "1.1\\1.2")
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        localizer_path = directory / "Localizer.dcm"
        writer.SetFileName(str(localizer_path))
        writer.Execute(localizer)
        dataset = pydicom.dcmread(localizer_path)
        dataset.ImageType = ["DERIVED", "SECONDARY", "LOCALIZER"]
        dataset.StudyInstanceUID = "1.2.3.1"
        dataset.SeriesInstanceUID = f"{series_uid}.99"
        dataset.SOPInstanceUID = f"{series_uid}.98"
        dataset.InstanceNumber = 1
        dataset.PixelSpacing = [1.1, 1.2]
        dataset.ImagePositionPatient = [10.0, 20.0, 30.0]
        dataset.ImageOrientationPatient = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        dataset.save_as(localizer_path, enforce_file_format=True)

    @staticmethod
    def _restore_test_geometry(output_path: Path, *, position: list[float]):
        """恢复测试 DICOM 会被 GDCM 按二维默认值覆盖的空间标签。"""

        dataset = pydicom.dcmread(output_path)
        dataset.PixelSpacing = [0.7, 0.8]
        dataset.ImagePositionPatient = position
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.SliceThickness = "1"
        dataset.SpacingBetweenSlices = "1"
        dataset.save_as(output_path, enforce_file_format=True)


if __name__ == "__main__":
    unittest.main()
