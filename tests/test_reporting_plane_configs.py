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
                "ScanThickness": 1.5,
                "plane_configs": [
                    {
                        "Plane": "冠状位",
                        "SliceThickness": 5,
                        "SliceSpacing": 10,
                        "WindowPresets": ["骨窗", "纵隔窗"],
                    }
                ],
            })
            context = ProcessingContext(request)

            report_path = ReportingStage()._generate_report(context)

            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertEqual(
                set(report),
                {"StudyPart", "StudyType", "ScanThickness", "PlaneList"},
            )
            self.assertEqual(report["StudyPart"], "腰椎")
            self.assertEqual(report["StudyType"], "CT")
            self.assertEqual(report["ScanThickness"], 1.5)
            self.assertEqual(report["PlaneList"], [
                {
                    "ConfigId": "配置01",
                    "Plane": "冠状位",
                    "SliceType": "MPR",
                    "SliceThickness": 5.0,
                    "SliceSpacing": 10.0,
                    "SliceImageCount": None,
                    "line_extent_mode": "整平面",
                    "WindowPresetList": [
                        {"Name": "骨窗", "RegionList": []},
                        {"Name": "纵隔窗", "RegionList": []},
                    ],
                }
            ])

    def test_delivery_report_adds_region_count_and_window_path(self):
        """窗位下的重建区域返回实际数量和目录。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dcm_dir = output_dir / "横状位" / "骨窗"
            self._write_dicom_series(dcm_dir, count=2, series_uid="1.2.3.10")
            context = self._make_context(output_dir, dcm_dir, slice_count=2)

            report_path = ReportingStage()._generate_report(context)

            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            axial = next(
                item for item in report["PlaneList"]
                if item["Plane"] == "横状位"
            )
            window = axial["WindowPresetList"][0]
            self.assertEqual(window["Name"], "骨窗")
            self.assertEqual(
                window["RegionList"],
                [{
                    "Region": "default",
                    "ActualSliceThickness": 1.0,
                    "ActualSliceSpacing": 1.0,
                    "ActualSliceImageCount": 2,
                    "Path": str(dcm_dir),
                }],
            )

    def test_delivery_report_places_each_region_under_its_window(self):
        """报告层级必须与状位/窗位/重建区域的磁盘层级一致。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.0,
            "plane_configs": [{
                "Plane": "横状位",
                "SliceType": "椎间隙",
                "SliceThickness": 5,
                "SliceSpacing": 5,
                "WindowPresets": ["骨窗", "肺窗"],
            }],
        })
        context = ProcessingContext(request)
        context.set_result("reconstruction", {"outputs": [{
            "config_id": "配置01",
            "plane": "横状位",
            "slice_type": "椎间隙",
            "group_name": "L1_L2",
            "slice_thickness": 5.0,
            "slice_spacing": 5.0,
            "slice_count": 20,
            "window_outputs": {
                "骨窗": "/tmp/output/横状位/骨窗/L1_L2",
                "肺窗": "/tmp/output/横状位/肺窗/L1_L2",
            },
        }]})

        windows = ReportingStage()._build_delivery_report(context)[
            "PlaneList"
        ][0]["WindowPresetList"]

        self.assertEqual([window["Name"] for window in windows], ["骨窗", "肺窗"])
        self.assertEqual(
            [window["RegionList"][0]["Region"] for window in windows],
            ["L1_L2", "L1_L2"],
        )
        self.assertEqual(
            [window["RegionList"][0]["Path"] for window in windows],
            [
                "/tmp/output/横状位/骨窗/L1_L2",
                "/tmp/output/横状位/肺窗/L1_L2",
            ],
        )

    def test_delivery_report_lists_every_interspace_region_count(self):
        """椎间隙数量按每个区域执行，并位于对应窗位之下。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request.from_dict({
                "dicom_dir": "/tmp/dicom",
                "output_dir": temp_dir,
                "StudyPart": "腰椎",
                "StudyType": "CT",
                "ScanThickness": 1.0,
                "plane_configs": [{
                    "Plane": "横状位",
                    "SliceType": "椎间隙",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "SliceImageCount": 20,
                    "WindowPresets": ["骨窗"],
                }],
            })
            context = ProcessingContext(request)
            groups = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]
            context.set_result("reconstruction", {
                "outputs": [
                    {
                        "plan_id": f"腰椎_椎间隙_{group}_横状位",
                        "config_id": "配置01",
                        "plane": "横状位",
                        "slice_type": "椎间隙",
                        "group_name": group,
                        "slice_thickness": 5.0,
                        "slice_spacing": 5.0,
                        "slice_count": 20,
                        "sampling_details": {
                            "effective_range_mm": 10.0,
                            "total_range_mm": 100.0,
                            "start_extension_mm": 45.0,
                            "end_extension_mm": 45.0,
                            "adjusted": False,
                            "adjustment_reason": "",
                        },
                        "window_outputs": {
                            "骨窗": str(
                                Path(temp_dir)
                                / "横状位"
                                / "骨窗"
                                / group
                            ),
                        },
                    }
                    for group in groups
                ],
            })

            report = ReportingStage()._build_delivery_report(context)

            axial = report["PlaneList"][0]
            self.assertEqual(axial["SliceImageCount"], 20)
            regions = axial["WindowPresetList"][0]["RegionList"]
            self.assertEqual(len(regions), 5)
            self.assertTrue(all(
                region["ActualSliceImageCount"] == 20
                for region in regions
            ))
            self.assertEqual(
                [region["Region"] for region in regions],
                groups,
            )
            self.assertTrue(all(
                region["Path"] == str(
                    Path(temp_dir)
                    / "横状位"
                    / "骨窗"
                    / region["Region"]
                )
                for region in regions
            ))

    def test_zero_requested_count_is_preserved_beside_actual_count(self):
        """传入数量 0 必须原样报告，实际计算数量单独返回。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.0,
            "plane_configs": [{
                "Plane": "矢状位",
                "SliceThickness": 5,
                "SliceSpacing": 5,
                "SliceImageCount": 0,
                "WindowPresets": ["骨窗"],
            }],
        })
        context = ProcessingContext(request)
        context.set_result("reconstruction", {"outputs": [{
            "config_id": "配置01",
            "plane": "矢状位",
            "slice_type": "MPR",
            "group_name": "whole_range",
            "slice_thickness": 5.0,
            "slice_spacing": 5.0,
            "slice_count": 21,
            "window_outputs": {"骨窗": "/tmp/矢状位/骨窗"},
        }]})

        plane = ReportingStage()._build_delivery_report(context)["PlaneList"][0]

        self.assertEqual(plane["SliceImageCount"], 0)
        self.assertEqual(
            plane["WindowPresetList"][0]["RegionList"][0][
                "ActualSliceImageCount"
            ],
            21,
        )

    def test_delivery_report_uses_actual_slice_type_and_sampling_per_plane(self):
        """横状位椎间隙、矢状位 MPR、冠状位 CPR 必须按实际输出分别报告。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "StudyPart": "腰椎",
            "StudyType": "CT",
            "ScanThickness": 1.0,
            "plane_configs": [
                {
                    "Plane": "横状位",
                    "SliceType": "椎间隙",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "SliceImageCount": 0,
                    "WindowPresets": ["骨窗"],
                },
                {
                    "Plane": "矢状位",
                    "SliceType": "MPR",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "SliceImageCount": 0,
                    "WindowPresets": ["骨窗"],
                },
                {
                    "Plane": "冠状位",
                    "SliceType": "CPR",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "SliceImageCount": 0,
                    "WindowPresets": ["骨窗"],
                },
            ],
        })
        context = ProcessingContext(request)
        context.set_result("reconstruction", {
            "outputs": [
                {
                    "config_id": config_id,
                    "plane": plane,
                    "slice_type": slice_type,
                    "group_name": group_name,
                    "slice_thickness": thickness,
                    "slice_spacing": spacing,
                    "slice_count": count,
                    "window_outputs": {"骨窗": f"/tmp/{plane}/{slice_type}"},
                }
                for config_id, plane, slice_type, group_name, thickness, spacing, count in (
                    ("配置01", "横状位", "椎间隙", "L1_L2", 5.0, 5.0, 12),
                    ("配置02", "矢状位", "MPR", "whole_range", 4.0, 4.0, 21),
                    ("配置03", "冠状位", "CPR", "CPR", 3.0, 3.0, 31),
                )
            ],
        })

        report = ReportingStage()._build_delivery_report(context)

        configs = {
            item["Plane"]: item
            for item in report["PlaneList"]
        }
        expected = {
            "横状位": ("椎间隙", 5.0, 5.0, 12),
            "矢状位": ("MPR", 4.0, 4.0, 21),
            "冠状位": ("CPR", 3.0, 3.0, 31),
        }
        for plane, values in expected.items():
            with self.subTest(plane=plane):
                slice_type, thickness, spacing, count = values
                config = configs[plane]
                self.assertEqual(config["SliceType"], slice_type)
                region = config["WindowPresetList"][0]["RegionList"][0]
                self.assertEqual(region["ActualSliceThickness"], thickness)
                self.assertEqual(region["ActualSliceSpacing"], spacing)
                self.assertEqual(region["ActualSliceImageCount"], count)

        self.assertEqual(
            [item["ConfigId"] for item in report["PlaneList"]],
            ["配置01", "配置02", "配置03"],
        )

    def test_global_config_ids_do_not_mix_different_plane_outputs(self):
        """全局配置编号必须让不同状位关联各自实际输出。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.0,
            "plane_configs": [
                {
                    "Plane": "横状位",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "WindowPresets": ["骨窗"],
                },
                {
                    "Plane": "冠状位",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "WindowPresets": ["骨窗"],
                },
            ],
        })
        context = ProcessingContext(request)
        context.set_result("reconstruction", {
            "outputs": [
                {
                    "config_id": config_id,
                    "plane": plane,
                    "slice_type": "MPR",
                    "group_name": "whole_range",
                    "slice_thickness": 5.0,
                    "slice_spacing": 5.0,
                    "slice_count": count,
                    "window_outputs": {"骨窗": f"/tmp/{plane}"},
                }
                for config_id, plane, count in (
                    ("配置01", "横状位", 10),
                    ("配置02", "冠状位", 20),
                )
            ],
        })

        report = ReportingStage()._build_delivery_report(context)

        configs = {
            item["Plane"]: item
            for item in report["PlaneList"]
        }
        self.assertEqual(
            configs["横状位"]["WindowPresetList"][0]["RegionList"][0][
                "ActualSliceImageCount"
            ],
            10,
        )
        self.assertEqual(
            configs["冠状位"]["WindowPresetList"][0]["RegionList"][0][
                "ActualSliceImageCount"
            ],
            20,
        )

    def test_same_plane_sampling_configs_match_their_own_outputs(self):
        """同状位多份配置必须按配置编号关联各自的实际参数和目录。"""

        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.0,
            "plane_configs": [
                {
                    "Plane": "横状位",
                    "SliceThickness": 5,
                    "SliceSpacing": 5,
                    "WindowPresets": ["骨窗"],
                },
                {
                    "Plane": "横状位",
                    "SliceThickness": 3,
                    "SliceSpacing": 3,
                    "WindowPresets": ["骨窗"],
                },
            ],
        })
        context = ProcessingContext(request)
        context.set_result("reconstruction", {
            "outputs": [
                {
                    "config_id": config_id,
                    "plane": output_plane,
                    "source_plane": "横状位",
                    "slice_type": "MPR",
                    "group_name": "whole_range",
                    "slice_thickness": thickness,
                    "slice_spacing": thickness,
                    "slice_count": count,
                    "window_outputs": {"骨窗": path},
                }
                for config_id, output_plane, thickness, count, path in (
                    ("配置01", "横状位", 5.0, 20, "/tmp/横状位/骨窗"),
                    ("配置02", "横状位1", 3.0, 30, "/tmp/横状位1/骨窗"),
                )
            ],
        })

        report = ReportingStage()._build_delivery_report(context)

        first, second = report["PlaneList"]
        self.assertEqual(first["ConfigId"], "配置01")
        self.assertEqual(first["Plane"], "横状位")
        self.assertEqual(first["SliceThickness"], 5.0)
        self.assertEqual(
            first["WindowPresetList"][0]["RegionList"][0]["Path"],
            "/tmp/横状位/骨窗",
        )
        self.assertEqual(second["ConfigId"], "配置02")
        self.assertEqual(second["Plane"], "横状位1")
        self.assertEqual(second["SliceThickness"], 3.0)
        self.assertEqual(
            second["WindowPresetList"][0]["RegionList"][0]["Path"],
            "/tmp/横状位1/骨窗",
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
                "src.stages.stage_reporting.is_debug_file_enabled",
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
                "src.stages.stage_reporting.is_debug_file_enabled",
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
                "src.stages.stage_reporting.is_debug_file_enabled",
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
                "slice_type": "MPR",
                "group_name": "整切",
                "geometry_type": "plane_stack",
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
                "slice_type": "MPR",
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
