import json
import tempfile
import unittest
from pathlib import Path

import SimpleITK as sitk
import numpy as np

from src.utils.Logger_utils import _logger_manager
_logger_manager.debug_log_file_enabled = True

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request, WindowPresetEnum
from src.models.enums import CutLineExtentModeEnum
from src.models.enums import BodyPartEnum, CutModeEnum, PlaneTypeEnum
from src.planning.base import select_planner
from src.planning.lumbar import LumbarInterspacePlanner, LumbarWholeRangePlanner
from src.planning.single_label import SingleLabelWholeRangePlanner
from src.stages.stage_mapping import MappingStage


class Stage4SlicePlanningTest(unittest.TestCase):
    def test_select_planner_supports_configured_whole_range_body_parts(self):
        config_path = Path("src/config/localization_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        whole_range_body_parts = [
            body_part
            for body_part, cut_configs in config["targets"].items()
            if "whole_range" in cut_configs
        ]

        self.assertEqual(whole_range_body_parts, [
            "颈椎",
            "胸椎",
            "腰椎",
            "肺部",
            "头骨",
            "髋骨",
        ])

        for body_part in whole_range_body_parts:
            with self.subTest(body_part=body_part):
                planner = select_planner(body_part, CutModeEnum.整切.value)
                if body_part == BodyPartEnum.腰椎.value:
                    self.assertIsInstance(planner, LumbarWholeRangePlanner)
                else:
                    self.assertIsInstance(planner, SingleLabelWholeRangePlanner)

    def test_single_label_whole_range_planner_builds_axial_plan_from_bbox(self):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=[WindowPresetEnum.肺窗],
                )
            ],
        )
        localization_result = {
            "body_part": "肺部",
            "cut_mode": "whole_range",
            "correction_scope_bbox": {
                "z_min": 0,
                "z_max": 10,
                "y_min": 20,
                "y_max": 80,
                "x_min": 30,
                "x_max": 90,
            },
            "image": {
                "spacing_xyz": [1.0, 1.0, 1.0],
                "origin_xyz": [0.0, 0.0, 0.0],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
        }

        result = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        )

        self.assertEqual(result.planner_name, "SingleLabelWholeRangePlanner")
        self.assertEqual(len(result.plans), 1)
        plan = result.plans[0]
        self.assertEqual(plan.plan_id, "肺部_whole_range_横状位")
        self.assertEqual(len(plan.slices), 2)
        self.assertEqual(plan.slices[1].center.to_dict(), {
            "x": 60.0,
            "y": 50.0,
            "z": 7.0,
        })
        self.assertEqual(plan.window_presets, ["肺窗"])

    def test_axial_whole_range_defaults_to_full_plane_extent(self):
        request = Request(
            dicom_dir="/tmp/dicom", output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部, cut_mode=CutModeEnum.整切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位, slice_thickness=3.0,
                slice_spacing=5.0, window_presets=[WindowPresetEnum.肺窗],
            )],
        )
        loc = self._lumbar_localization_result()
        loc["image"]["size_xyz"] = [100, 120, 30]
        loc["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]

        plan = SingleLabelWholeRangePlanner().build_plans(request, loc).plans[0]

        self.assertEqual(plan.extent_mm.to_dict(), {"width": 99.0, "height": 238.0})

    def test_axial_whole_range_can_limit_extent_to_target_bbox(self):
        request = Request(
            dicom_dir="/tmp/dicom", output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部, cut_mode=CutModeEnum.整切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位, slice_thickness=3.0,
                slice_spacing=5.0, window_presets=[WindowPresetEnum.肺窗],
                line_extent_mode=CutLineExtentModeEnum.部位范围,
            )],
        )
        loc = self._lumbar_localization_result()
        loc["image"]["size_xyz"] = [100, 120, 30]
        loc["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]

        plan = SingleLabelWholeRangePlanner().build_plans(request, loc).plans[0]

        self.assertEqual(plan.extent_mm.to_dict(), {"width": 10.0, "height": 20.0})

    def test_axial_whole_range_insets_center_range_by_half_thickness(self):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=[WindowPresetEnum.肺窗],
                )
            ],
        )
        localization_result = {
            "body_part": "肺部",
            "cut_mode": "whole_range",
            "correction_scope_bbox": {
                "z_min": 0,
                "z_max": 10,
                "y_min": 20,
                "y_max": 80,
                "x_min": 30,
                "x_max": 90,
            },
            "image": {
                "spacing_xyz": [1.0, 1.0, 1.0],
                "origin_xyz": [0.0, 0.0, 0.0],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
        }

        result = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        )

        plan = result.plans[0]
        self.assertEqual(plan.range_start.z, 2.0)
        self.assertEqual(plan.range_end.z, 8.0)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            2.0,
            7.0,
        ])

    def test_axial_whole_range_short_target_uses_midpoint(self):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=[WindowPresetEnum.肺窗],
                )
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["correction_scope_bbox"]["z_min"] = 4
        localization_result["correction_scope_bbox"]["z_max"] = 6

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.range_start.z, 5.0)
        self.assertEqual(plan.range_end.z, 5.0)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [5.0])

    def test_lumbar_whole_range_planner_expands_slices_from_l1_to_s1(self):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=[WindowPresetEnum.骨窗],
                )
            ],
        )
        localization_result = self._lumbar_localization_result()

        result = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        )

        self.assertEqual(len(result.plans), 1)
        plan = result.plans[0]
        self.assertEqual(plan.plan_id, "腰椎_whole_range_横状位")
        self.assertEqual(len(plan.slices), 2)
        self.assertEqual(plan.slices[0].center.z, 2.0)
        self.assertEqual(plan.slices[-1].center.z, 7.0)

    def test_lumbar_whole_range_axial_uses_outer_bbox_z_range(self):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=[WindowPresetEnum.骨窗],
                )
            ],
        )
        localization_result = self._lumbar_localization_result()
        # 椎体质心仍是 0 到 10，但真实目标最外侧范围更大。
        # 横状位整切应该覆盖最外侧 bbox，而不是只切到端点质心。
        localization_result["correction_scope_bbox"]["z_min"] = -2
        localization_result["correction_scope_bbox"]["z_max"] = 12

        result = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        )

        plan = result.plans[0]
        self.assertEqual(plan.range_start.z, 0.0)
        self.assertEqual(plan.range_end.z, 10.0)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            0.0,
            5.0,
            10.0,
        ])

    def test_lumbar_whole_range_axial_uses_bbox_even_without_endpoint_centroids(self):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=[WindowPresetEnum.骨窗],
                )
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["coords"] = {}

        result = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        )

        self.assertEqual(result.warnings, [])
        self.assertEqual(len(result.plans), 1)
        plan = result.plans[0]
        self.assertEqual(plan.range_start.z, 2.0)
        self.assertEqual(plan.range_end.z, 8.0)
        self.assertEqual(plan.source_labels, ["correction_scope_bbox"])

    def test_lumbar_interspace_planner_builds_one_plan_per_available_interval(self):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.椎间切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=2.0,
                    slice_spacing=2.0,
                    window_presets=[WindowPresetEnum.骨窗],
                )
            ],
        )
        localization_result = self._lumbar_localization_result()

        result = LumbarInterspacePlanner().build_plans(
            request,
            localization_result,
        )

        self.assertEqual([plan.group_name for plan in result.plans], [
            "L1-L2",
            "L2-L3",
            "L3-L4",
            "L4-L5",
            "L5-S1",
        ])
        self.assertTrue(all(plan.slices for plan in result.plans))

    def test_mapping_stage_writes_slice_plans_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.腰椎,
                cut_mode=CutModeEnum.整切,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=[WindowPresetEnum.骨窗],
                    )
                ],
            )
            context = ProcessingContext(request)
            context.set_result("localization", self._lumbar_localization_result())

            MappingStage().execute(context)

            result = context.get_result("mapping")
            metadata_path = Path(result["metadata_path"])
            self.assertTrue(metadata_path.exists())
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(data["plan_count"], 1)
            self.assertEqual(data["plans"][0]["plan_id"], "腰椎_whole_range_横状位")
            self.assertIn("slices", data["plans"][0])
            self.assertEqual(data["diagnostics"][0]["slice_count"], 2)
            self.assertEqual(data["diagnostics"][0]["slice_thickness"], 5.0)
            self.assertEqual(data["diagnostics"][0]["slice_spacing"], 5.0)
            self.assertEqual(data["diagnostics"][0]["spacing_minus_thickness"], 0.0)
            self.assertIn("按层间距", data["diagnostics"][0]["calculation_note"])
            self.assertEqual(data["diagnostics"][0]["actual_spacing_min_mm"], 5.0)
            self.assertEqual(data["diagnostics"][0]["actual_spacing_max_mm"], 5.0)
            self.assertFalse(data["diagnostics"][0]["has_nonuniform_spacing"])
            self.assertFalse(data["diagnostics"][0]["has_extra_endpoint_slice"])
            self.assertEqual(data["diagnostics"][0]["leading_uncovered_mm"], 0.0)
            self.assertEqual(data["diagnostics"][0]["trailing_uncovered_mm"], 1.0)
            self.assertEqual(data["diagnostics"][0]["target_length_mm"], 11.0)
            self.assertFalse(data["diagnostics"][0]["target_shorter_than_thickness"])
            self.assertEqual(data["diagnostics"][0]["first_coverage_start"]["z"], -0.5)
            self.assertEqual(data["diagnostics"][0]["last_coverage_end"]["z"], 9.5)

    def test_mapping_stage_writes_slice_plan_line_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))

            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.腰椎,
                cut_mode=CutModeEnum.整切,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=[WindowPresetEnum.骨窗],
                    )
                ],
            )
            localization_result = self._lumbar_localization_result()
            localization_result["correction_scope_bbox"]["z_min"] = 3
            localization_result["correction_scope_bbox"]["z_max"] = 13
            localization_result["seg_for_reconstruction_path"] = str(seg_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overlay_path = Path(result["debug_slice_plan_overlay_path"])
            overlay_image = sitk.ReadImage(str(overlay_path))
            overlay_array = sitk.GetArrayFromImage(overlay_image)
            self.assertTrue(overlay_path.exists())
            self.assertEqual(overlay_image.GetSize(), image.GetSize())
            # 11 表示第一组 plan 的中心路径线，12 表示切片中心点。
            # 13 表示每张 slice 的平面引导线，应该离中心点有一段距离也能看到。
            self.assertEqual(overlay_array[5, 50, 60], 12)
            self.assertEqual(overlay_array[5, 50, 65], 13)

    def test_mapping_stage_writes_single_layer_overview_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))

            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.腰椎,
                cut_mode=CutModeEnum.整切,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=[WindowPresetEnum.骨窗],
                    )
                ],
            )
            localization_result = self._lumbar_localization_result()
            localization_result["correction_scope_bbox"]["z_min"] = 3
            localization_result["correction_scope_bbox"]["z_max"] = 13
            localization_result["seg_for_reconstruction_path"] = str(seg_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overview_path = Path(result["debug_slice_plan_overview_overlay_path"])
            overview_image = sitk.ReadImage(str(overview_path))
            overview_array = sitk.GetArrayFromImage(overview_image)
            self.assertTrue(overview_path.exists())
            self.assertEqual(overview_image.GetSpacing(), (1.0, 1.0, 0.5))
            self.assertEqual(overview_image.GetSize(), (80, 80, 35))
            # 5 mm 中心间距在 0.5 mm 检查网格上恒为 10 个体素。
            center_rows = np.unique(np.argwhere(overview_array == 14)[:, 0])
            self.assertEqual(center_rows.tolist(), [10, 20])
            self.assertEqual(int(center_rows[1] - center_rows[0]), 10)
            native_path = Path(result["debug_slice_plan_native_overview_overlay_path"])
            native_image = sitk.ReadImage(str(native_path))
            self.assertEqual(native_image.GetSize(), image.GetSize())

    def test_mapping_stage_overview_draws_thickness_bands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))

            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.腰椎,
                cut_mode=CutModeEnum.整切,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=3.0,
                        slice_spacing=5.0,
                        window_presets=[WindowPresetEnum.骨窗],
                    )
                ],
            )
            localization_result = self._lumbar_localization_result()
            localization_result["correction_scope_bbox"]["z_min"] = 3
            localization_result["correction_scope_bbox"]["z_max"] = 13
            localization_result["seg_for_reconstruction_path"] = str(seg_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overview_path = Path(result["debug_slice_plan_overview_overlay_path"])
            overview_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overview_path)))
            # bbox 3..13、层厚 3、中心间距 5：中心在 4 和 9，
            # 对应两个厚度覆盖带，中间仍保留配置允许的空档。
            center_rows = np.unique(np.argwhere(overview_array == 14)[:, 0])
            self.assertEqual(center_rows.tolist(), [8, 18])
            # 3 mm 层厚在 0.5 mm 网格上固定占 6 行；中心仅为短刻度，
            # 不再把整条厚度带分成“黄色在上/黄色在下”。
            band_rows = np.unique(np.argwhere(overview_array == 15)[:, 0])
            self.assertEqual(band_rows.tolist(), [5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20])
            self.assertEqual(overview_array[8, 45, 60], 15)
            self.assertEqual(overview_array[8, 50, 60], 14)
            self.assertEqual(overview_array[12, 50, 60], 0)

    def test_mapping_stage_overview_overlay_respects_image_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 100.0))
            image.SetDirection((
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
            ))
            sitk.WriteImage(image, str(seg_path))

            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.腰椎,
                cut_mode=CutModeEnum.整切,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=[WindowPresetEnum.骨窗],
                    )
                ],
            )
            localization_result = self._lumbar_localization_result()
            localization_result["image"]["origin_xyz"] = [0.0, 0.0, 100.0]
            localization_result["image"]["direction"] = [
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
            ]
            localization_result["correction_scope_bbox"]["z_min"] = 3
            localization_result["correction_scope_bbox"]["z_max"] = 13
            localization_result["seg_for_reconstruction_path"] = str(seg_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overview_path = Path(result["debug_slice_plan_overview_overlay_path"])
            overview_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overview_path)))
            self.assertEqual(overview_array[10, 50, 60], 14)
            self.assertEqual(overview_array[20, 50, 60], 14)

    @staticmethod
    def _lumbar_localization_result():
        def coord(name, z):
            return {
                "label_name": f"vertebrae_{name}",
                "label_ids": [1],
                "voxel_count": 10,
                "found": True,
                "centroid": {"z": z, "y": 50.0, "x": 60.0},
                "bbox": {
                    "z_min": int(z),
                    "z_max": int(z),
                    "y_min": 45,
                    "y_max": 55,
                    "x_min": 55,
                    "x_max": 65,
                },
            }

        return {
            "body_part": "腰椎",
            "cut_mode": "whole_range",
            "correction_scope_bbox": {
                "z_min": 0,
                "z_max": 10,
                "y_min": 45,
                "y_max": 55,
                "x_min": 55,
                "x_max": 65,
            },
            "image": {
                "spacing_xyz": [1.0, 1.0, 1.0],
                "origin_xyz": [0.0, 0.0, 0.0],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "coords": {
                "vertebrae_L1": coord("L1", 0.0),
                "vertebrae_L2": coord("L2", 2.0),
                "vertebrae_L3": coord("L3", 4.0),
                "vertebrae_L4": coord("L4", 6.0),
                "vertebrae_L5": coord("L5", 8.0),
                "vertebrae_S1": coord("S1", 10.0),
            },
        }


if __name__ == "__main__":
    unittest.main()
