import json
import tempfile
import unittest
from pathlib import Path

import SimpleITK as sitk
import numpy as np

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Request, WindowPresetEnum
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
        self.assertEqual(len(plan.slices), 4)
        self.assertEqual(plan.slices[1].center.to_dict(), {
            "x": 60.0,
            "y": 50.0,
            "z": 2.5,
        })
        self.assertEqual(plan.window_presets, ["肺窗"])

    def test_axial_whole_range_expands_center_range_by_half_thickness(self):
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
        self.assertEqual(plan.range_start.z, -2.5)
        self.assertEqual(plan.range_end.z, 12.5)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            -2.5,
            2.5,
            7.5,
            12.5,
        ])

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
        self.assertEqual(len(plan.slices), 4)
        self.assertEqual(plan.slices[0].center.z, -2.5)
        self.assertEqual(plan.slices[-1].center.z, 12.5)

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
        self.assertEqual(plan.range_start.z, -4.5)
        self.assertEqual(plan.range_end.z, 14.5)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            -4.5,
            0.5,
            5.5,
            10.5,
            14.5,
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
        self.assertEqual(plan.range_start.z, -2.5)
        self.assertEqual(plan.range_end.z, 12.5)
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
            self.assertEqual(data["diagnostics"][0]["slice_count"], 4)
            self.assertEqual(data["diagnostics"][0]["slice_thickness"], 5.0)
            self.assertEqual(data["diagnostics"][0]["slice_spacing"], 5.0)
            self.assertEqual(data["diagnostics"][0]["spacing_minus_thickness"], 0.0)
            self.assertIn("按层间距", data["diagnostics"][0]["calculation_note"])

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
            self.assertEqual(overlay_array[3, 50, 60], 11)
            self.assertEqual(overlay_array[6, 50, 60], 12)
            self.assertEqual(overlay_array[6, 50, 65], 13)

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
            self.assertEqual(overview_image.GetSize(), image.GetSize())
            # 14 表示第一组 plan 的单层总览线。所有横状位切片都画在 x=60 这个矢状位层上。
            self.assertEqual(overview_array[1, 45, 60], 14)
            self.assertEqual(overview_array[6, 50, 60], 14)
            self.assertEqual(overview_array[11, 55, 60], 14)
            self.assertEqual(overview_array[16, 50, 60], 14)
            self.assertEqual(overview_array[6, 50, 61], 0)

    def test_mapping_stage_overview_overlay_only_draws_center_lines(self):
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
            overview_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overview_path)))
            # 总览图只画每张输出图的中心线。厚层覆盖范围用 slice_plans.json
            # 里的 diagnostics 判断，不再叠加边界线，避免人工看图时更乱。
            self.assertEqual(overview_array[1, 50, 60], 14)
            self.assertEqual(overview_array[6, 50, 60], 14)
            self.assertEqual(overview_array[11, 50, 60], 14)
            self.assertEqual(overview_array[16, 50, 60], 14)
            self.assertEqual(int((overview_array == 15).sum()), 0)

    def test_mapping_stage_overview_overlay_respects_image_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 100.0))
            image.SetDirection((
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
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
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
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
            self.assertEqual(overview_array[1, 45, 60], 14)
            self.assertEqual(overview_array[6, 50, 60], 14)
            self.assertEqual(overview_array[11, 55, 60], 14)
            self.assertEqual(overview_array[16, 50, 60], 14)

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
