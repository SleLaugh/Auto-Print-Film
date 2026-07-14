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
from src.planning.whole_range import (
    build_whole_range_plan_from_bbox,
    plane_extent_mm,
)
from src.stages.stage_mapping import MappingStage


class Stage4SlicePlanningTest(unittest.TestCase):
    def test_overview_with_image_preserves_ct_and_highlights_overlay_labels(self):
        raw_array = np.array([
            [[0, 10], [20, 30]],
            [[40, 50], [60, 70]],
        ], dtype=np.int16)
        raw_image = sitk.GetImageFromArray(raw_array)
        raw_image.SetSpacing((1.0, 1.0, 1.0))

        reference_image = sitk.Image(2, 3, 2, sitk.sitkUInt16)
        reference_image.SetSpacing((1.0, 0.5, 1.0))
        overlay_array = np.zeros((2, 3, 2), dtype=np.uint16)
        overlay_array[0, 1, 0] = 15
        overlay_array[0, 1, 1] = 14

        result = MappingStage._build_overview_with_image(
            raw_image,
            reference_image,
            overlay_array,
        )
        result_array = sitk.GetArrayFromImage(result)

        self.assertEqual(result.GetSize(), reference_image.GetSize())
        self.assertEqual(result.GetPixelID(), sitk.sitkInt16)
        self.assertEqual(result.GetSpacing(), reference_image.GetSpacing())
        self.assertEqual(result.GetOrigin(), reference_image.GetOrigin())
        self.assertEqual(result.GetDirection(), reference_image.GetDirection())
        self.assertAlmostEqual(float(result_array[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(result_array[0, 2, 1]), 30.0)
        self.assertGreater(float(result_array[0, 1, 0]), 70.0)
        self.assertGreater(
            float(result_array[0, 1, 1]),
            float(result_array[0, 1, 0]),
        )

    def test_debug_filename_suffix_uses_short_plane_and_cut_mode_labels(self):
        self.assertEqual(
            MappingStage._debug_plan_suffix([
                {"plane": "冠状位", "cut_mode": "interspace"},
            ]),
            "冠_区",
        )

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
            "z": 3.0,
        })
        self.assertEqual(plan.window_presets, ["肺窗"])

    def test_single_label_planner_builds_sagittal_whole_range_plan(self):
        """单目标整切 planner 应把矢状位交给通用直线构造器。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            cut_mode=CutModeEnum.整切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.矢状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=[WindowPresetEnum.肺窗],
            )],
        )

        result = SingleLabelWholeRangePlanner().build_plans(
            request,
            self._lumbar_localization_result(),
        )

        self.assertEqual(len(result.plans), 1)
        self.assertEqual(result.plans[0].plane, "矢状位")
        self.assertEqual(result.warnings, [])

    def test_whole_range_builder_builds_all_straight_planes(self):
        """通用整切构造器应按请求顺序生成横、矢、冠三种默认直线几何。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.横状位,
                    3.0,
                    5.0,
                    [WindowPresetEnum.肺窗],
                ),
                PlaneConfig(
                    PlaneTypeEnum.矢状位,
                    3.0,
                    5.0,
                    [WindowPresetEnum.肺窗],
                ),
                PlaneConfig(
                    PlaneTypeEnum.冠状位,
                    3.0,
                    5.0,
                    [WindowPresetEnum.肺窗],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]
        localization_result["image"]["size_xyz"] = [100, 120, 30]

        plans = []
        for plane_config in request.plane_configs:
            try:
                plans.append(build_whole_range_plan_from_bbox(
                    request,
                    plane_config,
                    localization_result,
                ))
            except ValueError as error:
                self.fail(
                    f"默认直线整切不应拒绝{plane_config.plane.value}: {error}"
                )
        axial, sagittal, coronal = plans

        self.assertEqual(
            [axial.plane, sagittal.plane, coronal.plane],
            ["横状位", "矢状位", "冠状位"],
        )
        self.assertEqual(
            axial.output_spacing_mm.to_dict(),
            {"x": 1.0, "y": 2.0, "z": 0.0},
        )
        self.assertEqual(
            sagittal.output_spacing_mm.to_dict(),
            {"x": 2.0, "y": 3.0, "z": 0.0},
        )
        self.assertEqual(
            coronal.output_spacing_mm.to_dict(),
            {"x": 1.0, "y": 3.0, "z": 0.0},
        )
        self.assertEqual(
            sagittal.slices[0].normal.to_dict(),
            {"x": -1.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(
            sagittal.slices[0].col_dir.to_dict(),
            {"x": 0.0, "y": 0.0, "z": -1.0},
        )
        self.assertEqual(
            coronal.slices[0].normal.to_dict(),
            {"x": 0.0, "y": 1.0, "z": 0.0},
        )
        self.assertEqual(
            coronal.slices[0].col_dir.to_dict(),
            {"x": 0.0, "y": 0.0, "z": -1.0},
        )
        self.assertGreater(sagittal.range_start.x, sagittal.range_end.x)
        self.assertLess(coronal.range_start.y, coronal.range_end.y)

    def test_axial_whole_range_orders_slices_from_superior_to_inferior(self):
        """横状位应从头侧向足侧输出，同时保持单张图像的方向不变。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=[WindowPresetEnum.骨窗],
            )],
        )

        plan = build_whole_range_plan_from_bbox(
            request,
            request.plane_configs[0],
            self._lumbar_localization_result(),
        )

        # 测试数据采用单位方向矩阵，物理 z 越大越靠头侧。
        self.assertGreater(plan.range_start.z, plan.range_end.z)
        self.assertGreater(plan.slices[0].center.z, plan.slices[-1].center.z)
        # 序列倒序不能连带翻转单张横状位图像及其 DICOM 平面法向。
        self.assertEqual(
            plan.slices[0].row_dir.to_dict(),
            {"x": 1.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(
            plan.slices[0].col_dir.to_dict(),
            {"x": 0.0, "y": 1.0, "z": 0.0},
        )
        self.assertEqual(
            plan.slices[0].normal.to_dict(),
            {"x": 0.0, "y": 0.0, "z": 1.0},
        )

    def test_whole_range_target_extent_uses_plane_axes(self):
        """部位范围模式应选取当前平面内的两个 bbox 轴。"""

        localization_result = self._lumbar_localization_result()
        localization_result["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]
        bbox = localization_result["correction_scope_bbox"]
        image_info = localization_result["image"]
        expected_extents = {
            PlaneTypeEnum.横状位: {"width": 10.0, "height": 20.0},
            PlaneTypeEnum.矢状位: {"width": 20.0, "height": 30.0},
            PlaneTypeEnum.冠状位: {"width": 10.0, "height": 30.0},
        }

        for plane, expected in expected_extents.items():
            with self.subTest(plane=plane.value):
                extent = plane_extent_mm(
                    bbox,
                    image_info,
                    plane,
                    CutLineExtentModeEnum.部位范围,
                )
                self.assertEqual(extent.to_dict(), expected)

    def test_projected_extent_uses_selected_region_and_plane_frame(self):
        """直线范围投影应根据整面或部位框返回对应宽高和中心。"""

        from src.planning.straight import (
            default_straight_frame,
            resolve_projected_extent,
        )

        localization_result = self._lumbar_localization_result()
        localization_result["image"]["size_xyz"] = [100, 120, 30]
        frame = default_straight_frame(
            PlaneTypeEnum.横状位,
            localization_result["image"],
        )

        full_extent = resolve_projected_extent(
            CutLineExtentModeEnum.整平面,
            localization_result,
            localization_result["correction_scope_bbox"],
            frame,
        )
        target_extent = resolve_projected_extent(
            CutLineExtentModeEnum.部位范围,
            localization_result,
            localization_result["correction_scope_bbox"],
            frame,
        )

        self.assertEqual(full_extent.extent_mm.to_dict(), {
            "width": 99.0,
            "height": 119.0,
        })
        self.assertEqual(full_extent.center_u, 49.5)
        self.assertEqual(full_extent.center_v, 59.5)
        self.assertEqual(target_extent.extent_mm.to_dict(), {
            "width": 10.0,
            "height": 10.0,
        })
        self.assertEqual(target_extent.center_u, 60.0)
        self.assertEqual(target_extent.center_v, 50.0)

    def test_straight_plan_recenters_full_plane_without_moving_slice_plane(self):
        """整面切应覆盖原图中心，同时保持目标区间对应的切片层面。"""

        from src.planning.straight import (
            CutSegment,
            build_straight_slice_plan,
            default_straight_frame,
        )
        from src.planning.whole_range import bbox_target_points

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=[WindowPresetEnum.骨窗],
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["image"]["size_xyz"] = [100, 120, 30]
        target_start, target_end = bbox_target_points(
            localization_result,
            PlaneTypeEnum.横状位,
        )
        segment = CutSegment(
            group_name="whole_range",
            target_start=target_start,
            target_end=target_end,
            target_region_bbox=localization_result["correction_scope_bbox"],
            source_labels=("correction_scope_bbox",),
        )
        frame = default_straight_frame(
            PlaneTypeEnum.横状位,
            localization_result["image"],
        )

        plan = build_straight_slice_plan(
            request=request,
            plane_config=request.plane_configs[0],
            localization_result=localization_result,
            segment=segment,
            frame=frame,
        )

        self.assertEqual(plan.slices[0].center.x, 49.5)
        self.assertEqual(plan.slices[0].center.y, 59.5)
        self.assertEqual(plan.slices[0].center.z, 9.0)
        self.assertEqual(plan.extent_mm.to_dict(), {
            "width": 99.0,
            "height": 119.0,
        })

    def test_straight_plan_insets_thickness_along_final_tilted_normal(self):
        """未来加入倾角后，首末层中心仍应沿最终法向各内缩半层厚。"""

        from src.models import Point3D
        from src.planning.straight import (
            CutSegment,
            StraightPlaneFrame,
            build_straight_slice_plan,
        )

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.椎间切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=2.0,
                slice_spacing=2.0,
                window_presets=[WindowPresetEnum.骨窗],
                line_extent_mode=CutLineExtentModeEnum.部位范围,
            )],
        )
        localization_result = self._lumbar_localization_result()
        inverse_sqrt_two = 2 ** -0.5
        frame = StraightPlaneFrame(
            row_dir=Point3D(1.0, 0.0, 0.0),
            col_dir=Point3D(0.0, inverse_sqrt_two, -inverse_sqrt_two),
            normal=Point3D(0.0, inverse_sqrt_two, inverse_sqrt_two),
        )
        segment = CutSegment(
            group_name="L1-L2",
            target_start=Point3D(5.0, 5.0, 10.0),
            target_end=Point3D(5.0, 5.0, 0.0),
            target_region_bbox=localization_result["correction_scope_bbox"],
            source_labels=("vertebrae_L1", "vertebrae_L2"),
        )

        plan = build_straight_slice_plan(
            request=request,
            plane_config=request.plane_configs[0],
            localization_result=localization_result,
            segment=segment,
            frame=frame,
        )

        target_projection_length = 10.0 * inverse_sqrt_two
        planned_projection_length = (
            (plan.range_start.x - plan.range_end.x) * frame.normal.x
            + (plan.range_start.y - plan.range_end.y) * frame.normal.y
            + (plan.range_start.z - plan.range_end.z) * frame.normal.z
        )
        self.assertAlmostEqual(
            planned_projection_length,
            target_projection_length - 2.0,
        )

    def test_whole_range_builder_uses_projected_full_plane_center(self):
        """现有整切公开入口应接入通用直线范围投影流水线。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=[WindowPresetEnum.骨窗],
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["image"]["size_xyz"] = [100, 120, 30]

        plan = build_whole_range_plan_from_bbox(
            request,
            request.plane_configs[0],
            localization_result,
        )

        self.assertEqual(plan.slices[0].center.x, 49.5)
        self.assertEqual(plan.slices[0].center.y, 59.5)
        self.assertEqual(plan.slices[0].center.z, 9.0)

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
        self.assertEqual(plan.range_start.z, 8.0)
        self.assertEqual(plan.range_end.z, 2.0)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            8.0,
            3.0,
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
        self.assertEqual(plan.slices[0].center.z, 8.0)
        self.assertEqual(plan.slices[-1].center.z, 3.0)

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
        self.assertEqual(plan.range_start.z, 10.0)
        self.assertEqual(plan.range_end.z, 0.0)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            10.0,
            5.0,
            0.0,
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
        self.assertEqual(plan.range_start.z, 8.0)
        self.assertEqual(plan.range_end.z, 2.0)
        self.assertEqual(plan.source_labels, ["correction_scope_bbox"])

    def test_lumbar_whole_range_planner_builds_three_straight_planes(self):
        """腰椎整切也应复用通用构造器生成三个默认直线状位。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=plane,
                    slice_thickness=3.0,
                    slice_spacing=5.0,
                    window_presets=[WindowPresetEnum.骨窗],
                )
                for plane in (
                    PlaneTypeEnum.横状位,
                    PlaneTypeEnum.矢状位,
                    PlaneTypeEnum.冠状位,
                )
            ],
        )

        result = LumbarWholeRangePlanner().build_plans(
            request,
            self._lumbar_localization_result(),
        )

        self.assertEqual(
            [plan.plane for plan in result.plans],
            ["横状位", "矢状位", "冠状位"],
        )
        self.assertEqual(result.warnings, [])

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

    def test_lumbar_interspace_resolves_one_segment_per_available_gap(self):
        """腰椎区间解析应使用相邻椎体相向 bbox 表面生成独立区间。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        coords = localization_result["coords"]
        coords["vertebrae_L1"]["bbox"].update({"z_min": 9, "z_max": 11})
        coords["vertebrae_L2"]["bbox"].update({"z_min": 5, "z_max": 7})
        coords["vertebrae_L3"]["bbox"].update({"z_min": 1, "z_max": 3})
        for label_name in ("vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            coords[label_name]["found"] = False

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual(
            [segment.group_name for segment in segments],
            ["L1-L2", "L2-L3"],
        )
        self.assertEqual(segments[0].target_start.z, 8.5)
        self.assertEqual(segments[0].target_end.z, 7.5)
        self.assertEqual(
            segments[0].source_labels,
            ("vertebrae_L1", "vertebrae_L2"),
        )
        self.assertEqual(segments[0].target_region_bbox["z_min"], 5)
        self.assertEqual(segments[0].target_region_bbox["z_max"], 11)
        self.assertTrue(any("L3-L4" in warning for warning in warnings))

    def test_lumbar_interspace_target_extent_uses_current_label_pair(self):
        """区间范围切应使用当前相邻椎体 bbox 并集构造横状位计划。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.椎间切,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=0.5,
                slice_spacing=1.0,
                window_presets=[WindowPresetEnum.骨窗],
                line_extent_mode=CutLineExtentModeEnum.部位范围,
            )],
        )
        localization_result = self._lumbar_localization_result()
        coords = localization_result["coords"]
        coords["vertebrae_L1"]["bbox"].update({
            "x_min": 20, "x_max": 30,
            "y_min": 30, "y_max": 40,
            "z_min": 9, "z_max": 11,
        })
        coords["vertebrae_L2"]["bbox"].update({
            "x_min": 15, "x_max": 35,
            "y_min": 25, "y_max": 45,
            "z_min": 5, "z_max": 7,
        })
        coords["vertebrae_L1"]["centroid"].update({"x": 25.0, "y": 35.0})
        coords["vertebrae_L2"]["centroid"].update({"x": 25.0, "y": 35.0})
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            coords[label_name]["found"] = False

        result = LumbarInterspacePlanner().build_plans(
            request,
            localization_result,
        )

        self.assertEqual(len(result.plans), 1)
        plan = result.plans[0]
        self.assertEqual(plan.group_name, "L1-L2")
        self.assertEqual(plan.extent_mm.to_dict(), {
            "width": 20.0,
            "height": 20.0,
        })
        self.assertEqual(plan.slices[0].center.x, 25.0)
        self.assertEqual(plan.slices[0].center.y, 35.0)

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
            self.assertEqual(data["diagnostics"][0]["first_coverage_start"]["z"], 10.5)
            self.assertEqual(data["diagnostics"][0]["last_coverage_end"]["z"], 0.5)

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
            self.assertEqual(overlay_path.name, "切片线_横_整.nii.gz")
            overlay_image = sitk.ReadImage(str(overlay_path))
            overlay_array = sitk.GetArrayFromImage(overlay_image)
            self.assertTrue(overlay_path.exists())
            self.assertEqual(overlay_image.GetSize(), image.GetSize())
            # 11 表示第一组 plan 的中心路径线，12 表示切片中心点。
            # 13 表示每张 slice 的平面引导线，应该离中心点有一段距离也能看到。
            self.assertEqual(overlay_array[6, 50, 60], 12)
            self.assertEqual(overlay_array[6, 50, 65], 13)
            self.assertEqual(result["debug_slice_plan_overview_paths"], {})
            self.assertEqual(
                result["debug_slice_plan_overview_with_image_paths"],
                {},
            )
            self.assertEqual(
                set(result["debug_slice_plan_native_overview_paths"]),
                {"横状位"},
            )

    def test_mapping_stage_writes_separate_overviews_for_three_planes(self):
        """横、矢、冠应分别生成普通检查网格和原始网格总览。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            raw_path = temp_path / "raw.nii.gz"
            image = sitk.GetImageFromArray(
                np.zeros((18, 80, 80), dtype=np.uint8),
            )
            image.SetSpacing((1.0, 2.0, 3.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))
            raw_image = sitk.Cast(image, sitk.sitkInt16) + 100
            sitk.WriteImage(raw_image, str(raw_path))

            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.腰椎,
                cut_mode=CutModeEnum.整切,
                plane_configs=[
                    PlaneConfig(
                        plane=plane,
                        slice_thickness=3.0,
                        slice_spacing=5.0,
                        window_presets=[WindowPresetEnum.骨窗],
                    )
                    for plane in (
                        PlaneTypeEnum.横状位,
                        PlaneTypeEnum.矢状位,
                        PlaneTypeEnum.冠状位,
                    )
                ],
            )
            localization_result = self._lumbar_localization_result()
            localization_result["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]
            localization_result["seg_for_reconstruction_path"] = str(seg_path)
            localization_result["raw_for_reconstruction_path"] = str(raw_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overview_paths = result["debug_slice_plan_overview_paths"]
            overview_with_image_paths = result[
                "debug_slice_plan_overview_with_image_paths"
            ]
            native_paths = result["debug_slice_plan_native_overview_paths"]
            expected_planes = {"横状位", "矢状位", "冠状位"}
            self.assertEqual(set(overview_paths), expected_planes)
            self.assertEqual(overview_with_image_paths, overview_paths)
            self.assertEqual(set(native_paths), expected_planes)
            self.assertEqual(
                Path(native_paths["矢状位"]).name,
                "切片原始网格总览_矢_整.nii.gz",
            )
            self.assertEqual(
                Path(native_paths["冠状位"]).name,
                "切片原始网格总览_冠_整.nii.gz",
            )

            expected_spacing = {
                "横状位": (1.0, 2.0, 0.5),
                "矢状位": (0.5, 2.0, 3.0),
                "冠状位": (1.0, 0.5, 3.0),
            }
            for plane, path in overview_paths.items():
                with self.subTest(kind="uniform", plane=plane):
                    self.assertIn("切片总览含原图_", Path(path).name)
                    overview_image = sitk.ReadImage(path)
                    overview_array = sitk.GetArrayFromImage(overview_image)
                    self.assertEqual(
                        overview_image.GetSpacing(),
                        expected_spacing[plane],
                    )
                    self.assertTrue(np.any(overview_array > 0))

            self.assertFalse(any(
                temp_path.glob("metadata/切片总览_*.nii.gz")
            ))

            for plane, path in native_paths.items():
                with self.subTest(kind="native", plane=plane):
                    native_image = sitk.ReadImage(path)
                    native_array = sitk.GetArrayFromImage(native_image)
                    self.assertEqual(native_image.GetSize(), image.GetSize())
                    self.assertEqual(native_image.GetSpacing(), image.GetSpacing())
                    self.assertEqual(native_image.GetOrigin(), image.GetOrigin())
                    self.assertEqual(
                        native_image.GetDirection(),
                        image.GetDirection(),
                    )
                    self.assertTrue(np.any(native_array > 0))

            self.assertEqual(
                result["debug_slice_plan_overview_overlay_path"],
                overview_paths["横状位"],
            )
            self.assertEqual(
                result["debug_slice_plan_native_overview_overlay_path"],
                native_paths["横状位"],
            )
            metadata = json.loads(
                Path(result["metadata_path"]).read_text(encoding="utf-8"),
            )
            self.assertEqual(
                metadata["debug_slice_plan_overview_paths"],
                overview_paths,
            )
            self.assertEqual(
                metadata["debug_slice_plan_overview_with_image_paths"],
                overview_paths,
            )
            self.assertEqual(
                metadata["debug_slice_plan_native_overview_paths"],
                native_paths,
            )

    def test_mapping_stage_writes_single_layer_overview_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            raw_path = temp_path / "raw.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))
            sitk.WriteImage(sitk.Cast(image, sitk.sitkInt16), str(raw_path))

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
            localization_result["raw_for_reconstruction_path"] = str(raw_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overview_path = Path(result["debug_slice_plan_overview_overlay_path"])
            self.assertEqual(overview_path.name, "切片总览含原图_横_整.nii.gz")
            overview_image = sitk.ReadImage(str(overview_path))
            overview_array = sitk.GetArrayFromImage(overview_image)
            self.assertTrue(overview_path.exists())
            self.assertEqual(overview_image.GetSpacing(), (1.0, 1.0, 0.5))
            self.assertEqual(overview_image.GetSize(), (80, 80, 35))
            self.assertEqual(overview_image.GetPixelID(), sitk.sitkInt16)
            self.assertEqual(overview_image.GetOrigin(), (0.0, 0.0, 0.0))
            # 5 mm 中心间距在 0.5 mm 检查网格上恒为 10 个体素。
            center_rows = np.unique(np.argwhere(overview_array == 2)[:, 0])
            self.assertEqual(center_rows.tolist(), [12, 22])
            self.assertEqual(int(center_rows[1] - center_rows[0]), 10)
            native_path = Path(result["debug_slice_plan_native_overview_overlay_path"])
            self.assertEqual(native_path.name, "切片原始网格总览_横_整.nii.gz")
            native_image = sitk.ReadImage(str(native_path))
            self.assertEqual(native_image.GetSize(), image.GetSize())

    def test_mapping_stage_overview_draws_thickness_bands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            raw_path = temp_path / "raw.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))
            sitk.WriteImage(sitk.Cast(image, sitk.sitkInt16), str(raw_path))

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
            localization_result["raw_for_reconstruction_path"] = str(raw_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overview_path = Path(result["debug_slice_plan_overview_overlay_path"])
            overview_image = sitk.ReadImage(str(overview_path))
            overview_array = sitk.GetArrayFromImage(overview_image)
            # bbox 3..13、层厚 3、中心间距 5：从上向下的中心在 12 和 7，
            # 对应两个厚度覆盖带，中间仍保留配置允许的空档。
            center_rows = np.unique(np.argwhere(overview_array == 2)[:, 0])
            self.assertEqual(center_rows.tolist(), [14, 24])
            # 3 mm 层厚在 0.5 mm 网格上固定占 6 行；中心仅为短刻度，
            # 不再把整条厚度带分成“黄色在上/黄色在下”。
            band_rows = np.unique(np.argwhere(overview_array == 1)[:, 0])
            self.assertEqual(band_rows.tolist(), [
                11, 12, 13, 14, 15, 16,
                21, 22, 23, 24, 25, 26,
            ])
            band_index = MappingStage._physical_point_to_array_index(
                overview_image,
                {"x": 60.0, "y": 45.0, "z": 7.0},
            )
            center_index = MappingStage._physical_point_to_array_index(
                overview_image,
                {"x": 60.0, "y": 50.0, "z": 7.0},
            )
            gap_index = MappingStage._physical_point_to_array_index(
                overview_image,
                {"x": 60.0, "y": 50.0, "z": 9.5},
            )
            self.assertEqual(overview_array[band_index], 1)
            self.assertEqual(overview_array[center_index], 2)
            self.assertEqual(overview_array[gap_index], 0)

    def test_mapping_stage_overview_overlay_respects_image_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            raw_path = temp_path / "raw.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((18, 80, 80), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 100.0))
            image.SetDirection((
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
            ))
            sitk.WriteImage(image, str(seg_path))
            sitk.WriteImage(sitk.Cast(image, sitk.sitkInt16), str(raw_path))

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
            localization_result["raw_for_reconstruction_path"] = str(raw_path)
            context = ProcessingContext(request)
            context.set_result("localization", localization_result)

            MappingStage().execute(context)

            result = context.get_result("mapping")
            overview_path = Path(result["debug_slice_plan_overview_overlay_path"])
            overview_image = sitk.ReadImage(str(overview_path))
            overview_array = sitk.GetArrayFromImage(overview_image)
            center_rows = np.unique(np.argwhere(overview_array == 2)[:, 0])
            self.assertEqual(center_rows.tolist(), [10, 20])

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
                "vertebrae_L1": coord("L1", 10.0),
                "vertebrae_L2": coord("L2", 8.0),
                "vertebrae_L3": coord("L3", 6.0),
                "vertebrae_L4": coord("L4", 4.0),
                "vertebrae_L5": coord("L5", 2.0),
                "vertebrae_S1": coord("S1", 0.0),
            },
        }


if __name__ == "__main__":
    unittest.main()
