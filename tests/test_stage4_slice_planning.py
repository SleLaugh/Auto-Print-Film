import json
import tempfile
import unittest
from pathlib import Path

import SimpleITK as sitk
import numpy as np

from src.utils.Logger_utils import _logger_manager
_logger_manager.debug_log_file_enabled = True

from src.core.context import ProcessingContext
from src.models import PlaneConfig, Point3D, Request
from src.models.enums import CutLineExtentModeEnum
from src.models.enums import BodyPartEnum, CutModeEnum, PlaneTypeEnum
from src.planning.base import select_planner
from src.planning.lumbar import LumbarInterspacePlanner, LumbarWholeRangePlanner
from src.planning.single_label import SingleLabelWholeRangePlanner
from src.planning.spine_cpr import (
    SPINE_CPR_LABELS,
    _resolve_endpoint_face_center,
)
from src.planning.whole_range import (
    build_whole_range_plan_from_bbox,
    plane_extent_mm,
)
from src.stages.stage_mapping import MappingStage


class Stage4SlicePlanningTest(unittest.TestCase):
    """汇总 `Stage4SlicePlanningTest` 相关测试场景。"""

    @staticmethod
    def _max_line_fit_residual(points_zyx: np.ndarray) -> float:
        """返回二维高亮点到最佳直线的最大像素残差。"""

        rows = points_zyx[:, 0].astype(float)
        columns = points_zyx[:, 1].astype(float)
        matrix = np.column_stack([rows, np.ones_like(rows)])
        slope, intercept = np.linalg.lstsq(matrix, columns, rcond=None)[0]
        fitted = slope * rows + intercept
        return float(np.max(np.abs(columns - fitted)))

    def test_mapping_derives_lumbar_axial_interspace_and_sagittal_straight_mpr(self):
        """腰椎横状位保持区间切基线，矢状位回到普通直线 MPR。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.腰椎,
                plane_configs=[
                    PlaneConfig(
                        PlaneTypeEnum.横状位, 1.0, 3.0,
                        ["骨窗"],
                    ),
                    PlaneConfig(
                        PlaneTypeEnum.矢状位, 1.0, 3.0,
                        ["骨窗"],
                    ),
                ],
            )
            context = ProcessingContext(request)
            context.set_result("localization", self._lumbar_localization_result())

            MappingStage().execute(context)

            plans = context.get_result("mapping")["plans"]
            self.assertEqual(
                [
                    (
                        plan["plane"],
                        plan["cut_mode"],
                        plan["geometry_type"],
                    )
                    for plan in plans
                ],
                [
                    ("横状位", "interspace", "plane_stack"),
                    ("横状位", "interspace", "plane_stack"),
                    ("横状位", "interspace", "plane_stack"),
                    ("横状位", "interspace", "plane_stack"),
                    ("横状位", "interspace", "plane_stack"),
                    ("矢状位", "whole_range", "plane_stack"),
                ],
            )
            sagittal_plan = plans[-1]
            self.assertEqual(sagittal_plan["group_name"], "whole_range")
            self.assertEqual(sagittal_plan["localizer_plane"], "冠状位")
            self.assertNotIn("curve_points", sagittal_plan)
            self.assertNotIn("offsets_mm", sagittal_plan)

    def test_spine_coronal_cpr_falls_back_when_fewer_than_three_labels(self):
        """有效椎体少于 3 个时，冠状位 CPR 应降级为现有直线 MPR。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            localization_result["coords"][label_name]["found"] = False

        result = LumbarWholeRangePlanner().build_plans(request, localization_result)

        self.assertEqual(result.plans[0].geometry_type, "plane_stack")
        self.assertIn("CPR 降级", result.warnings[0])

    def test_spine_cpr_extends_to_first_and_last_vertebra_end_faces(self):
        """缺少主体端面几何时，CPR 首尾应回退到首末标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["coords"]["vertebrae_L1"]["centroid"]["z"] = 10.0
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 8,
            "z_max": 13,
        })
        localization_result["coords"]["vertebrae_S1"]["centroid"]["z"] = 0.0
        localization_result["coords"]["vertebrae_S1"]["bbox"].update({
            "z_min": -3,
            "z_max": 2,
        })

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.geometry_type, "curved_stack")
        self.assertAlmostEqual(plan.curve_points[0].z, 10.0)
        self.assertAlmostEqual(plan.curve_points[-1].z, 0.0)
        self.assertAlmostEqual(plan.range_start.z, 10.0)
        self.assertAlmostEqual(plan.range_end.z, 0.0)

    def test_spine_cpr_endpoint_faces_extend_vertically_from_endpoint_centroids(self):
        """回退路径的首尾不应被全量 bbox 端面拉出标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["coords"]["vertebrae_L1"]["centroid"].update({
            "z": 40.0, "y": 10.0, "x": 60.0,
        })
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 36, "z_max": 44,
            "y_min": 8, "y_max": 12,
            "x_min": 55, "x_max": 65,
        })
        localization_result["coords"]["vertebrae_L2"]["centroid"].update({
            "z": 30.0, "y": 20.0, "x": 60.0,
        })
        localization_result["coords"]["vertebrae_L3"]["centroid"].update({
            "z": 20.0, "y": 30.0, "x": 60.0,
        })
        localization_result["coords"]["vertebrae_L4"]["centroid"].update({
            "z": 10.0, "y": 40.0, "x": 60.0,
        })
        localization_result["coords"]["vertebrae_L5"]["centroid"].update({
            "z": 0.0, "y": 50.0, "x": 60.0,
        })
        localization_result["coords"]["vertebrae_S1"]["centroid"].update({
            "z": -10.0, "y": 80.0, "x": 60.0,
        })
        localization_result["coords"]["vertebrae_S1"]["bbox"].update({
            "z_min": -18, "z_max": -6,
            "y_min": 78, "y_max": 82,
            "x_min": 55, "x_max": 65,
        })

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.geometry_type, "curved_stack")
        self.assertAlmostEqual(plan.curve_points[0].z, 40.0)
        self.assertAlmostEqual(plan.curve_points[0].y, 10.0)
        self.assertAlmostEqual(plan.curve_points[-1].z, -10.0)
        self.assertAlmostEqual(plan.curve_points[-1].y, 80.0)

    def test_spine_cpr_endpoint_prefers_face_aligned_with_adjacent_centroid_line(self):
        """端点应优先选择与相邻质心连线同向的端面候选。"""

        annotation = {
            "found": True,
            "local_superior_inferior_face_centroids": {
                "min": {"x": -2.0, "y": 5.0, "z": 0.0},
                "max": {"x": 2.0, "y": 15.0, "z": 0.0},
            },
            "local_superior_inferior_face_candidates": [
                {
                    "side": "max",
                    "projection": 2.0,
                    "physical": {"x": 0.0, "y": 13.0, "z": 0.0},
                },
                {
                    "side": "min",
                    "projection": -2.0,
                    "physical": {"x": 0.0, "y": 7.0, "z": 0.0},
                },
                {
                    "side": "max",
                    "projection": 3.0,
                    "physical": {"x": 2.0, "y": 15.0, "z": 0.0},
                },
                {
                    "side": "min",
                    "projection": -3.0,
                    "physical": {"x": -2.0, "y": 5.0, "z": 0.0},
                },
            ],
        }

        endpoint = _resolve_endpoint_face_center(
            annotation,
            Point3D(0.0, 10.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
            toward_start=False,
        )

        self.assertEqual(endpoint, Point3D(2.0, 15.0, 0.0))
        start_endpoint = _resolve_endpoint_face_center(
            annotation,
            Point3D(0.0, 10.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
            toward_start=True,
        )

        self.assertEqual(start_endpoint, Point3D(-2.0, 5.0, 0.0))

        annotation["local_superior_inferior_face_candidates"][2][
            "face_valid"
        ] = False
        annotation["local_superior_inferior_face_candidates"][3][
            "face_valid"
        ] = False
        endpoint_on_real_face = _resolve_endpoint_face_center(
            annotation,
            Point3D(0.0, 10.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
            toward_start=False,
        )

        self.assertEqual(endpoint_on_real_face, Point3D(0.0, 13.0, 0.0))

    def test_spine_cpr_endpoint_rejects_sparse_terminal_corner_candidate(self):
        """终末端稀薄小片不应覆盖真实端面中心。"""

        annotation = {
            "found": True,
            "local_superior_inferior_face_candidates": [
                {
                    "side": "min",
                    "projection": -16.5,
                    "point_count": 238,
                    "face_valid": True,
                    "transverse_span_mm": {
                        "first": 4.6,
                        "second": 10.0,
                    },
                    "physical": {"x": 0.0, "y": 0.0, "z": -16.5},
                },
                {
                    "side": "min",
                    "projection": -13.8,
                    "point_count": 637,
                    "face_valid": True,
                    "transverse_span_mm": {
                        "first": 10.2,
                        "second": 23.1,
                    },
                    "physical": {"x": 0.0, "y": 0.0, "z": -13.8},
                },
                {
                    "side": "min",
                    "projection": -12.5,
                    "point_count": 1044,
                    "face_valid": True,
                    "transverse_span_mm": {
                        "first": 13.4,
                        "second": 28.4,
                    },
                    "physical": {"x": 0.0, "y": 0.0, "z": -12.5},
                },
            ],
        }

        endpoint = _resolve_endpoint_face_center(
            annotation,
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, 0.0, -1.0),
            toward_start=False,
        )

        self.assertEqual(endpoint, Point3D(0.0, 0.0, -13.8))

    def test_spine_cpr_endpoint_uses_body_face_opposite_to_incoming_centroid_line(self):
        """有主体四面几何时，端点应取质心入射面的对侧大面中心。"""

        annotation = {
            "found": True,
            "sagittal_body_faces": [
                {
                    "axis_index": 0,
                    "side": "min",
                    "point_count": 2000,
                    "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "center": {"x": 0.0, "y": 0.0, "z": 4.0},
                },
                {
                    "axis_index": 0,
                    "side": "max",
                    "point_count": 1800,
                    "normal": {"x": 0.0, "y": 0.2, "z": -0.98},
                    "center": {"x": 0.0, "y": 4.0, "z": -14.0},
                },
                {
                    "axis_index": 1,
                    "side": "max",
                    "point_count": 80,
                    "normal": {"x": 0.0, "y": 0.01, "z": -0.99995},
                    "center": {"x": 0.0, "y": 20.0, "z": -16.0},
                },
            ],
            "local_superior_inferior_face_candidates": [
                {
                    "side": "min",
                    "projection": -16.0,
                    "point_count": 80,
                    "face_valid": True,
                    "physical": {"x": 0.0, "y": 20.0, "z": -16.0},
                },
            ],
        }

        endpoint = _resolve_endpoint_face_center(
            annotation,
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, 0.0, -1.0),
            toward_start=False,
        )

        self.assertEqual(endpoint, Point3D(0.0, 4.0, -14.0))

    def test_spine_cpr_connects_annotation_centroids(self):
        """基准 CPR 只按解剖顺序连接过滤标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        for label_name, centroid_y in {
            "vertebrae_L1": 52.0,
            "vertebrae_L2": 56.0,
            "vertebrae_L3": 60.0,
        }.items():
            coord = localization_result["coords"][label_name]
            center_z = coord["centroid"]["z"]
            coord["bbox"].update({
                "z_min": int(center_z) - 1,
                "z_max": int(center_z) + 1,
            })
            localization_result.setdefault("spine_geometry", {}).setdefault(
                "filtered_annotation_centroids", {}
            )[label_name] = {
                "found": True,
                "voxel_count": 20,
                "centroid_voxel": {
                    "z": center_z,
                    "y": centroid_y,
                    "x": 60.0,
                },
                "centroid_physical": {
                    "x": 60.0,
                    "y": centroid_y,
                    "z": center_z,
                },
                "local_superior_inferior_axis": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                },
                "local_superior_inferior_extent_mm": {
                    "min": -1.0,
                    "max": 1.0,
                },
            }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.geometry_type, "curved_stack")
        self.assertEqual(len(plan.curve_points), 3)
        self.assertEqual(
            [point.y for point in plan.curve_points],
            [52.0, 56.0, 60.0],
        )
        for point, label_name in zip(
            plan.curve_points,
            ("vertebrae_L1", "vertebrae_L2", "vertebrae_L3"),
        ):
            annotation = localization_result[
                "spine_geometry"
            ]["filtered_annotation_centroids"][label_name]
            expected = annotation["centroid_physical"]
            self.assertAlmostEqual(point.x, expected["x"])
            self.assertAlmostEqual(point.y, expected["y"])
            self.assertAlmostEqual(point.z, expected["z"])

    def test_spine_cpr_baseline_ignores_local_segments(self):
        """基准 CPR 不使用椎体局部线段，避免端点造成折返。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        coord = localization_result["coords"]["vertebrae_L3"]
        coord["centroid"].update({"z": 6.0, "y": 50.0, "x": 60.0})
        coord["bbox"].update({
            "z_min": 4,
            "z_max": 8,
            "y_min": 48,
            "y_max": 52,
        })
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {
                label_name: {
                    "found": True,
                    "centroid_physical": {
                        "x": 60.0,
                        "y": 50.0,
                        "z": float(localization_result["coords"][label_name]["centroid"]["z"]),
                    },
                    "local_superior_inferior_axis": (
                        {"x": 0.0, "y": 0.70710678, "z": 0.70710678}
                        if label_name == "vertebrae_L3"
                        else {"x": 0.0, "y": 0.0, "z": 1.0}
                    ),
                    "local_superior_inferior_extent_mm": (
                        {"min": -1.0, "max": 1.0}
                        if label_name == "vertebrae_L3"
                        else {"min": -1.0, "max": 1.0}
                    ),
                }
                for label_name in (
                    "vertebrae_L1",
                    "vertebrae_L2",
                    "vertebrae_L3",
                    "vertebrae_L4",
                    "vertebrae_L5",
                    "vertebrae_S1",
                )
            },
        }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(len(plan.curve_points), 6)
        self.assertTrue(any(
            abs(point.x - 60.0) < 1e-6
            and abs(point.y - 50.0) < 1e-6
            and abs(point.z - 6.0) < 1e-6
            for point in plan.curve_points
        ))

    def test_spine_cpr_uses_body_face_centers_and_all_annotation_centroids(self):
        """CPR 锚点应包含首尾端面中点和全部过滤标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        filtered = {}
        for label_name, coord in localization_result["coords"].items():
            centroid = coord["centroid"]
            z = float(centroid["z"])
            filtered[label_name] = {
                "found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": 50.0,
                    "z": z,
                },
                "local_superior_inferior_face_centroids": {
                    "min": {"x": 60.0, "y": 50.0, "z": z - 1.5},
                    "max": {"x": 60.0, "y": 50.0, "z": z + 1.5},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": filtered,
        }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(len(plan.curve_points), 8)
        # 解剖顺序从头侧到足侧时，物理 z 是递减的，所以 min/max 需要按路径方向解释。
        self.assertEqual(
            [point.z for point in plan.curve_points],
            [11.5, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0, -1.5],
        )

    def test_spine_cpr_uses_sacrum_as_s1_alias(self):
        """数据集只有 sacrum 时，腰椎 CPR 仍应保留第六个标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["coords"]["sacrum"] = (
            localization_result["coords"].pop("vertebrae_S1")
        )
        filtered = {
            label_name: {
                "found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": 50.0,
                    "z": float(coord["centroid"]["z"]),
                },
            }
            for label_name, coord in localization_result["coords"].items()
        }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": filtered,
        }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(len(plan.curve_points), 6)
        self.assertEqual(plan.source_labels[-1], "sacrum")
        self.assertAlmostEqual(plan.curve_points[-1].z, 0.0)

    def test_spine_cpr_keeps_face_centers_and_all_centroids_as_direct_anchors(self):
        """首尾端面偏移时仍应保留端面中心和全部质心锚点。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        y_by_label = {
            "vertebrae_L1": 50.0,
            "vertebrae_L2": 49.0,
            "vertebrae_L3": 48.0,
            "vertebrae_L4": 48.0,
            "vertebrae_L5": 50.0,
            "vertebrae_S1": 53.0,
        }
        filtered = {}
        for label_name, coord in localization_result["coords"].items():
            z = float(coord["centroid"]["z"])
            y = y_by_label[label_name]
            filtered[label_name] = {
                "found": True,
                "centroid_physical": {"x": 60.0, "y": y, "z": z},
            }
        filtered["vertebrae_L1"]["sagittal_body_faces"] = [
            {
                "axis_index": 0,
                "side": "min",
                "point_count": 100,
                "center": {"x": 60.0, "y": 55.0, "z": 8.0},
                "normal": {"x": 0.0, "y": -0.4472, "z": -0.8944},
            },
            {
                "axis_index": 0,
                "side": "max",
                "point_count": 100,
                "center": {"x": 60.0, "y": 40.0, "z": 12.0},
                "normal": {"x": 0.0, "y": 0.4472, "z": 0.8944},
            },
        ]
        filtered["vertebrae_S1"]["sagittal_body_faces"] = [
            {
                "axis_index": 0,
                "side": "min",
                "point_count": 100,
                "center": {"x": 60.0, "y": 48.0, "z": 2.0},
                "normal": {"x": 0.0, "y": -0.8321, "z": 0.5547},
            },
            {
                "axis_index": 0,
                "side": "max",
                "point_count": 100,
                "center": {"x": 60.0, "y": 62.0, "z": -2.0},
                "normal": {"x": 0.0, "y": 0.8321, "z": -0.5547},
            },
        ]
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": filtered,
        }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(
            plan.curve_points[0],
            Point3D(60.0, 40.0, 12.0),
        )
        self.assertEqual(
            plan.curve_points[-1],
            Point3D(60.0, 62.0, -2.0),
        )

        self.assertEqual(len(plan.curve_points), 8)
        self.assertEqual(
            plan.curve_points[1:-1],
            [
                Point3D(60.0, y_by_label[label_name], z)
                for label_name, z in (
                    ("vertebrae_L1", 10.0),
                    ("vertebrae_L2", 8.0),
                    ("vertebrae_L3", 6.0),
                    ("vertebrae_L4", 4.0),
                    ("vertebrae_L5", 2.0),
                    ("vertebrae_S1", 0.0),
                )
            ],
        )

    def test_spine_cpr_terminal_face_follows_last_annotation_centroid(self):
        """末端路径应由最后一个标注质心直接连接到对立端面中心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 1.0,
                    ["骨窗"],
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        y_by_label = {
            "vertebrae_L1": -98.0,
            "vertebrae_L2": -106.0,
            "vertebrae_L3": -115.0,
            "vertebrae_L4": -121.0,
            "vertebrae_L5": -123.0,
            "vertebrae_S1": -118.0,
        }
        filtered = {}
        for label_name, coord in localization_result["coords"].items():
            z = float(coord["centroid"]["z"])
            filtered[label_name] = {
                "found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": y_by_label[label_name],
                    "z": z,
                },
            }
        filtered["vertebrae_L1"]["local_superior_inferior_face_centroids"] = {
            "min": {"x": 60.0, "y": -98.0, "z": 11.5},
            "max": {"x": 60.0, "y": -98.0, "z": 8.5},
        }
        filtered["vertebrae_S1"]["local_superior_inferior_face_centroids"] = {
            "min": {"x": 60.0, "y": -110.0, "z": -1.5},
            "max": {"x": 60.0, "y": -110.0, "z": 1.5},
        }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": filtered,
        }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(
            plan.curve_points[-1],
            Point3D(60.0, -110.0, -1.5),
        )
        self.assertEqual(
            plan.curve_points[-2],
            Point3D(60.0, y_by_label["vertebrae_S1"], 0.0),
        )

    def test_cervical_sagittal_stays_straight_and_coronal_uses_cpr(self):
        """颈椎矢状位保持直线 MPR，冠状位走 CPR。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.颈椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.横状位, 1.0, 3.0,
                    ["骨窗"],
                ),
                PlaneConfig(
                    PlaneTypeEnum.矢状位, 1.0, 3.0,
                    ["骨窗"],
                ),
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                ),
            ],
        )

        result = SingleLabelWholeRangePlanner().build_plans(
            request,
            self._spine_localization_result(("C1", "C2", "C3", "C4"), "颈椎"),
        )

        self.assertEqual(
            [(plan.plane, plan.geometry_type) for plan in result.plans],
            [
                ("横状位", "plane_stack"),
                ("矢状位", "plane_stack"),
                ("冠状位", "curved_stack"),
            ],
        )
        self.assertEqual(
            result.plans[-1].source_labels,
            ["vertebrae_C2", "vertebrae_C3", "vertebrae_C4"],
        )

    def test_cervical_cpr_label_sequence_excludes_c1_ring(self):
        """颈椎 CPR 源序列从 C2 开始，避免 C1 环形结构干扰曲线。"""

        self.assertEqual(
            SPINE_CPR_LABELS["颈椎"],
            [f"vertebrae_C{index}" for index in range(2, 8)],
        )

    def test_overview_with_image_preserves_ct_and_highlights_overlay_labels(self):
        """验证 `test_overview_with_image_preserves_ct_and_highlights_overlay_labels` 场景符合预期。"""

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
        # 旧层厚带标签不再烧录，只高亮最细中心线。
        self.assertAlmostEqual(float(result_array[0, 1, 0]), 10.0)
        self.assertGreater(float(result_array[0, 1, 1]), 70.0)

    def test_debug_filename_suffix_uses_short_plane_and_cut_mode_labels(self):
        """验证 `test_debug_filename_suffix_uses_short_plane_and_cut_mode_labels` 场景符合预期。"""

        self.assertEqual(
            MappingStage._debug_plan_suffix([
                {"plane": "冠状位", "cut_mode": "interspace"},
            ]),
            "冠_区",
        )

    def test_select_planner_supports_configured_whole_range_body_parts(self):
        """验证 `test_select_planner_supports_configured_whole_range_body_parts` 场景符合预期。"""

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

    def test_checked_in_range_extension_keeps_configured_defaults(self):
        """验证当前默认值，不限制后续按需增加其他部位或状位。"""

        config_path = Path("src/config/localization_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for body_part in ("肺部", "髋骨"):
            with self.subTest(body_part=body_part):
                whole_range = config["targets"][body_part]["whole_range"]
                self.assertEqual(whole_range["slice_range_extension"], {
                    "横状位": {"start": 3.0, "end": 3.0},
                })

    def test_single_label_whole_range_planner_builds_axial_plan_from_bbox(self):
        """验证 `test_single_label_whole_range_planner_builds_axial_plan_from_bbox` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=["肺窗"],
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
        self.assertEqual(
            [planned_slice.center.z for planned_slice in plan.slices],
            [12.5, 7.5, 2.5, -2.5],
        )
        self.assertEqual(plan.window_presets, ["肺窗"])

    def test_lung_axial_slice_centers_cover_real_mask_bounds(self):
        """真实肺部层间距和原点下，首末成片中心必须越过掩膜上下边界。"""

        spacing_z = 0.6499999761581421
        origin_z = -1511.0999755859375
        mask_min_index = 103.0
        mask_max_index = 522.0
        mask_min_z = origin_z + mask_min_index * spacing_z
        mask_max_z = origin_z + mask_max_index * spacing_z
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=["肺窗"],
                )
            ],
        )
        localization_result = {
            "body_part": "肺部",
            "cut_mode": "whole_range",
            "cut_configs": {
                "whole_range": {
                    "start": "lung",
                    "end": "lung",
                    "slice_range_extension": {
                        "横状位": {"start": 3.0, "end": 3.0},
                    },
                },
            },
            "correction_scope_bbox": {
                "z_min": 103,
                "z_max": 522,
                "y_min": 0,
                "y_max": 511,
                "x_min": 0,
                "x_max": 511,
            },
            "image": {
                "size_xyz": [512, 512, 556],
                "spacing_xyz": [1.0, 1.0, spacing_z],
                "origin_xyz": [0.0, 0.0, origin_z],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "anatomical_geometry": {
                "global_frame": {
                    "origin": {
                        "x": 255.5,
                        "y": 255.5,
                        "z": (mask_min_z + mask_max_z) / 2.0,
                    },
                    "left_right_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "anterior_posterior_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "superior_inferior_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "bounds": {
                        "left_right": {"minimum": 0.0, "maximum": 511.0},
                        "anterior_posterior": {"minimum": 0.0, "maximum": 511.0},
                        "superior_inferior": {
                            "minimum": mask_min_z,
                            "maximum": mask_max_z,
                        },
                    },
                    "method": "test_real_lung_bounds",
                    "confidence": 1.0,
                    "applied": True,
                    "warnings": [],
                },
                "local_frames": {},
            },
        }

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]
        center_indices = [
            (planned_slice.center.z - origin_z) / spacing_z
            for planned_slice in plan.slices
        ]

        configured_start_extension = 3.0
        configured_end_extension = 3.0
        extended_length = (
            mask_max_z
            - mask_min_z
            + configured_start_extension
            + configured_end_extension
        )
        expected_step_count = int(np.ceil(extended_length / 5.0))
        expected_center_overflow = (
            expected_step_count * 5.0 - extended_length
        )
        expected_min_index = (
            mask_min_z
            - configured_end_extension
            - expected_center_overflow / 2.0
            - origin_z
        ) / spacing_z
        expected_max_index = (
            mask_max_z
            + configured_start_extension
            + expected_center_overflow / 2.0
            - origin_z
        ) / spacing_z

        self.assertEqual(plan.source_labels, ["anatomical_geometry.global_frame"])
        self.assertEqual(len(center_indices), expected_step_count + 1)
        self.assertLessEqual(min(center_indices), mask_min_index)
        self.assertGreaterEqual(max(center_indices), mask_max_index)
        self.assertAlmostEqual(min(center_indices), expected_min_index, places=6)
        self.assertAlmostEqual(max(center_indices), expected_max_index, places=6)
        for previous, current in zip(center_indices, center_indices[1:]):
            self.assertAlmostEqual(
                abs(current - previous) * spacing_z,
                5.0,
                places=6,
            )

    def test_lung_whole_range_uses_anatomical_frame_for_center_and_parallel_slices(self):
        """验证 `test_lung_whole_range_uses_anatomical_frame_for_center_and_parallel_slices` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=4.0,
                slice_spacing=10.0,
                window_presets=["肺窗"],
            )],
        )
        localization_result = {
            "body_part": "肺部",
            "correction_scope_bbox": {
                "z_min": 0, "z_max": 40,
                "y_min": 0, "y_max": 60,
                "x_min": 0, "x_max": 100,
            },
            "image": {
                "size_xyz": [120, 100, 80],
                "spacing_xyz": [1.0, 1.0, 1.0],
                "origin_xyz": [0.0, 0.0, 0.0],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "anatomical_geometry": {
                "global_frame": {
                    "origin": {"x": 16.0, "y": 40.0, "z": 62.0},
                    "left_right_axis": {"x": 0.8, "y": 0.0, "z": 0.6},
                    "anterior_posterior_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "superior_inferior_axis": {"x": -0.6, "y": 0.0, "z": 0.8},
                    "bounds": {
                        "left_right": {"minimum": 0.0, "maximum": 100.0},
                        "anterior_posterior": {"minimum": 10.0, "maximum": 70.0},
                        "superior_inferior": {"minimum": 20.0, "maximum": 60.0},
                    },
                    "method": "lung_superior_envelope",
                    "confidence": 0.9,
                    "applied": True,
                    "warnings": [],
                },
                "local_frames": {},
            },
        }

        result = SingleLabelWholeRangePlanner().build_plans(request, localization_result)

        plan = result.plans[0]
        # 未指定局部范围时，斜切也必须覆盖整张原图在切片平面上的投影。
        self.assertAlmostEqual(plan.extent_mm.width, 142.6, places=6)
        self.assertAlmostEqual(plan.extent_mm.height, 99.0, places=6)
        self.assertEqual(plan.slices[0].normal.to_dict(), {"x": -0.6, "y": 0.0, "z": 0.8})
        self.assertEqual(plan.slices[0].row_dir.to_dict(), {"x": 0.8, "y": 0.0, "z": 0.6})
        self.assertEqual(plan.slices[0].col_dir.to_dict(), {"x": 0.0, "y": 1.0, "z": 0.0})
        # 切片平面保持原目标层面，但画面中心移动到整张原图的投影中心。
        # 解剖 SI 轴已统一指向上方，因此横状位从最大投影向最小投影输出。
        self.assertEqual(len(plan.slices), 5)
        self.assertAlmostEqual(plan.range_start.x, 21.04, places=6)
        self.assertAlmostEqual(plan.range_start.y, 49.5, places=6)
        self.assertAlmostEqual(plan.range_start.z, 90.78, places=6)
        self.assertAlmostEqual(plan.range_end.x, 45.04, places=6)
        self.assertAlmostEqual(plan.range_end.z, 58.78, places=6)
        for previous, current in zip(plan.slices, plan.slices[1:]):
            distance = ((current.center.x - previous.center.x) ** 2
                        + (current.center.y - previous.center.y) ** 2
                        + (current.center.z - previous.center.z) ** 2) ** 0.5
            self.assertAlmostEqual(distance, 10.0, places=6)

        request.plane_configs[0].line_extent_mode = CutLineExtentModeEnum.部位范围
        target_plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]
        self.assertEqual(
            target_plan.extent_mm.to_dict(),
            {"width": 100.0, "height": 60.0},
        )
        self.assertAlmostEqual(target_plan.range_start.x, 4.0, places=6)
        self.assertAlmostEqual(target_plan.range_start.y, 40.0, places=6)
        self.assertAlmostEqual(target_plan.range_start.z, 78.0, places=6)

    def test_cervical_three_planes_follow_complete_spine_axis(self):
        """颈椎三状位应沿脊柱轴切割，矢状画面按原图头脚方向摆正。"""

        spine_axis = np.asarray([0.2, 0.3, 0.9327379053], dtype=float)
        spine_axis /= np.linalg.norm(spine_axis)
        dicom_lr = np.asarray([1.0, 0.0, 0.0])
        left_right = dicom_lr - np.dot(dicom_lr, spine_axis) * spine_axis
        left_right /= np.linalg.norm(left_right)
        anterior_posterior = np.cross(spine_axis, left_right)

        def vector(value):
            return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.颈椎,
            plane_configs=[
                PlaneConfig(plane, 4.0, 10.0, ["骨窗"])
                for plane in (
                    PlaneTypeEnum.横状位,
                    PlaneTypeEnum.矢状位,
                    PlaneTypeEnum.冠状位,
                )
            ],
        )
        localization_result = {
            "body_part": "颈椎",
            "correction_scope_bbox": {
                "z_min": 10, "z_max": 70,
                "y_min": 10, "y_max": 70,
                "x_min": 10, "x_max": 70,
            },
            "image": {
                "size_xyz": [100, 100, 100],
                "spacing_xyz": [1.0, 1.0, 1.0],
                "origin_xyz": [0.0, 0.0, 0.0],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "anatomical_geometry": {
                "global_frame": {
                    "origin": {"x": 40.0, "y": 40.0, "z": 40.0},
                    "left_right_axis": vector(left_right),
                    "anterior_posterior_axis": vector(anterior_posterior),
                    "superior_inferior_axis": vector(spine_axis),
                    "bounds": {
                        "left_right": {"minimum": 10.0, "maximum": 50.0},
                        "anterior_posterior": {"minimum": 5.0, "maximum": 45.0},
                        "superior_inferior": {"minimum": 20.0, "maximum": 80.0},
                    },
                    "method": "spine_endpoint_centroids",
                    "confidence": 1.0,
                    "applied": True,
                    "warnings": [],
                },
                "local_frames": {},
            },
        }

        plans = SingleLabelWholeRangePlanner().build_plans(request, localization_result).plans

        axial, sagittal, coronal = plans
        axial_normal = np.asarray(list(axial.slices[0].normal.to_dict().values()))
        self.assertTrue(np.allclose(axial_normal, spine_axis))
        first_projection = np.dot(
            list(axial.slices[0].center.to_dict().values()), spine_axis
        )
        last_projection = np.dot(
            list(axial.slices[-1].center.to_dict().values()), spine_axis
        )
        self.assertGreater(first_projection, last_projection)
        for plan in plans:
            normals = [slice_.normal.to_dict() for slice_ in plan.slices]
            self.assertTrue(all(value == normals[0] for value in normals))
            for previous, current in zip(plan.slices, plan.slices[1:]):
                self.assertAlmostEqual(
                    np.linalg.norm(
                        np.asarray(list(current.center.to_dict().values()))
                        - np.asarray(list(previous.center.to_dict().values()))
                    ),
                    10.0,
                    places=6,
                )
        for plan in (sagittal, coronal):
            normal = np.asarray(list(plan.slices[0].normal.to_dict().values()))
            self.assertAlmostEqual(float(np.dot(normal, spine_axis)), 0.0, places=6)

        # 矢状位切割平面不变，但画面上下应尽量贴近原图默认的足侧方向，
        # 不能继续把负脊柱轴直接作为 col_dir 导致整张成片跟着旋转。
        sagittal_normal = np.asarray(
            list(sagittal.slices[0].normal.to_dict().values())
        )
        original_inferior = np.asarray([0.0, 0.0, -1.0])
        expected_col = original_inferior - np.dot(
            original_inferior, sagittal_normal
        ) * sagittal_normal
        expected_col /= np.linalg.norm(expected_col)
        sagittal_row = np.asarray(
            list(sagittal.slices[0].row_dir.to_dict().values())
        )
        sagittal_col = np.asarray(
            list(sagittal.slices[0].col_dir.to_dict().values())
        )
        self.assertTrue(np.allclose(sagittal_col, expected_col))
        self.assertTrue(
            np.allclose(np.cross(sagittal_row, sagittal_col), sagittal_normal)
        )
        expected_reference = (
            left_right * 30.0
            + anterior_posterior * 25.0
            + spine_axis * 50.0
        )
        self.assertTrue(np.allclose(
            list(sagittal.to_dict()["localizer_reference"].values()),
            expected_reference,
        ))

    def test_non_lung_keeps_legacy_axes_even_if_anatomical_geometry_is_present(self):
        """验证 `test_non_lung_keeps_legacy_axes_even_if_anatomical_geometry_is_present` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.头骨,
            plane_configs=[PlaneConfig(
                PlaneTypeEnum.横状位, 1.0, 5.0, ["骨窗"]
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["anatomical_geometry"] = {
            "global_frame": {
                "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
                "left_right_axis": {"x": 0.8, "y": 0.0, "z": 0.6},
                "anterior_posterior_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "superior_inferior_axis": {"x": -0.6, "y": 0.0, "z": 0.8},
                "bounds": {
                    "left_right": {"minimum": 0.0, "maximum": 10.0},
                    "anterior_posterior": {"minimum": 0.0, "maximum": 10.0},
                    "superior_inferior": {"minimum": 0.0, "maximum": 10.0},
                },
                "method": "test", "confidence": 1.0, "applied": True, "warnings": [],
            },
            "local_frames": {},
        }

        plan = SingleLabelWholeRangePlanner().build_plans(request, localization_result).plans[0]

        self.assertEqual(plan.slices[0].normal.to_dict(), {"x": 0.0, "y": 0.0, "z": 1.0})

    def test_thoracic_axial_uses_default_image_axis_but_sagittal_keeps_spine_axis(self):
        """胸椎横状位应水平切，矢状位仍应使用胸椎整体走向。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.胸椎,
            plane_configs=[
                PlaneConfig(PlaneTypeEnum.横状位, 2.0, 5.0, ["骨窗"]),
                PlaneConfig(PlaneTypeEnum.矢状位, 2.0, 5.0, ["骨窗"]),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["body_part"] = "胸椎"
        localization_result["image"]["size_xyz"] = [120, 120, 120]
        localization_result["anatomical_geometry"] = {
            "global_frame": {
                "origin": {"x": 60.0, "y": 50.0, "z": 50.0},
                "left_right_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "anterior_posterior_axis": {
                    "x": 0.0, "y": 0.8944271909999159, "z": -0.4472135954999579,
                },
                "superior_inferior_axis": {
                    "x": 0.0, "y": 0.4472135954999579, "z": 0.8944271909999159,
                },
                "bounds": {
                    "left_right": {"minimum": 50.0, "maximum": 70.0},
                    "anterior_posterior": {"minimum": 10.0, "maximum": 30.0},
                    "superior_inferior": {"minimum": 20.0, "maximum": 80.0},
                },
                "method": "spine_endpoint_centroids",
                "confidence": 1.0,
                "applied": True,
                "warnings": [],
            },
            "local_frames": {},
        }

        axial, sagittal = SingleLabelWholeRangePlanner().build_plans(
            request, localization_result
        ).plans

        self.assertEqual(
            axial.slices[0].normal.to_dict(),
            {"x": 0.0, "y": 0.0, "z": 1.0},
        )
        self.assertGreater(axial.slices[0].center.z, axial.slices[-1].center.z)
        self.assertEqual(
            sagittal.slices[0].normal.to_dict(),
            {"x": -1.0, "y": -0.0, "z": -0.0},
        )

    def test_thoracic_coronal_uses_cpr_curve_on_sagittal_localizer(self):
        """胸椎冠状位应按 CPR 曲线规划，并在矢状位定位图显示 T 段。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.胸椎,
            plane_configs=[
                PlaneConfig(PlaneTypeEnum.冠状位, 2.0, 5.0, ["骨窗"]),
            ],
        )
        localization_result = self._spine_localization_result(
            ("C7", "T1", "T2", "T3", "T12", "L1"),
            "胸椎",
        )
        localization_result["correction_scope_bbox"] = {
            "z_min": 0, "z_max": 100,
            "y_min": 10, "y_max": 90,
            "x_min": 0, "x_max": 100,
        }
        localization_result["coords"]["vertebrae_T1"]["bbox"].update({
            "z_min": 70, "z_max": 75,
            "y_min": 40, "y_max": 60,
            "x_min": 45, "x_max": 55,
        })
        localization_result["coords"]["vertebrae_T1"]["centroid"].update({
            "z": 75.0, "y": 50.0, "x": 50.0,
        })
        localization_result["coords"]["vertebrae_T2"]["bbox"].update({
            "z_min": 60, "z_max": 65,
            "y_min": 40, "y_max": 60,
            "x_min": 45, "x_max": 55,
        })
        localization_result["coords"]["vertebrae_T2"]["centroid"].update({
            "z": 65.0, "y": 50.0, "x": 50.0,
        })
        localization_result["coords"]["vertebrae_T3"]["bbox"].update({
            "z_min": 50, "z_max": 55,
            "y_min": 40, "y_max": 60,
            "x_min": 45, "x_max": 55,
        })
        localization_result["coords"]["vertebrae_T3"]["centroid"].update({
            "z": 55.0, "y": 50.0, "x": 50.0,
        })
        localization_result["coords"]["vertebrae_T12"]["bbox"].update({
            "z_min": 10, "z_max": 15,
            "y_min": 40, "y_max": 60,
            "x_min": 45, "x_max": 55,
        })
        localization_result["coords"]["vertebrae_T12"]["centroid"].update({
            "z": 10.0, "y": 50.0, "x": 50.0,
        })
        localization_result["anatomical_geometry"] = {
            "global_frame": {
                "origin": {"x": 50.0, "y": 50.0, "z": 50.0},
                "left_right_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "anterior_posterior_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "superior_inferior_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                "bounds": {
                    "left_right": {"minimum": 0.0, "maximum": 100.0},
                    "anterior_posterior": {"minimum": 10.0, "maximum": 90.0},
                    "superior_inferior": {"minimum": 0.0, "maximum": 100.0},
                },
                "method": "spine_endpoint_centroids",
                "confidence": 1.0,
                "applied": True,
                "warnings": [],
            },
            "local_frames": {},
        }

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.geometry_type, "curved_stack")
        self.assertEqual(plan.cut_mode, "cpr")
        self.assertEqual(plan.group_name, "CPR")
        self.assertEqual(plan.plane, "冠状位")
        self.assertEqual(plan.localizer_plane, "矢状位")
        self.assertEqual(plan.source_labels, [
            "vertebrae_T1",
            "vertebrae_T2",
            "vertebrae_T3",
            "vertebrae_T12",
        ])
        self.assertEqual(plan.offset_axis.to_dict(), {"x": 0.0, "y": 1.0, "z": 0.0})
        self.assertIn(0.0, plan.offsets_mm)
        self.assertGreater(len(plan.offsets_mm), 1)
        self.assertAlmostEqual(plan.extent_mm.width, 10.0)
        self.assertAlmostEqual(plan.extent_mm.height, 65.0)
        self.assertAlmostEqual(plan.localizer_reference.x, 50.0)
        self.assertAlmostEqual(plan.localizer_reference.z, 51.25)

    def test_thoracic_coronal_cpr_ignores_c7_l1_endpoint_helpers(self):
        """胸椎冠状位 CPR 曲线只使用 T 段椎体，不把 C7/L1 画进去。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.胸椎,
            plane_configs=[
                PlaneConfig(PlaneTypeEnum.冠状位, 2.0, 5.0, ["骨窗"]),
            ],
        )
        localization_result = self._spine_localization_result(
            ("C7", "T1", "T2", "T3", "L1"),
            "胸椎",
        )
        localization_result["coords"]["vertebrae_C7"]["centroid"]["z"] = 100.0
        localization_result["coords"]["vertebrae_L1"]["centroid"]["z"] = -100.0
        localization_result["coords"]["vertebrae_T1"]["centroid"]["z"] = 30.0
        localization_result["coords"]["vertebrae_T2"]["centroid"]["z"] = 20.0
        localization_result["coords"]["vertebrae_T3"]["centroid"]["z"] = 10.0
        localization_result["coords"]["vertebrae_T1"]["bbox"].update({
            "z_min": 29,
            "z_max": 31,
        })
        localization_result["coords"]["vertebrae_T2"]["bbox"].update({
            "z_min": 19,
            "z_max": 21,
        })
        localization_result["coords"]["vertebrae_T3"]["bbox"].update({
            "z_min": 9,
            "z_max": 11,
        })

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.geometry_type, "curved_stack")
        self.assertEqual(plan.localizer_plane, "矢状位")
        self.assertEqual(
            [point.z for point in plan.curve_points],
            [30.0, 20.0, 10.0],
        )

    def test_hip_only_tilts_axial_and_sets_three_localizer_planes(self):
        """髋骨仅横状位使用 AB 角度，矢冠保持默认轴并指定定位背景。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.髋骨,
            plane_configs=[
                PlaneConfig(plane, 2.0, 5.0, ["骨窗"])
                for plane in (
                    PlaneTypeEnum.横状位,
                    PlaneTypeEnum.矢状位,
                    PlaneTypeEnum.冠状位,
                )
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["body_part"] = "髋骨"
        localization_result["image"]["size_xyz"] = [100, 100, 100]
        localization_result["correction_scope_bbox"] = {
            "z_min": 10, "z_max": 70,
            "y_min": 20, "y_max": 80,
            "x_min": 15, "x_max": 85,
        }
        localization_result["anatomical_geometry"] = {
            "global_frame": {
                "origin": {"x": 51.0, "y": 52.0, "z": 43.0},
                "left_right_axis": {
                    "x": 0.9805806756909202, "y": 0.0,
                    "z": -0.19611613513818404,
                },
                "anterior_posterior_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "superior_inferior_axis": {
                    "x": 0.19611613513818404, "y": 0.0,
                    "z": 0.9805806756909202,
                },
                "bounds": {
                    "left_right": {"minimum": 10.0, "maximum": 90.0},
                    "anterior_posterior": {"minimum": 20.0, "maximum": 80.0},
                    "superior_inferior": {"minimum": 15.0, "maximum": 75.0},
                },
                "method": "hip_outer_superior_caps",
                "confidence": 1.0,
                "applied": True,
                "warnings": [],
            },
            "local_frames": {},
        }

        axial, sagittal, coronal = SingleLabelWholeRangePlanner().build_plans(
            request, localization_result
        ).plans

        self.assertTrue(np.allclose(
            list(axial.slices[0].normal.to_dict().values()),
            [0.19611613513818404, 0.0, 0.9805806756909202],
        ))
        self.assertEqual(
            sagittal.slices[0].normal.to_dict(),
            {"x": -1.0, "y": -0.0, "z": -0.0},
        )
        self.assertEqual(
            coronal.slices[0].normal.to_dict(),
            {"x": 0.0, "y": 1.0, "z": 0.0},
        )
        expected_reference = {"x": 51.0, "y": 52.0, "z": 43.0}
        for plan in (axial, sagittal, coronal):
            self.assertEqual(plan.localizer_reference.to_dict(), expected_reference)
        self.assertEqual(
            [plan.localizer_plane for plan in (axial, sagittal, coronal)],
            ["冠状位", "冠状位", "矢状位"],
        )

    def test_oblique_plan_overview_draws_the_actual_sloped_guide(self):
        """验证 `test_oblique_plan_overview_draws_the_actual_sloped_guide` 场景符合预期。"""

        from src.stages.stage_mapping import MappingStage, _OVERVIEW_AXES

        image = sitk.Image(64, 64, 64, sitk.sitkUInt8)
        overlay = np.zeros((64, 64, 64), dtype=np.uint16)
        plan = {
            "extent_mm": {"width": 40.0, "height": 20.0},
            "slices": [{
                "center": {"x": 32.0, "y": 32.0, "z": 32.0},
                "normal": {"x": -0.6, "y": 0.0, "z": 0.8},
                "row_dir": {"x": 0.8, "y": 0.0, "z": 0.6},
                "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
            }],
        }

        MappingStage()._paint_plan_overview(
            reference_image=image,
            overlay_array=overlay,
            plan=plan,
            axes=_OVERVIEW_AXES["横状位"],
            value=14,
        )

        painted = np.argwhere(overlay == 14)
        self.assertGreater(len(np.unique(painted[:, 0])), 1)
        self.assertGreater(len(np.unique(painted[:, 2])), 1)

    def test_single_label_planner_builds_sagittal_whole_range_plan(self):
        """单目标整切 planner 应把矢状位交给通用直线构造器。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.矢状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=["肺窗"],
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
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.横状位,
                    3.0,
                    5.0,
                    ["肺窗"],
                ),
                PlaneConfig(
                    PlaneTypeEnum.矢状位,
                    3.0,
                    5.0,
                    ["肺窗"],
                ),
                PlaneConfig(
                    PlaneTypeEnum.冠状位,
                    3.0,
                    5.0,
                    ["肺窗"],
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
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=["骨窗"],
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
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=["骨窗"],
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
            cut_mode="whole_range",
            plane_config=request.plane_configs[0],
            localization_result=localization_result,
            segment=segment,
            frame=frame,
        )

        self.assertEqual(plan.slices[0].center.x, 49.5)
        self.assertEqual(plan.slices[0].center.y, 59.5)
        self.assertEqual(plan.slices[0].center.z, 12.5)
        self.assertEqual(plan.extent_mm.to_dict(), {
            "width": 99.0,
            "height": 119.0,
        })

    def test_straight_plan_covers_target_along_final_tilted_normal(self):
        """倾斜区间也应沿最终法向等距展开并完整覆盖首尾。"""

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
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=2.0,
                slice_spacing=2.0,
                window_presets=["骨窗"],
                line_extent_mode=CutLineExtentModeEnum.部位范围,
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["cut_configs"] = {
            "interspace": {
                "intervals": [{"start": "L1", "end": "L2"}],
                "slice_range_extension": {
                    "横状位": {"start": 1.0, "end": 3.0},
                },
            },
        }
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
            cut_mode="interspace",
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
        self.assertAlmostEqual(planned_projection_length, 12.0)
        self.assertGreaterEqual(
            planned_projection_length,
            target_projection_length,
        )
        self.assertEqual(len(plan.slices), 7)
        start_extra = (
            (plan.range_start.x - plan.target_start.x) ** 2
            + (plan.range_start.y - plan.target_start.y) ** 2
            + (plan.range_start.z - plan.target_start.z) ** 2
        ) ** 0.5
        end_extra = (
            (plan.range_end.x - plan.target_end.x) ** 2
            + (plan.range_end.y - plan.target_end.y) ** 2
            + (plan.range_end.z - plan.target_end.z) ** 2
        ) ** 0.5
        self.assertAlmostEqual(end_extra - start_extra, 2.0, places=6)
        for previous, current in zip(plan.slices, plan.slices[1:]):
            distance = ((current.center.x - previous.center.x) ** 2
                        + (current.center.y - previous.center.y) ** 2
                        + (current.center.z - previous.center.z) ** 2) ** 0.5
            self.assertAlmostEqual(distance, 2.0, places=6)

    def test_whole_range_builder_uses_projected_full_plane_center(self):
        """现有整切公开入口应接入通用直线范围投影流水线。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=["骨窗"],
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
        self.assertEqual(plan.slices[0].center.z, 12.5)

    def test_axial_whole_range_defaults_to_full_plane_extent(self):
        """验证 `test_axial_whole_range_defaults_to_full_plane_extent` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom", output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位, slice_thickness=3.0,
                slice_spacing=5.0, window_presets=["肺窗"],
            )],
        )
        loc = self._lumbar_localization_result()
        loc["image"]["size_xyz"] = [100, 120, 30]
        loc["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]

        plan = SingleLabelWholeRangePlanner().build_plans(request, loc).plans[0]

        self.assertEqual(plan.extent_mm.to_dict(), {"width": 99.0, "height": 238.0})

    def test_axial_whole_range_can_limit_extent_to_target_bbox(self):
        """验证 `test_axial_whole_range_can_limit_extent_to_target_bbox` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom", output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位, slice_thickness=3.0,
                slice_spacing=5.0, window_presets=["肺窗"],
                line_extent_mode=CutLineExtentModeEnum.部位范围,
            )],
        )
        loc = self._lumbar_localization_result()
        loc["image"]["size_xyz"] = [100, 120, 30]
        loc["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]

        plan = SingleLabelWholeRangePlanner().build_plans(request, loc).plans[0]

        self.assertEqual(plan.extent_mm.to_dict(), {"width": 10.0, "height": 20.0})

    def test_axial_whole_range_symmetrically_expands_to_cover_both_ends(self):
        """不能整除间距的整切范围应增加一张并在首尾对称外扩。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=["肺窗"],
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
        self.assertEqual(plan.target_start.z, 10.5)
        self.assertEqual(plan.target_end.z, -0.5)
        self.assertEqual(plan.range_start.z, 12.5)
        self.assertEqual(plan.range_end.z, -2.5)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            12.5,
            7.5,
            2.5,
            -2.5,
        ])

    def test_axial_whole_range_short_nonzero_target_uses_two_outer_centers(self):
        """短于层厚和间距的非零 bbox 仍应由首末中心跨过两端。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=["肺窗"],
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

        self.assertEqual(plan.range_start.z, 7.5)
        self.assertEqual(plan.range_end.z, 2.5)
        self.assertEqual(
            [planned_slice.center.z for planned_slice in plan.slices],
            [7.5, 2.5],
        )

    def test_lumbar_whole_range_planner_expands_slices_from_l1_to_s1(self):
        """验证 `test_lumbar_whole_range_planner_expands_slices_from_l1_to_s1` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=["骨窗"],
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
        self.assertEqual(plan.slices[0].center.z, 12.5)
        self.assertEqual(plan.slices[-1].center.z, -2.5)

    def test_lumbar_whole_range_axial_uses_outer_bbox_z_range(self):
        """验证 `test_lumbar_whole_range_axial_uses_outer_bbox_z_range` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=["骨窗"],
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
        self.assertEqual(plan.range_start.z, 12.5)
        self.assertEqual(plan.range_end.z, -2.5)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            12.5,
            7.5,
            2.5,
            -2.5,
        ])

    def test_lumbar_whole_range_axial_uses_bbox_even_without_endpoint_centroids(self):
        """验证 `test_lumbar_whole_range_axial_uses_bbox_even_without_endpoint_centroids` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=5.0,
                    slice_spacing=5.0,
                    window_presets=["骨窗"],
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
        self.assertEqual(plan.range_start.z, 12.5)
        self.assertEqual(plan.range_end.z, -2.5)
        self.assertEqual(plan.source_labels, ["correction_scope_bbox"])

    def test_lumbar_whole_range_planner_builds_three_straight_planes(self):
        """腰椎整切也应复用通用构造器生成三个默认直线状位。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    plane=plane,
                    slice_thickness=3.0,
                    slice_spacing=5.0,
                    window_presets=["骨窗"],
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
        """验证 `test_lumbar_interspace_planner_builds_one_plan_per_available_interval` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=2.0,
                    slice_spacing=2.0,
                    window_presets=["骨窗"],
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
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=0.5,
                slice_spacing=1.0,
                window_presets=["骨窗"],
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
        """验证 `test_mapping_stage_writes_slice_plans_json` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.肺部,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=["骨窗"],
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
            self.assertEqual(data["plans"][0]["plan_id"], "肺部_whole_range_横状位")
            self.assertIn("slices", data["plans"][0])
            self.assertEqual(data["diagnostics"][0]["slice_count"], 4)
            self.assertEqual(data["diagnostics"][0]["slice_thickness"], 5.0)
            self.assertEqual(data["diagnostics"][0]["slice_spacing"], 5.0)
            self.assertEqual(data["diagnostics"][0]["spacing_minus_thickness"], 0.0)
            self.assertIn("首尾对称", data["diagnostics"][0]["calculation_note"])
            self.assertEqual(data["diagnostics"][0]["actual_spacing_min_mm"], 5.0)
            self.assertEqual(data["diagnostics"][0]["actual_spacing_max_mm"], 5.0)
            self.assertFalse(data["diagnostics"][0]["has_nonuniform_spacing"])
            self.assertFalse(data["diagnostics"][0]["has_extra_endpoint_slice"])
            self.assertEqual(data["diagnostics"][0]["leading_uncovered_mm"], 0.0)
            self.assertEqual(data["diagnostics"][0]["trailing_uncovered_mm"], 0.0)
            self.assertEqual(data["diagnostics"][0]["leading_overflow_mm"], 4.5)
            self.assertEqual(data["diagnostics"][0]["trailing_overflow_mm"], 4.5)
            self.assertEqual(
                data["diagnostics"][0]["leading_center_uncovered_mm"],
                0.0,
            )
            self.assertEqual(
                data["diagnostics"][0]["trailing_center_uncovered_mm"],
                0.0,
            )
            self.assertEqual(
                data["diagnostics"][0]["leading_center_overflow_mm"],
                2.0,
            )
            self.assertEqual(
                data["diagnostics"][0]["trailing_center_overflow_mm"],
                2.0,
            )
            self.assertEqual(data["diagnostics"][0]["target_length_mm"], 11.0)
            self.assertFalse(data["diagnostics"][0]["target_shorter_than_thickness"])
            self.assertEqual(data["diagnostics"][0]["first_coverage_start"]["z"], 15.0)
            self.assertEqual(data["diagnostics"][0]["last_coverage_end"]["z"], -5.0)

    def test_mapping_diagnostics_ignore_sub_tolerance_center_gaps(self):
        """中心越界量在规划容差内时，不应被诊断为首尾缺失。"""

        diagnostics = MappingStage()._build_plan_diagnostics([{
            "plan_id": "center_tolerance",
            "plane": "横状位",
            "range_start": {"x": 0.0, "y": 0.0, "z": 0.0005},
            "range_end": {"x": 0.0, "y": 0.0, "z": 10.0005},
            "target_start": {"x": 0.0, "y": 0.0, "z": 0.0},
            "target_end": {"x": 0.0, "y": 0.0, "z": 10.001},
            "slice_thickness": 1.0,
            "slice_spacing": 5.0,
            "slices": [
                {"center": {"x": 0.0, "y": 0.0, "z": z}}
                for z in (0.0005, 5.0005, 10.0005)
            ],
        }])

        self.assertEqual(diagnostics[0]["leading_center_uncovered_mm"], 0.0)
        self.assertEqual(diagnostics[0]["trailing_center_uncovered_mm"], 0.0)

    def test_mapping_stage_writes_slice_plan_line_overlay(self):
        """验证 `test_mapping_stage_writes_slice_plan_line_overlay` 场景符合预期。"""

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
                body_part=BodyPartEnum.肺部,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=["骨窗"],
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
                plane_configs=[
                    PlaneConfig(
                        plane=plane,
                        slice_thickness=3.0,
                        slice_spacing=5.0,
                        window_presets=["骨窗"],
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
            center_cross_path = Path(result["debug_center_cross_overlay_path"])
            self.assertEqual(
                center_cross_path.name,
                "十字中线_横+矢+冠_区+整+CPR.nii.gz",
            )
            self.assertTrue(center_cross_path.exists())
            center_cross_image = sitk.ReadImage(str(center_cross_path))
            center_cross_array = sitk.GetArrayFromImage(center_cross_image)
            self.assertEqual(center_cross_image.GetSize(), image.GetSize())
            # 三个状位合并绘制，但全部使用 uint16 最大值显示为纯白色。
            self.assertEqual(
                set(np.unique(center_cross_array)) - {0},
                {65535},
            )
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
                "切片原始网格总览_冠_CPR.nii.gz",
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
                metadata["debug_center_cross_overlay_path"],
                str(center_cross_path),
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

    def test_mapping_cpr_overview_draws_curve_on_sagittal_background(self):
        """CPR 调试总览应按曲线点画在矢状位背景上。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((36, 36, 36), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))
            context = ProcessingContext(
                Request(
                    dicom_dir="/tmp/dicom",
                    output_dir=temp_dir,
                    body_part=BodyPartEnum.腰椎,
                    plane_configs=[],
                )
            )
            localization_result = {
                "seg_for_reconstruction_path": str(seg_path),
            }
            plan = {
                "plane": "矢状位",
                "geometry_type": "curved_stack",
                "cut_mode": "cpr",
                "localizer_plane": "矢状位",
                "localizer_reference": {"x": 18.0, "y": 18.0, "z": 18.0},
                "extent_mm": {"width": 20.0, "height": 28.0},
                "curve_points": [
                    {"x": 18.0, "y": 10.0, "z": 5.0},
                    {"x": 18.0, "y": 16.0, "z": 14.0},
                    {"x": 18.0, "y": 8.0, "z": 23.0},
                    {"x": 18.0, "y": 14.0, "z": 32.0},
                ],
                "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "offsets_mm": [0.0],
                "slices": [{
                    "index": 1,
                    "center": {"x": 18.0, "y": 12.0, "z": 18.0},
                    "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "col_dir": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "offset_mm": 0.0,
                }],
            }

            _, native_paths = MappingStage()._write_slice_plan_overview_overlays(
                context,
                localization_result,
                [plan],
            )
            overview = sitk.ReadImage(native_paths["矢状位"])
            highlighted = np.argwhere(sitk.GetArrayFromImage(overview) > 0)

            self.assertGreater(len(highlighted), 20)
            self.assertGreater(self._max_line_fit_residual(highlighted), 2.0)

    def test_mapping_cpr_overview_draws_only_baseline_curve(self):
        """CPR 调试总览只画基准中心曲线。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_path = temp_path / "seg.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((36, 36, 36), dtype=np.uint8))
            image.SetSpacing((1.0, 1.0, 1.0))
            image.SetOrigin((0.0, 0.0, 0.0))
            sitk.WriteImage(image, str(seg_path))
            context = ProcessingContext(
                Request(
                    dicom_dir="/tmp/dicom",
                    output_dir=temp_dir,
                    body_part=BodyPartEnum.腰椎,
                    plane_configs=[],
                )
            )
            localization_result = {
                "seg_for_reconstruction_path": str(seg_path),
            }
            plan = {
                "plane": "矢状位",
                "geometry_type": "curved_stack",
                "cut_mode": "cpr",
                "localizer_plane": "矢状位",
                "localizer_reference": {"x": 18.0, "y": 18.0, "z": 18.0},
                "extent_mm": {"width": 20.0, "height": 28.0},
                "curve_points": [
                    {"x": 18.0, "y": 18.0, "z": 5.0},
                    {"x": 18.0, "y": 18.0, "z": 14.0},
                    {"x": 18.0, "y": 18.0, "z": 23.0},
                    {"x": 18.0, "y": 18.0, "z": 32.0},
                ],
                "offset_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "offsets_mm": [-8.0, 0.0, 8.0],
                "slices": [{
                    "index": 1,
                    "center": {"x": 18.0, "y": 18.0, "z": 18.0},
                    "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "col_dir": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "offset_mm": 0.0,
                }],
            }

            _, native_paths = MappingStage()._write_slice_plan_overview_overlays(
                context,
                localization_result,
                [plan],
            )
            overview = sitk.ReadImage(native_paths["矢状位"])
            highlighted = np.argwhere(sitk.GetArrayFromImage(overview) > 0)

            self.assertEqual(np.unique(highlighted[:, 1]).tolist(), [18])

    def test_center_cross_uses_lung_anatomical_origin_in_long_scan(self):
        """颈到盆长扫描的十字中线应落在肺中心，而不是整幅图像中心。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            seg_path = Path(temp_dir) / "seg.nii.gz"
            image = sitk.GetImageFromArray(
                np.zeros((100, 40, 40), dtype=np.uint8),
            )
            sitk.WriteImage(image, str(seg_path))
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                body_part=BodyPartEnum.肺部,
            )
            context = ProcessingContext(request)
            localization_result = {
                "seg_for_reconstruction_path": str(seg_path),
                "anatomical_geometry": {
                    "global_frame": {
                        "origin": {"x": 20.0, "y": 20.0, "z": 80.0},
                        "left_right_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "anterior_posterior_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                        "superior_inferior_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                        "bounds": {
                            "left_right": {"minimum": 10.0, "maximum": 30.0},
                            "anterior_posterior": {"minimum": 10.0, "maximum": 30.0},
                            "superior_inferior": {"minimum": 70.0, "maximum": 90.0},
                        },
                        "method": "test_lung_frame",
                        "confidence": 1.0,
                        "applied": True,
                        "warnings": [],
                    },
                    "local_frames": {},
                },
            }
            plans = [
                {
                    "plane": "横状位",
                    "cut_mode": "whole_range",
                    "extent_mm": {"width": 30.0, "height": 30.0},
                    "slices": [{
                        "center": {"x": 20.0, "y": 20.0, "z": 50.0},
                        "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                    }],
                },
                {
                    "plane": "矢状位",
                    "cut_mode": "whole_range",
                    "extent_mm": {"width": 30.0, "height": 30.0},
                    "slices": [{
                        "center": {"x": 20.0, "y": 20.0, "z": 50.0},
                        "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                        "col_dir": {"x": 0.0, "y": 0.0, "z": -1.0},
                    }],
                },
                {
                    "plane": "冠状位",
                    "cut_mode": "whole_range",
                    "extent_mm": {"width": 30.0, "height": 30.0},
                    "slices": [{
                        "center": {"x": 20.0, "y": 20.0, "z": 50.0},
                        "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "col_dir": {"x": 0.0, "y": 0.0, "z": -1.0},
                    }],
                },
            ]

            output_path = MappingStage()._write_center_cross_overlay(
                context,
                localization_result,
                plans,
            )
            overlay = sitk.GetArrayFromImage(sitk.ReadImage(output_path))

            self.assertEqual(int(overlay[80, 20, 30]), 65535)
            self.assertEqual(int(overlay[50, 20, 30]), 0)

    def test_mapping_stage_writes_single_layer_overview_overlay(self):
        """验证 `test_mapping_stage_writes_single_layer_overview_overlay` 场景符合预期。"""

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
                body_part=BodyPartEnum.肺部,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=["骨窗"],
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
            # 5 mm 真实间距在 0.5 mm 等距展示网格上恒定相差 10 行。
            center_rows = np.unique(np.argwhere(overview_array > 0)[:, 0])
            self.assertEqual(center_rows.tolist(), [1, 11, 21, 31])
            self.assertTrue(np.all(np.diff(center_rows) == 10))
            native_path = Path(result["debug_slice_plan_native_overview_overlay_path"])
            self.assertEqual(native_path.name, "切片原始网格总览_横_整.nii.gz")
            native_image = sitk.ReadImage(str(native_path))
            self.assertEqual(native_image.GetSize(), image.GetSize())

    def test_mapping_stage_overview_uses_minimum_line_width(self):
        """验证 `test_mapping_stage_overview_uses_minimum_line_width` 场景符合预期。"""

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
                body_part=BodyPartEnum.肺部,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=3.0,
                        slice_spacing=5.0,
                        window_presets=["骨窗"],
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
            # 层厚不再决定显示宽度；每张切片只占等距网格一行。
            painted_rows = np.unique(np.argwhere(overview_array > 0)[:, 0])
            self.assertEqual(painted_rows.tolist(), [1, 11, 21, 31])

    def test_mapping_stage_overview_overlay_respects_image_direction(self):
        """验证 `test_mapping_stage_overview_overlay_respects_image_direction` 场景符合预期。"""

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
                body_part=BodyPartEnum.肺部,
                plane_configs=[
                    PlaneConfig(
                        plane=PlaneTypeEnum.横状位,
                        slice_thickness=5.0,
                        slice_spacing=5.0,
                        window_presets=["骨窗"],
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
            center_rows = np.unique(np.argwhere(overview_array > 0)[:, 0])
            self.assertEqual(center_rows.tolist(), [1, 11, 21, 31])

    @staticmethod
    def _spine_localization_result(labels, body_part):
        """生成最小脊柱定位结果，供 CPR planner 测试使用。"""

        coords = {}
        for index, name in enumerate(labels):
            z = 10.0 - index * 3.0
            coords[f"vertebrae_{name}"] = {
                "label_name": f"vertebrae_{name}",
                "label_ids": [index + 1],
                "voxel_count": 10,
                "found": True,
                "centroid": {"z": z, "y": 50.0, "x": 60.0},
                "bbox": {
                    "z_min": int(z) - 1,
                    "z_max": int(z) + 1,
                    "y_min": 45,
                    "y_max": 55,
                    "x_min": 55,
                    "x_max": 65,
                },
            }

        return {
            "body_part": body_part,
            "cut_mode": "whole_range",
            "correction_scope_bbox": {
                "z_min": 0,
                "z_max": 12,
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
                "size_xyz": [120, 120, 40],
            },
            "coords": coords,
        }

    @staticmethod
    def _lumbar_localization_result():
        """为测试准备 `_lumbar_localization_result` 所需的数据或记录行为。"""

        def coord(name, z):
            """为测试准备 `coord` 所需的数据或记录行为。"""

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
