import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import SimpleITK as sitk
import numpy as np

from src.utils.Logger_utils import _logger_manager
_logger_manager.debug_file_enabled = True

from src.core.context import ProcessingContext
from src.config import (
    localization_profiles_for,
    load_slice_part_config,
    slice_rules_for,
)
from src.models import PlaneConfig, Point3D, Request
from src.models.enums import SlicePartEnum, PlaneTypeEnum, SliceType
from src.planning.base import select_planner
from src.planning.cervical import CervicalInterspacePlanner
from src.planning.geometry import normalize_vector, subtract_points
from src.planning.lumbar import LumbarInterspacePlanner, LumbarWholeRangePlanner
from src.planning.single_label import SingleLabelWholeRangePlanner
from src.planning.spine_interspace import configured_interspace_definitions
from src.planning.spine_cpr import (
    _build_annotation_centroid_path,
    _correct_reversing_cpr_centroids,
    _extend_curve_endpoints_to_terminal_raw_labels,
    _filtered_annotation_offset_interval_mm,
    _get_spine_annotation_centers,
    _resolve_sampling_offsets_mm,
    _resolve_endpoint_face_center,
    _smooth_cpr_centroid_path_with_straight_endpoints,
    configured_spine_cpr_labels,
)
from src.planning.whole_range import (
    build_whole_range_plan_from_bbox,
    plane_bbox_extent_mm,
)
from src.stages.stage_planning import PlanningStage


class Stage6SlicePlanningTest(unittest.TestCase):
    """覆盖第六阶段直线切片、椎间切和 CPR 规划。"""

    _ANATOMICAL_RESULT_KEYS = {
        "anatomical_geometry",
        "anatomical_geometry_diagnostics",
        "spine_geometry",
    }

    @staticmethod
    def _configured_localization_contract(slice_part):
        """从正式统一配置生成规划测试使用的定位配置契约。"""

        config = load_slice_part_config()
        aliases = config["label_aliases"]
        cut_configs = localization_profiles_for(config, slice_part)

        def alias_values(name):
            value = aliases.get(name, name)
            return list(value) if isinstance(value, list) else [value]

        resolved_cut_configs = {}
        for profile, cut_config in cut_configs.items():
            if profile == "interspace":
                resolved_cut_configs[profile] = {
                    "intervals": [
                        {
                            **interval,
                            "start_labels": alias_values(interval["start"]),
                            "end_labels": alias_values(interval["end"]),
                        }
                        for interval in cut_config
                    ]
                }
            else:
                resolved_cut_configs[profile] = {
                    **cut_config,
                    "start_labels": alias_values(cut_config["start"]),
                    "end_labels": alias_values(cut_config["end"]),
                }

        cpr_config = config["slice_parts"][slice_part]["localization"].get(
            "spine_cpr",
            {},
        )
        return {
            "cut_config": cut_configs.get("whole_range"),
            "cut_configs": cut_configs,
            "resolved_cut_config": resolved_cut_configs.get("whole_range"),
            "resolved_cut_configs": resolved_cut_configs,
            "slice_rules": slice_rules_for(config, slice_part),
            "spine_cpr_config": {
                "full_label": cpr_config.get("full_label", True),
                "excluded_labels": [
                    alias_values(name)[0]
                    for name in cpr_config.get("excluded_labels", [])
                ],
            },
        }

    @classmethod
    def _set_planning_inputs(cls, context, combined_result):
        """把合成 fixture 按当前定位/解剖契约拆入上下文。"""

        context.set_result(
            "localization",
            {
                key: value
                for key, value in combined_result.items()
                if key not in cls._ANATOMICAL_RESULT_KEYS
            },
        )
        context.set_result(
            "anatomical_analysis",
            {
                key: value
                for key, value in combined_result.items()
                if key in cls._ANATOMICAL_RESULT_KEYS
            },
        )

    @staticmethod
    def _max_line_fit_residual(points_zyx: np.ndarray) -> float:
        """返回二维高亮点到最佳直线的最大像素残差。"""

        rows = points_zyx[:, 0].astype(float)
        columns = points_zyx[:, 1].astype(float)
        matrix = np.column_stack([rows, np.ones_like(rows)])
        slope, intercept = np.linalg.lstsq(matrix, columns, rcond=None)[0]
        fitted = slope * rows + intercept
        return float(np.max(np.abs(columns - fitted)))

    @staticmethod
    def _point_close(left: Point3D, right: Point3D, tolerance: float = 1e-6) -> bool:
        """判断两个物理点是否足够接近。"""

        return (
            abs(left.x - right.x) <= tolerance
            and abs(left.y - right.y) <= tolerance
            and abs(left.z - right.z) <= tolerance
        )

    def assertCurveContainsPoint(self, curve_points, expected):
        """断言 CPR 平滑曲线仍穿过指定锚点。"""

        self.assertTrue(
            any(self._point_close(point, expected) for point in curve_points),
            f"curve does not contain anchor {expected}",
        )

    @staticmethod
    def _min_distance_to_curve(curve_points, expected):
        """返回锚点到 CPR 曲线采样点的最近距离。"""

        return min(
            float(np.linalg.norm([
                point.x - expected.x,
                point.y - expected.y,
                point.z - expected.z,
            ]))
            for point in curve_points
        )

    @staticmethod
    def _raw_label_exit_case(temp_dir, with_single_slice_hole=False):
        """构造 bbox 远大于白线沿线标签范围的首尾椎体。"""

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[40:47, 18:23, 18:23] = 1
        segmentation[14:21, 18:23, 18:23] = 3
        if with_single_slice_hole:
            segmentation[44, 18:23, 18:23] = 0
        segmentation_image = sitk.GetImageFromArray(segmentation)
        segmentation_path = Path(temp_dir) / "seg.nii.gz"
        sitk.WriteImage(segmentation_image, str(segmentation_path))

        centers = [
            ("vertebrae_L1", Point3D(20.0, 20.0, 40.0)),
            ("vertebrae_L2", Point3D(20.0, 20.0, 30.0)),
            ("vertebrae_L3", Point3D(20.0, 20.0, 20.0)),
        ]
        localization_result = {
            "seg_for_reconstruction_path": str(segmentation_path),
            "image": {
                "spacing_xyz": [1.0, 1.0, 1.0],
                "origin_xyz": [0.0, 0.0, 0.0],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
                "size_xyz": [64, 64, 64],
            },
            "coords": {
                "vertebrae_L1": {
                    "label_ids": [1],
                    "bbox": {
                        "x_min": 0, "x_max": 63,
                        "y_min": 0, "y_max": 63,
                        "z_min": 40, "z_max": 60,
                    },
                },
                "vertebrae_L3": {
                    "label_ids": [3],
                    "bbox": {
                        "x_min": 0, "x_max": 63,
                        "y_min": 0, "y_max": 63,
                        "z_min": 0, "z_max": 20,
                    },
                },
            },
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": {"x": 20.0, "y": 20.0, "z": 42.0},
                            "max": {"x": 20.0, "y": 20.0, "z": 38.0},
                        },
                    },
                    "vertebrae_L3": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": {"x": 20.0, "y": 20.0, "z": 22.0},
                            "max": {"x": 20.0, "y": 20.0, "z": 18.0},
                        },
                    },
                },
            },
        }
        return localization_result, centers

    def test_planning_derives_lumbar_axial_interspace_and_sagittal_straight_mpr(self):
        """腰椎横状位保持区间切基线，矢状位回到普通直线 MPR。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                slice_part=SlicePartEnum.腰椎,
                plane_configs=[
                    PlaneConfig(
                        PlaneTypeEnum.横状位, 1.0, 3.0,
                        ["骨窗"],
                        slice_type=SliceType.椎间隙,
                    ),
                    PlaneConfig(
                        PlaneTypeEnum.矢状位, 1.0, 3.0,
                        ["骨窗"],
                    ),
                ],
            )
            context = ProcessingContext(request)
            self._set_planning_inputs(
                context,
                self._lumbar_localization_result(),
            )

            PlanningStage().execute(context)

            plans = context.get_result("planning")["plans"]
            self.assertEqual(
                [
                    (
                        plan["plane"],
                        plan["slice_type"],
                        plan["geometry_type"],
                    )
                    for plan in plans
                ],
                [
                    ("横状位", "椎间隙", "plane_stack"),
                    ("横状位", "椎间隙", "plane_stack"),
                    ("横状位", "椎间隙", "plane_stack"),
                    ("横状位", "椎间隙", "plane_stack"),
                    ("横状位", "椎间隙", "plane_stack"),
                    ("矢状位", "MPR", "plane_stack"),
                ],
            )
            sagittal_plan = plans[-1]
            self.assertEqual(sagittal_plan["group_name"], "whole_range")
            self.assertEqual(sagittal_plan["localizer_plane"], "冠状位")
            self.assertNotIn("curve_points", sagittal_plan)
            self.assertNotIn("offsets_mm", sagittal_plan)

    def test_spine_coronal_cpr_fails_when_fewer_than_three_labels(self):
        """显式 CPR 条件不足时必须失败，不能静默改成 MPR。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            localization_result["coords"][label_name]["found"] = False

        with self.assertRaisesRegex(ValueError, "CPR 规划失败"):
            LumbarWholeRangePlanner().build_plans(request, localization_result)

    def test_spine_cpr_prefers_prism_centroid(self):
        """第五阶段几何可用时，CPR 应优先使用四棱柱体素质心。"""

        localization_result = {
            "coords": {},
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "centroid_physical": {
                            "x": 1.0, "y": 2.0, "z": 3.0,
                        },
                        "prism_found": True,
                        "prism_centroid_physical": {
                            "x": 4.0, "y": 5.0, "z": 6.0,
                        },
                    },
                },
            },
        }

        centers, warnings = _get_spine_annotation_centers(
            localization_result,
            ["vertebrae_L1"],
        )

        self.assertEqual(
            centers,
            [("vertebrae_L1", Point3D(4.0, 5.0, 6.0))],
        )
        self.assertEqual(warnings, [])

    def test_spine_cpr_warns_when_new_prism_contract_falls_back_to_body(self):
        """新结果明确四棱柱失败时，应回退主体质心并给出 label 警告。"""

        localization_result = {
            "coords": {},
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "centroid_physical": {
                            "x": 1.0, "y": 2.0, "z": 3.0,
                        },
                        "prism_found": False,
                        "prism_centroid_physical": None,
                    },
                },
            },
        }

        centers, warnings = _get_spine_annotation_centers(
            localization_result,
            ["vertebrae_L1"],
        )

        self.assertEqual(
            centers[0][1],
            Point3D(1.0, 2.0, 3.0),
        )
        self.assertIn("vertebrae_L1 四棱柱生成失败", warnings[0])

    def test_spine_cpr_warns_when_prism_state_is_missing(self):
        """输入缺少四棱柱状态时，使用主体质心并明确告警。"""

        localization_result = {
            "coords": {},
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "centroid_physical": {
                            "x": 1.0, "y": 2.0, "z": 3.0,
                        },
                    },
                },
            },
        }

        centers, warnings = _get_spine_annotation_centers(
            localization_result,
            ["vertebrae_L1"],
        )

        self.assertEqual(
            centers[0][1],
            Point3D(1.0, 2.0, 3.0),
        )
        self.assertIn("vertebrae_L1 四棱柱生成失败", warnings[0])

    def test_spine_cpr_falls_back_when_prism_centroid_is_incomplete(self):
        """四棱柱质心字段不完整时不能抛错，应回退主体质心并告警。"""

        localization_result = {
            "coords": {},
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "centroid_physical": {
                            "x": 1.0, "y": 2.0, "z": 3.0,
                        },
                        "prism_found": True,
                        "prism_centroid_physical": {"x": 4.0},
                    },
                },
            },
        }

        centers, warnings = _get_spine_annotation_centers(
            localization_result,
            ["vertebrae_L1"],
        )

        self.assertEqual(
            centers[0][1],
            Point3D(1.0, 2.0, 3.0),
        )
        self.assertIn("vertebrae_L1 四棱柱生成失败", warnings[0])

    def test_spine_cpr_rejects_nonfinite_prism_centroid(self):
        """四棱柱质心含非有限数值时，应回退到有效主体质心。"""

        localization_result = {
            "coords": {},
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "centroid_physical": {
                            "x": 1.0, "y": 2.0, "z": 3.0,
                        },
                        "prism_found": True,
                        "prism_centroid_physical": {
                            "x": math.nan, "y": 5.0, "z": 6.0,
                        },
                    },
                },
            },
        }

        centers, warnings = _get_spine_annotation_centers(
            localization_result,
            ["vertebrae_L1"],
        )

        self.assertEqual(
            centers[0][1],
            Point3D(1.0, 2.0, 3.0),
        )
        self.assertIn("vertebrae_L1 四棱柱生成失败", warnings[0])

    def test_spine_cpr_skips_malformed_body_without_blocking_other_labels(self):
        """单椎体主体质心损坏时，其他椎体仍应正常进入 CPR。"""

        localization_result = {
            "coords": {},
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "centroid_physical": {"x": 1.0},
                        "prism_found": False,
                        "prism_centroid_physical": None,
                    },
                    "vertebrae_L2": {
                        "found": True,
                        "centroid_physical": {
                            "x": 4.0, "y": 5.0, "z": 6.0,
                        },
                        "prism_found": True,
                        "prism_centroid_physical": {
                            "x": 7.0, "y": 8.0, "z": 9.0,
                        },
                    },
                },
            },
        }

        centers, warnings = _get_spine_annotation_centers(
            localization_result,
            ["vertebrae_L1", "vertebrae_L2"],
        )

        self.assertEqual(
            centers,
            [("vertebrae_L2", Point3D(7.0, 8.0, 9.0))],
        )
        self.assertTrue(any("vertebrae_L1" in warning for warning in warnings))

    def test_spine_cpr_extends_to_first_and_last_vertebra_end_faces(self):
        """缺少主体端面几何时，CPR 首尾应回退到首末标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
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
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
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

    def test_spine_cpr_minimally_corrects_reversed_endpoint_faces(self):
        """反向首末端面只消除矢状负分量，并保留左右坐标和全部质心。"""

        centers = [
            ("vertebrae_L1", Point3D(10.0, 0.0, 10.0)),
            ("vertebrae_L2", Point3D(20.0, 2.0, 8.0)),
            ("vertebrae_L3", Point3D(30.0, 4.0, 6.0)),
        ]
        first_direction = 2.0 ** -0.5
        localization_result = {
            "image": {
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "sagittal_body_faces": [
                            {
                                "axis_index": 0,
                                "side": "min",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": first_direction,
                                    "z": -first_direction,
                                },
                                "center": {
                                    "x": 30.0, "y": -2.0, "z": 12.0,
                                },
                            },
                            {
                                "axis_index": 0,
                                "side": "max",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": -first_direction,
                                    "z": first_direction,
                                },
                                "center": {
                                    "x": 30.0, "y": 2.0, "z": 8.0,
                                },
                            },
                        ],
                    },
                    "vertebrae_L3": {
                        "found": True,
                        "sagittal_body_faces": [
                            {
                                "axis_index": 0,
                                "side": "min",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": -first_direction,
                                    "z": first_direction,
                                },
                                "center": {
                                    "x": -10.0, "y": 6.0, "z": 4.0,
                                },
                            },
                            {
                                "axis_index": 0,
                                "side": "max",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": first_direction,
                                    "z": -first_direction,
                                },
                                "center": {
                                    "x": -10.0, "y": 2.0, "z": 8.0,
                                },
                            },
                        ],
                    },
                },
            },
        }

        path = _build_annotation_centroid_path(localization_result, centers)

        self.assertTrue(self._point_close(
            path[0],
            Point3D(30.0, 0.0, 10.0),
        ))
        self.assertTrue(self._point_close(
            path[-1],
            Point3D(-10.0, 4.0, 6.0),
        ))
        self.assertEqual(path[1:-1], [center for _, center in centers])

    def test_spine_cpr_keeps_aligned_endpoint_faces_unchanged(self):
        """已沿局部弧线向外的端面中心不应被吸附或移动。"""

        centers = [
            ("vertebrae_L1", Point3D(10.0, 0.0, 10.0)),
            ("vertebrae_L2", Point3D(20.0, 2.0, 8.0)),
            ("vertebrae_L3", Point3D(30.0, 4.0, 6.0)),
        ]
        localization_result = {
            "image": {
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": {"x": 30.0, "y": -2.0, "z": 12.0},
                            "max": {"x": 30.0, "y": 2.0, "z": 8.0},
                        },
                    },
                    "vertebrae_L3": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": {"x": -10.0, "y": 2.0, "z": 8.0},
                            "max": {"x": -10.0, "y": 6.0, "z": 4.0},
                        },
                    },
                },
            },
        }

        path = _build_annotation_centroid_path(localization_result, centers)

        self.assertEqual(path[0], Point3D(30.0, -2.0, 12.0))
        self.assertEqual(path[-1], Point3D(-10.0, 6.0, 4.0))
        self.assertEqual(path[1:-1], [center for _, center in centers])

    def test_spine_cpr_projects_endpoint_faces_that_reverse_arc_turn(self):
        """端面虽继续向外但反弧时，应最短投影到质心弧线延长线。"""

        centers = [
            ("vertebrae_L1", Point3D(10.0, 0.0, 10.0)),
            ("vertebrae_L2", Point3D(20.0, 2.0, 8.0)),
            ("vertebrae_L3", Point3D(30.0, 5.0, 6.0)),
        ]
        localization_result = {
            "image": {
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": {"x": 30.0, "y": -4.0, "z": 11.0},
                            "max": {"x": 30.0, "y": 2.0, "z": 8.0},
                        },
                    },
                    "vertebrae_L3": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": {"x": -10.0, "y": 2.0, "z": 8.0},
                            "max": {"x": -10.0, "y": 6.0, "z": 2.0},
                        },
                    },
                },
            },
        }

        path = _build_annotation_centroid_path(localization_result, centers)

        self.assertTrue(self._point_close(
            path[0],
            Point3D(30.0, -2.5, 12.5),
        ))
        self.assertTrue(self._point_close(
            path[-1],
            Point3D(-10.0, 5.0 + 33.0 / 13.0, 6.0 - 22.0 / 13.0),
        ))
        self.assertEqual(path[1:-1], [center for _, center in centers])

    def test_spine_cpr_projects_out_of_range_face_to_outward_centroid_ray(
        self,
    ):
        """端面越出局部弧线区间时，应最短投影到零角度外延线。"""

        inner_angle = math.radians(10.0)
        face_angle = math.radians(20.0)
        centers = [
            ("vertebrae_L1", Point3D(10.0, 0.0, 0.0)),
            ("vertebrae_L2", Point3D(20.0, 0.0, -10.0)),
            (
                "vertebrae_L3",
                Point3D(
                    30.0,
                    -10.0 * math.sin(inner_angle),
                    -10.0 - 10.0 * math.cos(inner_angle),
                ),
            ),
        ]
        raw_face = Point3D(
            30.0,
            -5.0 * math.sin(face_angle),
            5.0 * math.cos(face_angle),
        )
        localization_result = {
            "image": {
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": raw_face.to_dict(),
                            "max": {
                                "x": 30.0, "y": 0.0, "z": -10.0,
                            },
                        },
                    },
                },
            },
        }

        path = _build_annotation_centroid_path(localization_result, centers)

        corrected_delta = np.array([
            path[0].y - centers[0][1].y,
            path[0].z - centers[0][1].z,
        ])
        corrected_angle = math.atan2(
            -corrected_delta[0],
            corrected_delta[1],
        )
        outward_ray = np.array([0.0, 1.0])
        correction = np.array([
            raw_face.y - path[0].y,
            raw_face.z - path[0].z,
        ])

        self.assertAlmostEqual(
            corrected_angle,
            0.0,
            places=6,
        )
        self.assertAlmostEqual(path[0].x, raw_face.x)
        self.assertAlmostEqual(
            float(np.dot(correction, outward_ray)),
            0.0,
            places=6,
        )
        self.assertEqual(path[1:], [center for _, center in centers])

    def test_spine_cpr_keeps_nonzero_face_inside_local_arc_interval(self):
        """端面位于正常非零弧线区间内时不应被吸附到零角度。"""

        inner_angle = math.radians(10.0)
        face_angle = math.radians(5.0)
        centers = [
            ("vertebrae_L1", Point3D(10.0, 0.0, 0.0)),
            ("vertebrae_L2", Point3D(20.0, 0.0, -10.0)),
            (
                "vertebrae_L3",
                Point3D(
                    30.0,
                    -10.0 * math.sin(inner_angle),
                    -10.0 - 10.0 * math.cos(inner_angle),
                ),
            ),
        ]
        raw_face = Point3D(
            30.0,
            -5.0 * math.sin(face_angle),
            5.0 * math.cos(face_angle),
        )
        localization_result = {
            "image": {
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "local_superior_inferior_face_centroids": {
                            "min": raw_face.to_dict(),
                            "max": {
                                "x": 30.0, "y": 0.0, "z": -10.0,
                            },
                        },
                    },
                },
            },
        }

        path = _build_annotation_centroid_path(localization_result, centers)

        self.assertEqual(path[0], raw_face)
        self.assertEqual(path[1:], [center for _, center in centers])

    def test_spine_cpr_extends_existing_endpoint_lines_to_raw_label_exit(self):
        """首尾已有端线只应延伸到白线穿出原始标签的位置。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            localization_result, centers = self._raw_label_exit_case(temp_dir)

            path = _build_annotation_centroid_path(
                localization_result,
                centers,
            )

        self.assertAlmostEqual(path[0].z, 46.0)
        self.assertAlmostEqual(path[-1].z, 13.5)
        self.assertEqual(path[1:-1], [center for _, center in centers])
        for centroid, extended_endpoint in (
            (centers[0][1], path[0]),
            (centers[-1][1], path[-1]),
        ):
            endpoint_direction = normalize_vector(
                subtract_points(extended_endpoint, centroid)
            )
            self.assertEqual(endpoint_direction.x, 0.0)
            self.assertEqual(endpoint_direction.y, 0.0)

    def test_spine_cpr_ignores_single_voxel_hole_when_finding_raw_label_exit(
        self,
    ):
        """原始标签内部的单层孔洞不应让端线提前停止。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            localization_result, centers = self._raw_label_exit_case(
                temp_dir,
                with_single_slice_hole=True,
            )

            path = _build_annotation_centroid_path(
                localization_result,
                centers,
            )

        self.assertAlmostEqual(path[0].z, 46.0)
        self.assertGreater(path[0].z, 44.0)

    def test_spine_cpr_extends_curve_endpoints_to_terminal_raw_label_edges(
        self,
    ):
        """首尾曲线应沿现有切线补足到终端原始标签投影外缘。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            localization_result, _ = self._raw_label_exit_case(temp_dir)
            segmentation = sitk.ReadImage(
                localization_result["seg_for_reconstruction_path"]
            )
            array = sitk.GetArrayFromImage(segmentation)
            array[50, 24, 24] = 1
            array[9, 24, 24] = 3
            sitk.WriteImage(
                sitk.GetImageFromArray(array),
                localization_result["seg_for_reconstruction_path"],
            )
            curve_points = [
                Point3D(20.0, 20.0, 46.0),
                Point3D(20.0, 20.0, 30.0),
                Point3D(20.0, 20.0, 13.5),
            ]

            extended = _extend_curve_endpoints_to_terminal_raw_labels(
                localization_result,
                ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3"],
                curve_points,
            )

        self.assertGreater(extended[0].z, 52.0)
        self.assertLess(extended[-1].z, 7.5)
        self.assertEqual(extended[1], curve_points[1])
        self.assertAlmostEqual(extended[0].x, 20.0)
        self.assertAlmostEqual(extended[-1].y, 20.0)

    def test_spine_cpr_matches_terminal_labels_when_curve_order_is_reversed(
        self,
    ):
        """曲线点顺序反向时，首末原始标签也不能上下互换延伸。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            localization_result, _ = self._raw_label_exit_case(temp_dir)
            segmentation = sitk.ReadImage(
                localization_result["seg_for_reconstruction_path"]
            )
            array = sitk.GetArrayFromImage(segmentation)
            array[50, 24, 24] = 1
            array[9, 24, 24] = 3
            sitk.WriteImage(
                sitk.GetImageFromArray(array),
                localization_result["seg_for_reconstruction_path"],
            )
            curve_points = [
                Point3D(20.0, 20.0, 13.5),
                Point3D(20.0, 20.0, 30.0),
                Point3D(20.0, 20.0, 46.0),
            ]

            extended = _extend_curve_endpoints_to_terminal_raw_labels(
                localization_result,
                ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3"],
                curve_points,
            )

        self.assertLess(extended[0].z, 7.5)
        self.assertGreater(extended[-1].z, 52.0)
        self.assertEqual(extended[1], curve_points[1])

    def test_spine_cpr_keeps_existing_endpoints_without_raw_segmentation(self):
        """缺少原始分割时不得继续使用 bbox 极值延长端点。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            localization_result, centers = self._raw_label_exit_case(temp_dir)
            localization_result.pop("seg_for_reconstruction_path")

            path = _build_annotation_centroid_path(
                localization_result,
                centers,
            )

        self.assertEqual(path[0], Point3D(20.0, 20.0, 42.0))
        self.assertEqual(path[-1], Point3D(20.0, 20.0, 18.0))

    def test_spine_cpr_omits_endpoint_face_collapsed_onto_centroid(self):
        """反向端面修正到端部质心时，不应留下零长度重复锚点。"""

        centers = [
            ("vertebrae_L1", Point3D(10.0, 0.0, 10.0)),
            ("vertebrae_L2", Point3D(10.0, 2.0, 8.0)),
            ("vertebrae_L3", Point3D(10.0, 4.0, 6.0)),
        ]
        first_direction = 2.0 ** -0.5
        localization_result = {
            "image": {
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
            "spine_geometry": {
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {
                        "found": True,
                        "sagittal_body_faces": [
                            {
                                "axis_index": 0,
                                "side": "min",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": first_direction,
                                    "z": -first_direction,
                                },
                                "center": {
                                    "x": 10.0, "y": -2.0, "z": 12.0,
                                },
                            },
                            {
                                "axis_index": 0,
                                "side": "max",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": -first_direction,
                                    "z": first_direction,
                                },
                                "center": {
                                    "x": 10.0, "y": 2.0, "z": 8.0,
                                },
                            },
                        ],
                    },
                    "vertebrae_L3": {
                        "found": True,
                        "sagittal_body_faces": [
                            {
                                "axis_index": 0,
                                "side": "min",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": -first_direction,
                                    "z": first_direction,
                                },
                                "center": {
                                    "x": 10.0, "y": 6.0, "z": 4.0,
                                },
                            },
                            {
                                "axis_index": 0,
                                "side": "max",
                                "point_count": 100,
                                "normal": {
                                    "x": 0.0,
                                    "y": first_direction,
                                    "z": -first_direction,
                                },
                                "center": {
                                    "x": 10.0, "y": 2.0, "z": 8.0,
                                },
                            },
                        ],
                    },
                },
            },
        }

        path = _build_annotation_centroid_path(localization_result, centers)

        self.assertEqual(path, [center for _, center in centers])

    def test_spine_cpr_smooths_annotation_centroid_curve(self):
        """CPR 基准曲线应基于过滤质心圆滑加密，而不是生硬折线。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        y_by_label = {
            "vertebrae_L1": 52.0,
            "vertebrae_L2": 56.0,
            "vertebrae_L3": 60.0,
            "vertebrae_L4": 62.0,
            "vertebrae_L5": 60.0,
            "vertebrae_S1": 56.0,
        }
        for label_name, centroid_y in y_by_label.items():
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
        self.assertGreater(len(plan.curve_points), len(y_by_label))
        for label_name in y_by_label:
            annotation = localization_result[
                "spine_geometry"
            ]["filtered_annotation_centroids"][label_name]
            expected = annotation["centroid_physical"]
            self.assertCurveContainsPoint(
                plan.curve_points,
                Point3D(expected["x"], expected["y"], expected["z"]),
            )

        segment_lengths = [
            np.linalg.norm([
                current.x - previous.x,
                current.y - previous.y,
                current.z - previous.z,
            ])
            for previous, current in zip(plan.curve_points, plan.curve_points[1:])
        ]
        self.assertLess(max(segment_lengths), 1.25)
        signed_offsets = [
            point.y - np.interp(
                point.z,
                [plan.curve_points[-1].z, plan.curve_points[0].z],
                [plan.curve_points[-1].y, plan.curve_points[0].y],
            )
            for point in plan.curve_points[1:-1]
        ]
        self.assertTrue(all(offset >= -1e-6 for offset in signed_offsets))

    def test_cpr_centroid_reverse_correction_keeps_normal_c_curve(self):
        """正常单向 C 型残差方向一致时不应移动质心。"""

        points = [
            Point3D(60.0, y, z)
            for y, z in (
                (50.0, 10.0),
                (51.0, 8.0),
                (54.0, 6.0),
                (59.0, 4.0),
                (66.0, 2.0),
                (75.0, 0.0),
            )
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            1.0,
            Point3D(0.0, 0.0, 1.0),
        )

        self.assertEqual(corrected, points)

    def test_cpr_centroid_correction_limits_terminal_curvature_spike(self):
        """终端前单点残差突增并折返时应收敛，不能形成尖锐内凹。"""

        points = [
            Point3D(0.0, y, z)
            for y, z in (
                (-95.852, -597.822),
                (-106.403, -629.598),
                (-115.775, -663.083),
                (-123.043, -697.367),
                (-125.132, -731.661),
                (-114.963, -762.067),
            )
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            0.9765625,
            Point3D(0.0, 0.0, 1.0),
        )

        incoming = np.asarray([
            corrected[-2].y - corrected[-3].y,
            corrected[-2].z - corrected[-3].z,
        ])
        outgoing = np.asarray([
            corrected[-1].y - corrected[-2].y,
            corrected[-1].z - corrected[-2].z,
        ])
        turn_angle = math.degrees(math.acos(np.clip(
            float(np.dot(incoming, outgoing))
            / float(np.linalg.norm(incoming) * np.linalg.norm(outgoing)),
            -1.0,
            1.0,
        )))
        self.assertGreater(corrected[-2].y, points[-2].y + 1.0)
        self.assertLess(turn_angle, 17.0)
        self.assertEqual(corrected[-1], points[-1])

    def test_cpr_straight_terminal_segments_join_spline_tangentially(self):
        """首尾直线端段与质心样条必须切向连续，不能只把位置拼在一起。"""

        centroids = [
            Point3D(0.0, y, z)
            for y, z in (
                (0.0, 10.0),
                (1.0, 8.0),
                (2.0, 6.0),
                (3.0, 4.0),
                (4.0, 2.0),
            )
        ]
        anchors = [Point3D(0.0, -4.0, 12.0), *centroids,
                   Point3D(0.0, 8.0, 0.0)]

        curve = _smooth_cpr_centroid_path_with_straight_endpoints(
            anchors,
            centroids,
            0.05,
            Point3D(0.0, 0.0, 1.0),
        )

        for centroid in (centroids[0], centroids[-1]):
            index = next(
                point_index
                for point_index, point in enumerate(curve)
                if self._point_close(point, centroid)
            )
            incoming = np.asarray([
                curve[index].x - curve[index - 1].x,
                curve[index].y - curve[index - 1].y,
                curve[index].z - curve[index - 1].z,
            ])
            outgoing = np.asarray([
                curve[index + 1].x - curve[index].x,
                curve[index + 1].y - curve[index].y,
                curve[index + 1].z - curve[index].z,
            ])
            turn_angle = math.degrees(math.acos(np.clip(
                float(np.dot(incoming, outgoing))
                / float(np.linalg.norm(incoming) * np.linalg.norm(outgoing)),
                -1.0,
                1.0,
            )))
            self.assertLess(turn_angle, 1.0)

    def test_cpr_centroid_reverse_correction_removes_middle_wave(self):
        """单个中间点与两侧趋势相反时应回正到双邻点预测位置。"""

        points = [
            Point3D(60.0, y, z)
            for y, z in (
                (50.0, 10.0),
                (50.0, 8.0),
                (56.0, 6.0),
                (50.0, 4.0),
                (50.0, 2.0),
            )
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            1.0,
            Point3D(0.0, 0.0, 1.0),
        )

        self.assertEqual(corrected[0], points[0])
        self.assertEqual(corrected[1], points[1])
        self.assertEqual(corrected[3], points[3])
        self.assertEqual(corrected[4], points[4])
        self.assertEqual(corrected[2], Point3D(60.0, 50.0, 6.0))

    def test_cpr_centroid_reverse_correction_can_adjust_l5_without_moving_s1(self):
        """L5 与前方腰椎趋势相反时可回正，S1 末端仍保持不动。"""

        points = [
            Point3D(60.0, y, z)
            for y, z in (
                (50.0, 10.0),
                (51.0, 8.0),
                (54.0, 6.0),
                (59.0, 4.0),
                (52.0, 2.0),
                (75.0, 0.0),
            )
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            1.0,
            Point3D(0.0, 0.0, 1.0),
        )

        self.assertEqual(corrected[3], points[3])
        self.assertEqual(corrected[-1], points[-1])
        self.assertEqual(corrected[-2], Point3D(60.0, 67.0, 2.0))

    def test_cpr_centroid_reverse_correction_keeps_real_c4_case_detectable(self):
        """颈1实测 C4 反向残差应仍能穿过强度门槛被回正。"""

        points = [
            Point3D(x, y, z)
            for x, y, z in (
                (0.0, -123.411, -8.995),
                (-1.355, -119.567, -33.414),
                (-4.177, -117.943, -46.599),
                (-2.531, -110.432, -63.507),
                (-0.949, -105.716, -78.641),
                (0.633, -102.110, -95.029),
            )
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            0.31640625,
            Point3D(0.0, 0.0, 1.0),
        )

        self.assertEqual(corrected[0], points[0])
        self.assertEqual(corrected[1], points[1])
        self.assertEqual(corrected[3], points[3])
        self.assertEqual(corrected[4], points[4])
        self.assertEqual(corrected[5], points[5])
        self.assertAlmostEqual(corrected[2].x, -1.870255, places=6)
        self.assertAlmostEqual(corrected[2].y, -115.564575, places=6)
        self.assertAlmostEqual(corrected[2].z, -46.599, places=6)

    def test_cpr_centroid_correction_retracts_lumbar_l3_local_corner(self):
        """腰3实测 L3 局部折角应缩回，L2/L4/L5/S1 不被推出。"""

        points = [
            Point3D(x, y, z)
            for x, y, z in (
                (33.277, -99.975, -276.978),
                (33.639, -107.069, -309.942),
                (34.000, -116.791, -342.365),
                (31.290, -121.420, -376.356),
                (29.303, -123.908, -410.553),
                (28.779, -116.026, -436.052),
            )
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            0.361328125,
            Point3D(0.0, 0.0, 1.0),
        )

        self.assertEqual(corrected[0], points[0])
        self.assertEqual(corrected[1], points[1])
        self.assertEqual(corrected[3], points[3])
        self.assertEqual(corrected[4], points[4])
        self.assertEqual(corrected[5], points[5])
        self.assertAlmostEqual(corrected[2].x, 32.492229, places=6)
        self.assertAlmostEqual(corrected[2].y, -114.075090, places=6)
        self.assertAlmostEqual(corrected[2].z, -342.365, places=6)

    def test_cpr_centroid_correction_keeps_smooth_inflection_peak(self):
        """合法 S 型峰值是方向拐点，不应按局部折角缩回。"""

        points = [
            Point3D(60.0, y, z)
            for y, z in (
                (0.0, 10.0),
                (1.0, 8.0),
                (4.0, 6.0),
                (2.0, 4.0),
                (0.0, 2.0),
                (-1.0, 0.0),
            )
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            1.0,
            Point3D(0.0, 0.0, 1.0),
        )

        self.assertEqual(corrected, points)

    def test_cpr_centroid_reverse_correction_falls_back_on_invalid_longitudinal_order(self):
        """纵向顺序异常时保持原质心链，避免隐藏输入错误。"""

        points = [
            Point3D(60.0, 50.0, 10.0),
            Point3D(60.0, 51.0, 8.0),
            Point3D(60.0, 56.0, 8.0),
            Point3D(60.0, 50.0, 6.0),
            Point3D(60.0, 50.0, 2.0),
        ]

        corrected = _correct_reversing_cpr_centroids(
            points,
            1.0,
            Point3D(0.0, 0.0, 1.0),
        )

        self.assertEqual(corrected, points)

    def test_spine_cpr_reverse_centroid_correction_removes_anchor_wave(self):
        """CPR 规划应绕过反向中间质心，而不是让 Localizer 穿过波峰。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        y_by_label = {
            "vertebrae_L1": 50.0,
            "vertebrae_L2": 50.0,
            "vertebrae_L3": 56.0,
            "vertebrae_L4": 50.0,
            "vertebrae_L5": 50.0,
            "vertebrae_S1": 50.0,
        }
        for label_name, centroid_y in y_by_label.items():
            coord = localization_result["coords"][label_name]
            center_z = float(coord["centroid"]["z"])
            localization_result.setdefault("spine_geometry", {}).setdefault(
                "filtered_annotation_centroids", {}
            )[label_name] = {
                "found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": centroid_y,
                    "z": center_z,
                },
            }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertGreater(
            self._min_distance_to_curve(
                plan.curve_points,
                Point3D(60.0, 56.0, 6.0),
            ),
            1.0,
        )
        self.assertCurveContainsPoint(
            plan.curve_points,
            Point3D(60.0, 50.0, 6.0),
        )

    def test_spine_cpr_builds_terminal_path_after_centroid_correction(self):
        """端面和标签出口必须使用回正后的首尾质心方向，且只回正一次。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        corrected = [
            Point3D(60.0, 40.0 + index, 10.0 - index * 2.0)
            for index in range(6)
        ]

        with patch(
            "src.planning.spine_cpr._correct_reversing_cpr_centroids",
            return_value=corrected,
        ) as correction, patch(
            "src.planning.spine_cpr._build_annotation_centroid_path",
            side_effect=lambda _localization, centers: [
                center for _, center in centers
            ],
        ) as build_path:
            plan = LumbarWholeRangePlanner().build_plans(
                request,
                localization_result,
            ).plans[0]

        self.assertEqual(correction.call_count, 1)
        self.assertEqual(
            [center for _, center in build_path.call_args.args[1]],
            corrected,
        )
        self.assertCurveContainsPoint(plan.curve_points, corrected[-2])

    def test_spine_cpr_baseline_ignores_local_segments(self):
        """基准 CPR 不使用椎体局部线段，避免端点造成折返。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
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

        self.assertGreater(len(plan.curve_points), 6)
        self.assertCurveContainsPoint(
            plan.curve_points,
            Point3D(60.0, 50.0, 6.0),
        )

    def test_spine_cpr_uses_body_face_centers_and_all_annotation_centroids(self):
        """CPR 锚点应包含首尾端面中点和全部过滤标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
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

        # 解剖顺序从头侧到足侧时，物理 z 是递减的，所以 min/max 需要按路径方向解释。
        self.assertGreater(len(plan.curve_points), 8)
        self.assertEqual(plan.curve_points[0], Point3D(60.0, 50.0, 11.5))
        self.assertEqual(plan.curve_points[-1], Point3D(60.0, 50.0, -1.5))
        self.assertTrue(all(abs(point.y - 50.0) <= 1e-6 for point in plan.curve_points))

    def test_spine_cpr_uses_sacrum_as_s1_alias(self):
        """数据集只有 sacrum 时，腰椎 CPR 仍应保留第六个标注质心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
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

        self.assertGreater(len(plan.curve_points), 6)
        self.assertEqual(plan.source_labels[-1], "sacrum")
        self.assertAlmostEqual(plan.curve_points[-1].z, 0.0)

    def test_spine_cpr_corrects_reversed_face_and_keeps_all_centroids_as_anchors(self):
        """反向端面最小回正后，全部质心仍应作为 CPR 硬锚点。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
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

        self.assertTrue(self._point_close(
            plan.curve_points[0],
            Point3D(60.0, 50.0, 10.0),
        ))
        self.assertTrue(self._point_close(
            plan.curve_points[-1],
            Point3D(
                60.0,
                53.0 + 93.0 / 13.0,
                -62.0 / 13.0,
            ),
        ))

        self.assertGreater(len(plan.curve_points), 8)
        for label_name, z in (
            ("vertebrae_L1", 10.0),
            ("vertebrae_L2", 8.0),
            ("vertebrae_L3", 6.0),
            ("vertebrae_L4", 4.0),
            ("vertebrae_L5", 2.0),
            ("vertebrae_S1", 0.0),
        ):
            self.assertCurveContainsPoint(
                plan.curve_points,
                Point3D(60.0, y_by_label[label_name], z),
            )

        first_centroid = Point3D(
            60.0, y_by_label["vertebrae_L1"], 10.0,
        )
        second_centroid = Point3D(
            60.0, y_by_label["vertebrae_L2"], 8.0,
        )
        previous_centroid = Point3D(
            60.0, y_by_label["vertebrae_L5"], 2.0,
        )
        last_centroid = Point3D(
            60.0, y_by_label["vertebrae_S1"], 0.0,
        )
        first_index = next(
            index
            for index, point in enumerate(plan.curve_points)
            if self._point_close(point, first_centroid)
        )
        last_index = next(
            index
            for index, point in enumerate(plan.curve_points)
            if self._point_close(point, last_centroid)
        )

        first_tangent = np.array([
            second_centroid.y - first_centroid.y,
            second_centroid.z - first_centroid.z,
        ])
        first_segment = np.array([
            first_centroid.y - plan.curve_points[0].y,
            first_centroid.z - plan.curve_points[0].z,
        ])
        for previous, current in zip(
            plan.curve_points[:first_index],
            plan.curve_points[1:first_index + 1],
        ):
            delta = np.array([
                current.y - previous.y,
                current.z - previous.z,
            ])
            self.assertGreaterEqual(
                float(np.dot(delta, first_tangent)),
                -1e-6,
            )
            relative = np.array([
                current.y - plan.curve_points[0].y,
                current.z - plan.curve_points[0].z,
            ])
            self.assertAlmostEqual(
                float(
                    relative[0] * first_segment[1]
                    - relative[1] * first_segment[0]
                ),
                0.0,
                places=6,
            )

        last_tangent = np.array([
            last_centroid.y - previous_centroid.y,
            last_centroid.z - previous_centroid.z,
        ])
        last_segment = np.array([
            plan.curve_points[-1].y - last_centroid.y,
            plan.curve_points[-1].z - last_centroid.z,
        ])
        for previous, current in zip(
            plan.curve_points[last_index:-1],
            plan.curve_points[last_index + 1:],
        ):
            delta = np.array([
                current.y - previous.y,
                current.z - previous.z,
            ])
            self.assertGreaterEqual(
                float(np.dot(delta, last_tangent)),
                -1e-6,
            )
            relative = np.array([
                current.y - last_centroid.y,
                current.z - last_centroid.z,
            ])
            self.assertAlmostEqual(
                float(
                    relative[0] * last_segment[1]
                    - relative[1] * last_segment[0]
                ),
                0.0,
                places=6,
            )

    def test_spine_cpr_terminal_face_follows_last_annotation_centroid(self):
        """末端路径应由最后一个标注质心直接连接到对立端面中心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 1.0,
                    ["骨窗"],
                    slice_type=SliceType.CPR,
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
        self.assertCurveContainsPoint(
            plan.curve_points,
            Point3D(60.0, y_by_label["vertebrae_S1"], 0.0),
        )

    def test_cervical_sagittal_stays_straight_and_coronal_uses_cpr(self):
        """颈椎矢状位保持直线 MPR，冠状位走 CPR。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.颈椎,
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
                    slice_type=SliceType.CPR,
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

        localization_result = self._spine_localization_result(
            ("C1", "C2", "C3", "C4", "C5", "C6", "C7"),
            "颈椎",
        )
        self.assertEqual(
            configured_spine_cpr_labels(localization_result),
            [f"vertebrae_C{index}" for index in range(2, 8)],
        )

    def test_cervical_interspace_starts_at_c2_c3_and_keeps_five_gaps(self):
        """颈椎横状位不生成环椎 C1 与 C2 之间的区间。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.颈椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=2.0,
                slice_spacing=2.0,
                window_presets=["骨窗"],
                slice_type=SliceType.椎间隙,
            )],
        )
        localization_result = self._spine_localization_result(
            ("C1", "C2", "C3", "C4", "C5", "C6", "C7"),
            "颈椎",
        )
        for index, coord in enumerate(localization_result["coords"].values()):
            z = 24.0 - index * 4.0
            coord["centroid"]["z"] = z
            coord["bbox"]["z_min"] = int(z) - 1
            coord["bbox"]["z_max"] = int(z) + 1

        with patch(
            "src.planning.cervical.regularize_cervical_interspace_segments",
            side_effect=lambda _, segments: (list(segments), []),
        ) as regularize:
            result = CervicalInterspacePlanner().build_plans(
                request,
                localization_result,
            )

        regularize.assert_called_once()
        self.assertEqual(
            [plan.group_name for plan in result.plans],
            ["C2_C3", "C3_C4", "C4_C5", "C5_C6", "C6_C7"],
        )
        self.assertNotIn("C1_C2", [plan.group_name for plan in result.plans])
        self.assertEqual(
            configured_interspace_definitions(localization_result)[0],
            ("C2_C3", "vertebrae_C2", "vertebrae_C3"),
        )
        self.assertTrue(all(
            "vertebrae_C1" not in plan.source_labels
            for plan in result.plans
        ))

    def test_select_planner_supports_cervical_interspace(self):
        """颈椎椎间隙请求必须选择专用规划器。"""

        planner = select_planner("颈椎", SliceType.椎间隙.value)

        self.assertIsInstance(planner, CervicalInterspacePlanner)

    def test_select_planner_supports_configured_whole_range_slice_parts(self):
        """验证 `test_select_planner_supports_configured_whole_range_slice_parts` 场景符合预期。"""

        config_path = Path("src/config/slice_part_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        whole_range_slice_parts = [
            slice_part
            for slice_part, part_config in config["slice_parts"].items()
            if part_config["enabled"]
            and any(
                region["type"] == "whole_range"
                for region in part_config["localization"]["regions"].values()
            )
        ]

        self.assertEqual(set(whole_range_slice_parts), {
            "颈椎", "胸椎", "腰椎", "肺部", "头骨", "髋骨",
        })

        for slice_part in whole_range_slice_parts:
            with self.subTest(slice_part=slice_part):
                planner = select_planner(slice_part, SliceType.MPR.value)
                if slice_part == SlicePartEnum.腰椎.value:
                    self.assertIsInstance(planner, LumbarWholeRangePlanner)
                else:
                    self.assertIsInstance(planner, SingleLabelWholeRangePlanner)

    def test_checked_in_range_extension_keeps_configured_defaults(self):
        """验证当前默认值，不限制后续按需增加其他部位或状位。"""

        config_path = Path("src/config/slice_part_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for slice_part in ("肺部", "髋骨"):
            with self.subTest(slice_part=slice_part):
                rule = config["slice_parts"][slice_part]["slice_rules"][
                    "横状位"
                ]["MPR"]
                self.assertEqual(
                    rule["range_extension"],
                    {"start_mm": 3.0, "end_mm": 3.0},
                )

    def test_checked_in_localizer_planes_cover_every_slice_rule(self):
        """每个状位与切法都应显式配置不同于重建面的定位图状位。"""

        config = load_slice_part_config()
        for slice_part, part in config["slice_parts"].items():
            for plane, type_rules in part["slice_rules"].items():
                for slice_type, rule in type_rules.items():
                    with self.subTest(
                        slice_part=slice_part,
                        plane=plane,
                        slice_type=slice_type,
                    ):
                        self.assertIn(
                            rule["localizer_plane"],
                            {"横状位", "矢状位", "冠状位"},
                        )
                        self.assertNotEqual(rule["localizer_plane"], plane)
        self.assertEqual(
            config["slice_parts"]["肺部"]["slice_rules"]["横状位"]
            ["MPR"]["localizer_plane"],
            "冠状位",
        )

    def test_checked_in_interspace_regions_are_selected_by_slice_rules(self):
        """椎间隙输出名称、顺序和端点统一由区域及切法引用决定。"""

        config = json.loads(
            Path("src/config/slice_part_config.json").read_text(encoding="utf-8")
        )
        expected_counts = {"颈椎": 5, "腰椎": 5}
        for slice_part, expected_count in expected_counts.items():
            with self.subTest(slice_part=slice_part):
                part = config["slice_parts"][slice_part]
                selected = part["slice_rules"]["横状位"]["椎间隙"]["regions"]
                self.assertEqual(len(selected), expected_count)
                self.assertTrue(all(
                    part["localization"]["regions"][name]["type"] == "interspace"
                    for name in selected
                ))
        self.assertEqual(
            config["slice_parts"]["颈椎"]["slice_rules"]["横状位"]
            ["椎间隙"]["regions"],
            [f"C{index}_C{index + 1}" for index in range(2, 7)],
        )

    def test_single_label_whole_range_planner_builds_axial_plan_from_bbox(self):
        """验证 `test_single_label_whole_range_planner_builds_axial_plan_from_bbox` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
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
            "slice_part": "肺部",
            "localization_profile": "whole_range",
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
        localization_result.update(
            self._configured_localization_contract("肺部")
        )
        localization_result["slice_rules"]["横状位"]["MPR"][
            "range_extension"
        ] = {"start_mm": 0.0, "end_mm": 0.0}

        result = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        )

        self.assertEqual(result.planner_name, "SingleLabelWholeRangePlanner")
        self.assertEqual(len(result.plans), 1)
        plan = result.plans[0]
        self.assertEqual(plan.plan_id, "肺部_MPR_横状位")
        self.assertEqual(len(plan.slices), 3)
        self.assertEqual(
            [planned_slice.center.z for planned_slice in plan.slices],
            [10.0, 5.0, 0.0],
        )
        self.assertEqual(plan.window_presets, ["肺窗"])
        self.assertEqual(plan.localizer_plane, "冠状位")

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
            slice_part=SlicePartEnum.肺部,
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
            **self._configured_localization_contract("肺部"),
            "slice_part": "肺部",
            "localization_profile": "whole_range",
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
        expected_count = int(np.ceil((extended_length - 5.0) / 5.0)) + 1
        expected_total_range = 5.0 + (expected_count - 1) * 5.0
        expected_range_overflow = expected_total_range - extended_length
        expected_min_index = (
            mask_min_z
            - configured_end_extension
            - expected_range_overflow / 2.0
            + 2.5
            - origin_z
        ) / spacing_z
        expected_max_index = (
            mask_max_z
            + configured_start_extension
            + expected_range_overflow / 2.0
            - 2.5
            - origin_z
        ) / spacing_z

        self.assertEqual(plan.source_labels, ["anatomical_geometry.global_frame"])
        self.assertEqual(len(center_indices), expected_count)
        self.assertLessEqual(
            min(center_indices) * spacing_z - 2.5,
            mask_min_index * spacing_z,
        )
        self.assertGreaterEqual(
            max(center_indices) * spacing_z + 2.5,
            mask_max_index * spacing_z,
        )
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
            slice_part=SlicePartEnum.肺部,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=4.0,
                slice_spacing=10.0,
                window_presets=["肺窗"],
            )],
        )
        localization_result = {
            "slice_part": "肺部",
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

        request.plane_configs[0].fill_percent = 100
        target_plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]
        self.assertEqual(
            target_plan.extent_mm.to_dict(),
            plan.extent_mm.to_dict(),
        )
        # FillPercent 由第七阶段按整组应用，不改第六阶段采样平面。
        self.assertEqual(target_plan.range_start, plan.range_start)

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
            slice_part=SlicePartEnum.颈椎,
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
            "slice_part": "颈椎",
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

    def test_non_lung_keeps_default_axes_even_if_anatomical_geometry_is_present(self):
        """未接入专属坐标的部位即使携带解剖几何，也继续使用默认图像轴。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.头骨,
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
            slice_part=SlicePartEnum.胸椎,
            plane_configs=[
                PlaneConfig(PlaneTypeEnum.横状位, 2.0, 5.0, ["骨窗"]),
                PlaneConfig(PlaneTypeEnum.矢状位, 2.0, 5.0, ["骨窗"]),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["slice_part"] = "胸椎"
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

    def test_thoracic_coronal_mpr_keeps_spine_target_range(self):
        """胸椎冠状位 MPR 应使用胸椎段 bbox，且不进入 CPR。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.胸椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 2.0, 5.0, ["骨窗"],
                ),
            ],
        )
        localization_result = self._spine_localization_result(
            ("C7", "T1", "T2", "T12", "L1"),
            "胸椎",
        )

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.slice_type, "MPR")
        self.assertEqual(plan.geometry_type, "plane_stack")
        self.assertEqual(plan.localizer_plane, "矢状位")
        self.assertEqual(plan.source_labels, [
            "vertebrae_C7",
            "vertebrae_T1",
            "vertebrae_T2",
            "vertebrae_T12",
            "vertebrae_L1",
        ])

    def test_thoracic_coronal_uses_cpr_curve_on_sagittal_localizer(self):
        """胸椎冠状位应按 CPR 曲线规划，并在矢状位定位图显示 T 段。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.胸椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 2.0, 5.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
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
        for label_name, z_value in (("vertebrae_C7", 85.0), ("vertebrae_L1", 0.0)):
            localization_result["coords"][label_name]["bbox"].update({
                "z_min": int(z_value), "z_max": int(z_value) + 5,
                "y_min": 40, "y_max": 60,
                "x_min": 45, "x_max": 55,
            })
            localization_result["coords"][label_name]["centroid"].update({
                "z": z_value, "y": 50.0, "x": 50.0,
            })
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
        self.assertEqual(plan.slice_type, "CPR")
        self.assertEqual(plan.group_name, "CPR")
        self.assertEqual(plan.plane, "冠状位")
        self.assertEqual(plan.localizer_plane, "矢状位")
        self.assertEqual(plan.source_labels, [
            "vertebrae_C7",
            "vertebrae_T1",
            "vertebrae_T2",
            "vertebrae_T3",
            "vertebrae_T12",
            "vertebrae_L1",
        ])
        self.assertEqual(plan.offset_axis.to_dict(), {"x": 0.0, "y": 1.0, "z": 0.0})
        self.assertIn(0.0, plan.offsets_mm)
        self.assertGreater(len(plan.offsets_mm), 1)
        self.assertAlmostEqual(plan.extent_mm.width, 10.0)
        self.assertAlmostEqual(plan.extent_mm.height, 85.0)
        self.assertAlmostEqual(plan.localizer_reference.x, 50.0)
        self.assertAlmostEqual(plan.localizer_reference.z, 48.333333333333336)

    def test_thoracic_coronal_cpr_uses_configured_c7_l1_range(self):
        """胸椎冠状位 CPR 应按配置使用 C7 到 L1 的完整范围。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.胸椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 2.0, 5.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )
        localization_result = self._spine_localization_result(
            ("C7", "T1", "T2", "T3", "L1"),
            "胸椎",
        )
        localization_result["resolved_cut_config"] = {
            "start": "C7",
            "end": "L1",
            "start_labels": ["vertebrae_C7"],
            "end_labels": ["vertebrae_L1"],
        }
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
        self.assertEqual(plan.source_labels, [
            "vertebrae_C7",
            "vertebrae_T1",
            "vertebrae_T2",
            "vertebrae_T3",
            "vertebrae_L1",
        ])
        self.assertGreater(len(plan.curve_points), 3)
        for z in (100.0, 30.0, 20.0, 10.0, -100.0):
            self.assertCurveContainsPoint(
                plan.curve_points,
                Point3D(60.0, 50.0, z),
            )

    def test_spine_cpr_offsets_cover_original_mask_with_asymmetric_counts(self):
        """CPR offset 张数应按原始标注体素外缘计算，且两侧可不对称。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            segmentation = np.zeros((32, 32, 32), dtype=np.uint8)
            segmentation[8:24, 8:14, 10:22] = 1
            segmentation[8:24, 14:26, 10:22] = 2
            segmentation[8:24, 26:31, 10:22] = 3
            segmentation_image = sitk.GetImageFromArray(segmentation)
            segmentation_path = Path(temp_dir) / "raw_seg.nii.gz"
            sitk.WriteImage(segmentation_image, str(segmentation_path))

            localization_result = self._spine_localization_result(
                ("T1", "T2", "T3"),
                "胸椎",
            )
            localization_result["seg_for_reconstruction_path"] = str(
                segmentation_path
            )
            for label_name, label_id in (
                ("vertebrae_T1", 1),
                ("vertebrae_T2", 2),
                ("vertebrae_T3", 3),
            ):
                localization_result["coords"][label_name]["label_ids"] = [
                    label_id
                ]
                localization_result["coords"][label_name]["bbox"].update({
                    "x_min": 8, "x_max": 23,
                    "y_min": 14, "y_max": 17,
                    "z_min": 10, "z_max": 21,
                })
                localization_result["coords"][label_name]["centroid"].update({
                    "x": 16.0,
                    "y": 16.0,
                })

            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir="/tmp/output",
                slice_part=SlicePartEnum.胸椎,
                plane_configs=[
                    PlaneConfig(
                        PlaneTypeEnum.冠状位, 4.0, 5.0, ["骨窗"],
                        slice_type=SliceType.CPR,
                    ),
                ],
            )

            plan = SingleLabelWholeRangePlanner().build_plans(
                request,
                localization_result,
            ).plans[0]

            self.assertEqual(
                plan.offsets_mm,
                [-7.0, -2.0, 3.0, 8.0, 13.0],
            )
            self.assertEqual(
                [planned.index for planned in plan.slices],
                [1, 2, 3, 4, 5],
            )
            self.assertEqual(
                [planned.offset_mm for planned in plan.slices],
                plan.offsets_mm,
            )

    def test_spine_cpr_offsets_keep_raw_label_extent_when_filtered_prism_exists(self):
        """四棱柱决定 CPR 中心线，但 offset 范围仍应覆盖原始完整标注。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            segmentation = np.zeros((24, 128, 128), dtype=np.uint8)
            segmentation[8:16, 10:101, 55:66] = 1
            segmentation_path = Path(temp_dir) / "raw_seg.nii.gz"
            sitk.WriteImage(
                sitk.GetImageFromArray(segmentation),
                str(segmentation_path),
            )

            localization_result = self._spine_localization_result(
                ("L1", "L2", "L3"),
                "腰椎",
            )
            localization_result["seg_for_reconstruction_path"] = str(
                segmentation_path
            )
            localization_result["spine_geometry"] = {
                "filtered_annotation_centroids": {}
            }
            for label_name in ("vertebrae_L1", "vertebrae_L2", "vertebrae_L3"):
                localization_result["coords"][label_name]["label_ids"] = [1]
                localization_result["spine_geometry"][
                    "filtered_annotation_centroids"
                ][label_name] = {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": 10.0},
                    "prism_found": True,
                    "prism_bbox": {
                        "z_min": 8,
                        "z_max": 12,
                        "y_min": 45,
                        "y_max": 55,
                        "x_min": 55,
                        "x_max": 65,
                    },
                }

            offsets, sampling = _resolve_sampling_offsets_mm(
                localization_result,
                ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3"],
                [
                    Point3D(60.0, 50.0, 16.0),
                    Point3D(60.0, 50.0, 12.0),
                    Point3D(60.0, 50.0, 8.0),
                ],
                Point3D(0.0, 1.0, 0.0),
                slice_thickness=4.0,
                slice_spacing=5.0,
                slice_image_count=None,
            )

        self.assertEqual(
            offsets,
            [float(value) for value in range(-40, 55, 5)],
        )
        self.assertEqual(sampling.slice_image_count, 19)

    def test_spine_cpr_full_label_false_uses_filtered_prism_extent(self):
        """full_label=false 时，CPR offset 应按四棱柱/主体范围生成。"""

        localization_result = self._spine_localization_result(
            ("L1", "L2", "L3"),
            "腰椎",
        )
        localization_result["spine_cpr_config"] = {"full_label": False}
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {}
        }
        for index, label_name in enumerate(("vertebrae_L1", "vertebrae_L2", "vertebrae_L3")):
            z_center = 12.0 - index * 4.0
            localization_result["coords"][label_name]["bbox"].update({
                "y_min": 10,
                "y_max": 100,
            })
            localization_result["spine_geometry"][
                "filtered_annotation_centroids"
            ][label_name] = {
                "found": True,
                "centroid_physical": {"x": 60.0, "y": 50.0, "z": z_center},
                "prism_found": True,
                "prism_bbox": {
                    "z_min": int(z_center) - 2,
                    "z_max": int(z_center) + 2,
                    "y_min": 45,
                    "y_max": 55,
                    "x_min": 55,
                    "x_max": 65,
                },
            }
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 4.0, 5.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.cpr_range_mode, "body_prism")
        self.assertEqual(plan.plan_id, "腰椎_CPR_冠状位_body_prism")
        self.assertEqual(plan.group_name, "CPR_主椎体")
        self.assertEqual(plan.offsets_mm, [-5.0, 0.0, 5.0])

    def test_spine_cpr_body_prism_offsets_use_local_curve_reference(self):
        """body_prism offset 应相对局部曲线计算，避免胸椎弧度放大范围。"""

        localization_result = self._spine_localization_result(
            ("C7", "T1", "T2", "T3", "T4"),
            "胸椎",
        )
        curve_points = [
            Point3D(60.0, y, z)
            for y, z in (
                (-150.0, 100.0),
                (-135.0, 75.0),
                (-120.0, 50.0),
                (-105.0, 25.0),
                (-90.0, 0.0),
            )
        ]
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {}
        }
        for label_name, curve_point in zip(
            (
                "vertebrae_C7",
                "vertebrae_T1",
                "vertebrae_T2",
                "vertebrae_T3",
                "vertebrae_T4",
            ),
            curve_points,
        ):
            center_y = int(round(curve_point.y))
            center_z = int(round(curve_point.z))
            localization_result["spine_geometry"][
                "filtered_annotation_centroids"
            ][label_name] = {
                "found": True,
                "centroid_physical": {
                    "x": curve_point.x,
                    "y": curve_point.y,
                    "z": curve_point.z,
                },
                "prism_found": True,
                "prism_bbox": {
                    "z_min": center_z - 1,
                    "z_max": center_z + 1,
                    "y_min": center_y - 10,
                    "y_max": center_y + 10,
                    "x_min": 55,
                    "x_max": 65,
                },
            }

        offsets, sampling = _resolve_sampling_offsets_mm(
            localization_result,
            [
                "vertebrae_C7",
                "vertebrae_T1",
                "vertebrae_T2",
                "vertebrae_T3",
                "vertebrae_T4",
            ],
            curve_points,
            Point3D(0.0, 1.0, 0.0),
            slice_thickness=5.0,
            slice_spacing=5.0,
            slice_image_count=None,
            range_mode="body_prism",
        )

        self.assertEqual(
            offsets,
            [-7.5, -2.5, 2.5, 7.5],
        )
        self.assertEqual(sampling.slice_image_count, 4)

    def test_spine_cpr_body_prism_uses_oriented_prism_vertices(self):
        """歪斜椎体不能用轴对齐 bbox 角点拼出不存在的 offset。"""

        localization_result = self._spine_localization_result(
            ("S1",),
            "腰椎",
        )
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {
                "vertebrae_S1": {
                    "found": True,
                    "prism_found": True,
                    "prism_bbox": {
                        "z_min": 0,
                        "z_max": 10,
                        "y_min": 10,
                        "y_max": 22,
                        "x_min": 58,
                        "x_max": 62,
                    },
                    "sagittal_body_quadrilateral": {
                        "vertices": [
                            {"x": 60.0, "y": 10.0, "z": 0.0},
                            {"x": 60.0, "y": 12.0, "z": 0.0},
                            {"x": 60.0, "y": 20.0, "z": 10.0},
                            {"x": 60.0, "y": 22.0, "z": 10.0},
                        ],
                        "left_right_extent_mm": {
                            "min": -2.0,
                            "max": 2.0,
                        },
                    },
                },
            },
        }
        interval = _filtered_annotation_offset_interval_mm(
            localization_result,
            ["vertebrae_S1"],
            [
                Point3D(60.0, 0.0, 0.0),
                Point3D(60.0, 10.0, 10.0),
                Point3D(60.0, 20.0, 20.0),
            ],
            Point3D(0.0, 1.0, 0.0),
        )

        self.assertEqual(interval, (10.0, 12.0))

    def test_spine_cpr_full_label_true_keeps_full_extent_and_stable_identity(self):
        """full_label=true 时，保持整体范围和稳定的 CPR 输出命名。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            segmentation = np.zeros((24, 128, 128), dtype=np.uint8)
            segmentation[8:16, 10:101, 55:66] = 1
            segmentation_path = Path(temp_dir) / "raw_seg.nii.gz"
            sitk.WriteImage(
                sitk.GetImageFromArray(segmentation),
                str(segmentation_path),
            )
            localization_result = self._spine_localization_result(
                ("L1", "L2", "L3"),
                "腰椎",
            )
            localization_result["seg_for_reconstruction_path"] = str(
                segmentation_path
            )
            localization_result["spine_cpr_config"] = {"full_label": True}
            localization_result["spine_geometry"] = {
                "filtered_annotation_centroids": {}
            }
            for index, label_name in enumerate(("vertebrae_L1", "vertebrae_L2", "vertebrae_L3")):
                z_center = 16.0 - index * 4.0
                localization_result["coords"][label_name]["label_ids"] = [1]
                localization_result["spine_geometry"][
                    "filtered_annotation_centroids"
                ][label_name] = {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": z_center},
                    "prism_found": True,
                    "prism_bbox": {
                        "z_min": int(z_center) - 2,
                        "z_max": int(z_center) + 2,
                        "y_min": 45,
                        "y_max": 55,
                        "x_min": 55,
                        "x_max": 65,
                    },
                }
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir="/tmp/output",
                slice_part=SlicePartEnum.腰椎,
                plane_configs=[
                    PlaneConfig(
                        PlaneTypeEnum.冠状位, 4.0, 5.0, ["骨窗"],
                        slice_type=SliceType.CPR,
                    ),
                ],
            )

            plans = SingleLabelWholeRangePlanner().build_plans(
                request,
                localization_result,
            ).plans

        self.assertEqual(
            [plan.cpr_range_mode for plan in plans],
            ["full_label"],
        )
        self.assertEqual(
            [plan.group_name for plan in plans],
            ["CPR"],
        )
        self.assertEqual(
            [plan.plan_id for plan in plans],
            ["腰椎_CPR_冠状位"],
        )
        self.assertGreater(len(plans[0].offsets_mm), 3)

    def test_checked_in_spine_cpr_full_label_config_lives_under_spine_targets(self):
        """CPR 范围配置只放在三个脊柱部位下，且必须使用 bool。"""

        config_path = Path("src/config/slice_part_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertNotIn("spine_cpr", config)
        for slice_part in ("颈椎", "胸椎", "腰椎"):
            with self.subTest(slice_part=slice_part):
                self.assertIsInstance(
                    config["slice_parts"][slice_part]["localization"]
                    ["spine_cpr"].get("full_label"),
                    bool,
                )
        for slice_part in ("肺部", "头骨", "髋骨"):
            with self.subTest(slice_part=slice_part):
                self.assertNotIn(
                    "spine_cpr",
                    config["slice_parts"][slice_part]["localization"],
                )

    def test_checked_in_spine_cpr_uses_full_label_range(self):
        """当前脊柱 CPR 配置统一覆盖完整标注范围。"""

        config_path = Path("src/config/slice_part_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(
            config["slice_parts"]["胸椎"]["localization"]
            ["spine_cpr"]["full_label"],
        )
        self.assertTrue(
            config["slice_parts"]["腰椎"]["localization"]
            ["spine_cpr"]["full_label"],
        )

    def test_spine_cpr_rejects_missing_full_label(self):
        """CPR 定位配置缺少 full_label 时应明确失败。"""

        localization_result = self._spine_localization_result(
            ("L1", "L2", "L3"),
            "腰椎",
        )
        localization_result["spine_cpr_config"] = {"range_mode": "body_prism"}
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 4.0, 5.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "spine_cpr_config.full_label 必须是布尔值",
        ):
            SingleLabelWholeRangePlanner().build_plans(
                request,
                localization_result,
            )

    def test_spine_cpr_offsets_read_coronal_whole_range_extension(self):
        """冠状位 CPR 应用整切冠状位扩展增加左右 offset 范围。"""

        localization_result = self._spine_localization_result(
            ("L1", "L2", "L3"),
            "腰椎",
        )
        localization_result["slice_rules"]["冠状位"]["CPR"][
            "range_extension"
        ] = {"start_mm": 4.0, "end_mm": 6.0}
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 4.0, 5.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.geometry_type, "curved_stack")
        self.assertEqual(plan.offsets_mm, [-9.0, -4.0, 1.0, 6.0, 11.0])
        self.assertAlmostEqual(plan.extent_mm.width, 10.0)

    def test_lumbar_coronal_cpr_uses_configured_whole_range_subset(self):
        """腰椎 CPR 应按配置范围取椎体，不应固定使用 L1 到 S1。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["resolved_cut_config"] = {
            "start": "L2",
            "end": "L5",
            "start_labels": ["vertebrae_L2"],
            "end_labels": ["vertebrae_L5"],
        }
        localization_result["resolved_cut_configs"]["whole_range"] = (
            localization_result["resolved_cut_config"]
        )

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.geometry_type, "curved_stack")
        self.assertEqual(plan.source_labels, [
            "vertebrae_L2",
            "vertebrae_L3",
            "vertebrae_L4",
            "vertebrae_L5",
        ])

    def test_lumbar_cpr_uses_whole_range_in_mixed_interspace_request(self):
        """混合请求以椎间隙定位时，CPR 仍使用独立的整段范围。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    PlaneTypeEnum.冠状位, 1.0, 3.0, ["骨窗"],
                    slice_type=SliceType.CPR,
                ),
            ],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["localization_profile"] = "interspace"
        localization_result["resolved_cut_config"] = {
            "intervals": [
                {
                    "start_labels": ["vertebrae_L1"],
                    "end_labels": ["vertebrae_L2"],
                },
            ],
        }
        localization_result["resolved_cut_configs"] = {
            "whole_range": {
                "start_labels": ["vertebrae_L2"],
                "end_labels": ["vertebrae_L5"],
            },
            "interspace": localization_result["resolved_cut_config"],
        }

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.source_labels, [
            "vertebrae_L2",
            "vertebrae_L3",
            "vertebrae_L4",
            "vertebrae_L5",
        ])

    def test_hip_only_tilts_axial_and_sets_three_localizer_planes(self):
        """髋骨仅横状位使用 AB 角度，矢冠保持默认轴并指定定位背景。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.髋骨,
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
        localization_result.update(
            self._configured_localization_contract("髋骨")
        )
        localization_result["slice_part"] = "髋骨"
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

    def test_single_label_planner_builds_sagittal_whole_range_plan(self):
        """单目标整切 planner 应把矢状位交给通用直线构造器。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
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
            slice_part=SlicePartEnum.肺部,
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
            slice_part=SlicePartEnum.腰椎,
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
                extent = plane_bbox_extent_mm(
                    bbox,
                    image_info,
                    plane,
                )
                self.assertEqual(extent.to_dict(), expected)

    def test_projected_extent_uses_selected_region_and_plane_frame(self):
        """直线范围投影应根据整面或部位框返回对应宽高和中心。"""

        from src.planning.straight import (
            default_straight_frame,
            resolve_projected_extent,
            project_region_extent,
            bbox_physical_corners,
        )

        localization_result = self._lumbar_localization_result()
        localization_result["image"]["size_xyz"] = [100, 120, 30]
        frame = default_straight_frame(
            PlaneTypeEnum.横状位,
            localization_result["image"],
        )

        full_extent = resolve_projected_extent(
            localization_result,
            localization_result["correction_scope_bbox"],
            frame,
        )
        target_extent = project_region_extent(
            bbox_physical_corners(localization_result["correction_scope_bbox"], localization_result["image"]),
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
            slice_part=SlicePartEnum.腰椎,
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
            slice_type="MPR",
            plane_config=request.plane_configs[0],
            localization_result=localization_result,
            segment=segment,
            frame=frame,
        )

        self.assertEqual(plan.slices[0].center.x, 49.5)
        self.assertEqual(plan.slices[0].center.y, 59.5)
        self.assertEqual(plan.slices[0].center.z, 10.0)
        self.assertEqual(plan.extent_mm.to_dict(), {
            "width": 99.0,
            "height": 119.0,
        })

    def test_interspace_plan_keeps_shortest_gap_at_vertebral_position(self):
        """椎间切不能把最近点沿平面方向搬到整幅图像中心。"""

        from src.models import Point3D
        from src.planning.straight import (
            CutSegment,
            build_straight_slice_plan,
            default_straight_frame,
        )

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=1.0,
                slice_spacing=1.0,
                window_presets=["骨窗"],
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["image"]["size_xyz"] = [200, 200, 40]
        frame = default_straight_frame(
            PlaneTypeEnum.横状位,
            localization_result["image"],
        )
        segment = CutSegment(
            group_name="L1_L2",
            target_start=Point3D(20.0, 30.0, 20.0),
            target_end=Point3D(20.0, 30.0, 18.0),
            target_region_bbox=localization_result["correction_scope_bbox"],
            source_labels=("vertebrae_L1", "vertebrae_L2"),
            recenter_target_in_plane=False,
        )

        plan = build_straight_slice_plan(
            request=request,
            slice_type="椎间隙",
            plane_config=request.plane_configs[0],
            localization_result=localization_result,
            segment=segment,
            frame=frame,
        )

        self.assertAlmostEqual(plan.target_start.x, 20.0)
        self.assertAlmostEqual(plan.target_end.x, 20.0)
        self.assertAlmostEqual(plan.target_start.y, 30.0)
        self.assertAlmostEqual(plan.extent_mm.width, 358.0)
        self.assertAlmostEqual(plan.extent_mm.height, 338.0)

    def test_interspace_uses_convex_prism_closest_pair_and_perpendicular_normal(self):
        """椎间线应由两个四棱柱最近点决定，法向必须与最近连线一致。"""

        from src.models import Point3D
        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        coords = localization_result["coords"]
        for label_name in (
            "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1",
        ):
            coords[label_name]["found"] = False
        coords["vertebrae_L1"]["bbox"].update({"z_min": 6, "z_max": 10})
        coords["vertebrae_L2"]["bbox"].update({"z_min": 0, "z_max": 4})
        annotations = {}
        for label_name, vertices, center in (
            (
                "vertebrae_L1",
                [(0, 10), (2, 10), (2, 6), (0, 6)],
                (1, 8),
            ),
            (
                "vertebrae_L2",
                [(4, 4), (6, 4), (6, 0), (4, 0)],
                (5, 2),
            ),
        ):
            annotations[label_name] = {
                "found": True,
                "prism_found": True,
                "centroid_physical": {"x": 20.0, "y": center[0], "z": center[1]},
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        {"x": 20.0, "y": y, "z": z}
                        for y, z in vertices
                    ],
                    "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(localization_result)

        self.assertEqual(
            [segment.group_name for segment in segments],
            ["L1_L2"],
        )
        self.assertFalse(any("L1_L2" in warning for warning in warnings))
        segment = segments[0]
        self.assertFalse(segment.recenter_target_in_plane)
        vector = Point3D(
            segment.target_end.x - segment.target_start.x,
            segment.target_end.y - segment.target_start.y,
            segment.target_end.z - segment.target_start.z,
        )
        length = (vector.x ** 2 + vector.y ** 2 + vector.z ** 2) ** 0.5
        self.assertAlmostEqual(length, 2.0 * 2.0 ** 0.5, places=6)
        self.assertAlmostEqual(vector.y / length, 2.0 ** 0.5 / 2.0, places=6)
        self.assertAlmostEqual(vector.z / length, -2.0 ** 0.5 / 2.0, places=6)
        self.assertAlmostEqual(
            segment.frame.normal.x * vector.x
            + segment.frame.normal.y * vector.y
            + segment.frame.normal.z * vector.z,
            length,
            places=6,
        )

        localization_result["image"]["direction"][0] = -1.0
        reflected_segments, reflected_warnings = (
            resolve_lumbar_interspace_segments(localization_result)
        )
        self.assertEqual(
            [item.group_name for item in reflected_segments],
            ["L1_L2"],
        )
        self.assertIsNotNone(reflected_segments[0].frame)
        self.assertFalse(any(
            "L1_L2" in warning
            for warning in reflected_warnings
        ))

    def test_interspace_closest_pair_rejects_crossing_prism_polygons(self):
        """边相交但顶点互不包含的四边形也属于重叠，不能伪造非零间隙。"""

        from src.planning.spine_common import _closest_convex_polygon_pair_2d

        horizontal = [(-3.0, -1.0), (3.0, -1.0), (3.0, 1.0), (-3.0, 1.0)]
        vertical = [(-1.0, -3.0), (1.0, -3.0), (1.0, 3.0), (-1.0, 3.0)]

        self.assertIsNone(
            _closest_convex_polygon_pair_2d(horizontal, vertical),
        )

    def test_interspace_facing_pair_uses_full_shared_midpoint_for_small_overlap(self):
        """共同中点仍有足够间隙时，不能把切片中心拉向正间隙一端。"""

        from src.planning.spine_common import (
            _facing_convex_polygon_boundary_pair_2d,
        )

        upper = [
            (-118.26528292399925, -90.54306294485141),
            (-96.08122018642437, -83.28981372543835),
            (-97.19533011351777, -69.20201696556389),
            (-114.90242903822754, -71.93928749669388),
        ]
        lower = [
            (-110.38675671206288, -104.37629001222695),
            (-93.26371015102883, -99.78155479906933),
            (-92.84843218229823, -84.01150535360331),
            (-110.79563093203059, -87.87899157998513),
        ]

        pair = _facing_convex_polygon_boundary_pair_2d(upper, lower)

        self.assertIsNotNone(pair)
        upper_point, lower_point, _ = pair
        self.assertAlmostEqual(
            (upper_point[0] + lower_point[0]) / 2.0,
            -103.33403352661477,
            places=6,
        )
        self.assertAlmostEqual(
            (upper_point[1] + lower_point[1]) / 2.0,
            -85.97040727160639,
            places=6,
        )

    def test_interspace_facing_pair_keeps_wider_positive_subrange(self):
        """共同中点间隙明显变窄时，仍使用更可靠的正间隙子区间。"""

        from src.planning.spine_common import (
            _facing_convex_polygon_boundary_pair_2d,
        )

        upper = [
            (-130.5633441505035, -48.838557129339236),
            (-112.833983375, -34.48777944333965),
            (-112.833983375, -23.891889132339184),
            (-128.8847361128766, -25.9484475253361),
        ]
        lower = [
            (-125.17370432990435, -53.000001),
            (-111.35713977564123, -53.000001),
            (-109.02013944764123, -38.749999),
            (-122.50209284383028, -38.749999),
        ]

        pair = _facing_convex_polygon_boundary_pair_2d(upper, lower)

        self.assertIsNotNone(pair)
        upper_point, lower_point, squared_gap = pair
        self.assertAlmostEqual(
            (upper_point[0] + lower_point[0]) / 2.0,
            -115.08961066169228,
            places=6,
        )
        self.assertGreater(math.sqrt(squared_gap), 2.0)

    def test_cervical_low_volume_body_limits_abnormal_interspace_normal(self):
        """低体积颈椎主体不能把椎间法向拉离主体中心线超过十度。"""

        from src.models import Point3D
        from src.planning.spine_common import (
            regularize_cervical_interspace_segments,
        )
        from src.planning.straight import CutSegment, StraightPlaneFrame

        annotations = {
            "vertebrae_C3": {
                "found": True,
                "voxel_count": 500,
                "centroid_physical": {"x": 0.0, "y": -2.0, "z": 20.0},
            },
            "vertebrae_C4": {
                "found": True,
                "voxel_count": 370,
                "centroid_physical": {"x": 0.0, "y": 0.0, "z": 10.0},
            },
            "vertebrae_C5": {
                "found": True,
                "voxel_count": 600,
                "centroid_physical": {"x": 0.0, "y": 4.0, "z": 0.0},
            },
        }
        localization_result = {
            "slice_part": "颈椎",
            "spine_geometry": {
                "filtered_annotation_centroids": annotations,
            },
        }
        original = CutSegment(
            group_name="C4_C5",
            target_start=Point3D(0.0, 2.0, 6.0),
            target_end=Point3D(0.0, 2.0, 4.0),
            target_region_bbox={
                "x_min": 0,
                "x_max": 1,
                "y_min": 0,
                "y_max": 1,
                "z_min": 0,
                "z_max": 1,
            },
            source_labels=("vertebrae_C4", "vertebrae_C5"),
            frame=StraightPlaneFrame(
                row_dir=Point3D(1.0, 0.0, 0.0),
                col_dir=Point3D(0.0, -1.0, 0.0),
                normal=Point3D(0.0, 0.0, -1.0),
            ),
            recenter_target_in_plane=False,
        )

        corrected, warnings = regularize_cervical_interspace_segments(
            localization_result,
            [original],
        )

        self.assertEqual(len(corrected), 1)
        centerline = normalize_vector(Point3D(0.0, 4.0, -10.0))
        normal = corrected[0].frame.normal
        deviation = math.degrees(math.acos(max(-1.0, min(1.0,
            normal.x * centerline.x
            + normal.y * centerline.y
            + normal.z * centerline.z
        ))))
        self.assertAlmostEqual(deviation, 10.0, places=6)
        self.assertAlmostEqual(
            math.dist(
                tuple(corrected[0].target_start.to_dict().values()),
                tuple(corrected[0].target_end.to_dict().values()),
            ),
            2.0,
            places=6,
        )
        self.assertTrue(any("C4_C5" in warning for warning in warnings))

        annotations["vertebrae_C4"]["voxel_count"] = 380
        unchanged, unchanged_warnings = (
            regularize_cervical_interspace_segments(
                localization_result,
                [original],
            )
        )
        self.assertEqual(unchanged[0].frame.normal, original.frame.normal)
        self.assertEqual(unchanged_warnings, [])

    def test_interspace_body_pair_translates_underestimated_prism_gap(self):
        """真实主体间隙明显更大时只平移切片，不改变长度和方向。"""

        from src.planning.spine_common import (
            _translate_interspace_to_body_pair,
        )
        from src.planning.straight import StraightPlaneFrame

        localization_result = {
            **self._configured_localization_contract("颈椎"),
            "slice_part": "颈椎",
            "spine_geometry": {
                "interspace_body_pairs": {
                    "C6_C7": {
                        "found": True,
                        "minimum_distance_mm": 2.0,
                        "support_point_count": 12,
                        "midpoint_physical": {
                            "x": 0.0,
                            "y": -4.0,
                            "z": -2.0,
                        },
                    },
                },
            },
        }
        frame = StraightPlaneFrame(
            row_dir=Point3D(1.0, 0.0, 0.0),
            col_dir=Point3D(0.0, -1.0, 0.0),
            normal=Point3D(0.0, 0.0, -1.0),
        )
        target_start = Point3D(0.0, 0.0, 0.25)
        target_end = Point3D(0.0, 0.0, -0.25)

        translated_start, translated_end, warning = (
            _translate_interspace_to_body_pair(
                localization_result,
                "C6_C7",
                target_start,
                target_end,
                frame,
            )
        )

        self.assertEqual(translated_start, Point3D(0.0, 0.0, -1.75))
        self.assertEqual(translated_end, Point3D(0.0, 0.0, -2.25))
        self.assertAlmostEqual(
            math.dist(
                tuple(translated_start.to_dict().values()),
                tuple(translated_end.to_dict().values()),
            ),
            0.5,
        )
        self.assertIn("C6_C7", warning)

        localization_result["spine_geometry"]["interspace_body_pairs"][
            "C6_C7"
        ]["minimum_distance_mm"] = 1.5
        unchanged_start, unchanged_end, unchanged_warning = (
            _translate_interspace_to_body_pair(
                localization_result,
                "C6_C7",
                target_start,
                target_end,
                frame,
            )
        )
        self.assertEqual(unchanged_start, target_start)
        self.assertEqual(unchanged_end, target_end)
        self.assertIsNone(unchanged_warning)

    def test_terminal_cervical_interspace_moves_toward_c7_from_high_center(self):
        """C6_C7 高于相邻主体中面时只向 C7 侧受限平移。"""

        from src.planning.spine_common import (
            _translate_interspace_to_body_pair,
        )
        from src.planning.straight import StraightPlaneFrame

        localization_result = {
            **self._configured_localization_contract("颈椎"),
            "slice_part": "颈椎",
            "spine_geometry": {
                "interspace_body_pairs": {
                    "C6_C7": {
                        "found": True,
                        "minimum_distance_mm": 0.8,
                        "support_point_count": 12,
                        "midpoint_physical": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                        },
                        "upper_label": "vertebrae_C6",
                        "lower_label": "vertebrae_C7",
                    },
                },
                "filtered_annotation_centroids": {
                    "vertebrae_C6": {
                        "centroid_physical": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 1.0,
                        },
                    },
                    "vertebrae_C7": {
                        "centroid_physical": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": -1.6,
                        },
                    },
                },
            },
        }
        frame = StraightPlaneFrame(
            row_dir=Point3D(1.0, 0.0, 0.0),
            col_dir=Point3D(0.0, -1.0, 0.0),
            normal=Point3D(0.0, 0.0, -1.0),
        )
        target_start = Point3D(0.0, 0.0, 0.25)
        target_end = Point3D(0.0, 0.0, -0.25)

        translated_start, translated_end, warning = (
            _translate_interspace_to_body_pair(
                localization_result,
                "C6_C7",
                target_start,
                target_end,
                frame,
            )
        )

        self.assertEqual(translated_start, Point3D(0.0, 0.0, -0.55))
        self.assertEqual(translated_end, Point3D(0.0, 0.0, -1.05))
        self.assertIn("向 C7 侧", warning)

        localization_result["spine_geometry"][
            "filtered_annotation_centroids"
        ]["vertebrae_C7"]["centroid_physical"]["z"] = -1.4
        unchanged_start, unchanged_end, unchanged_warning = (
            _translate_interspace_to_body_pair(
                localization_result,
                "C6_C7",
                target_start,
                target_end,
                frame,
            )
        )
        self.assertEqual(unchanged_start, target_start)
        self.assertEqual(unchanged_end, target_end)
        self.assertIsNone(unchanged_warning)

    def test_interspace_overlapping_prisms_keep_ordered_boundary_pair(self):
        """外包络轻微穿入时仍按上下相向边界保留椎间区间。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        for label_name in (
            "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1",
        ):
            localization_result["coords"][label_name]["found"] = False
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 6,
            "z_max": 10,
        })
        localization_result["coords"]["vertebrae_L2"]["bbox"].update({
            "z_min": 0,
            "z_max": 4,
        })
        annotations = {}
        for label_name, vertices, center_z in (
            (
                "vertebrae_L1",
                [(-4.0, 6.0), (4.0, 6.0), (4.0, 0.0), (-4.0, 0.0)],
                3.0,
            ),
            (
                "vertebrae_L2",
                [(-3.0, -4.0), (5.0, -4.0), (5.0, 2.0), (-3.0, 2.0)],
                -1.0,
            ),
        ):
            annotations[label_name] = {
                "found": True,
                "prism_found": True,
                "centroid_physical": {"x": 60.0, "y": 50.0, "z": center_z},
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        {"x": 60.0, "y": y, "z": z}
                        for y, z in vertices
                    ],
                    "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertFalse(any("L1_L2" in warning for warning in warnings))
        self.assertFalse(segments[0].recenter_target_in_plane)
        self.assertIsNotNone(segments[0].frame)
        normal = segments[0].frame.normal
        self.assertLess(normal.z, 0.0)
        self.assertGreater(abs(normal.z), abs(normal.y))

    def test_interspace_wedge_uses_shared_face_middle_instead_of_corner(self):
        """楔形椎间隙应取相向端面的共同中段，不能塌缩到最近角点。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        for label_name in (
            "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1",
        ):
            localization_result["coords"][label_name]["found"] = False
        annotations = {}
        for label_name, vertices, center in (
            (
                "vertebrae_L1",
                [(-10.0, 12.0), (10.0, 10.0),
                 (10.0, 20.0), (-10.0, 20.0)],
                (0.0, 15.5),
            ),
            (
                "vertebrae_L2",
                [(-10.0, -10.0), (10.0, -10.0),
                 (10.0, 9.0), (-10.0, 5.0)],
                (0.0, -1.5),
            ),
        ):
            annotations[label_name] = {
                "found": True,
                "prism_found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": center[0],
                    "z": center[1],
                },
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        {"x": 60.0, "y": y, "z": z}
                        for y, z in vertices
                    ],
                    "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertFalse(any("L1_L2" in warning for warning in warnings))
        segment = segments[0]
        midpoint_y = (segment.target_start.y + segment.target_end.y) / 2.0
        gap_length = math.sqrt(
            (segment.target_end.y - segment.target_start.y) ** 2
            + (segment.target_end.z - segment.target_start.z) ** 2
        )
        self.assertAlmostEqual(midpoint_y, 0.0, delta=1.0)
        self.assertGreater(gap_length, 3.5)
        self.assertLess(segment.frame.normal.z, -0.9)

    def test_lumbar3_l5_s1_overlapping_prisms_keep_si_dominant_normal(self):
        """腰3_1mm 的 L5_S1 外包络相交时不能把短横差放大为切面主方向。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        for label_name in (
            "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
        ):
            localization_result["coords"][label_name]["found"] = False

        centers = {
            "vertebrae_L5": (-123.10293044891787, -410.31570840052825),
            "vertebrae_S1": (-116.59805918909188, -434.89392781679226),
        }
        quadrilaterals = {
            "vertebrae_L5": [
                (-140.01093044891786, -426.04945555956414),
                (-104.64912718593234, -419.60182431960976),
                (-105.07298892474044, -394.1231547015047),
                (-140.0109304489179, -401.3750337079068),
            ],
            "vertebrae_S1": [
                (-137.10976984825587, -434.29396779511086),
                (-118.75482443937416, -456.84273637031487),
                (-99.48807646578821, -433.9378718149762),
                (-105.80409339429733, -412.08530237074734),
            ],
        }
        annotations = {}
        for label_name in ("vertebrae_L5", "vertebrae_S1"):
            center_y, center_z = centers[label_name]
            localization_result["coords"][label_name]["centroid"].update({
                "y": center_y,
                "z": center_z,
            })
            annotations[label_name] = {
                "found": True,
                "prism_found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": center_y,
                    "z": center_z,
                },
                "sagittal_body_quadrilateral": {
                    "center": {
                        "x": 60.0,
                        "y": center_y,
                        "z": center_z,
                    },
                    "vertices": [
                        {"x": 60.0, "y": y, "z": z}
                        for y, z in quadrilaterals[label_name]
                    ],
                    "left_right_extent_mm": {"min": -20.0, "max": 20.0},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual(
            [segment.group_name for segment in segments],
            ["L5_S1"],
        )
        self.assertFalse(any("L5_S1" in warning for warning in warnings))
        normal = segments[0].frame.normal
        center_direction = np.asarray([
            centers["vertebrae_S1"][0] - centers["vertebrae_L5"][0],
            centers["vertebrae_S1"][1] - centers["vertebrae_L5"][1],
        ])
        center_direction /= np.linalg.norm(center_direction)
        alignment = (
            normal.y * center_direction[0]
            + normal.z * center_direction[1]
        )
        self.assertGreater(alignment, 0.9)
        self.assertGreater(abs(normal.z), abs(normal.y))
        gap_length = math.sqrt(
            (segments[0].target_end.y - segments[0].target_start.y) ** 2
            + (segments[0].target_end.z - segments[0].target_start.z) ** 2
        )
        self.assertGreater(gap_length, 3.0)

    def test_interspace_invalid_quadrilateral_does_not_resurrect_face_geometry(self):
        """存在但相交的四边形无效时，不能再用旧端面中心绕过碰撞检查。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        for label_name in (
            "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1",
        ):
            localization_result["coords"][label_name]["found"] = False
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 8,
            "z_max": 10,
        })
        localization_result["coords"]["vertebrae_L2"]["bbox"].update({
            "z_min": 4,
            "z_max": 6,
        })
        annotations = {}
        for label_name, vertices, center_z in (
            (
                "vertebrae_L1",
                [(-3.0, 7.0), (3.0, 7.0), (3.0, 9.0), (-3.0, 9.0)],
                9.0,
            ),
            (
                "vertebrae_L2",
                [(-1.0, 5.0), (1.0, 5.0), (1.0, 11.0), (-1.0, 11.0)],
                5.0,
            ),
        ):
            annotations[label_name] = {
                "found": True,
                "prism_found": True,
                "centroid_physical": {"x": 60.0, "y": 50.0, "z": center_z},
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        {"x": 60.0, "y": 50.0 + y, "z": z}
                        for y, z in vertices
                    ],
                    "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                },
                "sagittal_body_faces": [
                    {
                        "axis_index": 0,
                        "side": "min",
                        "point_count": 100,
                        "center": {"x": 60.0, "y": 50.0, "z": center_z - 1.0},
                        "normal": {"x": 0.0, "y": 0.0, "z": -1.0},
                    },
                    {
                        "axis_index": 0,
                        "side": "max",
                        "point_count": 100,
                        "center": {"x": 60.0, "y": 50.0, "z": center_z + 1.0},
                        "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
                    },
                ],
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertIsNone(segments[0].frame)
        self.assertTrue(segments[0].recenter_target_in_plane)
        self.assertTrue(any("L1_L2" in warning for warning in warnings))

    def test_interspace_prism_rejects_nonfinite_frame_and_concave_polygon(self):
        """非有限图像方向或非凸四边形必须被拒绝，不能产生 NaN 坐标系。"""

        from src.planning.spine_common import (
            _parse_sagittal_prism,
            _resolve_interspace_prism_geometry,
        )
        from src.planning.straight import default_straight_frame

        localization_result = self._lumbar_localization_result()
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {
                label_name: {
                    "found": True,
                    "prism_found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": center_z},
                    "sagittal_body_quadrilateral": {
                        "vertices": [
                            {"x": 60.0, "y": y, "z": z}
                            for y, z in vertices
                        ],
                        "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                    },
                }
                for label_name, center_z, vertices in (
                    (
                        "vertebrae_L1",
                        9.0,
                        [(45.0, 8.0), (55.0, 8.0), (50.0, 9.0), (55.0, 10.0)],
                    ),
                    (
                        "vertebrae_L2",
                        5.0,
                        [(45.0, 4.0), (55.0, 4.0), (55.0, 6.0), (45.0, 6.0)],
                    ),
                )
            },
        }
        frame = default_straight_frame(
            PlaneTypeEnum.横状位,
            localization_result["image"],
        )
        upper = localization_result["spine_geometry"][
            "filtered_annotation_centroids"
        ]["vertebrae_L1"]
        self.assertIsNone(_parse_sagittal_prism(upper, frame))

        localization_result["image"]["direction"][0] = math.nan
        self.assertIsNone(_resolve_interspace_prism_geometry(
            localization_result,
            "vertebrae_L1",
            "vertebrae_L2",
        ))

    def test_interspace_rejects_quadrilateral_when_prism_was_invalidated(self):
        """第五阶段判定四棱柱碰撞失效后，第六阶段不能继续使用残留四边形。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        for label_name in (
            "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1",
        ):
            localization_result["coords"][label_name]["found"] = False
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 8,
            "z_max": 10,
        })
        localization_result["coords"]["vertebrae_L2"]["bbox"].update({
            "z_min": 4,
            "z_max": 6,
        })
        annotations = {}
        for label_name, z_min, z_max in (
            ("vertebrae_L1", 8.0, 10.0),
            ("vertebrae_L2", 4.0, 6.0),
        ):
            annotations[label_name] = {
                "found": True,
                "prism_found": False,
                "centroid_physical": {
                    "x": 60.0,
                    "y": 50.0,
                    "z": (z_min + z_max) / 2.0,
                },
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        {"x": 60.0, "y": y, "z": z}
                        for y, z in (
                            (45.0, z_min),
                            (55.0, z_min),
                            (55.0, z_max),
                            (45.0, z_max),
                        )
                    ],
                    "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertIsNone(segments[0].frame)
        self.assertTrue(segments[0].recenter_target_in_plane)
        self.assertTrue(any("L1_L2" in warning for warning in warnings))

    def test_interspace_missing_label_bbox_recenters_centroid_fallback(self):
        """四棱柱有效但 label bbox 缺失时，质心回退不能保留 prism 定位标记。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        for label_name in (
            "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1",
        ):
            localization_result["coords"][label_name]["found"] = False
        localization_result["coords"]["vertebrae_L1"].pop("bbox")
        annotations = {}
        for label_name, center_z in (
            ("vertebrae_L1", 10.0),
            ("vertebrae_L2", 8.0),
        ):
            annotations[label_name] = {
                "found": True,
                "prism_found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": 50.0,
                    "z": center_z,
                },
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        {"x": 60.0, "y": y, "z": z}
                        for y, z in (
                            (45.0, center_z - 0.5),
                            (55.0, center_z - 0.5),
                            (55.0, center_z + 0.5),
                            (45.0, center_z + 0.5),
                        )
                    ],
                    "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertIsNone(segments[0].frame)
        self.assertTrue(segments[0].recenter_target_in_plane)
        self.assertTrue(any("L1_L2" in warning for warning in warnings))

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
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=2.0,
                slice_spacing=2.0,
                window_presets=["骨窗"],
                fill_percent=100,
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["slice_rules"]["横状位"]["MPR"][
            "range_extension"
        ] = {"start_mm": 1.0, "end_mm": 3.0}
        inverse_sqrt_two = 2 ** -0.5
        frame = StraightPlaneFrame(
            row_dir=Point3D(1.0, 0.0, 0.0),
            col_dir=Point3D(0.0, inverse_sqrt_two, -inverse_sqrt_two),
            normal=Point3D(0.0, inverse_sqrt_two, inverse_sqrt_two),
        )
        segment = CutSegment(
            group_name="L1_L2",
            target_start=Point3D(5.0, 5.0, 10.0),
            target_end=Point3D(5.0, 5.0, 0.0),
            target_region_bbox=localization_result["correction_scope_bbox"],
            source_labels=("vertebrae_L1", "vertebrae_L2"),
        )

        plan = build_straight_slice_plan(
            request=request,
            slice_type="MPR",
            plane_config=request.plane_configs[0],
            localization_result=localization_result,
            segment=segment,
            frame=frame,
        )

        target_projection_length = 10.0 * inverse_sqrt_two
        center_projection_length = (
            (plan.range_start.x - plan.range_end.x) * frame.normal.x
            + (plan.range_start.y - plan.range_end.y) * frame.normal.y
            + (plan.range_start.z - plan.range_end.z) * frame.normal.z
        )
        planned_projection_length = center_projection_length + plan.slice_thickness
        self.assertAlmostEqual(planned_projection_length, 12.0)
        self.assertGreaterEqual(
            planned_projection_length,
            target_projection_length,
        )
        self.assertEqual(len(plan.slices), 6)
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
            slice_part=SlicePartEnum.腰椎,
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
        self.assertEqual(plan.slices[0].center.z, 10.0)

    def test_axial_whole_range_defaults_to_full_plane_extent(self):
        """验证 `test_axial_whole_range_defaults_to_full_plane_extent` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom", output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
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

    def test_fill_percent_leaves_base_sampling_extent_unchanged(self):
        """取景比例由重建整组应用，规划仍保留原始采样视野。"""

        request = Request(
            dicom_dir="/tmp/dicom", output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位, slice_thickness=3.0,
                slice_spacing=5.0, window_presets=["肺窗"],
                fill_percent=100,
            )],
        )
        loc = self._lumbar_localization_result()
        loc["image"]["size_xyz"] = [100, 120, 30]
        loc["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]

        plan = SingleLabelWholeRangePlanner().build_plans(request, loc).plans[0]

        self.assertEqual(plan.extent_mm.to_dict(), {"width": 99.0, "height": 238.0})
        self.assertEqual(plan.to_dict()["fill_percent"], 100)

    def test_axial_whole_range_symmetrically_expands_to_cover_both_ends(self):
        """不能整除间距的整切范围应增加一张并在首尾对称外扩。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
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
            "slice_part": "肺部",
            "localization_profile": "whole_range",
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
        self.assertEqual(plan.range_start.z, 10.0)
        self.assertEqual(plan.range_end.z, 0.0)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            10.0,
            5.0,
            0.0,
        ])

    def test_axial_whole_range_short_nonzero_target_uses_one_thick_slice(self):
        """短于层厚的非零有效范围使用一张厚层切片完整覆盖。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
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

        self.assertEqual(plan.range_start.z, 5.0)
        self.assertEqual(plan.range_end.z, 5.0)
        self.assertEqual(
            [planned_slice.center.z for planned_slice in plan.slices],
            [5.0],
        )

    def test_lumbar_whole_range_planner_expands_slices_from_l1_to_s1(self):
        """验证 `test_lumbar_whole_range_planner_expands_slices_from_l1_to_s1` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
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
        self.assertEqual(plan.plan_id, "腰椎_MPR_横状位")
        self.assertEqual(len(plan.slices), 3)
        self.assertEqual(plan.slices[0].center.z, 10.0)
        self.assertEqual(plan.slices[-1].center.z, 0.0)

    def test_lumbar_whole_range_axial_uses_outer_bbox_z_range(self):
        """验证 `test_lumbar_whole_range_axial_uses_outer_bbox_z_range` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
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
        self.assertEqual(plan.range_start.z, 10.0)
        self.assertEqual(plan.range_end.z, 0.0)
        self.assertEqual([planned_slice.center.z for planned_slice in plan.slices], [
            10.0,
            5.0,
            0.0,
        ])

    def test_lumbar_whole_range_axial_uses_bbox_even_without_endpoint_centroids(self):
        """验证 `test_lumbar_whole_range_axial_uses_bbox_even_without_endpoint_centroids` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
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
        self.assertEqual(plan.range_start.z, 10.0)
        self.assertEqual(plan.range_end.z, 0.0)
        self.assertEqual(plan.source_labels, ["correction_scope_bbox"])

    def test_lumbar_whole_range_planner_builds_three_straight_planes(self):
        """腰椎整切也应复用通用构造器生成三个默认直线状位。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
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

    def test_lumbar_sagittal_localizer_uses_body_center_without_moving_slices(self):
        """矢状整切定位图应经过椎体，整平面成片仍保持原图中心。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.矢状位,
                slice_thickness=3.0,
                slice_spacing=5.0,
                window_presets=["骨窗"],
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["image"]["size_xyz"] = [120, 220, 40]
        self._add_lumbar_localizer_body_geometry(localization_result)

        plan = LumbarWholeRangePlanner().build_plans(
            request,
            localization_result,
        ).plans[0]

        self.assertEqual(plan.localizer_plane, "冠状位")
        self.assertEqual(plan.localizer_reference.to_dict(), {
            "x": 61.0,
            "y": 52.5,
            "z": 5.0,
        })
        self.assertTrue(all(
            planned_slice.center.y == 109.5
            for planned_slice in plan.slices
        ))
        self.assertNotEqual(
            plan.slices[0].center.y,
            plan.localizer_reference.y,
        )

    def test_lumbar_interspace_planner_builds_one_plan_per_available_interval(self):
        """验证 `test_lumbar_interspace_planner_builds_one_plan_per_available_interval` 场景符合预期。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
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
            "L1_L2",
            "L2_L3",
            "L3_L4",
            "L4_L5",
            "L5_S1",
        ])
        self.assertTrue(all(plan.slices for plan in result.plans))

    def test_lumbar_interspace_count_applies_to_every_interval(self):
        """数量 20 表示每个有效椎间隙分别生成 20 张。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=2.0,
                slice_spacing=2.0,
                slice_image_count=20,
                window_presets=["骨窗"],
                slice_type=SliceType.椎间隙,
            )],
        )

        result = LumbarInterspacePlanner().build_plans(
            request,
            self._lumbar_localization_result(),
        )

        self.assertEqual(len(result.plans), 5)
        self.assertTrue(all(len(plan.slices) == 20 for plan in result.plans))
        self.assertEqual(sum(len(plan.slices) for plan in result.plans), 100)

    def test_straight_plan_calculates_missing_spacing_from_count(self):
        """给定层厚和数量后，每个计划根据自己的有效范围计算层间距。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=5.0,
                slice_spacing=None,
                slice_image_count=4,
                window_presets=["肺窗"],
            )],
        )

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            self._lumbar_localization_result(),
        ).plans[0]

        self.assertEqual(len(plan.slices), 4)
        self.assertEqual(plan.slice_thickness, 5.0)
        self.assertEqual(plan.slice_spacing, 2.0)
        self.assertEqual(plan.sampling_details["effective_range_mm"], 11.0)
        self.assertEqual(plan.sampling_details["total_range_mm"], 11.0)

    def test_cpr_count_generates_exact_number_of_offsets(self):
        """CPR 数量直接决定每套偏移曲面的实际输出张数。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.冠状位,
                slice_thickness=2.0,
                slice_spacing=2.0,
                slice_image_count=20,
                window_presets=["骨窗"],
                slice_type=SliceType.CPR,
            )],
        )

        plan = SingleLabelWholeRangePlanner().build_plans(
            request,
            self._spine_localization_result(("L1", "L2", "L3"), "腰椎"),
        ).plans[0]

        self.assertEqual(len(plan.slices), 20)
        self.assertEqual(len(plan.offsets_mm), 20)
        self.assertEqual(plan.sampling_details["slice_image_count"], 20)

    def test_planning_names_multiple_configs_for_same_plane_in_order(self):
        """同一状位多套配置必须生成唯一计划编号和连续输出名称。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.肺部,
            plane_configs=[
                PlaneConfig(PlaneTypeEnum.横状位, 5.0, 5.0, ["肺窗"]),
                PlaneConfig(
                    PlaneTypeEnum.横状位,
                    5.0,
                    None,
                    ["肺窗"],
                    slice_image_count=4,
                ),
            ],
        )
        context = ProcessingContext(request)
        self._set_planning_inputs(context, self._lumbar_localization_result())

        PlanningStage().execute(context)

        plans = context.get_result("planning")["plans"]
        self.assertEqual(len(plans), 2)
        self.assertEqual(len({plan["plan_id"] for plan in plans}), 2)
        self.assertEqual(
            [plan["output_plane"] for plan in plans],
            ["横状位", "横状位1"],
        )

    def test_plane_config_ids_are_global_in_request_order(self):
        """配置编号在整个请求内唯一，不按状位分别重新编号。"""

        plane_configs = [
            PlaneConfig(PlaneTypeEnum.横状位, 5.0, 5.0, ["骨窗"]),
            PlaneConfig(PlaneTypeEnum.矢状位, 5.0, 5.0, ["骨窗"]),
            PlaneConfig(PlaneTypeEnum.冠状位, 5.0, 5.0, ["骨窗"]),
        ]

        PlanningStage._assign_plane_config_ids(plane_configs)

        self.assertEqual(
            [config.config_id for config in plane_configs],
            ["配置01", "配置02", "配置03"],
        )
        self.assertEqual(
            [config.output_plane for config in plane_configs],
            ["横状位", "矢状位", "冠状位"],
        )

    def test_repeated_plane_output_names_increment_from_one(self):
        """重复状位依次使用原名、后缀1和后缀2。"""

        plane_configs = [
            PlaneConfig(PlaneTypeEnum.横状位, 5.0, 5.0, ["骨窗"]),
            PlaneConfig(PlaneTypeEnum.横状位, 3.0, 3.0, ["骨窗"]),
            PlaneConfig(PlaneTypeEnum.横状位, 2.0, 2.0, ["骨窗"]),
        ]

        PlanningStage._assign_plane_config_ids(plane_configs)

        self.assertEqual(
            [config.output_plane for config in plane_configs],
            ["横状位", "横状位1", "横状位2"],
        )

    def test_lumbar_interspace_plans_share_one_body_centered_sagittal_localizer(self):
        """五个椎间计划可有不同切面，但必须共用一张含全部椎体的底图。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=2.0,
                slice_spacing=2.0,
                window_presets=["骨窗"],
            )],
        )
        localization_result = self._lumbar_localization_result()
        localization_result["image"]["size_xyz"] = [120, 220, 40]
        self._add_lumbar_localizer_body_geometry(localization_result)

        plans = LumbarInterspacePlanner().build_plans(
            request,
            localization_result,
        ).plans

        expected_reference = {"x": 61.0, "y": 52.5, "z": 5.0}
        self.assertTrue(all(
            plan.localizer_plane == "矢状位"
            for plan in plans
        ))
        self.assertTrue(all(
            plan.localizer_reference.to_dict() == expected_reference
            for plan in plans
        ))
        self.assertGreater(
            len({round(plan.slices[0].center.z, 6) for plan in plans}),
            1,
        )

    def test_spine_localizer_reference_falls_back_to_maximum_body_coverage(self):
        """侧弯导致主体无共同层时，应选覆盖数量最多且靠近中位中心的层。"""

        from src.planning.spine_common import resolve_spine_localizer_reference

        localization_result = self._lumbar_localization_result()
        for label_name in ("vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            localization_result["coords"][label_name]["found"] = False
        annotations = {}
        for label_name, x_bounds in (
            ("vertebrae_L1", (0.0, 2.0)),
            ("vertebrae_L2", (10.0, 12.0)),
            ("vertebrae_L3", (11.0, 13.0)),
        ):
            coord = localization_result["coords"][label_name]
            annotations[label_name] = {
                "found": True,
                "body_bbox": {
                    **coord["bbox"],
                    "x_min": x_bounds[0],
                    "x_max": x_bounds[1],
                },
                "centroid_voxel": {
                    **coord["centroid"],
                    "x": sum(x_bounds) / 2.0,
                },
                "prism_found": False,
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        reference = resolve_spine_localizer_reference(
            localization_result,
            [
                "vertebrae_L1",
                "vertebrae_L2",
                "vertebrae_L3",
                "vertebrae_L4",
                "vertebrae_L5",
                "vertebrae_S1",
            ],
        )

        self.assertEqual(reference.to_dict(), {
            "x": 11.0,
            "y": 50.0,
            "z": 8.0,
        })

    def test_lumbar_interspace_resolves_five_valid_prism_frames_and_bbox_unions(self):
        """L1–S1 五个四棱柱间隙均应生成有限斜法向并覆盖相邻全量 bbox。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments
        from src.planning.spine_common import union_bboxes

        localization_result = self._lumbar_localization_result()
        annotations = {}
        labels = ("L1", "L2", "L3", "L4", "L5", "S1")
        for index, label in enumerate(labels):
            center_y = 40.0 + 3.0 * index
            center_z = 10.0 - 2.0 * index
            annotations[f"vertebrae_{label}"] = {
                "found": True,
                "prism_found": True,
                "centroid_physical": {
                    "x": 60.0,
                    "y": center_y,
                    "z": center_z,
                },
                "sagittal_body_quadrilateral": {
                    "vertices": [
                        {"x": 60.0, "y": y, "z": z}
                        for y, z in (
                            (center_y - 0.5, center_z - 0.5),
                            (center_y + 0.5, center_z - 0.5),
                            (center_y + 0.5, center_z + 0.5),
                            (center_y - 0.5, center_z + 0.5),
                        )
                    ],
                    "left_right_extent_mm": {"min": -5.0, "max": 5.0},
                },
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual(
            [segment.group_name for segment in segments],
            ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"],
        )
        self.assertEqual(warnings, [])
        for index, segment in enumerate(segments):
            with self.subTest(group_name=segment.group_name):
                self.assertIsNotNone(segment.frame)
                self.assertFalse(segment.recenter_target_in_plane)
                self.assertTrue(all(
                    math.isfinite(value)
                    for direction in (
                        segment.frame.row_dir,
                        segment.frame.col_dir,
                        segment.frame.normal,
                    )
                    for value in (direction.x, direction.y, direction.z)
                ))
                self.assertGreater(abs(segment.frame.normal.y), 0.1)
                self.assertNotAlmostEqual(abs(segment.frame.normal.z), 1.0)
                upper_bbox = localization_result["coords"][
                    f"vertebrae_{labels[index]}"
                ]["bbox"]
                lower_bbox = localization_result["coords"][
                    f"vertebrae_{labels[index + 1]}"
                ]["bbox"]
                self.assertEqual(
                    segment.target_region_bbox,
                    union_bboxes(upper_bbox, lower_bbox),
                )

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
            ["L1_L2", "L2_L3"],
        )
        self.assertEqual(segments[0].target_start.z, 8.5)
        self.assertEqual(segments[0].target_end.z, 7.5)
        self.assertEqual(
            segments[0].source_labels,
            ("vertebrae_L1", "vertebrae_L2"),
        )
        self.assertEqual(segments[0].target_region_bbox["z_min"], 5)
        self.assertEqual(segments[0].target_region_bbox["z_max"], 11)
        self.assertTrue(segments[0].recenter_target_in_plane)
        self.assertTrue(any("L3_L4" in warning for warning in warnings))

    def test_lumbar_interspace_uses_sagittal_faces_when_full_bboxes_overlap(self):
        """全量 bbox 重叠时，仍应使用四棱柱相向端面生成斜横切区间。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        coords = localization_result["coords"]
        coords["vertebrae_L1"]["bbox"].update({"z_min": 8, "z_max": 14})
        coords["vertebrae_L2"]["bbox"].update({"z_min": 6, "z_max": 12})
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            coords[label_name]["found"] = False

        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {
                "vertebrae_L1": {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": 12.0},
                    "sagittal_body_faces": [
                        {
                            "axis_index": 0,
                            "side": "min",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 11.0},
                            "normal": {"x": 0.0, "y": -0.5, "z": -0.8660254},
                        },
                        {
                            "axis_index": 0,
                            "side": "max",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 13.0},
                            "normal": {"x": 0.0, "y": 0.5, "z": 0.8660254},
                        },
                    ],
                },
                "vertebrae_L2": {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": 8.0},
                    "sagittal_body_faces": [
                        {
                            "axis_index": 0,
                            "side": "min",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 7.0},
                            "normal": {"x": 0.0, "y": -0.5, "z": -0.8660254},
                        },
                        {
                            "axis_index": 0,
                            "side": "max",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 9.0},
                            "normal": {"x": 0.0, "y": 0.5, "z": 0.8660254},
                        },
                    ],
                },
            },
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertFalse(any("L1_L2" in warning for warning in warnings))
        segment = segments[0]
        self.assertAlmostEqual(segment.target_start.z, 11.0)
        self.assertAlmostEqual(segment.target_end.z, 9.0)
        self.assertIsNotNone(segment.frame)
        self.assertTrue(segment.recenter_target_in_plane)
        self.assertLess(segment.frame.normal.z, -0.8)
        self.assertLess(segment.frame.normal.y, -0.2)
        self.assertNotEqual(segment.frame.normal.to_dict(), {
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
        })

    def test_lumbar_interspace_planner_consumes_segment_face_frame(self):
        """腰椎 planner 保留端面切角，并把画面上下朝向对齐原图。"""

        localization_result = self._lumbar_localization_result()
        self._add_lumbar_face_geometry(localization_result)
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 8,
            "z_max": 14,
        })
        localization_result["coords"]["vertebrae_L2"]["bbox"].update({
            "z_min": 6,
            "z_max": 12,
        })
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            localization_result["coords"][label_name]["found"] = False
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=1.0,
                slice_spacing=1.0,
                window_presets=["骨窗"],
                fill_percent=100,
            )],
        )

        result = LumbarInterspacePlanner().build_plans(
            request,
            localization_result,
        )

        self.assertEqual(len(result.plans), 1)
        plan = result.plans[0]
        self.assertEqual(plan.group_name, "L1_L2")
        self.assertGreater(plan.slices[0].normal.y, 0.2)
        self.assertGreater(plan.slices[0].normal.z, 0.8)
        self.assertGreater(plan.slices[0].col_dir.y, 0.8)
        self.assertLess(plan.slices[0].col_dir.z, -0.2)
        self.assertAlmostEqual(plan.extent_mm.width, 10.0)
        self.assertAlmostEqual(plan.extent_mm.height, 12.660254, places=5)

    def test_lumbar_interspace_warns_and_falls_back_when_faces_are_invalid(self):
        """端面字段非法时应回退 bbox，并只对当前区间写 warning。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        coords = localization_result["coords"]
        coords["vertebrae_L1"]["bbox"].update({"z_min": 9, "z_max": 11})
        coords["vertebrae_L2"]["bbox"].update({"z_min": 5, "z_max": 7})
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            coords[label_name]["found"] = False
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {
                "vertebrae_L1": {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": 10.0},
                    "sagittal_body_faces": [],
                },
                "vertebrae_L2": {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": 6.0},
                    "sagittal_body_faces": [],
                },
            },
        }

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertIsNone(segments[0].frame)
        self.assertTrue(any("L1_L2" in warning and "端面" in warning for warning in warnings))

    def test_lumbar_interspace_falls_back_when_face_interval_is_degenerate(self):
        """端面中心退化时不能丢弃有效 bbox 区间。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        self._add_lumbar_face_geometry(localization_result)
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 9,
            "z_max": 11,
        })
        localization_result["coords"]["vertebrae_L2"]["bbox"].update({
            "z_min": 5,
            "z_max": 7,
        })
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            localization_result["coords"][label_name]["found"] = False
        for annotation in localization_result["spine_geometry"][
            "filtered_annotation_centroids"
        ].values():
            for face in annotation["sagittal_body_faces"]:
                face["center"]["z"] = 8.0

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertIsNone(segments[0].frame)
        self.assertTrue(any("L1_L2" in warning and "端面" in warning for warning in warnings))

    def test_lumbar_interspace_tolerates_malformed_spine_geometry_container(self):
        """损坏的 spine_geometry 容器应安全回退到 bbox。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 9,
            "z_max": 11,
        })
        localization_result["coords"]["vertebrae_L2"]["bbox"].update({
            "z_min": 5,
            "z_max": 7,
        })
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            localization_result["coords"][label_name]["found"] = False
        localization_result["spine_geometry"] = None

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertIsNone(segments[0].frame)
        self.assertFalse(any("L1_L2" in warning for warning in warnings))

    def test_lumbar_interspace_rejects_incomplete_face_pair(self):
        """只有一条头脚面时应视为不完整几何并回退 bbox。"""

        from src.planning.lumbar import resolve_lumbar_interspace_segments

        localization_result = self._lumbar_localization_result()
        self._add_lumbar_face_geometry(localization_result)
        localization_result["coords"]["vertebrae_L1"]["bbox"].update({
            "z_min": 9,
            "z_max": 11,
        })
        localization_result["coords"]["vertebrae_L2"]["bbox"].update({
            "z_min": 5,
            "z_max": 7,
        })
        for label_name in ("vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1"):
            localization_result["coords"][label_name]["found"] = False
        for annotation in localization_result["spine_geometry"][
            "filtered_annotation_centroids"
        ].values():
            annotation["sagittal_body_faces"] = annotation["sagittal_body_faces"][:1]

        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )

        self.assertEqual([segment.group_name for segment in segments], ["L1_L2"])
        self.assertIsNone(segments[0].frame)
        self.assertTrue(any("L1_L2" in warning and "端面" in warning for warning in warnings))

    def test_lumbar_interspace_target_extent_uses_current_label_pair(self):
        """区间范围切应使用当前相邻椎体 bbox 并集构造横状位计划。"""

        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=SlicePartEnum.腰椎,
            plane_configs=[PlaneConfig(
                plane=PlaneTypeEnum.横状位,
                slice_thickness=0.5,
                slice_spacing=1.0,
                window_presets=["骨窗"],
                fill_percent=100,
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
        self.assertEqual(plan.group_name, "L1_L2")
        self.assertEqual(plan.extent_mm.to_dict(), {
            "width": 20.0,
            "height": 20.0,
        })
        self.assertEqual(plan.slices[0].center.x, 25.0)
        self.assertEqual(plan.slices[0].center.y, 35.0)

    def test_planning_stage_writes_slice_plans_json(self):
        """验证 `test_planning_stage_writes_slice_plans_json` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                slice_part=SlicePartEnum.肺部,
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
            self._set_planning_inputs(
                context,
                self._lumbar_localization_result(),
            )

            PlanningStage().execute(context)

            result = context.get_result("planning")
            metadata_path = Path(result["metadata_path"])
            self.assertTrue(metadata_path.exists())
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(data["plan_count"], 1)
            self.assertEqual(data["plans"][0]["plan_id"], "肺部_MPR_横状位")
            self.assertIn("slices", data["plans"][0])
            self.assertEqual(data["diagnostics"][0]["slice_count"], 3)
            self.assertEqual(data["diagnostics"][0]["slice_thickness"], 5.0)
            self.assertEqual(data["diagnostics"][0]["slice_spacing"], 5.0)
            self.assertEqual(data["diagnostics"][0]["spacing_minus_thickness"], 0.0)
            self.assertIn("均等外扩", data["diagnostics"][0]["calculation_note"])
            self.assertEqual(data["diagnostics"][0]["actual_spacing_min_mm"], 5.0)
            self.assertEqual(data["diagnostics"][0]["actual_spacing_max_mm"], 5.0)
            self.assertFalse(data["diagnostics"][0]["has_nonuniform_spacing"])
            self.assertFalse(data["diagnostics"][0]["has_extra_endpoint_slice"])
            self.assertEqual(data["diagnostics"][0]["leading_uncovered_mm"], 0.0)
            self.assertEqual(data["diagnostics"][0]["trailing_uncovered_mm"], 0.0)
            self.assertEqual(data["diagnostics"][0]["leading_overflow_mm"], 2.0)
            self.assertEqual(data["diagnostics"][0]["trailing_overflow_mm"], 2.0)
            self.assertEqual(
                data["diagnostics"][0]["leading_center_uncovered_mm"],
                0.5,
            )
            self.assertEqual(
                data["diagnostics"][0]["trailing_center_uncovered_mm"],
                0.5,
            )
            self.assertEqual(
                data["diagnostics"][0]["leading_center_overflow_mm"],
                0.0,
            )
            self.assertEqual(
                data["diagnostics"][0]["trailing_center_overflow_mm"],
                0.0,
            )
            self.assertEqual(data["diagnostics"][0]["target_length_mm"], 11.0)
            self.assertFalse(data["diagnostics"][0]["target_shorter_than_thickness"])
            self.assertEqual(data["diagnostics"][0]["first_coverage_start"]["z"], 12.5)
            self.assertEqual(data["diagnostics"][0]["last_coverage_end"]["z"], -2.5)

    def test_planning_diagnostics_ignore_sub_tolerance_center_gaps(self):
        """中心越界量在规划容差内时，不应被诊断为首尾缺失。"""

        diagnostics = PlanningStage()._build_plan_diagnostics([{
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

    def test_planning_writes_json_without_debug_nifti(self):
        """混合切片请求只写规划 JSON，不再生成规划调试 NIfTI。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request(
                dicom_dir="/tmp/dicom",
                output_dir=temp_dir,
                slice_part=SlicePartEnum.腰椎,
                plane_configs=[
                    PlaneConfig(
                        PlaneTypeEnum.横状位, 3.0, 5.0, ["骨窗"],
                        slice_type=SliceType.椎间隙,
                    ),
                    PlaneConfig(
                        PlaneTypeEnum.矢状位, 3.0, 5.0, ["骨窗"],
                    ),
                    PlaneConfig(
                        PlaneTypeEnum.冠状位, 3.0, 5.0, ["骨窗"],
                        slice_type=SliceType.CPR,
                    ),
                ],
            )
            context = ProcessingContext(request)
            self._set_planning_inputs(
                context,
                self._lumbar_localization_result(),
            )

            PlanningStage().execute(context)

            result = context.get_result("planning")
            metadata_path = Path(result["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            obsolete_keys = {
                "debug_slice_plan_overlay_path",
                "debug_center_cross_overlay_path",
                "debug_slice_plan_overview_overlay_path",
                "debug_slice_plan_native_overview_overlay_path",
                "debug_slice_plan_overview_paths",
                "debug_slice_plan_overview_with_image_paths",
                "debug_slice_plan_native_overview_paths",
            }

            self.assertEqual(result["plan_count"], 7)
            self.assertEqual(
                {plan["slice_type"] for plan in result["plans"]},
                {"MPR", "CPR", "椎间隙"},
            )
            self.assertFalse(obsolete_keys & result.keys())
            self.assertFalse(obsolete_keys & metadata.keys())
            self.assertEqual(
                list((Path(temp_dir) / "metadata").glob("*.nii.gz")),
                [],
            )

    @staticmethod
    def _spine_localization_result(labels, slice_part):
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
            **Stage6SlicePlanningTest._configured_localization_contract(
                slice_part
            ),
            "slice_part": slice_part,
            "localization_profile": "whole_range",
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
    def _add_lumbar_face_geometry(localization_result):
        """为测试注入一对相邻椎体的有效矢状端面。"""

        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": {
                "vertebrae_L1": {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": 12.0},
                    "sagittal_body_faces": [
                        {
                            "axis_index": 0,
                            "side": "min",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 11.0},
                            "normal": {"x": 0.0, "y": -0.5, "z": -0.8660254},
                        },
                        {
                            "axis_index": 0,
                            "side": "max",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 13.0},
                            "normal": {"x": 0.0, "y": 0.5, "z": 0.8660254},
                        },
                    ],
                },
                "vertebrae_L2": {
                    "found": True,
                    "centroid_physical": {"x": 60.0, "y": 50.0, "z": 8.0},
                    "sagittal_body_faces": [
                        {
                            "axis_index": 0,
                            "side": "min",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 7.0},
                            "normal": {"x": 0.0, "y": -0.5, "z": -0.8660254},
                        },
                        {
                            "axis_index": 0,
                            "side": "max",
                            "point_count": 100,
                            "center": {"x": 60.0, "y": 50.0, "z": 9.0},
                            "normal": {"x": 0.0, "y": 0.5, "z": 0.8660254},
                        },
                    ],
                },
            },
        }

    @staticmethod
    def _add_lumbar_localizer_body_geometry(localization_result):
        """注入位置不同但存在共同中心层的 L1-S1 椎体主体几何。"""

        bounds = {
            "vertebrae_L1": ((50, 68), (40, 60)),
            "vertebrae_L2": ((52, 70), (42, 62)),
            "vertebrae_L3": ((54, 72), (44, 64)),
            "vertebrae_L4": ((53, 69), (43, 61)),
            "vertebrae_L5": ((55, 71), (45, 63)),
            "vertebrae_S1": ((51, 67), (41, 60)),
        }
        annotations = {}
        for label_name, (x_bounds, y_bounds) in bounds.items():
            coord = localization_result["coords"][label_name]
            body_bbox = {
                **coord["bbox"],
                "x_min": x_bounds[0],
                "x_max": x_bounds[1],
                "y_min": y_bounds[0],
                "y_max": y_bounds[1],
            }
            body_centroid = {
                **coord["centroid"],
                "x": sum(x_bounds) / 2.0,
                "y": sum(y_bounds) / 2.0,
            }
            # 全量标签还包含椎弓等后方结构，故意把它扩到主体之外，验证选层
            # 始终优先使用第五阶段过滤出的 body_bbox。
            coord["bbox"].update({
                "x_min": x_bounds[0] - 4,
                "x_max": x_bounds[1] + 8,
                "y_min": y_bounds[0] - 3,
                "y_max": y_bounds[1] + 70,
            })
            coord["centroid"].update({
                "x": sum(x_bounds) / 2.0 + 2.0,
                "y": sum(y_bounds) / 2.0 + 20.0,
            })
            annotations[label_name] = {
                "found": True,
                "body_bbox": body_bbox,
                "centroid_voxel": body_centroid,
                "prism_found": False,
            }
        localization_result["spine_geometry"] = {
            "filtered_annotation_centroids": annotations,
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
            **Stage6SlicePlanningTest._configured_localization_contract("腰椎"),
            "slice_part": "腰椎",
            "localization_profile": "whole_range",
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
