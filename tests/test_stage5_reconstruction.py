import copy
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import pydicom
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import (
    BodyPartEnum,
    CutModeEnum,
    PlaneConfig,
    PlaneTypeEnum,
    Request,
    WindowPreset,
)
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.reconstruction import dicom as dicom_writer
from src.reconstruction.curved_stack import CurvedStackResampler
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image
from src.stages.stage_reconstruction import ReconstructionStage
from src.utils.Logger_utils import _logger_manager


class Stage5ReconstructionTest(unittest.TestCase):
    """汇总 `Stage5ReconstructionTest` 相关测试场景。"""

    @staticmethod
    def _max_line_fit_residual(points_zyx: np.ndarray) -> float:
        """返回二维高亮点到最佳直线的最大像素残差。"""

        rows = points_zyx[:, 0].astype(float)
        columns = points_zyx[:, 1].astype(float)
        matrix = np.column_stack([rows, np.ones_like(rows)])
        slope, intercept = np.linalg.lstsq(matrix, columns, rcond=None)[0]
        fitted = slope * rows + intercept
        return float(np.max(np.abs(columns - fitted)))

    def test_plans_are_grouped_by_plane_without_changing_first_seen_order(self):
        """验证 `test_plans_are_grouped_by_plane_without_changing_first_seen_order` 场景符合预期。"""

        plans = [
            {"plan_id": "横状位_1", "plane": "横状位"},
            {"plan_id": "矢状位_1", "plane": "矢状位"},
            {"plan_id": "横状位_2", "plane": "横状位"},
            {"plan_id": "冠状位_1", "plane": "冠状位"},
        ]

        grouped = ReconstructionStage._group_plans_by_plane(plans)

        self.assertEqual(list(grouped), ["横状位", "矢状位", "冠状位"])
        self.assertEqual(
            [plan["plan_id"] for plan in grouped["横状位"]],
            ["横状位_1", "横状位_2"],
        )

    def test_parallel_windows_start_only_after_all_mpr_slices_are_sampled(self):
        """验证 `test_parallel_windows_start_only_after_all_mpr_slices_are_sampled` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                window_presets=["骨窗", "脑窗"],
            )
            plan = context.get_result("mapping")["plans"][0]
            sampled_indices = []
            first_window_sample_count = []
            active_workers = 0
            max_active_workers = 0
            worker_names = set()
            lock = threading.Lock()

            class RecordingResampler:
                """汇总 `RecordingResampler` 相关测试场景。"""

                def sample_slice(self, image, current_plan, planned_slice):
                    """为测试准备 `sample_slice` 所需的数据或记录行为。"""

                    sampled_indices.append(planned_slice["index"])
                    return planned_slice["index"]

            def recording_window(sampled_image, window):
                """为测试准备 `recording_window` 所需的数据或记录行为。"""

                nonlocal active_workers, max_active_workers
                with lock:
                    if not first_window_sample_count:
                        first_window_sample_count.append(len(sampled_indices))
                    active_workers += 1
                    max_active_workers = max(max_active_workers, active_workers)
                    worker_names.add(threading.current_thread().name)
                time.sleep(0.03)
                with lock:
                    active_workers -= 1
                return sampled_image

            stage = ReconstructionStage({
                "parallel": True,
                "max_window_workers": 2,
            })
            with mock.patch(
                "src.stages.stage_reconstruction.PlaneStackResampler",
                return_value=RecordingResampler(),
            ), mock.patch(
                "src.stages.stage_reconstruction.apply_window_to_image",
                side_effect=recording_window,
            ), mock.patch(
                "src.stages.stage_reconstruction.write_ct_dicom",
            ):
                stage._execute_plan(
                    image=object(),
                    plan=plan,
                    output_root=Path(context.request.output_dir),
                    study_uid="1.2.3",
                    source_metadata={},
                )

            self.assertEqual(sampled_indices, [1, 2])
            self.assertEqual(first_window_sample_count, [2])
            self.assertGreaterEqual(max_active_workers, 2)
            self.assertGreaterEqual(len(worker_names), 2)

    def test_gdcm_writer_lifecycle_is_serialized_across_window_threads(self):
        """四个窗位可并发准备图像，但 GDCM 写入器生命周期不能重叠。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(temp_dir, slice_count=1)
            plan = context.get_result("mapping")["plans"][0]
            planned_slice = plan["slices"][0]
            output_root = Path(temp_dir) / "gdcm-writers"
            start_barrier = threading.Barrier(4)
            state_lock = threading.Lock()
            active_writers = 0
            max_active_writers = 0

            class RecordingImageFileWriter:
                """记录测试期间同时进入 Execute 的原生写入器数量。"""

                def __init__(self):
                    self.output_path: Path | None = None

                def SetFileName(self, value):
                    self.output_path = Path(value)

                def KeepOriginalImageUIDOn(self):
                    return None

                def Execute(self, image):
                    nonlocal active_writers, max_active_writers
                    self.assert_ready()
                    with state_lock:
                        active_writers += 1
                        max_active_writers = max(
                            max_active_writers,
                            active_writers,
                        )
                    try:
                        self.output_path.touch()
                        # 释放 GIL 并扩大重叠窗口，使没有串行保护的实现稳定失败。
                        time.sleep(0.05)
                    finally:
                        with state_lock:
                            active_writers -= 1

                def assert_ready(self):
                    if self.output_path is None:
                        raise AssertionError("测试写入器尚未设置输出路径")

            def write_window(window_index: int) -> Path:
                """让四个窗位同时进入真实 write_ct_dicom 调用链。"""

                start_barrier.wait(timeout=2.0)
                output_path = output_root / str(window_index) / "1.dcm"
                dicom_writer.write_ct_dicom(
                    image=sitk.GetImageFromArray(
                        np.zeros((1, 4, 4), dtype=np.uint16),
                    ),
                    output_path=output_path,
                    plan=plan,
                    planned_slice=planned_slice,
                    window=WindowPreset(
                        name=f"测试窗{window_index}",
                        window_width=100.0,
                        window_center=50.0,
                    ),
                    series_uid=f"1.2.826.0.1.3680043.2.1125.{window_index}",
                    study_uid="1.2.826.0.1.3680043.2.1125.100",
                    total_slices=1,
                    source_metadata={},
                )
                return output_path

            with mock.patch.object(
                dicom_writer.sitk,
                "ImageFileWriter",
                RecordingImageFileWriter,
            ), mock.patch.object(
                dicom_writer,
                "_restore_planned_slice_geometry_tags",
            ):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    output_paths = list(executor.map(write_window, range(1, 5)))

            self.assertTrue(all(path.exists() for path in output_paths))
            self.assertEqual(max_active_writers, 1)

    def test_windowing_bakes_hu_into_fixed_uint16_grayscale(self):
        """验证 `test_windowing_bakes_hu_into_fixed_uint16_grayscale` 场景符合预期。"""

        image = sitk.GetImageFromArray(
            np.asarray([[[-850, 400, 1650]]], dtype=np.int16),
        )

        output = apply_window_to_image(image, "骨窗")

        self.assertEqual(output.GetPixelID(), sitk.sitkUInt16)
        self.assertEqual(
            sitk.GetArrayFromImage(output).tolist(),
            [[[0, 32768, 65535]]],
        )

    def test_interval_plan_keeps_group_directory(self):
        """验证 `test_interval_plan_keeps_group_directory` 场景符合预期。"""

        plan = {"cut_mode": "interspace", "group_name": "L1-L2"}

        self.assertEqual(ReconstructionStage._plan_group_name(plan), "L1-L2")

    def test_cpr_plan_writes_directly_under_window_directory(self):
        """CPR 输出不应在窗位目录下额外创建 CPR 子目录。"""

        plan = {
            "geometry_type": "curved_stack",
            "cut_mode": "cpr",
            "group_name": "CPR",
        }

        self.assertEqual(ReconstructionStage._plan_group_name(plan), "")

    def test_localizer_uses_representative_original_slice_instead_of_projection(self):
        """验证 `test_localizer_uses_representative_original_slice_instead_of_projection` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            image_array = np.zeros((8, 8, 8), dtype=np.int16)
            image_array[:, :, 2] = 100
            image_array[:, :, 7] = 1000
            image = sitk.GetImageFromArray(image_array)
            plan = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
            ).get_result("mapping")["plans"][0]
            plan["slice_thickness"] = 5.0

            localizer = ReconstructionStage._build_plan_localizer(image, plan)
            localizer_array = sitk.GetArrayFromImage(localizer)

            self.assertEqual(localizer.GetSize(), (8, 8))
            self.assertEqual(float(localizer_array[0, 0]), 100.0)
            self.assertGreater(float(np.max(localizer_array)), 1000.0)
            highlighted_rows = np.unique(
                np.argwhere(localizer_array > 1000.0)[:, 0]
            )
            self.assertEqual(highlighted_rows.tolist(), [3, 5])

    def test_localizer_uses_target_reference_instead_of_recentered_mpr_center(self):
        """矢状位定位图应经过目标椎体中心，而不是整幅图像显示中心。"""

        image_array = np.zeros((8, 8, 8), dtype=np.int16)
        for y_index in range(8):
            image_array[:, y_index, :] = y_index * 100
        image = sitk.GetImageFromArray(image_array)
        plan = {
            "plane": "矢状位",
            "extent_mm": {"width": 6.0, "height": 6.0},
            "localizer_reference": {"x": 3.0, "y": 2.0, "z": 3.0},
            "slices": [{
                "index": 1,
                # 成片为了整图覆盖居中在 y=5，但目标椎体实际位于 y=2。
                "center": {"x": 3.0, "y": 5.0, "z": 3.0},
                "normal": {"x": -1.0, "y": 0.0, "z": 0.0},
                "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 0.0, "z": -1.0},
            }],
        }

        localizer = ReconstructionStage._build_plan_localizer(image, plan)
        localizer_array = sitk.GetArrayFromImage(localizer)

        # 交线只改少量像素，背景中位数能稳定表示所选原始冠状层。
        self.assertEqual(float(np.median(localizer_array)), 200.0)

    def test_localizer_plane_explicitly_selects_coronal_background(self):
        """计划指定冠状定位图时，应固定原图 y 层且不再由斜率自动改面。"""

        image_array = np.zeros((8, 8, 8), dtype=np.int16)
        for y_index in range(8):
            image_array[:, y_index, :] = y_index * 100
        image = sitk.GetImageFromArray(image_array)
        plan = {
            "plane": "横状位",
            "localizer_plane": "冠状位",
            "localizer_reference": {"x": 3.0, "y": 2.0, "z": 3.0},
            "extent_mm": {"width": 6.0, "height": 6.0},
            "slices": [{
                "index": 1,
                "center": {"x": 3.0, "y": 3.0, "z": 3.0},
                # 该角度的旧自动逻辑会选择矢状定位图；显式字段必须覆盖它。
                "normal": {"x": 0.0, "y": 0.2, "z": 0.9797958971},
                "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 0.9797958971, "z": -0.2},
            }],
        }

        localizer = ReconstructionStage._build_plan_localizer(image, plan)

        self.assertEqual(localizer.GetSize(), (8, 8))
        self.assertEqual(float(np.median(sitk.GetArrayFromImage(localizer))), 200.0)

    def test_oblique_axial_plan_draws_sloped_line_on_localizer(self):
        """斜横状位计划应在定位图上显示与真实切片平面一致的斜线。"""

        image = sitk.GetImageFromArray(np.zeros((32, 32, 32), dtype=np.int16))
        plan = {
            "plane": "横状位",
            "extent_mm": {"width": 20.0, "height": 20.0},
            "slices": [{
                "index": 1,
                "center": {"x": 15.0, "y": 15.0, "z": 15.0},
                "normal": {"x": 0.4472136, "y": 0.0, "z": 0.8944272},
                "row_dir": {"x": 0.8944272, "y": 0.0, "z": -0.4472136},
                "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
            }],
        }

        localizer = ReconstructionStage._build_plan_localizer(image, plan)
        highlighted = np.argwhere(sitk.GetArrayFromImage(localizer) > 0.0)

        self.assertGreater(len(highlighted), 5)
        self.assertGreater(int(np.ptp(highlighted[:, 0])), 2)
        self.assertGreater(int(np.ptp(highlighted[:, 1])), 2)

    def test_curved_stack_localizer_draws_curve_on_sagittal_background(self):
        """CPR 定位图应按曲线点绘制，不应退化成首尾斜直线。"""

        image = sitk.GetImageFromArray(np.zeros((36, 36, 36), dtype=np.int16))
        plan = {
            "plane": "矢状位",
            "geometry_type": "curved_stack",
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

        localizer = ReconstructionStage._build_plan_localizer(image, plan)
        highlighted = np.argwhere(sitk.GetArrayFromImage(localizer) > 0.0)

        self.assertGreater(len(highlighted), 20)
        self.assertGreater(self._max_line_fit_residual(highlighted), 2.0)

    def test_curved_stack_localizer_draws_equally_spaced_offsets(self):
        """CPR 定位图应只画等间距 offset 中心线。"""

        image = sitk.GetImageFromArray(np.zeros((36, 36, 36), dtype=np.int16))
        plan = {
            "plane": "矢状位",
            "geometry_type": "curved_stack",
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
            "slice_thickness": 4.0,
            "slices": [{
                "index": 1,
                "center": {"x": 18.0, "y": 18.0, "z": 18.0},
                "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
                "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 0.0, "z": 1.0},
                "offset_mm": 0.0,
            }],
        }

        localizer = ReconstructionStage._build_plan_localizer(image, plan)
        highlighted = np.argwhere(sitk.GetArrayFromImage(localizer) > 0.0)

        self.assertEqual(
            np.unique(highlighted[:, 1]).tolist(),
            [10, 18, 26],
        )

    def test_curved_stack_localizer_offsets_follow_bend_normals(self):
        """CPR 弯曲段定位线应随局部法向偏移，而不是固定方向平移。"""

        image = sitk.GetImageFromArray(np.zeros((64, 64, 64), dtype=np.int16))
        plan = {
            "plane": "冠状位",
            "geometry_type": "curved_stack",
            "localizer_plane": "矢状位",
            "localizer_reference": {"x": 32.0, "y": 32.0, "z": 32.0},
            "extent_mm": {"width": 20.0, "height": 40.0},
            "curve_points": [
                {"x": 32.0, "y": 20.0, "z": 0.0},
                {"x": 32.0, "y": 20.0, "z": 10.0},
                {"x": 32.0, "y": 21.0, "z": 20.0},
                {"x": 32.0, "y": 25.0, "z": 29.0},
                {"x": 32.0, "y": 35.0, "z": 35.0},
                {"x": 32.0, "y": 45.0, "z": 38.0},
                {"x": 32.0, "y": 55.0, "z": 40.0},
            ],
            "offset_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
            "offsets_mm": [-10.0, 10.0],
            "slice_thickness": 5.0,
            "slices": [{
                "index": 1,
                "center": {"x": 32.0, "y": 32.0, "z": 32.0},
                "normal": {"x": 0.0, "y": 1.0, "z": 0.0},
                "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 0.0, "z": 1.0},
                "offset_mm": 0.0,
            }],
        }

        localizer = ReconstructionStage._build_plan_localizer(image, plan)
        highlighted = np.argwhere(sitk.GetArrayFromImage(localizer) > 0.0)

        self.assertGreater(int(np.ptp(highlighted[:, 0])), 30)
        self.assertGreaterEqual(len(np.unique(highlighted[:, 1])), 2)

    def test_curved_stack_localizer_uses_actual_offset_axis_on_bends(self):
        """CPR 定位线应按实际采样 offset_axis 平移。"""

        curve_points = [
            {"x": 32.0, "y": 24.0, "z": 12.0},
            {"x": 32.0, "y": 24.0, "z": 32.0},
            {"x": 32.0, "y": 44.0, "z": 42.0},
        ]
        plan = {
            "curve_points": curve_points,
            "offset_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
            "offsets_mm": [-10.0, 0.0, 10.0],
            "localizer_plane": "矢状位",
        }

        projected = ReconstructionStage._curved_localizer_projected_curves(
            plan=plan,
            fixed_axis=0,
        )

        lower = np.asarray(projected[-10.0][-1], dtype=float)
        center_curve = np.asarray(projected[0.0], dtype=float)
        center = center_curve[-1]
        upper = np.asarray(projected[10.0][-1], dtype=float)
        self.assertAlmostEqual(float(np.linalg.norm(center - lower)), 10.0)
        self.assertAlmostEqual(float(np.linalg.norm(upper - center)), 10.0)
        self.assertAlmostEqual(lower[2], center[2])
        self.assertAlmostEqual(upper[2], center[2])

    def test_curved_stack_localizer_offsets_match_actual_sampling_geometry(self):
        """每条定位线必须是实际采样曲线的刚体平移。"""

        plan = {
            "curve_points": [
                {"x": 32.0, "y": 24.0, "z": 12.0},
                {"x": 32.0, "y": 24.0, "z": 32.0},
                {"x": 32.0, "y": 44.0, "z": 42.0},
            ],
            "offset_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
            "offsets_mm": [-10.0, 0.0, 10.0, 20.0],
        }

        projected = ReconstructionStage._curved_localizer_projected_curves(
            plan=plan,
            fixed_axis=0,
        )
        center = np.asarray(projected[0.0], dtype=float)
        offset = np.asarray(projected[10.0], dtype=float)
        deltas = offset - center
        expected_delta = np.asarray([0.0, 10.0, 0.0])
        expected_deltas = np.tile(expected_delta, (len(center), 1))
        np.testing.assert_allclose(deltas, expected_deltas, atol=1e-6)

        for lower, upper in [(-10.0, 0.0), (0.0, 10.0), (10.0, 20.0)]:
            gap = np.asarray(projected[upper], dtype=float) - np.asarray(
                projected[lower], dtype=float
            )
            np.testing.assert_allclose(gap, expected_deltas, atol=1e-6)

    def test_curved_stack_localizer_limits_normal_turn_for_extended_offsets(self):
        """slice_range_extension 较大时应限制法向旋转，避免内弯相交。"""

        plan = {
            "curve_points": [
                {"x": 32.0, "y": 24.0, "z": 12.0},
                {"x": 32.0, "y": 24.0, "z": 32.0},
                {"x": 32.0, "y": 44.0, "z": 42.0},
            ],
            "offset_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
            "offsets_mm": [-70.0, 0.0, 70.0],
        }

        projected = ReconstructionStage._curved_localizer_projected_curves(
            plan=plan,
            fixed_axis=0,
        )
        curve = np.asarray(projected[0.0], dtype=float)
        extended = np.asarray(projected[70.0], dtype=float)
        center_steps = np.diff(curve[:, [1, 2]], axis=0)
        extended_steps = np.diff(extended[:, [1, 2]], axis=0)

        self.assertTrue(np.all(
            np.sum(center_steps * extended_steps, axis=1) > 0.0
        ))

    def test_curved_stack_localizer_removes_lateral_backtracking(self):
        """弯曲末端不能沿显示横向先回折再折回。"""

        curve = np.asarray([
            [32.0, 20.0, 0.0],
            [32.0, 20.0, 10.0],
            [32.0, 22.0, 20.0],
            [32.0, 28.0, 29.0],
            [32.0, 38.0, 35.0],
            [32.0, 45.0, 38.0],
            [32.0, 43.0, 40.0],
        ])

        projected = ReconstructionStage._curved_localizer_projected_curves(
            plan={
                "curve_points": [
                    {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
                    for point in curve
                ],
                "offset_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "offsets_mm": [70.0],
            },
            fixed_axis=0,
        )
        adjusted = np.asarray(projected[70.0])
        center = np.asarray(projected[0.0])

        np.testing.assert_allclose(adjusted[:, 2], center[:, 2])
        np.testing.assert_allclose(
            np.diff(adjusted[:, 2]),
            np.diff(center[:, 2]),
        )

    def test_curved_stack_localizer_shares_longitudinal_samples(self):
        """所有 offset 线应共用纵向采样，避免弯角处回环。"""

        plan = {
            "curve_points": [
                {"x": 32.0, "y": 20.0, "z": 0.0},
                {"x": 32.0, "y": 20.0, "z": 10.0},
                {"x": 32.0, "y": 24.0, "z": 20.0},
                {"x": 32.0, "y": 38.0, "z": 29.0},
                {"x": 32.0, "y": 50.0, "z": 35.0},
            ],
            "offset_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
            "offsets_mm": [-20.0, 0.0, 20.0],
        }

        projected = ReconstructionStage._curved_localizer_projected_curves(
            plan=plan,
            fixed_axis=0,
        )
        center = np.asarray(projected[0.0])

        np.testing.assert_allclose(np.asarray(projected[-20.0])[:, 2], center[:, 2])
        np.testing.assert_allclose(np.asarray(projected[20.0])[:, 2], center[:, 2])

    def test_curved_stack_localizer_draws_every_planned_offset(self):
        """CPR 定位图必须按实际切片 offset 一张对应一条线。"""

        plan = {
            "offsets_mm": [float(value) for value in range(-35, 70, 5)],
        }

        self.assertEqual(
            ReconstructionStage._curved_localizer_offsets(plan),
            [float(value) for value in range(-35, 70, 5)],
        )
        self.assertEqual(len(plan["offsets_mm"]), 21)

    def test_full_xyz_oblique_plan_is_resampled_with_planned_direction(self):
        """第五步应原样执行含 XYZ 三方向倾斜的通用平面计划。"""

        image = sitk.GetImageFromArray(np.ones((32, 32, 32), dtype=np.int16))
        normal = np.asarray([0.2, 0.3, 0.9327379053], dtype=float)
        normal /= np.linalg.norm(normal)
        row = np.asarray([1.0, 0.0, 0.0], dtype=float)
        row -= np.dot(row, normal) * normal
        row /= np.linalg.norm(row)
        col = np.cross(normal, row)

        def point(value):
            return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}

        planned_slice = {
            "index": 1,
            "center": {"x": 15.5, "y": 15.5, "z": 15.5},
            "normal": point(normal),
            "row_dir": point(row),
            "col_dir": point(col),
        }
        plan = {
            "plane": "横状位",
            "extent_mm": {"width": 20.0, "height": 20.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 1.0,
            "slices": [planned_slice],
        }

        output = PlaneStackResampler().sample_slice(image, plan, planned_slice)
        expected_direction = tuple(
            component
            for coordinate in range(3)
            for component in (row[coordinate], col[coordinate], normal[coordinate])
        )

        self.assertTrue(np.allclose(output.GetDirection(), expected_direction))
        self.assertTrue(np.any(sitk.GetArrayFromImage(output) > -1024.0))

    def test_localizer_puts_physical_superior_side_at_top(self):
        """验证 `test_localizer_puts_physical_superior_side_at_top` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            for z_direction in (1.0, -1.0):
                image_array = np.zeros((8, 8, 8), dtype=np.int16)
                for z_index in range(8):
                    superior_value = (
                        z_index if z_direction > 0 else 7 - z_index
                    )
                    image_array[z_index, :, :] = superior_value * 100
                image = sitk.GetImageFromArray(image_array)
                image.SetDirection((
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, z_direction,
                ))

                for plane in ("横状位", "矢状位", "冠状位"):
                    with self.subTest(z_direction=z_direction, plane=plane):
                        plan = self._make_context_with_one_plan(
                            temp_dir,
                            slice_count=1,
                        ).get_result("mapping")["plans"][0]
                        plan["plane"] = plane
                        plan["extent_mm"] = {"width": 0.0, "height": 0.0}

                        localizer = ReconstructionStage._build_plan_localizer(
                            image,
                            plan,
                        )
                        localizer_array = sitk.GetArrayFromImage(localizer)

                        self.assertGreater(
                            float(np.median(localizer_array[0])),
                            float(np.median(localizer_array[-1])),
                        )
                        orientation = tuple(float(value) for value in (
                            localizer.GetMetaData("0020|0037").split("\\")
                        ))
                        self.assertEqual(len(orientation), 6)
                        # DICOM 画面列方向必须真正沿物理头脚轴，
                        # 且从画面上向下指向足侧。
                        self.assertLess(orientation[5], 0.0)

    def test_multiple_interval_plans_write_to_independent_group_directories(self):
        """第五阶段应保留多个区间各自的名称和独立输出目录。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=1,
                window_presets=["骨窗"],
            )
            first_plan = context.get_result("mapping")["plans"][0]
            first_plan.update({
                "plan_id": "腰椎_interspace_L1-L2_横状位",
                "cut_mode": "interspace",
                "group_name": "L1-L2",
            })
            second_plan = copy.deepcopy(first_plan)
            second_plan.update({
                "plan_id": "腰椎_interspace_L2-L3_横状位",
                "group_name": "L2-L3",
            })
            context.set_result("mapping", {
                "plans": [first_plan, second_plan],
            })

            ReconstructionStage(config={"parallel": False}).execute(context)

            output_root = Path(context.request.output_dir)
            self.assertEqual(
                len(list(
                    (output_root / "横状位" / "骨窗" / "L1-L2")
                    .glob("[0-9]*.dcm")
                )),
                1,
            )
            self.assertTrue(
                (output_root / "横状位" / "骨窗" / "L1-L2" / "Localizer.dcm")
                .exists()
            )
            self.assertTrue(
                (output_root / "横状位" / "骨窗" / "L2-L3" / "Localizer.dcm")
                .exists()
            )
            self.assertEqual(
                len(list(
                    (output_root / "横状位" / "骨窗" / "L2-L3")
                    .glob("[0-9]*.dcm")
                )),
                1,
            )
            self.assertEqual(
                context.get_result("reconstruction")["plan_count"],
                2,
            )

    def test_plane_stack_resampler_uses_slice_center_and_in_plane_extent(self):
        """验证 `test_plane_stack_resampler_uses_slice_center_and_in_plane_extent` 场景符合预期。"""

        image = self._make_ramp_image(size=(8, 8, 8))
        plan = {
            "extent_mm": {"width": 4.0, "height": 4.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 1.0,
        }
        planned_slice = {
            "center": {"x": 2.0, "y": 2.0, "z": 2.0},
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
            "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
        }

        output = PlaneStackResampler().sample_slice(image, plan, planned_slice)

        self.assertEqual(output.GetSize(), (5, 5, 1))
        self.assertEqual(output.GetSpacing(), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(
            float(sitk.GetArrayFromImage(output)[0, 0, 0]),
            200.0,
        )

    def test_curved_stack_resampler_samples_straight_sagittal_curve(self):
        """直线 CPR 曲线应沿曲线弧长和前后方向连续采样。"""

        image = self._make_ramp_image(size=(8, 8, 8))
        plan = {
            "plan_id": "腰椎_CPR_矢状位",
            "geometry_type": "curved_stack",
            "curve_points": [
                {"x": 2.0, "y": 3.0, "z": 1.0},
                {"x": 2.0, "y": 3.0, "z": 5.0},
            ],
            "curve_spacing_mm": 1.0,
            "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
            "offsets_mm": [0.0],
            "extent_mm": {"width": 4.0, "height": 4.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 1.0,
            "slice_spacing": 1.0,
            "slices": [{
                "index": 1,
                "offset_mm": 0.0,
                "center": {"x": 2.0, "y": 3.0, "z": 3.0},
                "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
                "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 0.0, "z": 1.0},
            }],
        }

        output = CurvedStackResampler().sample_slice(
            image,
            plan,
            plan["slices"][0],
        )

        output_array = sitk.GetArrayFromImage(output)
        self.assertEqual(output.GetSize(), (5, 5, 1))
        self.assertAlmostEqual(float(output_array[0, 0, 0]), 112.0)
        self.assertAlmostEqual(float(output_array[0, 2, 2]), 332.0)
        self.assertAlmostEqual(float(output_array[0, 4, 4]), 552.0)

    def test_curved_stack_resampler_averages_slice_thickness_slab(self):
        """curved_stack 应按 slice_thickness 对 offset 方向做厚层平均。"""

        array = np.zeros((8, 8, 8), dtype=np.float32)
        for x_index in range(8):
            array[:, :, x_index] = float(x_index * x_index * 100)
        image = sitk.GetImageFromArray(array)
        plan = {
            "plan_id": "腰椎_CPR_冠状位",
            "geometry_type": "curved_stack",
            "curve_points": [
                {"x": 3.0, "y": 3.0, "z": 1.0},
                {"x": 3.0, "y": 3.0, "z": 5.0},
            ],
            "curve_spacing_mm": 1.0,
            "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
            "offsets_mm": [0.0],
            "extent_mm": {"width": 0.0, "height": 4.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 4.0,
            "slice_spacing": 5.0,
            "slices": [{
                "index": 1,
                "offset_mm": 0.0,
                "center": {"x": 3.0, "y": 3.0, "z": 3.0},
                "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
                "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 0.0, "z": 1.0},
            }],
        }

        output = CurvedStackResampler().sample_slice(
            image,
            plan,
            plan["slices"][0],
        )

        output_array = sitk.GetArrayFromImage(output)
        self.assertAlmostEqual(float(output_array[0, 0, 0]), 1050.0)

    def test_curved_stack_resampler_logs_sampling_perf_summary(self):
        """CPR 重采样应输出性能摘要，便于定位一轮耗时。"""

        image = self._make_ramp_image(size=(8, 8, 8))
        plan = {
            "plan_id": "腰椎_CPR_矢状位",
            "geometry_type": "curved_stack",
            "curve_points": [
                {"x": 2.0, "y": 3.0, "z": 1.0},
                {"x": 2.0, "y": 3.0, "z": 5.0},
            ],
            "curve_spacing_mm": 1.0,
            "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
            "offsets_mm": [0.0],
            "extent_mm": {"width": 4.0, "height": 4.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 2.0,
            "slice_spacing": 1.0,
            "slices": [{
                "index": 7,
                "offset_mm": 0.0,
                "center": {"x": 2.0, "y": 3.0, "z": 3.0},
                "normal": {"x": 1.0, "y": 0.0, "z": 0.0},
                "row_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 0.0, "z": 1.0},
            }],
        }

        with mock.patch("src.reconstruction.curved_stack.DebugLog") as debug_log:
            CurvedStackResampler().sample_slice(image, plan, plan["slices"][0])

        messages = [call.args[0] for call in debug_log.call_args_list]
        self.assertTrue(any("[CPR采样性能]" in message for message in messages))
        self.assertTrue(any("slab_count=" in message for message in messages))

    def test_curved_stack_resampler_reuses_nearest_valid_in_plane_direction(self):
        """局部切线平行堆叠轴时，不应因单行方向退化而中断 CPR。"""

        image = self._make_ramp_image(size=(8, 8, 8))
        plan = {
            "curve_points": [
                {"x": 1.0, "y": 3.0, "z": 1.0},
                {"x": 3.0, "y": 3.0, "z": 1.0},
                {"x": 3.0, "y": 3.0, "z": 5.0},
            ],
            "curve_spacing_mm": 1.0,
            "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
            "extent_mm": {"width": 2.0, "height": 6.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
        }
        planned_slice = {"offset_mm": 0.0}

        output = CurvedStackResampler().sample_slice(
            image,
            plan,
            planned_slice,
        )

        self.assertEqual(output.GetSize(), (3, 7, 1))
        self.assertTrue(np.isfinite(sitk.GetArrayFromImage(output)).all())

    def test_plane_stack_resampler_consumes_three_straight_whole_range_plans(self):
        """第五阶段应直接执行通用构造器生成的横、矢、冠直线计划。"""

        image = self._make_ramp_image(size=(12, 12, 12))
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            body_part=BodyPartEnum.肺部,
            plane_configs=[
                PlaneConfig(
                    plane=plane,
                    slice_thickness=1.0,
                    slice_spacing=2.0,
                    window_presets=["肺窗"],
                )
                for plane in (
                    PlaneTypeEnum.横状位,
                    PlaneTypeEnum.矢状位,
                    PlaneTypeEnum.冠状位,
                )
            ],
        )
        localization_result = {
            "correction_scope_bbox": {
                "x_min": 2,
                "x_max": 8,
                "y_min": 1,
                "y_max": 9,
                "z_min": 3,
                "z_max": 7,
            },
            "image": {
                "size_xyz": [12, 12, 12],
                "spacing_xyz": [1.0, 1.0, 1.0],
                "origin_xyz": [0.0, 0.0, 0.0],
                "direction": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            },
        }
        resampler = PlaneStackResampler()

        for plane_config in request.plane_configs:
            with self.subTest(plane=plane_config.plane.value):
                plan = build_whole_range_plan_from_bbox(
                    request,
                    plane_config,
                    localization_result,
                ).to_dict()
                planned_slice = plan["slices"][0]

                output = resampler.sample_slice(image, plan, planned_slice)

                expected_direction = (
                    planned_slice["row_dir"]["x"],
                    planned_slice["col_dir"]["x"],
                    planned_slice["normal"]["x"],
                    planned_slice["row_dir"]["y"],
                    planned_slice["col_dir"]["y"],
                    planned_slice["normal"]["y"],
                    planned_slice["row_dir"]["z"],
                    planned_slice["col_dir"]["z"],
                    planned_slice["normal"]["z"],
                )
                self.assertEqual(output.GetSize(), (12, 12, 1))
                self.assertEqual(output.GetDirection(), expected_direction)
                self.assertTrue(
                    np.any(sitk.GetArrayFromImage(output) > -1024.0)
                )

    def test_axis_aligned_slab_weights_partial_source_layers_by_overlap(self):
        """验证 `test_axis_aligned_slab_weights_partial_source_layers_by_overlap` 场景符合预期。"""

        image = sitk.GetImageFromArray(
            np.asarray([[[0]], [[10]], [[20]], [[30]], [[40]], [[50]]], dtype=np.int16),
        )
        image.SetSpacing((1.0, 1.0, 0.65))
        plan = {
            "extent_mm": {"width": 0.0, "height": 0.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 3.0,
        }
        planned_slice = {
            "center": {"x": 0.0, "y": 0.0, "z": 2.0},
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
            "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
        }

        output = PlaneStackResampler().sample_slice(image, plan, planned_slice)

        # slab 为 0.5..3.5mm：0.475*10 + 0.65*20 + 0.65*30 +
        # 0.65*40 + 0.575*50，再除以 3mm。
        self.assertAlmostEqual(
            float(sitk.GetArrayFromImage(output)[0, 0, 0]),
            92.0 / 3.0,
            delta=1e-3,
        )

    def test_reconstruction_consumes_mapping_plans_and_reports_physical_slice_count(self):
        """验证 `test_reconstruction_consumes_mapping_plans_and_reports_physical_slice_count` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                window_presets=["骨窗"],
            )

            ReconstructionStage(config={"parallel": False}).execute(context)

            result = context.get_result("reconstruction")
            self.assertEqual(result["plan_count"], 1)
            self.assertEqual(result["slice_count"], 2)
            self.assertEqual(result["image_count"], 2)
            self.assertEqual(result["planes"]["横状位"]["slice_count"], 2)

    def test_reconstruction_dispatches_curved_stack_resampler(self):
        """阶段五应按 geometry_type 把 CPR plan 交给 curved_stack resampler。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                geometry_type="curved_stack",
            )
            plan = context.get_result("mapping")["plans"][0]
            plan.update({
                "plan_id": "腰椎_CPR_矢状位",
                "cut_mode": "cpr",
                "plane": "矢状位",
                "group_name": "CPR",
                "curve_points": [
                    {"x": 2.0, "y": 2.0, "z": 2.0},
                    {"x": 2.0, "y": 2.0, "z": 4.0},
                ],
                "curve_spacing_mm": 1.0,
                "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "offsets_mm": [0.0],
            })
            plan["slices"] = [dict(plan["slices"][0], offset_mm=0.0)]

            sampled_geometry_types = []

            class RecordingCurvedResampler:
                """记录阶段五是否选择 CPR 重采样器。"""

                def sample_slice(self, image, current_plan, planned_slice):
                    """返回一张最小 HU 图像，并记录 geometry_type。"""

                    sampled_geometry_types.append(current_plan["geometry_type"])
                    return sitk.Image(1, 1, 1, sitk.sitkFloat32)

            with mock.patch(
                "src.stages.stage_reconstruction.CurvedStackResampler",
                return_value=RecordingCurvedResampler(),
            ), mock.patch(
                "src.stages.stage_reconstruction.write_ct_dicom",
            ):
                ReconstructionStage(config={"parallel": False}).execute(context)

            self.assertEqual(sampled_geometry_types, ["curved_stack"])

    def test_coronal_cpr_writes_only_localizer_without_sampling_slices(self):
        """冠状位 CPR 暂时只写定位图，不执行耗时的曲面切片重建。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                geometry_type="curved_stack",
            )
            plan = context.get_result("mapping")["plans"][0]
            plan.update({
                "plan_id": "腰椎_CPR_冠状位",
                "cut_mode": "cpr",
                "plane": "冠状位",
                "group_name": "CPR",
                "curve_points": [
                    {"x": 2.0, "y": 2.0, "z": 2.0},
                    {"x": 2.0, "y": 3.0, "z": 4.0},
                    {"x": 2.0, "y": 2.0, "z": 6.0},
                ],
                "curve_spacing_mm": 1.0,
                "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "offsets_mm": [0.0],
                "localizer_plane": "矢状位",
                "localizer_reference": {"x": 2.0, "y": 2.0, "z": 4.0},
            })
            plan["slices"] = [dict(plan["slices"][0], offset_mm=0.0)]

            class FailingCurvedResampler:
                """冠状位 CPR localizer-only 时不应被调用。"""

                def sample_slice(self, image, current_plan, planned_slice):
                    raise AssertionError("冠状位 CPR 不应采样实际切片")

            with mock.patch(
                "src.stages.stage_reconstruction.CurvedStackResampler",
                return_value=FailingCurvedResampler(),
            ), mock.patch(
                "src.stages.stage_reconstruction.write_ct_dicom",
            ) as write_slice:
                ReconstructionStage(config={"parallel": False}).execute(context)

            output_dir = Path(context.request.output_dir) / "冠状位" / "骨窗"
            result = context.get_result("reconstruction")

            write_slice.assert_not_called()
            self.assertTrue((output_dir / "Localizer.dcm").exists())
            self.assertFalse((output_dir / "1.dcm").exists())
            self.assertEqual(result["coronal"]["plan_count"], 1)
            self.assertEqual(result["coronal"]["slice_count"], 0)
            self.assertEqual(result["coronal"]["image_count"], 0)
            self.assertTrue(result["outputs"][0]["localizer_only"])

    def test_debug_mode_writes_cpr_localizer_png_in_window_directory(self):
        """CPR 定位图 PNG 应直接写在对应状位和窗位目录下。"""

        original_debug = _logger_manager.debug_log_file_enabled
        _logger_manager.debug_log_file_enabled = True
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                context = self._make_context_with_one_plan(
                    temp_dir,
                    geometry_type="curved_stack",
                )
                plan = context.get_result("mapping")["plans"][0]
                plan.update({
                    "plan_id": "腰椎_CPR_矢状位",
                    "cut_mode": "cpr",
                    "plane": "矢状位",
                    "group_name": "CPR",
                    "curve_points": [
                        {"x": 2.0, "y": 2.0, "z": 2.0},
                        {"x": 2.0, "y": 3.0, "z": 4.0},
                        {"x": 2.0, "y": 2.0, "z": 6.0},
                    ],
                    "curve_spacing_mm": 1.0,
                    "offset_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "offsets_mm": [0.0],
                    "localizer_plane": "矢状位",
                    "localizer_reference": {"x": 2.0, "y": 2.0, "z": 4.0},
                })
                plan["slices"] = [dict(plan["slices"][0], offset_mm=0.0)]

                ReconstructionStage(config={"parallel": False}).execute(context)

                output_dir = (
                    Path(context.request.output_dir)
                    / "矢状位"
                    / "骨窗"
                )
                self.assertTrue((output_dir / "Localizer.dcm").exists())
                self.assertTrue((output_dir / "Localizer.png").exists())
                self.assertFalse((output_dir / "CPR").exists())
        finally:
            _logger_manager.debug_log_file_enabled = original_debug

    def test_reconstruction_rejects_unsupported_geometry_type_with_plan_id(self):
        """未知 geometry_type 应明确报出 plan_id。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                geometry_type="surface_stack",
            )

            with self.assertRaisesRegex(ValueError, r"plan_id=.*geometry_type"):
                ReconstructionStage(config={"parallel": False}).execute(context)

    def test_slab_thickness_averages_source_planes_without_adding_slices(self):
        """验证 `test_slab_thickness_averages_source_planes_without_adding_slices` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=1,
                window_presets=["骨窗"],
            )
            raw_path = Path(
                context.get_result("localization")["raw_for_reconstruction_path"]
            )
            raw_image = sitk.GetImageFromArray(
                np.asarray([[[0]], [[10]], [[20]]], dtype=np.int16),
            )
            raw_image.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(raw_image, str(raw_path))

            plan = context.get_result("mapping")["plans"][0]
            plan["slice_thickness"] = 3.0
            plan["slice_spacing"] = 5.0
            plan["extent_mm"] = {"width": 0.0, "height": 0.0}
            plan["slices"][0]["center"] = {"x": 0.0, "y": 0.0, "z": 1.0}

            ReconstructionStage(config={"parallel": False}).execute(context)

            result = context.get_result("reconstruction")
            self.assertEqual(result["slice_count"], 1)
            dcm_path = next(Path(temp_dir).rglob("1.dcm"))
            output = sitk.ReadImage(str(dcm_path))
            expected_gray = round((10.0 + 850.0) / 2500.0 * 65535.0)
            self.assertEqual(
                int(sitk.GetArrayFromImage(output)[0, 0, 0]),
                expected_gray,
            )

    def test_reconstruction_writes_one_series_per_window_preset(self):
        """验证 `test_reconstruction_writes_one_series_per_window_preset` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                window_presets=["骨窗", "脑窗"],
            )
            plan = context.get_result("mapping")["plans"][0]
            plan["slice_thickness"] = 5.0
            plan["slice_spacing"] = 5.0
            plan["range_end"]["z"] = 7.0
            plan["path_points"][-1]["z"] = 7.0
            plan["slices"][-1]["center"]["z"] = 7.0

            # 多窗位使用并行路径，验证并发写出的两个 DICOM 序列都完整。
            ReconstructionStage(config={"parallel": True}).execute(context)

            root = Path(context.request.output_dir)
            self.assertEqual(
                len(list((root / "横状位" / "骨窗").rglob("*.dcm"))),
                3,
            )
            self.assertEqual(
                len(list((root / "横状位" / "脑窗").rglob("*.dcm"))),
                3,
            )
            self.assertEqual(
                len(list((root / "横状位" / "骨窗").glob("[0-9]*.dcm"))),
                2,
            )
            self.assertFalse((root / "横状位" / "骨窗" / "整切").exists())
            bone_paths = [
                root / "横状位" / "骨窗" / "1.dcm",
                root / "横状位" / "骨窗" / "2.dcm",
            ]
            self.assertTrue(all(path.exists() for path in bone_paths))
            for path in bone_paths:
                dataset = pydicom.dcmread(str(path))
                self.assertEqual(float(dataset.SliceThickness), 5.0)
                self.assertEqual(float(dataset.SpacingBetweenSlices), 5.0)
            first_position = pydicom.dcmread(str(bone_paths[0])).ImagePositionPatient
            second_position = pydicom.dcmread(str(bone_paths[1])).ImagePositionPatient
            self.assertAlmostEqual(
                abs(float(second_position[2]) - float(first_position[2])),
                5.0,
            )
            bone_images = [sitk.ReadImage(str(path)) for path in bone_paths]
            self.assertEqual(
                len({image.GetMetaData("0020|000e") for image in bone_images}),
                1,
            )
            self.assertIn("_1_2", bone_images[0].GetMetaData("0008|103e"))
            self.assertIn("_2_2", bone_images[1].GetMetaData("0008|103e"))
            self.assertEqual(
                context.get_result("reconstruction")["image_count"],
                4,
            )

    def test_written_dicom_contains_slice_geometry_and_window_tags(self):
        """验证 `test_written_dicom_contains_slice_geometry_and_window_tags` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=1,
                window_presets=["骨窗"],
            )

            ReconstructionStage(config={"parallel": False}).execute(context)

            dcm_path = next(Path(context.request.output_dir).rglob("1.dcm"))
            image = sitk.ReadImage(str(dcm_path))
            self.assertEqual(image.GetMetaData("0008|0060"), "CT")
            self.assertEqual(float(image.GetMetaData("0018|0050")), 1.0)
            self.assertEqual(image.GetPixelID(), sitk.sitkUInt16)
            self.assertEqual(image.GetMetaData("0028|1050").strip(), "32768")
            self.assertEqual(image.GetMetaData("0028|1051").strip(), "65535")
            self.assertIn("骨窗", image.GetMetaData("0008|103e"))
            self.assertIn("\\", image.GetMetaData("0020|0037"))

    def test_geometry_tag_restoration_overrides_gdcm_default_values(self):
        """GDCM 写成默认值后，应能按计划原子恢复层厚和层间距。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "slice.dcm"
            image = sitk.GetImageFromArray(np.zeros((2, 2), dtype=np.uint16))
            image.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
            image.SetMetaData("0008|0018", "1.2.826.0.1.3680043.2.1125.901")
            image.SetMetaData("0018|0050", "1")
            image.SetMetaData("0018|0088", "1")
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(str(output_path))
            writer.Execute(image)

            dicom_writer._restore_planned_slice_geometry_tags(
                output_path,
                slice_thickness=5.0,
                slice_spacing=5.0,
            )

            dataset = pydicom.dcmread(str(output_path))
            self.assertEqual(float(dataset.SliceThickness), 5.0)
            self.assertEqual(float(dataset.SpacingBetweenSlices), 5.0)

    def test_localizer_restores_sagittal_geometry_after_gdcm_write(self):
        """定位图写盘后仍应保留矢状位的物理行列方向。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "Localizer.dcm"
            image = sitk.GetImageFromArray(
                np.zeros((1, 4, 5), dtype=np.uint16),
            )
            image.SetMetaData("0020|0032", "10\\20\\30")
            image.SetMetaData("0020|0037", "0\\1\\0\\0\\0\\-1")

            dicom_writer.write_localizer_dicom(
                image=image,
                output_path=output_path,
                window=WindowPreset(
                    name="骨窗",
                    window_width=1000.0,
                    window_center=400.0,
                ),
                series_uid="1.2.826.0.1.3680043.2.1125.701",
                study_uid="1.2.826.0.1.3680043.2.1125.702",
            )

            dataset = pydicom.dcmread(str(output_path))
            self.assertEqual(
                [float(value) for value in dataset.ImagePositionPatient],
                [10.0, 20.0, 30.0],
            )
            self.assertEqual(
                [float(value) for value in dataset.ImageOrientationPatient],
                [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            )

    def test_ct_dicom_write_failure_does_not_leave_partial_formal_file(self):
        """标签校正失败时，不应在正式文件名下留下未校正的 DICOM。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(temp_dir, slice_count=1)

            with mock.patch.object(
                pydicom.dataset.Dataset,
                "save_as",
                side_effect=RuntimeError("模拟标签校正失败"),
            ):
                with self.assertRaisesRegex(RuntimeError, "模拟标签校正失败"):
                    ReconstructionStage(config={"parallel": False}).execute(context)

            output_dir = Path(context.request.output_dir) / "横状位" / "骨窗"
            self.assertFalse((output_dir / "1.dcm").exists())
            self.assertEqual(list(output_dir.glob(".*.tmp.dcm")), [])

    def test_stage_writes_localizer_without_counting_it_as_slice(self):
        """验证 `test_stage_writes_localizer_without_counting_it_as_slice` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                window_presets=["骨窗"],
            )
            ReconstructionStage(config={"parallel": False}).execute(context)

            output_dir = Path(context.request.output_dir) / "横状位" / "骨窗"
            self.assertTrue((output_dir / "1.dcm").exists())
            self.assertTrue((output_dir / "2.dcm").exists())
            localizer_path = output_dir / "Localizer.dcm"
            self.assertTrue(localizer_path.exists())
            localizer = sitk.ReadImage(str(localizer_path))
            self.assertIn(
                "LOCALIZER",
                localizer.GetMetaData("0008|0008").upper(),
            )
            orientation = tuple(float(value) for value in (
                localizer.GetMetaData("0020|0037").split("\\")
            ))
            self.assertEqual(orientation, (0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
            result = context.get_result("reconstruction")
            self.assertEqual(result["slice_count"], 2)
            self.assertEqual(result["image_count"], 2)

    def test_debug_mode_writes_localizer_png_next_to_localizer_dicom(self):
        """debug 模式下每个窗位目录应同时写出定位图 PNG。"""

        original_debug = _logger_manager.debug_log_file_enabled
        _logger_manager.debug_log_file_enabled = True
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                context = self._make_context_with_one_plan(
                    temp_dir,
                    slice_count=2,
                    window_presets=["骨窗"],
                )

                ReconstructionStage(config={"parallel": False}).execute(context)

                output_dir = Path(context.request.output_dir) / "横状位" / "骨窗"
                png_path = output_dir / "Localizer.png"
                self.assertTrue(png_path.exists())
                png_image = sitk.ReadImage(str(png_path))
                self.assertEqual(png_image.GetSize(), (8, 8))
        finally:
            _logger_manager.debug_log_file_enabled = original_debug

    def test_non_debug_mode_does_not_write_localizer_png(self):
        """非 debug 模式不应把定位图 PNG 放进正式窗位目录。"""

        original_debug = _logger_manager.debug_log_file_enabled
        _logger_manager.debug_log_file_enabled = False
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                context = self._make_context_with_one_plan(
                    temp_dir,
                    slice_count=2,
                    window_presets=["骨窗"],
                )

                ReconstructionStage(config={"parallel": False}).execute(context)

                output_dir = Path(context.request.output_dir) / "横状位" / "骨窗"
                self.assertTrue((output_dir / "Localizer.dcm").exists())
                self.assertFalse((output_dir / "Localizer.png").exists())
        finally:
            _logger_manager.debug_log_file_enabled = original_debug

    def test_derived_dicom_inherits_source_study_and_patient_metadata(self):
        """验证 `test_derived_dicom_inherits_source_study_and_patient_metadata` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            source_dir.mkdir()
            source_path = source_dir / "source.dcm"
            source = sitk.GetImageFromArray(np.asarray([[10]], dtype=np.int16))
            source.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
            source.SetMetaData("0008|0018", "1.2.826.0.1.3680043.2.1125.100")
            source.SetMetaData("0008|0060", "CT")
            source.SetMetaData("0008|0020", "20260710")
            source.SetMetaData("0008|0030", "121314")
            source.SetMetaData("0008|0050", "ACC-001")
            source.SetMetaData("0010|0010", "TEST^PATIENT")
            source.SetMetaData("0010|0020", "PID-001")
            source.SetMetaData("0010|0030", "19800101")
            source.SetMetaData("0010|0040", "F")
            source.SetMetaData("0018|0050", "1")
            source.SetMetaData("0018|0088", "1")
            source.SetMetaData("0020|000d", "1.2.826.0.1.3680043.2.1125.200")
            source.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125.300")
            source.SetMetaData("0020|0010", "STUDY-001")
            source.SetMetaData("0020|0052", "1.2.826.0.1.3680043.2.1125.400")
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(str(source_path))
            writer.Execute(source)

            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                dicom_dir=str(source_dir),
            )
            plan = context.get_result("mapping")["plans"][0]
            plan["slice_thickness"] = 5.0
            plan["slice_spacing"] = 5.0
            plan["range_end"]["z"] = 7.0
            plan["path_points"][-1]["z"] = 7.0
            plan["slices"][-1]["center"]["z"] = 7.0
            ReconstructionStage(config={"parallel": False}).execute(context)

            output_paths = [
                Path(context.request.output_dir) / "横状位" / "骨窗" / "1.dcm",
                Path(context.request.output_dir) / "横状位" / "骨窗" / "2.dcm",
            ]
            first = sitk.ReadImage(str(output_paths[0]))
            second = sitk.ReadImage(str(output_paths[1]))
            self.assertEqual(first.GetMetaData("0010|0010").strip(), "TEST^PATIENT")
            self.assertEqual(first.GetMetaData("0010|0020").strip(), "PID-001")
            self.assertEqual(first.GetMetaData("0010|0040").strip(), "F")
            self.assertEqual(first.GetMetaData("0008|0020").strip(), "20260710")
            self.assertEqual(first.GetMetaData("0008|0050").strip(), "ACC-001")
            self.assertEqual(
                first.GetMetaData("0020|000d").strip(),
                "1.2.826.0.1.3680043.2.1125.200",
            )
            self.assertEqual(
                first.GetMetaData("0020|0052").strip(),
                "1.2.826.0.1.3680043.2.1125.400",
            )
            self.assertNotEqual(
                first.GetMetaData("0020|000e"),
                "1.2.826.0.1.3680043.2.1125.300",
            )
            self.assertEqual(
                first.GetMetaData("0020|000e"),
                second.GetMetaData("0020|000e"),
            )
            self.assertNotEqual(
                first.GetMetaData("0008|0018"),
                second.GetMetaData("0008|0018"),
            )
            dataset = pydicom.dcmread(str(output_paths[0]))
            self.assertEqual(float(dataset.SliceThickness), 5.0)
            self.assertEqual(float(dataset.SpacingBetweenSlices), 5.0)

    def test_derived_dicom_decodes_gb18030_surrogate_metadata(self):
        """验证 `test_derived_dicom_decodes_gb18030_surrogate_metadata` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(temp_dir, slice_count=1)
            context.set_result("preprocessing", {
                "source_dicom_metadata": {
                    "0008|0005": "GB18030",
                    "0008|1030": (
                        "11_\udcd0\udcd8ȫ\udcb8\udcb9ƽɨ"
                        "\udca3\udca8\udcb3\udcc9\udcc8ˣ\udca9"
                    ),
                    "0010|0010": "张三".encode("gb18030").decode(
                        "utf-8",
                        errors="surrogateescape",
                    ),
                    "0020|000d": "1.2.826.0.1.3680043.2.1125.500",
                },
            })

            ReconstructionStage(config={"parallel": False}).execute(context)

            dcm_path = next(Path(context.request.output_dir).rglob("1.dcm"))
            image = sitk.ReadImage(str(dcm_path))
            description = image.GetMetaData("0008|1030")
            self.assertNotRegex(description, r"[\ud800-\udfff]")
            self.assertEqual(description.strip(), "11_胸全腹平扫（成人）")
            dataset = pydicom.dcmread(str(dcm_path))
            self.assertEqual(dataset.SpecificCharacterSet, "ISO_IR 192")
            self.assertEqual(str(dataset.PatientName), "张三")
            self.assertEqual(dataset.StudyDescription, "11_胸全腹平扫（成人）")

    @staticmethod
    def _make_context_with_one_plan(
        temp_dir: str,
        *,
        slice_count: int = 2,
        window_presets: list[str] | None = None,
        geometry_type: str = "plane_stack",
        dicom_dir: str = "/tmp/dicom",
    ) -> ProcessingContext:
        """为测试准备 `_make_context_with_one_plan` 所需的数据或记录行为。"""

        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "raw.nii.gz"
        raw_image = sitk.GetImageFromArray(
            np.zeros((8, 8, 8), dtype=np.int16),
        )
        raw_image.SetSpacing((1.0, 1.0, 1.0))
        raw_image.SetOrigin((0.0, 0.0, 0.0))
        raw_image.SetDirection((
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ))
        sitk.WriteImage(raw_image, str(raw_path))

        centers = [2.0 + 2.0 * index for index in range(slice_count)]
        plan = {
            "plan_id": "测试_整切_横状位",
            "body_part": "腰椎",
            "cut_mode": "whole_range",
            "plane": "横状位",
            "group_name": "whole_range",
            "geometry_type": geometry_type,
            "range_start": {"x": 2.0, "y": 2.0, "z": 2.0},
            "range_end": {"x": 2.0, "y": 2.0, "z": centers[-1]},
            "path_points": [
                {"x": 2.0, "y": 2.0, "z": 2.0},
                {"x": 2.0, "y": 2.0, "z": centers[-1]},
            ],
            "slices": [
                {
                    "index": index + 1,
                    "center": {"x": 2.0, "y": 2.0, "z": center},
                    "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                }
                for index, center in enumerate(centers)
            ],
            "extent_mm": {"width": 4.0, "height": 4.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 1.0,
            "slice_spacing": 2.0,
            "window_presets": window_presets or ["骨窗"],
            "source_labels": ["correction_scope_bbox"],
        }

        request = Request(
            dicom_dir=dicom_dir,
            output_dir=str(output_dir),
            body_part=BodyPartEnum.腰椎,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=1.0,
                    slice_spacing=2.0,
                    window_presets=["骨窗"],
                ),
            ],
        )
        context = ProcessingContext(request)
        context.set_result("localization", {
            "raw_for_reconstruction_path": str(raw_path),
        })
        context.set_result("mapping", {"plans": [plan]})
        return context

    @staticmethod
    def _make_ramp_image(size: tuple[int, int, int]) -> sitk.Image:
        """为测试准备 `_make_ramp_image` 所需的数据或记录行为。"""

        size_x, size_y, size_z = size
        array = np.zeros((size_z, size_y, size_x), dtype=np.int16)
        for z in range(size_z):
            for y in range(size_y):
                for x in range(size_x):
                    array[z, y, x] = x + 10 * y + 100 * z
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection((
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ))
        return image


if __name__ == "__main__":
    unittest.main()
