import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.utils.Logger_utils import _logger_manager
_logger_manager.debug_log_file_enabled = True

from src.core.context import ProcessingContext
from src.models import (
    LabelCoordinate,
    LabelTarget,
    LocalizationStageResult,
    Request,
    VoxelBBox,
    VoxelPoint,
)
from src.stages.stage_localization import LocalizationStage
from src.stages.stage_reconstruction import ReconstructionStage


class Stage3TargetAxialCorrectionTest(unittest.TestCase):
    """汇总 `Stage3TargetAxialCorrectionTest` 相关测试场景。"""

    def test_slice_range_extension_config_accepts_optional_endpoints(self):
        """切片范围扩展允许只配置单端，未配置端由第四步按零处理。"""

        LocalizationStage._validate_cut_config("whole_range", {
            "start": "lung",
            "end": "lung",
            "slice_range_extension": {
                "横状位": {"start": 3.0},
                "矢状位": {"end": 2.0},
            },
        })

    def test_slice_range_extension_config_rejects_invalid_values(self):
        """范围扩展仅接受现有状位、start/end 字段和非负有限毫米数。"""

        invalid_extensions = (
            ({"斜状位": {"start": 1.0}}, "不支持的状位"),
            ({"横状位": []}, "必须是对象"),
            ({"横状位": {"before": 1.0}}, "仅支持 start/end"),
            ({"横状位": {"start": -1.0}}, "不能小于 0"),
            ({"横状位": {"start": True}}, "必须是有限数字"),
            ({"横状位": {"end": math.inf}}, "必须是有限数字"),
        )
        for extension, message in invalid_extensions:
            with self.subTest(extension=extension):
                with self.assertRaisesRegex(ValueError, message):
                    LocalizationStage._validate_cut_config("whole_range", {
                        "start": "lung",
                        "end": "lung",
                        "slice_range_extension": extension,
                    })

    def test_lung_builds_typed_anatomical_geometry_for_stage4(self):
        """验证 `test_lung_builds_typed_anatomical_geometry_for_stage4` 场景符合预期。"""

        stage = LocalizationStage()
        mask = np.zeros((32, 24, 48), dtype=bool)
        for x in range(6, 42):
            top = 5 + int(round((x - 6) * 0.1))
            mask[top:28, 4:20, x] = True
        image = sitk.GetImageFromArray(mask.astype(np.uint8))

        geometry, diagnostics = stage._build_anatomical_geometry(
            body_part="肺部",
            target_mask=mask,
            image=image,
        )

        self.assertIsNotNone(geometry)
        self.assertEqual(geometry.global_frame.method, "lung_dicom_axes")
        self.assertFalse(geometry.global_frame.applied)
        self.assertEqual(diagnostics["angle_source"], "disabled")
        self.assertEqual(geometry.local_frames, {})

    def test_non_lung_does_not_enable_anatomical_geometry_yet(self):
        """验证 `test_non_lung_does_not_enable_anatomical_geometry_yet` 场景符合预期。"""

        stage = LocalizationStage()
        mask = np.ones((4, 4, 4), dtype=bool)
        image = sitk.GetImageFromArray(mask.astype(np.uint8))

        geometry, diagnostics = stage._build_anatomical_geometry(
            body_part="头骨",
            target_mask=mask,
            image=image,
        )

        self.assertIsNone(geometry)
        self.assertEqual(diagnostics, {})

    def test_cervical_and_thoracic_build_spine_geometry_from_configured_endpoints(self):
        """颈椎和胸椎应共用首尾椎体质心确定的三维解剖坐标。"""

        stage = LocalizationStage()
        shape = (40, 32, 40)
        image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
        for body_part, start_label, end_label in (
            ("颈椎", "vertebrae_C1", "vertebrae_C7"),
            ("胸椎", "vertebrae_C7", "vertebrae_L1"),
        ):
            label_masks = {
                start_label: np.zeros(shape, dtype=bool),
                end_label: np.zeros(shape, dtype=bool),
            }
            label_masks[start_label][30:34, 8:12, 10:14] = True
            label_masks[end_label][6:10, 20:24, 26:30] = True
            target_mask = np.logical_or.reduce(list(label_masks.values()))

            geometry, diagnostics = stage._build_anatomical_geometry(
                body_part=body_part,
                target_mask=target_mask,
                image=image,
                label_masks=label_masks,
                start_label=start_label,
                end_label=end_label,
            )

            self.assertIsNotNone(geometry)
            self.assertEqual(geometry.global_frame.method, "spine_endpoint_centroids")
            self.assertTrue(geometry.global_frame.applied)
            self.assertEqual(diagnostics["start_label"], start_label)
            self.assertEqual(diagnostics["end_label"], end_label)

    def test_hip_builds_geometry_from_left_and_right_masks(self):
        """髋骨应使用配置中的左右独立 Mask 计算外侧顶部横切坐标。"""

        stage = LocalizationStage()
        shape = (32, 20, 40)
        image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
        label_masks = {
            "hip_left": np.zeros(shape, dtype=bool),
            "hip_right": np.zeros(shape, dtype=bool),
        }
        label_masks["hip_left"][8:28, 5:15, 4:14] = True
        label_masks["hip_right"][6:24, 5:15, 26:36] = True

        geometry, diagnostics = stage._build_anatomical_geometry(
            body_part="髋骨",
            target_mask=np.logical_or.reduce(list(label_masks.values())),
            image=image,
            label_masks=label_masks,
            start_label="hip_left",
            end_label="hip_right",
        )

        self.assertIsNotNone(geometry)
        self.assertEqual(geometry.global_frame.method, "hip_outer_superior_caps")
        self.assertEqual(diagnostics["left_label"], "hip_left")
        self.assertEqual(diagnostics["right_label"], "hip_right")

    def test_target_mask_and_bbox_only_include_requested_labels(self):
        # 阶段三不能拿整套 volume 判断歪斜，必须先合成“本次任务真正关心的目标 mask”。
        """验证 `test_target_mask_and_bbox_only_include_requested_labels` 场景符合预期。"""

        stage = LocalizationStage()
        seg_array = np.zeros((5, 12, 12), dtype=np.uint8)
        seg_array[1:3, 2:5, 4:7] = 5
        seg_array[4, 6, 8] = 9
        seg_array[0, 0, 0] = 99
        label_targets = {
            "vertebrae_T1": LabelTarget("vertebrae_T1", [5]),
            "vertebrae_T2": LabelTarget("vertebrae_T2", [9]),
        }

        target_mask = stage._build_target_mask(seg_array, label_targets)
        target_bbox = stage._build_target_bbox(target_mask)

        self.assertEqual(int(target_mask.sum()), 19)
        self.assertFalse(target_mask[0, 0, 0])
        self.assertEqual(target_bbox.to_dict(), {
            "z_min": 1,
            "z_max": 4,
            "y_min": 2,
            "y_max": 6,
            "x_min": 4,
            "x_max": 8,
        })

    def test_detect_axial_skew_uses_xy_projection_angle(self):
        # 横状位歪斜会影响后续矢状位/冠状位切片，所以第一版先检测 XY 投影主方向。
        """验证 `test_detect_axial_skew_uses_xy_projection_angle` 场景符合预期。"""

        stage = LocalizationStage()
        target_mask = np.zeros((1, 80, 80), dtype=bool)
        expected_angle = 30.0
        radians = math.radians(expected_angle)
        for distance in np.linspace(-24, 24, 160):
            x = int(round(40 + distance * math.cos(radians)))
            y = int(round(40 + distance * math.sin(radians)))
            target_mask[0, y - 1:y + 2, x - 1:x + 2] = True

        skew_info = stage._detect_axial_skew(target_mask)

        self.assertTrue(skew_info["enough_points"])
        self.assertAlmostEqual(abs(skew_info["angle_degrees"]), expected_angle, delta=3.0)

    def test_detect_axial_skew_reports_not_enough_points(self):
        # 目标体素太少时 PCA 方向没有意义，应该明确标记而不是给一个假角度。
        """验证 `test_detect_axial_skew_reports_not_enough_points` 场景符合预期。"""

        stage = LocalizationStage()
        target_mask = np.zeros((2, 5, 5), dtype=bool)
        target_mask[0, 2, 2] = True

        skew_info = stage._detect_axial_skew(target_mask)

        self.assertFalse(skew_info["enough_points"])
        self.assertEqual(skew_info["angle_degrees"], 0.0)
        self.assertFalse(skew_info["should_correct"])

    def test_find_dataset_json_prefers_segmentation_sidecar_file(self):
        # nnUNet 模型目录可能有多个 dataset.json；seg 同目录的 sidecar 最贴近当前分割结果。
        """验证 `test_find_dataset_json_prefers_segmentation_sidecar_file` 场景符合预期。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_dir = temp_path / "segmentation"
            model_dir = temp_path / "nnUNet_results" / "Dataset001_model"
            seg_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            seg_path = seg_dir / "seg.nii.gz"
            sidecar_dataset_json = seg_dir / "dataset.json"
            model_dataset_json = model_dir / "dataset.json"
            self._write_image(seg_path, np.zeros((1, 4, 4), dtype=np.uint8))
            sidecar_dataset_json.write_text(
                json.dumps({"labels": {"background": 0, "vertebrae_C1": 26}}),
                encoding="utf-8",
            )
            model_dataset_json.write_text(
                json.dumps({"labels": {"background": 0, "wrong_model_label": 1}}),
                encoding="utf-8",
            )
            stage._nnunet_paths = {
                "nnUNet_results": str(temp_path / "nnUNet_results"),
                "nnUNet_preprocessed": "",
                "nnUNet_raw": "",
            }

            dataset_json_path = stage._find_dataset_json({"dataset_id": 1}, seg_path=str(seg_path))

            self.assertEqual(dataset_json_path, sidecar_dataset_json)

    def test_whole_range_expands_ordered_vertebrae_between_start_and_end(self):
        # 脊柱整切不能只定位两端点；C1 到 C7 应覆盖中间所有颈椎 label。
        """验证 `test_whole_range_expands_ordered_vertebrae_between_start_and_end` 场景符合预期。"""

        stage = LocalizationStage()
        resolved_cut_config = {
            "start": "C1",
            "end": "C7",
            "start_labels": ["vertebrae_C1"],
            "end_labels": ["vertebrae_C7"],
        }
        label_map = {
            f"vertebrae_C{index}": [label_id]
            for index, label_id in zip(range(1, 8), range(26, 19, -1))
        }

        label_targets = stage._resolve_label_targets(resolved_cut_config, label_map)

        self.assertEqual(list(label_targets.keys()), [
            "vertebrae_C1",
            "vertebrae_C2",
            "vertebrae_C3",
            "vertebrae_C4",
            "vertebrae_C5",
            "vertebrae_C6",
            "vertebrae_C7",
        ])

    def test_cervical_whole_range_keeps_c1_for_cut_coverage(self):
        """颈椎切图覆盖仍需保留环形 C1，避免其他状位缺少上端范围。"""

        config = json.loads(
            Path("src/config/localization_config.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["targets"]["颈椎"]["whole_range"], {
            "start": "C1",
            "end": "C7",
        })

    def test_s1_range_prefers_vertebrae_s1_over_sacrum_when_available(self):
        # S1 这个业务名表示第一骶椎。模型同时提供 vertebrae_S1 和 sacrum 时，
        # 整切范围应该纳入 vertebrae_S1，不能直接跳到整块骶骨。
        """验证 `test_s1_range_prefers_vertebrae_s1_over_sacrum_when_available` 场景符合预期。"""

        stage = LocalizationStage()
        resolved_cut_config = {
            "start": "L1",
            "end": "S1",
            "start_labels": ["vertebrae_L1"],
            "end_labels": ["vertebrae_S1"],
        }
        label_map = {
            "vertebrae_L1": [7],
            "vertebrae_L2": [6],
            "vertebrae_L3": [5],
            "vertebrae_L4": [4],
            "vertebrae_L5": [3],
            "vertebrae_S1": [2],
            "sacrum": [1],
        }

        label_targets = stage._resolve_label_targets(resolved_cut_config, label_map)

        self.assertEqual(list(label_targets.keys()), [
            "vertebrae_L1",
            "vertebrae_L2",
            "vertebrae_L3",
            "vertebrae_L4",
            "vertebrae_L5",
            "vertebrae_S1",
        ])

    def test_s1_range_falls_back_to_sacrum_when_vertebrae_s1_missing(self):
        # 部分模型没有 vertebrae_S1，只提供 sacrum；这种情况下继续回退到 sacrum。
        """验证 `test_s1_range_falls_back_to_sacrum_when_vertebrae_s1_missing` 场景符合预期。"""

        stage = LocalizationStage()
        resolved_cut_config = {
            "start": "L1",
            "end": "S1",
            "start_labels": ["vertebrae_L1"],
            "end_labels": ["vertebrae_S1"],
        }
        label_map = {
            "vertebrae_L1": [7],
            "vertebrae_L2": [6],
            "vertebrae_L3": [5],
            "vertebrae_L4": [4],
            "vertebrae_L5": [3],
            "sacrum": [1],
        }

        label_targets = stage._resolve_label_targets(resolved_cut_config, label_map)

        self.assertEqual(list(label_targets.keys()), [
            "vertebrae_L1",
            "vertebrae_L2",
            "vertebrae_L3",
            "vertebrae_L4",
            "vertebrae_L5",
            "sacrum",
        ])

    def test_vertebrae_axial_skew_aggregates_per_label_angles_and_skips_boundary_labels(self):
        # 脊柱类目标用每个椎体的角度做稳健聚合；贴边残缺 label 不参与回正判断。
        """验证 `test_vertebrae_axial_skew_aggregates_per_label_angles_and_skips_boundary_labels` 场景符合预期。"""

        stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
        seg_array = np.zeros((3, 100, 100), dtype=np.uint8)
        self._draw_oriented_label(seg_array, label_id=20, z=1, center_y=35, center_x=40, angle_degrees=20.0)
        self._draw_oriented_label(seg_array, label_id=21, z=1, center_y=60, center_x=55, angle_degrees=22.0)
        self._draw_oriented_label(seg_array, label_id=22, z=0, center_y=5, center_x=5, angle_degrees=-60.0)
        label_targets = {
            "vertebrae_C7": LabelTarget("vertebrae_C7", [20]),
            "vertebrae_C6": LabelTarget("vertebrae_C6", [21]),
            "vertebrae_C5": LabelTarget("vertebrae_C5", [22]),
        }

        skew_info = stage._detect_target_axial_skew(seg_array, label_targets)

        self.assertEqual(skew_info["method"], "vertebrae_label_angle_aggregation")
        self.assertAlmostEqual(skew_info["angle_degrees"], 21.0, delta=3.0)
        self.assertTrue(skew_info["should_correct"])
        self.assertEqual(skew_info["used_label_names"], ["vertebrae_C7", "vertebrae_C6"])
        self.assertEqual(skew_info["skipped_label_names"], ["vertebrae_C5"])

    def test_vertebrae_axial_skew_excludes_c1_ring_from_positioning(self):
        """回正角度定位不使用环形 C1，但仍允许其他流程保留 C1 坐标。"""

        stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
        seg_array = np.zeros((3, 100, 100), dtype=np.uint8)
        self._draw_oriented_label(seg_array, label_id=20, z=1, center_y=35, center_x=40, angle_degrees=20.0)
        self._draw_oriented_label(seg_array, label_id=21, z=1, center_y=60, center_x=55, angle_degrees=22.0)
        self._draw_oriented_label(seg_array, label_id=22, z=1, center_y=78, center_x=75, angle_degrees=-60.0)
        label_targets = {
            "vertebrae_C2": LabelTarget("vertebrae_C2", [20]),
            "vertebrae_C3": LabelTarget("vertebrae_C3", [21]),
            "vertebrae_C1": LabelTarget("vertebrae_C1", [22]),
        }

        skew_info = stage._detect_target_axial_skew(seg_array, label_targets)

        self.assertEqual(skew_info["used_label_names"], ["vertebrae_C2", "vertebrae_C3"])
        self.assertEqual(skew_info["skipped_label_names"], ["vertebrae_C1"])
        self.assertAlmostEqual(skew_info["angle_degrees"], 21.0, delta=3.0)

    def test_debug_bbox_overlay_writes_edges_on_reference_grid(self):
        # 研发期用这个 overlay 在 ITK-SNAP 里直接看定位框；它只画边框，不填满盒子。
        """验证 `test_debug_bbox_overlay_writes_edges_on_reference_grid` 场景符合预期。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            reference_image = sitk.GetImageFromArray(np.zeros((6, 10, 10), dtype=np.uint8))
            reference_image.SetSpacing((0.5, 0.6, 1.5))
            reference_image.SetOrigin((10.0, 20.0, 30.0))
            label_coordinates = {
                "vertebrae_C1": LabelCoordinate(
                    label_name="vertebrae_C1",
                    label_ids=[26],
                    voxel_count=10,
                    found=True,
                    centroid=VoxelPoint(z=2.0, y=3.0, x=4.0),
                    bbox=VoxelBBox(z_min=1, z_max=3, y_min=2, y_max=5, x_min=3, x_max=6),
                ),
                "vertebrae_C2": LabelCoordinate(
                    label_name="vertebrae_C2",
                    label_ids=[25],
                    voxel_count=8,
                    found=True,
                    centroid=VoxelPoint(z=4.0, y=6.0, x=6.0),
                    bbox=VoxelBBox(z_min=3, z_max=4, y_min=6, y_max=8, x_min=5, x_max=8),
                ),
            }
            overlay_path = temp_path / "定位框.nii.gz"

            written_path = stage._write_debug_bbox_overlay(
                reference_image=reference_image,
                label_coordinates=label_coordinates,
                correction_scope_bbox=VoxelBBox(z_min=0, z_max=5, y_min=0, y_max=9, x_min=0, x_max=9),
                output_path=overlay_path,
            )

            overlay_image = sitk.ReadImage(str(overlay_path))
            overlay_array = sitk.GetArrayFromImage(overlay_image)
            self.assertEqual(written_path, str(overlay_path))
            self.assertEqual(overlay_image.GetSize(), reference_image.GetSize())
            for actual, expected in zip(overlay_image.GetSpacing(), reference_image.GetSpacing()):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(overlay_image.GetOrigin(), reference_image.GetOrigin())
            self.assertEqual(overlay_array[0, 0, 0], 1)
            self.assertEqual(overlay_array[1, 2, 3], 0)
            self.assertEqual(overlay_array[2, 3, 4], 0)

            label_overlay_path = temp_path / "localization_label_bbox_overlay.nii.gz"
            stage._write_debug_bbox_overlay(
                reference_image=reference_image,
                label_coordinates=label_coordinates,
                correction_scope_bbox=VoxelBBox(z_min=0, z_max=5, y_min=0, y_max=9, x_min=0, x_max=9),
                output_path=label_overlay_path,
                include_label_bboxes=True,
            )
            label_overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(label_overlay_path)))
            self.assertEqual(label_overlay_array[1, 2, 3], 26)
            self.assertEqual(label_overlay_array[3, 6, 5], 25)

    def test_debug_target_mask_overlay_writes_only_requested_target_voxels(self):
        # bbox 是外接盒，矢状位/冠状位上会被极值体素撑大；研发期再输出一个
        # 真实目标 mask overlay，用来在 ITK-SNAP 里核对定位目标本身有没有偏。
        """验证 `test_debug_target_mask_overlay_writes_only_requested_target_voxels` 场景符合预期。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((3, 5, 6), dtype=np.uint8)
            seg_array[0, 1, 2] = 20
            seg_array[1, 2, 3] = 21
            seg_array[2, 3, 4] = 99
            reference_image = sitk.GetImageFromArray(seg_array)
            reference_image.SetSpacing((0.5, 0.6, 1.5))
            reference_image.SetOrigin((10.0, 20.0, 30.0))
            label_targets = {
                "vertebrae_C7": LabelTarget("vertebrae_C7", [20]),
                "vertebrae_C6": LabelTarget("vertebrae_C6", [21]),
            }
            overlay_path = temp_path / "目标掩膜.nii.gz"

            written_path = stage._write_debug_target_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_image = sitk.ReadImage(str(overlay_path))
            overlay_array = sitk.GetArrayFromImage(overlay_image)
            self.assertEqual(written_path, str(overlay_path))
            self.assertEqual(overlay_image.GetSize(), reference_image.GetSize())
            self.assertEqual(overlay_image.GetOrigin(), reference_image.GetOrigin())
            for actual, expected in zip(overlay_image.GetSpacing(), reference_image.GetSpacing()):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(overlay_array[0, 1, 2], 1)
            self.assertEqual(overlay_array[1, 2, 3], 1)
            self.assertEqual(overlay_array[2, 3, 4], 0)
            self.assertEqual(set(np.unique(overlay_array).tolist()), {0, 1})

    def test_debug_label_mask_overlay_keeps_each_requested_label_id(self):
        # 这个 overlay 用于查看每个椎体自己的位置区间：目标 label 保留原始
        # 模型 id，非本次目标 label 清零。
        """验证 `test_debug_label_mask_overlay_keeps_each_requested_label_id` 场景符合预期。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((3, 5, 6), dtype=np.uint16)
            seg_array[0, 1, 2] = 20
            seg_array[1, 2, 3] = 21
            seg_array[2, 3, 4] = 99
            reference_image = sitk.GetImageFromArray(seg_array)
            reference_image.SetSpacing((0.5, 0.6, 1.5))
            reference_image.SetOrigin((10.0, 20.0, 30.0))
            label_targets = {
                "vertebrae_C7": LabelTarget("vertebrae_C7", [20]),
                "vertebrae_C6": LabelTarget("vertebrae_C6", [21]),
            }
            overlay_path = temp_path / "标签掩膜.nii.gz"

            written_path = stage._write_debug_label_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_image = sitk.ReadImage(str(overlay_path))
            overlay_array = sitk.GetArrayFromImage(overlay_image)
            self.assertEqual(written_path, str(overlay_path))
            self.assertEqual(overlay_image.GetSize(), reference_image.GetSize())
            self.assertEqual(overlay_image.GetOrigin(), reference_image.GetOrigin())
            for actual, expected in zip(overlay_image.GetSpacing(), reference_image.GetSpacing()):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(overlay_array[0, 1, 2], 20)
            self.assertEqual(overlay_array[1, 2, 3], 21)
            self.assertEqual(overlay_array[2, 3, 4], 0)
            self.assertEqual(set(np.unique(overlay_array).tolist()), {0, 20, 21})

    def test_debug_vertebral_body_mask_overlay_colors_body_voxels_and_excludes_posterior_elements(self):
        """椎体主体校验 overlay 应填充主体颜色，不只画框，也不包含后方窄附件。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((8, 12, 14), dtype=np.uint16)
            # 模拟椎体主体：横状位上前方较宽的矩形区域。
            seg_array[2:6, 2:6, 3:10] = 20
            # 模拟后方椎弓/棘突：同一 label，但 AP 方向更靠后且更窄。
            seg_array[2:6, 6:10, 6:8] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            reference_image.SetSpacing((0.5, 0.6, 1.5))
            reference_image.SetOrigin((10.0, 20.0, 30.0))
            label_targets = {
                "vertebrae_L3": LabelTarget("vertebrae_L3", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            written_path = stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_image = sitk.ReadImage(str(overlay_path))
            overlay_array = sitk.GetArrayFromImage(overlay_image)
            self.assertEqual(written_path, str(overlay_path))
            self.assertEqual(overlay_image.GetSize(), reference_image.GetSize())
            self.assertEqual(overlay_image.GetOrigin(), reference_image.GetOrigin())
            for actual, expected in zip(overlay_image.GetSpacing(), reference_image.GetSpacing()):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(overlay_array[2, 2, 3], 20)
            self.assertEqual(overlay_array[3, 3, 4], 20)
            self.assertEqual(overlay_array[4, 4, 8], 20)
            self.assertEqual(overlay_array[5, 5, 9], 20)
            self.assertEqual(overlay_array[2, 9, 7], 0)
            self.assertEqual(overlay_array[5, 6, 7], 0)

    def test_debug_vertebral_body_mask_overlay_keeps_body_per_axial_slice(self):
        """逐层提取主体，避免胸椎这类整体窄小椎体在全局阈值下大片丢失。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((8, 14, 14), dtype=np.uint16)
            # 上下端层主体较窄，中间层主体较宽；后方附件在 AP 方向更靠后。
            seg_array[2, 3:7, 5:9] = 20
            seg_array[3:5, 3:8, 4:10] = 20
            seg_array[5, 3:7, 5:9] = 20
            seg_array[2:6, 9:12, 6:8] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_T8": LabelTarget("vertebrae_T8", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(overlay_array[2, 4, 6], 20)
            self.assertEqual(overlay_array[5, 4, 7], 20)
            self.assertEqual(overlay_array[3, 10, 7], 0)

    def test_debug_vertebral_body_mask_overlay_prefers_anterior_body_over_wider_posterior_process(self):
        """后方附件比主体更宽时，主体候选仍应取前方第一段宽区域。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((5, 14, 16), dtype=np.uint16)
            seg_array[2, 2:6, 5:11] = 20
            seg_array[2, 7, 7:9] = 20
            seg_array[2, 9:12, 2:15] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_L5": LabelTarget("vertebrae_L5", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(overlay_array[2, 3, 6], 20)
            self.assertEqual(overlay_array[2, 10, 6], 0)

    def test_debug_vertebral_body_mask_overlay_keeps_central_lr_segment_only(self):
        """同一主体 AP 行存在左右侧块时，只保留靠近椎体左右中心的连续段。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((5, 10, 18), dtype=np.uint16)
            seg_array[2, 2:6, 6:12] = 20
            seg_array[2, 3:5, 1:4] = 20
            seg_array[2, 3:5, 14:17] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_C5": LabelTarget("vertebrae_C5", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(overlay_array[2, 3, 8], 20)
            self.assertEqual(overlay_array[2, 3, 2], 0)
            self.assertEqual(overlay_array[2, 3, 15], 0)

    def test_debug_vertebral_body_mask_overlay_excludes_posterior_midline_fragments(self):
        """后方正中小段即使靠近左右中心，也不应跟随前方主体一起上色。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((5, 14, 16), dtype=np.uint16)
            seg_array[2, 2:6, 5:11] = 20
            seg_array[2, 8:11, 7:10] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_T9": LabelTarget("vertebrae_T9", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(overlay_array[2, 3, 8], 20)
            self.assertEqual(overlay_array[2, 9, 8], 0)

    def test_debug_vertebral_body_mask_overlay_stops_when_posterior_row_splits(self):
        """主体后缘分裂成左右附件时，不应因整体跨度仍很宽而继续上色。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((5, 14, 18), dtype=np.uint16)
            seg_array[2, 2:7, 5:13] = 20
            # 后方附件左右分裂，但左右极值仍让整行外接跨度看起来很宽。
            seg_array[2, 7:10, 5:9] = 20
            seg_array[2, 7:10, 11:15] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_T10": LabelTarget("vertebrae_T10", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(overlay_array[2, 5, 7], 20)
            self.assertEqual(overlay_array[2, 8, 7], 0)
            self.assertEqual(overlay_array[2, 8, 13], 0)

    def test_debug_vertebral_body_mask_overlay_skips_axial_slices_with_only_posterior_attachment(self):
        """同一 label 的端层只有后方附件时，不应单独生成一段伪椎体主体。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((8, 14, 14), dtype=np.uint16)
            seg_array[2, 8:11, 6:8] = 20
            seg_array[3:6, 3:7, 5:9] = 20
            seg_array[3:6, 8:11, 6:8] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_T11": LabelTarget("vertebrae_T11", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(overlay_array[3, 4, 6], 20)
            self.assertEqual(overlay_array[2, 9, 7], 0)

    def test_debug_vertebral_body_mask_overlay_does_not_fallback_to_full_label(self):
        """找不到可信主体候选时，应保持为空而不是回退标记整个 label。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((4, 8, 8), dtype=np.uint16)
            seg_array[2, 4, 4] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_T12": LabelTarget("vertebrae_T12", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(int(overlay_array.sum()), 0)

    def test_debug_vertebral_body_mask_overlay_removes_isolated_3d_fragments(self):
        """主体候选中的孤立三维小块不应出现在最终 overlay。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((9, 14, 18), dtype=np.uint16)
            seg_array[2:6, 2:7, 5:12] = 20
            # 与主体没有三维连接，但在自身轴向层上足够宽，模拟外侧残渣。
            seg_array[6, 2:6, 13:17] = 20
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_L4": LabelTarget("vertebrae_L4", [20]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(overlay_array[4, 4, 7], 20)
            self.assertEqual(overlay_array[6, 3, 14], 0)

    def test_debug_vertebral_body_mask_overlay_excludes_c1_but_keeps_c2(self):
        """主体定位 overlay 排除 C1 环形结构，但保留 C2 及以下椎体。"""

        stage = LocalizationStage()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seg_array = np.zeros((6, 12, 18), dtype=np.uint16)
            seg_array[2:5, 2:6, 2:8] = 20
            seg_array[2:5, 2:6, 10:16] = 21
            reference_image = sitk.GetImageFromArray(seg_array)
            label_targets = {
                "vertebrae_C1": LabelTarget("vertebrae_C1", [20]),
                "vertebrae_C2": LabelTarget("vertebrae_C2", [21]),
            }
            overlay_path = temp_path / "椎体主体掩膜.nii.gz"

            stage._write_debug_vertebral_body_mask_overlay(
                reference_image=reference_image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=overlay_path,
            )

            overlay_array = sitk.GetArrayFromImage(sitk.ReadImage(str(overlay_path)))
            self.assertEqual(int((overlay_array == 20).sum()), 0)
            self.assertGreater(int((overlay_array == 21).sum()), 0)

    def test_stage3_spine_geometry_uses_body_centroid_and_excludes_c1(self):
        """阶段三应输出独立主体质心，不能改写全量定位质心，且 C1 不进入 CPR 几何。"""

        stage = LocalizationStage()
        seg_array = np.zeros((8, 14, 18), dtype=np.uint16)
        seg_array[2:6, 2:6, 5:13] = 20  # C1：应保留在全量定位，但不输出主体 CPR 点
        seg_array[2:6, 2:6, 5:13] = 21  # C2 主体
        seg_array[2:6, 6:10, 7:9] = 21  # C2 后方附件
        image = sitk.GetImageFromArray(seg_array)
        image.SetSpacing((0.5, 0.6, 1.5))
        image.SetOrigin((10.0, 20.0, 30.0))
        label_targets = {
            "vertebrae_C1": LabelTarget("vertebrae_C1", [20]),
            "vertebrae_C2": LabelTarget("vertebrae_C2", [21]),
        }

        geometry = stage._build_spine_body_geometry(
            body_part="颈椎",
            image=image,
            seg_array=seg_array,
            label_targets=label_targets,
        )

        self.assertEqual(geometry["method"], "vertebral_body_candidate_v1")
        self.assertNotIn("vertebrae_C1", geometry["body_centroids"])
        c2 = geometry["body_centroids"]["vertebrae_C2"]
        self.assertTrue(c2["found"])
        self.assertGreater(c2["voxel_count"], 0)
        self.assertEqual(set(c2["centroid_voxel"]), {"z", "y", "x"})
        self.assertEqual(set(c2["centroid_physical"]), {"x", "y", "z"})
        self.assertAlmostEqual(c2["centroid_physical"]["x"], 14.25, places=3)

    def test_stage3_spine_geometry_is_not_generated_for_non_spine(self):
        """肺部等非脊柱部位不应生成椎体主体几何。"""

        image = sitk.Image(4, 4, 4, sitk.sitkUInt8)
        geometry = LocalizationStage._build_spine_body_geometry(
            body_part="肺部",
            image=image,
            seg_array=np.zeros((4, 4, 4), dtype=np.uint8),
            label_targets={},
        )

        self.assertIsNone(geometry)

    def test_no_correction_uses_original_raw_and_seg_paths(self):
        # 未超过阈值时不要生成假的 corrected 文件，后续阶段应直接吃原始 raw/seg。
        """验证 `test_no_correction_uses_original_raw_and_seg_paths` 场景符合预期。"""

        stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw_0000.nii.gz"
            seg_path = temp_path / "seg.nii.gz"
            self._write_image(raw_path, np.zeros((2, 8, 8), dtype=np.float32))
            self._write_image(seg_path, np.zeros((2, 8, 8), dtype=np.uint8))

            correction = stage._correct_images_if_needed(
                raw_path=str(raw_path),
                seg_path=str(seg_path),
                output_dir=temp_path,
                skew_info={"angle_degrees": 2.0, "should_correct": False},
            )

            self.assertFalse(correction["correction_applied"])
            self.assertEqual(correction["raw_for_reconstruction_path"], str(raw_path))
            self.assertEqual(correction["seg_for_reconstruction_path"], str(seg_path))
            self.assertFalse((temp_path / "raw_corrected.nii.gz").exists())
            self.assertFalse((temp_path / "seg_corrected.nii.gz").exists())

    def test_correction_step_is_todo_by_default_even_when_skewed(self):
        # 阶段三默认保留“回正”流程位置，但算法还是 TODO，不能自动改写体数据。
        """验证 `test_correction_step_is_todo_by_default_even_when_skewed` 场景符合预期。"""

        stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw_0000.nii.gz"
            seg_path = temp_path / "seg.nii.gz"
            self._write_image(raw_path, np.zeros((2, 16, 16), dtype=np.float32))
            self._write_image(seg_path, np.zeros((2, 16, 16), dtype=np.uint8))

            correction = stage._correct_images_if_needed(
                raw_path=str(raw_path),
                seg_path=str(seg_path),
                output_dir=temp_path,
                skew_info={"angle_degrees": 43.0, "should_correct": True},
            )

            self.assertFalse(correction["correction_applied"])
            self.assertEqual(correction["raw_for_reconstruction_path"], str(raw_path))
            self.assertEqual(correction["seg_for_reconstruction_path"], str(seg_path))
            self.assertFalse((temp_path / "raw_corrected.nii.gz").exists())
            self.assertFalse((temp_path / "seg_corrected.nii.gz").exists())
            self.assertTrue(correction["correction_params"]["correction_requested"])
            self.assertEqual(correction["correction_params"]["correction_status"], "todo")
            self.assertIn(
                "回正逻辑待实现",
                correction["correction_params"]["correction_todo_reason"],
            )

    def test_correction_writes_raw_and_seg_with_nearest_neighbor_labels(self):
        # 低层旋转 helper 保留给后续算法验证；主流程当前不会调用它写 corrected 文件。
        """验证 `test_correction_writes_raw_and_seg_with_nearest_neighbor_labels` 场景符合预期。"""

        stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
        seg_array = np.zeros((3, 32, 32), dtype=np.uint8)
        seg_array[:, 10:22, 12:20] = 7
        seg_image = sitk.GetImageFromArray(seg_array)

        corrected_seg = stage._rotate_image_xy(
            image=seg_image,
            rotation_degrees=18.0,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )

        corrected_labels = set(np.unique(sitk.GetArrayFromImage(corrected_seg)))
        self.assertIn(7, corrected_labels)
        self.assertTrue(corrected_labels.issubset({0, 7}))

    def test_correction_reduces_detected_axial_angle(self):
        # 这是低层旋转 helper 的合成数据测试，不代表真实颈椎回正算法已可用。
        """验证 `test_correction_reduces_detected_axial_angle` 场景符合预期。"""

        stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
        seg_array = np.zeros((1, 80, 80), dtype=np.uint8)
        expected_angle = 30.0
        radians = math.radians(expected_angle)
        for distance in np.linspace(-24, 24, 160):
            x = int(round(40 + distance * math.cos(radians)))
            y = int(round(40 + distance * math.sin(radians)))
            seg_array[0, y - 1:y + 2, x - 1:x + 2] = 7
        seg_image = sitk.GetImageFromArray(seg_array)

        before = stage._detect_axial_skew(seg_array == 7)
        corrected_seg = sitk.GetArrayFromImage(
            stage._rotate_image_xy(
                image=seg_image,
                rotation_degrees=before["angle_degrees"],
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0,
            )
        )
        after = stage._detect_axial_skew(corrected_seg == 7)

        self.assertGreater(before["abs_angle_degrees"], 20.0)
        self.assertLess(after["abs_angle_degrees"], 3.0)

    def test_correction_uses_target_bbox_center_to_avoid_swinging_target(self):
        # 目标不在整张图像中心时，绕整图中心旋转会把目标甩到另一侧；
        # helper 显式开启时应绕目标 bbox 中心旋转，让定位目标的位置尽量稳定。
        """验证 `test_correction_uses_target_bbox_center_to_avoid_swinging_target` 场景符合预期。"""

        stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
        seg_array = np.zeros((1, 100, 100), dtype=np.uint8)
        radians = math.radians(25.0)
        for distance in np.linspace(-20, 20, 140):
            x = int(round(25 + distance * math.cos(radians)))
            y = int(round(30 + distance * math.sin(radians)))
            seg_array[0, y - 1:y + 2, x - 1:x + 2] = 7
        target_mask = seg_array == 7
        before_centroid = np.argwhere(target_mask).mean(axis=0)
        bbox = stage._build_target_bbox(target_mask)
        skew_info = stage._detect_axial_skew(target_mask)
        seg_image = sitk.GetImageFromArray(seg_array)
        center_index_xyz = stage._rotation_center_index_xyz(bbox, seg_image)

        corrected_seg = sitk.GetArrayFromImage(
            stage._rotate_image_xy(
                image=seg_image,
                rotation_degrees=skew_info["angle_degrees"],
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0,
                center_index_xyz=center_index_xyz,
            )
        )

        after_centroid = np.argwhere(corrected_seg == 7).mean(axis=0)
        self.assertLess(abs(after_centroid[1] - before_centroid[1]), 1.5)
        self.assertLess(abs(after_centroid[2] - before_centroid[2]), 1.5)

    def test_reconstruction_returns_empty_result_when_mapping_has_no_plans(self):
        # 第五步只执行第四步 plans；没有计划时不读取旧的 axial_params 占位字段。
        """验证 `test_reconstruction_returns_empty_result_when_mapping_has_no_plans` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            context = ProcessingContext(Request(dicom_dir="/tmp/dicom", output_dir=temp_dir))
            context.set_result("localization", {
                "raw_for_reconstruction_path": "/tmp/new_raw.nii.gz",
                "corrected_raw_path": "/tmp/old_raw.nii.gz",
            })
            context.set_result("mapping", {
                "plans": [],
            })
            stage = ReconstructionStage({"parallel": False})

            stage.execute(context)

            result = context.get_result("reconstruction")
            self.assertEqual(result["plan_count"], 0)
            self.assertEqual(result["slice_count"], 0)

    def test_execute_writes_reconstruction_paths_and_metadata(self):
        # 覆盖完整阶段三主流程：解析 label、定位、判断不回正、写出新旧兼容字段。
        """验证 `test_execute_writes_reconstruction_paths_and_metadata` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "nifti" / "raw_0000.nii.gz"
            seg_path = temp_path / "segmentation" / "seg.nii.gz"
            dataset_dir = temp_path / "nnUNet_results" / "Dataset001_test"
            dataset_json_path = dataset_dir / "dataset.json"
            raw_path.parent.mkdir(parents=True)
            seg_path.parent.mkdir(parents=True)
            dataset_dir.mkdir(parents=True)

            raw_array = np.zeros((2, 40, 40), dtype=np.float32)
            seg_array = np.zeros((2, 40, 40), dtype=np.uint8)
            seg_array[0, 10:15, 18:23] = 5
            seg_array[1, 24:29, 18:23] = 9
            self._write_image(raw_path, raw_array)
            self._write_image(seg_path, seg_array)
            dataset_json_path.write_text(
                json.dumps({
                    "labels": {
                        "background": 0,
                        "vertebrae_L1": 5,
                        "vertebrae_S1": 9,
                    }
                }),
                encoding="utf-8",
            )

            context = ProcessingContext(Request(dicom_dir="/tmp/dicom", output_dir=temp_dir))
            context.set_result("preprocessing", {"nifti_path": str(raw_path)})
            context.set_result("segmentation", {
                "seg_path": str(seg_path),
                "model": {"dataset_id": 1},
            })
            stage = LocalizationStage({"correction_angle_threshold_degrees": 3.0})
            stage._nnunet_paths = {
                "nnUNet_results": str(temp_path / "nnUNet_results"),
                "nnUNet_preprocessed": "",
                "nnUNet_raw": "",
            }
            stage._localization_config = {
                "label_aliases": {
                    "L1": "vertebrae_L1",
                    "S1": "vertebrae_S1",
                },
                "targets": {
                    "腰椎": {
                        "whole_range": {
                            "start": "L1",
                            "end": "S1",
                            "slice_range_extension": {
                                "矢状位": {"start": 2.0},
                            },
                        },
                        "interspace": {
                            "intervals": [{"start": "L1", "end": "S1"}],
                            "slice_range_extension": {
                                "横状位": {"end": 1.0},
                            },
                        },
                    }
                },
            }

            stage.execute(context)

            result = context.get_result("localization")
            self.assertIsInstance(
                context.require_stage_result(
                    "localization", LocalizationStageResult
                ),
                LocalizationStageResult,
            )
            metadata = json.loads(
                (temp_path / "metadata" / "localization_result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["correction_applied"])
            self.assertEqual(result["raw_for_reconstruction_path"], str(raw_path))
            self.assertEqual(result["seg_for_reconstruction_path"], str(seg_path))
            self.assertEqual(result["corrected_raw_path"], str(raw_path))
            self.assertEqual(result["image"]["spacing_xyz"], [1.0, 1.0, 1.0])
            self.assertEqual(result["image"], metadata["image"])
            self.assertIn("cut_configs", result)
            self.assertEqual(
                result["cut_configs"]["whole_range"]
                ["slice_range_extension"]["矢状位"]["start"],
                2.0,
            )
            self.assertEqual(
                metadata["cut_configs"]["interspace"]
                ["slice_range_extension"]["横状位"]["end"],
                1.0,
            )
            self.assertEqual(metadata["raw_for_reconstruction_path"], str(raw_path))
            self.assertEqual(metadata["seg_for_reconstruction_path"], str(seg_path))
            self.assertIn("correction_scope_bbox", metadata)
            overlay_path = temp_path / "metadata" / "定位框.nii.gz"
            target_overlay_path = temp_path / "metadata" / "目标掩膜.nii.gz"
            label_overlay_path = temp_path / "metadata" / "标签掩膜.nii.gz"
            self.assertEqual(result["debug_bbox_overlay_path"], str(overlay_path))
            self.assertEqual(metadata["debug_bbox_overlay_path"], str(overlay_path))
            self.assertEqual(result["debug_target_mask_overlay_path"], str(target_overlay_path))
            self.assertEqual(metadata["debug_target_mask_overlay_path"], str(target_overlay_path))
            self.assertEqual(result["debug_label_mask_overlay_path"], str(label_overlay_path))
            self.assertEqual(metadata["debug_label_mask_overlay_path"], str(label_overlay_path))
            self.assertTrue(overlay_path.exists())
            self.assertTrue(target_overlay_path.exists())
            self.assertTrue(label_overlay_path.exists())
            self.assertEqual(
                result["spine_geometry"]["method"],
                "vertebral_body_candidate_v1",
            )
            self.assertIn(
                "vertebrae_L1",
                result["spine_geometry"]["body_centroids"],
            )
            self.assertIn("spine_geometry", metadata)

    @staticmethod
    def _write_image(path: Path, array: np.ndarray) -> None:
        """为测试准备 `_write_image` 所需的数据或记录行为。"""

        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(path))

    @staticmethod
    def _draw_oriented_label(
        seg_array: np.ndarray,
        label_id: int,
        z: int,
        center_y: int,
        center_x: int,
        angle_degrees: float,
    ) -> None:
        """为测试准备 `_draw_oriented_label` 所需的数据或记录行为。"""

        radians = math.radians(angle_degrees)
        for distance in np.linspace(-18, 18, 90):
            x = int(round(center_x + distance * math.cos(radians)))
            y = int(round(center_y + distance * math.sin(radians)))
            y_min = max(0, y - 1)
            y_max = min(seg_array.shape[1], y + 2)
            x_min = max(0, x - 1)
            x_max = min(seg_array.shape[2], x + 2)
            seg_array[z, y_min:y_max, x_min:x_max] = label_id


if __name__ == "__main__":
    unittest.main()
