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
    def test_target_mask_and_bbox_only_include_requested_labels(self):
        # 阶段三不能拿整套 volume 判断歪斜，必须先合成“本次任务真正关心的目标 mask”。
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
        stage = LocalizationStage()
        target_mask = np.zeros((2, 5, 5), dtype=bool)
        target_mask[0, 2, 2] = True

        skew_info = stage._detect_axial_skew(target_mask)

        self.assertFalse(skew_info["enough_points"])
        self.assertEqual(skew_info["angle_degrees"], 0.0)
        self.assertFalse(skew_info["should_correct"])

    def test_find_dataset_json_prefers_segmentation_sidecar_file(self):
        # nnUNet 模型目录可能有多个 dataset.json；seg 同目录的 sidecar 最贴近当前分割结果。
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

    def test_s1_range_prefers_vertebrae_s1_over_sacrum_when_available(self):
        # S1 这个业务名表示第一骶椎。模型同时提供 vertebrae_S1 和 sacrum 时，
        # 整切范围应该纳入 vertebrae_S1，不能直接跳到整块骶骨。
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

    def test_debug_bbox_overlay_writes_edges_on_reference_grid(self):
        # 研发期用这个 overlay 在 ITK-SNAP 里直接看定位框；它只画边框，不填满盒子。
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

    def test_no_correction_uses_original_raw_and_seg_paths(self):
        # 未超过阈值时不要生成假的 corrected 文件，后续阶段应直接吃原始 raw/seg。
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
                        },
                        "interspace": {
                            "intervals": [{"start": "L1", "end": "S1"}],
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

    @staticmethod
    def _write_image(path: Path, array: np.ndarray) -> None:
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
