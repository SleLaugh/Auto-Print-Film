#!/src/services/stage_localization.py
# -*- coding: utf-8 -*-

"""
阶段三：定位与回正占位
椎体/目标 label 定位、目标 bbox 输出，以及回正 TODO 状态记录
"""

from pathlib import Path
from typing import Dict, Any
import json
import math
import os

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import (
    LabelCoordinate,
    LabelTarget,
    LocalizationResult,
    NiftiImageInfo,
    VoxelBBox,
    VoxelPoint,
)
from src.utils import DebugLog


class LocalizationStage(BaseStage):
    """定位与回正占位阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="localization", config=config)
        # 这些阈值暂时只给实验开关使用。默认主流程不会自动回正，但保留
        # 阈值字段，方便后续真正设计回正算法时继续沿用 metadata 结构。
        self.correction_angle_threshold_degrees = float(
            self.config.get('correction_angle_threshold_degrees', 3.0)
        )
        # PCA 至少需要一定数量的目标体素；太少时方向不稳定，直接跳过回正。
        self.correction_min_points = int(
            self.config.get('correction_min_points', 10)
        )
        # 脊柱整段回正至少需要两个可用椎体；只有一个椎体时无法代表整段姿态。
        self.correction_min_vertebrae = int(
            self.config.get('correction_min_vertebrae', 2)
        )
        # 当前“椎体 PCA -> 全局刚性回正”的角度定义已在真实颈椎 CT 上验证不可靠。
        # 阶段三保留“回正”流程位置，但算法状态是 TODO：默认请求回正、记录
        # 待实现原因，不再自动生成 raw_corrected/seg_corrected，避免正片被误旋转。
        self.correction_requested = bool(
            self.config.get('correction_requested', True)
        )
        self.enable_global_correction = bool(
            self.config.get('enable_experimental_global_correction', False)
        )
        self._load_config()
        self._load_nnunet_config()

    def _load_config(self):
        """
        加载定位/切割范围配置。

        配置只描述“某个部位在某种切割方式下，需要哪些定位范围”。
        层厚、层间距、切几张等采样参数不放在这里，后续由外部请求传入。
        """
        config_paths = [
            Path.cwd() / "config" / "localization_config.json",
            Path.cwd() / "localization_config.json",
            Path(__file__).parent.parent / "config" / "localization_config.json",
        ]

        self._localization_config = {}
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._localization_config = json.load(f)
                DebugLog(f"[定位] 加载配置文件: {config_path}")
                return

        DebugLog("[定位] 未找到 localization_config.json，使用空配置")

    def _load_nnunet_config(self):
        """
        加载 nnUNet 路径配置，用于根据模型 ID 查找 dataset.json。

        定位阶段不重新匹配模型，只复用 segmentation 阶段得到的 model_id。
        这里读取路径配置，是为了把 label 名称转换成真实 label id。
        """
        config_paths = [
            Path.cwd() / "config" / "nnuNet_config.json",
            Path.cwd() / "nnuNet_config.json",
            Path(__file__).parent.parent / "config" / "nnuNet_config.json",
        ]

        self._nnunet_paths = {}
        project_root = Path(__file__).resolve().parents[2]
        for config_path in config_paths:
            if not config_path.exists():
                continue
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._nnunet_paths = {
                'nnUNet_results': self._resolve_config_path(config.get('nnUNet_results', ''), project_root),
                'nnUNet_preprocessed': self._resolve_config_path(config.get('nnUNet_preprocessed', ''), project_root),
                'nnUNet_raw': self._resolve_config_path(config.get('nnUNet_raw', ''), project_root),
            }
            return

    @staticmethod
    def _resolve_config_path(value: str, project_root: Path) -> str:
        if not value:
            return ""
        expanded = os.path.expanduser(os.path.expandvars(str(value)))
        path = Path(expanded)
        if not path.is_absolute():
            path = project_root / path
        return str(path)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        prev_result = context.get_result('segmentation')
        preprocessing_result = context.get_result('preprocessing') or {}
        
        if not prev_result:
            raise ValueError("前置阶段 'segmentation' 结果不存在")
        
        seg_path = prev_result.get('seg_path')
        raw_path = (
            preprocessing_result.get('nifti_path')
            or prev_result.get('input_nifti')
            or str(Path(request.output_dir) / 'nifti' / 'raw_0000.nii.gz')
        )
        model_info = prev_result.get('model') or {}
        if not seg_path:
            raise ValueError("分割结果路径不存在")
        
        DebugLog("=" * 60)
        DebugLog("【阶段三】定位与回正占位 - 目标Label定位")
        DebugLog(f"  输入原图: {raw_path}")
        DebugLog(f"  输入分割: {seg_path}")
        DebugLog(f"  部位: {request.body_part.value}")
        cut_config = self._resolve_cut_config(request)
        resolved_cut_config = self._resolve_label_aliases(cut_config)
        DebugLog(f"  切割方式: {request.cut_mode.value}")
        DebugLog(f"  切割配置: {cut_config}")
        DebugLog(f"  标签配置: {resolved_cut_config}")
        DebugLog("=" * 60)
        
        output_path = Path(request.output_dir) / 'metadata'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # [1] resolve_labels: 获取Label列表
        DebugLog("  [1] resolve_labels: 获取目标Label列表...")
        dataset_json_path = self._find_dataset_json(model_info, seg_path=seg_path)
        label_map = self._load_label_map(dataset_json_path)
        label_targets = self._resolve_label_targets(resolved_cut_config, label_map)
        DebugLog(f"    dataset.json: {dataset_json_path}")
        DebugLog(f"    目标Label: {label_targets}")
        
        # [2] extract_label_coordinates: 提取目标坐标
        DebugLog("  [2] extract_label_coordinates: 提取目标质心+包围盒...")
        image = sitk.ReadImage(str(seg_path))
        seg_array = sitk.GetArrayFromImage(image)
        label_coordinates = self._extract_label_coordinates(seg_array, label_targets)
        self._validate_found_labels(label_coordinates)
        target_mask = self._build_target_mask(seg_array, label_targets)
        correction_scope_bbox = self._build_target_bbox(target_mask)
        DebugLog(f"    已提取 {len(label_coordinates)} 个Label坐标")
        DebugLog(f"    目标整体bbox: {correction_scope_bbox.to_dict()}")
        
        # [3] detect_skew: 当前只保留定位，回正算法先作为 TODO 占位。
        # 旧算法会把椎体形态/颈椎走形误当成病人横断面旋转角，真实正片会被
        # 硬转歪。这里不再给主流程输出可执行角度，只记录禁用原因。
        if self.enable_global_correction:
            DebugLog("  [3] detect_skew: 基于目标范围检测横状位歪斜...")
            skew_info = self._detect_target_axial_skew(seg_array, label_targets)
            DebugLog(f"    横状位偏角: {skew_info['angle_degrees']:.3f}°")
            DebugLog(f"    是否需要回正: {skew_info['should_correct']}")
        else:
            DebugLog("  [3] detect_skew: 回正逻辑 TODO，阶段三只输出定位结果")
            skew_info = self._todo_correction_info()
        
        # [4] correct_skew: 回正变换
        nifti_path = Path(request.output_dir) / 'nifti'
        DebugLog("  [4] correct_skew: 回正逻辑 TODO，占位返回原始 raw/seg...")
        correction_result = self._correct_images_if_needed(
            raw_path=str(raw_path),
            seg_path=str(seg_path),
            output_dir=nifti_path,
            skew_info=skew_info,
            correction_center_bbox=correction_scope_bbox,
        )

        if correction_result['correction_applied']:
            DebugLog("    已写出回正图像，重新从 seg_corrected.nii.gz 提取坐标")
            image = sitk.ReadImage(correction_result['seg_for_reconstruction_path'])
            seg_array = sitk.GetArrayFromImage(image)
            label_coordinates = self._extract_label_coordinates(seg_array, label_targets)
            self._validate_found_labels(label_coordinates)
        else:
            DebugLog("    未执行回正，后续阶段继续使用原始 raw/seg")

        # 研发期调试输出：给 ITK-SNAP 看的 bbox 边框 overlay。
        # 这不是业务结果，不参与后续阶段；后续稳定后删除这一处调用和
        # _write_debug_bbox_overlay 方法即可。
        debug_bbox_overlay_path = output_path / 'localization_bbox_overlay.nii.gz'
        self._write_debug_bbox_overlay(
            reference_image=image,
            label_coordinates=label_coordinates,
            correction_scope_bbox=correction_scope_bbox,
            output_path=debug_bbox_overlay_path,
        )
        DebugLog(f"    调试bbox overlay: {debug_bbox_overlay_path}")
        debug_target_mask_overlay_path = output_path / 'localization_target_mask_overlay.nii.gz'
        self._write_debug_target_mask_overlay(
            reference_image=image,
            seg_array=seg_array,
            label_targets=label_targets,
            output_path=debug_target_mask_overlay_path,
        )
        DebugLog(f"    调试目标mask overlay: {debug_target_mask_overlay_path}")
        debug_label_mask_overlay_path = output_path / 'localization_label_mask_overlay.nii.gz'
        self._write_debug_label_mask_overlay(
            reference_image=image,
            seg_array=seg_array,
            label_targets=label_targets,
            output_path=debug_label_mask_overlay_path,
        )
        DebugLog(f"    调试分label mask overlay: {debug_label_mask_overlay_path}")

        localization_result_path = output_path / 'localization_result.json'
        image_info = NiftiImageInfo(
            size_xyz=list(image.GetSize()),
            spacing_xyz=list(image.GetSpacing()),
            origin_xyz=list(image.GetOrigin()),
            direction=list(image.GetDirection()),
        ).to_dict()
        result = {
            'coords': self._coordinates_to_dict(label_coordinates),
            'cut_config': cut_config,
            'resolved_cut_config': resolved_cut_config,
            'label_targets': self._label_targets_to_dict(label_targets),
            'dataset_json_path': str(dataset_json_path),
            'localization_result_path': str(localization_result_path),
            # 阶段四需要图像空间信息把第三步的 z/y/x 体素坐标换算成 x/y/z
            # 物理坐标；这里只写 JSON 文件不够，必须同步放入 context 结果。
            'image': image_info,
            'raw_for_reconstruction_path': correction_result['raw_for_reconstruction_path'],
            'seg_for_reconstruction_path': correction_result['seg_for_reconstruction_path'],
            'correction_applied': correction_result['correction_applied'],
            'correction_angle_degrees': skew_info['angle_degrees'],
            'correction_scope_bbox': correction_scope_bbox.to_dict(),
            'correction_params': correction_result['correction_params'],
            'corrected_raw_path': correction_result['corrected_raw_path'],
            'corrected_seg_path': correction_result['corrected_seg_path'],
            'is_skewed': correction_result['correction_applied'],
            'skew_params': correction_result['correction_params'],
            'debug_bbox_overlay_path': str(debug_bbox_overlay_path),
            'debug_target_mask_overlay_path': str(debug_target_mask_overlay_path),
            'debug_label_mask_overlay_path': str(debug_label_mask_overlay_path),
        }

        self._write_localization_result(
            localization_result_path,
            request=request,
            model_info=model_info,
            cut_config=cut_config,
            resolved_cut_config=resolved_cut_config,
            label_targets=label_targets,
            label_coordinates=label_coordinates,
            dataset_json_path=dataset_json_path,
            image=image,
            seg_path=result['seg_for_reconstruction_path'],
            raw_for_reconstruction_path=result['raw_for_reconstruction_path'],
            seg_for_reconstruction_path=result['seg_for_reconstruction_path'],
            correction_applied=result['correction_applied'],
            correction_angle_degrees=result['correction_angle_degrees'],
            correction_scope_bbox=correction_scope_bbox,
            correction_params=result['correction_params'],
            corrected_raw_path=result['corrected_raw_path'],
            corrected_seg_path=result['corrected_seg_path'],
            debug_bbox_overlay_path=result['debug_bbox_overlay_path'],
            debug_target_mask_overlay_path=result['debug_target_mask_overlay_path'],
            debug_label_mask_overlay_path=result['debug_label_mask_overlay_path'],
            is_skewed=result['is_skewed'],
            skew_params=result['skew_params'],
        )
        
        context.set_result(self.name, result)
        
        DebugLog("  ✅ 阶段三完成")
        DebugLog("=" * 60)

        return context

    def _find_dataset_json(self, model_info: Dict[str, Any], seg_path: str = None) -> Path:
        """
        根据 segmentation 阶段匹配到的模型 ID 查找对应 dataset.json。

        优先使用 seg.nii.gz 同目录的 sidecar dataset.json，因为它和
        当前分割输出一一对应。真实运行中模型目录可能存在多个相同 ID 的
        dataset.json，sidecar 文件能避免定位阶段读到不匹配的 label 字典。

        如果 sidecar 不存在，再按 nnUNet 目录查找。nnUNet 目录常见形态
        是 Dataset292_xxx 或 Dataset315_xxx。优先从 nnUNet_results 查找，
        因为它最贴近实际用于推理的模型；找不到再查 preprocessed/raw。
        """
        if seg_path:
            sidecar_dataset_json = Path(seg_path).parent / "dataset.json"
            if sidecar_dataset_json.exists():
                return sidecar_dataset_json

        dataset_id = model_info.get('dataset_id') or model_info.get('model_id')
        if dataset_id is None:
            raise ValueError("分割结果缺少 model_id/dataset_id，无法查找 dataset.json")

        dataset_prefix = f"Dataset{int(dataset_id):03d}"
        roots = [
            self._nnunet_paths.get('nnUNet_results', ''),
            self._nnunet_paths.get('nnUNet_preprocessed', ''),
            self._nnunet_paths.get('nnUNet_raw', ''),
        ]

        for root_value in roots:
            if not root_value:
                continue
            root = Path(root_value)
            if not root.exists():
                continue
            matches = [
                path for path in root.rglob("dataset.json")
                if any(part.startswith(dataset_prefix) for part in path.parts)
            ]
            if matches:
                return self._choose_dataset_json(matches)

        raise FileNotFoundError(f"未找到模型 {dataset_prefix} 对应的 dataset.json")

    @staticmethod
    def _choose_dataset_json(paths: list[Path]) -> Path:
        """
        从候选 dataset.json 中选择一个更像模型根目录的文件。

        结果目录下可能有 crossval 或子任务目录。路径越短，通常越接近
        当前模型配置根目录；同长度时按路径名排序，保证结果稳定。
        """
        return sorted(paths, key=lambda p: (len(p.parts), str(p)))[0]

    @staticmethod
    def _load_label_map(dataset_json_path: Path) -> Dict[str, list[int]]:
        """
        读取 dataset.json 中的 labels，并统一成 label_name -> [label_id]。

        目前主要处理普通整数标签；如果后续遇到 nnUNet region label
        的列表写法，这里也会保留成多个 id。
        """
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        labels = dataset.get('labels') or {}
        if not labels:
            raise ValueError(f"dataset.json 中缺少 labels: {dataset_json_path}")

        label_map: Dict[str, list[int]] = {}
        for label_name, label_value in labels.items():
            if isinstance(label_value, list):
                label_map[label_name] = [int(value) for value in label_value]
            else:
                label_map[label_name] = [int(label_value)]
        return label_map

    def _resolve_label_targets(
        self,
        resolved_cut_config: Dict[str, Any],
        label_map: Dict[str, list[int]]
    ) -> Dict[str, LabelTarget]:
        """
        把定位配置中用到的 label 名称解析为 id，并去重。

        返回结构按 label 名称组织，后续提坐标时只需要遍历一次，避免
        椎间切中 L2、L3 等标签重复计算。
        """
        label_names = self._collect_label_names(resolved_cut_config, label_map)
        missing_names = [name for name in label_names if name not in label_map]
        if missing_names:
            available = ", ".join(label_map.keys())
            missing = ", ".join(missing_names)
            raise ValueError(f"dataset.json 中找不到定位标签: {missing}; 可用标签: {available}")

        targets = {}
        for label_name in label_names:
            targets[label_name] = LabelTarget(
                label_name=label_name,
                label_ids=label_map[label_name]
            )
        return targets

    def _collect_label_names(
        self,
        resolved_cut_config: Dict[str, Any],
        label_map: Dict[str, list[int]] = None
    ) -> list[str]:
        """
        收集本次定位需要的 label 名称。

        普通目标仍然收集 start/end。脊柱整切的 start/end 是范围边界，
        不是只取两个端点；比如 C1 到 C7 会展开为 C1-C7 全部椎体。
        """
        label_names = []
        if 'intervals' in resolved_cut_config:
            for interval in resolved_cut_config['intervals']:
                label_names.extend(self._expand_label_pair(
                    interval['start_labels'],
                    interval['end_labels'],
                    label_map,
                ))
        else:
            label_names.extend(self._expand_label_pair(
                resolved_cut_config['start_labels'],
                resolved_cut_config['end_labels'],
                label_map,
            ))

        seen = set()
        unique_names = []
        for label_name in label_names:
            if label_name in seen:
                continue
            seen.add(label_name)
            unique_names.append(label_name)
        return unique_names

    def _expand_label_pair(
        self,
        start_labels: list[str],
        end_labels: list[str],
        label_map: Dict[str, list[int]] = None,
    ) -> list[str]:
        """
        展开一组 start/end label。

        如果 start/end 都是脊柱 label，就按解剖顺序展开中间椎体；否则
        保持原来的端点语义，适配 skull、lung 这类整体目标。
        """
        expanded_names = []
        for start_label in start_labels:
            for end_label in end_labels:
                expanded_range = self._expand_vertebrae_range(start_label, end_label, label_map)
                if expanded_range:
                    expanded_names.extend(expanded_range)
                else:
                    expanded_names.extend([start_label, end_label])
        return expanded_names

    @classmethod
    def _expand_vertebrae_range(
        cls,
        start_label: str,
        end_label: str,
        label_map: Dict[str, list[int]] = None,
    ) -> list[str]:
        """
        按 C/T/L/S 的解剖顺序展开脊柱范围。

        label_map 传入时，只返回当前模型认识的 label，避免把不存在的中间
        label 强行加入定位目标。S1 在不同模型里可能叫 vertebrae_S1 或
        sacrum，这里按实际可用 label 选择。
        """
        start_index = cls._vertebra_order_index(start_label)
        end_index = cls._vertebra_order_index(end_label)
        if start_index is None or end_index is None:
            return []

        order = cls._vertebra_order_labels()
        if start_index <= end_index:
            vertebra_keys = order[start_index:end_index + 1]
        else:
            vertebra_keys = list(reversed(order[end_index:start_index + 1]))

        expanded_names = []
        for vertebra_key in vertebra_keys:
            label_name = cls._vertebra_key_to_available_label(vertebra_key, label_map)
            if label_name:
                expanded_names.append(label_name)
        return expanded_names

    @staticmethod
    def _vertebra_order_labels() -> list[str]:
        """
        统一的脊柱顺序，方向为头侧到足侧。

        这里保存业务短名，真正的 dataset.json label 名在
        _vertebra_key_to_available_label 中转换。
        """
        return (
            [f"C{index}" for index in range(1, 8)] +
            [f"T{index}" for index in range(1, 13)] +
            [f"L{index}" for index in range(1, 6)] +
            ["S1"]
        )

    @classmethod
    def _vertebra_order_index(cls, label_name: str) -> int | None:
        vertebra_key = cls._vertebra_key(label_name)
        if not vertebra_key:
            return None
        order = cls._vertebra_order_labels()
        if vertebra_key not in order:
            return None
        return order.index(vertebra_key)

    @staticmethod
    def _vertebra_key(label_name: str) -> str | None:
        """
        把 dataset label 或业务短名转换成 C1/T12/L5/S1 这种统一 key。
        """
        name = str(label_name)
        if name == "sacrum":
            return "S1"
        if name.startswith("vertebrae_"):
            name = name[len("vertebrae_"):]
        if len(name) < 2:
            return None
        prefix = name[0]
        number = name[1:]
        if prefix in {"C", "T", "L", "S"} and number.isdigit():
            return f"{prefix}{int(number)}"
        return None

    @staticmethod
    def _vertebra_key_to_available_label(
        vertebra_key: str,
        label_map: Dict[str, list[int]] = None,
    ) -> str | None:
        candidates = (
            ["vertebrae_S1", "sacrum"]
            if vertebra_key == "S1"
            else [f"vertebrae_{vertebra_key}"]
        )
        if label_map is None:
            return candidates[0]
        for candidate in candidates:
            if candidate in label_map:
                return candidate
        return None

    def _extract_label_coordinates(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget]
    ) -> Dict[str, LabelCoordinate]:
        """
        从分割体素中提取每个目标 label 的 bbox 和 centroid。

        SimpleITK 转 numpy 后数组顺序是 (z, y, x)。这里先输出体素索引
        坐标；物理坐标 mm 需要后续结合 affine/origin/spacing/direction。
        """
        coordinates = {}
        for label_name, target in label_targets.items():
            mask = np.isin(seg_array, target.label_ids)
            indices = np.argwhere(mask)
            if indices.size == 0:
                coordinates[label_name] = LabelCoordinate(
                    label_name=label_name,
                    label_ids=target.label_ids,
                    voxel_count=0,
                    found=False,
                )
                continue

            z_min, y_min, x_min = indices.min(axis=0)
            z_max, y_max, x_max = indices.max(axis=0)
            centroid_z, centroid_y, centroid_x = indices.mean(axis=0)
            coordinates[label_name] = LabelCoordinate(
                label_name=label_name,
                label_ids=target.label_ids,
                voxel_count=int(indices.shape[0]),
                found=True,
                centroid=VoxelPoint(
                    z=float(centroid_z),
                    y=float(centroid_y),
                    x=float(centroid_x),
                ),
                bbox=VoxelBBox(
                    z_min=int(z_min),
                    z_max=int(z_max),
                    y_min=int(y_min),
                    y_max=int(y_max),
                    x_min=int(x_min),
                    x_max=int(x_max),
                )
            )
        return coordinates

    @staticmethod
    def _build_target_mask(
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget]
    ) -> np.ndarray:
        """
        把本次任务关心的所有 label 合成一个整体 mask。

        横状位回正不能用整套 CT 判断，因为同一段片子里可能头歪、胸椎正，
        也可能反过来。这里先按 body_part + cut_mode 解析出的 label 范围
        缩小目标，再只看这个目标范围的姿态。
        """
        target_label_ids = []
        for target in label_targets.values():
            target_label_ids.extend(target.label_ids)
        if not target_label_ids:
            return np.zeros(seg_array.shape, dtype=bool)
        return np.isin(seg_array, target_label_ids)

    @staticmethod
    def _build_target_bbox(target_mask: np.ndarray) -> VoxelBBox:
        """
        计算目标整体 mask 的 3D bbox。

        这个 bbox 记录的是“本次判断回正的空间范围”，后续排查为什么回正或
        为什么没回正时，比只看单个 label 的 bbox 更直观。
        """
        indices = np.argwhere(target_mask)
        if indices.size == 0:
            raise ValueError("目标 mask 为空，无法计算定位范围 bbox")

        z_min, y_min, x_min = indices.min(axis=0)
        z_max, y_max, x_max = indices.max(axis=0)
        return VoxelBBox(
            z_min=int(z_min),
            z_max=int(z_max),
            y_min=int(y_min),
            y_max=int(y_max),
            x_min=int(x_min),
            x_max=int(x_max),
        )

    def _detect_axial_skew(self, target_mask: np.ndarray) -> Dict[str, Any]:
        """
        用目标 mask 的 XY 投影估计横状位旋转偏角。

        SimpleITK 转出的数组是 (z, y, x)。横状位是否歪，第一版只看
        XY 平面中的方向：先把目标体素投影到 XY，再用 PCA 找主方向。
        由于目标天然可能沿 X 轴或 Y 轴更长，最终角度取“相对最近标准轴
        0/90 度的偏离”，避免把一个正着的竖向结构误判为 90 度歪斜。
        """
        indices = np.argwhere(target_mask)
        point_count = int(indices.shape[0])
        base_info = {
            'method': 'pca_xy_projection',
            'point_count': point_count,
            'min_points': self.correction_min_points,
            'threshold_degrees': self.correction_angle_threshold_degrees,
            'raw_pca_angle_degrees': 0.0,
            'angle_degrees': 0.0,
            'abs_angle_degrees': 0.0,
            'enough_points': False,
            'should_correct': False,
        }
        if point_count < self.correction_min_points:
            base_info['reason'] = '目标体素太少，PCA 方向不稳定'
            return base_info

        xy_points = indices[:, [2, 1]].astype(np.float64)
        centered_points = xy_points - xy_points.mean(axis=0)
        covariance = np.cov(centered_points, rowvar=False)
        if not np.all(np.isfinite(covariance)):
            base_info['reason'] = '目标点协方差异常'
            return base_info

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        max_index = int(np.argmax(eigenvalues))
        if float(eigenvalues[max_index]) <= 0:
            base_info['reason'] = '目标点没有可用主方向'
            return base_info

        principal_vector = eigenvectors[:, max_index]
        raw_angle = math.degrees(math.atan2(principal_vector[1], principal_vector[0]))
        normalized_raw_angle = self._normalize_pca_angle(raw_angle)
        deviation_angle = self._deviation_from_nearest_axial_axis(normalized_raw_angle)
        abs_angle = abs(deviation_angle)

        return {
            **base_info,
            'raw_pca_angle_degrees': float(normalized_raw_angle),
            'angle_degrees': float(deviation_angle),
            'abs_angle_degrees': float(abs_angle),
            'enough_points': True,
            'should_correct': abs_angle > self.correction_angle_threshold_degrees,
        }

    def _detect_target_axial_skew(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
    ) -> Dict[str, Any]:
        """
        根据目标类型选择横状位角度检测方法。

        脊柱类目标优先用单个椎体的横状位主轴做稳健聚合；头骨、肺部等
        非脊柱目标没有天然的椎体顺序，继续使用整体 mask PCA。

        注意：这个方法目前只允许在 enable_experimental_global_correction
        显式打开时进入主流程。默认流程不会使用这里的角度执行回正。
        """
        if self._is_vertebrae_targets(label_targets):
            vertebrae_info = self._detect_vertebrae_axial_skew(seg_array, label_targets)
            if vertebrae_info.get('enough_points'):
                return vertebrae_info

            fallback_mask = self._build_target_mask(seg_array, label_targets)
            fallback_info = self._detect_axial_skew(fallback_mask)
            fallback_info.update({
                'method': 'pca_xy_projection_fallback',
                'fallback_reason': vertebrae_info.get('reason', '脊柱椎体级角度不可用'),
                'vertebrae_skew_params': vertebrae_info,
            })
            return fallback_info

        target_mask = self._build_target_mask(seg_array, label_targets)
        return self._detect_axial_skew(target_mask)

    def _todo_correction_info(self) -> Dict[str, Any]:
        """
        构造“回正逻辑待实现”的诊断信息。

        这里故意不计算候选角度，避免把旧 PCA 角度继续写入结果后被误解为
        可执行的回正角。阶段三当前只承诺输出定位坐标和目标 bbox。
        """
        return {
            'method': 'global_correction_todo',
            'point_count': 0,
            'min_points': self.correction_min_points,
            'min_vertebrae': self.correction_min_vertebrae,
            'threshold_degrees': self.correction_angle_threshold_degrees,
            'raw_pca_angle_degrees': 0.0,
            'angle_degrees': 0.0,
            'abs_angle_degrees': 0.0,
            'enough_points': False,
            'should_correct': False,
            'correction_requested': self.correction_requested,
            'correction_status': 'todo',
            'global_correction_enabled': self.enable_global_correction,
            'correction_todo_reason': (
                '回正逻辑待实现，阶段三当前只保留定位坐标和目标 bbox'
            ),
        }

    @staticmethod
    def _is_vertebrae_targets(label_targets: Dict[str, LabelTarget]) -> bool:
        """
        判断当前目标是否属于脊柱类。

        sacrum 在模型里有时不叫 vertebrae_S1，但它仍然属于腰骶整段定位。
        """
        if not label_targets:
            return False
        for label_name in label_targets:
            if label_name == "sacrum":
                continue
            if not label_name.startswith("vertebrae_"):
                return False
        return True

    def _detect_vertebrae_axial_skew(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
    ) -> Dict[str, Any]:
        """
        对脊柱目标按单椎体角度聚合，避免整块体素 PCA 被椎体形状带偏。

        真实数据里首尾可能被扫描边界切掉。贴边 label 的形状不完整，
        这里直接跳过，避免残缺椎体主轴影响整段回正角度。
        """
        used_label_names = []
        skipped_label_names = []
        label_angles = []
        total_point_count = 0

        for label_name, target in label_targets.items():
            mask = np.isin(seg_array, target.label_ids)
            indices = np.argwhere(mask)
            point_count = int(indices.shape[0])
            if point_count < self.correction_min_points:
                skipped_label_names.append(label_name)
                continue

            bbox = self._build_target_bbox(mask)
            if self._bbox_touches_array_boundary(bbox, seg_array.shape):
                skipped_label_names.append(label_name)
                continue

            label_info = self._detect_axial_skew(mask)
            if not label_info.get('enough_points'):
                skipped_label_names.append(label_name)
                continue

            used_label_names.append(label_name)
            total_point_count += point_count
            label_angles.append({
                'label_name': label_name,
                'point_count': point_count,
                'raw_pca_angle_degrees': label_info['raw_pca_angle_degrees'],
                'angle_degrees': label_info['angle_degrees'],
                'bbox': bbox.to_dict(),
            })

        base_info = {
            'method': 'vertebrae_label_angle_aggregation',
            'point_count': total_point_count,
            'min_points': self.correction_min_points,
            'min_vertebrae': self.correction_min_vertebrae,
            'threshold_degrees': self.correction_angle_threshold_degrees,
            'raw_pca_angle_degrees': 0.0,
            'angle_degrees': 0.0,
            'abs_angle_degrees': 0.0,
            'enough_points': False,
            'should_correct': False,
            'used_label_names': used_label_names,
            'skipped_label_names': skipped_label_names,
            'label_angles': label_angles,
        }
        if len(label_angles) < self.correction_min_vertebrae:
            base_info['reason'] = '可用于聚合的完整椎体数量不足'
            return base_info

        mean_raw_angle = self._weighted_axis_angle_mean([
            (item['raw_pca_angle_degrees'], item['point_count'])
            for item in label_angles
        ])
        deviation_angle = self._deviation_from_nearest_axial_axis(mean_raw_angle)
        abs_angle = abs(deviation_angle)
        return {
            **base_info,
            'raw_pca_angle_degrees': float(mean_raw_angle),
            'angle_degrees': float(deviation_angle),
            'abs_angle_degrees': float(abs_angle),
            'enough_points': True,
            'should_correct': abs_angle > self.correction_angle_threshold_degrees,
        }

    @staticmethod
    def _bbox_touches_array_boundary(bbox: VoxelBBox, array_shape_zyx: tuple[int, ...]) -> bool:
        """
        判断目标 bbox 是否贴到数组边界。

        贴边通常表示该 label 被扫描范围截断，横状位主轴不再可靠。
        """
        z_size, y_size, x_size = array_shape_zyx
        return (
            bbox.z_min <= 0 or bbox.y_min <= 0 or bbox.x_min <= 0 or
            bbox.z_max >= z_size - 1 or bbox.y_max >= y_size - 1 or bbox.x_max >= x_size - 1
        )

    @classmethod
    def _weighted_axis_angle_mean(cls, angle_weight_pairs: list[tuple[float, int]]) -> float:
        """
        计算轴方向的加权平均角。

        PCA 主轴没有箭头，0 度和 180 度等价，所以用双角度圆统计求平均，
        再除以 2 回到 [-90, 90)。
        """
        sin_sum = 0.0
        cos_sum = 0.0
        for angle_degrees, weight in angle_weight_pairs:
            radians = math.radians(float(angle_degrees) * 2.0)
            sin_sum += float(weight) * math.sin(radians)
            cos_sum += float(weight) * math.cos(radians)

        if sin_sum == 0.0 and cos_sum == 0.0:
            return 0.0
        mean_angle = math.degrees(math.atan2(sin_sum, cos_sum)) / 2.0
        return cls._normalize_pca_angle(mean_angle)

    @staticmethod
    def _normalize_pca_angle(angle_degrees: float) -> float:
        """
        把 PCA 主方向规整到 [-90, 90)。

        PCA 的特征向量没有箭头方向，同一条轴既可以表示成 30 度，也可以
        表示成 210 度；规整后才能稳定比较。
        """
        angle = float(angle_degrees)
        while angle < -90.0:
            angle += 180.0
        while angle >= 90.0:
            angle -= 180.0
        return angle

    @staticmethod
    def _deviation_from_nearest_axial_axis(angle_degrees: float) -> float:
        """
        计算主方向偏离最近标准轴的角度。

        标准轴是 0 度和 90 度。比如 82 度其实更接近“竖着但偏了 -8 度”，
        不应按 82 度直接回正。
        """
        if angle_degrees >= 45.0:
            return angle_degrees - 90.0
        if angle_degrees < -45.0:
            return angle_degrees + 90.0
        return angle_degrees

    def _correct_images_if_needed(
        self,
        raw_path: str,
        seg_path: str,
        output_dir: Path,
        skew_info: Dict[str, Any],
        correction_center_bbox: VoxelBBox = None,
    ) -> Dict[str, Any]:
        """
        根据回正状态决定后续阶段该读取哪组 raw/seg。

        默认情况下 enable_experimental_global_correction=False，因此这里只
        返回原始路径，并把 correction_status 写成 todo。下面真正写
        raw_corrected/seg_corrected 的分支只留给后续算法实验，不能作为当前
        业务默认回正逻辑理解。
        """
        output_dir = Path(output_dir)
        raw_path = str(raw_path)
        seg_path = str(seg_path)
        should_correct = bool(skew_info.get('should_correct'))
        angle_degrees = float(skew_info.get('angle_degrees', 0.0))
        correction_params = {
            **skew_info,
            'applied_rotation_degrees': 0.0,
            'raw_interpolator': 'none',
            'seg_interpolator': 'none',
            'rotation_center_source': 'none',
            'rotation_center_index_xyz': None,
            'correction_requested': self.correction_requested,
            'correction_status': 'pending',
            'correction_todo_reason': '',
            'global_correction_enabled': self.enable_global_correction,
        }

        if not self.enable_global_correction:
            correction_params['correction_status'] = 'todo'
            correction_params['correction_todo_reason'] = (
                '回正逻辑待实现，阶段三当前只输出定位坐标，不改写 raw/seg'
            )
            return {
                'raw_for_reconstruction_path': raw_path,
                'seg_for_reconstruction_path': seg_path,
                'corrected_raw_path': raw_path,
                'corrected_seg_path': seg_path,
                'correction_applied': False,
                'correction_params': correction_params,
            }

        if not should_correct:
            correction_params['correction_status'] = 'not_needed'
            correction_params['correction_todo_reason'] = ''
            correction_params['correction_disabled_reason'] = '未超过回正阈值'
            return {
                'raw_for_reconstruction_path': raw_path,
                'seg_for_reconstruction_path': seg_path,
                'corrected_raw_path': raw_path,
                'corrected_seg_path': seg_path,
                'correction_applied': False,
                'correction_params': correction_params,
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        corrected_raw_path = output_dir / 'raw_corrected.nii.gz'
        corrected_seg_path = output_dir / 'seg_corrected.nii.gz'

        raw_image = sitk.ReadImage(raw_path)
        seg_image = sitk.ReadImage(seg_path)
        # SimpleITK Resample 的 transform 是“输出坐标 -> 输入坐标”。
        # 因此这里传入检测角度本身，图像内容会朝相反方向被采样回正。
        applied_rotation_degrees = angle_degrees
        rotation_center_index_xyz = self._rotation_center_index_xyz(
            correction_center_bbox,
            raw_image,
        )

        corrected_raw = self._rotate_image_xy(
            image=raw_image,
            rotation_degrees=applied_rotation_degrees,
            interpolator=sitk.sitkLinear,
            default_value=self._image_min_value(raw_image),
            center_index_xyz=rotation_center_index_xyz,
        )
        corrected_seg = self._rotate_image_xy(
            image=seg_image,
            rotation_degrees=applied_rotation_degrees,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
            center_index_xyz=rotation_center_index_xyz,
        )
        corrected_seg = sitk.Cast(corrected_seg, seg_image.GetPixelID())

        sitk.WriteImage(corrected_raw, str(corrected_raw_path))
        sitk.WriteImage(corrected_seg, str(corrected_seg_path))

        correction_params.update({
            'correction_status': 'applied',
            'applied_rotation_degrees': applied_rotation_degrees,
            'raw_interpolator': 'linear',
            'seg_interpolator': 'nearest_neighbor',
            'raw_default_value': self._image_min_value(raw_image),
            'seg_default_value': 0,
            'rotation_center_source': 'target_bbox' if correction_center_bbox else 'image_center',
            'rotation_center_index_xyz': rotation_center_index_xyz,
        })
        return {
            'raw_for_reconstruction_path': str(corrected_raw_path),
            'seg_for_reconstruction_path': str(corrected_seg_path),
            'corrected_raw_path': str(corrected_raw_path),
            'corrected_seg_path': str(corrected_seg_path),
            'correction_applied': True,
            'correction_params': correction_params,
        }

    @staticmethod
    def _rotate_image_xy(
        image: sitk.Image,
        rotation_degrees: float,
        interpolator: int,
        default_value: float,
        center_index_xyz: list[float] = None,
    ) -> sitk.Image:
        """
        在当前图像网格内围绕 Z 轴旋转，也就是只修正横状位 XY 平面。

        SimpleITK 的 Resample 使用“输出点映射回输入点”的 transform。
        这里保持尺寸、spacing、origin、direction 不变，只改变采样来源，
        保证 raw 和 seg 仍能与后续 MPR 使用同一套空间信息。
        """
        if image.GetDimension() != 3:
            raise ValueError(f"横状位回正只支持 3D 图像，当前维度: {image.GetDimension()}")

        size = image.GetSize()
        center_index = center_index_xyz or [
            (size[0] - 1) / 2.0,
            (size[1] - 1) / 2.0,
            (size[2] - 1) / 2.0,
        ]
        center_point = image.TransformContinuousIndexToPhysicalPoint(center_index)

        transform = sitk.Euler3DTransform()
        transform.SetCenter(center_point)
        transform.SetRotation(0.0, 0.0, math.radians(rotation_degrees))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(transform)
        return resampler.Execute(image)

    @staticmethod
    def _rotation_center_index_xyz(
        correction_center_bbox: VoxelBBox,
        image: sitk.Image,
    ) -> list[float]:
        """
        计算回正旋转中心的连续体素索引，顺序为 SimpleITK 使用的 x/y/z。

        优先使用本次定位目标的 bbox 中心；没有 bbox 时退回整图中心，兼容
        单元测试或外部直接调用 helper 的旧方式。
        """
        if correction_center_bbox:
            return [
                (correction_center_bbox.x_min + correction_center_bbox.x_max) / 2.0,
                (correction_center_bbox.y_min + correction_center_bbox.y_max) / 2.0,
                (correction_center_bbox.z_min + correction_center_bbox.z_max) / 2.0,
            ]

        size = image.GetSize()
        return [
            (size[0] - 1) / 2.0,
            (size[1] - 1) / 2.0,
            (size[2] - 1) / 2.0,
        ]

    @staticmethod
    def _image_min_value(image: sitk.Image) -> float:
        """
        读取图像最小值作为 raw 回正的边界填充值。

        CT 背景常常不是 0，用最小值通常比硬编码 0 更少引入亮边。
        """
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        return float(stats.GetMinimum())

    @staticmethod
    def _validate_found_labels(label_coordinates: Dict[str, LabelCoordinate]) -> None:
        """
        确认请求定位所需 label 在 seg.nii.gz 中真的存在。

        dataset.json 只说明模型认识哪些标签；实际推理结果里可能没有分割
        出来。缺少目标体素时继续往后切图没有意义，所以这里直接失败。
        """
        missing_names = [
            label_name
            for label_name, coord in label_coordinates.items()
            if not coord.found
        ]
        if missing_names:
            raise ValueError(f"seg.nii.gz 中未找到目标Label体素: {', '.join(missing_names)}")

    @staticmethod
    def _write_debug_bbox_overlay(
        reference_image: sitk.Image,
        label_coordinates: Dict[str, LabelCoordinate],
        correction_scope_bbox: VoxelBBox,
        output_path: Path,
        include_label_bboxes: bool = False,
    ) -> str:
        """
        写出研发期 bbox 可视化 overlay，方便在 ITK-SNAP 中直接看定位框。

        这个文件只用于人工验证定位，不参与后续 pipeline。overlay 与
        reference_image 使用同一套 size/spacing/origin/direction。数值约定：
        1 表示本次目标整体 bbox。默认不画每个 label 的 bbox，避免 ITK-SNAP
        里一屏叠太多框；如果临时需要细看单椎体范围，可把
        include_label_bboxes 打开，单个 label 的 bbox 会使用该 label 的第一个
        label id。因为只是研发辅助输出，后续稳定后删除本方法和调用点即可。
        """
        reference_array = sitk.GetArrayFromImage(reference_image)
        overlay_array = np.zeros(reference_array.shape, dtype=np.uint16)
        output_path = Path(output_path)

        def paint_bbox_edges(bbox: VoxelBBox, value: int) -> None:
            z_size, y_size, x_size = overlay_array.shape
            z_min = max(0, min(int(bbox.z_min), z_size - 1))
            z_max = max(0, min(int(bbox.z_max), z_size - 1))
            y_min = max(0, min(int(bbox.y_min), y_size - 1))
            y_max = max(0, min(int(bbox.y_max), y_size - 1))
            x_min = max(0, min(int(bbox.x_min), x_size - 1))
            x_max = max(0, min(int(bbox.x_max), x_size - 1))
            if z_min > z_max or y_min > y_max or x_min > x_max:
                return

            # 只画六个面的一层边框，不填满盒子。这样在 ITK-SNAP 里能看到
            # 定位范围，但不会遮住内部真实分割。
            overlay_array[z_min, y_min:y_max + 1, x_min:x_max + 1] = value
            overlay_array[z_max, y_min:y_max + 1, x_min:x_max + 1] = value
            overlay_array[z_min:z_max + 1, y_min, x_min:x_max + 1] = value
            overlay_array[z_min:z_max + 1, y_max, x_min:x_max + 1] = value
            overlay_array[z_min:z_max + 1, y_min:y_max + 1, x_min] = value
            overlay_array[z_min:z_max + 1, y_min:y_max + 1, x_max] = value

        paint_bbox_edges(correction_scope_bbox, 1)
        if include_label_bboxes:
            for coord in label_coordinates.values():
                if not coord.found or not coord.bbox or not coord.label_ids:
                    continue
                paint_bbox_edges(coord.bbox, int(coord.label_ids[0]))

        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    @staticmethod
    def _write_debug_target_mask_overlay(
        reference_image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        output_path: Path,
    ) -> str:
        """
        写出研发期真实目标 mask overlay，方便在 ITK-SNAP 中核对定位是否偏。

        bbox overlay 画的是轴对齐外接盒，在矢状位/冠状位上会被少量极值体素
        或斜着的目标形态撑大。这个 overlay 只保留本次请求关心的 label 体素，
        数值统一写成 1，用来回答“目标本身在哪里”，不表达外接盒范围。
        """
        target_mask = LocalizationStage._build_target_mask(seg_array, label_targets)
        overlay_array = target_mask.astype(np.uint8)
        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    @staticmethod
    def _write_debug_label_mask_overlay(
        reference_image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        output_path: Path,
    ) -> str:
        """
        写出研发期分 label mask overlay，用原始模型 label id 区分各个椎体。

        target mask overlay 会把本次目标统一写成 1，适合看整体是否对齐。
        这个 overlay 保留 seg.nii.gz 中的原始 label id，比如 C1/C2/C7 会
        使用各自模型 id，方便在 ITK-SNAP 中按颜色查看每个椎体自己的范围。
        """
        target_mask = LocalizationStage._build_target_mask(seg_array, label_targets)
        overlay_array = np.where(target_mask, seg_array, 0).astype(np.uint16)
        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    @staticmethod
    def _write_localization_result(
        output_path: Path,
        request,
        model_info: Dict[str, Any],
        cut_config: Dict[str, Any],
        resolved_cut_config: Dict[str, Any],
        label_targets: Dict[str, LabelTarget],
        label_coordinates: Dict[str, LabelCoordinate],
        dataset_json_path: Path,
        image: sitk.Image,
        seg_path: str,
        raw_for_reconstruction_path: str,
        seg_for_reconstruction_path: str,
        correction_applied: bool,
        correction_angle_degrees: float,
        correction_scope_bbox: VoxelBBox,
        correction_params: Dict[str, Any],
        corrected_raw_path: str,
        corrected_seg_path: str,
        debug_bbox_overlay_path: str,
        debug_target_mask_overlay_path: str,
        debug_label_mask_overlay_path: str,
        is_skewed: bool,
        skew_params: Dict[str, Any],
    ) -> None:
        """写出定位结果，供后续映射、MPR 和调试使用。"""
        localization_result = LocalizationResult(
            body_part=request.body_part.value,
            cut_mode=request.cut_mode.value,
            seg_path=str(seg_path),
            model=model_info,
            dataset_json_path=str(dataset_json_path),
            image=NiftiImageInfo(
                size_xyz=list(image.GetSize()),
                spacing_xyz=list(image.GetSpacing()),
                origin_xyz=list(image.GetOrigin()),
                direction=list(image.GetDirection()),
            ),
            cut_config=cut_config,
            resolved_cut_config=resolved_cut_config,
            label_targets=label_targets,
            coords=label_coordinates,
            raw_for_reconstruction_path=raw_for_reconstruction_path,
            seg_for_reconstruction_path=seg_for_reconstruction_path,
            correction_applied=correction_applied,
            correction_angle_degrees=correction_angle_degrees,
            correction_scope_bbox=correction_scope_bbox,
            correction_params=correction_params,
            corrected_raw_path=corrected_raw_path,
            corrected_seg_path=corrected_seg_path,
            debug_bbox_overlay_path=debug_bbox_overlay_path,
            debug_target_mask_overlay_path=debug_target_mask_overlay_path,
            debug_label_mask_overlay_path=debug_label_mask_overlay_path,
            is_skewed=is_skewed,
            skew_params=skew_params,
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(localization_result.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _label_targets_to_dict(label_targets: Dict[str, LabelTarget]) -> Dict[str, Dict[str, Any]]:
        return {
            name: target.to_dict()
            for name, target in label_targets.items()
        }

    @staticmethod
    def _coordinates_to_dict(label_coordinates: Dict[str, LabelCoordinate]) -> Dict[str, Dict[str, Any]]:
        return {
            name: coord.to_dict()
            for name, coord in label_coordinates.items()
        }

    def _resolve_cut_config(self, request) -> Dict[str, Any]:
        """
        根据请求中的部位和切割方式查找配置。

        外部只传 cut_mode；具体范围统一从 localization_config.json 读取。
        whole_range 使用 start/end；interspace 的每个 interval 也使用
        start/end，避免同一配置里混用 labels、数组等多种形态。
        """
        body_part = getattr(request.body_part, 'value', request.body_part)
        cut_mode = getattr(request.cut_mode, 'value', request.cut_mode)
        target_configs = self._localization_config.get('targets', {})
        target_config = target_configs.get(body_part)
        if not target_config:
            raise ValueError(f"未找到部位定位配置: {body_part}")

        cut_config = target_config.get(cut_mode)
        if not cut_config:
            raise ValueError(f"未找到切割方式配置: body_part={body_part}, cut_mode={cut_mode}")

        self._validate_cut_config(cut_mode, cut_config)
        return cut_config

    def _resolve_label_aliases(self, cut_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        把业务短名解析成模型/数据集中的标准 label 名。

        例如配置中可以写 L1、C7、skull；这里会根据 label_aliases
        转成 vertebrae_L1、vertebrae_C7、skull。后续再用这些标准名
        去 dataset.json 中查 label id。
        """
        if 'intervals' in cut_config:
            return {
                'intervals': [
                    {
                        'start': interval['start'],
                        'end': interval['end'],
                        'start_labels': self._label_alias(interval['start']),
                        'end_labels': self._label_alias(interval['end'])
                    }
                    for interval in cut_config['intervals']
                ]
            }

        return {
            'start': cut_config['start'],
            'end': cut_config['end'],
            'start_labels': self._label_alias(cut_config['start']),
            'end_labels': self._label_alias(cut_config['end'])
        }

    def _label_alias(self, name: str) -> list[str]:
        """
        获取一个定位名称对应的标准 label 名列表。

        alias 可以是字符串，也可以是字符串数组。肺部这类由左右肺共同
        构成的整体范围，可以用一个业务名 lung 映射到多个 label。
        """
        aliases = self._localization_config.get('label_aliases', {})
        value = aliases.get(name, name)
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _validate_cut_config(cut_mode: str, cut_config: Dict[str, Any]) -> None:
        """
        校验切割配置结构。

        配置统一使用 start/end 表达范围：
            whole_range: {"start": "L1", "end": "S1"}
            interspace: {"intervals": [{"start": "L1", "end": "L2"}]}
        """
        if cut_mode == 'whole_range':
            if not cut_config.get('start') or not cut_config.get('end'):
                raise ValueError(f"whole_range 配置必须包含 start/end: {cut_config}")
            return

        if cut_mode == 'interspace':
            intervals = cut_config.get('intervals')
            if not intervals:
                raise ValueError(f"interspace 配置必须包含 intervals: {cut_config}")
            for interval in intervals:
                if not interval.get('start') or not interval.get('end'):
                    raise ValueError(f"interspace interval 必须包含 start/end: {interval}")
            return

        raise ValueError(f"未知切割方式: {cut_mode}")
