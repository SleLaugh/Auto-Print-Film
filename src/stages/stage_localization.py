#!/src/services/stage_localization.py
# -*- coding: utf-8 -*-

"""
阶段三：定位与矫正
椎体定位、歪斜检测与回正
"""

from pathlib import Path
from typing import Dict, Any
import json
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
    """定位与矫正阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="localization", config=config)
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
        
        if not prev_result:
            raise ValueError("前置阶段 'segmentation' 结果不存在")
        
        seg_path = prev_result.get('seg_path')
        model_info = prev_result.get('model') or {}
        if not seg_path:
            raise ValueError("分割结果路径不存在")
        
        DebugLog("=" * 60)
        DebugLog("【阶段三】定位与矫正 - 椎体定位与歪斜检测")
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
        dataset_json_path = self._find_dataset_json(model_info)
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
        DebugLog(f"    已提取 {len(label_coordinates)} 个Label坐标")
        
        # [3] detect_skew: 检测歪斜
        DebugLog("  [3] detect_skew: 检测歪斜...")
        # TODO: 实现具体逻辑
        
        # [4] correct_skew: 回正变换
        DebugLog("  [4] correct_skew: 执行回正变换...")
        # TODO: 实现具体逻辑
        
        nifti_path = Path(request.output_dir) / 'nifti'
        localization_result_path = output_path / 'localization_result.json'
        result = {
            'coords': self._coordinates_to_dict(label_coordinates),
            'cut_config': cut_config,
            'resolved_cut_config': resolved_cut_config,
            'label_targets': self._label_targets_to_dict(label_targets),
            'dataset_json_path': str(dataset_json_path),
            'localization_result_path': str(localization_result_path),
            'corrected_raw_path': str(nifti_path / 'raw_corrected.nii.gz'),
            'corrected_seg_path': str(nifti_path / 'seg_corrected.nii.gz'),
            'is_skewed': False,
            'skew_params': {}
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
            seg_path=seg_path,
            is_skewed=result['is_skewed'],
            skew_params=result['skew_params'],
        )
        
        context.set_result(self.name, result)
        
        DebugLog("  ✅ 阶段三完成")
        DebugLog("=" * 60)

        return context

    def _find_dataset_json(self, model_info: Dict[str, Any]) -> Path:
        """
        根据 segmentation 阶段匹配到的模型 ID 查找对应 dataset.json。

        nnUNet 目录常见形态是 Dataset292_xxx 或 Dataset315_xxx。优先从
        nnUNet_results 查找，因为它最贴近实际用于推理的模型；找不到再
        查 preprocessed/raw。
        """
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
        label_names = self._collect_label_names(resolved_cut_config)
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

    @staticmethod
    def _collect_label_names(resolved_cut_config: Dict[str, Any]) -> list[str]:
        label_names = []
        if 'intervals' in resolved_cut_config:
            for interval in resolved_cut_config['intervals']:
                label_names.extend(interval['start_labels'])
                label_names.extend(interval['end_labels'])
        else:
            label_names.extend(resolved_cut_config['start_labels'])
            label_names.extend(resolved_cut_config['end_labels'])

        seen = set()
        unique_names = []
        for label_name in label_names:
            if label_name in seen:
                continue
            seen.add(label_name)
            unique_names.append(label_name)
        return unique_names

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
