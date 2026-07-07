#!/src/services/stage_localization.py
# -*- coding: utf-8 -*-

"""
阶段三：定位与矫正
椎体定位、歪斜检测与回正
"""

from pathlib import Path
from typing import Dict, Any
import json

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.utils import DebugLog


class LocalizationStage(BaseStage):
    """定位与矫正阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="localization", config=config)
        self._load_config()

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
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        prev_result = context.get_result('segmentation')
        
        if not prev_result:
            raise ValueError("前置阶段 'segmentation' 结果不存在")
        
        seg_path = prev_result.get('seg_path')
        
        DebugLog("=" * 60)
        DebugLog("【阶段三】定位与矫正 - 椎体定位与歪斜检测")
        DebugLog(f"  输入分割: {seg_path}")
        DebugLog(f"  部位: {request.body_part}")
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
        # TODO: 实现具体逻辑
        
        # [2] extract_vertebra_coordinates: 提取椎体坐标
        DebugLog("  [2] extract_vertebra_coordinates: 提取椎体质心+包围盒...")
        # TODO: 实现具体逻辑
        
        # [3] detect_skew: 检测歪斜
        DebugLog("  [3] detect_skew: 检测歪斜...")
        # TODO: 实现具体逻辑
        
        # [4] correct_skew: 回正变换
        DebugLog("  [4] correct_skew: 执行回正变换...")
        # TODO: 实现具体逻辑
        
        nifti_path = Path(request.output_dir) / 'nifti'
        result = {
            'coords': {},
            'cut_config': cut_config,
            'resolved_cut_config': resolved_cut_config,
            'corrected_raw_path': str(nifti_path / 'raw_corrected.nii.gz'),
            'corrected_seg_path': str(nifti_path / 'seg_corrected.nii.gz'),
            'is_skewed': False,
            'skew_params': {}
        }
        
        context.set_result(self.name, result)
        
        DebugLog("  ✅ 阶段三完成")
        DebugLog("=" * 60)
        
        return context

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

        例如配置中可以写 L1、C7、brain；这里会根据 label_aliases
        转成 vertebrae_L1、vertebrae_C7、brain。后续再用这些标准名
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
