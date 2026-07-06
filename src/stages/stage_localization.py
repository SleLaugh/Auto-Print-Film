#!/src/services/stage_localization.py
# -*- coding: utf-8 -*-

"""
阶段三：定位与矫正
椎体定位、歪斜检测与回正
"""

from pathlib import Path
from typing import Dict, Any

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.utils import DebugLog


class LocalizationStage(BaseStage):
    """定位与矫正阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="localization", config=config)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        prev_result = context.get_result('segmentation')
        
        if not prev_result:
            raise ValueError("前置阶段 'segmentation' 结果不存在")
        
        seg_path = prev_result.get('seg_path')
        
        DebugLog("=" * 60)
        DebugLog("【阶段三】定位与矫正 - 椎体定位与歪斜检测")
        DebugLog(f"  输入分割: {seg_path}")
        DebugLog(f"  部位类型: {request.body_part}")
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
            'corrected_raw_path': str(nifti_path / 'raw_corrected.nii.gz'),
            'corrected_seg_path': str(nifti_path / 'seg_corrected.nii.gz'),
            'is_skewed': False,
            'skew_params': {}
        }
        
        context.set_result(self.name, result)
        
        DebugLog("  ✅ 阶段三完成")
        DebugLog("=" * 60)
        
        return context