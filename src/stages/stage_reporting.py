#!/src/stages/stage_reporting.py
# # -*- coding: utf-8 -*-

"""
阶段六：结果汇总
生成报告与输出整理
"""

import json
from pathlib import Path
from typing import Dict, Any

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.utils import DebugLog


class ReportingStage(BaseStage):
    """结果汇总阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="reporting", config=config)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        
        DebugLog("=" * 60)
        DebugLog("【阶段六】结果汇总 - 生成报告")
        DebugLog(f"  输出目录: {request.output_dir}")
        DebugLog("=" * 60)
        
        # [1] generate_report: 生成报告
        DebugLog("  [1] generate_report: 生成处理报告...")
        report_path = self._generate_report(context)
        
        # [2] organize_output: 整理输出
        DebugLog("  [2] organize_output: 整理输出目录结构...")
        self._organize_output(context)
        
        result = {
            'report_path': str(report_path),
            'total_time': context.get_total_time()
        }
        
        context.set_result(self.name, result)
        
        DebugLog(f"  ✅ 阶段六完成，总耗时: {context.get_total_time():.2f}s")
        DebugLog("=" * 60)
        
        return context
    
    def _generate_report(self, context: ProcessingContext) -> Path:
        output_dir = Path(context.request.output_dir)
        report_path = output_dir / 'report.json'
        
        report_data = {
            'success': not context.has_errors(),
            # 请求参数统一由 Request.to_dict() 输出，避免新增字段后报告遗漏。
            'request': context.request.to_dict(),
            'timings': context.timings,
            'total_time': context.get_total_time(),
            'errors': [
                {'stage': e.stage, 'message': e.message}
                for e in context.errors
            ],
            'outputs': {
                'nifti': str(output_dir / 'nifti'),
                'axial': str(output_dir / '横状位'),
                'sagittal': str(output_dir / '矢状位'),
                'coronal': str(output_dir / '冠状位')
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        DebugLog(f"    报告已生成: {report_path}")
        return report_path
    
    def _organize_output(self, context: ProcessingContext):
        DebugLog("    [占位] 整理输出目录...")
        # TODO: 实现具体逻辑
