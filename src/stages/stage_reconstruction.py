#!/src/services/stage_reconstruction.py
# -*- coding: utf-8 -*-

"""
阶段五：重建与输出
MPR多平面重建与DICOM生成
支持三平面并行处理
"""

from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.utils import DebugLog, ErrorLog


class ReconstructionStage(BaseStage):
    """重建与输出阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="reconstruction", config=config)
        self.parallel = config.get('parallel', True) if config else True
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        
        loc_result = context.get_result('localization')
        map_result = context.get_result('mapping')
        
        if not loc_result:
            raise ValueError("前置阶段 'localization' 结果不存在")
        if not map_result:
            raise ValueError("前置阶段 'mapping' 结果不存在")
        
        corrected_path = loc_result.get('corrected_raw_path')
        axial_params = map_result.get('axial_params', {})
        sagittal_params = map_result.get('sagittal_params', {})
        coronal_params = map_result.get('coronal_params', {})
        
        DebugLog("=" * 60)
        DebugLog("【阶段五】重建与输出 - MPR重建与DICOM生成")
        DebugLog(f"  输入图像: {corrected_path}")
        DebugLog(f"  输出目录: {request.output_dir}")
        DebugLog("=" * 60)
        
        output_path = Path(request.output_dir)
        axial_dir = output_path / '横状位'
        sagittal_dir = output_path / '矢状位'
        coronal_dir = output_path / '冠状位'
        
        for d in [axial_dir, sagittal_dir, coronal_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        tasks = [
            ('axial', self._process_axial, corrected_path, axial_dir, axial_params),
            ('sagittal', self._process_sagittal, corrected_path, sagittal_dir, sagittal_params),
            ('coronal', self._process_coronal, corrected_path, coronal_dir, coronal_params)
        ]
        
        if self.parallel:
            DebugLog("  启动三平面并行处理...")
            results = self._run_parallel(tasks)
        else:
            DebugLog("  三平面串行处理...")
            results = self._run_serial(tasks)
        
        result = {
            'axial': results.get('axial', {'count': 0, 'output_dir': str(axial_dir)}),
            'sagittal': results.get('sagittal', {'count': 0, 'output_dir': str(sagittal_dir)}),
            'coronal': results.get('coronal', {'count': 0, 'output_dir': str(coronal_dir)})
        }
        
        context.set_result(self.name, result)
        
        DebugLog("  ✅ 阶段五完成")
        DebugLog("=" * 60)
        
        return context
    
    def _run_serial(self, tasks: List[tuple]) -> Dict[str, Dict]:
        results = {}
        for plane, func, *args in tasks:
            DebugLog(f"  [{plane}] 开始处理...")
            results[plane] = func(*args)
            DebugLog(f"  [{plane}] 完成")
        return results
    
    def _run_parallel(self, tasks: List[tuple]) -> Dict[str, Dict]:
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_map = {}
            for plane, func, *args in tasks:
                DebugLog(f"  [{plane}] 提交任务...")
                future = executor.submit(func, *args)
                future_map[future] = plane
            
            for future in as_completed(future_map):
                plane = future_map[future]
                try:
                    results[plane] = future.result()
                    DebugLog(f"  [{plane}] 完成")
                except Exception as e:
                    ErrorLog(f"  [{plane}] 失败", e)
                    results[plane] = {'count': 0, 'error': str(e)}
        return results
    
    def _process_axial(self, image_path: str, output_dir: Path, params: Dict) -> Dict:
        DebugLog("    [横状位] MPR重建...")
        # TODO: 实现 resample_axial
        # TODO: 实现 apply_window
        # TODO: 实现 generate_dicom
        return {'count': 0}
    
    def _process_sagittal(self, image_path: str, output_dir: Path, params: Dict) -> Dict:
        DebugLog("    [矢状位] MPR重建...")
        # TODO: 实现 resample_sagittal
        # TODO: 实现 apply_window
        # TODO: 实现 generate_dicom
        return {'count': 0}
    
    def _process_coronal(self, image_path: str, output_dir: Path, params: Dict) -> Dict:
        DebugLog("    [冠状位] MPR重建...")
        # TODO: 实现 resample_coronal
        # TODO: 实现 apply_window
        # TODO: 实现 generate_dicom
        return {'count': 0}