# -*- coding: utf-8 -*-

"""
Pipeline执行器模块

本模块定义了 Pipeline 执行器，负责编排和调度所有处理阶段。

设计思路：
    - 采用管道模式（Pipeline Pattern），将复杂处理流程拆分为多个阶段
    - 每个阶段独立实现，通过上下文传递数据
    - 支持阶段串行执行，单个阶段内部可支持并行（如三平面重建）
    - 统一的异常处理和错误收集机制

执行流程：
    1. 创建 ProcessingContext（包含输入请求）
    2. 按顺序依次执行每个阶段
    3. 每个阶段接收上下文，处理后返回更新后的上下文
    4. 如果某个阶段失败，可选择中断或继续（可配置）
    5. 最终生成 Response 对象返回

使用示例：
    pipeline = Pipeline()
    response = pipeline.run(request)
    if response.success:
        print("处理成功")
    else:
        print(f"失败: {response.errors}")
"""

import time
import shutil
from pathlib import Path
from typing import List, Optional, Any
from src.utils import ErrorLog, DebugLog
from .base_stage import BaseStage
from .context import ProcessingContext
from .response import Response


class Pipeline:
    """
    Pipeline执行器
    
    负责管理和执行所有处理阶段。
    
    属性：
        stages: 阶段列表，按顺序执行
        
    使用示例：
        # 使用默认阶段列表
        pipeline = Pipeline()
        response = pipeline.run(request)
        
        # 自定义阶段列表
        custom_stages = [MyStage1(), MyStage2()]
        pipeline = Pipeline(stages=custom_stages)
        response = pipeline.run(request)
    """
    
    def __init__(self, stages: Optional[List[BaseStage]] = None):
        """
        初始化Pipeline
        
        Args:
            stages: 阶段列表（按执行顺序），如果为 None 则使用默认的6个阶段
        """
        self.stages = stages or self._default_stages()
        """阶段列表，按执行顺序排列"""
    
    def _default_stages(self) -> List[BaseStage]:
        """
        获取默认的阶段列表
        
        按以下顺序执行：
            1. PreprocessingStage   - 预处理（DICOM → NIfTI）
            2. SegmentationStage    - 推理分割（模型匹配 + nnUNet推理）
            3. LocalizationStage    - 定位与回正占位（目标定位 + bbox + 回正 TODO 状态）
            4. MappingStage         - 坐标映射（三平面范围映射）
            5. ReconstructionStage  - 重建与输出（MPR重建 + DICOM生成）
            6. ReportingStage       - 结果汇总（生成报告）
        
        Returns:
            默认的阶段实例列表
        """
        
        from src.stages import ( # 延迟导入避免循环依赖
            PreprocessingStage,
            SegmentationStage,
            LocalizationStage,
            MappingStage,
            ReconstructionStage,
            ReportingStage
        )
        
        return [
            PreprocessingStage(),
            SegmentationStage(),
            LocalizationStage(),
            MappingStage(),
            ReconstructionStage(),
            ReportingStage()
        ]
    
    def run(self, request: Any) -> Response:
        """
        执行整个处理流程
        
        这是 Pipeline 的核心方法，按顺序执行所有阶段。
        
        Args:
            request: 输入请求对象（如 Request 实体）
            
        Returns:
            Response 对象，包含执行结果、状态、耗时等信息
        """
        DebugLog("Pipeline 开始执行")
        DebugLog(f"共 {len(self.stages)} 个阶段")
        
        self._prepare_output_dir(request)
        
        context = ProcessingContext(request) # 创建处理上下文
        pipeline_start = time.time()        
        
        for stage in self.stages: # 按顺序执行每个阶段
            try:
                DebugLog(f"▶ 执行阶段: {stage.name}")
                stage_start = time.time()
                
                
                context.set_current_stage(stage.name) # 设置当前阶段（用于计时）
                
                context = stage.execute(context) # 执行阶段
                
                stage_elapsed = time.time() - stage_start # 记录阶段耗时
                DebugLog(f"  ✅ 阶段完成: {stage.name} (耗时: {stage_elapsed:.2f}s)")
                
            except Exception as e:
                ErrorLog(f"阶段执行异常: {stage.name}", e)
                context.add_error(stage.name, e)
                
                # TODO: 根据配置决定是否继续执行后续阶段
                # 目前策略：遇到错误立即停止
                break
        
        pipeline_elapsed = time.time() - pipeline_start # 生成响应
        
        response = Response(
            success=not context.has_errors(),
            request=request,
            context=context,
            total_time=pipeline_elapsed
        )
        
        DebugLog("Pipeline 执行完成")
        DebugLog(f"状态: {'✅ 成功' if response.success else '❌ 失败'}")
        DebugLog(f"总耗时: {pipeline_elapsed:.2f}s")
        
        return response

    def _prepare_output_dir(self, request: Any) -> None:
        """
        清理并重建本次任务的输出根目录。

        输出目录是一次任务的完整产物目录。重新运行同一个 output_dir 时，
        先删除旧目录，避免读取到上一次运行留下的 NIfTI、分割或报告文件。
        """
        output_dir = getattr(request, 'output_dir', None)
        if not output_dir:
            return

        output_path = Path(output_dir)
        if output_path.exists():
            DebugLog(f"清理旧输出目录: {output_path}")
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
