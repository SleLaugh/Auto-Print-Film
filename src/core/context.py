# -*- coding: utf-8 -*-

"""
处理上下文模块

本模块定义了贯穿整个处理流程的上下文对象。
上下文是 Pipeline 各阶段之间传递数据的核心载体。

设计思路：
    - 每个 Pipeline 执行过程对应一个上下文实例
    - 上下文保存了输入请求、各阶段结果、错误信息和耗时统计
    - 阶段之间通过上下文的 set_result/get_result 方法传递数据
    - 错误信息会被统一收集，便于最终生成报告

数据流向：
    Request → ProcessingContext → Stage1 → Stage2 → ... → StageN → Response
                ↑                 ↑        ↑
                └── 各阶段结果 ─────┴────────┘
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ErrorInfo:
    """
    错误信息数据类
    
    用于记录阶段执行过程中发生的错误详情
    
    属性：
        stage: 发生错误的阶段名称
        message: 错误消息
        exception: 异常类型名称（如 "ValueError"）
        timestamp: 错误发生时间（ISO格式）
    """
    stage: str
    """发生错误的阶段名称"""
    
    message: str
    """错误消息描述"""
    
    exception: Optional[str] = None
    """异常类型名称（如 'ValueError', 'RuntimeError'）"""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """错误发生时间，ISO 8601 格式"""


class ProcessingContext:
    """
    处理上下文类
    
    贯穿整个 Pipeline 执行过程的数据容器。
    负责存储输入请求、各阶段执行结果、错误信息和性能统计。
    
    属性：
        request: 原始输入请求对象
        results: 各阶段结果字典 {阶段名: 结果数据}
        metadata: 元数据字典（用于存储额外信息，如文件大小、版本等）
        errors: 错误信息列表
        timings: 各阶段耗时字典 {阶段名: 耗时秒数}
    
    使用示例：
        # 在阶段中保存结果
        context.set_result("my_stage", {"output": "/path/to/file"})
        
        # 在后续阶段中获取结果
        prev = context.get_result("my_stage")
        if prev:
            file_path = prev.get("output")
    """
    
    def __init__(self, request: Any):
        """
        初始化处理上下文
        
        Args:
            request: 输入请求对象（如 Request 实体）
        """
        self.request = request
        """原始输入请求对象"""
        
        self.results: Dict[str, Any] = {}
        """各阶段执行结果，键为阶段名称，值为任意类型的结果数据"""
        
        self.metadata: Dict[str, Any] = {}
        """元数据存储，用于记录额外信息"""
        
        self.errors: List[ErrorInfo] = []
        """执行过程中收集的错误信息列表"""
        
        self.timings: Dict[str, float] = {}
        """各阶段耗时统计，键为阶段名称，值为耗时（秒）"""
        
        self._current_stage: Optional[str] = None
        """当前正在执行的阶段名称（内部使用，用于计时）"""
        
        self._start_time: Optional[datetime] = None
        """当前阶段的开始时间（内部使用，用于计时）"""
    
    def set_current_stage(self, stage_name: str) -> None:
        """
        设置当前正在执行的阶段（由 Pipeline 调用）
        
        此方法会记录阶段开始时间，用于后续计算耗时。
        
        Args:
            stage_name: 阶段名称
        """
        self._current_stage = stage_name
        self._start_time = datetime.now()
    
    def set_result(self, stage_name: str, data: Any) -> None:
        """
        保存某个阶段的执行结果
        
        该方法会自动记录该阶段的耗时（从 set_current_stage 开始计时）。
        
        Args:
            stage_name: 阶段名称
            data: 要保存的结果数据（可以是任意类型）
        """
        self.results[stage_name] = data
        
        # 自动计算并记录该阶段耗时
        if self._start_time and self._current_stage == stage_name:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.timings[stage_name] = elapsed
    
    def get_result(self, stage_name: str) -> Optional[Any]:
        """
        获取某个阶段的执行结果
        
        Args:
            stage_name: 阶段名称
            
        Returns:
            该阶段的执行结果，如果不存在则返回 None
        """
        return self.results.get(stage_name)
    
    def add_error(self, stage_name: str, exception: Exception) -> None:
        """
        添加错误信息到上下文
        
        Args:
            stage_name: 发生错误的阶段名称
            exception: 捕获到的异常对象
        """
        self.errors.append(ErrorInfo(
            stage=stage_name,
            message=str(exception),
            exception=type(exception).__name__
        ))
    
    def has_errors(self) -> bool:
        """
        判断执行过程中是否有错误发生
        
        Returns:
            True 表示有错误，False 表示无错误
        """
        return len(self.errors) > 0
    
    def get_total_time(self) -> float:
        """
        获取所有阶段的总耗时
        
        Returns:
            总耗时（秒）
        """
        return sum(self.timings.values())
    
    def get_stage_time(self, stage_name: str) -> float:
        """
        获取指定阶段的耗时
        
        Args:
            stage_name: 阶段名称
            
        Returns:
            该阶段的耗时（秒），如果不存在则返回 0.0
        """
        return self.timings.get(stage_name, 0.0)