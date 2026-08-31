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
import uuid

from .lifecycle import JobStatus, StageRecord


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
        metadata: 元数据字典（用于存储额外信息，如文件大小、版本等）
    
    贯穿整个 Pipeline 执行过程的数据容器。
    负责存储输入请求、各阶段执行结果、错误信息和性能统计。
    
    属性：
        request: 原始输入请求对象
        results: 各阶段结果字典 {阶段名: 结果数据}
        metadata: 元数据字典（用于存储额外信息，如文件大小、版本等）
        errors: 错误信息列表
        timings: 各阶段耗时字典 {阶段名: 耗时秒数}
    
        results: 各阶段结果字典 {阶段名: 结果数据}
        metadata: 元数据字典（用于存储额外信息，如文件大小、版本等）
    使用示例：
        # 在阶段中保存结果
        context.set_result("my_stage", {"output": "/path/to/file"})
        
        # 在后续阶段中获取结果
        prev = context.get_result("my_stage")
        if prev:
            file_path = prev.get("output")
    """
    
    def __init__(self, request: Any, stage_names: Optional[List[str]] = None):
        """
        初始化处理上下文
        
        Args:
            request: 输入请求对象（如 Request 实体）
        """
        self.request = request
        """原始输入请求对象"""

        self.job_id = str(uuid.uuid4())
        """本次任务唯一 ID，用于关联 Response、manifest 和日志。"""
        self.job_status = JobStatus.PENDING
        """整个任务的当前生命周期状态，由 Pipeline 统一修改。"""
        self.created_at = datetime.now().astimezone().isoformat()
        """Context 创建时间，使用带时区 ISO-8601 格式。"""
        self.started_at: Optional[str] = None
        """任务正式开始执行的时间；预检前保持为空。"""
        self.finished_at: Optional[str] = None
        """任务成功、失败或取消后的结束时间。"""
        self.stage_records: Dict[str, StageRecord] = {
            name: StageRecord(name)
            for name in (stage_names or [])
        }
        """按阶段名保存生命周期、耗时、警告和错误记录。"""
        self.warnings: List[Dict[str, Any]] = []
        """不导致任务失败的框架级警告。"""
        self.effective_config_summary: Dict[str, Any] = {}
        """不含绝对模型路径和隐私信息的生效配置摘要。"""
        self.software_version = "1.0.0"
        """生成本次结果的软件版本，写入 manifest 供追溯。"""
        
        self.results: Dict[str, Any] = {}
        """各阶段执行结果，键为阶段名称，值为任意类型的结果数据"""

        self.stage_results: Dict[str, Any] = {}
        """新框架使用的有类型阶段结果。"""
        
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

    def set_stage_result(self, result: Any) -> None:
        """保存有类型结果，并同步阶段业务字典供后续处理读取。"""

        stage_name = getattr(result, 'stage_name', '')
        to_dict = getattr(result, 'to_dict', None)
        if not stage_name or not callable(to_dict):
            from .errors import StageInputError
            raise StageInputError("阶段结果缺少 stage_name 或 to_dict()")
        self.stage_results[stage_name] = result
        result_data = to_dict()
        result_data.pop('stage_name', None)
        result_data.pop('schema_version', None)
        self.set_result(stage_name, result_data)

    def require_stage_result(self, stage_name: str, expected_type: Any) -> Any:
        """读取指定类型的前置结果；缺失或类型错误时给出标准错误。"""

        from .errors import StageInputError

        result = self.stage_results.get(stage_name)
        if result is None:
            raise StageInputError(
                f"前置阶段结果不存在: {stage_name}",
                stage=self._current_stage,
            )
        if not isinstance(result, expected_type):
            raise StageInputError(
                f"前置阶段结果类型错误: {stage_name}",
                stage=self._current_stage,
                details={
                    "expected": expected_type.__name__,
                    "actual": type(result).__name__,
                },
            )
        return result
    
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
