# -*- coding: utf-8 -*-

"""
Stage基类模块

本模块定义了所有处理阶段的基类 BaseStage。
所有具体的阶段（如预处理、分割、定位等）都需要继承此类并实现 execute 方法。

设计思路：
    - 采用模板方法模式，每个阶段都遵循相同的执行流程
    - 每个阶段接收 ProcessingContext 上下文，处理后返回更新后的上下文
    - 阶段之间通过上下文传递数据，实现解耦

使用示例：
    class MyStage(BaseStage):
        def __init__(self, config=None):
            super().__init__(name="my_stage", config=config)
        
        def execute(self, context: ProcessingContext) -> ProcessingContext:
            # 从上下文获取前置阶段的结果
            prev_result = context.get_result("previous_stage")
            # 执行业务逻辑...
            # 保存结果到上下文
            context.set_result(self.name, {"key": "value"})
            return context
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from .context import ProcessingContext


class BaseStage(ABC):
    """
    所有处理阶段的抽象基类
    
    属性：
        name: 阶段名称，用于日志标识和上下文中的结果索引
        config: 该阶段的专属配置字典，支持灵活扩展
    
    子类必须实现：
        execute: 执行阶段的核心业务逻辑
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化阶段实例
        
        Args:
            name: 阶段名称（建议使用英文小写加下划线，如 "preprocessing"）
            config: 该阶段的配置参数（可选），如超参数、开关等
        """
        self.name = name
        """阶段名称，用于日志和上下文索引"""
        
        self.config = config or {}
        """阶段专属配置字典"""
    
    @abstractmethod
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        执行当前阶段的核心业务逻辑
        
        子类必须实现此方法。典型实现步骤：
            1. 从 context 中获取前置阶段的结果（如需要）
            2. 执行业务逻辑（如读取文件、调用模型等）
            3. 将结果保存到 context 中
            4. 返回更新后的 context
        
        Args:
            context: 处理上下文对象，包含请求参数和所有阶段的结果
            
        Returns:
            更新后的处理上下文对象
            
        Raises:
            Exception: 执行失败时抛出异常，由 Pipeline 统一处理
        """
        pass