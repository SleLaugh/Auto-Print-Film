# -*- coding: utf-8 -*-

"""
响应对象模块

本模块定义了 Pipeline 执行完成后的响应对象。

设计思路：
    - 封装执行结果，提供统一的访问接口
    - 通过属性方法（@property）提供便捷的数据访问
    - 支持转换为字典，便于序列化（如 JSON 输出）
    - 与 ProcessingContext 解耦，只暴露必要的信息

使用示例：
    response = pipeline.run(request)
    
    if response.success:
        print(f"输出目录: {response.output_dir}")
        print(f"横状位: {response.get_axial_count()} 张")
        print(f"矢状位: {response.get_sagittal_count()} 张")
        print(f"冠状位: {response.get_coronal_count()} 张")
    else:
        for error in response.errors:
            print(f"错误: {error['stage']} - {error['message']}")
    
    # 转换为字典（用于API响应）
    result_dict = response.to_dict()
"""

from typing import Any, Optional, Dict, List
from dataclasses import dataclass

from .context import ProcessingContext


@dataclass
class Response:
    """
    Pipeline执行响应对象
    
    封装了 Pipeline 执行完成后的所有信息。
    
    属性：
        success: 是否全部阶段执行成功（无错误）
        request: 原始输入请求对象
        context: 处理上下文（包含详细结果和错误信息）
        total_time: 总执行耗时（秒）
    
    便捷属性：
        output_dir: 输出目录路径
        errors: 错误列表（格式化的字典列表）
        timings: 各阶段耗时字典
    """
    
    success: bool
    """是否全部阶段执行成功（无错误）"""
    
    request: Any
    """原始输入请求对象"""
    
    context: ProcessingContext
    """处理上下文，包含详细结果和错误信息"""
    
    total_time: float = 0.0
    """总执行耗时（秒）"""
    
    @property
    def output_dir(self) -> Optional[str]:
        """
        获取输出目录路径
        
        Returns:
            输出目录路径，如果请求对象没有该属性则返回 None
        """
        return getattr(self.request, 'output_dir', None)
    
    @property
    def errors(self) -> List[Dict[str, str]]:
        """
        获取格式化的错误列表
        
        Returns:
            错误列表，每个元素包含 'stage' 和 'message' 字段
        """
        return [
            {'stage': e.stage, 'message': e.message}
            for e in self.context.errors
        ]
    
    @property
    def timings(self) -> Dict[str, float]:
        """
        获取各阶段耗时统计
        
        Returns:
            字典 {阶段名称: 耗时(秒)}
        """
        return self.context.timings

    @property
    def status(self) -> str:
        """任务生命周期状态。"""

        return self.context.job_status.value

    @property
    def warnings(self) -> List[Dict[str, Any]]:
        """返回流水线执行期间收集的结构化警告列表。"""

        return list(self.context.warnings)
    
    def get_stage_result(self, stage_name: str) -> Optional[Any]:
        """
        获取指定阶段的执行结果
        
        Args:
            stage_name: 阶段名称（如 'preprocessing', 'segmentation'）
            
        Returns:
            该阶段的结果数据，如果不存在则返回 None
        """
        return self.context.get_result(stage_name)
    
    def get_axial_count(self) -> int:
        """
        获取横状位切片数量
        
        从 reconstruction 阶段的 axial 结果中提取 count 字段
        
        Returns:
            横状位切片数量，默认为 0
        """
        result = self.context.get_result('reconstruction')
        if result:
            axial = result.get('axial', {})
            return axial.get('count', 0)
        return 0
    
    def get_sagittal_count(self) -> int:
        """
        获取矢状位切片数量
        
        从 reconstruction 阶段的 sagittal 结果中提取 count 字段
        
        Returns:
            矢状位切片数量，默认为 0
        """
        result = self.context.get_result('reconstruction')
        if result:
            sagittal = result.get('sagittal', {})
            return sagittal.get('count', 0)
        return 0
    
    def get_coronal_count(self) -> int:
        """
        获取冠状位切片数量
        
        从 reconstruction 阶段的 coronal 结果中提取 count 字段
        
        Returns:
            冠状位切片数量，默认为 0
        """
        result = self.context.get_result('reconstruction')
        if result:
            coronal = result.get('coronal', {})
            return coronal.get('count', 0)
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将响应转换为字典格式
        
        便于序列化为 JSON，用于 API 响应或日志输出。
        
        Returns:
            包含所有关键信息的字典
        """
        return {
            'success': self.success,
            'output_dir': self.output_dir,
            'total_time': self.total_time,
            'errors': self.errors,
            'timings': self.timings,
            'axial_count': self.get_axial_count(),
            'sagittal_count': self.get_sagittal_count(),
            'coronal_count': self.get_coronal_count(),
            'job_id': self.context.job_id,
            'status': self.status,
            'warnings': self.warnings,
            'stages': {
                name: record.status.value
                for name, record in self.context.stage_records.items()
            },
        }
