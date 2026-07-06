"""
枚举定义模块
定义系统中使用的所有枚举类型
"""

from enum import Enum


class BodyPartEnum(Enum):
    """身体部位枚举"""
    颅脑 = "颅脑"
    颈椎 = "颈椎"
    胸椎 = "胸椎"
    腰椎 = "腰椎"
    肺部 = "肺部"
    骶骨 = "骶骨"
    髋骨 = "髋骨"


class PositioningModeEnum(Enum):
    """定位模式枚举"""
    单区间 = "single"          # 单区间模式（大多数部位使用）
    多区间 = "multi"           # 多区间模式（腰椎间盘突出需要多区间划分）


class PlaneTypeEnum(Enum):
    """重建平面类型枚举"""
    横状位 = "横状位"
    矢状位 = "矢状位"
    冠状位 = "冠状位"


class ProcessingStatusEnum(Enum):
    """处理状态枚举"""
    待处理 = "pending"
    处理中 = "running"
    成功 = "success"
    失败 = "failed"