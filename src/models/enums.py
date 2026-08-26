"""
枚举定义模块
定义系统中使用的所有枚举类型
"""

from enum import Enum


class SlicePartEnum(Enum):
    """切割部位枚举。"""
    头骨 = "头骨"
    颈椎 = "颈椎"
    胸椎 = "胸椎"
    腰椎 = "腰椎"
    肺部 = "肺部"
    骶骨 = "骶骨"
    髋骨 = "髋骨"


class StudyTypeEnum(Enum):
    """检查类型枚举"""
    CT = "CT"
    MR = "MR"


class SliceType(Enum):
    """外部请求可选择的切片方式。"""

    MPR = "MPR"
    CPR = "CPR"
    椎间隙 = "椎间隙"


class PlaneTypeEnum(Enum):
    """重建平面类型枚举"""
    # 横状位、矢状位、冠状位会直接对应后续输出目录和 MPR 采样方向。
    横状位 = "横状位"
    矢状位 = "矢状位"
    冠状位 = "冠状位"


class CutLineExtentModeEnum(Enum):
    """切割线在每张平面内的横向范围。"""
    整平面 = "整平面"
    部位范围 = "部位范围"


class ProcessingStatusEnum(Enum):
    """处理状态枚举"""
    待处理 = "pending"
    处理中 = "running"
    成功 = "success"
    失败 = "failed"
