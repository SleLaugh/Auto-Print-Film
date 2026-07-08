"""
枚举定义模块
定义系统中使用的所有枚举类型
"""

from enum import Enum


class BodyPartEnum(Enum):
    """部位枚举"""
    头骨 = "头骨"
    颈椎 = "颈椎"
    胸椎 = "胸椎"
    腰椎 = "腰椎"
    肺部 = "肺部"
    骶骨 = "骶骨"


class CheckTypeEnum(Enum):
    """检查类型枚举"""
    CT = "CT"
    MR = "MR"


class ScanThicknessEnum(Enum):
    """扫描厚度档位枚举"""
    默认 = "default"
    三毫米 = "3mm"
    六毫米 = "6mm"


class CutModeEnum(Enum):
    """切割方式枚举"""
    整切 = "whole_range"
    椎间切 = "interspace"


class PositioningModeEnum(Enum):
    """定位模式枚举"""
    单区间 = "single"          # 单区间模式（大多数部位使用）
    多区间 = "multi"           # 多区间模式（腰椎间盘突出需要多区间划分）


class PlaneTypeEnum(Enum):
    """重建平面类型枚举"""
    # 横状位、矢状位、冠状位会直接对应后续输出目录和 MPR 采样方向。
    横状位 = "横状位"
    矢状位 = "矢状位"
    冠状位 = "冠状位"


class WindowPresetEnum(Enum):
    """窗宽窗位预设枚举"""
    # 这里只保存业务预设名，不在枚举里写死 WindowCenter / WindowWidth 数值；
    # 具体窗宽窗位数值后续由重建/DICOM 输出阶段统一映射。
    骨窗 = "骨窗"
    肺窗 = "肺窗"
    纵隔窗 = "纵隔窗"


class ProcessingStatusEnum(Enum):
    """处理状态枚举"""
    待处理 = "pending"
    处理中 = "running"
    成功 = "success"
    失败 = "failed"
