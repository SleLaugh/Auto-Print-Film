#!/src/models/__init__.py
# # -*- coding: utf-8 -*-

"""
数据模型包
"""

# 对外暴露请求模型，方便入口、测试和阶段代码统一从 src.models 导入。
from .request import Request, PlaneConfig
from .dicom_metadata import DicomMetadata
from .localization import (
    VoxelPoint,
    VoxelBBox,
    LabelTarget,
    LabelCoordinate,
    NiftiImageInfo,
    LocalizationResult,
)
from .enums import (
    BodyPartEnum,
    CheckTypeEnum,
    ScanThicknessEnum,
    CutModeEnum,
    PositioningModeEnum,
    PlaneTypeEnum,
    WindowPresetEnum,
    ProcessingStatusEnum
)

__all__ = [
    'Request',
    'PlaneConfig',
    'DicomMetadata',
    'VoxelPoint',
    'VoxelBBox',
    'LabelTarget',
    'LabelCoordinate',
    'NiftiImageInfo',
    'LocalizationResult',
    'BodyPartEnum',
    'CheckTypeEnum',
    'ScanThicknessEnum',
    'CutModeEnum',
    'PositioningModeEnum',
    'PlaneTypeEnum',
    'WindowPresetEnum',
    'ProcessingStatusEnum'
]
