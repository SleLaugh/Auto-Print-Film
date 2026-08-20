#!/src/models/__init__.py
# # -*- coding: utf-8 -*-

"""
数据模型包
"""

# 对外暴露请求模型，方便入口、测试和阶段代码统一从 src.models 导入。
from .request import Request, PlaneConfig
from .dicom_metadata import DicomMetadata
from .localization import (
    CutConfig,
    VoxelPoint,
    VoxelBBox,
    LabelTarget,
    LabelCoordinate,
    NiftiImageInfo,
    LocalizationResult,
)
from .reconstruction import (
    Point3D,
    PlannedSlice,
    ExtentMM,
    SlicePlan,
)
from .anatomy import (
    AxisBounds,
    AnatomicalBounds,
    AnatomicalFrame,
    AnatomicalGeometryResult,
)
from .window import WindowPreset, get_window_preset
from .stage_results import (
    StageResult,
    PreprocessingResult,
    SegmentationResult,
    SpatialCorrectionStageResult,
    LocalizationStageResult,
    AnatomicalAnalysisStageResult,
    PlanningStageResult,
    ReconstructionStageResult,
    ReportingStageResult,
)
from .enums import (
    BodyPartEnum,
    CheckTypeEnum,
    ScanThicknessEnum,
    SliceType,
    PlaneTypeEnum,
    ProcessingStatusEnum
)

__all__ = [
    'Request',
    'PlaneConfig',
    'DicomMetadata',
    'CutConfig',
    'VoxelPoint',
    'VoxelBBox',
    'LabelTarget',
    'LabelCoordinate',
    'NiftiImageInfo',
    'LocalizationResult',
    'Point3D',
    'PlannedSlice',
    'ExtentMM',
    'SlicePlan',
    'AxisBounds',
    'AnatomicalBounds',
    'AnatomicalFrame',
    'AnatomicalGeometryResult',
    'WindowPreset',
    'get_window_preset',
    'BodyPartEnum',
    'CheckTypeEnum',
    'ScanThicknessEnum',
    'SliceType',
    'PlaneTypeEnum',
    'ProcessingStatusEnum',
    'StageResult',
    'PreprocessingResult',
    'SegmentationResult',
    'SpatialCorrectionStageResult',
    'LocalizationStageResult',
    'AnatomicalAnalysisStageResult',
    'PlanningStageResult',
    'ReconstructionStageResult',
    'ReportingStageResult',
]
