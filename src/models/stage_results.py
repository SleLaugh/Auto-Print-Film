"""
六阶段之间稳定交接的顶层结果契约。

旧阶段曾直接传递任意字典。本模块先约束每个阶段结果的顶层协议和核心路径字段，复杂
医学结构继续保留在 payload 中，便于渐进迁移而不一次性改写算法。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List


@dataclass
class StageResult:
    """所有阶段结果共享的警告、产物、诊断信息和序列化行为。"""

    warnings: List[str] = field(default_factory=list)
    """不导致阶段失败的中文警告列表。"""
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    """阶段产生的已登记产物描述。"""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    """不属于稳定业务协议的调试诊断数据。"""

    stage_name: ClassVar[str] = ""
    """结果所属阶段的固定名称，子类必须覆盖。"""
    schema_version: ClassVar[str] = "stage_result_v1"
    """阶段结果协议版本，用于 manifest 和兼容迁移。"""

    def to_dict(self) -> Dict[str, Any]:
        """转换成 JSON 友好的稳定顶层字典。"""
        data = asdict(self)
        return {
            "schema_version": self.schema_version,
            "stage_name": self.stage_name,
            **data,
        }


@dataclass
class PreprocessingResult(StageResult):
    """预处理阶段输出的 NIfTI 路径和源 DICOM 元数据。"""
    stage_name: ClassVar[str] = "preprocessing"
    schema_version: ClassVar[str] = "preprocessing_v1"

    nifti_path: str = ""
    """提供给 nnUNet 的原始 NIfTI 文件路径。"""
    metadata_path: str = ""
    """调试模式下写出的 DICOM 几何元数据路径。"""
    source_dicom_metadata: Dict[str, str] = field(default_factory=dict)
    """第五阶段生成派生 DICOM 时允许继承的脱敏标签集合。"""


@dataclass
class SegmentationResult(StageResult):
    """分割阶段输出的 mask 路径、输入路径和实际模型信息。"""
    stage_name: ClassVar[str] = "segmentation"
    schema_version: ClassVar[str] = "segmentation_v1"

    seg_path: str = ""
    """统一命名后的分割 NIfTI 文件路径。"""
    input_nifti: str = ""
    """本次推理实际读取的 NIfTI 路径。"""
    model: Dict[str, Any] = field(default_factory=dict)
    """实际命中的模型 ID、fold、configuration 等信息。"""


@dataclass
class LocalizationStageResult(StageResult):
    """定位阶段结果；复杂坐标和回正诊断暂时保留在 payload。"""
    stage_name: ClassVar[str] = "localization"
    schema_version: ClassVar[str] = "localization_v1"

    payload: Dict[str, Any] = field(default_factory=dict)
    """保持现有定位字典字段不变的兼容载荷。"""

    def to_dict(self) -> Dict[str, Any]:
        """把 payload 展平到旧字典顶层，保证下游旧代码继续读取。"""
        data = super().to_dict()
        payload = data.pop("payload")
        data.update(payload)
        return data


@dataclass
class MappingStageResult(LocalizationStageResult):
    """规划阶段结果，payload 内主要包含 SlicePlan 列表和诊断信息。"""
    stage_name: ClassVar[str] = "mapping"
    schema_version: ClassVar[str] = "mapping_v1"


@dataclass
class ReconstructionStageResult(LocalizationStageResult):
    """重建阶段结果，payload 内包含输出序列、张数和状位汇总。"""
    stage_name: ClassVar[str] = "reconstruction"
    schema_version: ClassVar[str] = "reconstruction_v1"


@dataclass
class ReportingStageResult(LocalizationStageResult):
    """报告阶段结果，payload 内包含报告路径、校验和图像数量。"""
    stage_name: ClassVar[str] = "reporting"
    schema_version: ClassVar[str] = "reporting_v1"
