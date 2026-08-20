#!/src/models/request.py
# # -*- coding: utf-8 -*-

"""
输入请求模型
"""

import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from .enums import (
    StudyPartEnum,
    StudyTypeEnum,
    PlaneTypeEnum,
    CutLineExtentModeEnum,
    SliceType,
)
from .window import get_window_preset


def _parse_scan_thickness(value: Any) -> float:
    """解析外部扫描层厚，并拒绝布尔值、负数及非有限数。"""

    if isinstance(value, bool):
        raise ValueError("ScanThickness 必须是非负有限数值")
    try:
        thickness = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("ScanThickness 必须是非负有限数值") from error
    if not math.isfinite(thickness) or thickness < 0:
        raise ValueError("ScanThickness 必须是非负有限数值")
    return thickness


@dataclass
class PlaneConfig:
    """单个重建状位的采样和显示配置。"""
    # 重建状位：横状位、矢状位或冠状位。
    plane: PlaneTypeEnum
    # 目标层厚，单位按业务输入约定使用 mm。
    slice_thickness: float
    # 目标层间距，单位按业务输入约定使用 mm。
    slice_spacing: float
    # 一个状位可以同时输出多套窗宽窗位图像，例如骨窗和纵隔窗。
    window_presets: List[str]
    line_extent_mode: CutLineExtentModeEnum = CutLineExtentModeEnum.整平面
    # 切片方式由调用方显式选择；普通直线重建默认使用 MPR。
    slice_type: SliceType = SliceType.MPR

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlaneConfig':
        """
        从外部 JSON 字典解析三平面配置。

        CLI 和外部接口传入的是字符串值；这里统一转换成枚举，后续阶段
        就不需要反复处理字符串合法性。
        """
        return cls(
            plane=PlaneTypeEnum(data['Plane']),
            slice_thickness=float(data['SliceThickness']),
            slice_spacing=float(data['SliceSpacing']),
            window_presets=[str(value) for value in data.get('WindowPresets', [])],
            slice_type=SliceType(data.get('SliceType', SliceType.MPR.value)),
            line_extent_mode=CutLineExtentModeEnum(
                data.get('line_extent_mode', CutLineExtentModeEnum.整平面.value)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        输出 JSON 友好的结构，用于报告和调试。

        枚举在内存里方便代码判断，但 JSON 报告和外部接口只能稳定消费字符串，
        所以这里统一转回中文协议值。
        """
        return {
            'Plane': self.plane.value,
            'SliceThickness': self.slice_thickness,
            'SliceSpacing': self.slice_spacing,
            'WindowPresets': list(self.window_presets),
            'SliceType': self.slice_type.value,
            'line_extent_mode': self.line_extent_mode.value,
        }


def default_plane_configs() -> List[PlaneConfig]:
    """
    默认输出当前已实现的横状位骨窗配置。

    矢状位和冠状位已经属于正式能力，但仍需调用方显式请求，避免默认参数
    在未确认业务需求时一次性输出三套重建。
    """
    return [
        PlaneConfig(PlaneTypeEnum.横状位, 1.0, 1.0, ["骨窗"]),
    ]

@dataclass
class Request:
    """处理请求实体"""
    # 原始 DICOM 序列目录。
    dicom_dir: str
    # 本次任务的输出根目录。
    output_dir: str
    # 目标部位决定模型匹配和定位配置。
    study_part: StudyPartEnum = StudyPartEnum.腰椎
    # 检查类型当前主要用于模型规则匹配。
    study_type: StudyTypeEnum = StudyTypeEnum.CT
    # 当前原始扫描层厚，单位 mm；模型选择按配置区间匹配。
    scan_thickness: float = 1.0
    # 三平面重建参数；调用方不传时使用 default_plane_configs。
    plane_configs: List[PlaneConfig] = field(default_factory=default_plane_configs)

    # ===== 后续扩展字段（取消注释即可使用）=====
    # patient_name: str = ""
    # patient_id: str = ""
    # enable_correction: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将 Request 转成 JSON 友好的字典。

        报告阶段和后续调试都走这个方法，避免新增请求字段后只在部分地方遗漏。
        """
        return {
            'dicom_dir': self.dicom_dir,
            'output_dir': self.output_dir,
            'StudyPart': self.study_part.value,
            'StudyType': self.study_type.value,
            'ScanThickness': self.scan_thickness,
            'plane_configs': [
                plane_config.to_dict()
                for plane_config in self.plane_configs
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Request':
        """
        从外部字典创建请求对象。

        外部传入的部位和三平面配置都是字符串；这里统一转成枚举和
        PlaneConfig，让后续 Pipeline 阶段处理的是强约束结构。
        """
        removed_fields = {
            'body_part',
            'check_type',
            'study_part',
            'study_type',
            'target_part',
        }
        present_removed_fields = sorted(removed_fields.intersection(data))
        if present_removed_fields:
            raise ValueError(
                "外部请求包含已移除字段: "
                + ", ".join(present_removed_fields)
            )
        if 'ScanThickness' not in data:
            raise ValueError("外部请求缺少必填字段: ScanThickness")

        return cls(
            dicom_dir=data['dicom_dir'],
            output_dir=data['output_dir'],
            study_part=StudyPartEnum(
                data.get('StudyPart', StudyPartEnum.腰椎.value)
            ),
            study_type=StudyTypeEnum(
                data.get('StudyType', StudyTypeEnum.CT.value)
            ),
            scan_thickness=_parse_scan_thickness(data['ScanThickness']),
            # 未提供 plane_configs 时使用默认横状位骨窗配置。
            plane_configs=[
                PlaneConfig.from_dict(item)
                for item in data.get('plane_configs', [
                    plane_config.to_dict()
                    for plane_config in default_plane_configs()
                ])
            ]
        )

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        校验请求参数是否足够执行流水线。

        这里先做结构层校验：路径不能为空、三平面配置不能冲突、层厚/层间距必须
        为正数。真实路径是否存在仍由各阶段按自己的读取逻辑处理。
        """
        if not self.dicom_dir:
            return False, "dicom_dir 不能为空"
        if not self.output_dir:
            return False, "output_dir 不能为空"

        if isinstance(self.scan_thickness, bool):
            return False, "ScanThickness 必须是非负有限数值"
        try:
            scan_thickness = float(self.scan_thickness)
        except (TypeError, ValueError):
            return False, "ScanThickness 必须是非负有限数值"
        if not math.isfinite(scan_thickness) or scan_thickness < 0:
            return False, "ScanThickness 必须是非负有限数值"

        if not self.plane_configs:
            return False, "plane_configs 不能为空"

        seen_planes = set()
        for plane_config in self.plane_configs:
            plane_value = plane_config.plane.value
            # 同一个状位重复出现会导致后续输出目录和采样参数语义不明确。
            if plane_value in seen_planes:
                return False, f"plane_configs 中存在重复状位: {plane_value}"
            seen_planes.add(plane_value)

            # 层厚和层间距是重采样的核心输入，0 或负数无法生成有效切片序列。
            if plane_config.slice_thickness <= 0:
                return False, f"{plane_value} 层厚必须大于 0"
            if plane_config.slice_spacing <= 0:
                return False, f"{plane_value} 层间距必须大于 0"
            # 没有窗宽窗位就无法决定最终图像显示版本。
            if not plane_config.window_presets:
                return False, f"{plane_value} 窗宽窗位不能为空"
            for window_name in plane_config.window_presets:
                try:
                    get_window_preset(window_name)
                except ValueError as exc:
                    return False, str(exc)

        return True, None
