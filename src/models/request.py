#!/src/models/request.py
# # -*- coding: utf-8 -*-

"""
输入请求模型
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from .enums import (
    BodyPartEnum,
    CheckTypeEnum,
    ScanThicknessEnum,
    CutModeEnum,
    PlaneTypeEnum,
    WindowPresetEnum,
)


@dataclass
class PlaneConfig:
    """单个重建状位的采样和显示配置。"""
    plane: PlaneTypeEnum
    slice_thickness: float
    slice_spacing: float
    window_presets: List[WindowPresetEnum]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlaneConfig':
        """
        从外部 JSON 字典解析三平面配置。

        CLI 和外部接口传入的是字符串值；这里统一转换成枚举，后续阶段
        就不需要反复处理字符串合法性。
        """
        return cls(
            plane=PlaneTypeEnum(data['plane']),
            slice_thickness=float(data['slice_thickness']),
            slice_spacing=float(data['slice_spacing']),
            window_presets=[
                WindowPresetEnum(value)
                for value in data.get('window_presets', [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        """输出 JSON 友好的结构，用于报告和调试。"""
        return {
            'plane': self.plane.value,
            'slice_thickness': self.slice_thickness,
            'slice_spacing': self.slice_spacing,
            'window_presets': [
                window.value
                for window in self.window_presets
            ],
        }


def default_plane_configs() -> List[PlaneConfig]:
    """
    默认输出三个状位的骨窗配置。

    旧命令没有传 plane_configs 时仍然能构造 Request；后续重建阶段
    可以直接读取这个列表，不需要再判断 None。
    """
    return [
        PlaneConfig(PlaneTypeEnum.横状位, 1.0, 1.0, [WindowPresetEnum.骨窗]),
        PlaneConfig(PlaneTypeEnum.矢状位, 1.0, 1.0, [WindowPresetEnum.骨窗]),
        PlaneConfig(PlaneTypeEnum.冠状位, 1.0, 1.0, [WindowPresetEnum.骨窗]),
    ]

@dataclass
class Request:
    """处理请求实体"""
    dicom_dir: str
    output_dir: str
    body_part: BodyPartEnum = BodyPartEnum.腰椎
    check_type: CheckTypeEnum = CheckTypeEnum.CT
    scan_thickness: ScanThicknessEnum = ScanThicknessEnum.默认
    cut_mode: CutModeEnum = CutModeEnum.整切
    plane_configs: List[PlaneConfig] = field(default_factory=default_plane_configs)

    # ===== 后续扩展字段（取消注释即可使用）=====
    # patient_name: str = ""
    # patient_id: str = ""
    # enable_correction: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dicom_dir': self.dicom_dir,
            'output_dir': self.output_dir,
            'body_part': self.body_part.value,
            'check_type': self.check_type.value,
            'scan_thickness': self.scan_thickness.value,
            'cut_mode': self.cut_mode.value,
            'plane_configs': [
                plane_config.to_dict()
                for plane_config in self.plane_configs
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Request':
        return cls(
            dicom_dir=data['dicom_dir'],
            output_dir=data['output_dir'],
            body_part=BodyPartEnum(data.get('body_part', data.get('target_part', BodyPartEnum.腰椎.value))),
            check_type=CheckTypeEnum(data.get('check_type', CheckTypeEnum.CT.value)),
            scan_thickness=ScanThicknessEnum(data.get('scan_thickness', ScanThicknessEnum.默认.value)),
            cut_mode=CutModeEnum(data.get('cut_mode', CutModeEnum.整切.value)),
            plane_configs=[
                PlaneConfig.from_dict(item)
                for item in data.get('plane_configs', [
                    plane_config.to_dict()
                    for plane_config in default_plane_configs()
                ])
            ]
        )

    @property
    def target_part(self) -> BodyPartEnum:
        """
        兼容旧代码中短暂使用过的 target_part 命名。

        当前主命名已经改回 body_part；保留这个属性可以让迁移期间的
        旧调用继续读取 request.target_part。
        """
        return self.body_part
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.dicom_dir:
            return False, "dicom_dir 不能为空"
        if not self.output_dir:
            return False, "output_dir 不能为空"

        if not self.plane_configs:
            return False, "plane_configs 不能为空"

        seen_planes = set()
        for plane_config in self.plane_configs:
            plane_value = plane_config.plane.value
            if plane_value in seen_planes:
                return False, f"plane_configs 中存在重复状位: {plane_value}"
            seen_planes.add(plane_value)

            if plane_config.slice_thickness <= 0:
                return False, f"{plane_value} 层厚必须大于 0"
            if plane_config.slice_spacing <= 0:
                return False, f"{plane_value} 层间距必须大于 0"
            if not plane_config.window_presets:
                return False, f"{plane_value} 窗宽窗位不能为空"

        return True, None
