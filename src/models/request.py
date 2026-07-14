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
    PlaneTypeEnum,
    WindowPresetEnum,
    CutLineExtentModeEnum,
)


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
    window_presets: List[WindowPresetEnum]
    line_extent_mode: CutLineExtentModeEnum = CutLineExtentModeEnum.整平面

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
            line_extent_mode=CutLineExtentModeEnum(
                data.get('line_extent_mode', CutLineExtentModeEnum.整平面.value)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        输出 JSON 友好的结构，用于报告和调试。

        枚举在内存里方便代码判断，但 JSON 报告和外部接口只能稳定消费字符串，
        所以这里统一转回中文枚举值。
        """
        return {
            'plane': self.plane.value,
            'slice_thickness': self.slice_thickness,
            'slice_spacing': self.slice_spacing,
            'window_presets': [
                window.value
                for window in self.window_presets
            ],
            'line_extent_mode': self.line_extent_mode.value,
        }


def default_plane_configs() -> List[PlaneConfig]:
    """
    默认输出当前已实现的横状位骨窗配置。

    未实现的矢状位和冠状位必须由调用方显式请求，并由能力预检给出清晰错误，
    避免默认参数宣称超过 planner 的真实能力。
    """
    return [
        PlaneConfig(PlaneTypeEnum.横状位, 1.0, 1.0, [WindowPresetEnum.骨窗]),
    ]

@dataclass
class Request:
    """处理请求实体"""
    # 原始 DICOM 序列目录。
    dicom_dir: str
    # 本次任务的输出根目录。
    output_dir: str
    # 目标部位决定模型匹配和定位配置。
    body_part: BodyPartEnum = BodyPartEnum.腰椎
    # 检查类型当前主要用于模型规则匹配。
    check_type: CheckTypeEnum = CheckTypeEnum.CT
    # 扫描厚度档位用于选择不同 nnUNet 模型。
    scan_thickness: ScanThicknessEnum = ScanThicknessEnum.默认
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
            'body_part': self.body_part.value,
            'check_type': self.check_type.value,
            'scan_thickness': self.scan_thickness.value,
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
        return cls(
            dicom_dir=data['dicom_dir'],
            output_dir=data['output_dir'],
            body_part=BodyPartEnum(data.get('body_part', data.get('target_part', BodyPartEnum.腰椎.value))),
            check_type=CheckTypeEnum(data.get('check_type', CheckTypeEnum.CT.value)),
            scan_thickness=ScanThicknessEnum(data.get('scan_thickness', ScanThicknessEnum.默认.value)),
            # 未提供 plane_configs 时走默认三平面配置，保证旧调用兼容。
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
        """
        校验请求参数是否足够执行流水线。

        这里先做结构层校验：路径不能为空、三平面配置不能冲突、层厚/层间距必须
        为正数。真实路径是否存在仍由各阶段按自己的读取逻辑处理。
        """
        if not self.dicom_dir:
            return False, "dicom_dir 不能为空"
        if not self.output_dir:
            return False, "output_dir 不能为空"

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

        return True, None
