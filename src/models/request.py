#!/src/models/request.py
# # -*- coding: utf-8 -*-

"""
输入请求模型
"""

import math
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from .enums import (
    SlicePartEnum,
    StudyTypeEnum,
    PlaneTypeEnum,
    SliceType,
)
from .window import (
    WindowPreset,
    get_window_preset,
    parse_window_presets,
)


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


def _parse_optional_positive_float(value: Any, name: str) -> float | None:
    """解析外部可选层厚或层间距；0 由 PlaneConfig 保留后再规范化。"""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name}必须是大于 0 的有限数值")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是大于 0 的有限数值") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name}必须是 0 或大于 0 的有限数值")
    return parsed


def _parse_optional_positive_integer(value: Any, name: str) -> int | None:
    """解析外部可选切片数量；0 由 PlaneConfig 保留后再规范化。"""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name}必须是 0 或大于 0 的整数")
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}必须是 0 或大于 0 的整数")
    return int(value)


def parse_fill_percent(value: Any) -> float | None:
    """解析 [0, 100] 的 JSON 数字；None/0 不缩放，后续平面上下裁空独立生效。"""

    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 100
        or not math.isfinite(value)
    ):
        raise ValueError("FillPercent 必须是 0 到 100 之间的有限数字；0 表示不缩放")
    return float(value)


@dataclass
class PlaneConfig:
    """单个重建状位的采样和显示配置。"""
    # 重建状位：横状位、矢状位或冠状位。
    plane: PlaneTypeEnum
    # 目标层厚，单位按业务输入约定使用 mm。
    slice_thickness: Optional[float]
    # 目标层间距，单位按业务输入约定使用 mm。
    slice_spacing: Optional[float]
    # 一个状位可以同时输出多套窗宽窗位图像，例如骨窗和纵隔窗。
    window_presets: List[str]
    # 整组目标包围框宽/高的最大占比；None/0 不缩放，平面成片仍可上下裁空。
    fill_percent: Optional[float] = None
    # 切片方式由调用方显式选择；普通直线重建默认使用 MPR。
    slice_type: SliceType = SliceType.MPR
    # 每个实际切割计划的切片数量；椎间隙会对每个间隙分别应用该数量。
    slice_image_count: Optional[int] = None
    # 以下字段只在第六阶段内部使用，不属于外部请求协议。
    config_id: str = field(default="", repr=False, compare=False)
    output_plane: str = field(default="", repr=False, compare=False)
    requested_slice_thickness: Optional[float] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    requested_slice_spacing: Optional[float] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    requested_slice_image_count: Optional[int] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """代码内直接构造配置时，也把三个采样参数的 0 规范化为未配置。"""

        self.requested_slice_thickness = self.slice_thickness
        self.requested_slice_spacing = self.slice_spacing
        self.requested_slice_image_count = self.slice_image_count
        if (
            not isinstance(self.slice_thickness, bool)
            and self.slice_thickness == 0
        ):
            self.slice_thickness = None
        if (
            not isinstance(self.slice_spacing, bool)
            and self.slice_spacing == 0
        ):
            self.slice_spacing = None
        if (
            not isinstance(self.slice_image_count, bool)
            and self.slice_image_count == 0
        ):
            self.slice_image_count = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlaneConfig':
        """
        从外部 JSON 字典解析三平面配置。

        CLI 和外部接口传入的是字符串值；这里统一转换成枚举，后续阶段
        就不需要反复处理字符串合法性。
        """
        thickness = data.get('SliceThickness')
        spacing = data.get('SliceSpacing')
        image_count = data.get('SliceImageCount')
        if 'line_extent_mode' in data:
            raise ValueError("line_extent_mode 已移除，请使用 FillPercent；省略时不缩放，平面成片仍裁上下空白")
        return cls(
            plane=PlaneTypeEnum(data['Plane']),
            slice_thickness=_parse_optional_positive_float(
                thickness,
                "层厚",
            ),
            slice_spacing=_parse_optional_positive_float(
                spacing,
                "层间距",
            ),
            window_presets=[str(value) for value in data.get('WindowPresets', [])],
            slice_type=SliceType(data.get('SliceType', SliceType.MPR.value)),
            fill_percent=parse_fill_percent(data.get('FillPercent')),
            slice_image_count=_parse_optional_positive_integer(
                image_count,
                "数量",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        输出 JSON 友好的结构，用于报告和调试。

        枚举在内存里方便代码判断，但 JSON 报告和外部接口只能稳定消费字符串，
        所以这里统一转回中文协议值。
        """
        data = {
            'Plane': self.plane.value,
            'WindowPresets': list(self.window_presets),
            'SliceType': self.slice_type.value,
        }
        if self.slice_thickness is not None:
            data['SliceThickness'] = self.slice_thickness
        if self.slice_spacing is not None:
            data['SliceSpacing'] = self.slice_spacing
        if self.slice_image_count is not None:
            data['SliceImageCount'] = self.slice_image_count
        if self.fill_percent is not None:
            # 整数比例以 JSON 整数回显，兼容调用方的整数属性；真实小数不截断。
            percent = parse_fill_percent(self.fill_percent)
            data['FillPercent'] = int(percent) if percent.is_integer() else percent
        return data


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
    # 切割部位决定模型匹配和定位配置。
    slice_part: SlicePartEnum = SlicePartEnum.腰椎
    # 检查类型当前主要用于模型规则匹配。
    study_type: StudyTypeEnum = StudyTypeEnum.CT
    # 当前原始扫描层厚，单位 mm；模型选择按配置区间匹配。
    scan_thickness: float = 1.0
    # 三平面重建参数；调用方不传时使用 default_plane_configs。
    plane_configs: List[PlaneConfig] = field(default_factory=default_plane_configs)
    # 本次任务的窗宽窗位覆盖列表；同名项优先于默认配置。
    window_presets: List[WindowPreset] = field(default_factory=list)
    # 持久化缓存根目录；第一阶段放在 nifti/，第二阶段按模型 ID 分目录。
    cache_dir: str = ""

    def __post_init__(self) -> None:
        # 兼容代码内直接传入字典的调用方，同时确保后续阶段拿到统一实体。
        self.window_presets = parse_window_presets(self.window_presets)
        if self.cache_dir is None:
            self.cache_dir = ""

    # ===== 后续扩展字段（取消注释即可使用）=====
    # patient_name: str = ""
    # patient_id: str = ""
    # enable_correction: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将 Request 转成 JSON 友好的字典。

        报告阶段和后续调试都走这个方法；传入的窗宽窗位覆盖列表只在非空时
        写入，避免改变未传参数时的既有请求结构。
        """
        data = {
            'dicom_dir': self.dicom_dir,
            'output_dir': self.output_dir,
            'SlicePart': self.slice_part.value,
            'StudyType': self.study_type.value,
            'ScanThickness': self.scan_thickness,
            'plane_configs': [
                plane_config.to_dict()
                for plane_config in self.plane_configs
            ]
        }
        if self.window_presets:
            data['window_presets'] = [
                preset.to_dict() for preset in self.window_presets
            ]
        if self.cache_dir:
            data['cache_dir'] = str(self.cache_dir)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Request':
        """
        从外部字典创建请求对象。

        外部传入的切割部位和三平面配置都是字符串；这里统一转成枚举和
        PlaneConfig，让后续 Pipeline 阶段处理的是强约束结构。顶层
        ``window_presets``（或兼容的 ``WindowPresets``）用于本次任务覆盖默认窗位。
        """
        removed_fields = {
            'StudyPart',
            'body_part',
            'check_type',
            'slice_part',
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

        custom_window_presets = parse_window_presets(
            data.get('window_presets', data.get('WindowPresets'))
        )
        return cls(
            dicom_dir=data['dicom_dir'],
            output_dir=data['output_dir'],
            slice_part=SlicePartEnum(
                data.get('SlicePart', SlicePartEnum.腰椎.value)
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
            ],
            window_presets=custom_window_presets,
            cache_dir=data.get(
                'cache_dir',
                data.get('CacheDir', ''),
            ),
        )

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        校验请求参数是否足够执行流水线。

        这里先做结构层校验：路径不能为空，每项状位配置必须提供两个或三个有效
        采样参数，并配置至少一个窗位。真实路径是否存在仍由各阶段读取时处理。
        """
        if not self.dicom_dir:
            return False, "dicom_dir 不能为空"
        if not self.output_dir:
            return False, "output_dir 不能为空"
        if not isinstance(self.cache_dir, (str, Path)):
            return False, "cache_dir 必须是路径字符串"
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

        for plane_config in self.plane_configs:
            try:
                parse_fill_percent(plane_config.fill_percent)
            except ValueError as exc:
                return False, str(exc)
            plane_value = plane_config.plane.value
            sampling_values = (
                plane_config.slice_thickness,
                plane_config.slice_spacing,
                plane_config.slice_image_count,
            )
            provided_count = sum(value is not None for value in sampling_values)
            if provided_count not in (2, 3):
                return False, (
                    f"{plane_value} 层厚、层间距、数量必须传入其中两个或三个"
                )

            for name, value in (
                ("层厚", plane_config.slice_thickness),
                ("层间距", plane_config.slice_spacing),
            ):
                if value is None:
                    continue
                if isinstance(value, bool):
                    return False, f"{plane_value} {name}必须是大于 0 的有限数值"
                try:
                    parsed_value = float(value)
                except (TypeError, ValueError):
                    return False, f"{plane_value} {name}必须是大于 0 的有限数值"
                if not math.isfinite(parsed_value) or parsed_value <= 0:
                    return False, f"{plane_value} {name}必须是大于 0 的有限数值"

            image_count = plane_config.slice_image_count
            if (
                image_count is not None
                and (
                    isinstance(image_count, bool)
                    or not isinstance(image_count, int)
                    or image_count <= 0
                )
            ):
                return False, f"{plane_value} 数量必须是大于 0 的整数"
            # 没有窗宽窗位就无法决定最终图像显示版本。
            if not plane_config.window_presets:
                return False, f"{plane_value} 窗宽窗位不能为空"
            for window_name in plane_config.window_presets:
                try:
                    get_window_preset(window_name, self.window_presets)
                except ValueError as exc:
                    return False, str(exc)

        return True, None
