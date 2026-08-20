#!/src/planning/sampling.py
# -*- coding: utf-8 -*-

"""切片层厚、层间距、数量和物理范围的统一计算。"""

from dataclasses import dataclass
import math
from typing import Any


RANGE_TOLERANCE_MM = 1e-3


@dataclass(frozen=True)
class ResolvedSampling:
    """第六阶段根据有效范围计算出的实际采样参数。"""

    slice_thickness: float
    slice_spacing: float
    slice_image_count: int
    effective_range_mm: float
    total_range_mm: float
    start_extension_mm: float
    end_extension_mm: float
    adjusted: bool = False
    adjustment_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """输出调试报告使用的完整采样计算结果。"""

        return {
            "slice_thickness": self.slice_thickness,
            "slice_spacing": self.slice_spacing,
            "slice_image_count": self.slice_image_count,
            "effective_range_mm": self.effective_range_mm,
            "total_range_mm": self.total_range_mm,
            "start_extension_mm": self.start_extension_mm,
            "end_extension_mm": self.end_extension_mm,
            "adjusted": self.adjusted,
            "adjustment_reason": self.adjustment_reason,
        }


def resolve_sampling(
    effective_range_mm: float,
    slice_thickness: float | None,
    slice_spacing: float | None,
    slice_image_count: int | None,
) -> ResolvedSampling:
    """三个采样参数传入两个或三个；值为 0 时按未配置处理。"""

    effective_range = _non_negative_finite_float(
        effective_range_mm,
        "有效范围",
    )
    thickness = _optional_positive_float(slice_thickness, "层厚")
    spacing = _optional_positive_float(slice_spacing, "层间距")
    image_count = _optional_positive_integer(slice_image_count, "数量")

    provided_count = sum(
        value is not None
        for value in (thickness, spacing, image_count)
    )
    if provided_count not in (2, 3):
        raise ValueError("层厚、层间距、数量必须传入其中两个或三个")

    adjusted = False
    adjustment_reason = ""

    if thickness is not None and spacing is not None and image_count is None:
        remaining_range = max(0.0, effective_range - thickness)
        if remaining_range <= RANGE_TOLERANCE_MM:
            image_count = 1
        else:
            image_count = (
                int(math.ceil(
                    max(0.0, remaining_range - RANGE_TOLERANCE_MM)
                    / spacing
                ))
                + 1
            )
    elif thickness is not None and image_count is not None and spacing is None:
        if image_count == 1:
            raise ValueError("数量为 1 时无法仅根据层厚计算层间距")
        spacing = (effective_range - thickness) / (image_count - 1)
        if spacing <= RANGE_TOLERANCE_MM:
            raise ValueError(
                "当前有效范围、层厚和数量无法计算出大于 0 的层间距"
            )
    elif spacing is not None and image_count is not None and thickness is None:
        thickness = effective_range - (image_count - 1) * spacing
        if thickness <= RANGE_TOLERANCE_MM:
            raise ValueError(
                "当前有效范围、层间距和数量无法计算出大于 0 的层厚"
            )

    if thickness is None or spacing is None or image_count is None:
        raise ValueError("层厚、层间距、数量计算失败")

    total_range = thickness + (image_count - 1) * spacing
    if total_range < effective_range - RANGE_TOLERANCE_MM:
        # 三个参数同时传入但总范围不足时，数量优先，层厚和层间距均分有效范围。
        if provided_count != 3:
            raise ValueError("计算后的总范围小于有效范围")
        equal_value = effective_range / image_count
        if equal_value <= RANGE_TOLERANCE_MM:
            raise ValueError("当前有效范围和数量无法计算出有效的层厚、层间距")
        thickness = equal_value
        spacing = equal_value
        total_range = effective_range
        adjusted = True
        adjustment_reason = (
            "传入参数计算出的总范围小于有效范围，已保持数量并将层厚、"
            "层间距调整为有效范围除以数量"
        )

    overflow = max(0.0, total_range - effective_range)
    extension = overflow / 2.0
    return ResolvedSampling(
        slice_thickness=float(thickness),
        slice_spacing=float(spacing),
        slice_image_count=int(image_count),
        effective_range_mm=float(effective_range),
        total_range_mm=float(total_range),
        start_extension_mm=float(extension),
        end_extension_mm=float(extension),
        adjusted=adjusted,
        adjustment_reason=adjustment_reason,
    )


def _optional_positive_float(value: Any, name: str) -> float | None:
    """校验可选的正有限小数，并把 0 规范化为未配置。"""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name}必须是大于 0 的有限数值")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是大于 0 的有限数值") from error
    if parsed == 0:
        return None
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name}必须是 0 或大于 0 的有限数值")
    return parsed


def _optional_positive_integer(value: Any, name: str) -> int | None:
    """校验可选的正整数，并把 0 规范化为未配置。"""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name}必须是 0 或大于 0 的整数")
    if value == 0:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}必须是 0 或大于 0 的整数")
    return int(value)


def _non_negative_finite_float(value: Any, name: str) -> float:
    """校验必填的非负有限小数。"""

    if isinstance(value, bool):
        raise ValueError(f"{name}必须是非负有限数值")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是非负有限数值") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name}必须是非负有限数值")
    return parsed
