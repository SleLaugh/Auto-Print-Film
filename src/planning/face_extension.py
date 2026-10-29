# -*- coding: utf-8 -*-

"""统一患者解剖方向的六面范围扩展。"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict

from src.anatomy.orientation import patient_reference_axes
from src.config import resolve_face_extension as resolve_config_face_extension
from src.models import AnatomicalBounds, AxisBounds, Point3D
from src.models.enums import PlaneTypeEnum
from src.planning.geometry import image_direction_matrix


@dataclass(frozen=True)
class FaceExtension:
    """沿患者上、下、前、后、左、右方向增加的距离。"""

    top_mm: float = 0.0
    bottom_mm: float = 0.0
    front_mm: float = 0.0
    back_mm: float = 0.0
    left_mm: float = 0.0
    right_mm: float = 0.0

    @property
    def is_zero(self) -> bool:
        """是否没有配置任何六面扩展。"""

        return not any(
            value > 0.0
            for value in (
                self.top_mm,
                self.bottom_mm,
                self.front_mm,
                self.back_mm,
                self.left_mm,
                self.right_mm,
            )
        )


def resolve_face_extension(
    localization_result: Dict[str, Any],
    slice_type: str,
    plane: PlaneTypeEnum,
) -> FaceExtension | None:
    """读取当前状位/切法规则的六面扩展，未配置时返回 ``None``。"""

    slice_rules = localization_result.get("slice_rules")
    if not isinstance(slice_rules, dict):
        return None
    plane_rules = slice_rules.get(plane.value)
    if not isinstance(plane_rules, dict):
        return None
    slice_type_value = str(getattr(slice_type, "value", slice_type))
    rule = plane_rules.get(slice_type_value)
    if not isinstance(rule, dict):
        return None
    values = resolve_config_face_extension(rule)
    if values is None:
        return None
    extension = FaceExtension(**values)
    return None if extension.is_zero else extension


def expand_anatomical_bounds(
    bounds: AnatomicalBounds,
    extension: FaceExtension,
) -> AnatomicalBounds:
    """按六面扩展已有解剖范围，保留原范围作为中心和定位依据。"""

    return AnatomicalBounds(
        left_right=AxisBounds(
            bounds.left_right.minimum - extension.right_mm,
            bounds.left_right.maximum + extension.left_mm,
        ),
        anterior_posterior=AxisBounds(
            bounds.anterior_posterior.minimum - extension.front_mm,
            bounds.anterior_posterior.maximum + extension.back_mm,
        ),
        superior_inferior=AxisBounds(
            bounds.superior_inferior.minimum - extension.bottom_mm,
            bounds.superior_inferior.maximum + extension.top_mm,
        ),
    )


def expand_physical_points(
    points: list[Point3D],
    image_info: Dict[str, Any],
    extension: FaceExtension,
) -> list[Point3D]:
    """把物理点集合按患者六个方向扩展，返回扩展后的角点集合。"""

    axes = patient_reference_axes(image_direction_matrix(image_info))
    left_right_axis = axes[:, 0]
    anterior_posterior_axis = axes[:, 1]
    superior_inferior_axis = axes[:, 2]
    expanded = []
    for point in points:
        base = [point.x, point.y, point.z]
        for left_right, anterior_posterior, superior_inferior in product(
            (-extension.right_mm, extension.left_mm),
            (-extension.front_mm, extension.back_mm),
            (-extension.bottom_mm, extension.top_mm),
        ):
            value = Point3D(
                base[0]
                + left_right_axis[0] * left_right
                + anterior_posterior_axis[0] * anterior_posterior
                + superior_inferior_axis[0] * superior_inferior,
                base[1]
                + left_right_axis[1] * left_right
                + anterior_posterior_axis[1] * anterior_posterior
                + superior_inferior_axis[1] * superior_inferior,
                base[2]
                + left_right_axis[2] * left_right
                + anterior_posterior_axis[2] * anterior_posterior
                + superior_inferior_axis[2] * superior_inferior,
            )
            expanded.append(value)
    return expanded
