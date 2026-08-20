#!/src/models/reconstruction.py
# -*- coding: utf-8 -*-

"""
重建规划模型

这些实体是第六阶段输出给第七阶段的稳定契约。第六阶段负责把切割路径展开成
具体切片清单；第七阶段只按这些切片逐张采样，不再反推切多少张或每张在哪里。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Point3D:
    """物理坐标点或方向向量，坐标顺序为 x/y/z，单位通常是 mm。"""

    x: float
    y: float
    z: float

    def to_dict(self) -> Dict[str, float]:
        """将物理点或方向向量序列化为 x/y/z 字典。"""

        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class PlannedSlice:
    """第六阶段已经规划好的一张待切图。"""

    index: int
    center: Point3D
    normal: Point3D
    row_dir: Point3D
    col_dir: Point3D
    offset_mm: float | None = None
    """CPR 曲面相对中心曲线的左右偏移；普通平面切片为空。"""

    def to_dict(self) -> Dict[str, Any]:
        """序列化单张计划切片的中心、法向和画面方向。"""

        data = {
            "index": self.index,
            "center": self.center.to_dict(),
            "normal": self.normal.to_dict(),
            "row_dir": self.row_dir.to_dict(),
            "col_dir": self.col_dir.to_dict(),
        }
        if self.offset_mm is not None:
            data["offset_mm"] = self.offset_mm
        return data


@dataclass
class ExtentMM:
    """单张输出图像覆盖的物理宽高范围。"""

    width: float
    height: float

    def to_dict(self) -> Dict[str, float]:
        """序列化单张切片覆盖的物理宽度和高度。"""

        return {"width": self.width, "height": self.height}


@dataclass
class SlicePlan:
    """一组已经展开好的切片规划，供第七阶段直接执行。"""

    plan_id: str
    body_part: str
    slice_type: str
    plane: str
    group_name: str
    geometry_type: str
    range_start: Point3D
    range_end: Point3D
    path_points: List[Point3D]
    slices: List[PlannedSlice]
    extent_mm: ExtentMM
    output_spacing_mm: Point3D
    slice_thickness: float
    slice_spacing: float
    window_presets: List[str]
    source_labels: List[str]
    curve_points: List[Point3D] | None = None
    """CPR 中心曲线物理点；仅 curved_stack 使用。"""
    curve_spacing_mm: float | None = None
    """CPR 沿曲线弧长方向的输出采样间距。"""
    offset_axis: Point3D | None = None
    """CPR 左右偏移方向单位向量。"""
    offsets_mm: List[float] | None = None
    """CPR 每张图相对中心曲线的左右偏移。"""
    cpr_range_mode: str | None = None
    """CPR offset 覆盖范围来源；full_label 为完整标注，body_prism 为主椎体。"""
    target_start: Point3D | None = None
    target_end: Point3D | None = None
    localizer_reference: Point3D | None = None
    """定位图选择原始参考层时使用的物理点；不参与正式切片重采样。"""
    localizer_plane: str | None = None
    """定位图使用的原始正交状位；为空时自动选择，不改变正式切片状位。"""
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化第七阶段可直接执行的一组完整切片计划。"""

        data = {
            "plan_id": self.plan_id,
            "body_part": self.body_part,
            "slice_type": self.slice_type,
            "plane": self.plane,
            "group_name": self.group_name,
            "geometry_type": self.geometry_type,
            "range_start": self.range_start.to_dict(),
            "range_end": self.range_end.to_dict(),
            "path_points": [point.to_dict() for point in self.path_points],
            "slices": [planned_slice.to_dict() for planned_slice in self.slices],
            "extent_mm": self.extent_mm.to_dict(),
            "output_spacing_mm": self.output_spacing_mm.to_dict(),
            "slice_thickness": self.slice_thickness,
            "slice_spacing": self.slice_spacing,
            "window_presets": self.window_presets,
            "source_labels": self.source_labels,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }
        if self.target_start is not None:
            data["target_start"] = self.target_start.to_dict()
        if self.target_end is not None:
            data["target_end"] = self.target_end.to_dict()
        if self.localizer_reference is not None:
            data["localizer_reference"] = self.localizer_reference.to_dict()
        if self.localizer_plane is not None:
            data["localizer_plane"] = self.localizer_plane
        if self.curve_points is not None:
            data["curve_points"] = [point.to_dict() for point in self.curve_points]
        if self.curve_spacing_mm is not None:
            data["curve_spacing_mm"] = self.curve_spacing_mm
        if self.offset_axis is not None:
            data["offset_axis"] = self.offset_axis.to_dict()
        if self.offsets_mm is not None:
            data["offsets_mm"] = list(self.offsets_mm)
        if self.cpr_range_mode is not None:
            data["cpr_range_mode"] = self.cpr_range_mode
        return data
