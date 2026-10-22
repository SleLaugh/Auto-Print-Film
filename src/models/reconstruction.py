#!/src/models/reconstruction.py
# -*- coding: utf-8 -*-

"""
重建规划模型

这些实体是第六阶段输出给第七阶段的稳定契约。第六阶段负责把切割路径展开成
具体切片清单；第七阶段保持切面、层厚、层间距和张数，仅按整组目标范围
调整面内中心与取景比例，再执行采样。
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
    """CPR 曲面沿当前 offset 轴相对中心曲线的偏移；普通平面切片为空。"""

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
    slice_part: str
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
    """CPR 真实三维中心曲线物理点；仅 curved_stack 使用。"""
    curve_spacing_mm: float | None = None
    """CPR 沿采样曲线（矢状位为冠状投影）弧长方向的输出间距。"""
    curve_projection_axis: Point3D | None = None
    """矢状位 CPR 的患者前后轴；采样前固定该坐标，仅展开左右侧弯。"""
    offset_axis: Point3D | None = None
    """CPR 首选堆叠方向：矢状位为左右轴，冠状位为前后轴。"""
    offsets_mm: List[float] | None = None
    """CPR 每张图沿当前 offset 轴相对中心曲线的偏移。"""
    cpr_range_mode: str | None = None
    """CPR offset 覆盖范围来源；full_label 为完整标注，body_prism 为主椎体。"""
    target_start: Point3D | None = None
    target_end: Point3D | None = None
    localizer_reference: Point3D | None = None
    """定位图正交选层或曲面显示参考网格的物理点；不参与正式切片重采样。"""
    localizer_plane: str | None = None
    """定位图参考显示状位；为空时自动选择，脊柱可用曲面背景，不改变正式切片状位。"""
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    config_id: str = ""
    """当前计划对应的状位配置标识，用于同状位多配置结果关联。"""
    output_plane: str = ""
    """用于隔离输出目录的唯一名称，例如横状位、横状位1。"""
    sampling_details: Dict[str, Any] | None = None
    """实际层厚、层间距、数量、有效范围和总范围的计算详情。"""
    fill_percent: float | None = None
    """第七阶段目标取景占比；None/0 不缩放；平面裁空按部位与取景策略执行。"""
    framing_superior_extension_mm: float | None = None
    """颅脑横状位获准的颅顶取景余量；同步范围扩展，不包含下端或取整余量。"""

    def to_dict(self) -> Dict[str, Any]:
        """序列化第七阶段可直接执行的一组完整切片计划。"""

        data = {
            "plan_id": self.plan_id,
            "slice_part": self.slice_part,
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
        if self.fill_percent is not None:
            data["fill_percent"] = self.fill_percent
        if self.framing_superior_extension_mm is not None:
            data["framing_superior_extension_mm"] = self.framing_superior_extension_mm
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
        if self.curve_projection_axis is not None:
            data["curve_projection_axis"] = self.curve_projection_axis.to_dict()
        if self.offset_axis is not None:
            data["offset_axis"] = self.offset_axis.to_dict()
        if self.offsets_mm is not None:
            data["offsets_mm"] = list(self.offsets_mm)
        if self.cpr_range_mode is not None:
            data["cpr_range_mode"] = self.cpr_range_mode
        if self.config_id:
            data["config_id"] = self.config_id
        if self.output_plane:
            data["output_plane"] = self.output_plane
        if self.sampling_details is not None:
            data["sampling_details"] = dict(self.sampling_details)
        data["slice_image_count"] = len(self.slices)
        return data
