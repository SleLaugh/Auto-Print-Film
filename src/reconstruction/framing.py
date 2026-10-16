"""按计划统一取景比例，正 FillPercent 时逐层居中目标（不依赖窗宽窗位）。"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from src.models.request import parse_fill_percent
from src.reconstruction.framing_context import FramingContext


@dataclass(frozen=True)
class InterspaceFramingGeometry:
    """同一椎间隙配置共用的输出矩阵、物理采样比例及其计算依据。"""

    size_xy: tuple[int, int]
    spacing_xy: tuple[float, float]
    scale: float
    target_size_mm: tuple[float, float]
    plan_ids: tuple[str, ...]
    reference_plan_ids: tuple[str, ...] = ()


class TargetFramer:
    """以计划目标为比例基准，正 FillPercent 时结合有界上下文逐层居中。"""

    def __init__(
        self,
        segmentation: sitk.Image,
        coords: dict,
        plan: dict,
        *,
        shared_geometry: InterspaceFramingGeometry | None = None,
        context: FramingContext | None = None,
    ):
        self.shared_geometry = shared_geometry
        names = plan.get("source_labels") or []
        # 旧 planner 的这两个名称表示整个目标范围，并非模型标签名。
        if not names or any(name in {
            "correction_scope_bbox", "anatomical_geometry.global_frame"
        } for name in names):
            names = list(coords)
        ids = {
            int(value)
            for name in names
            for value in coords.get(name, {}).get("label_ids", [])
            if int(value) != 0
        }
        if not ids:
            raise ValueError(f"plan={plan.get('plan_id')} 缺少用于居中的目标 label_ids")
        array = sitk.GetArrayViewFromImage(segmentation)
        selected = np.isin(array, list(ids))
        if not np.any(selected):
            raise ValueError(f"plan={plan.get('plan_id')} 分割图中未找到目标标签 {sorted(ids)}")
        direction = np.asarray(segmentation.GetDirection()).reshape(3, 3)
        self.voxel_axes = direction * np.asarray(segmentation.GetSpacing())
        self.context_label_ids = ()
        self._context_points = None
        percent = parse_fill_percent(plan.get("fill_percent"))
        if context is not None and percent is not None and percent > 0:
            extra_ids = set(context.label_ids) - ids - {0}
            if extra_ids:
                axis = np.asarray(context.axis_xyz, dtype=float)
                if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 0:
                    raise ValueError("显示支持范围的纵向轴必须是有限非零向量")
                axis /= np.linalg.norm(axis)
                # 限定在原目标的纵向覆盖内；相邻标签只补充显示，不延长切割范围。
                target_indices = np.argwhere(selected)
                axis_steps = self.voxel_axes.T @ axis
                # 两组体素共用原点，范围比较可省去相同的原点投影。
                target_depths = target_indices[:, ::-1] @ axis_steps
                half_support = float(np.abs(axis_steps).sum() / 2)
                lower = float(target_depths.min() - half_support)
                upper = float(target_depths.max() + half_support)
                extra_indices = np.argwhere(np.isin(array, list(extra_ids)))
                extra_depths = extra_indices[:, ::-1] @ axis_steps
                keep = (extra_depths + half_support > lower + 1e-7) & (
                    extra_depths - half_support < upper - 1e-7
                )
                extra_indices = extra_indices[keep]
                if len(extra_indices):
                    self.context_label_ids = tuple(sorted(map(int, np.unique(array[tuple(extra_indices.T)]))))
                    if plan.get("geometry_type") == "curved_stack":
                        selected[tuple(extra_indices.T)] = True
                    else:
                        self._context_points = (
                            extra_indices[:, ::-1] @ self.voxel_axes.T + segmentation.GetOrigin()
                        )
        self.mask = None
        if plan.get("geometry_type") == "curved_stack":
            # CPR 只使用展开后的掩膜并集，无需展开全体目标体素的三维坐标。
            self.mask = sitk.GetImageFromArray(selected.astype(np.uint8))
            self.mask.CopyInformation(segmentation)
        else:
            indices = np.argwhere(selected)
            self.points = indices[:, ::-1] @ self.voxel_axes.T + segmentation.GetOrigin()
        self._projection_key = None
        self._bounds = None
        self._projection_axes = None
        self._half_voxel = None
        self._normal_depths = None
        self._context_normal_depths = None
        self._framing_target_size = None

    @staticmethod
    def _vector(point: dict) -> np.ndarray:
        return np.array([point[axis] for axis in "xyz"], dtype=float)

    @staticmethod
    def _point(vector: np.ndarray) -> dict:
        return dict(zip("xyz", map(float, vector)))

    def plane_target_bounds(self, planned_slice: dict) -> tuple[np.ndarray, np.ndarray]:
        """当前计划全部目标标签在给定切面上的投影外边界。"""
        axes = np.column_stack([
            self._vector(planned_slice[key])
            for key in ("row_dir", "col_dir", "normal")
        ])
        axes /= np.linalg.norm(axes, axis=0)
        key = tuple(axes.ravel())
        if key != self._projection_key:
            projected = self.points @ axes[:, :2]
            half_voxel = np.abs(self.voxel_axes.T @ axes).sum(axis=0) / 2
            self._bounds = (
                projected.min(axis=0) - half_voxel[:2],
                projected.max(axis=0) + half_voxel[:2],
            )
            self._projection_axes = axes
            self._half_voxel = half_voxel
            self._normal_depths = None
            self._context_normal_depths = None
            self._framing_target_size = None
            self._projection_key = key
        return self._bounds

    def _slab_target_bounds(
        self, plan: dict, center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """求原目标及有界上下文与当前厚层正重叠的体素投影外边界。"""

        axes = self._projection_axes
        if self._normal_depths is None:
            self._normal_depths = self.points @ axes[:, 2]
        if self._context_points is not None and self._context_normal_depths is None:
            self._context_normal_depths = self._context_points @ axes[:, 2]
        center_depth = float(center @ axes[:, 2])
        half_support = float(plan.get("slice_thickness", 1.0)) / 2 + self._half_voxel[2]
        # 只接触厚层边界的体素没有采样厚度，不能让相邻层的目标拉偏中心。
        # 容差为亚纳米级毫米数；额外覆盖大物理坐标投影的浮点舍入误差。
        tolerance = max(1e-7, np.finfo(float).eps * max(abs(center_depth), half_support, 1.0) * 32)
        bounds = []
        for points, depths in (
            (self.points, self._normal_depths),
            (self._context_points, self._context_normal_depths),
        ):
            if points is None:
                continue
            selected = np.abs(depths - center_depth) < half_support - tolerance
            if np.any(selected):
                projected = points[selected] @ axes[:, :2]
                bounds.append((projected.min(axis=0), projected.max(axis=0)))
        if not bounds:
            return None
        return (
            np.min([item[0] for item in bounds], axis=0) - self._half_voxel[:2],
            np.max([item[1] for item in bounds], axis=0) + self._half_voxel[:2],
        )

    def framing_target_size(self, plan: dict, planned_slice: dict) -> np.ndarray:
        """预先容纳每层的完整显示支持范围，倍率仍对整组固定。"""

        lower, upper = self.plane_target_bounds(planned_slice)
        if self._framing_target_size is None:
            target_size = upper - lower
            if self._context_points is not None:
                # 计划内通常方向一致；有不同方向时各方向使用自己的投影缓存。
                normal = self._projection_axes[:, 2]
                for item in plan["slices"]:
                    item_axes = np.column_stack([
                        self._vector(item[key]) for key in ("row_dir", "col_dir", "normal")
                    ])
                    item_axes /= np.linalg.norm(item_axes, axis=0)
                    if not np.allclose(item_axes, self._projection_axes, rtol=0, atol=1e-12):
                        continue
                    center = self._vector(item["center"])
                    # 只读取层中心的法向位置，不使用规划时的面内偏移。
                    bounds = self._slab_target_bounds(plan, normal * float(center @ normal))
                    if bounds is not None:
                        target_size = np.maximum(target_size, bounds[1] - bounds[0])
            self._framing_target_size = target_size
        return self._framing_target_size

    def plane_geometry(self, plan: dict, planned_slice: dict) -> tuple[dict, dict]:
        """整组共用比例，正 FillPercent 时按当前厚层目标与上下文联合居中。

        原目标整组投影范围作为倍率基准，并预先容纳每层的联合显示范围。
        椎间隙配置可共用矩阵及比例；逐层只移动面内中心，不改变切平面。
        原目标与上下文均为空，或 FillPercent 未设置/为零时，保留整组投影中心。
        """

        lower, upper = self.plane_target_bounds(planned_slice)
        framing_target_size = self.framing_target_size(plan, planned_slice)
        axes = self._projection_axes
        center = self._vector(planned_slice["center"])
        center_local = center @ axes
        target_center = (lower + upper) / 2
        center_source = "group_target"
        slice_target_size = None
        percent = parse_fill_percent(plan.get("fill_percent"))
        if percent is not None and percent > 0:
            slice_bounds = self._slab_target_bounds(plan, center)
            if slice_bounds is None:
                center_source = "group_fallback"
            else:
                slice_lower, slice_upper = slice_bounds
                target_center = (slice_lower + slice_upper) / 2
                slice_target_size = list(map(float, slice_upper - slice_lower))
                center_source = "slice_target"
        new_center = center + axes[:, :2] @ (target_center - center_local[:2])
        if self.shared_geometry is None:
            size, spacing = _plane_matrix(plan)
            new_spacing, scale = fitted_spacing(size, spacing, framing_target_size, plan.get("fill_percent"))
        else:
            size = np.asarray(self.shared_geometry.size_xy)
            new_spacing = np.asarray(self.shared_geometry.spacing_xy)
            scale = self.shared_geometry.scale
        framed_plan = dict(plan)
        framed_plan["extent_mm"] = dict(zip(("width", "height"), (size - 1) * new_spacing))
        framed_plan["output_spacing_mm"] = {"x": float(new_spacing[0]), "y": float(new_spacing[1]), "z": 0.0}
        framed_slice = dict(planned_slice)
        framed_slice["center"] = self._point(new_center)
        framed_slice["framing"] = {
            "source": "group_target", "mode": "group",
            "target_size_mm": list(map(float, upper - lower)),
            "framing_target_size_mm": list(map(float, framing_target_size)),
            "context_label_ids": list(self.context_label_ids),
            "center_source": center_source,
            "slice_target_size_mm": slice_target_size,
            "scale": float(scale),
            "fill_percent": plan.get("fill_percent"),
            "center": framed_slice["center"],
            "extent_mm": framed_plan["extent_mm"],
            "output_spacing_mm": framed_plan["output_spacing_mm"],
        }
        if self.shared_geometry is not None:
            framed_slice["framing"].update({
                "output_size_xy": list(self.shared_geometry.size_xy),
                "scale_scope": "interspace_config",
                "shared_target_size_mm": list(self.shared_geometry.target_size_mm),
                "shared_plan_ids": list(self.shared_geometry.plan_ids),
                "canvas_reference_plan_ids": list(self.shared_geometry.reference_plan_ids),
            })
        return framed_plan, framed_slice


def _plane_matrix(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    spacing = np.array([plan["output_spacing_mm"][axis] for axis in "xy"], dtype=float)
    extent = np.array([plan["extent_mm"][key] for key in ("width", "height")], dtype=float)
    size = np.maximum(1, np.rint(extent / spacing).astype(int) + 1)
    return size, spacing


def prepare_interspace_framing(
    segmentation: sitk.Image,
    coords: dict,
    plans: list[dict],
    *,
    context: FramingContext | None = None,
) -> InterspaceFramingGeometry:
    """以前面各组画布中位数及全部计划的显示尺寸需求确定共同采样比例。

    显示需求覆盖原目标整组投影及每个厚层的目标与有界上下文联合范围。
    逐计划创建并释放目标点云，只保留尺寸统计，避免同时持有多套体素坐标。
    各间隙的空间位置不参与范围合并，否则不同高度会错误扩大取景视野。
    最后一个间隙不参与画布参考；只有一组时使用该组自身画布。
    """

    if not plans:
        raise ValueError("椎间隙统一取景至少需要一个计划")
    reference_plans = plans[:-1] or plans
    reference_sizes = []
    target_size = np.zeros(2, dtype=float)
    reference_spacing = None
    reference_percent = None
    plan_ids = []
    for index, plan in enumerate(plans):
        if plan.get("slice_type") != "椎间隙" or plan.get("geometry_type") != "plane_stack":
            raise ValueError("椎间隙统一取景仅支持椎间隙 plane_stack 计划")
        if not plan.get("slices"):
            raise ValueError(f"plan={plan.get('plan_id')} 缺少椎间隙切片")
        spacing = np.array([plan["output_spacing_mm"][axis] for axis in "xy"], dtype=float)
        percent = parse_fill_percent(plan.get("fill_percent")) or 0.0
        if reference_spacing is None:
            reference_spacing = spacing
            reference_percent = percent
        elif not np.array_equal(reference_spacing, spacing) or reference_percent != percent:
            raise ValueError("同一椎间隙配置需要一致的基础像素间距和 FillPercent")
        if index < len(reference_plans):
            reference_sizes.append(_plane_matrix(plan)[0])
        framer = TargetFramer(segmentation, coords, plan, context=context)
        for planned_slice in plan["slices"]:
            target_size = np.maximum(target_size, framer.framing_target_size(plan, planned_slice))
        del framer
        plan_ids.append(str(plan.get("plan_id", "")))
    shared_size = np.maximum(1, np.rint(np.median(reference_sizes, axis=0)).astype(int))
    new_spacing, scale = fitted_spacing(
        shared_size, reference_spacing, target_size, reference_percent,
    )
    return InterspaceFramingGeometry(
        size_xy=tuple(map(int, shared_size)),
        spacing_xy=tuple(map(float, new_spacing)),
        scale=float(scale),
        target_size_mm=tuple(map(float, target_size)),
        plan_ids=tuple(plan_ids),
        reference_plan_ids=tuple(str(plan.get("plan_id", "")) for plan in reference_plans),
    )


def fitted_spacing(size, spacing, target_size, fill_percent):
    """按像素外边界定义占比，保留输出矩阵和像素纵横比。"""

    percent = parse_fill_percent(fill_percent)
    spacing = np.asarray(spacing, dtype=float)
    if percent is None or percent == 0:
        return spacing, 1.0
    # 极小但有限的百分比也可能使物理视野溢出，不能把 inf/NaN 传给重采样器。
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        factor = np.max(np.asarray(target_size) / (np.asarray(size) * spacing)) / percent * 100
        new_spacing = spacing * factor
        field_of_view = np.asarray(size) * new_spacing
        scale = 1.0 / factor
    if (
        not np.all(np.isfinite(new_spacing))
        or not np.all(np.isfinite(field_of_view))
        or np.any(new_spacing <= 0)
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("FillPercent 对应的物理视野超出可表示范围，请调整百分比")
    return new_spacing, float(scale)


def curved_group_bounds(masks):
    """在共同展开图坐标中求整组标签并集，空白层不改变取景。"""

    bounds = []
    reference = None
    for mask in masks:
        geometry = (mask.GetSize(), mask.GetSpacing())
        if reference is not None and geometry != reference:
            raise ValueError("CPR 整组取景需要一致的展开图矩阵与像素间距")
        reference = geometry
        occupied = np.argwhere(sitk.GetArrayViewFromImage(mask)[0] > 0)
        if len(occupied):
            bounds.append((occupied.min(axis=0)[::-1] - 0.5, occupied.max(axis=0)[::-1] + 0.5))
    if not bounds:
        raise ValueError("CPR 全组切片没有目标标签，无法居中或计算 FillPercent")
    return np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)


def frame_curved_slice(image: sitk.Image, mask: sitk.Image, fill_percent, group_bounds=None):
    """共用展开图比例，正 FillPercent 按本张掩膜居中；保持 CPR 展开空间。"""

    lower, upper = curved_group_bounds([mask]) if group_bounds is None else group_bounds
    center_index = (lower + upper) / 2
    spacing = np.array(image.GetSpacing()[:2])
    size = np.array(image.GetSize()[:2])
    new_spacing, scale = fitted_spacing(size, spacing, (upper - lower) * spacing, fill_percent)
    center_source = "group_target"
    slice_target_size = None
    percent = parse_fill_percent(fill_percent)
    if percent is not None and percent > 0:
        occupied = np.argwhere(sitk.GetArrayViewFromImage(mask)[0] > 0)
        if len(occupied):
            slice_lower = occupied.min(axis=0)[::-1] - 0.5
            slice_upper = occupied.max(axis=0)[::-1] + 0.5
            center_index = (slice_lower + slice_upper) / 2
            slice_target_size = list(map(float, (slice_upper - slice_lower) * spacing))
            center_source = "slice_target"
        else:
            center_source = "group_fallback"
    center = np.array(image.TransformContinuousIndexToPhysicalPoint((*map(float, center_index), 0.0)))
    axes = np.asarray(image.GetDirection()).reshape(3, 3)
    origin = center - axes[:, :2] @ ((size - 1) * new_spacing / 2)
    result = sitk.Resample(
        image, image.GetSize(), sitk.Transform(), sitk.sitkLinear,
        tuple(origin), (*map(float, new_spacing), image.GetSpacing()[2]),
        image.GetDirection(), -1024.0, sitk.sitkFloat32,
    )
    return result, {
        "source": "group_target", "mode": "group",
        "center_source": center_source,
        "slice_target_size_mm": slice_target_size,
        "scale": float(scale), "fill_percent": fill_percent,
        "target_size_mm": list(map(float, (upper - lower) * spacing)),
        "center": TargetFramer._point(center),
        "output_spacing_mm": {"x": float(new_spacing[0]), "y": float(new_spacing[1]), "z": 0.0},
    }
