"""按计划统一比例，按选定部位的目标取景，不依赖窗宽窗位。"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from src.models.request import parse_fill_percent
from src.reconstruction.body_framing import BodyFramingSupport
from src.reconstruction.framing_context import FramingContext
from src.reconstruction.symmetry_framing import SymmetryFramingPlane


@dataclass(frozen=True)
class InterspaceFramingGeometry:
    """同一椎间隙配置共用的裁空前采样矩阵、物理比例及其计算依据。"""

    size_xy: tuple[int, int]
    spacing_xy: tuple[float, float]
    scale: float
    target_size_mm: tuple[float, float]
    plan_ids: tuple[str, ...]
    reference_plan_ids: tuple[str, ...] = ()


class TargetFramer:
    """按选定目标取景；显式传入的轮廓支持仅供独立调用使用。"""

    def __init__(
        self,
        segmentation: sitk.Image,
        coords: dict,
        plan: dict,
        *,
        shared_geometry: InterspaceFramingGeometry | None = None,
        context: FramingContext | None = None,
        symmetry_plane: SymmetryFramingPlane | None = None,
        merge_context_into_target: bool = False,
        body_support: BodyFramingSupport | None = None,
    ):
        self.shared_geometry = shared_geometry
        self.symmetry_plane = symmetry_plane
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
        if context is not None and (
            merge_context_into_target or (percent is not None and percent > 0)
        ):
            extra_ids = set(context.label_ids) - ids - {0}
            if extra_ids:
                axis = np.asarray(context.axis_xyz, dtype=float)
                if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 0:
                    raise ValueError("显示支持范围的纵向轴必须是有限非零向量")
                axis /= np.linalg.norm(axis)
                upper_extension = float(context.upper_extension_mm)
                if not np.isfinite(upper_extension) or upper_extension < 0:
                    raise ValueError("显示支持范围的上端余量必须是有限非负距离")
                # 限定在原目标的物理纵向支撑范围内，不能按数组 z 裁斜切。
                # 相邻标签仅补充显示，不延长切片层组。颅脑的骨性支持也须
                # 剔除脑底以外的面部/下颌；上端只增加计划明确允许的颅顶余量。
                # 零 FillPercent 同样受限，不能因保留原像素间距而漏掉颅顶取景。
                target_indices = np.argwhere(selected)
                axis_steps = self.voxel_axes.T @ axis
                # 两组体素共用原点，范围比较可省去相同的原点投影。
                target_depths = target_indices[:, ::-1] @ axis_steps
                half_support = float(np.abs(axis_steps).sum() / 2)
                lower = float(target_depths.min() - half_support)
                upper = float(target_depths.max() + half_support + upper_extension)
                extra_indices = np.argwhere(np.isin(array, list(extra_ids)))
                extra_depths = extra_indices[:, ::-1] @ axis_steps
                keep = (extra_depths + half_support > lower + 1e-7) & (
                    extra_depths - half_support < upper - 1e-7
                )
                extra_indices = extra_indices[keep]
                if len(extra_indices):
                    self.context_label_ids = tuple(sorted(map(int, np.unique(array[tuple(extra_indices.T)]))))
                    if merge_context_into_target or plan.get("geometry_type") == "curved_stack":
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
        self._body_points = None
        self._body_voxel_axes = None
        self._body_region = None
        if (
            body_support is not None
            and plan.get("slice_part") in {"肺部", "髋骨"}
            and plan.get("slice_type") == "MPR"
            and plan.get("geometry_type") == "plane_stack"
            and percent is not None and percent > 0
        ):
            # 只作显示支持，不扩大切片层组或修改原始目标标签。
            axis = np.asarray(body_support.longitudinal_axis_xyz, dtype=float)
            if plan.get("plane") == "横状位":
                # 横状切面可能相对源轴斜切，整幅身体边缘的 SI 会随面内
                # 位置变化。须沿实际法向裁支持，不能用中心的 SI 截断体表。
                axis = self._vector(plan["slices"][0]["normal"])
            axis /= np.linalg.norm(axis)
            body_axes = np.asarray(body_support.voxel_axes, dtype=float)
            points = np.asarray(body_support.points_xyz, dtype=float)
            target_depths = self.points @ axis
            half_target = float(np.abs(self.voxel_axes.T @ axis).sum() / 2)
            lower_depth = float(target_depths.min() - half_target)
            upper_depth = float(target_depths.max() + half_target)
            if plan.get("plane") == "横状位":
                # 数量/间距计算可能向肺尖、肺底外多出空目标厚层，这些层
                # 仍应显示同层身体，不能跳回少量肺组织的整组中心。
                half_slab = float(plan.get("slice_thickness", 1.0)) / 2
                depths = [float(self._vector(item["center"]) @ axis) for item in plan["slices"]]
                lower_depth = min(lower_depth, min(depths) - half_slab)
                upper_depth = max(upper_depth, max(depths) + half_slab)
            half_body = float(np.abs(body_axes.T @ axis).sum() / 2)
            body_depths = points @ axis
            keep = (body_depths + half_body > lower_depth + 1e-7) & (
                body_depths - half_body < upper_depth - 1e-7
            )
            if np.any(keep):
                self._body_points = points[keep]
                self._body_voxel_axes = body_axes
                self._body_region = body_support.region
        self._projection_key = None
        self._bounds = None
        self._projection_axes = None
        self._half_voxel = None
        self._normal_depths = None
        self._context_normal_depths = None
        self._body_normal_depths = None
        self._body_half_voxel = None
        self._body_group_bounds = None
        self._framing_target_size = None
        self._reference_framing = None

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
            self._body_normal_depths = None
            self._body_half_voxel = None
            self._body_group_bounds = None
            if self._body_points is not None:
                self._body_half_voxel = np.abs(self._body_voxel_axes.T @ axes).sum(axis=0) / 2
                projected_body = self._body_points @ axes[:, :2]
                self._body_group_bounds = (
                    projected_body.min(axis=0) - self._body_half_voxel[:2],
                    projected_body.max(axis=0) + self._body_half_voxel[:2],
                )
            self._framing_target_size = None
            self._reference_framing = None
            self._projection_key = key
        return self._bounds

    def _slab_target_bounds(
        self, plan: dict, center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """求原目标、有界上下文及身体轮廓与当前厚层正重叠的外边界。"""

        axes = self._projection_axes
        if self._normal_depths is None:
            self._normal_depths = self.points @ axes[:, 2]
        if self._context_points is not None and self._context_normal_depths is None:
            self._context_normal_depths = self._context_points @ axes[:, 2]
        if self._body_points is not None and self._body_normal_depths is None:
            self._body_normal_depths = self._body_points @ axes[:, 2]
        center_depth = float(center @ axes[:, 2])
        # 只接触厚层边界的体素没有采样厚度，不能让相邻层的目标拉偏中心。
        # 容差为亚纳米级毫米数；额外覆盖大物理坐标投影的浮点舍入误差。
        bounds = []
        for points, depths, half_voxel in (
            (self.points, self._normal_depths, self._half_voxel),
            (self._context_points, self._context_normal_depths, self._half_voxel),
            (self._body_points, self._body_normal_depths, self._body_half_voxel),
        ):
            if points is None:
                continue
            half_support = float(plan.get("slice_thickness", 1.0)) / 2 + half_voxel[2]
            tolerance = max(1e-7, np.finfo(float).eps * max(abs(center_depth), half_support, 1.0) * 32)
            selected = np.abs(depths - center_depth) < half_support - tolerance
            if np.any(selected):
                projected = points[selected] @ axes[:, :2]
                bounds.append((projected.min(axis=0) - half_voxel[:2],
                               projected.max(axis=0) + half_voxel[:2]))
        if not bounds:
            return None
        return (
            np.min([item[0] for item in bounds], axis=0),
            np.max([item[1] for item in bounds], axis=0),
        )

    def _slab_has_body_support(self, plan: dict, center: np.ndarray) -> bool:
        """本层确实存在身体边界时才标记身体轮廓居中。"""

        if self._body_points is None:
            return False
        depth = float(center @ self._projection_axes[:, 2])
        half_support = float(plan.get("slice_thickness", 1.0)) / 2 + self._body_half_voxel[2]
        tolerance = max(1e-7, np.finfo(float).eps * max(abs(depth), half_support, 1.0) * 32)
        return bool(np.any(np.abs(self._body_normal_depths - depth) < half_support - tolerance))

    def framing_target_size(self, plan: dict, planned_slice: dict) -> np.ndarray:
        """预先容纳每层的完整显示支持范围，倍率仍对整组固定。"""

        lower, upper = self.plane_target_bounds(planned_slice)
        if self._framing_target_size is None:
            target_size = upper - lower
            target_canvas = self._body_points is None and uses_target_centered_canvas(plan)
            if target_canvas and plan.get("slice_part") == "肺部":
                self._reference_framing = self._lung_reference_geometry(plan)
                if self._reference_framing is not None:
                    target_size = self._reference_framing["required_size"]
                self._framing_target_size = target_size
                return target_size
            if target_canvas:
                # 各层已经独立居中，跨层的位置漂移不能再撑大共同视野。
                # 只用实际计划厚层的目标尺寸定倍率，避免部位始终达不到 Fill。
                target_size = np.zeros(2, dtype=float)
            if self._body_group_bounds is not None:
                body_lower, body_upper = self._body_group_bounds
                # 各层已分别居中，位置漂移不能再撑大共同视野。
                # 身体支持按真实厚层的最大联合尺寸定倍率，使最宽层达到设定占比。
                target_size = np.zeros(2, dtype=float)
            percent = parse_fill_percent(plan.get("fill_percent"))
            use_symmetry = self.symmetry_plane is not None and percent is not None and percent > 0
            if self._context_points is not None or use_symmetry or self._body_points is not None or target_canvas:
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
                        lower_slice, upper_slice = bounds
                        required_size = upper_slice - lower_slice
                        if use_symmetry:
                            target_center, _ = self.symmetry_plane.centered_projection(
                                self._projection_axes, float(center @ normal), lower_slice, upper_slice,
                            )
                            if target_center is not None:
                                # 以镜像中心容纳左右两侧完整边界，Fill100 也不能裁掉长侧。
                                required_size = 2 * np.maximum(
                                    upper_slice - target_center, target_center - lower_slice,
                                )
                        target_size = np.maximum(target_size, required_size)
            if self._body_group_bounds is not None and not np.any(target_size > 0):
                # 所有计划层均无交集时沿用整组回退中心的范围，避免零尺寸倍率。
                target_size = np.maximum(upper, body_upper) - np.minimum(lower, body_lower)
            elif target_canvas and not np.any(target_size > 0):
                target_size = upper - lower
            self._framing_target_size = target_size
        return self._framing_target_size

    def _lung_reference_geometry(self, plan: dict) -> dict | None:
        """以最大实际目标厚层固定面内平移，并一次容纳其他层的完整目标。"""

        axes = self._projection_axes
        slabs = []
        reference = None
        reference_area = -1.0
        for item in plan["slices"]:
            item_axes = np.column_stack([
                self._vector(item[key]) for key in ("row_dir", "col_dir", "normal")
            ])
            item_axes /= np.linalg.norm(item_axes, axis=0)
            if not np.allclose(item_axes, axes, rtol=0, atol=1e-12):
                continue
            center = self._vector(item["center"])
            bounds = self._slab_target_bounds(plan, center)
            if bounds is None:
                continue
            lower, upper = bounds
            slabs.append((center, lower, upper))
            area = float(np.prod(upper - lower))
            # 相同截面的浮点投影舍入不改变参考层；面积并列时取计划中最先一层。
            tolerance = max(1e-9, np.finfo(float).eps * max(area, reference_area, 1.0) * 64)
            if area > reference_area + tolerance:
                reference = (item, center, lower, upper)
                reference_area = area
        if reference is None:
            return None
        item, center, lower, upper = reference
        shift = (lower + upper) / 2 - center @ axes[:, :2]
        required_size = upper - lower
        for planned_center, slab_lower, slab_upper in slabs:
            fixed_center = planned_center @ axes[:, :2] + shift
            required_size = np.maximum(required_size, 2 * np.maximum(
                slab_upper - fixed_center, fixed_center - slab_lower,
            ))
        return {
            "slice_index": item["index"], "shift": shift,
            "target_size": upper - lower, "required_size": required_size,
        }

    def plane_geometry(self, plan: dict, planned_slice: dict) -> tuple[dict, dict]:
        """整组共用比例；肺部固定参考层偏移，髋骨逐层目标居中。

        正式流水线不让体表改变选定部位的倍率或中心。肺部正数 MPR 以最大
        目标厚层的面内偏移取景，所有层沿用该偏移，一次放宽视野保护其他层。
        髋骨正数 MPR 仍按各实际厚层目标居中、以最大尺寸确定共同间距。
        独立调用显式提供身体支持时，才按实际厚层联合尺寸及轮廓中心取景。
        椎间隙配置可共用矩阵及比例；逐层只移动面内中心，不改变切平面。
        可靠镜像面只约束面内左右中心，其余方向继续使用当前厚层 bbox 中心。
        肺部没有实际参考层时、其余计划当前层没有目标时回退整组中心；
        FillPercent 未设置/为零时
        保留原有整组目标中心。肺/髋正数 MPR 成片不再单侧裁空；其余平面
        HU 采样后的上下无源支持行裁剪仍可改变最终 Rows 和图像垂直中心。
        """

        lower, upper = self.plane_target_bounds(planned_slice)
        framing_target_size = self.framing_target_size(plan, planned_slice)
        axes = self._projection_axes
        center = self._vector(planned_slice["center"])
        center_local = center @ axes
        target_center = (lower + upper) / 2
        center_source = "group_target"
        slice_target_size = None
        symmetry_details = None
        percent = parse_fill_percent(plan.get("fill_percent"))
        if percent is not None and percent > 0:
            slice_bounds = self._slab_target_bounds(plan, center)
            if self._reference_framing is not None:
                target_center = center_local[:2] + self._reference_framing["shift"]
                center_source = "group_reference_slice"
                if slice_bounds is not None:
                    slice_target_size = list(map(float, slice_bounds[1] - slice_bounds[0]))
            elif slice_bounds is None:
                center_source = "group_fallback"
                if self._body_group_bounds is not None:
                    body_lower, body_upper = self._body_group_bounds
                    target_center = (np.minimum(lower, body_lower) + np.maximum(upper, body_upper)) / 2
                    center_source = "group_body_fallback"
            else:
                slice_lower, slice_upper = slice_bounds
                target_center = (slice_lower + slice_upper) / 2
                slice_target_size = list(map(float, slice_upper - slice_lower))
                center_source = "slice_target"
                if self._slab_has_body_support(plan, center):
                    center_source = "slice_body_contour"
                if self.symmetry_plane is not None:
                    symmetry_center, symmetry_status = self.symmetry_plane.centered_projection(
                        axes, float(center_local[2]), slice_lower, slice_upper,
                    )
                    symmetry_details = {
                        "status": symmetry_status,
                        "source_labels": list(self.symmetry_plane.source_labels),
                        "normal_xyz": list(self.symmetry_plane.normal_xyz),
                        "point_xyz": list(self.symmetry_plane.point_xyz),
                    }
                    if symmetry_center is not None:
                        symmetry_details["shift_in_plane_mm"] = list(map(
                            float, symmetry_center - target_center,
                        ))
                        target_center = symmetry_center
                        center_source = "slice_symmetry_plane"
        new_center = center + axes[:, :2] @ (target_center - center_local[:2])
        if self.shared_geometry is None:
            size, spacing = _plane_matrix(plan)
            new_spacing, scale = fitted_spacing(size, spacing, framing_target_size, plan.get("fill_percent"))
            if (
                (self._body_points is None and uses_target_centered_canvas(plan))
                or (
                    self._body_points is not None
                    and plan.get("slice_part") == "肺部"
                    and plan.get("plane") == "横状位"
                    and percent is not None and percent > 0
                )
            ):
                # 保留目标中心和共同采样间距，只对称去掉画布的多余空气。
                # 近方形画布在宽胶片格内按高度缩小时会压低目标的显示占比。
                # 尺寸包含百分比所需留白和半体素外缘，并向上取整保护轮廓。
                required_pixels = framing_target_size / (percent / 100.0) / new_spacing
                tolerance = np.maximum(1e-9, np.abs(required_pixels) * np.finfo(float).eps * 32)
                compact_size = np.maximum(1, np.ceil(required_pixels - tolerance).astype(int))
                # 两侧各裁整数个像素，保持原采样网格相位，避免薄体表因
                # 半像素平移发生新的部分容积变化。最多额外保留一个像素空气。
                compact_size += (size - compact_size) % 2
                size = np.minimum(size, compact_size)
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
            "output_size_xy": list(map(int, size)),
        }
        if self.shared_geometry is not None:
            framed_slice["framing"].update({
                "output_size_xy": list(self.shared_geometry.size_xy),
                "scale_scope": "interspace_config",
                "shared_target_size_mm": list(self.shared_geometry.target_size_mm),
                "shared_plan_ids": list(self.shared_geometry.plan_ids),
                "canvas_reference_plan_ids": list(self.shared_geometry.reference_plan_ids),
            })
        if symmetry_details is not None:
            framed_slice["framing"]["symmetry_centering"] = symmetry_details
        if self._body_points is not None:
            framed_slice["framing"]["body_support_source"] = "ct_body_contour"
            framed_slice["framing"]["body_support_region"] = self._body_region
        if self._reference_framing is not None:
            framed_slice["framing"].update({
                "reference_slice_index": self._reference_framing["slice_index"],
                "reference_shift_mm": list(map(float, self._reference_framing["shift"])),
                "reference_target_size_mm": list(map(float, self._reference_framing["target_size"])),
            })
        return framed_plan, framed_slice


def uses_target_centered_canvas(plan: dict) -> bool:
    """肺正数 MPR 固定参考偏移、髋正数 MPR 逐层居中，均收紧目标画布。"""

    return (
        plan.get("slice_part") in {"肺部", "髋骨"}
        and plan.get("slice_type") == "MPR"
        and plan.get("geometry_type") == "plane_stack"
        and (parse_fill_percent(plan.get("fill_percent")) or 0) > 0
    )


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
    symmetry_planes: dict[str, SymmetryFramingPlane] | None = None,
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
        framer = TargetFramer(
            segmentation, coords, plan, context=context,
            symmetry_plane=(symmetry_planes or {}).get(str(plan.get("plan_id", ""))),
        )
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
    """按像素外边界定义占比，本步保留采样矩阵和像素纵横比。"""

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
