#!/src/services/stage_mapping.py
# -*- coding: utf-8 -*-

"""
阶段四：切片规划

本阶段负责回答“要切哪些图”。它会根据第三步定位结果、部位、切割方式和
状位配置生成 SlicePlan[]，并把每一张待切图的中心点和方向提前展开好。
第五步后续只负责执行这些计划，不再重新决定切片数量和切片位置。
"""

import json
import math
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import MappingStageResult, Point3D
from src.planning.geometry import normalize_vector, subtract_points, vector_length
from src.planning import select_planner
from src.planning.cut_policy import cut_mode_for_plane
from src.utils import DebugLog, is_debug_log_file_enabled


@dataclass(frozen=True)
class _OverviewAxes:
    """一个默认直线状位在输入图像索引空间中的总览轴配置。"""

    stack_axis: int
    """切片中心排列和层厚覆盖所在的输入轴，0/1/2 对应 x/y/z。"""

    fixed_axis: int
    """生成单层总览时固定不变的输入轴。"""

    line_axis: int
    """每条切片中心线和层厚覆盖带延伸的输入轴。"""


_OVERVIEW_AXES = {
    "横状位": _OverviewAxes(stack_axis=2, fixed_axis=0, line_axis=1),
    "矢状位": _OverviewAxes(stack_axis=0, fixed_axis=1, line_axis=2),
    "冠状位": _OverviewAxes(stack_axis=1, fixed_axis=0, line_axis=2),
}
"""横、矢、冠零角度直线计划对应的总览输入轴配置。"""


class MappingStage(BaseStage):
    """切片规划阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="mapping", config=config)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        prev_result = context.get_result('localization')
        
        if not prev_result:
            raise ValueError("前置阶段 'localization' 结果不存在")
        
        request = context.request
        coords = prev_result.get('coords', {})
        
        DebugLog("=" * 60)
        DebugLog("【阶段四】切片规划 - 生成第五步执行计划")
        DebugLog(f"  部位: {request.body_part.value}")
        DebugLog("  切割方式: 按部位和状位自动推导")
        DebugLog(f"  定位坐标数量: {len(coords)}")
        DebugLog("=" * 60)
        
        # 第四步内部拆成 planner + expander。planner 负责找区间和路径；
        # expander 已在 planner 内部调用，把路径展开成每张图的 slices[]。
        DebugLog("  [1] select_planner: 选择部位/切法 planner...")
        grouped_configs: Dict[str, list[Any]] = {}
        for plane_config in request.plane_configs:
            mode = cut_mode_for_plane(request.body_part, plane_config.plane)
            grouped_configs.setdefault(mode, []).append(plane_config)

        DebugLog("  [2] build_plans: 生成并展开切片计划...")
        all_plans = []
        all_warnings = []
        planner_names = []
        for cut_mode, plane_configs in grouped_configs.items():
            planner = select_planner(request.body_part.value, cut_mode)
            planner_names.append(planner.planner_name)
            DebugLog(f"    planner: {planner.planner_name}")
            scoped_request = copy(request)
            scoped_request.plane_configs = plane_configs
            partial_result = planner.build_plans(scoped_request, prev_result)
            all_plans.extend(partial_result.plans)
            all_warnings.extend(partial_result.warnings)
        planner_name = "+".join(planner_names)
        plans = [plan.to_dict() for plan in all_plans]
        diagnostics = self._build_plan_diagnostics(plans)

        debug_overlay_path = ""
        debug_overview_overlay_path = ""
        debug_native_overview_overlay_path = ""
        debug_overview_paths: Dict[str, str] = {}
        debug_native_overview_paths: Dict[str, str] = {}
        metadata_path = ""
        if is_debug_log_file_enabled():
            DebugLog("  [3] write_debug_overlay: 写出切割线和含原图总览...")
            debug_overlay_path = self._write_slice_plan_overlay(
                context=context,
                localization_result=prev_result,
                plans=plans,
            )
            (
                debug_overview_paths,
                debug_native_overview_paths,
            ) = self._write_slice_plan_overview_overlays(
                context=context,
                localization_result=prev_result,
                plans=plans,
            )
            debug_overview_overlay_path = self._legacy_overview_path(
                debug_overview_paths,
                plans,
            )
            debug_native_overview_overlay_path = self._legacy_overview_path(
                debug_native_overview_paths,
                plans,
            )
            DebugLog("  [4] write_metadata: 写出 slice_plans.json...")
            metadata_path = self._write_slice_plans(
                context,
                planner_name=planner_name,
                plans=plans,
                diagnostics=diagnostics,
                warnings=all_warnings,
                debug_overlay_path=debug_overlay_path,
                debug_overview_overlay_path=debug_overview_overlay_path,
                debug_native_overview_overlay_path=debug_native_overview_overlay_path,
                debug_overview_paths=debug_overview_paths,
                debug_native_overview_paths=debug_native_overview_paths,
            )

        result = {
            'planner': planner_name,
            'plans': plans,
            'warnings': all_warnings,
            'diagnostics': diagnostics,
            'plan_count': len(plans),
            'metadata_path': str(metadata_path),
            'debug_slice_plan_overlay_path': debug_overlay_path,
            'debug_slice_plan_overview_overlay_path': debug_overview_overlay_path,
            'debug_slice_plan_native_overview_overlay_path': debug_native_overview_overlay_path,
            'debug_slice_plan_overview_paths': debug_overview_paths,
            'debug_slice_plan_overview_with_image_paths': debug_overview_paths,
            'debug_slice_plan_native_overview_paths': debug_native_overview_paths,
        }
        
        context.set_stage_result(MappingStageResult(payload=result))
        
        DebugLog(f"    切片计划数量: {len(plans)}")
        for diagnostic in diagnostics:
            DebugLog(
                "    "
                f"{diagnostic['plan_id']}: "
                f"{diagnostic['slice_count']} 张, "
                f"区间 {diagnostic['range_length_mm']}mm, "
                f"层厚 {diagnostic['slice_thickness']}mm, "
                f"层间距 {diagnostic['slice_spacing']}mm"
            )
        if debug_overlay_path:
            DebugLog(f"    切割线 overlay: {debug_overlay_path}")
        if debug_overview_overlay_path:
            DebugLog(f"    切割线总览 overlay: {debug_overview_overlay_path}")
        for plane, path in debug_native_overview_paths.items():
            DebugLog(f"    {plane}原始网格总览 overlay: {path}")
        if all_warnings:
            DebugLog(f"    warnings: {all_warnings}")
        DebugLog("  ✅ 阶段四完成")
        DebugLog("=" * 60)
        
        return context

    def _write_slice_plans(
        self,
        context: ProcessingContext,
        planner_name: str,
        plans: list[Dict[str, Any]],
        diagnostics: list[Dict[str, Any]],
        warnings: list[str],
        debug_overlay_path: str,
        debug_overview_overlay_path: str,
        debug_native_overview_overlay_path: str,
        debug_overview_paths: Dict[str, str],
        debug_native_overview_paths: Dict[str, str],
    ) -> Path:
        """把阶段四输出写入 metadata/slice_plans.json，方便第五步和人工检查。"""

        metadata_dir = Path(context.request.output_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "slice_plans.json"
        data = {
            "schema_version": "slice_plan_v1",
            "planner": planner_name,
            "plan_count": len(plans),
            "plans": plans,
            "diagnostics": diagnostics,
            "warnings": warnings,
            "debug_slice_plan_overlay_path": debug_overlay_path,
            "debug_slice_plan_overview_overlay_path": debug_overview_overlay_path,
            "debug_slice_plan_native_overview_overlay_path": debug_native_overview_overlay_path,
            "debug_slice_plan_overview_paths": debug_overview_paths,
            "debug_slice_plan_overview_with_image_paths": debug_overview_paths,
            "debug_slice_plan_native_overview_paths": debug_native_overview_paths,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return metadata_path

    @staticmethod
    def _debug_plan_suffix(plans: list[Dict[str, Any]]) -> str:
        """把实际生成计划的状位和切割方式压缩成调试文件名后缀。"""

        plane_labels = {
            "横状位": "横",
            "矢状位": "矢",
            "冠状位": "冠",
        }
        cut_mode_labels = {
            "whole_range": "整",
            "整切": "整",
            "interspace": "区",
            "椎间切": "区",
            "区间切": "区",
        }

        planes = []
        cut_modes = []
        for plan in plans:
            plane = str(plan.get("plane", ""))
            plane_label = plane_labels.get(plane, plane)
            if plane_label and plane_label not in planes:
                planes.append(plane_label)

            cut_mode = str(plan.get("cut_mode", ""))
            cut_mode_label = cut_mode_labels.get(cut_mode, cut_mode)
            if cut_mode_label and cut_mode_label not in cut_modes:
                cut_modes.append(cut_mode_label)

        return (
            f"{'+'.join(planes) or '未知'}_"
            f"{'+'.join(cut_modes) or '未知'}"
        )

    def _build_plan_diagnostics(self, plans: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        为每个 plan 生成可读诊断信息。

        这部分不参与第五步执行，只用于排查“到底是计划张数少，还是 overlay
        看起来少”。尤其当层间距大于层厚时，输出图之间本来就会有物理空档，
        需要在 JSON 里直观看出来。
        """

        diagnostics = []
        for plan in plans:
            range_start = plan.get("range_start", {})
            range_end = plan.get("range_end", {})
            slice_spacing = float(plan.get("slice_spacing", 0.0))
            slice_thickness = float(plan.get("slice_thickness", 0.0))
            range_length = 0.0
            if range_start and range_end:
                range_length = vector_length(subtract_points(
                    end=self._dict_to_point(range_end),
                    start=self._dict_to_point(range_start),
                ))

            slice_count = len(plan.get("slices", []))
            slice_centers = [
                self._dict_to_point(planned_slice["center"])
                for planned_slice in plan.get("slices", [])
                if planned_slice.get("center")
            ]
            actual_spacings = [
                vector_length(subtract_points(end=current, start=previous))
                for previous, current in zip(slice_centers, slice_centers[1:])
            ]
            actual_spacing_min = min(actual_spacings, default=0.0)
            actual_spacing_max = max(actual_spacings, default=0.0)
            has_nonuniform_spacing = any(
                abs(actual_spacing - slice_spacing) > 1e-6
                for actual_spacing in actual_spacings
            )
            spacing_minus_thickness = slice_spacing - slice_thickness
            regular_center_count = (
                int(math.floor(range_length / slice_spacing)) + 1
                if slice_spacing > 0
                else 0
            )
            has_extra_endpoint_slice = False
            leading_uncovered = 0.0
            trailing_uncovered = 0.0
            target_start_data = plan.get("target_start")
            target_end_data = plan.get("target_end")
            target_length = 0.0
            first_coverage_start = None
            last_coverage_end = None
            target_shorter_than_thickness = False
            if target_start_data and target_end_data:
                target_start = self._dict_to_point(target_start_data)
                target_end = self._dict_to_point(target_end_data)
                target_vector = subtract_points(end=target_end, start=target_start)
                target_length = vector_length(target_vector)
                target_shorter_than_thickness = target_length < slice_thickness
            else:
                target_start = self._dict_to_point(range_start) if range_start else None
                target_end = self._dict_to_point(range_end) if range_end else None

            if slice_centers and target_start is not None and target_end is not None:
                target_vector = subtract_points(end=target_end, start=target_start)
                if vector_length(target_vector) > 0:
                    coverage_direction = normalize_vector(target_vector)
                else:
                    coverage_direction = Point3D(0.0, 0.0, 1.0)
                half_thickness = slice_thickness / 2.0
                first_coverage_start = Point3D(
                    slice_centers[0].x - coverage_direction.x * half_thickness,
                    slice_centers[0].y - coverage_direction.y * half_thickness,
                    slice_centers[0].z - coverage_direction.z * half_thickness,
                )
                last_coverage_end = Point3D(
                    slice_centers[-1].x + coverage_direction.x * half_thickness,
                    slice_centers[-1].y + coverage_direction.y * half_thickness,
                    slice_centers[-1].z + coverage_direction.z * half_thickness,
                )
                leading_delta = subtract_points(
                    end=first_coverage_start,
                    start=target_start,
                )
                trailing_delta = subtract_points(
                    end=target_end,
                    start=last_coverage_end,
                )
                leading_uncovered = max(
                    leading_delta.x * coverage_direction.x
                    + leading_delta.y * coverage_direction.y
                    + leading_delta.z * coverage_direction.z,
                    0.0,
                )
                trailing_uncovered = max(
                    trailing_delta.x * coverage_direction.x
                    + trailing_delta.y * coverage_direction.y
                    + trailing_delta.z * coverage_direction.z,
                    0.0,
                )
            calculation_note = (
                "阶段四按层间距从 range_start 到 range_end 展开每张输出图中心；"
                "区间末端不足一个完整层间距时不追加非等距中心。"
            )
            if spacing_minus_thickness > 0:
                calculation_note += (
                    f" 当前层间距比层厚大 {spacing_minus_thickness:.3f}mm，"
                    "厚层图之间会有未覆盖间隔，看起来会少线；"
                    "需要更多输出图时应减小 slice_spacing。"
                )

            diagnostics.append({
                "plan_id": plan.get("plan_id", ""),
                "plane": plan.get("plane", ""),
                "range_length_mm": round(range_length, 3),
                "slice_count": slice_count,
                "regular_center_count": regular_center_count,
                "has_extra_endpoint_slice": has_extra_endpoint_slice,
                "actual_spacing_min_mm": round(actual_spacing_min, 3),
                "actual_spacing_max_mm": round(actual_spacing_max, 3),
                "has_nonuniform_spacing": has_nonuniform_spacing,
                "target_start": target_start_data,
                "target_end": target_end_data,
                "target_length_mm": round(target_length, 3),
                "target_shorter_than_thickness": target_shorter_than_thickness,
                "first_coverage_start": (
                    first_coverage_start.to_dict() if first_coverage_start else None
                ),
                "last_coverage_end": (
                    last_coverage_end.to_dict() if last_coverage_end else None
                ),
                "leading_uncovered_mm": round(leading_uncovered, 3),
                "trailing_uncovered_mm": round(trailing_uncovered, 3),
                "slice_thickness": slice_thickness,
                "slice_spacing": slice_spacing,
                "spacing_minus_thickness": round(spacing_minus_thickness, 3),
                "calculation_note": calculation_note,
            })
        return diagnostics

    @staticmethod
    def _dict_to_point(point: Dict[str, Any]):
        """把 JSON 中的 x/y/z 字典转换成 Point3D，用于诊断距离计算。"""

        from src.models import Point3D

        return Point3D(
            x=float(point["x"]),
            y=float(point["y"]),
            z=float(point["z"]),
        )

    def _write_slice_plan_overlay(
        self,
        context: ProcessingContext,
        localization_result: Dict[str, Any],
        plans: list[Dict[str, Any]],
    ) -> tuple[str, str]:
        """
        写出第四步切割线调试 overlay。

        overlay 与第三步使用同一套 NIfTI 调试思路：参考图来自第三步声明的
        seg_for_reconstruction_path，输出图与参考图同 size/spacing/origin/direction。
        数值约定为：
        - plan_index * 10 + 1：该 plan 的中心路径线。
        - plan_index * 10 + 2：该 plan 已展开切片的中心点。
        - plan_index * 10 + 3：每张切片在该平面内的十字引导线。

        这个文件只用于人工核对“线画得对不对”，不作为第五步执行输入。
        """

        reference_path = localization_result.get("seg_for_reconstruction_path")
        if not reference_path:
            return ""

        reference_path = Path(reference_path)
        if not reference_path.exists():
            return ""

        reference_image = sitk.ReadImage(str(reference_path))
        reference_array = sitk.GetArrayFromImage(reference_image)
        overlay_array = np.zeros(reference_array.shape, dtype=np.uint16)

        for plan_index, plan in enumerate(plans, start=1):
            path_value = plan_index * 10 + 1
            center_value = plan_index * 10 + 2
            slice_guide_value = plan_index * 10 + 3
            path_points = plan.get("path_points", [])
            for start_point, end_point in zip(path_points, path_points[1:]):
                self._paint_physical_line(
                    reference_image=reference_image,
                    overlay_array=overlay_array,
                    start_point=start_point,
                    end_point=end_point,
                    value=path_value,
                )

            for planned_slice in plan.get("slices", []):
                self._paint_planned_slice_guides(
                    reference_image=reference_image,
                    overlay_array=overlay_array,
                    planned_slice=planned_slice,
                    extent_mm=plan.get("extent_mm", {}),
                    value=slice_guide_value,
                )
                self._paint_physical_marker(
                    reference_image=reference_image,
                    overlay_array=overlay_array,
                    point=planned_slice["center"],
                    value=center_value,
                )

        output_path = (
            Path(context.request.output_dir)
            / "metadata"
            / f"切片线_{self._debug_plan_suffix(plans)}.nii.gz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    def _write_slice_plan_overview_overlays(
        self,
        context: ProcessingContext,
        localization_result: Dict[str, Any],
        plans: list[Dict[str, Any]],
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        按横、矢、冠状位分别写出含 CT 背景和原始网格总览。

        每个文件只包含一个状位，避免三组互相垂直的切割线叠在一起。含原图
        总览会把当前状位的堆叠轴细化到 0.5mm，并将层厚带和中心线烧录到
        重采样后的 CT 中；原始网格总览保留输入图像网格，用于核对取整位置。
        缺少原始 CT 时只写原始网格总览。
        """

        reference_path = localization_result.get("seg_for_reconstruction_path")
        if not reference_path:
            return {}, {}

        reference_path = Path(reference_path)
        if not reference_path.exists():
            return {}, {}

        reference_image = sitk.ReadImage(str(reference_path))
        raw_path = localization_result.get("raw_for_reconstruction_path")
        raw_image = None
        if raw_path and Path(raw_path).exists():
            raw_image = sitk.ReadImage(str(raw_path))
        plans_by_plane: Dict[str, list[Dict[str, Any]]] = {}
        for plan in plans:
            plane = str(plan.get("plane", ""))
            if plane not in _OVERVIEW_AXES:
                continue
            plans_by_plane.setdefault(plane, []).append(plan)

        metadata_dir = Path(context.request.output_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        overview_paths: Dict[str, str] = {}
        native_overview_paths: Dict[str, str] = {}
        for plane, plane_plans in plans_by_plane.items():
            axes = _OVERVIEW_AXES[plane]
            native_overlay_array = np.zeros(
                tuple(reversed(reference_image.GetSize())),
                dtype=np.uint16,
            )
            uniform_reference = self._build_uniform_overview_reference(
                reference_image,
                axes.stack_axis,
            )
            uniform_overlay_array = np.zeros(
                tuple(reversed(uniform_reference.GetSize())),
                dtype=np.uint16,
            )

            for plan_index, plan in enumerate(plane_plans, start=1):
                overview_value = plan_index * 10 + 4
                thickness_value = plan_index * 10 + 5
                self._paint_plan_overview(
                    reference_image=reference_image,
                    overlay_array=native_overlay_array,
                    plan=plan,
                    axes=axes,
                    value=overview_value,
                    thickness_value=thickness_value,
                )
                self._paint_plan_overview(
                    reference_image=uniform_reference,
                    overlay_array=uniform_overlay_array,
                    plan=plan,
                    axes=axes,
                    value=overview_value,
                    thickness_value=thickness_value,
                )

            debug_suffix = self._debug_plan_suffix(plane_plans)
            output_path = metadata_dir / f"切片总览含原图_{debug_suffix}.nii.gz"
            native_output_path = (
                metadata_dir
                / f"切片原始网格总览_{debug_suffix}.nii.gz"
            )
            if raw_image is not None:
                overview_image = self._build_overview_with_image(
                    raw_image,
                    uniform_reference,
                    uniform_overlay_array,
                )
                sitk.WriteImage(overview_image, str(output_path))
                overview_paths[plane] = str(output_path)
            native_overlay_image = sitk.GetImageFromArray(native_overlay_array)
            native_overlay_image.CopyInformation(reference_image)
            sitk.WriteImage(native_overlay_image, str(native_output_path))
            native_overview_paths[plane] = str(native_output_path)

        return overview_paths, native_overview_paths

    @staticmethod
    def _legacy_overview_path(
        paths_by_plane: Dict[str, str],
        plans: list[Dict[str, Any]],
    ) -> str:
        """返回兼容旧单路径字段的横状位或第一个可用状位文件。"""

        if "横状位" in paths_by_plane:
            return paths_by_plane["横状位"]
        for plan in plans:
            plane = str(plan.get("plane", ""))
            if plane in paths_by_plane:
                return paths_by_plane[plane]
        return ""

    @staticmethod
    def _build_overview_with_image(
        raw_image: sitk.Image,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
    ) -> sitk.Image:
        """把 CT 重采样到总览网格，并烧录层厚带和中心线。"""

        raw_array = sitk.GetArrayViewFromImage(raw_image)
        source_min = float(np.min(raw_array))
        source_max = float(np.max(raw_array))
        int16_min = int(np.iinfo(np.int16).min)
        int16_max = int(np.iinfo(np.int16).max)
        resampled = sitk.Resample(
            raw_image,
            reference_image,
            sitk.Transform(),
            sitk.sitkLinear,
            min(max(source_min, int16_min), int16_max),
            sitk.sitkInt16,
        )
        result_array = sitk.GetArrayFromImage(resampled)
        intensity_span = max(source_max - source_min, 1.0)
        band_value = min(int(round(source_max + intensity_span)), int16_max - 1)
        center_value = min(int(round(band_value + intensity_span)), int16_max)
        if center_value <= band_value:
            band_value = int16_max - 1
            center_value = int16_max
        result_array[overlay_array % 10 == 5] = band_value
        result_array[overlay_array % 10 == 4] = center_value

        result = sitk.GetImageFromArray(result_array)
        result.CopyInformation(reference_image)
        return result

    @staticmethod
    def _build_uniform_overview_reference(
        reference_image: sitk.Image,
        stack_axis: int,
    ) -> sitk.Image:
        """只把当前状位堆叠轴细化为 0.5mm，创建同物理范围检查网格。"""

        if stack_axis not in (0, 1, 2):
            raise ValueError("总览堆叠轴只能是 0、1、2")
        spacing = list(reference_image.GetSpacing())
        size = list(reference_image.GetSize())
        uniform_spacing = 0.5
        physical_extent = (size[stack_axis] - 1) * spacing[stack_axis]
        size[stack_axis] = (
            int(math.ceil(physical_extent / uniform_spacing)) + 1
        )
        spacing[stack_axis] = uniform_spacing
        uniform_reference = sitk.Image(
            int(size[0]),
            int(size[1]),
            int(size[2]),
            sitk.sitkUInt16,
        )
        uniform_reference.SetSpacing(tuple(spacing))
        uniform_reference.SetOrigin(reference_image.GetOrigin())
        uniform_reference.SetDirection(reference_image.GetDirection())
        return uniform_reference

    def _paint_plan_overview(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        plan: Dict[str, Any],
        axes: _OverviewAxes,
        value: int,
        thickness_value: int,
    ) -> None:
        """
        把一组零角度直线 slice 投到当前状位的固定正交层上。

        总览图用不同标签绘制厚度覆盖带和输出切片中心线。当前方法使用输入
        图像 x/y/z 索引轴，只适用于本阶段已经支持的零角度横、矢、冠计划。
        """

        slices = plan.get("slices", [])
        if not slices:
            return

        extent_mm = plan.get("extent_mm", {})
        half_height = float(extent_mm.get("height", 0.0)) / 2.0
        if half_height <= 0:
            return

        center_indices = [
            reference_image.TransformPhysicalPointToContinuousIndex((
                float(planned_slice["center"]["x"]),
                float(planned_slice["center"]["y"]),
                float(planned_slice["center"]["z"]),
            ))
            for planned_slice in slices
        ]
        overview_fixed_index = int(round(float(np.median([
            index_xyz[axes.fixed_axis] for index_xyz in center_indices
        ]))))
        half_height_voxels = (
            half_height / reference_image.GetSpacing()[axes.line_axis]
        )
        half_thickness_voxels = (
            float(plan.get("slice_thickness", 0.0))
            / (2.0 * reference_image.GetSpacing()[axes.stack_axis])
        )

        slice_thickness = float(plan.get("slice_thickness", 0.0))
        center_tick_half_voxels = min(
            half_height_voxels * 0.25,
            5.0 / reference_image.GetSpacing()[axes.line_axis],
        )
        for center_index in center_indices:
            if slice_thickness > 0:
                self._paint_overview_index_band(
                    overlay_array=overlay_array,
                    overview_fixed_index=overview_fixed_index,
                    center_index=center_index,
                    axes=axes,
                    half_height_voxels=half_height_voxels,
                    half_thickness_voxels=half_thickness_voxels,
                    value=thickness_value,
                )

        # 中心线最后画，避免被同一张或相邻切片的厚度带覆盖。
        for center_index in center_indices:
            self._paint_overview_index_line(
                overlay_array=overlay_array,
                overview_fixed_index=overview_fixed_index,
                line_center_index=center_index[axes.line_axis],
                stack_index=int(round(center_index[axes.stack_axis])),
                axes=axes,
                half_height_voxels=center_tick_half_voxels,
                value=value,
            )

    def _paint_overview_index_band(
        self,
        overlay_array: np.ndarray,
        overview_fixed_index: int,
        center_index: tuple[float, float, float],
        axes: _OverviewAxes,
        half_height_voxels: float,
        half_thickness_voxels: float,
        value: int,
    ) -> None:
        """沿当前状位堆叠轴按物理层厚填充总览覆盖带。"""

        stack_center = center_index[axes.stack_axis]
        stack_start = int(math.ceil(
            stack_center - half_thickness_voxels - 1e-9,
        ))
        stack_end = int(math.ceil(
            stack_center + half_thickness_voxels - 1e-9,
        )) - 1
        for stack_index in range(stack_start, stack_end + 1):
            self._paint_overview_index_line(
                overlay_array=overlay_array,
                overview_fixed_index=overview_fixed_index,
                line_center_index=center_index[axes.line_axis],
                stack_index=stack_index,
                axes=axes,
                half_height_voxels=half_height_voxels,
                value=value,
            )

    def _paint_overview_index_line(
        self,
        overlay_array: np.ndarray,
        overview_fixed_index: int,
        line_center_index: float,
        stack_index: int,
        axes: _OverviewAxes,
        half_height_voxels: float,
        value: int,
    ) -> None:
        """在固定正交层上沿当前状位线轴画一条总览线。"""

        line_start = int(round(line_center_index - half_height_voxels))
        line_end = int(round(line_center_index + half_height_voxels))
        size_xyz = tuple(reversed(overlay_array.shape))
        if not 0 <= stack_index < size_xyz[axes.stack_axis]:
            return
        if not 0 <= overview_fixed_index < size_xyz[axes.fixed_axis]:
            return
        line_start = max(line_start, 0)
        line_end = min(line_end, size_xyz[axes.line_axis] - 1)
        for line_index in range(line_start, line_end + 1):
            index_xyz = [0, 0, 0]
            index_xyz[axes.stack_axis] = stack_index
            index_xyz[axes.fixed_axis] = overview_fixed_index
            index_xyz[axes.line_axis] = line_index
            overlay_array[
                index_xyz[2],
                index_xyz[1],
                index_xyz[0],
            ] = value

    @staticmethod
    def _physical_point_to_array_index(
        reference_image: sitk.Image,
        point: Dict[str, float],
    ) -> tuple[int, int, int]:
        """把 x/y/z 物理坐标转换成 numpy 数组使用的 z/y/x 索引。"""

        index_xyz = reference_image.TransformPhysicalPointToIndex((
            float(point["x"]),
            float(point["y"]),
            float(point["z"]),
        ))
        return int(index_xyz[2]), int(index_xyz[1]), int(index_xyz[0])

    @staticmethod
    def _is_inside_array(array: np.ndarray, z: int, y: int, x: int) -> bool:
        """判断 z/y/x 是否落在 overlay 数组内。"""

        z_size, y_size, x_size = array.shape
        return 0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size

    def _paint_physical_line(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        start_point: Dict[str, float],
        end_point: Dict[str, float],
        value: int,
    ) -> None:
        """在 overlay 上画一条连接两个物理坐标点的体素线。"""

        start_zyx = self._physical_point_to_array_index(reference_image, start_point)
        end_zyx = self._physical_point_to_array_index(reference_image, end_point)
        dz = end_zyx[0] - start_zyx[0]
        dy = end_zyx[1] - start_zyx[1]
        dx = end_zyx[2] - start_zyx[2]
        steps = max(abs(dz), abs(dy), abs(dx), 1)

        for step in range(steps + 1):
            z = int(round(start_zyx[0] + dz * step / steps))
            y = int(round(start_zyx[1] + dy * step / steps))
            x = int(round(start_zyx[2] + dx * step / steps))
            if self._is_inside_array(overlay_array, z, y, x):
                overlay_array[z, y, x] = value

    def _paint_planned_slice_guides(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        planned_slice: Dict[str, Any],
        extent_mm: Dict[str, float],
        value: int,
    ) -> None:
        """
        在单张 slice 所在平面内画十字引导线。

        中心路径线在横状位上常常只像一个点，不利于人工检查。这里用
        row_dir/col_dir 和 extent_mm 画两条穿过中心的平面内线，打开
        overlay 时能直接看到“每张图切在哪里”。
        """

        center = planned_slice["center"]
        row_dir = planned_slice["row_dir"]
        col_dir = planned_slice["col_dir"]
        half_width = float(extent_mm.get("width", 0.0)) / 2.0
        half_height = float(extent_mm.get("height", 0.0)) / 2.0

        if half_width > 0:
            self._paint_direction_line(
                reference_image=reference_image,
                overlay_array=overlay_array,
                center=center,
                direction=row_dir,
                half_length=half_width,
                value=value,
            )
        if half_height > 0:
            self._paint_direction_line(
                reference_image=reference_image,
                overlay_array=overlay_array,
                center=center,
                direction=col_dir,
                half_length=half_height,
                value=value,
            )

    def _paint_direction_line(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        center: Dict[str, float],
        direction: Dict[str, float],
        half_length: float,
        value: int,
    ) -> None:
        """沿某个方向从中心点向两侧画一条物理长度线段。"""

        start_point = {
            "x": center["x"] - direction["x"] * half_length,
            "y": center["y"] - direction["y"] * half_length,
            "z": center["z"] - direction["z"] * half_length,
        }
        end_point = {
            "x": center["x"] + direction["x"] * half_length,
            "y": center["y"] + direction["y"] * half_length,
            "z": center["z"] + direction["z"] * half_length,
        }
        self._paint_physical_line(
            reference_image=reference_image,
            overlay_array=overlay_array,
            start_point=start_point,
            end_point=end_point,
            value=value,
        )

    def _paint_physical_marker(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        point: Dict[str, float],
        value: int,
    ) -> None:
        """在 overlay 上标出一张 slice 的中心点，画成小十字方便查看。"""

        center_z, center_y, center_x = self._physical_point_to_array_index(
            reference_image,
            point,
        )
        offsets = [
            (0, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        for dz, dy, dx in offsets:
            z = center_z + dz
            y = center_y + dy
            x = center_x + dx
            if self._is_inside_array(overlay_array, z, y, x):
                overlay_array[z, y, x] = value
