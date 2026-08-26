#!/src/services/stage_planning.py
# -*- coding: utf-8 -*-

"""
切片规划实现：正式主链中为第六阶段。

本阶段负责回答“要切哪些图”。它会根据第四阶段定位结果、第五阶段解剖几何、部位、
切割方式和状位配置生成 SlicePlan[]，并把每一张待切图的中心点和方向提前展开好。
重建阶段只负责执行这些计划，不再重新决定切片数量和切片位置。
"""

import json
import time
from collections import Counter
from copy import copy
from pathlib import Path
from typing import Dict, Any

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import PlanningStageResult, Point3D
from src.planning.geometry import normalize_vector, subtract_points, vector_length
from src.planning.straight import CENTER_COVERAGE_TOLERANCE_MM
from src.planning import select_planner
from src.utils import DebugLog, is_debug_file_enabled


class PlanningStage(BaseStage):
    """把定位和解剖几何展开成可直接重建的 SlicePlan。"""

    def __init__(self, config: Dict[str, Any] = None):
        """初始化切片规划组件及其调试输出配置。"""

        super().__init__(name="planning", config=config)

    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        根据定位和解剖分析结果生成重建阶段可直接执行的切片计划。

        请求中的每个状位按外部指定的 SliceType 选择 planner，再统一写出
        `slice_plans.json`。规划阶段不再生成 NIfTI 调试总览。
        """

        prev_result = context.get_result('localization')

        if not prev_result:
            raise ValueError("前置阶段 'localization' 结果不存在")

        # 纯定位与解剖几何分属两个结果；部位 planner 消费二者的只读展平视图。
        anatomical_result = context.get_result('anatomical_analysis')
        if anatomical_result is None:
            raise ValueError("前置阶段 'anatomical_analysis' 结果不存在")
        prev_result = {**prev_result, **anatomical_result}

        request = context.request
        self._assign_plane_config_ids(request.plane_configs)
        coords = prev_result.get('coords', {})

        DebugLog("=" * 60)
        DebugLog("【第六阶段】切片规划 - 生成重建执行计划")
        DebugLog(f"  切割部位: {request.slice_part.value}")
        DebugLog(
            "  SliceType: "
            + ", ".join(config.slice_type.value for config in request.plane_configs)
        )
        DebugLog(f"  定位坐标数量: {len(coords)}")
        DebugLog("=" * 60)

        # 规划阶段内部拆成 planner + expander。planner 负责找区间和路径；
        # expander 已在 planner 内部调用，把路径展开成每张图的 slices[]。
        DebugLog("  [1] select_planner: 选择切割部位/切法 planner...")
        grouped_configs: Dict[str, list[Any]] = {}
        for plane_config in request.plane_configs:
            grouped_configs.setdefault(
                plane_config.slice_type.value,
                [],
            ).append(plane_config)

        DebugLog("  [2] build_plans: 生成并展开切片计划...")
        all_plans = []
        all_warnings = []
        planner_names = []
        for slice_type, plane_configs in grouped_configs.items():
            planner = select_planner(request.slice_part.value, slice_type)
            planner_names.append(planner.planner_name)
            DebugLog(f"    planner: {planner.planner_name}")
            scoped_request = copy(request)
            scoped_request.plane_configs = plane_configs
            planner_start = time.perf_counter()
            partial_result = planner.build_plans(scoped_request, prev_result)
            DebugLog(
                "    [切片规划性能] build_plans完成: "
                f"planner={planner.planner_name}, "
                f"plane_configs={len(plane_configs)}, "
                f"plans={len(partial_result.plans)}, "
                f"warnings={len(partial_result.warnings)}, "
                f"耗时={time.perf_counter() - planner_start:.3f}s"
            )
            all_plans.extend(partial_result.plans)
            all_warnings.extend(partial_result.warnings)
        for plan in all_plans:
            if plan.output_plane and plan.output_plane != plan.plane:
                plan.plan_id = f"{plan.plan_id}_{plan.config_id}"
        planner_name = "+".join(planner_names)
        plans = [plan.to_dict() for plan in all_plans]
        diagnostics = self._build_plan_diagnostics(plans)

        metadata_path = ""
        if is_debug_file_enabled():
            DebugLog("  [3] write_metadata: 写出 slice_plans.json...")
            metadata_start = time.perf_counter()
            metadata_path = self._write_slice_plans(
                context,
                planner_name=planner_name,
                plans=plans,
                diagnostics=diagnostics,
                warnings=all_warnings,
            )
            DebugLog(
                "    [切片规划性能] write_metadata完成: "
                f"耗时={time.perf_counter() - metadata_start:.3f}s"
            )

        result = {
            'planner': planner_name,
            'plans': plans,
            'warnings': all_warnings,
            'diagnostics': diagnostics,
            'plan_count': len(plans),
            'metadata_path': str(metadata_path),
        }

        context.set_stage_result(PlanningStageResult(payload=result))

        DebugLog(f"    切片计划数量: {len(plans)}")
        for diagnostic in diagnostics:
            DebugLog(
                "    "
                f"{diagnostic['plan_id']}: "
                f"数量 {diagnostic['slice_count']} 张, "
                f"层厚 {diagnostic['slice_thickness']}mm, "
                f"层间距 {diagnostic['slice_spacing']}mm, "
                f"有效范围 {diagnostic['effective_range_mm']}mm, "
                f"总范围 {diagnostic['total_range_mm']}mm"
            )
        if all_warnings:
            DebugLog(f"    warnings: {all_warnings}")
        DebugLog("  第六阶段完成")
        DebugLog("=" * 60)

        return context

    @staticmethod
    def _assign_plane_config_ids(plane_configs: list[Any]) -> None:
        """生成配置标识，并按状位出现顺序分配唯一输出名称。"""

        plane_ordinals: Counter[str] = Counter()
        for ordinal, config in enumerate(plane_configs, start=1):
            plane = config.plane.value
            plane_ordinal = plane_ordinals[plane]
            config.config_id = f"配置{ordinal:02d}"
            config.output_plane = (
                plane
                if plane_ordinal == 0
                else f"{plane}{plane_ordinal}"
            )
            plane_ordinals[plane] += 1

    def _write_slice_plans(
        self,
        context: ProcessingContext,
        planner_name: str,
        plans: list[Dict[str, Any]],
        diagnostics: list[Dict[str, Any]],
        warnings: list[str],
    ) -> Path:
        """把规划输出写入 metadata/slice_plans.json，方便重建和人工检查。"""

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
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return metadata_path

    def _build_plan_diagnostics(self, plans: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        为每个 plan 生成可读诊断信息。

        这部分不参与第七阶段执行，只用于从 JSON 排查计划张数和物理覆盖范围。
        尤其当层间距大于层厚时，输出图之间本来就会有物理空档，需要在诊断字段
        中明确表达。
        """

        diagnostics = []
        for plan in plans:
            range_start = plan.get("range_start", {})
            range_end = plan.get("range_end", {})
            slice_spacing = float(plan.get("slice_spacing", 0.0))
            slice_thickness = float(plan.get("slice_thickness", 0.0))
            sampling_details = plan.get("sampling_details") or {}
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
            regular_center_count = slice_count
            has_extra_endpoint_slice = False
            leading_uncovered = 0.0
            trailing_uncovered = 0.0
            leading_overflow = 0.0
            trailing_overflow = 0.0
            leading_center_uncovered = 0.0
            trailing_center_uncovered = 0.0
            leading_center_overflow = 0.0
            trailing_center_overflow = 0.0
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
                leading_gap = (
                    leading_delta.x * coverage_direction.x
                    + leading_delta.y * coverage_direction.y
                    + leading_delta.z * coverage_direction.z
                )
                trailing_gap = (
                    trailing_delta.x * coverage_direction.x
                    + trailing_delta.y * coverage_direction.y
                    + trailing_delta.z * coverage_direction.z
                )
                leading_center_delta = subtract_points(
                    end=slice_centers[0],
                    start=target_start,
                )
                trailing_center_delta = subtract_points(
                    end=target_end,
                    start=slice_centers[-1],
                )
                leading_center_gap = (
                    leading_center_delta.x * coverage_direction.x
                    + leading_center_delta.y * coverage_direction.y
                    + leading_center_delta.z * coverage_direction.z
                )
                trailing_center_gap = (
                    trailing_center_delta.x * coverage_direction.x
                    + trailing_center_delta.y * coverage_direction.y
                    + trailing_center_delta.z * coverage_direction.z
                )
                if abs(leading_gap) < 1e-6:
                    leading_gap = 0.0
                if abs(trailing_gap) < 1e-6:
                    trailing_gap = 0.0
                # 覆盖诊断与直线计划共用同一物理坐标容差，避免已被
                # 计划吸收的数微米误差又被报成首尾缺失。
                if abs(leading_center_gap) <= CENTER_COVERAGE_TOLERANCE_MM:
                    leading_center_gap = 0.0
                if abs(trailing_center_gap) <= CENTER_COVERAGE_TOLERANCE_MM:
                    trailing_center_gap = 0.0
                leading_uncovered = max(leading_gap, 0.0)
                trailing_uncovered = max(trailing_gap, 0.0)
                leading_overflow = max(-leading_gap, 0.0)
                trailing_overflow = max(-trailing_gap, 0.0)
                leading_center_uncovered = max(leading_center_gap, 0.0)
                trailing_center_uncovered = max(trailing_center_gap, 0.0)
                leading_center_overflow = max(-leading_center_gap, 0.0)
                trailing_center_overflow = max(-trailing_center_gap, 0.0)
            effective_range = float(
                sampling_details.get("effective_range_mm", target_length)
            )
            total_range = float(
                sampling_details.get(
                    "total_range_mm",
                    slice_thickness + max(0, slice_count - 1) * slice_spacing,
                )
            )
            calculation_note = (
                "第六阶段根据层厚、层间距、数量中的两个或三个参数计算实际采样，"
                "总范围超过有效范围的部分在开始和结束方向均等外扩。"
            )
            adjustment_reason = str(
                sampling_details.get("adjustment_reason", "")
            )
            if adjustment_reason:
                calculation_note += f" {adjustment_reason}。"
            if spacing_minus_thickness > 0:
                calculation_note += (
                    f" 当前层间距比层厚大 {spacing_minus_thickness:.3f}mm，"
                    "首尾仍完整覆盖，但厚层图之间会保留配置产生的空档；"
                    "需要更多输出图时应减小层间距。"
                )

            diagnostics.append({
                "plan_id": plan.get("plan_id", ""),
                "plane": plan.get("plane", ""),
                "range_length_mm": round(range_length, 3),
                "slice_count": slice_count,
                "slice_image_count": slice_count,
                "effective_range_mm": round(effective_range, 3),
                "total_range_mm": round(total_range, 3),
                "sampling_adjusted": bool(
                    sampling_details.get("adjusted", False)
                ),
                "sampling_adjustment_reason": adjustment_reason,
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
                "leading_overflow_mm": round(leading_overflow, 3),
                "trailing_overflow_mm": round(trailing_overflow, 3),
                "leading_center_uncovered_mm": round(
                    leading_center_uncovered,
                    3,
                ),
                "trailing_center_uncovered_mm": round(
                    trailing_center_uncovered,
                    3,
                ),
                "leading_center_overflow_mm": round(
                    leading_center_overflow,
                    3,
                ),
                "trailing_center_overflow_mm": round(
                    trailing_center_overflow,
                    3,
                ),
                "slice_thickness": slice_thickness,
                "slice_spacing": slice_spacing,
                "spacing_minus_thickness": round(spacing_minus_thickness, 3),
                "calculation_note": calculation_note,
            })
        return diagnostics

    @staticmethod
    def _dict_to_point(point: Dict[str, Any]):
        """把 JSON 中的 x/y/z 字典转换成 Point3D，用于诊断距离计算。"""

        return Point3D(
            x=float(point["x"]),
            y=float(point["y"]),
            z=float(point["z"]),
        )
