#!/src/planning/single_label.py
# -*- coding: utf-8 -*-

"""单目标整切 planner。"""

from typing import Any, Dict

from src.models.enums import PlaneTypeEnum, SliceType
from src.planning.spine_cpr import (
    build_spine_sagittal_cpr_plans,
    configured_spine_cpr_labels,
    spine_target_bbox,
)
from src.planning.base import PlanningResult
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.planning.straight import resolve_range_extension


class SingleLabelWholeRangePlanner:
    """肺部、颅脑等单目标三状位直线整切的兜底规划器。"""

    planner_name = "SingleLabelWholeRangePlanner"
    """写入第六阶段结果和调试报告的稳定 planner 名称。"""

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """根据第四阶段整体 bbox 为请求中的每个状位生成直线整切计划。"""

        plans = []
        warnings = []
        spine_labels = configured_spine_cpr_labels(
            localization_result,
        )
        for plane_config in request.plane_configs:
            target_region_bbox = None
            # 颅脑基础范围只用配置内的脑区标签；可选头骨是独立取景支持，
            # 不能通过 source_labels 再次把面部和下颌纳入实际切片范围。
            source_labels = (
                list(localization_result.get("label_targets") or {})
                if request.slice_part.value == "颅脑" else None
            )
            if (
                spine_labels
                and plane_config.slice_type is SliceType.CPR
            ):
                cpr_plans, cpr_warnings = build_spine_sagittal_cpr_plans(
                    request=request,
                    plane_config=plane_config,
                    localization_result=localization_result,
                    label_names=spine_labels,
                )
                warnings.extend(cpr_warnings)
                if not cpr_plans:
                    raise ValueError("CPR 规划失败: " + "; ".join(cpr_warnings))
                plans.extend(cpr_plans)
                continue
            if (
                spine_labels
                and request.slice_part.value == "胸椎"
                and plane_config.plane == PlaneTypeEnum.冠状位
            ):
                target_region_bbox, source_labels = spine_target_bbox(
                    localization_result,
                    spine_labels,
                )
            plan = build_whole_range_plan_from_bbox(
                request=request,
                plane_config=plane_config,
                localization_result=localization_result,
                source_labels=source_labels,
                target_region_bbox=target_region_bbox,
            )
            if (
                request.slice_part.value == "颅脑"
                and plane_config.plane == PlaneTypeEnum.横状位
                and plane_config.slice_type is SliceType.MPR
            ):
                # 横状位从上向下输出，start_mm 是颅顶余量。将本计划实际
                # 使用的配置快照交给取景阶段，避免新增顶部切片仍只按脑区取景。
                plan.framing_superior_extension_mm = resolve_range_extension(
                    localization_result, plane_config.slice_type.value, plane_config.plane,
                ).start_mm
            plans.append(plan)

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )
