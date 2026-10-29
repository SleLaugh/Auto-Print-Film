#!/src/planning/single_label.py
# -*- coding: utf-8 -*-

"""普通整切与显式脊柱 CPR 规划。"""

from copy import deepcopy
from typing import Any, Dict

from src.models.enums import PlaneTypeEnum, SliceType
from src.planning.spine_cpr import (
    build_spine_sagittal_cpr_plans,
    configured_spine_cpr_labels,
    configured_whole_range_spine_labels,
    spine_target_bbox,
)
from src.planning.base import PlanningResult
from src.planning.whole_range import build_whole_range_plan_from_bbox
from src.planning.face_extension import resolve_face_extension


class SingleLabelWholeRangePlanner:
    """复用整体范围生成三状位 MPR，并处理显式脊柱 CPR。"""

    planner_name = "SingleLabelWholeRangePlanner"
    """写入第六阶段结果和调试报告的稳定 planner 名称。"""

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """按请求的显式切法生成整体 MPR 或脊柱 CPR 计划。"""

        plans = []
        warnings = []
        spine_labels = configured_whole_range_spine_labels(
            localization_result,
        ) or []
        cpr_labels = configured_spine_cpr_labels(
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
                cpr_labels
                and plane_config.slice_type is SliceType.CPR
            ):
                cpr_plans, cpr_warnings = build_spine_sagittal_cpr_plans(
                    request=request,
                    plane_config=plane_config,
                    localization_result=localization_result,
                    label_names=cpr_labels,
                )
                warnings.extend(cpr_warnings)
                if not cpr_plans:
                    raise ValueError("CPR 规划失败: " + "; ".join(cpr_warnings))
                plans.extend(cpr_plans)
                continue
            if (
                spine_labels
                and request.slice_part.value in {"胸椎", "胸腰椎", "全脊柱"}
                and plane_config.plane == PlaneTypeEnum.冠状位
            ):
                # 普通 MPR 覆盖整体范围，不应用 CPR 专用的椎体排除项。
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
            if plane_config.slice_type is SliceType.MPR:
                slice_rules = localization_result.get("slice_rules")
                if isinstance(slice_rules, dict):
                    plane_rules = slice_rules.get(plane_config.plane.value)
                    if isinstance(plane_rules, dict):
                        rule = plane_rules.get("MPR")
                        if isinstance(rule, dict):
                            plan.framing_policy = deepcopy(
                                rule.get("framing_policy"),
                            )
            if (
                request.slice_part.value == "颅脑"
                and plane_config.plane == PlaneTypeEnum.横状位
                and plane_config.slice_type is SliceType.MPR
            ):
                # 横状位从上向下输出，top_mm 是颅顶余量。将本计划实际
                # 使用的配置快照交给取景阶段，避免新增顶部切片仍只按脑区取景。
                face_extension = resolve_face_extension(
                    localization_result,
                    plane_config.slice_type.value,
                    plane_config.plane,
                )
                plan.framing_superior_extension_mm = (
                    face_extension.top_mm
                    if face_extension is not None
                    else 0.0
                )
            plans.append(plan)

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )
