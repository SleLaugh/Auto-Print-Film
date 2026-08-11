#!/src/planning/single_label.py
# -*- coding: utf-8 -*-

"""单目标整切 planner。"""

from typing import Any, Dict

from src.models.enums import PlaneTypeEnum
from src.planning.spine_cpr import (
    SPINE_CPR_LABELS,
    build_spine_sagittal_cpr_plans,
    configured_spine_cpr_labels,
    spine_target_bbox,
)
from src.planning.base import PlanningResult
from src.planning.whole_range import build_whole_range_plan_from_bbox


class SingleLabelWholeRangePlanner:
    """肺部、头骨等单目标三状位直线整切的兜底规划器。"""

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
        spine_labels = SPINE_CPR_LABELS.get(request.body_part.value)
        if spine_labels is not None:
            spine_labels = configured_spine_cpr_labels(
                request.body_part.value,
                localization_result,
                spine_labels,
            )
        for plane_config in request.plane_configs:
            target_region_bbox = None
            source_labels = None
            if spine_labels is not None:
                cpr_plans, cpr_warnings = build_spine_sagittal_cpr_plans(
                    request=request,
                    plane_config=plane_config,
                    localization_result=localization_result,
                    label_names=spine_labels,
                )
                warnings.extend(cpr_warnings)
                if cpr_plans:
                    plans.extend(cpr_plans)
                    continue
                if (
                    request.body_part.value == "胸椎"
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
            if spine_labels is not None and plane_config.plane == PlaneTypeEnum.矢状位:
                # 脊柱矢状位恢复为普通直线 MPR，定位线继续画在冠状位背景上。
                plan.localizer_plane = PlaneTypeEnum.冠状位.value
            if (
                request.body_part.value == "胸椎"
                and plane_config.plane == PlaneTypeEnum.冠状位
                and target_region_bbox is not None
            ):
                plan.localizer_plane = PlaneTypeEnum.矢状位.value
            plans.append(plan)

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )
