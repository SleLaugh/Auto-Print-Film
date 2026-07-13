#!/src/planning/single_label.py
# -*- coding: utf-8 -*-

"""单目标整切 planner。"""

from typing import Any, Dict

from src.planning.base import PlanningResult
from src.planning.whole_range import build_whole_range_plan_from_bbox


class SingleLabelWholeRangePlanner:
    """肺部、头骨等单目标三状位直线整切的兜底规划器。"""

    planner_name = "SingleLabelWholeRangePlanner"
    """写入阶段四结果和调试报告的稳定 planner 名称。"""

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """根据第三步整体 bbox 为请求中的每个状位生成直线整切计划。"""

        plans = []
        for plane_config in request.plane_configs:
            plans.append(
                build_whole_range_plan_from_bbox(
                    request=request,
                    plane_config=plane_config,
                    localization_result=localization_result,
                )
            )

        return PlanningResult(
            plans=plans,
            warnings=[],
            planner_name=self.planner_name,
        )
