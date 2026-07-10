#!/src/planning/single_label.py
# -*- coding: utf-8 -*-

"""单目标整切 planner。"""

from typing import Any, Dict

from src.models.enums import PlaneTypeEnum
from src.planning.base import PlanningResult
from src.planning.whole_range import build_axial_whole_range_plan_from_bbox


class SingleLabelWholeRangePlanner:
    """肺部、头骨等单目标整切的兜底规划器。"""

    planner_name = "SingleLabelWholeRangePlanner"

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """根据第三步输出的整体 bbox 生成横状位整切计划。"""

        plans = []
        warnings = []
        for plane_config in request.plane_configs:
            if plane_config.plane != PlaneTypeEnum.横状位:
                # 状位参数允许提前传入，便于后续逐步实现；当前不能静默丢弃，
                # 必须把未执行原因写入阶段四和调试报告。
                warnings.append(
                    f"暂不支持{request.body_part.value}整切状位: "
                    f"{plane_config.plane.value}"
                )
                continue
            plans.append(
                build_axial_whole_range_plan_from_bbox(
                    request=request,
                    plane_config=plane_config,
                    localization_result=localization_result,
                )
            )

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )
