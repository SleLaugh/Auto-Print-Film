#!/src/planning/lumbar.py
# -*- coding: utf-8 -*-

"""腰椎切片 planner。"""

from typing import Any, Dict, List, Tuple

from src.models.enums import PlaneTypeEnum, SliceType
from src.planning.base import PlanningResult
from src.planning.spine_cpr import (
    build_spine_sagittal_cpr_plans,
    configured_spine_cpr_labels,
    configured_whole_range_spine_labels,
)
from src.planning.spine_common import (
    resolve_spine_interspace_segments,
    resolve_spine_localizer_reference,
)
from src.planning.spine_interspace import (
    build_spine_interspace_plans,
    configured_interspace_definitions,
)
from src.planning.straight import CutSegment
from src.planning.whole_range import build_whole_range_plan_from_bbox


def resolve_lumbar_interspace_segments(
    localization_result: Dict[str, Any],
) -> Tuple[List[CutSegment], List[str]]:
    """把腰椎相邻椎体解析成多个独立待切区间及其 warning。"""

    return resolve_spine_interspace_segments(
        localization_result,
        configured_interspace_definitions(localization_result),
    )


class LumbarWholeRangePlanner:
    """复用整体 bbox 生成腰椎三状位默认直线整切计划。"""

    planner_name = "LumbarWholeRangePlanner"
    """写入第六阶段结果和调试报告的稳定 planner 名称。"""

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """按腰椎整体 bbox 为请求中的每个状位展开整切切片。"""

        plans = []
        warnings = []
        spine_labels = configured_whole_range_spine_labels(
            localization_result,
        ) or []
        localizer_reference = resolve_spine_localizer_reference(
            localization_result,
            spine_labels,
        )
        cpr_labels = configured_spine_cpr_labels(
            localization_result,
        )
        for plane_config in request.plane_configs:
            if plane_config.slice_type is SliceType.CPR:
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
            plan = build_whole_range_plan_from_bbox(
                request=request,
                plane_config=plane_config,
                localization_result=localization_result,
            )
            if plane_config.plane == PlaneTypeEnum.矢状位:
                # 矢状位切片保持旧的直线 MPR；定位图单独选择穿过腰椎主体的
                # 冠状层，不能复用整平面成片为了显示覆盖而移动后的中心。
                plan.localizer_plane = PlaneTypeEnum.冠状位.value
                plan.localizer_reference = localizer_reference
            plans.append(plan)

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )


class LumbarInterspacePlanner:
    """腰椎横状位椎间隙规划器。"""

    planner_name = "LumbarInterspacePlanner"

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """逐个解析椎间区间，并通过通用直线构造器生成横状位计划。"""

        return build_spine_interspace_plans(
            request,
            localization_result,
            planner_name=self.planner_name,
        )
