#!/src/planning/lumbar.py
# -*- coding: utf-8 -*-

"""腰椎切片 planner。"""

from typing import Any, Dict, List, Tuple

from src.models.enums import PlaneTypeEnum
from src.planning.base import PlanningResult
from src.planning.spine_common import (
    build_axial_spine_plan,
    get_label_center,
)
from src.planning.whole_range import build_whole_range_plan_from_bbox


LUMBAR_LABELS = [
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "vertebrae_S1",
]

LUMBAR_INTERVALS: List[Tuple[str, str, str]] = [
    ("L1-L2", "vertebrae_L1", "vertebrae_L2"),
    ("L2-L3", "vertebrae_L2", "vertebrae_L3"),
    ("L3-L4", "vertebrae_L3", "vertebrae_L4"),
    ("L4-L5", "vertebrae_L4", "vertebrae_L5"),
    ("L5-S1", "vertebrae_L5", "vertebrae_S1"),
]


class LumbarWholeRangePlanner:
    """复用整体 bbox 生成腰椎三状位默认直线整切计划。"""

    planner_name = "LumbarWholeRangePlanner"
    """写入阶段四结果和调试报告的稳定 planner 名称。"""

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """按腰椎整体 bbox 为请求中的每个状位展开整切切片。"""

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


class LumbarInterspacePlanner:
    """腰椎区间切 planner，第一版只生成横状位 plane_stack。"""

    planner_name = "LumbarInterspacePlanner"

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """逐个找到椎间区间，并为每个可用区间展开自己的切片清单。"""

        plans = []
        warnings: List[str] = []
        for group_name, start_label, end_label in LUMBAR_INTERVALS:
            start = get_label_center(localization_result, start_label)
            end = get_label_center(localization_result, end_label)
            if not start or not end:
                warnings.append(f"跳过腰椎区间 {group_name}: 缺少端点 label")
                continue

            for plane_config in request.plane_configs:
                if plane_config.plane != PlaneTypeEnum.横状位:
                    warnings.append(f"暂不支持腰椎区间切状位: {plane_config.plane.value}")
                    continue
                plans.append(
                    build_axial_spine_plan(
                        request=request,
                        plane_config=plane_config,
                        group_name=group_name,
                        range_start=start,
                        range_end=end,
                        localization_result=localization_result,
                        source_labels=[start_label, end_label],
                    )
                )

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )
