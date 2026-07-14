#!/src/planning/lumbar.py
# -*- coding: utf-8 -*-

"""腰椎切片 planner。"""

from typing import Any, Dict, List, Tuple

from src.models.enums import PlaneTypeEnum
from src.planning.base import PlanningResult
from src.planning.spine_common import resolve_spine_interspace_segments
from src.planning.straight import (
    CutSegment,
    build_straight_slice_plan,
    default_straight_frame,
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


def resolve_lumbar_interspace_segments(
    localization_result: Dict[str, Any],
) -> Tuple[List[CutSegment], List[str]]:
    """把腰椎相邻椎体解析成多个独立待切区间及其 warning。"""

    return resolve_spine_interspace_segments(
        localization_result,
        LUMBAR_INTERVALS,
    )


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
        """逐个解析椎间区间，并通过通用直线构造器生成横状位计划。"""

        plans = []
        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )
        for segment in segments:
            for plane_config in request.plane_configs:
                if plane_config.plane != PlaneTypeEnum.横状位:
                    warnings.append(f"暂不支持腰椎区间切状位: {plane_config.plane.value}")
                    continue

                # 方向解析必须放在区间循环内部：当前返回零角度横状位；后续可在此
                # 根据矢状位上的当前椎间隙方向，为每个 CutSegment 单独计算倾角。
                frame = default_straight_frame(
                    plane_config.plane,
                    localization_result["image"],
                )
                plans.append(
                    build_straight_slice_plan(
                        request=request,
                        cut_mode="interspace",
                        plane_config=plane_config,
                        localization_result=localization_result,
                        segment=segment,
                        frame=frame,
                    )
                )

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )
