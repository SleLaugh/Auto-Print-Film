#!/src/planning/lumbar.py
# -*- coding: utf-8 -*-

"""腰椎切片 planner。"""

from typing import Any, Dict, List, Tuple

from src.models.enums import PlaneTypeEnum, SliceType
from src.planning.base import PlanningResult
from src.planning.spine_cpr import (
    build_spine_sagittal_cpr_plans,
    configured_spine_cpr_labels,
)
from src.planning.spine_common import (
    resolve_spine_interspace_segments,
    resolve_spine_localizer_reference,
)
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
    ("L1_L2", "vertebrae_L1", "vertebrae_L2"),
    ("L2_L3", "vertebrae_L2", "vertebrae_L3"),
    ("L3_L4", "vertebrae_L3", "vertebrae_L4"),
    ("L4_L5", "vertebrae_L4", "vertebrae_L5"),
    ("L5_S1", "vertebrae_L5", "vertebrae_S1"),
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
    """写入第六阶段结果和调试报告的稳定 planner 名称。"""

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """按腰椎整体 bbox 为请求中的每个状位展开整切切片。"""

        plans = []
        warnings = []
        localizer_reference = resolve_spine_localizer_reference(
            localization_result,
            LUMBAR_LABELS,
        )
        cpr_labels = configured_spine_cpr_labels(
            request.study_part.value,
            localization_result,
            LUMBAR_LABELS,
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
    """腰椎区间切 planner，第一版只生成横状位 plane_stack。"""

    planner_name = "LumbarInterspacePlanner"

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """逐个解析椎间区间，并通过通用直线构造器生成横状位计划。"""

        plans = []
        localizer_reference = resolve_spine_localizer_reference(
            localization_result,
            LUMBAR_LABELS,
        )
        segments, warnings = resolve_lumbar_interspace_segments(
            localization_result,
        )
        for segment in segments:
            for plane_config in request.plane_configs:
                if plane_config.plane != PlaneTypeEnum.横状位:
                    warnings.append(f"暂不支持腰椎区间切状位: {plane_config.plane.value}")
                    continue

                # 方向解析必须放在区间循环内部：每个椎间可携带自己的端面斜横切坐标系，
                # 定位结果没有该字段时才回退到零角度横状位。
                frame = segment.frame or default_straight_frame(
                    plane_config.plane,
                    localization_result["image"],
                )
                plan = build_straight_slice_plan(
                    request=request,
                    slice_type=SliceType.椎间隙.value,
                    plane_config=plane_config,
                    localization_result=localization_result,
                    segment=segment,
                    frame=frame,
                )
                # 五个椎间计划只共享定位图固定层，各自的斜切中心和方向不变。
                plan.localizer_plane = PlaneTypeEnum.矢状位.value
                plan.localizer_reference = localizer_reference
                plans.append(plan)

        return PlanningResult(
            plans=plans,
            warnings=warnings,
            planner_name=self.planner_name,
        )
