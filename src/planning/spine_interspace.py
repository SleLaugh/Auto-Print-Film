"""脊柱横状位椎间隙规划公共实现。"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.models.enums import PlaneTypeEnum, SliceType
from src.planning.base import PlanningResult
from src.planning.spine_common import (
    resolve_spine_interspace_segments,
    resolve_spine_localizer_reference,
)
from src.planning.straight import (
    CutSegment,
    build_straight_slice_plan,
    default_straight_frame,
)


InterspaceDefinition = Sequence[Tuple[str, str, str]]
SegmentResolver = Callable[
    [Dict[str, Any]],
    Tuple[List[CutSegment], List[str]],
]


def build_spine_interspace_plans(
    request: Any,
    localization_result: Dict[str, Any],
    *,
    label_names: Sequence[str],
    intervals: InterspaceDefinition,
    planner_name: str,
    segment_resolver: Optional[SegmentResolver] = None,
) -> PlanningResult:
    """按部位区间配置生成横状位椎间隙计划。"""

    plans = []
    localizer_reference = resolve_spine_localizer_reference(
        localization_result,
        label_names,
    )
    if segment_resolver is None:
        segments, warnings = resolve_spine_interspace_segments(
            localization_result,
            intervals,
        )
    else:
        segments, warnings = segment_resolver(localization_result)
    for segment in segments:
        for plane_config in request.plane_configs:
            if plane_config.plane != PlaneTypeEnum.横状位:
                warnings.append(
                    f"暂不支持{request.study_part.value}椎间隙状位: "
                    f"{plane_config.plane.value}"
                )
                continue

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
            plan.localizer_plane = PlaneTypeEnum.矢状位.value
            plan.localizer_reference = localizer_reference
            plans.append(plan)

    return PlanningResult(
        plans=plans,
        warnings=warnings,
        planner_name=planner_name,
    )
