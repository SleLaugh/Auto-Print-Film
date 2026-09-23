"""脊柱横状位椎间隙规划公共实现。"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.enums import PlaneTypeEnum, SliceType
from src.planning.base import PlanningResult
from src.planning.spine_common import (
    resolve_spine_interspace_segments,
    resolve_spine_localizer_reference,
)
from src.planning.spine_cpr import configured_whole_range_spine_labels
from src.planning.straight import (
    CutSegment,
    build_straight_slice_plan,
    default_straight_frame,
)


SegmentResolver = Callable[
    [Dict[str, Any]],
    Tuple[List[CutSegment], List[str]],
]


def configured_interspace_definitions(
    localization_result: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """从统一切割部位配置的解析结果读取椎间隙名称和端点标签。"""

    resolved_configs = localization_result.get("resolved_cut_configs")
    interspace_config = (
        resolved_configs.get("interspace")
        if isinstance(resolved_configs, dict)
        else None
    )
    intervals = (
        interspace_config.get("intervals")
        if isinstance(interspace_config, dict)
        else None
    )
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("当前定位结果缺少椎间隙区域配置")

    definitions = []
    for interval in intervals:
        region = interval.get("region")
        start_labels = interval.get("start_labels")
        end_labels = interval.get("end_labels")
        if (
            not isinstance(region, str)
            or not region
            or not isinstance(start_labels, list)
            or not start_labels
            or not isinstance(end_labels, list)
            or not end_labels
        ):
            raise ValueError(f"椎间隙区域配置无效: {interval}")
        definitions.append((region, start_labels[0], end_labels[0]))
    return definitions


def build_spine_interspace_plans(
    request: Any,
    localization_result: Dict[str, Any],
    *,
    planner_name: str,
    segment_resolver: Optional[SegmentResolver] = None,
) -> PlanningResult:
    """按部位区间配置生成横状位椎间隙计划。"""

    plans = []
    intervals = configured_interspace_definitions(localization_result)
    label_names = configured_whole_range_spine_labels(localization_result) or []
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
                    f"暂不支持{request.slice_part.value}椎间隙状位: "
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
            plan.localizer_reference = localizer_reference
            plans.append(plan)

    return PlanningResult(
        plans=plans,
        warnings=warnings,
        planner_name=planner_name,
    )
