"""脊柱横状位椎间隙规划公共实现。"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from src.anatomy.spine_labels import resolve_spine_label_name
from src.models import Point3D
from src.models.enums import PlaneTypeEnum, SliceType
from src.planning.base import PlanningResult
from src.planning.spine_common import (
    resolve_spine_interspace_segments,
    resolve_spine_localizer_reference,
)
from src.planning.spine_cpr import configured_whole_range_spine_labels
from src.planning.straight import (
    CutSegment,
    StraightPlaneFrame,
    build_straight_slice_plan,
    default_straight_frame,
)


SegmentResolver = Callable[
    [Dict[str, Any]],
    Tuple[List[CutSegment], List[str]],
]


def _source_aligned_display_frame(
    frame: StraightPlaneFrame,
    reference: StraightPlaneFrame,
) -> StraightPlaneFrame:
    """保留切平面和左右方向，让画面上下方向与原图一致。"""

    alignment = (
        frame.col_dir.x * reference.col_dir.x
        + frame.col_dir.y * reference.col_dir.y
        + frame.col_dir.z * reference.col_dir.z
    )
    if alignment >= 0.0:
        return frame
    # 区间几何法向指向上位→下位，不能直接决定画面上下。同步反转
    # 显示列轴和法向以保留右手坐标系；端点与切片排列顺序不变。
    return StraightPlaneFrame(
        row_dir=frame.row_dir,
        col_dir=Point3D(-frame.col_dir.x, -frame.col_dir.y, -frame.col_dir.z),
        normal=Point3D(-frame.normal.x, -frame.normal.y, -frame.normal.z),
    )


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

    # 持久配置保留标准标签名，执行计划必须使用本次已定位的模型标签。
    available_labels = (
        localization_result.get("label_targets") or localization_result.get("coords") or {}
    )
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
        definitions.append((
            region,
            resolve_spine_label_name(start_labels[0], available_labels),
            resolve_spine_label_name(end_labels[0], available_labels),
        ))
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

            reference_frame = default_straight_frame(
                plane_config.plane,
                localization_result["image"],
            )
            frame = _source_aligned_display_frame(
                segment.frame or reference_frame,
                reference_frame,
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
