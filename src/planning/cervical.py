"""颈椎专用切片规划。"""

from typing import Any, Dict, List, Tuple

from src.planning.base import PlanningResult
from src.planning.spine_common import (
    regularize_cervical_interspace_segments,
    resolve_spine_interspace_segments,
)
from src.planning.spine_interspace import (
    build_spine_interspace_plans,
    configured_interspace_definitions,
)
from src.planning.straight import CutSegment


def resolve_cervical_interspace_segments(
    localization_result: Dict[str, Any],
) -> Tuple[List[CutSegment], List[str]]:
    """解析配置允许输出的颈椎椎间隙。"""

    segments, warnings = resolve_spine_interspace_segments(
        localization_result,
        configured_interspace_definitions(localization_result),
    )
    segments, regularization_warnings = (
        regularize_cervical_interspace_segments(
            localization_result,
            segments,
        )
    )
    warnings.extend(regularization_warnings)
    return segments, warnings


class CervicalInterspacePlanner:
    """颈椎横状位椎间隙规划器。"""

    planner_name = "CervicalInterspacePlanner"

    def build_plans(
        self,
        request: Any,
        localization_result: Dict[str, Any],
    ) -> PlanningResult:
        """按配置生成独立颈椎横状位椎间隙计划。"""

        return build_spine_interspace_plans(
            request,
            localization_result,
            planner_name=self.planner_name,
            segment_resolver=resolve_cervical_interspace_segments,
        )
