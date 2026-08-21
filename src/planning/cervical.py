"""颈椎专用切片规划。"""

from typing import Any, Dict, List, Tuple

from src.planning.base import PlanningResult
from src.planning.spine_common import (
    regularize_cervical_interspace_segments,
    resolve_spine_interspace_segments,
)
from src.planning.spine_interspace import build_spine_interspace_plans
from src.planning.straight import CutSegment


CERVICAL_LABELS = [
    "vertebrae_C1",
    "vertebrae_C2",
    "vertebrae_C3",
    "vertebrae_C4",
    "vertebrae_C5",
    "vertebrae_C6",
    "vertebrae_C7",
]

# C1 为环椎，不生成 C1_C2 椎间隙计划；C1 仍保留在整体定位标签中。
CERVICAL_INTERVALS: List[Tuple[str, str, str]] = [
    ("C2_C3", "vertebrae_C2", "vertebrae_C3"),
    ("C3_C4", "vertebrae_C3", "vertebrae_C4"),
    ("C4_C5", "vertebrae_C4", "vertebrae_C5"),
    ("C5_C6", "vertebrae_C5", "vertebrae_C6"),
    ("C6_C7", "vertebrae_C6", "vertebrae_C7"),
]


def resolve_cervical_interspace_segments(
    localization_result: Dict[str, Any],
) -> Tuple[List[CutSegment], List[str]]:
    """解析 C2_C3 至 C6_C7 五个颈椎椎间隙。"""

    segments, warnings = resolve_spine_interspace_segments(
        localization_result,
        CERVICAL_INTERVALS,
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
        """生成 C2_C3 至 C6_C7 的独立横状位计划。"""

        return build_spine_interspace_plans(
            request,
            localization_result,
            label_names=CERVICAL_LABELS,
            intervals=CERVICAL_INTERVALS,
            planner_name=self.planner_name,
            segment_resolver=resolve_cervical_interspace_segments,
        )
