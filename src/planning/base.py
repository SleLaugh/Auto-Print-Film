#!/src/planning/base.py
# -*- coding: utf-8 -*-

"""第六阶段 planner 基础结构。"""

from dataclasses import dataclass, field
from typing import Any, List

from src.models import SlicePlan


@dataclass
class PlanningResult:
    """第六阶段 planner 的统一返回结果。"""

    plans: List[SlicePlan] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    planner_name: str = ""


def select_planner(slice_part: str, slice_type: str) -> Any:
    """
    根据切割部位和外部指定的 SliceType 选择第六阶段 planner。

    planner 的输出结构保持一致，差异只留在“从哪里切、怎么生成路径”的
    业务逻辑里。第七阶段只看 SlicePlan，不关心这里选中了哪个 planner。
    """

    from src.planning.cervical import CervicalInterspacePlanner
    from src.planning.lumbar import LumbarInterspacePlanner, LumbarWholeRangePlanner
    from src.planning.single_label import SingleLabelWholeRangePlanner

    planner_types = {
        ("颈椎", "MPR"): SingleLabelWholeRangePlanner,
        ("颈椎", "CPR"): SingleLabelWholeRangePlanner,
        ("颈椎", "椎间隙"): CervicalInterspacePlanner,
        ("胸椎", "MPR"): SingleLabelWholeRangePlanner,
        ("胸椎", "CPR"): SingleLabelWholeRangePlanner,
        ("脊柱", "MPR"): SingleLabelWholeRangePlanner,
        ("脊柱", "CPR"): SingleLabelWholeRangePlanner,
        ("腰椎", "MPR"): LumbarWholeRangePlanner,
        ("腰椎", "CPR"): LumbarWholeRangePlanner,
        ("腰椎", "椎间隙"): LumbarInterspacePlanner,
        ("肺部", "MPR"): SingleLabelWholeRangePlanner,
        ("颅脑", "MPR"): SingleLabelWholeRangePlanner,
        ("髋骨", "MPR"): SingleLabelWholeRangePlanner,
    }
    key = (slice_part, slice_type)
    if key not in planner_types:
        raise ValueError(
            f"未找到第六阶段 planner: slice_part={slice_part}, "
            f"SliceType={slice_type}"
        )
    return planner_types[key]()
