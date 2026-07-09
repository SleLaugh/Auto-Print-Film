#!/src/planning/base.py
# -*- coding: utf-8 -*-

"""阶段四 planner 基础结构。"""

from dataclasses import dataclass, field
from typing import Any, List

from src.models import SlicePlan


@dataclass
class PlanningResult:
    """阶段四 planner 的统一返回结果。"""

    plans: List[SlicePlan] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    planner_name: str = ""


def select_planner(body_part: str, cut_mode: str) -> Any:
    """
    根据部位和切割方式选择阶段四 planner。

    planner 的输出结构保持一致，差异只留在“从哪里切、怎么生成路径”的
    业务逻辑里。第五步后续只看 SlicePlan，不关心这里选中了哪个 planner。
    """

    from src.planning.lumbar import LumbarInterspacePlanner, LumbarWholeRangePlanner
    from src.planning.single_label import SingleLabelWholeRangePlanner

    planner_types = {
        ("颈椎", "whole_range"): SingleLabelWholeRangePlanner,
        ("胸椎", "whole_range"): SingleLabelWholeRangePlanner,
        ("腰椎", "whole_range"): LumbarWholeRangePlanner,
        ("腰椎", "interspace"): LumbarInterspacePlanner,
        ("肺部", "whole_range"): SingleLabelWholeRangePlanner,
        ("头骨", "whole_range"): SingleLabelWholeRangePlanner,
        ("髋骨", "whole_range"): SingleLabelWholeRangePlanner,
    }
    key = (body_part, cut_mode)
    if key not in planner_types:
        raise ValueError(
            f"未找到阶段四 planner: body_part={body_part}, cut_mode={cut_mode}"
        )
    return planner_types[key]()
