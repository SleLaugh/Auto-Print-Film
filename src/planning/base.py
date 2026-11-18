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

    # 普通整切消费统一方向和区域，不再为每个新部位增补白名单。
    # 只有确有专用行为的处理器在此集中注册。
    planner_types = {
        ("颈椎", "椎间隙"): CervicalInterspacePlanner,
        ("腰椎", "MPR"): LumbarWholeRangePlanner,
        ("腰椎", "CPR"): LumbarWholeRangePlanner,
        ("腰椎", "椎间隙"): LumbarInterspacePlanner,
    }
    key = (slice_part, slice_type)
    if key in planner_types:
        return planner_types[key]()
    if slice_type in {"MPR", "CPR"}:
        return SingleLabelWholeRangePlanner()
    raise ValueError(
        f"未找到第六阶段 planner: slice_part={slice_part}, "
        f"SliceType={slice_type}"
    )
