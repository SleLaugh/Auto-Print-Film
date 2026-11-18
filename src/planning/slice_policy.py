"""切片方式对应的定位范围策略。"""

from collections.abc import Iterable

from src.models.enums import SliceType


_LOCALIZATION_PROFILES = {
    SliceType.MPR.value: "whole_range",
    SliceType.CPR.value: "spine_range",
    SliceType.椎间隙.value: "interspace",
}


def localization_profile_for_slice_type(slice_type: object) -> str:
    """返回一种切片方式所需的定位范围方案。"""

    value = str(getattr(slice_type, "value", slice_type))
    try:
        return _LOCALIZATION_PROFILES[value]
    except KeyError as error:
        raise ValueError(f"未知 SliceType: {value}") from error


def localization_profile_for_request(slice_types: Iterable[object]) -> str:
    """为一次混合请求选择能覆盖全部切片方式的定位范围方案。"""

    profiles = {
        localization_profile_for_slice_type(slice_type)
        for slice_type in slice_types
    }
    # 主配置仅用于结果标识；各切法的标签依赖由定位阶段分别收集。
    for profile in ("interspace", "whole_range", "spine_range"):
        if profile in profiles:
            return profile
    raise ValueError("至少需要一种切片方式")
