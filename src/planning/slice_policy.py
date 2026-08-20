"""切片方式对应的定位范围策略。"""

from collections.abc import Iterable

from src.models.enums import SliceType


_LOCALIZATION_PROFILES = {
    SliceType.MPR.value: "whole_range",
    SliceType.CPR.value: "whole_range",
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
    # 椎间隙配置会列出所有相邻椎体，能够同时满足同部位 MPR/CPR 的标签需求。
    return "interspace" if "interspace" in profiles else "whole_range"
