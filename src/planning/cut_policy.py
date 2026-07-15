"""根据部位和重建状位推导内部切割方式。"""

from collections.abc import Iterable


# 后续新增特定部位的区间切时，只扩展这一份集合。
_INTERSPACE_PLANE_KEYS = frozenset({("腰椎", "横状位")})


def cut_mode_for_plane(body_part: object, plane: object) -> str:
    """返回指定部位和状位实际使用的内部切割方式。"""

    key = (_value(body_part), _value(plane))
    return "interspace" if key in _INTERSPACE_PLANE_KEYS else "whole_range"


def localization_cut_mode(body_part: object, planes: Iterable[object]) -> str:
    """定位阶段优先加载所选状位需要的区间标签全集。"""

    return (
        "interspace"
        if any(cut_mode_for_plane(body_part, plane) == "interspace" for plane in planes)
        else "whole_range"
    )


def _value(value: object) -> str:
    """统一读取枚举值或普通字符串，便于按中文部位和状位匹配策略。"""

    return str(getattr(value, "value", value))
