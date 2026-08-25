"""
当前已实现请求组合的唯一能力注册表。

枚举表示协议允许什么，能力注册表表示当前版本已经实现的部位、切片方式和状位组合。
预检直接拒绝未注册组合，避免后续阶段生成看似正常但语义不受支持的计划。
"""

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Mapping, Tuple

from src.config.study_parts import capability_entries, load_study_part_config

from .errors import CapabilityError


CapabilityKey = Tuple[str, str, str]
"""能力键依次为部位、SliceType 和重建状位。"""


@dataclass(frozen=True)
class CapabilityRegistry:
    """保存并校验当前版本实际支持的请求组合。"""

    supported: FrozenSet[CapabilityKey]
    """不可变能力集合，防止任务执行期间被动态修改。"""
    supported_study_types: FrozenSet[Tuple[str, str]]
    """配置中已启用部位对应的检查类型。"""

    @classmethod
    def current(
        cls,
        study_part_config: Mapping[str, Any] | None = None,
    ) -> "CapabilityRegistry":
        """从统一部位配置创建正式能力表。"""

        config = study_part_config or load_study_part_config()
        entries = tuple(capability_entries(config))
        combinations = {
            (study_part, slice_type, plane)
            for _, study_part, slice_type, plane in entries
        }
        study_types = {
            (study_type, study_part)
            for study_type, study_part, _, _ in entries
        }
        return cls(frozenset(combinations), frozenset(study_types))

    def validate(self, request) -> None:
        """
        校验检查类型以及部位/SliceType/状位组合是否存在。

        当前能力表已经包含状位维度，因此任何未注册的部位、切法或状位组合都会
        在清空输出目录前被拒绝。
        """
        study_type = _value(request.study_type)
        study_part = _value(request.study_part)
        if (study_type, study_part) not in self.supported_study_types:
            raise self._error(study_type, study_part, "", None)
        for plane_config in request.plane_configs:
            plane = _value(plane_config.plane)
            slice_type = _value(plane_config.slice_type)
            if (study_part, slice_type, plane) not in self.supported:
                raise self._error(study_type, study_part, slice_type, plane)

    def _error(
        self,
        study_type: str,
        study_part: str,
        slice_type: str,
        plane: str | None,
    ) -> CapabilityError:
        """构造包含完整请求组合的标准能力错误。"""
        details = {
            "study_type": study_type,
            "study_part": study_part,
            "SliceType": slice_type,
            "plane": plane,
        }
        return CapabilityError(
            "当前请求组合尚未实现",
            details=details,
        )

    def combinations(self) -> Iterable[CapabilityKey]:
        """按稳定顺序返回能力组合，供帮助文本和文档生成使用。"""
        return tuple(sorted(self.supported))


def _value(value) -> str:
    """兼容枚举和直接字符串调用，统一得到协议值。"""
    return str(getattr(value, "value", value))
