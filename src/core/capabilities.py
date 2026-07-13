"""
当前已实现请求组合的唯一能力注册表。

枚举表示系统未来可能支持什么，能力注册表表示当前版本已经存在的部位和切法主流程。
状位能力允许渐进开发：预检不阻断未来状位，阶段四负责生成已实现计划或给出 warning。
"""

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Tuple

from .errors import CapabilityError


CapabilityKey = Tuple[str, str, str]
"""能力键依次为部位、切割方式和重建状位。"""


@dataclass(frozen=True)
class CapabilityRegistry:
    """保存并校验当前版本实际支持的请求组合。"""

    supported: FrozenSet[CapabilityKey]
    """不可变能力集合，防止任务执行期间被动态修改。"""

    @classmethod
    def current(cls) -> "CapabilityRegistry":
        """创建与当前 planner 实现一致的正式能力表。"""
        whole_range_parts = ("颈椎", "胸椎", "腰椎", "肺部", "头骨", "髋骨")
        whole_range_planes = ("横状位", "矢状位", "冠状位")
        combinations = {
            (part, "whole_range", plane)
            for part in whole_range_parts
            for plane in whole_range_planes
        }
        # 椎间切仍是后续的“直线部分切”范围，本轮保持只支持横状位。
        combinations.add(("腰椎", "interspace", "横状位"))
        return cls(frozenset(combinations))

    def validate(self, request) -> None:
        """
        校验检查类型以及部位/切法主流程是否存在。

        这里故意不校验请求状位：矢状位和冠状位属于当前主流程上的后续扩展，阶段四
        可以暂时跳过并给出 warning；只有整个部位/切法都没有 planner 时才提前阻断。
        """
        check_type = _value(request.check_type)
        body_part = _value(request.body_part)
        cut_mode = _value(request.cut_mode)
        if check_type != "CT":
            raise self._error(check_type, body_part, cut_mode, None)
        supported_workflows = {
            (supported_body_part, supported_cut_mode)
            for supported_body_part, supported_cut_mode, _ in self.supported
        }
        if (body_part, cut_mode) not in supported_workflows:
            raise self._error(check_type, body_part, cut_mode, None)

    def _error(
        self,
        check_type: str,
        body_part: str,
        cut_mode: str,
        plane: str | None,
    ) -> CapabilityError:
        """构造包含完整请求组合的标准能力错误。"""
        details = {
            "check_type": check_type,
            "body_part": body_part,
            "cut_mode": cut_mode,
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
