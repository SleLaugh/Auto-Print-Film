"""从 JSON 配置加载窗宽窗位预设。"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class WindowPreset:
    """一个不可变的窗宽窗位预设。"""

    # 中文名称与外部请求中的预设名称保持一致。
    name: str
    # 窗宽和窗位来自默认配置，也可以由本次请求覆盖。
    window_width: float
    window_center: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为命令行和请求报告使用的公共字段格式。"""

        return {
            "Name": self.name,
            "windowWidth": self.window_width,
            "windowLevel": self.window_center,
        }


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "window_presets.json"


def load_window_presets(path: Path | None = None) -> dict[str, WindowPreset]:
    """加载并校验窗宽窗位配置。"""

    config_path = path or _DEFAULT_CONFIG_PATH
    try:
        data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取窗宽窗位配置: {config_path}") from exc
    if not isinstance(data, list):
        raise ValueError("窗宽窗位配置顶层必须是数组")

    presets: dict[str, WindowPreset] = {}
    expected_fields = {"name", "window_width", "window_center"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError(
                f"窗宽窗位配置第 {index + 1} 项必须且只能包含 "
                "name、window_width、window_center"
            )
        name = item["name"]
        width = item["window_width"]
        center = item["window_center"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"窗宽窗位配置第 {index + 1} 项 name 不能为空")
        if name in presets:
            raise ValueError(f"窗宽窗位配置名称重复: {name}")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or width <= 0:
            raise ValueError(f"窗宽窗位 {name} 的 window_width 必须大于 0")
        if isinstance(center, bool) or not isinstance(center, (int, float)):
            raise ValueError(f"窗宽窗位 {name} 的 window_center 必须是数值")
        presets[name] = WindowPreset(name, width, center)
    return presets


_WINDOW_PRESETS = load_window_presets()


def parse_window_preset(value: Any, index: int | None = None) -> WindowPreset:
    """解析外部 ``Name/windowWidth/windowLevel`` 窗宽窗位对象。"""

    label = f"第 {index + 1} 项" if index is not None else "窗宽窗位"
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}必须是对象")
    name = value.get("Name", value.get("name"))
    width = value.get("windowWidth", value.get("window_width"))
    center = value.get("windowLevel", value.get("window_center"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} Name 不能为空")
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not math.isfinite(width)
        or width <= 0
    ):
        raise ValueError(f"窗宽窗位 {name} 的 windowWidth 必须大于 0")
    if (
        isinstance(center, bool)
        or not isinstance(center, (int, float))
        or not math.isfinite(center)
    ):
        raise ValueError(f"窗宽窗位 {name} 的 windowLevel 必须是数值")
    return WindowPreset(name.strip(), width, center)


def parse_window_presets(values: Any) -> list[WindowPreset]:
    """解析外部窗宽窗位数组，并拒绝重复名称。"""

    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("window_presets 必须是 JSON 数组")
    result: list[WindowPreset] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        preset = (
            value
            if isinstance(value, WindowPreset)
            else parse_window_preset(value, index)
        )
        if preset.name in seen:
            raise ValueError(f"窗宽窗位名称重复: {preset.name}")
        seen.add(preset.name)
        result.append(preset)
    return result


def get_window_preset(
    preset: str,
    custom_presets: (
        Iterable[WindowPreset] | Mapping[str, WindowPreset] | None
    ) = None,
) -> WindowPreset:
    """按名称查找本次请求预设，找不到时回退到默认配置。"""

    custom: Mapping[str, WindowPreset]
    if isinstance(custom_presets, Mapping):
        custom = custom_presets
    else:
        custom = {item.name: item for item in (custom_presets or ())}
    try:
        return custom.get(preset) or _WINDOW_PRESETS[preset]
    except (TypeError, KeyError) as exc:
        raise ValueError(f"不支持的窗宽窗位预设: {preset}") from exc
