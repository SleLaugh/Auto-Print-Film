"""从 JSON 配置加载窗宽窗位预设。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WindowPreset:
    """一个不可变的窗宽窗位预设。"""

    # 中文名称与外部请求中的预设名称保持一致。
    name: str
    # 窗宽和窗位由项目内部固定定义，外部请求不能覆盖。
    window_width: float
    window_center: float


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


def get_window_preset(preset: str) -> WindowPreset:
    """按配置名称返回窗宽窗位实体。"""
    try:
        return _WINDOW_PRESETS[preset]
    except (TypeError, KeyError) as exc:
        raise ValueError(f"不支持的窗宽窗位预设: {preset}") from exc
