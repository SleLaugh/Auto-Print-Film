"""窗宽窗位固定预设。"""

from dataclasses import dataclass
from typing import Dict, Union

from .enums import WindowPresetEnum


@dataclass(frozen=True)
class WindowPreset:
    """一个不可变的窗宽窗位预设。"""

    # 中文名称与外部请求中的预设名称保持一致。
    name: str
    # 窗宽和窗位由项目内部固定定义，外部请求不能覆盖。
    window_width: int
    window_center: int


_WINDOW_PRESETS: Dict[WindowPresetEnum, WindowPreset] = {
    WindowPresetEnum.脑窗: WindowPreset("脑窗", 80, 40),
    WindowPresetEnum.骨窗: WindowPreset("骨窗", 2500, 400),
    WindowPresetEnum.肺窗: WindowPreset("肺窗", 1600, -600),
    WindowPresetEnum.纵隔窗: WindowPreset("纵隔窗", 350, 40),
    WindowPresetEnum.软组织窗: WindowPreset("软组织窗", 400, 50),
    WindowPresetEnum.腹部窗: WindowPreset("腹部窗", 350, 50),
    WindowPresetEnum.肝窗: WindowPreset("肝窗", 150, 60),
    WindowPresetEnum.肾窗: WindowPreset("肾窗", 350, 40),
    WindowPresetEnum.胰腺窗: WindowPreset("胰腺窗", 300, 40),
    WindowPresetEnum.骨盆窗: WindowPreset("骨盆窗", 400, 40),
    WindowPresetEnum.脊柱窗: WindowPreset("脊柱窗", 2000, 300),
}


def get_window_preset(
    preset: Union[WindowPresetEnum, str],
) -> WindowPreset:
    """按枚举或中文名称返回内部固定的窗宽窗位。"""
    try:
        preset_enum = (
            preset
            if isinstance(preset, WindowPresetEnum)
            else WindowPresetEnum(preset)
        )
        return _WINDOW_PRESETS[preset_enum]
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"不支持的窗宽窗位预设: {preset}") from exc
