"""第七阶段窗宽窗位解析。"""

from typing import Iterable, List, Mapping

import numpy as np
import SimpleITK as sitk

from src.models import WindowPreset, get_window_preset


def unique_window_presets(
    names: Iterable[str],
    custom_presets: (
        Iterable[WindowPreset] | Mapping[str, WindowPreset] | None
    ) = None,
) -> List[WindowPreset]:
    """按请求顺序去重，并按本次覆盖列表优先、默认配置回退的规则解析。"""

    result = []
    seen = set()
    for name in names:
        preset = get_window_preset(name, custom_presets)
        if preset.name in seen:
            continue
        seen.add(preset.name)
        result.append(preset)
    return result


def apply_window_to_image(
    image: sitk.Image,
    preset: WindowPreset | str,
) -> sitk.Image:
    """直接按所选窗宽窗位把重建 HU 映射为 uint16 出片显示灰度。"""

    window = preset if isinstance(preset, WindowPreset) else get_window_preset(preset)
    # 使用双精度执行窗口算术，避免高窗位/窄窗宽时把调用方参数先降为
    # float32；即使输入 HU 为 float32，窗口的下界和宽度也应保留精度。
    array = sitk.GetArrayFromImage(image).astype(np.float64)
    lower = float(window.window_center) - float(window.window_width) / 2.0
    scaled = (
        (array - lower)
        / float(window.window_width)
        * 65535.0
    )
    grayscale = np.rint(np.clip(scaled, 0.0, 65535.0)).astype(np.uint16)
    output = sitk.GetImageFromArray(grayscale)
    output.CopyInformation(image)
    return output
