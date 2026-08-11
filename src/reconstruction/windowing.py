"""第七阶段窗宽窗位解析。"""

from typing import Iterable, List

import numpy as np
import SimpleITK as sitk

from src.models import WindowPreset, get_window_preset


def unique_window_presets(names: Iterable[str]) -> List[WindowPreset]:
    """按请求顺序去重并解析窗宽窗位预设。"""

    result = []
    seen = set()
    for name in names:
        preset = get_window_preset(name)
        if preset.name in seen:
            continue
        seen.add(preset.name)
        result.append(preset)
    return result


def apply_window_to_image(
    image: sitk.Image,
    preset: WindowPreset | str,
) -> sitk.Image:
    """把 HU 图像按预设裁剪并映射为固定的 uint16 灰度图。"""

    window = preset if isinstance(preset, WindowPreset) else get_window_preset(preset)
    array = sitk.GetArrayFromImage(image).astype(np.float32)
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
