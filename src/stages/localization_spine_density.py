"""椎体二值分割的连续支撑统计。"""

from __future__ import annotations

import numpy as np


def si_pillar_continuous_support(
    mask: np.ndarray,
    maximum_gap_slices: int = 1,
) -> np.ndarray:
    """返回每个 y/x 柱最长连续 SI 链中实际占据的层数。

    相邻前景段之间不超过 ``maximum_gap_slices`` 的空层可以连接同一条链，
    但空层不增加支撑度，也不填充输入 mask。该参数是明确的层数容错，
    并非与图像层厚无关的毫米距离。逐层处理只需 O(y*x) 额外内存。
    """

    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("SI 连续支撑计算要求三维 mask")
    if (
        isinstance(maximum_gap_slices, (bool, np.bool_))
        or not isinstance(maximum_gap_slices, (int, np.integer))
        or maximum_gap_slices < 0
    ):
        raise ValueError("maximum_gap_slices 必须是非负整数")

    shape = mask.shape[1:]
    current_counts = np.zeros(shape, dtype=np.intp)
    gap_counts = np.zeros(shape, dtype=np.intp)
    longest_counts = np.zeros(shape, dtype=np.intp)
    for layer in mask:
        occupied = np.asarray(layer, dtype=bool)
        gap_counts += 1
        gap_counts[occupied] = 0
        current_counts[gap_counts > maximum_gap_slices] = 0
        current_counts[occupied] += 1
        np.maximum(longest_counts, current_counts, out=longest_counts)

    return longest_counts
