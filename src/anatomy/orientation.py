"""把 CT 存储方向的轴符号转换为患者解剖参考方向。"""

import numpy as np


def patient_reference_axes(direction: np.ndarray) -> np.ndarray:
    """返回按 LR/AP/SI 排列、朝向患者头侧的右手参考坐标。

    CT 的存储行、列和层轴可以各自反向；负行列式并不表示患者头侧翻转。
    保留输入轴所在的直线和斜扫描角度，先把层轴符号对齐患者 LPS +Z，
    再选择最接近 LPS 左/后方向的一对面内轴。该操作只确定方向符号，
    不能用于体素到物理点转换；原始 direction/origin 必须保持不变。
    """

    matrix = np.asarray(direction, dtype=float)
    if matrix.size != 9:
        raise ValueError("患者参考方向必须包含九个方向余弦")
    matrix = matrix.reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("患者参考方向必须为有限的单位正交三轴")
    lr, si = matrix[:, 0].copy(), matrix[:, 2].copy()
    if si[2] < 0.0:
        si = -si
    ap = np.cross(si, lr)
    # 两个面内方向必须成对选符号，否则又会改变已确定的患者头侧。
    if lr[0] + ap[1] < 0.0:
        lr, ap = -lr, -ap
    return np.column_stack((lr, ap, si))
