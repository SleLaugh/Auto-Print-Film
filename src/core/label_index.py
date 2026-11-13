"""在一次分割快照上复用目标标签的稀疏索引。"""

from __future__ import annotations

from collections.abc import Iterable
from operator import index

import numpy as np


class SparseLabelIndex:
    """一次扫描目标标签，随后只在命中体素中查找各标签组合。

    ``array`` 必须是三维数组，并在本对象生命周期内保持不变。对象强引用
    原数组以固定快照身份，但不复制整卷数据，也不缓存各组合的三维点云。
    修改分割或换用另一个数组后，调用方必须重新构造本对象。
    """

    def __init__(self, array: np.ndarray, label_ids: Iterable[int]):
        if not isinstance(array, np.ndarray):
            raise TypeError("标签快照必须是 numpy.ndarray")
        if array.ndim != 3:
            raise ValueError("标签快照必须是三维数组")
        ids = self._normalize_ids(label_ids)
        self._source = array
        self._shape = array.shape
        self._label_ids = frozenset(ids)
        if ids:
            # flatnonzero 与 argwhere 一样按 C-order 枚举，亦适用于非连续源数组。
            self._flat_indices = np.flatnonzero(np.isin(array, ids))
            # flat 的高级索引只复制命中值；ravel 会额外复制非 C 连续的整卷。
            self._values = array.flat[self._flat_indices]
        else:
            self._flat_indices = np.empty(0, dtype=np.intp)
            self._values = np.empty(0, dtype=array.dtype)

    @staticmethod
    def _normalize_ids(label_ids: Iterable[int]) -> tuple[int, ...]:
        # index 接受 Python/numpy 整数，拒绝浮点标签的隐式截断。
        return tuple(dict.fromkeys(index(value) for value in label_ids))

    @property
    def label_ids(self) -> frozenset[int]:
        """构造时收录的标签，包括快照中没有体素的标签。"""
        return self._label_ids

    def matches_source(self, array: np.ndarray) -> bool:
        """只接受原快照身份；内容相等的复制品也需要自己的索引。"""
        return array is self._source

    def contains_ids(self, label_ids: Iterable[int]) -> bool:
        return self._label_ids.issuperset(self._normalize_ids(label_ids))

    def indices(self, label_ids: Iterable[int]) -> np.ndarray:
        """返回与 ``np.argwhere(np.isin(array, ids))`` 同序的 Z/Y/X 坐标。

        返回数组为 ``np.intp``，保持 argwhere 的列连续布局，避免后续坐标
        矩阵乘法因内存布局改变而切换计算路径。未收录标签会显式报错。
        """
        ids = self._normalize_ids(label_ids)
        missing = set(ids).difference(self._label_ids)
        if missing:
            raise ValueError(f"稀疏索引未收录标签: {sorted(missing)}")
        if not ids:
            return np.empty((3, 0), dtype=np.intp).T
        flat_indices = self._flat_indices
        if frozenset(ids) != self._label_ids:
            flat_indices = flat_indices[np.isin(self._values, ids)]
        # 先构造 3xN 再转置，和 np.argwhere 的内存布局一致。
        return np.asarray(np.unravel_index(flat_indices, self._shape), dtype=np.intp).T
