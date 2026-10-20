"""解析仅用于取景的同类标签，不扩大切割规划的目标范围。"""

from dataclasses import dataclass
import json
from pathlib import Path
import re


_VERTEBRA_NAME = re.compile(r"(?:vertebrae_)?(?:C[1-7]|T(?:[1-9]|1[0-2])|L[1-5]|S1)")


@dataclass(frozen=True)
class FramingContext:
    """同类结构标签、纵向物理轴及计划显式允许的正轴端取景余量。"""

    label_ids: tuple[int, ...]
    axis_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    upper_extension_mm: float = 0.0
    """只放宽正轴端的骨性取景上界；颅脑使用 SI 正方向，脊柱默认为零。"""


def load_framing_context(
    dataset_json_path: str | Path | None,
    coords: dict,
) -> FramingContext | None:
    """从模型标签表补全椎体取景上下文，隔离其他器官及未选中的骶骨。

    未提供或找不到 sidecar 时保留旧取景行为；存在的文件必须可读且具有
    有效 labels，损坏文件会直接报错，不能静默退回容易裁掉目标的取景。
    """

    if not dataset_json_path:
        return None
    selected_sacrum = set()
    has_target = False
    for name, coordinate in coords.items():
        ids = {int(value) for value in coordinate.get("label_ids", []) if int(value) != 0}
        if not ids:
            continue
        has_target = True
        if name == "sacrum":
            selected_sacrum.update(ids)
        elif _VERTEBRA_NAME.fullmatch(name) is None:
            return None
    if not has_target:
        return None

    path = Path(dataset_json_path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as source:
        dataset = json.load(source)
    labels = dataset.get("labels") if isinstance(dataset, dict) else None
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"dataset.json 中缺少有效 labels: {path}")

    context_ids = set(selected_sacrum)
    for name, value in labels.items():
        ids = value if isinstance(value, list) else [value]
        if not ids or any(type(item) is not int or item < 0 for item in ids):
            raise ValueError(f"dataset.json 标签 {name!r} 必须为非负整数或整数列表: {path}")
        if _VERTEBRA_NAME.fullmatch(name) is not None:
            context_ids.update(item for item in ids if item != 0)
    if not context_ids:
        return None
    return FramingContext(label_ids=tuple(sorted(context_ids)))
