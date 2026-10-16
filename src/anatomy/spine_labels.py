"""脊柱模型标签的运行时兼容名，不修改标准部位配置。"""

from collections.abc import Collection


def resolve_spine_label_name(
    label_name: str | None,
    available_labels: Collection[str],
) -> str | None:
    """S1 优先使用第一骶椎，否则使用实际提供的 sacrum。

    仅处理已支持的 S1 别名；没有可用同义标签时保留原名，让既有缺标签
    校验继续报错，不能用任意相邻椎体替代缺失的真实端点。
    """

    if label_name in {"S1", "vertebrae_S1", "sacrum"}:
        for candidate in ("vertebrae_S1", "sacrum"):
            if candidate in available_labels:
                return candidate
    return label_name
