"""Resolve dataset IDs without nnUNet's ambiguous numeric-prefix matching."""

from pathlib import Path
from typing import Iterable

from src.runtime import resolve_application_path

from .errors import DependencyError


def resolve_nnunet_dataset_name(dataset_id: str | int, roots: Iterable[str]) -> str:
    """Return the complete, unique Dataset<ID>_name, preserving true ambiguity."""
    model_id = int(dataset_id)
    prefix = f"Dataset{model_id:03d}"
    candidates = set()
    similar = set()
    for value in roots:
        if not value:
            continue
        root = Path(resolve_application_path(value))
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.name.startswith(prefix) or not path.is_dir():
                continue
            if path.name.startswith(prefix + "_"):
                candidates.add(path.name)
            else:
                similar.add(path.name)
    if len(candidates) > 1:
        names = sorted(candidates)
        raise DependencyError(
            f"nnUNet 模型 {model_id} 对应多个数据集名称：{'、'.join(names)}。"
            "请检查 nnUNet_results 中该编号的目录名称，确保模型 ID 唯一。",
            stage="segmentation",
            details={"reason": "dataset_ambiguous", "model_id": model_id, "candidates": names},
        )
    if candidates:
        return next(iter(candidates))
    if similar:
        # Falling back here would let upstream select a different numeric ID.
        raise DependencyError(
            f"未找到模型 {model_id} 的精确数据集目录 {prefix}_*；"
            f"相似前缀目录 {'、'.join(sorted(similar))} 不属于该模型。",
            stage="segmentation",
            details={"reason": "dataset_not_found", "model_id": model_id},
        )
    # Retain the external CLI's existing missing-dataset diagnosis when no local
    # candidate is visible (including commands that set their own environment).
    return str(dataset_id)
