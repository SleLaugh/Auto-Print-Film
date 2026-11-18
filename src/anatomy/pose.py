"""按配置方法估计回正坐标；区域边界和切片余量不参与方向估计。"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk

from src.anatomy.brain import estimate_brain_frame_with_diagnostics
from src.anatomy.hip import estimate_hip_frame_with_diagnostics
from src.anatomy.lung import estimate_lung_frame_with_diagnostics
from src.anatomy.spine import estimate_spine_frame_with_diagnostics
from src.models import AnatomicalGeometryResult


@dataclass(frozen=True)
class PoseInputs:
    """只包含当前回正目标的数据；额外斜切策略由内部调用方选择。"""

    image: sitk.Image
    target_mask: np.ndarray
    label_masks: dict[str, np.ndarray] = field(default_factory=dict)
    label_indices_zyx: dict[str, np.ndarray] | None = None
    roles: dict[str, list[str]] = field(default_factory=dict)
    label_pose_annotations: dict[str, Any] | None = None
    include_axial_frame: bool = False
    lung_pose_support_mask: np.ndarray | None = None
    lung_pose_support_labels: tuple[str, ...] = ()


PoseEstimator = Callable[[PoseInputs], tuple[AnatomicalGeometryResult, dict[str, Any]]]


def _brain(inputs: PoseInputs):
    return estimate_brain_frame_with_diagnostics(inputs.target_mask, inputs.image)


def _lung(inputs: PoseInputs):
    return estimate_lung_frame_with_diagnostics(
        inputs.target_mask, inputs.image,
        pose_support_mask=inputs.lung_pose_support_mask,
        pose_support_source=inputs.lung_pose_support_labels,
    )


def _hip(inputs: PoseInputs):
    def role_mask(role: str):
        names = inputs.roles.get(role) or []
        if not names:
            raise ValueError(f"髋骨解剖坐标缺少 {role} 角色标签")
        missing = [name for name in names if name not in inputs.label_masks]
        if missing:
            raise ValueError(f"髋骨解剖坐标缺少 {role} 侧标签: {', '.join(missing)}")
        if len(names) == 1:
            return inputs.label_masks[names[0]]
        mask = inputs.label_masks[names[0]].copy()
        for name in names[1:]:
            mask |= inputs.label_masks[name]
        return mask

    geometry, diagnostics = estimate_hip_frame_with_diagnostics(
        left_mask=role_mask("left"), right_mask=role_mask("right"), image=inputs.image,
    )
    for role in ("left", "right"):
        names = inputs.roles[role]
        diagnostics[f"{role}_label"] = names[0] if len(names) == 1 else names
    return geometry, diagnostics


def _spine(inputs: PoseInputs):
    start = inputs.roles.get("start") or []
    end = inputs.roles.get("end") or []
    if len(start) != 1 or len(end) != 1:
        raise ValueError("脊柱回正 start/end 角色必须分别对应一个椎体标签")
    if not (inputs.label_masks or inputs.label_indices_zyx):
        raise ValueError("脊柱解剖坐标缺少椎体标签信息")
    return estimate_spine_frame_with_diagnostics(
        label_masks=inputs.label_masks,
        start_label=start[0], end_label=end[0], image=inputs.image,
        label_pose_annotations=inputs.label_pose_annotations,
        label_indices_zyx=inputs.label_indices_zyx,
        include_axial_frame=inputs.include_axial_frame,
    )


POSE_ESTIMATORS: dict[str, PoseEstimator] = {
    "brain": _brain, "lung": _lung, "hip": _hip, "spine": _spine,
}


def estimate_pose(method: str, inputs: PoseInputs):
    """新增回正方法只需扩展此入口，不改变定位、区域划分和重建流程。"""

    estimator = POSE_ESTIMATORS.get(method)
    if estimator is None:
        raise ValueError(f"未实现的回正方法: {method}")
    return estimator(inputs)
