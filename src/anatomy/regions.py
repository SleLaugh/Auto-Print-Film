#!/src/anatomy/regions.py
# -*- coding: utf-8 -*-

"""在已经确定的解剖方向中解析区域；标签引用均为实际分割标签名。"""

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import numpy as np
import SimpleITK as sitk

from src.models import AnatomicalBounds, AnatomicalFrame, AnatomicalGeometryResult, AxisBounds


_PROJECTION_CHUNK_SIZE = 65536
_FILTERED_POSE_METHODS = {"lung", "hip"}


def _labels(value: Any, location: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{location} 必须是非空标签数组")
    if any(not isinstance(label, str) or not label.strip() for label in value):
        raise ValueError(f"{location} 的标签名必须是非空字符串")
    return list(dict.fromkeys(value))


def _boundary(value: Any, location: str, side: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} 必须是边界配置对象")
    labels = _labels(value.get("labels"), f"{location}.labels")
    edge = value.get("edge")
    if edge not in {"upper", "lower"}:
        raise ValueError(f"{location}.edge 必须是 upper 或 lower")
    reduce = value.get("reduce")
    if reduce is None and len(labels) == 1:
        reduce = "max" if side == "upper" else "min"
    if reduce not in {"max", "min"}:
        raise ValueError(f"{location}.reduce 必须是 max 或 min，多标签时必须显式指定")
    offset = value.get("offset_mm", 0.0)
    if isinstance(offset, bool) or not isinstance(offset, (int, float)):
        raise ValueError(f"{location}.offset_mm 必须是有限毫米数值")
    if not math.isfinite(offset):
        raise ValueError(f"{location}.offset_mm 必须是有限毫米数值")
    return {"labels": labels, "edge": edge, "reduce": reduce, "offset_mm": float(offset)}


def _union_edge(boundary: dict[str, Any], labels: set[str]) -> bool:
    """判断该边界是否直接取完整标签并集的一端，不考虑其后应用的偏移。"""

    return (
        set(boundary["labels"]) == labels
        and boundary["reduce"] == ("max" if boundary["edge"] == "upper" else "min")
    )


def _index_array(label: str, indices: Mapping[str, Any]) -> np.ndarray:
    if label not in indices:
        raise ValueError(f"区域所需标签 {label} 缺失")
    array = np.asarray(indices[label])
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"标签 {label} 坐标必须是 N×3 的 z/y/x 索引")
    if not len(array):
        raise ValueError(f"区域所需标签 {label} 为空")
    return array


class _LabelProjector:
    """每标签每方向只投影一次，缓存范围，不保存全标签物理点或整卷 Mask。"""

    def __init__(
        self,
        image: sitk.Image,
        indices: Mapping[str, np.ndarray],
        pose_labels: set[str],
        pose_method: str,
        cleaned_indices: Mapping[str, np.ndarray],
        pose_roles: Mapping[str, list[str]] | None = None,
    ) -> None:
        self.indices = indices
        self.pose_labels = pose_labels
        self.pose_method = pose_method
        self.cleaned_indices = dict(cleaned_indices)
        self.pose_roles = pose_roles or {}
        self.size_zyx = np.asarray(image.GetSize()[::-1])
        self.origin = np.asarray(image.GetOrigin(), dtype=float)
        self.voxel_axes = (
            np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
            @ np.diag(image.GetSpacing())
        )
        self.cache: dict[tuple[bytes, str], np.ndarray] = {}

    def _validate_chunk(self, chunk: np.ndarray, label: str) -> None:
        try:
            valid = (
                np.all(np.isfinite(chunk))
                and np.all(chunk >= 0)
                and np.all(chunk < self.size_zyx)
                and np.all(chunk == np.floor(chunk))
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"标签 {label} 坐标必须是图像内的有限整数索引") from error
        if not valid:
            raise ValueError(f"标签 {label} 坐标必须是图像内的有限整数索引")

    def _clean_pose_indices(self, label: str) -> None:
        """仅组合区域需要逐标签范围时，在标签包围盒内复用已有连通块策略。"""

        if self.pose_method == "hip":
            # 一侧别名可包含多个实际标签；必须先合并该侧再选主体，与回正一致。
            labels = next(
                (names for names in self.pose_roles.values() if label in names), [label],
            )
        else:
            labels = sorted(self.pose_labels)
        arrays = {name: _index_array(name, self.indices) for name in labels}
        lower = self.size_zyx.copy()
        upper = np.zeros(3, dtype=np.int64)
        for name, indices in arrays.items():
            for start in range(0, len(indices), _PROJECTION_CHUNK_SIZE):
                chunk = indices[start:start + _PROJECTION_CHUNK_SIZE]
                self._validate_chunk(chunk, name)
                lower = np.minimum(lower, chunk.min(axis=0)).astype(np.int64)
                upper = np.maximum(upper, chunk.max(axis=0)).astype(np.int64)
        mask = np.zeros(tuple(upper - lower + 1), dtype=bool)
        for indices in arrays.values():
            for start in range(0, len(indices), _PROJECTION_CHUNK_SIZE):
                relative = indices[start:start + _PROJECTION_CHUNK_SIZE].astype(np.int64) - lower
                mask[tuple(relative.T)] = True
        if self.pose_method == "hip":
            from src.anatomy.hip import _retain_largest_component

            retained, _ = _retain_largest_component(mask, label)
        else:
            from src.anatomy.lung import retain_dominant_lung_components

            retained, _ = retain_dominant_lung_components(mask)
        del mask
        for name, indices in arrays.items():
            kept = []
            for start in range(0, len(indices), _PROJECTION_CHUNK_SIZE):
                chunk = indices[start:start + _PROJECTION_CHUNK_SIZE]
                relative = chunk.astype(np.int64) - lower
                kept.append(chunk[retained[tuple(relative.T)]])
            self.cleaned_indices[name] = np.concatenate(kept, axis=0)

    def bounds(self, label: str, frame: AnatomicalFrame) -> np.ndarray:
        axes = np.asarray([
            [axis.x, axis.y, axis.z]
            for axis in (
                frame.left_right_axis,
                frame.anterior_posterior_axis,
                frame.superior_inferior_axis,
            )
        ], dtype=float).T
        key = (axes.tobytes(), label)
        if key in self.cache:
            return self.cache[key]
        if self.pose_method in _FILTERED_POSE_METHODS and label in self.pose_labels:
            if label not in self.cleaned_indices:
                self._clean_pose_indices(label)
            indices = _index_array(label, self.cleaned_indices)
        else:
            indices = _index_array(label, self.indices)
        transform = self.voxel_axes.T @ axes
        origin_projection = self.origin @ axes
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for start in range(0, len(indices), _PROJECTION_CHUNK_SIZE):
            chunk = indices[start:start + _PROJECTION_CHUNK_SIZE]
            self._validate_chunk(chunk, label)
            projected = chunk @ transform[::-1, :] + origin_projection
            lower = np.minimum(lower, projected.min(axis=0))
            upper = np.maximum(upper, projected.max(axis=0))
        if self.pose_method == "brain":
            support = 0.5 * np.abs(transform).sum(axis=0)
            lower -= support
            upper += support
        result = np.stack((lower, upper))
        self.cache[key] = result
        return result


def _resolve_frame(
    frame: AnatomicalFrame,
    target_labels: list[str],
    boundaries: dict[str, dict[str, Any]],
    pose_labels: set[str],
    projector: _LabelProjector,
) -> AnatomicalFrame:
    if set(target_labels) == pose_labels:
        lr = frame.bounds.left_right
        ap = frame.bounds.anterior_posterior
    else:
        targets = [projector.bounds(label, frame) for label in target_labels]
        lr = AxisBounds(min(item[0, 0] for item in targets), max(item[1, 0] for item in targets))
        ap = AxisBounds(min(item[0, 1] for item in targets), max(item[1, 1] for item in targets))

    values = {}
    for side, boundary in boundaries.items():
        if _union_edge(boundary, pose_labels):
            axis_bounds = frame.bounds.superior_inferior
            value = axis_bounds.maximum if boundary["edge"] == "upper" else axis_bounds.minimum
        else:
            index = 1 if boundary["edge"] == "upper" else 0
            candidates = [
                float(projector.bounds(label, frame)[index, 2])
                for label in boundary["labels"]
            ]
            value = (max if boundary["reduce"] == "max" else min)(candidates)
        values[side] = value + boundary["offset_mm"]
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("区域偏移后的边界必须是有限毫米数值")
    if values["lower"] >= values["upper"]:
        raise ValueError(
            "区域上下边界反转或范围为空："
            f"lower={values['lower']:.6g} mm，upper={values['upper']:.6g} mm"
        )
    bounds = AnatomicalBounds(lr, ap, AxisBounds(values["lower"], values["upper"]))
    return frame if bounds == frame.bounds else replace(frame, bounds=bounds)


def resolve_anatomical_regions(
    resolved_regions: dict[str, dict[str, Any]],
    geometry: AnatomicalGeometryResult,
    image: sitk.Image,
    label_indices_zyx: dict[str, np.ndarray],
    pose_labels: list[str],
    pose_method: str,
    *,
    cleaned_label_indices_zyx: dict[str, np.ndarray] | None = None,
    pose_roles: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """将命名区域投影到全局及专用切片方向，返回可直接序列化的阶段结果。

    投影是患者物理点与轴的点积，与已有 ``AnatomicalFrame.bounds`` 一致，
    不减 frame.origin。正向沿参考系头侧，边界先取各标签的 edge，再 reduce，
    最后加有符号毫米偏移。LR/AP 只由 target_labels 支撑。

    完整 pose 标签并集复用现有 frame 边界，保持脑体素外支撑以及肺/髋骨主体
    去噪结果，默认配置不再扫描体素。肺/髋骨的非默认逐标签计算按需在局部标签
    包围盒内复用原连通块清理策略并缓存；若调用方已有清理后的稀疏坐标，可通过
    cleaned_label_indices_zyx 直接复用；不读取图像像素，仅按所需标签包围盒分配 Mask。
    hip 的 pose_roles 保留同侧多个实际标签的分组，清理时先合并同侧再分回各标签。
    constrained 表示配置不再等同于原始完整 pose 范围；即使偏移扩大范围也为
    True，供后续取景尊重显式区域。六方向外扩仍由规划阶段单独应用。
    """

    if image.GetDimension() != 3:
        raise ValueError("区域解析只支持三维图像")
    if not isinstance(resolved_regions, Mapping):
        raise ValueError("命名区域必须是配置对象")
    pose_set = set(_labels(pose_labels, "pose_labels"))
    projector = _LabelProjector(
        image, label_indices_zyx, pose_set, pose_method, cleaned_label_indices_zyx or {},
        pose_roles,
    )
    results = {}
    for name, region in resolved_regions.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(region, Mapping):
            raise ValueError("命名区域必须包含非空名称和配置对象")
        if region.get("type", "whole_range") != "whole_range":
            raise ValueError(f"区域 {name} 只支持 whole_range 类型")
        targets = _labels(region.get("target_labels"), f"区域 {name}.target_labels")
        raw_bounds = region.get("bounds")
        if not isinstance(raw_bounds, Mapping):
            raise ValueError(f"区域 {name}.bounds 必须包含 upper/lower 边界")
        boundaries = {
            side: _boundary(raw_bounds.get(side), f"区域 {name}.bounds.{side}", side)
            for side in ("upper", "lower")
        }
        required = set(targets).union(*(value["labels"] for value in boundaries.values()))
        # 默认路径只检查引用存在、数组形状和非空，不重复扫描已经分析过的体素。
        for label in sorted(required):
            _index_array(label, label_indices_zyx)
        def resolve_frame(frame_name: str, frame: AnatomicalFrame) -> dict[str, Any]:
            try:
                return _resolve_frame(
                    frame, targets, boundaries, pose_set, projector,
                ).to_dict()
            except ValueError as error:
                raise ValueError(f"区域 {name}，方向 {frame_name}: {error}") from error
        unchanged = set(targets) == pose_set and all(
            _union_edge(boundaries[side], pose_set)
            and boundaries[side]["edge"] == side
            and boundaries[side]["offset_mm"] == 0.0
            for side in ("upper", "lower")
        )
        results[name] = {
            "global_frame": resolve_frame("global_frame", geometry.global_frame),
            "local_frames": {
                frame_name: resolve_frame(frame_name, frame)
                for frame_name, frame in geometry.local_frames.items()
            },
            "target_labels": targets,
            "bounds_config": boundaries,
            "constrained": not unchanged,
        }
    return results
