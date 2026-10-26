#!/src/services/stage_localization.py
# -*- coding: utf-8 -*-

"""空间校正、标签定位和解剖分析阶段共用的医学几何算法。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import json
import math

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.resources import resource_operation
from src.config import (
    load_slice_part_config,
    validate_slice_part_config,
)
from src.models import (
    AnatomicalGeometryResult,
    LabelCoordinate,
    LabelTarget,
    VoxelBBox,
    VoxelPoint,
)
from src.anatomy.lung import estimate_lung_frame_with_diagnostics
from src.anatomy.hip import estimate_hip_frame_with_diagnostics
from src.anatomy.spine import estimate_spine_frame_with_diagnostics
from src.anatomy.spine_labels import resolve_spine_label_name
from src.anatomy.brain import estimate_brain_frame_with_diagnostics
from src.anatomy.spine_pose import estimate_spine_axial_symmetry, estimate_spine_label_symmetry
from src.utils import DebugLog
from src.runtime import config_candidates, resolve_application_path

from .localization_contracts import LocalizationContractMixin
from .localization_execution import LocalizationExecutionMixin
from .localization_geometry import LocalizationSpineGeometryMixin
from .localization_spine_density import si_pillar_continuous_support


@dataclass(frozen=True)
class _LabelVoxelRegion:
    """同一分割空间内可复用的 label 体素索引和包围盒。"""

    indices_zyx: np.ndarray
    bbox: VoxelBBox | None


_TERMINAL_EXTENSION_MINIMUM_COMPONENT_FRACTION = 0.10
_TERMINAL_EXTENSION_MAXIMUM_SI_GAP = 1
_TERMINAL_EXTENSION_MINIMUM_TRANSVERSE_OVERLAP_FRACTION = 0.75
_TERMINAL_EXTENSION_MINIMUM_SI_SPAN = 2
_BODY_REMNANT_MAX_LABEL_VOLUME_FRACTION = 0.10
_BODY_REMNANT_MAX_LABEL_SI_SPAN_FRACTION = 0.5
_PRISM_SEPARATOR_MAXIMUM_PROJECTION_INTRUSION_FRACTION = 0.05
_LOCALIZATION_WORKSPACE_KEY = "localization_workspace_v1"


class LocalizationOperations(
    LocalizationSpineGeometryMixin,
    LocalizationContractMixin,
    BaseStage,
):
    """校正、定位和解剖分析共用的已验证算法集合。"""
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        *,
        stage_name: str = "localization_operations",
    ):
        """初始化共用定位配置、校正阈值和 nnUNet 数据集路径。"""

        super().__init__(name=stage_name, config=config)
        # 脊柱候选超过阈值且通过一致性检查时进入第三阶段回正。
        # 颅脑、髋和肺的左右旋转/侧倾交由第五阶段的物理解剖坐标处理；
        # 颅脑的形态俯仰也在第五阶段估计，不在此处旋转整卷图像。
        self.correction_angle_threshold_degrees = float(
            self.config.get('correction_angle_threshold_degrees', 3.0)
        )
        self.correction_non_spine_angle_threshold_degrees = float(
            self.config.get('correction_non_spine_angle_threshold_degrees', 8.0)
        )
        # PCA 至少需要一定数量的目标体素；太少时方向不稳定，直接跳过回正。
        self.correction_min_points = int(
            self.config.get('correction_min_points', 10)
        )
        # 脊柱整段回正至少需要两个可用椎体；只有一个椎体时无法代表整段姿态。
        self.correction_min_vertebrae = int(
            self.config.get('correction_min_vertebrae', 2)
        )
        # 多个椎体的横状位候选角必须彼此接近；离散过大通常说明不是
        # 整体体位旋转，而是椎体形态、曲度或局部附件在拉动 PCA。
        self.correction_max_label_angle_spread_degrees = float(
            self.config.get('correction_max_label_angle_spread_degrees', 15.0)
        )
        # 第三阶段先写候选回正图和诊断信息；候选 raw/seg 双文件可用时，
        # 第四阶段定位和第五阶段解剖分析统一切到候选图像空间。
        # 实验开关只服务 `_correct_images_if_needed` 的独立验证入口。
        self.correction_requested = bool(
            self.config.get('correction_requested', True)
        )
        self.enable_global_correction = bool(
            self.config.get('enable_experimental_global_correction', False)
        )
        self._load_config()
        self._load_nnunet_config()

    def _load_config(self):
        """
        加载定位/切割范围配置。

        统一切割部位配置同时描述模型、定位区域和允许切法。本阶段只消费其中的
        label_aliases、localization 和 slice_rules。
        """
        injected_config = self.config.get("slice_part_config")
        if injected_config:
            validate_slice_part_config(injected_config)
            self._slice_part_config = injected_config
            return

        config_paths = config_candidates("slice_part_config.json")

        for config_path in config_paths:
            if config_path.exists():
                self._slice_part_config = load_slice_part_config(config_path)
                DebugLog(f"[定位] 加载配置文件: {config_path}")
                return

        raise FileNotFoundError("未找到 slice_part_config.json")

    def _load_nnunet_config(self):
        """
        加载 nnUNet 路径配置，用于根据模型 ID 查找 dataset.json。

        定位阶段不重新匹配模型，只复用 segmentation 阶段得到的 model_id。
        这里读取路径配置，是为了把 label 名称转换成真实 label id。
        """
        injected_keys = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
        if any(key in self.config for key in injected_keys):
            self._nnunet_paths = {
                key: resolve_application_path(self.config.get(key, ''))
                for key in injected_keys
            }
            return

        config_paths = config_candidates("nnuNet_config.json")

        self._nnunet_paths = {}
        for config_path in config_paths:
            if not config_path.exists():
                continue
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._nnunet_paths = {
                key: resolve_application_path(config.get(key, ''))
                for key in injected_keys
            }
            return
    
    @staticmethod
    def _build_anatomical_geometry(
        slice_part: str,
        target_mask: np.ndarray,
        image: sitk.Image,
        label_masks: Dict[str, np.ndarray] | None = None,
        start_label: str | None = None,
        end_label: str | None = None,
        label_pose_annotations: Dict[str, Any] | None = None,
        label_indices_zyx: Dict[str, np.ndarray] | None = None,
        lung_pose_support_mask: np.ndarray | None = None,
        lung_pose_support_labels: tuple[str, ...] = (),
    ) -> tuple[AnatomicalGeometryResult | None, Dict[str, Any]]:
        """只为已经适配的部位计算统一解剖坐标，其余部位保持旧行为。"""

        if slice_part == "肺部":
            return estimate_lung_frame_with_diagnostics(
                target_mask, image,
                pose_support_mask=lung_pose_support_mask,
                pose_support_source=lung_pose_support_labels,
            )
        if slice_part == "颅脑":
            # 方向和切片三轴范围只使用脑区，骨性取景支持不进入几何估计。
            return estimate_brain_frame_with_diagnostics(target_mask, image)
        if slice_part in {"颈椎", "胸椎", "腰椎", "脊柱"}:
            if not (label_masks or label_indices_zyx) or not start_label or not end_label:
                raise ValueError(f"{slice_part} 解剖坐标缺少首尾椎体标签信息")
            return estimate_spine_frame_with_diagnostics(
                label_masks=label_masks,
                start_label=start_label,
                end_label=end_label,
                image=image,
                label_pose_annotations=label_pose_annotations,
                label_indices_zyx=label_indices_zyx,
            )
        if slice_part == "髋骨":
            if not label_masks or not start_label or not end_label:
                raise ValueError("髋骨解剖坐标缺少左右髋骨标签信息")
            if start_label not in label_masks:
                raise ValueError(f"髋骨解剖坐标缺少左侧标签: {start_label}")
            if end_label not in label_masks:
                raise ValueError(f"髋骨解剖坐标缺少右侧标签: {end_label}")
            geometry, diagnostics = estimate_hip_frame_with_diagnostics(
                left_mask=label_masks[start_label],
                right_mask=label_masks[end_label],
                image=image,
            )
            # 配置决定左右标签来源，诊断中保留原名，便于现场核对模型映射。
            diagnostics["left_label"] = start_label
            diagnostics["right_label"] = end_label
            return geometry, diagnostics
        return None, {}

    def _find_dataset_json(self, model_info: Dict[str, Any], seg_path: str = None) -> Path:
        """
        根据 segmentation 阶段匹配到的模型 ID 查找对应 dataset.json。

        优先使用 seg.nii.gz 同目录的 sidecar dataset.json，因为它和
        当前分割输出一一对应。真实运行中模型目录可能存在多个相同 ID 的
        dataset.json，sidecar 文件能避免定位阶段读到不匹配的 label 字典。

        如果 sidecar 不存在，再按 nnUNet 目录查找。nnUNet 目录常见形态
        是 Dataset292_xxx 或 Dataset315_xxx。优先从 nnUNet_results 查找，
        因为它最贴近实际用于推理的模型；找不到再查 preprocessed/raw。
        """
        if seg_path:
            sidecar_dataset_json = Path(seg_path).parent / "dataset.json"
            if sidecar_dataset_json.exists():
                return sidecar_dataset_json

        dataset_id = model_info.get('dataset_id') or model_info.get('model_id')
        if dataset_id is None:
            raise ValueError("分割结果缺少 model_id/dataset_id，无法查找 dataset.json")

        dataset_prefix = f"Dataset{int(dataset_id):03d}"
        roots = [
            self._nnunet_paths.get('nnUNet_results', ''),
            self._nnunet_paths.get('nnUNet_preprocessed', ''),
            self._nnunet_paths.get('nnUNet_raw', ''),
        ]

        for root_value in roots:
            if not root_value:
                continue
            root = Path(root_value)
            if not root.exists():
                continue
            matches = [
                path for path in root.rglob("dataset.json")
                if any(part.startswith(dataset_prefix) for part in path.parts)
            ]
            if matches:
                return self._choose_dataset_json(matches)

        raise FileNotFoundError(f"未找到模型 {dataset_prefix} 对应的 dataset.json")

    @staticmethod
    def _choose_dataset_json(paths: list[Path]) -> Path:
        """
        从候选 dataset.json 中选择一个更像模型根目录的文件。

        结果目录下可能有 crossval 或子任务目录。路径越短，通常越接近
        当前模型配置根目录；同长度时按路径名排序，保证结果稳定。
        """
        return sorted(paths, key=lambda p: (len(p.parts), str(p)))[0]

    @staticmethod
    def _load_label_map(dataset_json_path: Path) -> Dict[str, list[int]]:
        """
        读取 dataset.json 中的 labels，并统一成 label_name -> [label_id]。

        目前主要处理普通整数标签；如果后续遇到 nnUNet region label
        的列表写法，这里也会保留成多个 id。
        """
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        labels = dataset.get('labels') or {}
        if not labels:
            raise ValueError(f"dataset.json 中缺少 labels: {dataset_json_path}")

        label_map: Dict[str, list[int]] = {}
        for label_name, label_value in labels.items():
            if isinstance(label_value, list):
                label_map[label_name] = [int(value) for value in label_value]
            else:
                label_map[label_name] = [int(label_value)]
        return label_map

    def _resolve_label_targets(
        self,
        resolved_cut_config: Dict[str, Any],
        label_map: Dict[str, list[int]]
    ) -> Dict[str, LabelTarget]:
        """
        把定位配置中用到的 label 名称解析为 id，并去重。

        返回结构按 label 名称组织，后续提坐标时只需要遍历一次，避免
        椎间切中 L2、L3 等标签重复计算。
        """
        label_names = self._collect_label_names(resolved_cut_config, label_map)
        missing_names = [name for name in label_names if name not in label_map]
        if missing_names:
            available = ", ".join(label_map.keys())
            missing = ", ".join(missing_names)
            raise ValueError(f"dataset.json 中找不到定位标签: {missing}; 可用标签: {available}")

        targets = {}
        for label_name in label_names:
            targets[label_name] = LabelTarget(
                label_name=label_name,
                label_ids=label_map[label_name]
            )
        return targets

    def _resolve_combined_label_targets(
        self,
        resolved_cut_configs: Dict[str, Dict[str, Any]],
        label_map: Dict[str, list[int]],
    ) -> Dict[str, LabelTarget]:
        """合并全部范围方案的标签，供空间校正、整体定位和解剖分析使用。

        即使本次只请求椎间隙，也会合并整切范围并校验其中的标签体素；
        因此扩展整切端点后，新增椎体同样必须存在于分割结果中。
        这里仅合并定位标签，不会据此增加椎间隙区域。
        """

        combined_targets: Dict[str, LabelTarget] = {}
        for resolved_cut_config in resolved_cut_configs.values():
            combined_targets.update(
                self._resolve_label_targets(resolved_cut_config, label_map)
            )
        return combined_targets

    def _collect_label_names(
        self,
        resolved_cut_config: Dict[str, Any],
        label_map: Dict[str, list[int]] = None
    ) -> list[str]:
        """
        收集本次定位需要的 label 名称。

        普通目标仍然收集 start/end。脊柱整切的 start/end 是范围边界，
        不是只取两个端点；例如 C1 到 T1 会按解剖顺序收集 C1-C7、T1，
        T12 到 S1 会收集 T12、L1-L5、S1。
        """
        label_names = []
        if 'intervals' in resolved_cut_config:
            for interval in resolved_cut_config['intervals']:
                label_names.extend(self._expand_label_pair(
                    interval['start_labels'],
                    interval['end_labels'],
                    label_map,
                ))
        else:
            label_names.extend(self._expand_label_pair(
                resolved_cut_config['start_labels'],
                resolved_cut_config['end_labels'],
                label_map,
            ))

        seen = set()
        unique_names = []
        for label_name in label_names:
            if label_name in seen:
                continue
            seen.add(label_name)
            unique_names.append(label_name)
        return unique_names

    def _expand_label_pair(
        self,
        start_labels: list[str],
        end_labels: list[str],
        label_map: Dict[str, list[int]] = None,
    ) -> list[str]:
        """
        展开一组 start/end label。

        如果 start/end 都是脊柱 label，就按解剖顺序展开中间椎体；否则
        保持原来的端点语义，适配 brain、lung 这类整体目标。
        """
        expanded_names = []
        for start_label in start_labels:
            for end_label in end_labels:
                expanded_range = self._expand_vertebrae_range(start_label, end_label, label_map)
                if expanded_range:
                    expanded_names.extend(expanded_range)
                else:
                    expanded_names.extend([start_label, end_label])
        return expanded_names

    @classmethod
    def _expand_vertebrae_range(
        cls,
        start_label: str,
        end_label: str,
        label_map: Dict[str, list[int]] = None,
    ) -> list[str]:
        """
        按 C/T/L/S 的解剖顺序展开脊柱范围。

        label_map 传入时，只返回当前模型认识的 label，避免把不存在的中间
        label 强行加入定位目标。S1 在不同模型里可能叫 vertebrae_S1 或
        sacrum，这里按实际可用 label 选择。
        """
        start_index = cls._vertebra_order_index(start_label)
        end_index = cls._vertebra_order_index(end_label)
        if start_index is None or end_index is None:
            return []

        order = cls._vertebra_order_labels()
        if start_index <= end_index:
            vertebra_keys = order[start_index:end_index + 1]
        else:
            vertebra_keys = list(reversed(order[end_index:start_index + 1]))

        expanded_names = []
        for vertebra_key in vertebra_keys:
            label_name = cls._vertebra_key_to_available_label(vertebra_key, label_map)
            if label_name:
                expanded_names.append(label_name)
        return expanded_names

    @staticmethod
    def _vertebra_order_labels() -> list[str]:
        """
        统一的脊柱顺序，方向为头侧到足侧。

        这里保存业务短名，真正的 dataset.json label 名在
        _vertebra_key_to_available_label 中转换。
        """
        return (
            [f"C{index}" for index in range(1, 8)] +
            [f"T{index}" for index in range(1, 13)] +
            [f"L{index}" for index in range(1, 6)] +
            ["S1"]
        )

    @classmethod
    def _vertebra_order_index(cls, label_name: str) -> int | None:
        """返回椎体标签在头侧到足侧统一序列中的位置，无法识别时返回空。"""

        vertebra_key = cls._vertebra_key(label_name)
        if not vertebra_key:
            return None
        order = cls._vertebra_order_labels()
        if vertebra_key not in order:
            return None
        return order.index(vertebra_key)

    @staticmethod
    def _vertebra_key(label_name: str) -> str | None:
        """
        把 dataset label 或业务短名转换成 C1/T12/L5/S1 这种统一 key。
        """
        name = str(label_name)
        if name == "sacrum":
            return "S1"
        if name.startswith("vertebrae_"):
            name = name[len("vertebrae_"):]
        if len(name) < 2:
            return None
        prefix = name[0]
        number = name[1:]
        if prefix in {"C", "T", "L", "S"} and number.isdigit():
            return f"{prefix}{int(number)}"
        return None

    @staticmethod
    def _vertebra_key_to_available_label(
        vertebra_key: str,
        label_map: Dict[str, list[int]] = None,
    ) -> str | None:
        """把统一椎体短名转换为当前 dataset.json 中实际存在的标签名。"""

        candidates = (
            ["vertebrae_S1", "sacrum"]
            if vertebra_key == "S1"
            else [f"vertebrae_{vertebra_key}"]
        )
        if label_map is None:
            return candidates[0]
        for candidate in candidates:
            if candidate in label_map:
                return candidate
        return None

    @staticmethod
    def _resolve_spine_endpoint_label(
        label_name: str | None,
        label_targets: Dict[str, LabelTarget],
    ) -> str | None:
        """按本次实际定位标签解析 S1，保留其余端点的缺失校验语义。

        配置保留标准名称；部分模型只提供 sacrum。端点必须与第四阶段
        已验证的 label_targets 一致，同时存在时仍优先第一骶椎标签。
        """

        return resolve_spine_label_name(label_name, label_targets)

    def _extract_label_coordinates(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
    ) -> Dict[str, LabelCoordinate]:
        """
        从分割体素中提取每个目标 label 的 bbox 和 centroid。

        SimpleITK 转 numpy 后数组顺序是 (z, y, x)。这里先输出体素索引
        坐标；物理坐标 mm 需要后续结合 affine/origin/spacing/direction。
        """
        if label_regions is None:
            label_regions = self._build_label_voxel_regions(
                seg_array,
                label_targets,
            )

        coordinates = {}
        for label_name, target in label_targets.items():
            region = label_regions[label_name]
            indices = region.indices_zyx
            if indices.size == 0:
                coordinates[label_name] = LabelCoordinate(
                    label_name=label_name,
                    label_ids=target.label_ids,
                    voxel_count=0,
                    found=False,
                )
                continue

            centroid_z, centroid_y, centroid_x = indices.mean(axis=0)
            coordinates[label_name] = LabelCoordinate(
                label_name=label_name,
                label_ids=target.label_ids,
                voxel_count=int(indices.shape[0]),
                found=True,
                centroid=VoxelPoint(
                    z=float(centroid_z),
                    y=float(centroid_y),
                    x=float(centroid_x),
                ),
                bbox=region.bbox,
            )
        return coordinates

    @staticmethod
    def _build_label_voxel_regions(
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
    ) -> Dict[str, _LabelVoxelRegion]:
        """扫描一次各目标 label，保留后续阶段共同需要的稀疏索引。"""

        regions = {}
        for label_name, target in label_targets.items():
            indices = np.argwhere(np.isin(seg_array, target.label_ids))
            bbox = None
            if indices.size:
                z_min, y_min, x_min = indices.min(axis=0)
                z_max, y_max, x_max = indices.max(axis=0)
                bbox = VoxelBBox(
                    z_min=int(z_min),
                    z_max=int(z_max),
                    y_min=int(y_min),
                    y_max=int(y_max),
                    x_min=int(x_min),
                    x_max=int(x_max),
                )
            regions[label_name] = _LabelVoxelRegion(
                indices_zyx=indices,
                bbox=bbox,
            )
        return regions

    @staticmethod
    def _build_target_mask(
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget]
    ) -> np.ndarray:
        """
        把本次任务关心的所有 label 合成一个整体 mask。

        横状位回正不能用整套 CT 判断，因为同一段片子里可能头歪、胸椎正，
        也可能反过来。这里先按部位和定位范围方案解析出的 label 范围
        缩小目标，再只看这个目标范围的姿态。
        """
        target_label_ids = []
        for target in label_targets.values():
            target_label_ids.extend(target.label_ids)
        if not target_label_ids:
            return np.zeros(seg_array.shape, dtype=bool)
        return np.isin(seg_array, target_label_ids)

    @staticmethod
    def _build_target_bbox(target_mask: np.ndarray) -> VoxelBBox:
        """
        计算目标整体 mask 的 3D bbox。

        这个 bbox 记录的是“本次判断回正的空间范围”，后续排查为什么回正或
        为什么没回正时，比只看单个 label 的 bbox 更直观。
        """
        indices = np.argwhere(target_mask)
        if indices.size == 0:
            raise ValueError("目标 mask 为空，无法计算定位范围 bbox")

        z_min, y_min, x_min = indices.min(axis=0)
        z_max, y_max, x_max = indices.max(axis=0)
        return VoxelBBox(
            z_min=int(z_min),
            z_max=int(z_max),
            y_min=int(y_min),
            y_max=int(y_max),
            x_min=int(x_min),
            x_max=int(x_max),
        )

    def _detect_axial_skew(self, target_mask: np.ndarray) -> Dict[str, Any]:
        """
        用目标 mask 的 XY 投影估计横状位旋转偏角。

        SimpleITK 转出的数组是 (z, y, x)。横状位是否歪，第一版只看
        XY 平面中的方向：先把目标体素投影到 XY，再用 PCA 找主方向。
        由于目标天然可能沿 X 轴或 Y 轴更长，最终角度取“相对最近标准轴
        0/90 度的偏离”，避免把一个正着的竖向结构误判为 90 度歪斜。
        """
        return self._detect_axial_skew_indices(np.argwhere(target_mask))

    def _detect_axial_skew_indices(self, indices: np.ndarray) -> Dict[str, Any]:
        """从按 z/y/x 排列的目标体素索引估计横状位旋转偏角。"""

        point_count = int(indices.shape[0])
        base_info = {
            'method': 'pca_xy_projection',
            'point_count': point_count,
            'min_points': self.correction_min_points,
            'threshold_degrees': self.correction_angle_threshold_degrees,
            'raw_pca_angle_degrees': 0.0,
            'angle_degrees': 0.0,
            'abs_angle_degrees': 0.0,
            'enough_points': False,
            'should_correct': False,
        }
        if point_count < self.correction_min_points:
            base_info['reason'] = '目标体素太少，PCA 方向不稳定'
            return base_info

        xy_points = indices[:, [2, 1]].astype(np.float64)
        centered_points = xy_points - xy_points.mean(axis=0)
        covariance = np.cov(centered_points, rowvar=False)
        if not np.all(np.isfinite(covariance)):
            base_info['reason'] = '目标点协方差异常'
            return base_info

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        max_index = int(np.argmax(eigenvalues))
        if float(eigenvalues[max_index]) <= 0:
            base_info['reason'] = '目标点没有可用主方向'
            return base_info

        principal_vector = eigenvectors[:, max_index]
        raw_angle = math.degrees(math.atan2(principal_vector[1], principal_vector[0]))
        normalized_raw_angle = self._normalize_pca_angle(raw_angle)
        deviation_angle = self._deviation_from_nearest_axial_axis(normalized_raw_angle)
        abs_angle = abs(deviation_angle)

        return {
            **base_info,
            'raw_pca_angle_degrees': float(normalized_raw_angle),
            'angle_degrees': float(deviation_angle),
            'abs_angle_degrees': float(abs_angle),
            'enough_points': True,
            'should_correct': abs_angle > self.correction_angle_threshold_degrees,
        }

    def _detect_target_axial_skew(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
        excluded_labels: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """
        根据目标类型选择横状位角度检测方法。

        脊柱类目标优先用单个椎体的横状位主轴做稳健聚合；颅脑、肺部等
        非脊柱目标没有天然的椎体顺序，候选入口会用更高阈值约束整体
        mask PCA，避免小角度形态偏差生成误导性候选。

        该方法只估计候选角并生成诊断；是否提升候选空间由第三阶段入口统一决定。
        """
        excluded_label_set = set(excluded_labels)
        if self._is_vertebrae_targets(label_targets):
            vertebrae_info = self._detect_vertebrae_axial_skew(
                seg_array,
                label_targets,
                label_regions=label_regions,
                excluded_labels=excluded_label_set,
            )
            if vertebrae_info.get('enough_points'):
                return vertebrae_info

            fallback_targets = {
                name: target
                for name, target in label_targets.items()
                if name not in excluded_label_set
            }
            fallback_mask = self._build_target_mask(
                seg_array,
                fallback_targets,
            )
            fallback_info = self._detect_axial_skew(fallback_mask)
            fallback_info.update({
                'method': 'pca_xy_projection_fallback',
                'fallback_reason': vertebrae_info.get('reason', '脊柱椎体级角度不可用'),
                'vertebrae_skew_params': vertebrae_info,
            })
            return fallback_info

        target_mask = self._build_target_mask(seg_array, label_targets)
        return self._detect_axial_skew(target_mask)

    def _todo_correction_info(self) -> Dict[str, Any]:
        """
        构造“回正逻辑待实现”的诊断信息。

        这里故意不计算候选角度，避免把 PCA 诊断误解为可执行回正角。该结果只供
        未启用全局实验回正的独立 helper 使用，不代表第三阶段主流程输出。
        """
        return {
            'method': 'global_correction_todo',
            'point_count': 0,
            'min_points': self.correction_min_points,
            'min_vertebrae': self.correction_min_vertebrae,
            'threshold_degrees': self.correction_angle_threshold_degrees,
            'raw_pca_angle_degrees': 0.0,
            'angle_degrees': 0.0,
            'abs_angle_degrees': 0.0,
            'enough_points': False,
            'should_correct': False,
            'correction_requested': self.correction_requested,
            'correction_status': 'todo',
            'global_correction_enabled': self.enable_global_correction,
            'correction_todo_reason': (
                '回正逻辑待实现，实验 helper 不改写 raw/seg'
            ),
        }

    def _build_xy_correction_candidate(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        slice_part: str | None = None,
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
        image: sitk.Image | None = None,
    ) -> Dict[str, Any]:
        """估计横状位回正候选角；图像写出与应用由第三阶段决定。

        主流程脊柱使用完整椎体的可靠镜像面对齐角。未传图像的兼容 helper
        保留归一偏角 PCA；头、髋、肺由第三阶段转交第五阶段的统一坐标处理。
        """

        if not self.correction_requested:
            return {
                'method': 'xy_correction_disabled',
                'axis': 'image_z',
                'angle_degrees': 0.0,
                'abs_angle_degrees': 0.0,
                'confidence': 0.0,
                'correction_status': 'disabled',
                'correction_applied': False,
                'reason': 'correction_requested 为 false',
            }
        if image is not None and slice_part in {"颅脑", "髋骨", "肺部"}:
            # 三维姿态在第五阶段计算，重建时一次采样到最终方向。
            # 此处不再分配整块 PCA 点云或先做整卷 XY 插值。
            return {
                "method": "anatomical_frame_deferred",
                "axis": "image_z",
                "angle_degrees": 0.0,
                "abs_angle_degrees": 0.0,
                "correction_status": "deferred_to_anatomical_frame",
                "correction_applied": False,
                "should_correct": False,
                "reason": "由第五阶段左右对称面统一确定各状位方向",
            }
        if not self._is_vertebrae_targets(label_targets):
            target_mask = self._build_target_mask(seg_array, label_targets)
            skew_info = self._detect_axial_skew(target_mask)
            abs_angle = float(skew_info.get('abs_angle_degrees', 0.0))
            enough_points = bool(skew_info.get('enough_points'))
            should_correct = (
                enough_points
                and abs_angle > self.correction_non_spine_angle_threshold_degrees
            )
            reason = ''
            if not enough_points:
                reason = str(skew_info.get('reason', '有效目标体素不足'))
            elif not should_correct:
                reason = '非脊柱候选角未超过高置信阈值'
            return {
                **skew_info,
                'method': 'pca_xy_projection_non_spine',
                'axis': 'image_z',
                'angle_degrees': float(skew_info.get('angle_degrees', 0.0)),
                'abs_angle_degrees': abs_angle,
                'confidence': max(0.0, min(1.0, abs_angle / 15.0)),
                'correction_status': 'candidate' if should_correct else 'rejected',
                'correction_applied': False,
                'non_spine_threshold_degrees': (
                    self.correction_non_spine_angle_threshold_degrees
                ),
                'should_correct': should_correct,
                'slice_part': slice_part,
                'reason': reason,
            }
        if image is not None:
            regions = label_regions or self._build_label_voxel_regions(seg_array, label_targets)
            excluded = set(
                self._resolve_spine_cpr_config(slice_part).get("excluded_labels", [])
                if slice_part else []
            )
            skipped = [
                name for name, region in regions.items()
                if name in excluded or region.bbox is None
                or len(region.indices_zyx) < self.correction_min_points
                or self._bbox_touches_array_boundary(region.bbox, seg_array.shape)
            ]
            candidate = estimate_spine_axial_symmetry(
                {name: region.indices_zyx for name, region in regions.items() if name not in skipped},
                image,
                threshold_degrees=self.correction_angle_threshold_degrees,
                min_labels=self.correction_min_vertebrae,
                max_angle_spread_degrees=self.correction_max_label_angle_spread_degrees,
            )
            candidate["skipped_label_names"] = list(dict.fromkeys(
                skipped + candidate.get("skipped_label_names", [])
            ))
            should_correct = bool(candidate.get("should_correct"))
            return {
                **candidate,
                "axis": "image_z",
                "correction_status": "candidate" if should_correct else "rejected",
                "correction_applied": False,
            }
        skew_info = self._detect_target_axial_skew(
            seg_array,
            label_targets,
            label_regions=label_regions,
            excluded_labels=(
                self._resolve_spine_cpr_config(slice_part).get(
                    "excluded_labels",
                    [],
                )
                if slice_part
                else []
            ),
        )
        abs_angle = float(skew_info.get('abs_angle_degrees', 0.0))
        enough_points = bool(skew_info.get('enough_points'))
        should_correct = bool(skew_info.get('should_correct'))
        status = 'candidate' if enough_points and should_correct else 'rejected'
        reason = ''
        if not enough_points:
            reason = str(skew_info.get('reason', '有效目标体素不足'))
        elif not should_correct:
            reason = str(skew_info.get('reason') or '候选角未超过回正阈值')

        return {
            **skew_info,
            'axis': 'image_z',
            'angle_degrees': float(skew_info.get('angle_degrees', 0.0)),
            'abs_angle_degrees': abs_angle,
            'confidence': max(0.0, min(1.0, abs_angle / 15.0)),
            'correction_status': status,
            'correction_applied': False,
            'reason': reason,
        }

    @staticmethod
    def _is_vertebrae_targets(label_targets: Dict[str, LabelTarget]) -> bool:
        """
        判断当前目标是否属于脊柱类。

        sacrum 在模型里有时不叫 vertebrae_S1，但它仍然属于腰骶整段定位。
        """
        if not label_targets:
            return False
        for label_name in label_targets:
            if label_name == "sacrum":
                continue
            if not label_name.startswith("vertebrae_"):
                return False
        return True

    def _detect_vertebrae_axial_skew(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
        excluded_labels: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """
        对脊柱目标按单椎体角度聚合，避免整块体素 PCA 被椎体形状带偏。

        真实数据里首尾可能被扫描边界切掉。贴边 label 的形状不完整，
        这里直接跳过，避免残缺椎体主轴影响整段回正角度。
        """
        used_label_names = []
        skipped_label_names = []
        label_angles = []
        total_point_count = 0

        if label_regions is None:
            label_regions = self._build_label_voxel_regions(
                seg_array,
                label_targets,
            )

        excluded_label_set = set(excluded_labels)
        for label_name in label_targets:
            if label_name in excluded_label_set:
                skipped_label_names.append(label_name)
                continue
            region = label_regions[label_name]
            indices = region.indices_zyx
            point_count = int(indices.shape[0])
            if point_count < self.correction_min_points:
                skipped_label_names.append(label_name)
                continue

            bbox = region.bbox
            if self._bbox_touches_array_boundary(bbox, seg_array.shape):
                skipped_label_names.append(label_name)
                continue

            label_info = self._detect_axial_skew_indices(indices)
            if not label_info.get('enough_points'):
                skipped_label_names.append(label_name)
                continue

            used_label_names.append(label_name)
            total_point_count += point_count
            label_angles.append({
                'label_name': label_name,
                'point_count': point_count,
                'raw_pca_angle_degrees': label_info['raw_pca_angle_degrees'],
                'angle_degrees': label_info['angle_degrees'],
                'bbox': bbox.to_dict(),
            })

        base_info = {
            'method': 'vertebrae_label_angle_aggregation',
            'point_count': total_point_count,
            'min_points': self.correction_min_points,
            'min_vertebrae': self.correction_min_vertebrae,
            'threshold_degrees': self.correction_angle_threshold_degrees,
            'max_label_angle_spread_degrees': (
                self.correction_max_label_angle_spread_degrees
            ),
            'angle_spread_degrees': 0.0,
            'raw_pca_angle_degrees': 0.0,
            'angle_degrees': 0.0,
            'abs_angle_degrees': 0.0,
            'enough_points': False,
            'should_correct': False,
            'used_label_names': used_label_names,
            'skipped_label_names': skipped_label_names,
            'label_angles': label_angles,
        }
        if len(label_angles) < self.correction_min_vertebrae:
            base_info['reason'] = '可用于聚合的完整椎体数量不足'
            return base_info

        angle_values = [
            float(item['angle_degrees'])
            for item in label_angles
        ]
        median_angle = float(np.median(angle_values))
        angle_spread = max(
            abs(angle - median_angle)
            for angle in angle_values
        )
        if angle_spread > self.correction_max_label_angle_spread_degrees:
            return {
                **base_info,
                'angle_spread_degrees': float(angle_spread),
                'median_label_angle_degrees': median_angle,
                'enough_points': True,
                'should_correct': False,
                'reason': '椎体横状位候选角离散过大，拒绝整体 XY 回正候选',
            }

        # PCA 最长轴可能在左右/前后轴之间切换。先归一成偏角再聚合，
        # 不能把约 -85° 与 +5° 两条等价基准混合成相反的旋转。
        deviation_angle = self._weighted_axis_angle_mean([
            (item['angle_degrees'], item['point_count'])
            for item in label_angles
        ])
        abs_angle = abs(deviation_angle)
        return {
            **base_info,
            'angle_spread_degrees': float(angle_spread),
            'median_label_angle_degrees': median_angle,
            'raw_pca_angle_degrees': float(deviation_angle),
            'angle_degrees': float(deviation_angle),
            'abs_angle_degrees': float(abs_angle),
            'enough_points': True,
            'should_correct': abs_angle > self.correction_angle_threshold_degrees,
        }

    @staticmethod
    def _bbox_touches_array_boundary(bbox: VoxelBBox, array_shape_zyx: tuple[int, ...]) -> bool:
        """
        判断目标 bbox 是否贴到数组边界。

        贴边通常表示该 label 被扫描范围截断，横状位主轴不再可靠。
        """
        z_size, y_size, x_size = array_shape_zyx
        return (
            bbox.z_min <= 0 or bbox.y_min <= 0 or bbox.x_min <= 0 or
            bbox.z_max >= z_size - 1 or bbox.y_max >= y_size - 1 or bbox.x_max >= x_size - 1
        )

    @classmethod
    def _weighted_axis_angle_mean(cls, angle_weight_pairs: list[tuple[float, int]]) -> float:
        """
        计算轴方向的加权平均角。

        PCA 主轴没有箭头，0 度和 180 度等价，所以用双角度圆统计求平均，
        再除以 2 回到 [-90, 90)。
        """
        sin_sum = 0.0
        cos_sum = 0.0
        for angle_degrees, weight in angle_weight_pairs:
            radians = math.radians(float(angle_degrees) * 2.0)
            sin_sum += float(weight) * math.sin(radians)
            cos_sum += float(weight) * math.cos(radians)

        if sin_sum == 0.0 and cos_sum == 0.0:
            return 0.0
        mean_angle = math.degrees(math.atan2(sin_sum, cos_sum)) / 2.0
        return cls._normalize_pca_angle(mean_angle)

    @staticmethod
    def _normalize_pca_angle(angle_degrees: float) -> float:
        """
        把 PCA 主方向规整到 [-90, 90)。

        PCA 的特征向量没有箭头方向，同一条轴既可以表示成 30 度，也可以
        表示成 210 度；规整后才能稳定比较。
        """
        angle = float(angle_degrees)
        while angle < -90.0:
            angle += 180.0
        while angle >= 90.0:
            angle -= 180.0
        return angle

    @staticmethod
    def _deviation_from_nearest_axial_axis(angle_degrees: float) -> float:
        """
        计算主方向偏离最近标准轴的角度。

        标准轴是 0 度和 90 度。比如 82 度其实更接近“竖着但偏了 -8 度”，
        不应按 82 度直接回正。
        """
        if angle_degrees >= 45.0:
            return angle_degrees - 90.0
        if angle_degrees < -45.0:
            return angle_degrees + 90.0
        return angle_degrees

    def _correct_images_if_needed(
        self,
        raw_path: str,
        seg_path: str,
        output_dir: Path,
        skew_info: Dict[str, Any],
        correction_center_bbox: VoxelBBox = None,
    ) -> Dict[str, Any]:
        """
        根据回正状态决定后续阶段该读取哪组 raw/seg。

        默认情况下 enable_experimental_global_correction=False，因此这里只
        返回原始路径，并把 correction_status 写成 todo。下面真正写
        raw_corrected/seg_corrected 的分支只留给后续算法实验，不能作为当前
        业务默认回正逻辑理解。
        """
        output_dir = Path(output_dir)
        raw_path = str(raw_path)
        seg_path = str(seg_path)
        should_correct = bool(skew_info.get('should_correct'))
        angle_degrees = float(skew_info.get('angle_degrees', 0.0))
        correction_params = {
            **skew_info,
            'applied_rotation_degrees': 0.0,
            'raw_interpolator': 'none',
            'seg_interpolator': 'none',
            'rotation_center_source': 'none',
            'rotation_center_index_xyz': None,
            'correction_requested': self.correction_requested,
            'correction_status': 'pending',
            'correction_todo_reason': '',
            'global_correction_enabled': self.enable_global_correction,
        }

        if not self.enable_global_correction:
            correction_params['correction_status'] = 'todo'
            correction_params['correction_todo_reason'] = (
                '回正逻辑待实现，实验 helper 不改写 raw/seg'
            )
            return {
                'raw_for_reconstruction_path': raw_path,
                'seg_for_reconstruction_path': seg_path,
                'corrected_raw_path': raw_path,
                'corrected_seg_path': seg_path,
                'correction_applied': False,
                'correction_params': correction_params,
            }

        if not should_correct:
            correction_params['correction_status'] = 'not_needed'
            correction_params['correction_todo_reason'] = ''
            correction_params['correction_disabled_reason'] = '未超过回正阈值'
            return {
                'raw_for_reconstruction_path': raw_path,
                'seg_for_reconstruction_path': seg_path,
                'corrected_raw_path': raw_path,
                'corrected_seg_path': seg_path,
                'correction_applied': False,
                'correction_params': correction_params,
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        corrected_raw_path = output_dir / 'raw_corrected.nii.gz'
        corrected_seg_path = output_dir / 'seg_corrected.nii.gz'

        with resource_operation("spatial_correction", "read_rotation_inputs"):
            raw_image = sitk.ReadImage(raw_path)
            seg_image = sitk.ReadImage(seg_path)
        # SimpleITK Resample 的 transform 是“输出坐标 -> 输入坐标”。
        # 因此这里传入检测角度本身，图像内容会朝相反方向被采样回正。
        applied_rotation_degrees = angle_degrees
        rotation_center_index_xyz = self._rotation_center_index_xyz(
            correction_center_bbox,
            raw_image,
        )

        corrected_raw = self._rotate_image_xy(
            image=raw_image,
            rotation_degrees=applied_rotation_degrees,
            interpolator=sitk.sitkLinear,
            default_value=self._image_min_value(raw_image),
            center_index_xyz=rotation_center_index_xyz,
        )
        corrected_seg = self._rotate_image_xy(
            image=seg_image,
            rotation_degrees=applied_rotation_degrees,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
            center_index_xyz=rotation_center_index_xyz,
        )
        corrected_seg = sitk.Cast(corrected_seg, seg_image.GetPixelID())

        sitk.WriteImage(corrected_raw, str(corrected_raw_path))
        sitk.WriteImage(corrected_seg, str(corrected_seg_path))

        correction_params.update({
            'correction_status': 'applied',
            'applied_rotation_degrees': applied_rotation_degrees,
            'raw_interpolator': 'linear',
            'seg_interpolator': 'nearest_neighbor',
            'raw_default_value': self._image_min_value(raw_image),
            'seg_default_value': 0,
            'rotation_center_source': 'target_bbox' if correction_center_bbox else 'image_center',
            'rotation_center_index_xyz': rotation_center_index_xyz,
        })
        return {
            'raw_for_reconstruction_path': str(corrected_raw_path),
            'seg_for_reconstruction_path': str(corrected_seg_path),
            'corrected_raw_path': str(corrected_raw_path),
            'corrected_seg_path': str(corrected_seg_path),
            'correction_applied': True,
            'correction_params': correction_params,
        }

    @staticmethod
    def _promote_xy_correction_candidate_if_available(
        correction_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        候选 raw/seg 均存在时，把候选回正结果提升为本轮定位和重建输入。

        第四阶段、第五阶段以及后续规划重建只消费稳定的
        `raw_for_reconstruction_path` 和 `seg_for_reconstruction_path`。这里集中
        完成路径提升，后续阶段不需要各自判断候选文件。
        """

        result = dict(correction_result)
        params = dict(result.get('correction_params') or {})
        candidate_raw_path = str(result.get('candidate_raw_path') or '')
        candidate_seg_path = str(result.get('candidate_seg_path') or '')
        candidate_paths = (candidate_raw_path, candidate_seg_path)

        if result.get('correction_applied'):
            params.setdefault('promoted_candidate_for_reconstruction', False)
            result['correction_params'] = params
            return result

        source_status = str(params.get('correction_status') or '')
        if source_status != 'candidate':
            params['candidate_promotion_status'] = 'not_candidate'
            params['promoted_candidate_for_reconstruction'] = False
            result['correction_params'] = params
            return result

        if not all(candidate_paths):
            params['candidate_promotion_status'] = 'not_available'
            params['promoted_candidate_for_reconstruction'] = False
            result['correction_params'] = params
            return result

        if not all(Path(path).exists() for path in candidate_paths):
            params['candidate_promotion_status'] = 'missing_files'
            params['promoted_candidate_for_reconstruction'] = False
            result['correction_params'] = params
            return result

        result.update({
            'raw_for_reconstruction_path': candidate_raw_path,
            'seg_for_reconstruction_path': candidate_seg_path,
            'corrected_raw_path': candidate_raw_path,
            'corrected_seg_path': candidate_seg_path,
            'correction_applied': True,
        })
        params.update({
            'correction_status': 'applied',
            'source_correction_status': source_status,
            'candidate_promotion_status': 'promoted',
            'promoted_candidate_for_reconstruction': True,
            'correction_applied': True,
        })
        result['correction_params'] = params
        return result

    def _write_xy_correction_candidate_images(
        self,
        raw_path: str,
        seg_path: str,
        output_dir: Path,
        candidate: Dict[str, Any],
        correction_center_bbox: VoxelBBox | None,
    ) -> Dict[str, Any]:
        """写出候选回正 raw/seg；是否切换主流程由调用方统一决定。"""

        raw_path = str(raw_path)
        seg_path = str(seg_path)
        output_dir = Path(output_dir)
        params = {
            **candidate,
            'candidate_raw_path': '',
            'candidate_seg_path': '',
            'raw_interpolator': 'none',
            'seg_interpolator': 'none',
            'rotation_center_source': 'none',
            'rotation_center_index_xyz': None,
        }
        if candidate.get('correction_status') != 'candidate':
            return {
                'raw_for_reconstruction_path': raw_path,
                'seg_for_reconstruction_path': seg_path,
                'corrected_raw_path': raw_path,
                'corrected_seg_path': seg_path,
                'candidate_raw_path': '',
                'candidate_seg_path': '',
                'correction_applied': False,
                'correction_params': params,
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_raw_path = output_dir / 'raw_corrected_candidate.nii.gz'
        candidate_seg_path = output_dir / 'seg_corrected_candidate.nii.gz'
        raw_image = sitk.ReadImage(raw_path)
        seg_image = sitk.ReadImage(seg_path)
        center_index_xyz = self._rotation_center_index_xyz(
            correction_center_bbox,
            raw_image,
        )
        angle_degrees = float(candidate.get('angle_degrees', 0.0))
        with resource_operation(
            "spatial_correction", "rotate_raw", size_xyz=raw_image.GetSize(),
            angle_degrees=angle_degrees,
        ):
            corrected_raw = self._rotate_image_xy(
                image=raw_image,
                rotation_degrees=angle_degrees,
                interpolator=sitk.sitkLinear,
                default_value=self._image_min_value(raw_image),
                center_index_xyz=center_index_xyz,
            )
        with resource_operation(
            "spatial_correction", "rotate_segmentation", size_xyz=seg_image.GetSize(),
        ):
            corrected_seg = self._rotate_image_xy(
                image=seg_image,
                rotation_degrees=angle_degrees,
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0,
                center_index_xyz=center_index_xyz,
            )
            corrected_seg = sitk.Cast(corrected_seg, seg_image.GetPixelID())

        with resource_operation("spatial_correction", "write_rotated_raw"):
            sitk.WriteImage(corrected_raw, str(candidate_raw_path))
        with resource_operation("spatial_correction", "write_rotated_segmentation"):
            sitk.WriteImage(corrected_seg, str(candidate_seg_path))

        params.update({
            'candidate_raw_path': str(candidate_raw_path),
            'candidate_seg_path': str(candidate_seg_path),
            'raw_interpolator': 'linear',
            'seg_interpolator': 'nearest_neighbor',
            'raw_default_value': self._image_min_value(raw_image),
            'seg_default_value': 0,
            'rotation_center_source': (
                'target_bbox' if correction_center_bbox else 'image_center'
            ),
            'rotation_center_index_xyz': center_index_xyz,
        })
        return {
            'raw_for_reconstruction_path': raw_path,
            'seg_for_reconstruction_path': seg_path,
            'corrected_raw_path': raw_path,
            'corrected_seg_path': seg_path,
            'candidate_raw_path': str(candidate_raw_path),
            'candidate_seg_path': str(candidate_seg_path),
            'correction_applied': False,
            'correction_params': params,
        }

    @staticmethod
    def _rotate_image_xy(
        image: sitk.Image,
        rotation_degrees: float,
        interpolator: int,
        default_value: float,
        center_index_xyz: list[float] = None,
    ) -> sitk.Image:
        """
        在当前图像网格内围绕 Z 轴旋转，也就是只修正横状位 XY 平面。

        SimpleITK 的 Resample 使用“输出点映射回输入点”的 transform。
        这里保持尺寸、spacing、origin、direction 不变，只改变采样来源，
        保证 raw 和 seg 仍能与后续 MPR 使用同一套空间信息。
        """
        if image.GetDimension() != 3:
            raise ValueError(f"横状位回正只支持 3D 图像，当前维度: {image.GetDimension()}")

        size = image.GetSize()
        center_index = center_index_xyz or [
            (size[0] - 1) / 2.0,
            (size[1] - 1) / 2.0,
            (size[2] - 1) / 2.0,
        ]
        center_point = image.TransformContinuousIndexToPhysicalPoint(center_index)

        # 候选角属于图像索引 XY 平面。映射到物理坐标后再重采样，
        # 斜扫描和反向输入不能直接当作患者物理 Z 轴旋转。
        angle = math.radians(rotation_degrees)
        local_rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        physical_rotation = direction @ local_rotation @ np.linalg.inv(direction)
        transform = sitk.AffineTransform(3)
        transform.SetCenter(center_point)
        transform.SetMatrix(tuple(physical_rotation.ravel()))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(transform)
        return resampler.Execute(image)

    @staticmethod
    def _rotation_center_index_xyz(
        correction_center_bbox: VoxelBBox,
        image: sitk.Image,
    ) -> list[float]:
        """
        计算回正旋转中心的连续体素索引，顺序为 SimpleITK 使用的 x/y/z。

        优先使用本次定位目标的 bbox 中心；没有 bbox 时退回整图中心，保证独立
        helper 调用也有确定的旋转中心。
        """
        if correction_center_bbox:
            return [
                (correction_center_bbox.x_min + correction_center_bbox.x_max) / 2.0,
                (correction_center_bbox.y_min + correction_center_bbox.y_max) / 2.0,
                (correction_center_bbox.z_min + correction_center_bbox.z_max) / 2.0,
            ]

        size = image.GetSize()
        return [
            (size[0] - 1) / 2.0,
            (size[1] - 1) / 2.0,
            (size[2] - 1) / 2.0,
        ]

    @staticmethod
    def _image_min_value(image: sitk.Image) -> float:
        """
        读取图像最小值作为 raw 回正的边界填充值。

        CT 背景常常不是 0，用最小值通常比硬编码 0 更少引入亮边。
        """
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        return float(stats.GetMinimum())

    @staticmethod
    def _validate_found_labels(label_coordinates: Dict[str, LabelCoordinate]) -> None:
        """
        确认请求定位所需 label 在 seg.nii.gz 中真的存在。

        dataset.json 只说明模型认识哪些标签；实际推理结果里可能没有分割
        出来。缺少目标体素时继续往后切图没有意义，所以这里直接失败。
        """
        missing_names = [
            label_name
            for label_name, coord in label_coordinates.items()
            if not coord.found
        ]
        if missing_names:
            raise ValueError(f"seg.nii.gz 中未找到目标Label体素: {', '.join(missing_names)}")

    @staticmethod
    def _write_debug_label_mask_overlay(
        reference_image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        output_path: Path,
    ) -> str:
        """
        写出研发期分 label mask overlay，用原始模型 label id 区分各个椎体。

        target mask overlay 会把本次目标统一写成 1，适合看整体是否对齐。
        这个 overlay 保留 seg.nii.gz 中的原始 label id，比如 C1/C2/C7 会
        使用各自模型 id，方便在 ITK-SNAP 中按颜色查看每个椎体自己的范围。
        """
        target_mask = LocalizationStage._build_target_mask(seg_array, label_targets)
        overlay_array = np.where(target_mask, seg_array, 0).astype(np.uint16)
        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    @staticmethod
    def _write_debug_vertebral_body_mask_overlay(
        reference_image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        output_path: Path,
        excluded_labels: Iterable[str] = (),
    ) -> str:
        """
        写出椎体主体候选 overlay，用于人工校验是否能排除后方附件。

        该产物只服务 CPR/区间切前的几何可行性验证，不参与主流程。普通椎体
        逐横状位层面提取前方连续 AP 区间和中央 x 段；S1 与生产四棱柱一致，
        使用左右投影支撑筛选和相连上缘补回。最后按原 label id 填色。
        """

        reference_array = sitk.GetArrayFromImage(reference_image)
        overlay_array = np.zeros(reference_array.shape, dtype=np.uint16)
        output_path = Path(output_path)

        excluded_label_set = set(excluded_labels)
        for target in label_targets.values():
            vertebra_key = LocalizationStage._vertebra_key(target.label_name)
            if target.label_name in excluded_label_set:
                continue
            if vertebra_key == "S1":
                body_mask = LocalizationStage._s1_body_candidate_mask(
                    seg_array,
                    target,
                )
            else:
                body_mask = LocalizationStage._vertebral_body_candidate_mask(
                    seg_array,
                    target,
                    spacing_zyx=tuple(reversed(reference_image.GetSpacing())),
                )
            if body_mask is None or not target.label_ids:
                continue
            overlay_array[body_mask] = int(target.label_ids[0])

        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    @staticmethod
    def _write_debug_vertebral_body_prism_mask_overlay(
        reference_image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        output_path: Path,
        excluded_labels: Iterable[str] = (),
    ) -> str:
        """写出按主体四边形和左右边界填满的三维四棱柱 Mask。"""

        overlay_array = np.zeros(seg_array.shape, dtype=np.uint16)
        _, prism_regions = (
            LocalizationStage._build_spine_body_prism_regions(
                seg_array,
                label_targets,
                reference_image,
                excluded_labels=excluded_labels,
            )
        )
        for label_name, target in label_targets.items():
            prism_region = prism_regions.get(label_name)
            if prism_region is None or not target.label_ids:
                continue
            minimum, local_mask = prism_region
            region_slices = tuple(
                slice(int(start), int(start) + int(size))
                for start, size in zip(minimum, local_mask.shape)
            )
            overlay_region = overlay_array[region_slices]
            overlay_region[local_mask] = int(target.label_ids[0])

        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    @classmethod
    def _build_spine_body_prism_regions(
        cls,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        image: sitk.Image,
        sagittal_geometries: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        ] | None = None,
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
        excluded_labels: Iterable[str] = (),
        body_diagnostics: dict[str, dict] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, tuple[np.ndarray, np.ndarray] | None],
    ]:
        """逐椎体提取稀疏主体点和局部棱柱，避免复制整幅三维 mask。"""

        body_indices: dict[str, np.ndarray] = {}
        prism_regions: dict[
            str,
            tuple[np.ndarray, np.ndarray] | None,
        ] = {}
        excluded_label_set = set(excluded_labels)
        for label_name, target in label_targets.items():
            vertebra_key = cls._vertebra_key(label_name)
            if vertebra_key is None or label_name in excluded_label_set:
                continue
            label_region = cls._target_label_region(
                seg_array,
                target,
                label_region=(
                    label_regions.get(label_name)
                    if label_regions is not None
                    else None
                ),
            )
            if label_region is None:
                body_indices[label_name] = np.empty((0, 3), dtype=int)
                prism_regions[label_name] = None
                if sagittal_geometries is not None:
                    sagittal_geometries[label_name] = None
                continue
            region_minimum, local_seg_array = label_region
            # S1 的骶骨方向和腰椎不同，不能套用“前方稳定行 + 中央 x 行”
            # 的普通腰椎启发式；改用专用左右支撑筛选和局部上缘补回。
            extraction_diagnostic = {}
            if vertebra_key == "S1":
                body_mask = cls._s1_body_candidate_mask(
                    local_seg_array,
                    target,
                )
            else:
                body_mask = cls._vertebral_body_candidate_mask(
                    local_seg_array,
                    target,
                    spacing_zyx=tuple(reversed(image.GetSpacing())),
                    diagnostics=extraction_diagnostic,
                )
            if body_diagnostics is not None and extraction_diagnostic:
                core_bbox = extraction_diagnostic.get("core_bbox_zyx")
                if core_bbox is not None:
                    extraction_diagnostic["core_bbox_zyx"] = {
                        side: (np.asarray(core_bbox[side]) + region_minimum).tolist()
                        for side in ("min", "max")
                    }
                body_diagnostics[label_name] = extraction_diagnostic
            indices = (
                np.argwhere(body_mask) + region_minimum
                if body_mask is not None
                else np.empty((0, 3), dtype=int)
            )
            body_indices[label_name] = indices
            sagittal_geometry = (
                cls._sagittal_body_quadrilateral(
                    indices,
                    image,
                    retain_all_projection=(vertebra_key == "S1"),
                )
                if indices.size
                else None
            )
            if sagittal_geometries is not None:
                sagittal_geometries[label_name] = sagittal_geometry
            prism_region = (
                cls._vertebral_body_prism_region(
                    indices,
                    image,
                    retain_all_projection=(vertebra_key == "S1"),
                    sagittal_geometry=sagittal_geometry,
                )
                if indices.size
                else None
            )
            if vertebra_key == "S1" and body_mask is not None:
                prism_region = cls._clip_prism_region_to_mask(
                    prism_region,
                    body_mask,
                    mask_minimum_zyx=region_minimum,
                )
            prism_regions[label_name] = prism_region

        return (
            body_indices,
            cls._resolve_non_overlapping_prism_regions(
                prism_regions,
                protected_indices=body_indices,
                voxel_spacing_zyx=np.asarray(image.GetSpacing())[::-1],
            ),
        )

    @staticmethod
    def _clip_prism_region_to_mask(
        prism_region: tuple[np.ndarray, np.ndarray] | None,
        mask: np.ndarray,
        mask_minimum_zyx: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """把局部棱柱限制在给定 mask 内，用于 S1 这类不规则主体。"""

        if prism_region is None:
            return None
        minimum, local_prism_mask = prism_region
        mask_minimum = (
            np.asarray(mask_minimum_zyx, dtype=int)
            if mask_minimum_zyx is not None
            else np.zeros(3, dtype=int)
        )
        minimum = np.asarray(minimum, dtype=int)
        prism_maximum = minimum + np.asarray(local_prism_mask.shape, dtype=int)
        mask_maximum = mask_minimum + np.asarray(mask.shape, dtype=int)
        overlap_minimum = np.maximum(minimum, mask_minimum)
        overlap_maximum = np.minimum(prism_maximum, mask_maximum)
        if np.any(overlap_minimum >= overlap_maximum):
            return None

        prism_slices = tuple(
            slice(int(start - prism_start), int(stop - prism_start))
            for start, stop, prism_start in zip(
                overlap_minimum,
                overlap_maximum,
                minimum,
            )
        )
        mask_slices = tuple(
            slice(int(start - mask_start), int(stop - mask_start))
            for start, stop, mask_start in zip(
                overlap_minimum,
                overlap_maximum,
                mask_minimum,
            )
        )
        clipped = local_prism_mask[prism_slices] & mask[mask_slices]
        if not np.any(clipped):
            return None

        occupied = np.argwhere(clipped)
        crop_minimum = occupied.min(axis=0)
        crop_maximum = occupied.max(axis=0) + 1
        crop_slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(crop_minimum, crop_maximum)
        )
        return overlap_minimum + crop_minimum, clipped[crop_slices]

    @classmethod
    def _resolve_non_overlapping_prism_regions(
        cls,
        prism_regions: dict[
            str,
            tuple[np.ndarray, np.ndarray] | None,
        ],
        protected_indices: dict[str, np.ndarray] | None = None,
        voxel_spacing_zyx: np.ndarray | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
        """用单一矢状分隔平面拆分重叠；主体不可分时整体失效。"""

        maximum_resolvable_overlap_fraction = 0.05
        resolved_regions = {
            label_name: (
                None
                if prism_region is None
                else (
                    np.asarray(prism_region[0], dtype=int).copy(),
                    np.asarray(prism_region[1], dtype=bool).copy(),
                )
            )
            for label_name, prism_region in prism_regions.items()
        }
        valid_labels = [
            label_name
            for label_name, prism_region in resolved_regions.items()
            if prism_region is not None
        ]
        voxel_counts: dict[str, int] = {}
        centroids: dict[str, np.ndarray] = {}
        protected_masks: dict[str, np.ndarray] = {}
        protected_points: dict[str, np.ndarray] = {}
        for label_name in valid_labels:
            minimum, mask = resolved_regions[label_name]
            protected_mask = np.zeros_like(mask, dtype=bool)
            indices = (
                protected_indices.get(label_name)
                if protected_indices is not None
                else None
            )
            protected_points[label_name] = (
                np.asarray(indices, dtype=float)
                if indices is not None and len(indices)
                else np.empty((0, 3), dtype=float)
            )
            if indices is not None and len(indices):
                local_indices = np.asarray(indices, dtype=int) - minimum
                valid = np.all(
                    (local_indices >= 0)
                    & (local_indices < np.asarray(mask.shape)),
                    axis=1,
                )
                if np.any(valid):
                    protected_mask[
                        tuple(local_indices[valid].T)
                    ] = True
            protected_masks[label_name] = protected_mask
        for label_name in valid_labels:
            minimum, mask = resolved_regions[label_name]
            mask_indices = np.argwhere(mask)
            voxel_counts[label_name] = len(mask_indices)
            centroids[label_name] = minimum + mask_indices.mean(axis=0)

        collided_labels: set[str] = set()
        separating_planes: dict[
            tuple[str, str],
            tuple[np.ndarray, float],
        ] = {}
        for left_index, left_label in enumerate(valid_labels):
            left_minimum, left_mask = resolved_regions[left_label]
            left_maximum = left_minimum + np.asarray(left_mask.shape)
            for right_label in valid_labels[left_index + 1:]:
                right_minimum, right_mask = resolved_regions[right_label]
                right_maximum = right_minimum + np.asarray(right_mask.shape)
                overlap_minimum = np.maximum(left_minimum, right_minimum)
                overlap_maximum = np.minimum(left_maximum, right_maximum)
                if np.any(overlap_minimum >= overlap_maximum):
                    continue
                left_slices = tuple(
                    slice(
                        int(overlap_minimum[axis] - left_minimum[axis]),
                        int(overlap_maximum[axis] - left_minimum[axis]),
                    )
                    for axis in range(3)
                )
                right_slices = tuple(
                    slice(
                        int(overlap_minimum[axis] - right_minimum[axis]),
                        int(overlap_maximum[axis] - right_minimum[axis]),
                    )
                    for axis in range(3)
                )
                overlap_voxel_count = int(
                    np.count_nonzero(
                        left_mask[left_slices] & right_mask[right_slices]
                    )
                )
                if overlap_voxel_count == 0:
                    continue
                left_points = protected_points[left_label]
                right_points = protected_points[right_label]
                protected_overlap = int(
                    np.count_nonzero(
                        protected_masks[left_label][left_slices]
                        & protected_masks[right_label][right_slices]
                    )
                )
                # 主体真实占用同一体素时不存在无损直线拆分，不能使用二维
                # 投影容错掩盖三维碰撞。
                if protected_overlap > 0:
                    collided_labels.update((left_label, right_label))
                    continue
                if len(left_points) and len(right_points):
                    separating_plane = cls._sagittal_body_separating_plane(
                        left_points,
                        right_points,
                        voxel_spacing_zyx=voxel_spacing_zyx,
                    )
                    if separating_plane is None:
                        collided_labels.update((left_label, right_label))
                    else:
                        separating_planes[(left_label, right_label)] = (
                            separating_plane
                        )
                left_overlap_fraction = (
                    overlap_voxel_count / voxel_counts[left_label]
                )
                right_overlap_fraction = (
                    overlap_voxel_count / voxel_counts[right_label]
                )
                if max(left_overlap_fraction, right_overlap_fraction) > (
                    maximum_resolvable_overlap_fraction
                ):
                    # 外包络变大后，相邻椎体的规则化空间可能大范围相交，
                    # 只要两者主体本身不相交，就继续拆分外部重叠；只有主体
                    # 真的相交，或调用方没有主体保护信息时，才按碰撞整体失效。
                    if (
                        protected_indices is None
                        or protected_overlap > 0
                    ):
                        collided_labels.update((left_label, right_label))
        for label_name in collided_labels:
            resolved_regions[label_name] = None

        resolvable_labels = [
            label_name
            for label_name in valid_labels
            if label_name not in collided_labels
        ]
        for left_index, left_label in enumerate(resolvable_labels):
            left_minimum, left_mask = resolved_regions[left_label]
            left_maximum = left_minimum + np.asarray(left_mask.shape)
            for right_label in resolvable_labels[left_index + 1:]:
                right_minimum, right_mask = resolved_regions[right_label]
                right_maximum = right_minimum + np.asarray(right_mask.shape)
                overlap_minimum = np.maximum(left_minimum, right_minimum)
                overlap_maximum = np.minimum(left_maximum, right_maximum)
                if np.any(overlap_minimum >= overlap_maximum):
                    continue
                left_slices = tuple(
                    slice(
                        int(overlap_minimum[axis] - left_minimum[axis]),
                        int(overlap_maximum[axis] - left_minimum[axis]),
                    )
                    for axis in range(3)
                )
                right_slices = tuple(
                    slice(
                        int(overlap_minimum[axis] - right_minimum[axis]),
                        int(overlap_maximum[axis] - right_minimum[axis]),
                    )
                    for axis in range(3)
                )
                left_overlap = left_mask[left_slices]
                right_overlap = right_mask[right_slices]
                left_protected = protected_masks[left_label][left_slices]
                right_protected = protected_masks[right_label][right_slices]
                overlap_indices = np.argwhere(left_overlap & right_overlap)
                if len(overlap_indices) == 0:
                    continue

                global_overlap_indices = overlap_minimum + overlap_indices
                separating_plane = separating_planes.get(
                    (left_label, right_label)
                )
                if separating_plane is not None:
                    plane_normal, plane_offset = separating_plane
                    spacing = cls._normalized_voxel_spacing_zyx(
                        voxel_spacing_zyx,
                    )
                    sagittal_points = (
                        global_overlap_indices[:, :2].astype(float)
                        * spacing[:2]
                    )
                    plane_values = sagittal_points @ plane_normal
                    assign_to_left = plane_values < plane_offset
                    on_separator = np.isclose(
                        plane_values,
                        plane_offset,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    assign_to_left[on_separator] = left_label < right_label
                else:
                    left_distances = np.sum(
                        (global_overlap_indices - centroids[left_label]) ** 2,
                        axis=1,
                    )
                    right_distances = np.sum(
                        (global_overlap_indices - centroids[right_label]) ** 2,
                        axis=1,
                    )
                    assign_to_left = left_distances < right_distances
                    equal_distances = np.isclose(
                        left_distances,
                        right_distances,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    assign_to_left[equal_distances] = left_label < right_label
                left_is_protected = left_protected[tuple(overlap_indices.T)]
                right_is_protected = right_protected[
                    tuple(overlap_indices.T)
                ]
                # 容差直线允许极少数投影点越线；逐 LR 体素的主体保护
                # 仍优先于直线归属，确保主体体素不会被误删。
                assign_to_left[
                    left_is_protected & ~right_is_protected
                ] = True
                assign_to_left[
                    ~left_is_protected & right_is_protected
                ] = False
                remove_from_left = ~assign_to_left & ~left_is_protected
                remove_from_right = assign_to_left & ~right_is_protected
                if np.any(remove_from_left):
                    indices = overlap_indices[remove_from_left]
                    left_overlap[tuple(indices.T)] = False
                if np.any(remove_from_right):
                    indices = overlap_indices[remove_from_right]
                    right_overlap[tuple(indices.T)] = False
        return resolved_regions

    @staticmethod
    def _normalized_voxel_spacing_zyx(
        voxel_spacing_zyx: np.ndarray | None,
    ) -> np.ndarray:
        """返回可用于索引点物理缩放的 zyx spacing。"""

        if voxel_spacing_zyx is None:
            return np.ones(3, dtype=float)
        spacing = np.asarray(voxel_spacing_zyx, dtype=float)
        if (
            spacing.shape != (3,)
            or np.any(~np.isfinite(spacing))
            or np.any(spacing <= 0.0)
        ):
            return np.ones(3, dtype=float)
        return spacing

    @staticmethod
    def _unique_packed_integer_rows_2d(
        values: np.ndarray,
        *,
        return_inverse: bool = False,
        return_counts: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """用一维整数键稳定去重二维整数坐标。"""

        rows = np.asarray(values)
        if rows.ndim != 2 or rows.shape[1] != 2:
            raise ValueError("二维坐标去重输入必须是 N x 2 数组")
        if not np.issubdtype(rows.dtype, np.integer):
            raise TypeError("二维坐标去重输入必须是整数数组")

        if rows.size == 0:
            inverse = (
                np.empty(0, dtype=np.intp)
                if return_inverse
                else None
            )
            counts = (
                np.empty(0, dtype=np.intp)
                if return_counts
                else None
            )
            return np.empty((0, 2), dtype=np.int64), inverse, counts

        if np.issubdtype(rows.dtype, np.unsignedinteger):
            maximum = int(np.max(rows))
            if maximum > np.iinfo(np.int64).max:
                raise ValueError("二维坐标超出 int64 可表示范围")
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        first_minimum = int(np.min(rows[:, 0]))
        first_maximum = int(np.max(rows[:, 0]))
        second_minimum = int(np.min(rows[:, 1]))
        second_maximum = int(np.max(rows[:, 1]))
        first_span = first_maximum - first_minimum + 1
        second_span = second_maximum - second_minimum + 1
        maximum_key = first_span * second_span - 1

        if (
            second_span <= np.iinfo(np.int64).max
            and maximum_key <= np.iinfo(np.int64).max
        ):
            first_shifted = rows[:, 0] - first_minimum
            second_shifted = rows[:, 1] - second_minimum
            packed_keys = np.ascontiguousarray(
                first_shifted * second_span + second_shifted,
                dtype=np.int64,
            )
            unique_result = np.unique(
                packed_keys,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            if return_inverse or return_counts:
                unique_keys = unique_result[0]
                result_index = 1
                inverse = (
                    unique_result[result_index]
                    if return_inverse
                    else None
                )
                result_index += int(return_inverse)
                counts = (
                    unique_result[result_index]
                    if return_counts
                    else None
                )
            else:
                unique_keys = unique_result
                inverse = None
                counts = None
            unique_rows = np.column_stack((
                unique_keys // second_span + first_minimum,
                unique_keys % second_span + second_minimum,
            )).astype(np.int64, copy=False)
            return unique_rows, inverse, counts

        # 医学图像索引不会触发该分支；保留 Python 整数回退，避免极端
        # 坐标跨度在打包乘法中溢出后悄悄合并不同坐标。
        row_tuples = [
            (int(first), int(second))
            for first, second in rows
        ]
        unique_tuples = sorted(set(row_tuples))
        unique_rows = np.asarray(unique_tuples, dtype=np.int64)
        row_to_index = {
            row: index
            for index, row in enumerate(unique_tuples)
        }
        inverse_for_counts = None
        if return_inverse or return_counts:
            inverse_for_counts = np.fromiter(
                (row_to_index[row] for row in row_tuples),
                dtype=np.intp,
                count=len(row_tuples),
            )
        inverse = inverse_for_counts if return_inverse else None
        counts = (
            np.bincount(
                inverse_for_counts,
                minlength=len(unique_tuples),
            ).astype(np.intp, copy=False)
            if return_counts
            else None
        )
        return unique_rows, inverse, counts

    @classmethod
    def _sagittal_body_separating_plane(
        cls,
        left_indices: np.ndarray,
        right_indices: np.ndarray,
        voxel_spacing_zyx: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """求严格分隔直线；仅对不超过 5% 的投影侵入启用稳健回退。"""

        left = np.asarray(left_indices, dtype=float)
        right = np.asarray(right_indices, dtype=float)
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[1] < 2
            or right.shape[1] < 2
            or len(left) == 0
            or len(right) == 0
        ):
            return None
        spacing = cls._normalized_voxel_spacing_zyx(voxel_spacing_zyx)
        if (
            np.any(~np.isfinite(left[:, :2]))
            or np.any(~np.isfinite(right[:, :2]))
        ):
            return None
        left_projection_indices = np.rint(left[:, :2]).astype(np.int64)
        right_projection_indices = np.rint(right[:, :2]).astype(np.int64)
        if (
            not np.allclose(left[:, :2], left_projection_indices)
            or not np.allclose(right[:, :2], right_projection_indices)
        ):
            return None
        left_unique_indices, _, _ = cls._unique_packed_integer_rows_2d(
            left_projection_indices,
        )
        right_unique_indices, _, _ = cls._unique_packed_integer_rows_2d(
            right_projection_indices,
        )
        left_points = left_unique_indices.astype(float) * spacing[:2]
        right_points = right_unique_indices.astype(float) * spacing[:2]
        left_hull = cls._convex_hull_2d(left_points)
        right_hull = cls._convex_hull_2d(right_points)
        center_direction = right_points.mean(axis=0) - left_points.mean(axis=0)
        center_length = float(np.linalg.norm(center_direction))
        if center_length <= 1e-9:
            return None

        candidate_normals: list[np.ndarray] = [
            center_direction / center_length,
        ]
        for hull in (left_hull, right_hull):
            if len(hull) < 2:
                continue
            for start, end in zip(hull, np.roll(hull, -1, axis=0)):
                edge = end - start
                edge_length = float(np.linalg.norm(edge))
                if edge_length <= 1e-9:
                    continue
                candidate_normals.append(
                    np.asarray([edge[1], -edge[0]], dtype=float)
                    / edge_length
                )

        best: tuple[float, np.ndarray, float] | None = None
        for candidate in candidate_normals:
            normal = np.asarray(candidate, dtype=float)
            if float(normal @ center_direction) < 0.0:
                normal = -normal
            left_projection = left_points @ normal
            right_projection = right_points @ normal
            left_maximum = float(np.max(left_projection))
            right_minimum = float(np.min(right_projection))
            gap = right_minimum - left_maximum
            if best is None or gap > best[0]:
                best = (
                    gap,
                    normal,
                    (left_maximum + right_minimum) / 2.0,
                )
        if best is not None and best[0] > 1e-6:
            return best[1], best[2]

        maximum_intrusion_fraction = (
            _PRISM_SEPARATOR_MAXIMUM_PROJECTION_INTRUSION_FRACTION
        )
        robust_best: tuple[
            tuple[float, float, float, float, float, float, float],
            np.ndarray,
            float,
        ] | None = None
        # 仅在严格分隔失败后逐候选投影，避免同时保存“候选数 × 点数”
        # 的临时数组，也保证容错不会改变原有严格分隔结果。
        for candidate in candidate_normals:
            normal = np.asarray(candidate, dtype=float)
            if float(normal @ center_direction) < 0.0:
                normal = -normal
            left_projection = left_points @ normal
            right_projection = right_points @ normal
            left_sorted = np.sort(left_projection)
            right_sorted = np.sort(right_projection)
            thresholds = np.unique(np.concatenate((
                left_sorted,
                right_sorted,
                np.asarray([
                    (
                        float(np.median(left_sorted))
                        + float(np.median(right_sorted))
                    ) / 2.0,
                ]),
            )))
            left_wrong = (
                len(left_sorted)
                - np.searchsorted(left_sorted, thresholds, side="right")
            ) / len(left_sorted)
            right_wrong = np.searchsorted(
                right_sorted,
                thresholds,
                side="left",
            ) / len(right_sorted)
            maximum_wrong = np.maximum(left_wrong, right_wrong)
            valid = maximum_wrong <= maximum_intrusion_fraction + 1e-12
            if not np.any(valid):
                continue

            candidate_indices = np.flatnonzero(valid)
            center_threshold = (
                float(np.median(left_sorted))
                + float(np.median(right_sorted))
            ) / 2.0
            selected_index = min(
                candidate_indices,
                key=lambda index: (
                    float(maximum_wrong[index]),
                    float(left_wrong[index] + right_wrong[index]),
                    abs(float(left_wrong[index] - right_wrong[index])),
                    abs(float(thresholds[index]) - center_threshold),
                    int(index),
                ),
            )
            threshold = float(thresholds[selected_index])
            canonical_normal = np.asarray(normal, dtype=float).copy()
            canonical_threshold = threshold
            first_nonzero = next(
                (
                    float(value)
                    for value in canonical_normal
                    if abs(float(value)) > 1e-12
                ),
                0.0,
            )
            if first_nonzero < 0.0:
                canonical_normal = -canonical_normal
                canonical_threshold = -canonical_threshold
            score = (
                float(maximum_wrong[selected_index]),
                float(left_wrong[selected_index] + right_wrong[selected_index]),
                abs(float(left_wrong[selected_index] - right_wrong[selected_index])),
                abs(threshold - center_threshold),
                float(canonical_normal[0]),
                float(canonical_normal[1]),
                float(canonical_threshold),
            )
            if robust_best is None or score < robust_best[0]:
                robust_best = (score, normal, threshold)

        if robust_best is None:
            return None
        return robust_best[1], robust_best[2]

    @classmethod
    def _build_spine_body_geometry(
        cls,
        slice_part: str,
        image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
        excluded_labels: Iterable[str] = (),
        interspace_regions: Iterable[Dict[str, Any]] | None = None,
        raw_array_loader: Callable[[], np.ndarray] | None = None,
    ) -> Dict[str, Any] | None:
        """
        为第五阶段生成脊柱过滤主体、四棱柱和相邻主体间隙几何。

        这里不修改原始 label 的质心和 bbox。完整 label 负责定位及覆盖范围，
        过滤主体负责 CPR 质心、椎间四棱柱和真实最近间隙参考；配置排除的标签仍
        保留在完整定位中，但不进入这些主体几何。物理坐标直接由当前工作图像转换，
        保证与第四阶段定位空间一致。
        """

        if slice_part not in {"颈椎", "胸椎", "腰椎", "脊柱"}:
            return None

        excluded_label_set = set(excluded_labels)
        sagittal_geometries: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        ] = {}
        body_diagnostics: dict[str, dict] = {}
        body_indices, prism_regions = cls._build_spine_body_prism_regions(
            seg_array,
            label_targets,
            image,
            sagittal_geometries=sagittal_geometries,
            label_regions=label_regions,
            excluded_labels=excluded_label_set,
            body_diagnostics=body_diagnostics,
        )

        raw_array = None
        if raw_array_loader is not None and body_diagnostics:
            try:
                loaded = raw_array_loader()
                if loaded.shape != seg_array.shape:
                    raise ValueError("端面密度验证的原图与分割形状不一致")
                raw_array = loaded
            except (RuntimeError, OSError, ValueError) as error:
                DebugLog(f"  原图端面密度验证不可用，保留几何回退: {error}")

        filtered_annotation_centroids: Dict[str, Dict[str, Any]] = {}
        for label_name, target in label_targets.items():
            vertebra_key = cls._vertebra_key(label_name)
            if vertebra_key is None or label_name in excluded_label_set:
                continue

            indices = body_indices.get(
                label_name,
                np.empty((0, 3), dtype=int),
            )
            if indices.size == 0:
                filtered_annotation_centroids[label_name] = {
                    "found": False,
                    "voxel_count": 0,
                    "body_bbox": None,
                    "centroid_voxel": None,
                    "centroid_physical": None,
                    "local_superior_inferior_axis": None,
                    "local_superior_inferior_extent_mm": None,
                    "local_superior_inferior_face_centroids": None,
                    "local_superior_inferior_face_candidates": None,
                    "sagittal_body_faces": None,
                    "sagittal_body_quadrilateral": None,
                    "prism_found": False,
                    "prism_voxel_count": 0,
                    "prism_bbox": None,
                    "prism_centroid_voxel": None,
                    "prism_centroid_physical": None,
                }
                continue

            centroid_z, centroid_y, centroid_x = indices.mean(axis=0)
            physical_x, physical_y, physical_z = image.TransformContinuousIndexToPhysicalPoint(
                (float(centroid_x), float(centroid_y), float(centroid_z))
            )
            (
                local_axis,
                local_axis_extent,
                local_face_centroids,
                local_face_candidates,
            ) = cls._estimate_local_superior_inferior_geometry(
                indices,
                image,
            )
            sagittal_body_faces = cls._estimate_sagittal_body_faces(
                indices,
                image,
                retain_all_projection=(vertebra_key == "S1"),
                sagittal_geometry=sagittal_geometries.get(label_name),
            )
            sagittal_geometry = sagittal_geometries.get(label_name)
            sagittal_body_quadrilateral = None
            if sagittal_geometry is not None:
                sagittal_center, sagittal_axes, sagittal_quad = sagittal_geometry
                indices_xyz = indices[:, [2, 1, 0]].astype(float)
                spacing = np.asarray(image.GetSpacing(), dtype=float)
                origin = np.asarray(image.GetOrigin(), dtype=float)
                direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
                physical_points = origin + (indices_xyz * spacing) @ direction.T
                left_right_axis = direction[:, 0]
                left_right_projection = (
                    (physical_points - sagittal_center) @ left_right_axis
                )
                physical_vertices = sagittal_center + (
                    sagittal_quad @ sagittal_axes.T
                )
                sagittal_body_quadrilateral = {
                    "center": {
                        "x": float(sagittal_center[0]),
                        "y": float(sagittal_center[1]),
                        "z": float(sagittal_center[2]),
                    },
                    "vertices": [
                        {
                            "x": float(vertex[0]),
                            "y": float(vertex[1]),
                            "z": float(vertex[2]),
                        }
                        for vertex in physical_vertices
                    ],
                    "left_right_extent_mm": {
                        "min": float(np.min(left_right_projection)),
                        "max": float(np.max(left_right_projection)),
                    },
                }
            prism_region = prism_regions.get(label_name)
            if prism_region is not None:
                prism_minimum, local_prism_mask = prism_region
                local_coordinates = np.nonzero(local_prism_mask)
                prism_voxel_count = int(len(local_coordinates[0]))
            else:
                prism_minimum = None
                local_coordinates = None
                prism_voxel_count = 0
            if prism_voxel_count:
                prism_z, prism_y, prism_x = (
                    float(prism_minimum[axis])
                    + float(np.mean(local_coordinates[axis]))
                    for axis in range(3)
                )
                prism_physical = image.TransformContinuousIndexToPhysicalPoint(
                    (
                        float(prism_x),
                        float(prism_y),
                        float(prism_z),
                    )
                )
            else:
                prism_z = prism_y = prism_x = None
                prism_physical = None
            filtered_annotation_centroids[label_name] = {
                "found": True,
                "voxel_count": int(indices.shape[0]),
                "body_bbox": cls._bbox_from_indices(indices),
                "centroid_voxel": {
                    "z": float(centroid_z),
                    "y": float(centroid_y),
                    "x": float(centroid_x),
                },
                "centroid_physical": {
                    "x": float(physical_x),
                    "y": float(physical_y),
                    "z": float(physical_z),
                },
                "local_superior_inferior_axis": (
                    {
                        "x": float(local_axis[0]),
                        "y": float(local_axis[1]),
                        "z": float(local_axis[2]),
                    }
                    if local_axis is not None
                    else None
                ),
                "local_superior_inferior_extent_mm": (
                    {
                        "min": float(local_axis_extent[0]),
                        "max": float(local_axis_extent[1]),
                    }
                    if local_axis_extent is not None
                    else None
                ),
                "local_superior_inferior_face_centroids": (
                    {
                        "min": {
                            "x": float(local_face_centroids[0][0]),
                            "y": float(local_face_centroids[0][1]),
                            "z": float(local_face_centroids[0][2]),
                        },
                        "max": {
                            "x": float(local_face_centroids[1][0]),
                            "y": float(local_face_centroids[1][1]),
                            "z": float(local_face_centroids[1][2]),
                        },
                    }
                    if local_face_centroids is not None
                    else None
                ),
                "local_superior_inferior_face_candidates": local_face_candidates,
                "sagittal_body_faces": sagittal_body_faces,
                "sagittal_body_quadrilateral": sagittal_body_quadrilateral,
                "prism_found": bool(prism_voxel_count),
                "prism_voxel_count": prism_voxel_count,
                "prism_bbox": (
                    cls._bbox_from_region(prism_minimum, local_prism_mask)
                    if prism_voxel_count and prism_minimum is not None
                    else None
                ),
                "prism_centroid_voxel": (
                    {
                        "z": float(prism_z),
                        "y": float(prism_y),
                        "x": float(prism_x),
                    }
                    if prism_voxel_count
                    else None
                ),
                "prism_centroid_physical": (
                    {
                        "x": float(prism_physical[0]),
                        "y": float(prism_physical[1]),
                        "z": float(prism_physical[2]),
                    }
                    if prism_physical is not None
                    else None
                ),
            }
            full_indices = (
                label_regions[label_name].indices_zyx
                if label_regions is not None and label_name in label_regions
                else np.argwhere(np.isin(seg_array, target.label_ids))
            )
            extraction_diagnostic = body_diagnostics.get(label_name)
            if extraction_diagnostic:
                filtered_annotation_centroids[label_name]["body_extraction"] = extraction_diagnostic
                endplates = cls._estimate_supported_body_endplates(
                    indices, image, extraction_diagnostic,
                    source_indices_zyx=full_indices,
                    raw_array=raw_array,
                )
                if endplates is not None:
                    filtered_annotation_centroids[label_name]["sagittal_body_endplates"] = endplates
            # 完整椎体的左右对称面与过滤主体端面职责不同，不能用主体
            # 截面最长轴代替左右方向。将可靠姿态传给整切和椎间斜切规划。
            filtered_annotation_centroids[label_name].update(
                estimate_spine_label_symmetry(full_indices, image)
            )

        interspace_body_pairs: Dict[str, Dict[str, Any]] = {}
        pair_definitions = []
        if interspace_regions is None:
            ordered_body_labels = sorted(
                (
                    label_name
                    for label_name, indices in body_indices.items()
                    if indices.size > 0
                ),
                key=lambda label_name: cls._vertebra_order_index(label_name),
            )
            pair_definitions = [
                (
                    f"{cls._vertebra_key(upper_label)}_"
                    f"{cls._vertebra_key(lower_label)}",
                    upper_label,
                    lower_label,
                )
                for upper_label, lower_label in zip(
                    ordered_body_labels,
                    ordered_body_labels[1:],
                )
                if (
                    cls._vertebra_order_index(upper_label) is not None
                    and cls._vertebra_order_index(lower_label)
                    == cls._vertebra_order_index(upper_label) + 1
                )
            ]
        else:
            for interval in interspace_regions:
                region = interval.get("region")
                start_labels = interval.get("start_labels") or []
                end_labels = interval.get("end_labels") or []
                if not region or not start_labels or not end_labels:
                    raise ValueError(f"椎间隙区域配置无效: {interval}")
                pair_definitions.append(
                    (
                        str(region),
                        cls._resolve_spine_endpoint_label(str(start_labels[0]), label_targets),
                        cls._resolve_spine_endpoint_label(str(end_labels[0]), label_targets),
                    )
                )

        recovered_labels = {
            name for name, diagnostic in body_diagnostics.items()
            if diagnostic.get("method") == "density_core_recovery"
        }
        for _, upper_label, lower_label in pair_definitions:
            if not recovered_labels.intersection((upper_label, lower_label)):
                continue
            for label_name in (upper_label, lower_label):
                if cls._vertebra_key(label_name) in {"C1", "C2", "S1"}:
                    continue
                annotation = filtered_annotation_centroids.get(label_name, {})
                indices = body_indices.get(label_name)
                if annotation.get("sagittal_body_endplates") or indices is None or not indices.size:
                    continue
                # 覆盖包络的端边可能被裁平；恢复体的正常邻体也需用自然
                # 边界配对。只补充端面证据，主体 mask 和质心不改变。
                minimum, maximum = np.quantile(indices, (0.25, 0.75), axis=0)
                support = {
                    "method": "adjacent_recovered_body",
                    "core_bbox_zyx": {"min": np.ceil(minimum).astype(int).tolist(),
                                       "max": np.floor(maximum).astype(int).tolist()},
                }
                source_indices = (
                    label_regions[label_name].indices_zyx
                    if label_regions is not None and label_name in label_regions
                    else np.argwhere(np.isin(seg_array, label_targets[label_name].label_ids))
                )
                endplates = cls._estimate_supported_body_endplates(
                    indices, image, support, source_indices_zyx=source_indices,
                    raw_array=raw_array,
                )
                if endplates is not None:
                    annotation["sagittal_body_endplates"] = endplates
                    annotation["sagittal_endplate_support"] = support

        for group_name, upper_label, lower_label in pair_definitions:
            upper_indices = body_indices.get(upper_label)
            lower_indices = body_indices.get(lower_label)
            if (
                upper_indices is None
                or lower_indices is None
                or upper_indices.size == 0
                or lower_indices.size == 0
            ):
                continue
            geometry = cls._nearest_body_interspace_geometry(
                image,
                upper_indices,
                lower_indices,
            )
            if geometry is None:
                continue
            interspace_body_pairs[group_name] = {
                "upper_label": upper_label,
                "lower_label": lower_label,
                **geometry,
            }

        return {
            "filtered_annotation": {
                "method": "vertebral_body_candidate_v3",
                "source": "full_nnunet_label",
                "target": "vertebral_body",
            },
            "stage": "stage5_anatomical_analysis",
            "filtered_annotation_centroids": filtered_annotation_centroids,
            "interspace_body_pairs": interspace_body_pairs,
            "excluded_from_annotation_centroids": sorted(excluded_label_set),
            "full_label_geometry": "coords",
            "full_label_bbox": "coords.*.bbox",
        }

    @staticmethod
    def _nearest_body_interspace_geometry(
        image: sitk.Image,
        upper_indices: np.ndarray,
        lower_indices: np.ndarray,
    ) -> Dict[str, Any] | None:
        """从相邻过滤主体体素提取有实际支撑的最近间隙中点。"""

        if upper_indices.size == 0 or lower_indices.size == 0:
            return None

        region_minimum = np.minimum(
            upper_indices.min(axis=0),
            lower_indices.min(axis=0),
        )
        region_maximum = np.maximum(
            upper_indices.max(axis=0),
            lower_indices.max(axis=0),
        )
        region_shape = region_maximum - region_minimum + 1
        lower_mask = np.zeros(tuple(region_shape), dtype=np.uint8)
        lower_local = lower_indices - region_minimum
        lower_mask[tuple(lower_local.T)] = 1

        lower_image = sitk.GetImageFromArray(lower_mask)
        lower_image.SetSpacing(image.GetSpacing())
        distance_filter = sitk.DanielssonDistanceMapImageFilter()
        distance_filter.SetInputIsBinary(True)
        distance_filter.SetUseImageSpacing(True)
        try:
            distance_image = distance_filter.Execute(lower_image)
        except RuntimeError:
            return None
        distance_array = sitk.GetArrayFromImage(distance_image)
        vector_array = sitk.GetArrayFromImage(
            distance_filter.GetVectorDistanceMap()
        )

        upper_local = upper_indices - region_minimum
        upper_distances = distance_array[tuple(upper_local.T)]
        finite_distances = upper_distances[np.isfinite(upper_distances)]
        if finite_distances.size == 0:
            return None
        minimum_distance = float(np.min(finite_distances))
        if minimum_distance <= 0.0:
            return None

        spacing = np.asarray(image.GetSpacing(), dtype=float)
        support_tolerance = max(float(np.min(spacing)) * 0.25, 1e-6)
        supported = np.isfinite(upper_distances) & (
            upper_distances <= minimum_distance + support_tolerance
        )
        supported_upper = upper_local[supported]
        if supported_upper.size == 0:
            return None
        nearest_offsets_xyz = vector_array[tuple(supported_upper.T)]
        supported_lower = (
            supported_upper
            + nearest_offsets_xyz[:, ::-1].astype(int)
        )
        valid = np.all(
            (supported_lower >= 0) & (supported_lower < region_shape),
            axis=1,
        )
        if not np.any(valid):
            return None
        supported_upper = supported_upper[valid]
        supported_lower = supported_lower[valid]
        valid = lower_mask[tuple(supported_lower.T)] > 0
        if not np.any(valid):
            return None
        supported_upper = supported_upper[valid] + region_minimum
        supported_lower = supported_lower[valid] + region_minimum

        origin = np.asarray(image.GetOrigin(), dtype=float)
        direction = np.asarray(
            image.GetDirection(),
            dtype=float,
        ).reshape(3, 3)

        def to_physical(indices_zyx: np.ndarray) -> np.ndarray:
            indices_xyz = indices_zyx[:, ::-1].astype(float)
            return origin + (indices_xyz * spacing) @ direction.T

        upper_physical = to_physical(supported_upper)
        lower_physical = to_physical(supported_lower)
        midpoint_physical = np.median(
            (upper_physical + lower_physical) / 2.0,
            axis=0,
        )
        return {
            "found": True,
            "minimum_distance_mm": minimum_distance,
            "support_point_count": int(len(supported_upper)),
            "midpoint_physical": {
                "x": float(midpoint_physical[0]),
                "y": float(midpoint_physical[1]),
                "z": float(midpoint_physical[2]),
            },
        }

    @staticmethod
    def _estimate_local_superior_inferior_geometry(
        indices_zyx: np.ndarray,
        image: sitk.Image,
    ) -> tuple[
        np.ndarray | None,
        tuple[float, float] | None,
        tuple[np.ndarray, np.ndarray] | None,
        list[dict[str, Any]] | None,
    ]:
        """估计主体头脚轴、范围，以及两端表面中心。"""

        if len(indices_zyx) < 3:
            return None, None, None, None
        indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        origin = np.asarray(image.GetOrigin(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        physical_points = origin + (indices_xyz * spacing) @ direction.T
        centered = physical_points - np.mean(physical_points, axis=0)
        covariance = (centered.T @ centered) / float(len(centered))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.all(np.isfinite(eigenvalues)):
            return None, None, None, None
        axis = eigenvectors[:, int(np.argmin(eigenvalues))]
        axis_index = int(np.argmin(eigenvalues))
        transverse_axes = eigenvectors[:, [index for index in range(3) if index != axis_index]]
        length = float(np.linalg.norm(axis))
        if not math.isfinite(length) or length <= 0.0:
            return None, None, None, None
        axis /= length
        projections = centered @ axis
        min_projection = float(np.min(projections))
        max_projection = float(np.max(projections))
        face_tolerance = max(float(np.min(spacing)) * 1.25, 1e-3)
        def select_face_points(target_projection: float) -> np.ndarray:
            selected = physical_points[
                np.abs(projections - target_projection) <= face_tolerance
            ]
            if selected.size > 0:
                return selected

            # 体素层间距大于容差时，仍取距离目标平面最近的一层。
            nearest_distance = float(
                np.min(np.abs(projections - target_projection))
            )
            return physical_points[
                np.abs(projections - target_projection)
                <= nearest_distance + 1e-6
            ]

        min_face_points = select_face_points(min_projection)
        max_face_points = select_face_points(max_projection)
        if min_face_points.size == 0 or max_face_points.size == 0:
            return axis, (min_projection, max_projection), None, None

        def face_center(
            face_points: np.ndarray,
            boundary_projection: float,
        ) -> np.ndarray:
            """取端面平面内两个主方向范围的中点。"""

            transverse_points = (face_points - physical_points.mean(axis=0)) @ transverse_axes
            transverse_min = np.min(transverse_points, axis=0)
            transverse_max = np.max(transverse_points, axis=0)
            transverse_midpoint = (transverse_min + transverse_max) / 2.0
            return (
                physical_points.mean(axis=0)
                + axis * boundary_projection
                + transverse_axes @ transverse_midpoint
            )

        def face_transverse_spans(face_points: np.ndarray) -> np.ndarray:
            transverse_points = (
                face_points - physical_points.mean(axis=0)
            ) @ transverse_axes
            return np.max(transverse_points, axis=0) - np.min(
                transverse_points,
                axis=0,
            )

        candidates: list[dict[str, Any]] = []
        axis_span = max_projection - min_projection
        terminal_span = axis_span * 0.45
        if terminal_span > 0.0:
            candidate_projections = (
                ("min", np.linspace(
                    min_projection,
                    min_projection + terminal_span,
                    12,
                )),
                ("max", np.linspace(
                    max_projection - terminal_span,
                    max_projection,
                    12,
                )),
            )
            for side, projections_for_side in candidate_projections:
                for candidate_projection in projections_for_side:
                    candidate_points = select_face_points(
                        float(candidate_projection)
                    )
                    if candidate_points.size == 0:
                        continue
                    center = face_center(
                        candidate_points,
                        float(candidate_projection),
                    )
                    transverse_spans = face_transverse_spans(candidate_points)
                    candidates.append({
                        "side": side,
                        "projection": float(candidate_projection),
                        "point_count": int(len(candidate_points)),
                        "transverse_span_mm": {
                            "first": float(transverse_spans[0]),
                            "second": float(transverse_spans[1]),
                        },
                        "face_valid": bool(
                            len(candidate_points) >= 3
                            and float(np.max(transverse_spans))
                            >= float(np.min(spacing)) * 0.75
                        ),
                        "physical": {
                            "x": float(center[0]),
                            "y": float(center[1]),
                            "z": float(center[2]),
                        },
                    })

        return (
            axis,
            (min_projection, max_projection),
            (
                face_center(min_face_points, min_projection),
                face_center(max_face_points, max_projection),
            ),
            candidates,
        )

    @staticmethod
    def _sagittal_body_quadrilateral(
        indices_zyx: np.ndarray,
        image: sitk.Image,
        retain_all_projection: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """返回主体物理中心、矢状二维轴和有序四边形角点。"""

        if len(indices_zyx) < 4:
            return None

        indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        origin = np.asarray(image.GetOrigin(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        physical_points = origin + (indices_xyz * spacing) @ direction.T
        if not retain_all_projection:
            retained_projection_mask = (
                LocalizationStage._retained_sagittal_projection_mask_by_lr_support(
                    indices_zyx,
                    retain_aligned_terminal_extensions=True,
                )
            )
            if np.count_nonzero(retained_projection_mask) >= 4:
                physical_points = physical_points[retained_projection_mask]
        center = physical_points.mean(axis=0)
        sagittal_axes = direction[:, [1, 2]]
        sagittal_points = (physical_points - center) @ sagittal_axes
        rounded_points = np.round(sagittal_points, 3)
        if np.any(~np.isfinite(rounded_points)):
            return None
        scaled_points = np.rint(rounded_points * 1000.0)
        if np.any(np.abs(scaled_points) > np.iinfo(np.int64).max):
            return None
        unique_scaled_points, _, _ = (
            LocalizationStage._unique_packed_integer_rows_2d(
                scaled_points.astype(np.int64),
            )
        )
        unique_points = unique_scaled_points.astype(float) / 1000.0
        hull = LocalizationStage._convex_hull_2d(unique_points)
        quad = LocalizationStage._simplify_hull_to_quadrilateral(hull)
        if quad is None:
            return None
        quad = LocalizationStage._expand_quadrilateral_to_contain_points(
            quad,
            sagittal_points,
        )
        if quad is None:
            quad = LocalizationStage._oriented_enclosing_quadrilateral(
                sagittal_points,
                reference_quadrilateral=hull,
            )
            if quad is None:
                return None
        return center, sagittal_axes, quad

    @staticmethod
    def _retained_sagittal_projection_mask_by_lr_support(
        indices_zyx: np.ndarray,
        minimum_support_fraction: float = 0.35,
        retain_aligned_terminal_extensions: bool = False,
    ) -> np.ndarray:
        """只让 LR 支撑足够的 Y/Z 投影点参与矢状四边形拟合。"""

        if len(indices_zyx) == 0:
            return np.zeros(0, dtype=bool)

        projection_input = np.ascontiguousarray(
            indices_zyx[:, [0, 1]],
            dtype=np.intp,
        )
        projection_points, inverse, support_counts = (
            LocalizationStage._unique_packed_integer_rows_2d(
                projection_input,
                return_inverse=True,
                return_counts=True,
            )
        )
        if support_counts.size == 0:
            return np.ones(len(indices_zyx), dtype=bool)

        max_support = int(np.max(support_counts))
        if max_support <= 2:
            return np.ones(len(indices_zyx), dtype=bool)

        minimum_support_fraction = min(
            1.0,
            max(0.0, float(minimum_support_fraction)),
        )
        minimum_support = max(
            2,
            int(math.ceil(max_support * minimum_support_fraction)),
        )
        retained_projection_points = support_counts >= minimum_support
        supported_indices = np.flatnonzero(retained_projection_points)
        supported_points = projection_points[supported_indices]
        if len(supported_points) < 4:
            return np.ones(len(indices_zyx), dtype=bool)

        minimum = supported_points.min(axis=0)
        local_points = supported_points - minimum
        local_shape = tuple(
            (supported_points.max(axis=0) - minimum + 1).astype(int)
        )
        support_mask = np.zeros(local_shape, dtype=np.uint8)
        support_mask[local_points[:, 0], local_points[:, 1]] = 1
        components = sitk.GetArrayFromImage(
            sitk.ConnectedComponent(
                sitk.GetImageFromArray(support_mask),
                True,
            )
        )
        component_ids = components[
            local_points[:, 0],
            local_points[:, 1],
        ]
        weighted_center = np.average(
            projection_points,
            axis=0,
            weights=support_counts,
        )

        def component_key(
            component_id: int,
        ) -> tuple[int, int, float, int]:
            member_mask = component_ids == component_id
            member_indices = supported_indices[member_mask]
            member_support = support_counts[member_indices]
            member_center = np.average(
                projection_points[member_indices],
                axis=0,
                weights=member_support,
            )
            distance_squared = float(np.sum(
                (member_center - weighted_center) ** 2
            ))
            return (
                int(np.count_nonzero(member_mask)),
                int(np.sum(member_support)),
                -distance_squared,
                -int(component_id),
            )

        selected_component = max(
            (
                int(value)
                for value in np.unique(component_ids)
                if value > 0
            ),
            key=component_key,
        )
        selected_members = component_ids == selected_component
        if np.count_nonzero(selected_members) < 4:
            return np.ones(len(indices_zyx), dtype=bool)

        selected_components = {selected_component}
        if retain_aligned_terminal_extensions:
            # 只补回紧邻主投影上下端、AP 高度重合且自身左右支撑稳定的块。
            # 远端附件即使投影面积较大，也不会重新进入四边形拟合。
            main_points = supported_points[selected_members]
            main_support = int(np.sum(
                support_counts[supported_indices[selected_members]],
            ))
            main_minimum = main_points.min(axis=0)
            main_maximum = main_points.max(axis=0)
            main_point_count = int(len(main_points))

            for component_id in np.unique(component_ids):
                component_id = int(component_id)
                if component_id <= 0 or component_id == selected_component:
                    continue
                component_members = component_ids == component_id
                component_points = supported_points[component_members]
                component_support = int(np.sum(
                    support_counts[supported_indices[component_members]],
                ))
                if (
                    len(component_points)
                    < main_point_count
                    * _TERMINAL_EXTENSION_MINIMUM_COMPONENT_FRACTION
                    or component_support
                    < main_support
                    * _TERMINAL_EXTENSION_MINIMUM_COMPONENT_FRACTION
                ):
                    continue

                component_minimum = component_points.min(axis=0)
                component_maximum = component_points.max(axis=0)
                if not LocalizationStage._is_aligned_terminal_component_bbox(
                    main_minimum,
                    main_maximum,
                    component_minimum,
                    component_maximum,
                    transverse_axes=(1,),
                ):
                    continue
                selected_components.add(component_id)

        selected_projection_points = np.zeros(
            len(projection_points),
            dtype=bool,
        )
        selected_projection_points[
            supported_indices[np.isin(component_ids, list(selected_components))]
        ] = True
        retained_mask = selected_projection_points[inverse]
        return retained_mask

    @staticmethod
    def _estimate_supported_body_endplates(
        body_indices_zyx: np.ndarray,
        image: sitk.Image,
        extraction_diagnostic: dict,
        *,
        source_indices_zyx: np.ndarray | None = None,
        raw_array: np.ndarray | None = None,
    ) -> list[dict] | None:
        """把密实核心支撑的真实上下边界转换到当前工作物理空间。"""

        if extraction_diagnostic.get("method") not in {
            "density_core_recovery", "adjacent_recovered_body",
        }:
            return None
        if raw_array is None:
            return None
        from .spine_body_core import fit_body_endplates_from_core

        faces = fit_body_endplates_from_core(
            body_indices_zyx,
            tuple(reversed(image.GetSpacing())),
            extraction_diagnostic.get("core_bbox_zyx", {}),
            source_indices_zyx=source_indices_zyx,
            raw_array=raw_array,
        )
        if faces is None:
            return None
        direction = np.asarray(image.GetDirection()).reshape(3, 3)
        result = []
        for face in faces:
            endpoints = [
                image.TransformContinuousIndexToPhysicalPoint(tuple(point[::-1]))
                for point in face["endpoints_zyx"]
            ]
            normal = direction @ np.asarray(face["normal_zyx"])[::-1]
            result.append({
                "side": face["side"],
                "endpoints": [dict(zip("xyz", point)) for point in endpoints],
                "normal": dict(zip("xyz", map(float, normal))),
                "point_count": face["point_count"],
                "median_residual_mm": face["median_residual_mm"],
                "method": face["method"],
                "source": face.get("source", "original_label"),
                **{key: face[key] for key in (
                    "lr_profile_count", "lr_profile_available", "lr_profile_extent_voxels",
                    "ct_validation", "ct_min_positive_fraction", "ct_min_normalized_drop",
                    "median_profile_residual_mm",
                ) if key in face},
            })
        return result

    @staticmethod
    def _estimate_sagittal_body_faces(
        indices_zyx: np.ndarray,
        image: sitk.Image,
        retain_all_projection: bool = False,
        sagittal_geometry: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None,
    ) -> list[dict[str, Any]] | None:
        """把过滤椎体主体抽象为矢状投影轮廓上的四个大面。"""

        geometry = sagittal_geometry
        if geometry is None:
            geometry = LocalizationStage._sagittal_body_quadrilateral(
                indices_zyx,
                image,
                retain_all_projection=retain_all_projection,
            )
        if geometry is None:
            return None
        center, sagittal_axes, quad = geometry

        indices_xyz = indices_zyx[:, [2, 1, 0]].astype(float)
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        origin = np.asarray(image.GetOrigin(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        physical_points = origin + (indices_xyz * spacing) @ direction.T
        sagittal_points = (physical_points - center) @ sagittal_axes

        tolerance = max(
            float(min(spacing[1], spacing[2])) * 1.5,
            1e-3,
        )

        faces: list[dict[str, Any]] = []
        signed_area = LocalizationStage._signed_area_2d(quad)
        if abs(signed_area) <= 1e-6:
            return None

        for edge_index in range(4):
            start = quad[edge_index]
            end = quad[(edge_index + 1) % 4]
            edge = end - start
            edge_length = float(np.linalg.norm(edge))
            if not math.isfinite(edge_length) or edge_length <= 1e-6:
                continue
            if signed_area > 0:
                normal_2d = np.asarray([edge[1], -edge[0]], dtype=float)
            else:
                normal_2d = np.asarray([-edge[1], edge[0]], dtype=float)
            normal_2d = normal_2d / float(np.linalg.norm(normal_2d))
            distances = np.abs((sagittal_points - start) @ normal_2d)
            selected = distances <= tolerance
            if not np.any(selected):
                nearest = float(np.min(distances))
                selected = distances <= nearest + 1e-6

            face_center_2d = (start + end) / 2.0
            normal_3d = sagittal_axes @ normal_2d
            normal_length = float(np.linalg.norm(normal_3d))
            if not math.isfinite(normal_length) or normal_length <= 0.0:
                continue
            normal_3d = normal_3d / normal_length
            face_center = center + sagittal_axes @ face_center_2d
            faces.append({
                "axis_index": int(edge_index % 2),
                "side": "min" if edge_index < 2 else "max",
                "point_count": int(np.count_nonzero(selected)),
                "center": {
                    "x": float(face_center[0]),
                    "y": float(face_center[1]),
                    "z": float(face_center[2]),
                },
                "normal": {
                    "x": float(normal_3d[0]),
                    "y": float(normal_3d[1]),
                    "z": float(normal_3d[2]),
                },
            })

        return faces if len(faces) == 4 else None

    @staticmethod
    def _vertebral_body_prism_mask(
        body_mask: np.ndarray,
        image: sitk.Image,
    ) -> np.ndarray | None:
        """把局部棱柱还原为与输入同尺寸的布尔 mask，供独立检查使用。"""

        body_indices = np.argwhere(body_mask)
        prism_region = LocalizationStage._vertebral_body_prism_region(
            body_indices,
            image,
        )
        if prism_region is None:
            return None
        minimum, local_mask = prism_region
        prism_mask = np.zeros_like(body_mask, dtype=bool)
        region_slices = tuple(
            slice(int(start), int(start) + int(size))
            for start, size in zip(minimum, local_mask.shape)
        )
        prism_mask[region_slices] = local_mask
        return prism_mask

    @staticmethod
    def _vertebral_body_prism_region(
        body_indices: np.ndarray,
        image: sitk.Image,
        retain_all_projection: bool = False,
        sagittal_geometry: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """在四棱柱自身外接范围内栅格化，返回局部 zyx mask。"""

        geometry = sagittal_geometry
        if geometry is None:
            geometry = LocalizationStage._sagittal_body_quadrilateral(
                body_indices,
                image,
                retain_all_projection=retain_all_projection,
            )
        if geometry is None:
            return None
        center, sagittal_axes, quad = geometry
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        origin = np.asarray(image.GetOrigin(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)

        body_xyz = body_indices[:, [2, 1, 0]].astype(float)
        body_points = origin + (body_xyz * spacing) @ direction.T
        left_right_axis = direction[:, 0]
        body_lr = (body_points - center) @ left_right_axis
        lr_limits = (float(np.min(body_lr)), float(np.max(body_lr)))

        sagittal_corners = center + np.asarray(quad, dtype=float) @ (
            np.asarray(sagittal_axes, dtype=float).T
        )
        prism_corners = np.asarray([
            sagittal_corner + left_right_axis * lr_limit
            for sagittal_corner in sagittal_corners
            for lr_limit in lr_limits
        ])
        continuous_xyz = ((prism_corners - origin) @ direction) / spacing
        minimum_xyz = np.floor(np.min(continuous_xyz, axis=0)).astype(int)
        maximum_xyz = np.ceil(np.max(continuous_xyz, axis=0)).astype(int)
        image_maximum_xyz = np.asarray(image.GetSize(), dtype=int) - 1
        minimum_xyz = np.maximum(minimum_xyz, 0)
        maximum_xyz = np.minimum(maximum_xyz, image_maximum_xyz)
        if np.any(minimum_xyz > maximum_xyz):
            return None
        minimum = minimum_xyz[::-1]
        maximum = maximum_xyz[::-1]

        grids = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1),
            np.arange(minimum[2], maximum[2] + 1),
            indexing="ij",
        )
        candidates_zyx = np.stack(grids, axis=-1).reshape(-1, 3)
        candidates_xyz = candidates_zyx[:, [2, 1, 0]].astype(float)
        candidate_points = origin + (candidates_xyz * spacing) @ direction.T

        candidate_relative = candidate_points - center
        candidate_lr = candidate_relative @ left_right_axis
        candidate_sagittal = candidate_relative @ sagittal_axes
        tolerance = max(
            float(min(spacing[1], spacing[2])) * 1e-3,
            1e-3,
        )
        selected = (
            (candidate_lr >= lr_limits[0] - tolerance)
            & (candidate_lr <= lr_limits[1] + tolerance)
            & LocalizationStage._points_inside_convex_polygon(
                candidate_sagittal,
                quad,
                tolerance,
            )
        )
        if not np.any(selected):
            return None

        local_mask = np.zeros(
            tuple((maximum - minimum + 1).astype(int)),
            dtype=bool,
        )
        selected_local_zyx = candidates_zyx[selected] - minimum
        local_mask[
            selected_local_zyx[:, 0],
            selected_local_zyx[:, 1],
            selected_local_zyx[:, 2],
        ] = True
        occupied = np.argwhere(local_mask)
        occupied_minimum = occupied.min(axis=0)
        occupied_maximum = occupied.max(axis=0)
        occupied_slices = tuple(
            slice(int(start), int(end) + 1)
            for start, end in zip(occupied_minimum, occupied_maximum)
        )
        return minimum + occupied_minimum, local_mask[occupied_slices]

    @staticmethod
    def _target_label_region(
        seg_array: np.ndarray,
        target: LabelTarget,
        label_region: _LabelVoxelRegion | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """返回目标 label 的全局最小 zyx 起点和局部分割视图。"""

        if label_region is not None:
            if label_region.bbox is None:
                return None
            bbox = label_region.bbox
            minimum = np.asarray(
                [bbox.z_min, bbox.y_min, bbox.x_min],
                dtype=int,
            )
            maximum = np.asarray(
                [bbox.z_max, bbox.y_max, bbox.x_max],
                dtype=int,
            )
        else:
            mask = np.isin(seg_array, target.label_ids)
            indices = np.argwhere(mask)
            if indices.size == 0:
                return None
            minimum = indices.min(axis=0)
            maximum = indices.max(axis=0)
        region_slices = tuple(
            slice(int(start), int(stop) + 1)
            for start, stop in zip(minimum, maximum)
        )
        return minimum, seg_array[region_slices]

    @staticmethod
    def _vertebral_body_candidate_mask(
        seg_array: np.ndarray,
        target: LabelTarget,
        *,
        spacing_zyx: tuple[float, float, float] | None = None,
        diagnostics: dict | None = None,
    ) -> np.ndarray | None:
        """从连续左右支撑和三维体量中选择前方椎体主体。"""

        mask = np.isin(seg_array, target.label_ids)
        if not np.any(mask):
            return None
        # 原始 label 的最大块可能是后方附件。先保留每个块的前方支撑候选，
        # 再比较主体证据，不能在识别主体前按原始最大块的体量删除小块。
        candidates = LocalizationStage._vertebral_body_support_candidates(
            mask, spacing_zyx=spacing_zyx,
        )
        raw_indices = np.argwhere(mask)
        label_voxel_count = len(raw_indices)
        label_si_span = int(np.ptp(raw_indices[:, 0]) + 1)
        center_x = float(np.median(raw_indices[:, 2]))
        del raw_indices
        body_mask = LocalizationStage._select_vertebral_body_support(
            candidates, center_x=center_x,
            spacing_zyx=spacing_zyx,
            label_voxel_count=label_voxel_count, label_si_span=label_si_span,
        )

        if not np.any(body_mask):
            return body_mask
        body_mask = (
            LocalizationStage._retain_main_3d_component_with_terminal_extensions(
                body_mask,
            )
        )
        body_mask = LocalizationStage._trim_thin_terminal_si_slices(body_mask)
        body_mask = LocalizationStage._trim_lr_edge_columns_by_si_height(body_mask)
        # 横向断崖裁剪后，末端残留的相对支撑度会重新下降；再做一次端部
        # 检查，处理“薄片-小回增-薄片”这类分段残留。
        body_mask = LocalizationStage._trim_thin_terminal_si_slices(body_mask)
        # 密度只附加验证已保留的边缘，不能降低原参考高度或放宽旧裁剪。
        body_mask = LocalizationStage._trim_transverse_edges_by_density(body_mask)
        if spacing_zyx is not None and LocalizationStage._vertebra_key(target.label_name) != "C2":
            from .spine_body_core import recover_body_from_dense_core

            recovered = recover_body_from_dense_core(
                mask, body_mask, spacing_zyx, diagnostics=diagnostics,
            )
            if recovered is not None:
                # 已确认旧主体漏掉密实内部。恢复受原 label 和物理窗口限制，
                # 不再用旧的全局 SI 柱高把倾斜端部或 LR 裂开的主体删掉。
                body_mask = recovered
        body_indices = np.argwhere(body_mask)
        if len(body_indices) and (
            len(body_indices) < label_voxel_count * _BODY_REMNANT_MAX_LABEL_VOLUME_FRACTION
            and np.ptp(body_indices[:, 0]) + 1
            < label_si_span * _BODY_REMNANT_MAX_LABEL_SI_SPAN_FRACTION
        ):
            # 同时丢失绝大部分体量和头脚支撑时，只剩短残块；不能把它
            # 当作完整主体生成端面。完整的小主体不按原始最大块比例淘汰。
            body_mask[:] = False
        return body_mask

    @staticmethod
    def _vertebral_body_support_candidates(
        mask: np.ndarray,
        *,
        spacing_zyx: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """用连续 LR 段提取前方带，跨层确认后再跳过薄前突起。"""

        spacing = LocalizationStage._normalized_voxel_spacing_zyx(spacing_zyx)
        # AP/LR 物理长度比较，两边同时除以 LR spacing；等距采样时
        # 比值恰为 1，保留原有阈值的浮点边界行为。
        ap_lr_spacing_ratio = float(spacing[1] / spacing[2])

        def relatively_small_front(bands) -> bool:
            if len(bands) < 2:
                return False
            front, rear = bands[:2]
            # 比较同一轴的前后两带，spacing 因子分别抵消。相近 LR
            # 宽度且 AP 深度不足一半，才能把前带视为相对小的前突起；
            # 大而宽的后方附件、AP 深度相近的真前主体不满足此条件。
            return (len(front[0]) * 2 <= len(rear[0])
                    and front[1] * 3 >= rear[1] * 2
                    and front[1] * 2 <= rear[1] * 3)

        components = sitk.GetArrayFromImage(sitk.ConnectedComponent(
            sitk.GetImageFromArray(mask.astype(np.uint8)), True,
        ))
        indices = np.argwhere(mask)
        component_ids = components[tuple(indices.T)]
        order = np.argsort(component_ids)
        boundaries = np.flatnonzero(np.diff(component_ids[order])) + 1
        candidates = np.zeros_like(mask, dtype=bool)
        for member_indices in np.split(order, boundaries):
            points = indices[member_indices]
            if len(points) < 2:
                continue
            minimum = points.min(axis=0)
            local_points = points - minimum
            local_mask = np.zeros(tuple(local_points.max(axis=0) + 1), dtype=bool)
            local_mask[tuple(local_points.T)] = True
            center_x = float(np.median(local_points[:, 2]))
            si_support_by_y = local_mask.any(axis=2).sum(axis=0)
            stable_si_support = float(si_support_by_y.max())
            slices = []
            for z_index in np.unique(local_points[:, 0]):
                slice_mask = local_mask[int(z_index)]
                row_runs = {
                    int(y): LocalizationStage._central_x_run(slice_mask[int(y)], center_x)
                    for y in np.flatnonzero(slice_mask.any(axis=1))
                }
                # 左右分叉之间的空白不计入主体宽度；少数宽翼也不决定阈值。
                stable_scale = float(np.percentile(
                    [len(run) for run in row_runs.values()], 90,
                ))
                threshold = max(2, int(math.ceil(stable_scale * 0.35)))
                supported = np.asarray([
                    y for y, run in row_runs.items() if len(run) >= threshold
                ], dtype=int)
                if not len(supported):
                    continue
                groups = np.split(supported, np.flatnonzero(np.diff(supported) > 1) + 1)
                row_counts = LocalizationStage._slice_row_run_counts(slice_mask)
                bands = []
                for group in groups:
                    rows = group.tolist()
                    while len(rows) > 1 and row_counts[rows[-1]] > 1:
                        rows.pop()
                    widths = np.array([len(row_runs[y]) for y in rows])
                    median_width = float(np.median(widths))
                    stable_width = int(max(widths[widths <= median_width * 1.25]))
                    # 前后两个宽块之间的细桥只在少数层出现时，不能把整段
                    # AP 当成一个主体。仅切开两端均有宽支撑的连续内部窄颈；
                    # 单侧渐窄的端缘及单行凹陷仍保留。
                    narrow = np.flatnonzero(widths < stable_width * 0.5)
                    necks = np.split(narrow, np.flatnonzero(np.diff(narrow) > 1) + 1)
                    cuts = [neck for neck in necks if len(neck) >= 2
                            and neck[0] > 0 and neck[-1] < len(rows) - 1
                            and np.max(si_support_by_y[np.asarray(rows)[neck]])
                            < stable_si_support * 0.25]
                    keep = np.ones(len(rows), dtype=bool)
                    for neck in cuts:
                        keep[neck] = False
                    kept_rows = np.asarray(rows)[keep]
                    for part in np.split(kept_rows, np.flatnonzero(np.diff(kept_rows) > 1) + 1):
                        if not len(part):
                            continue
                        part_widths = np.array([len(row_runs[int(y)]) for y in part])
                        middle = float(np.median(part_widths))
                        width = int(max(part_widths[part_widths <= middle * 1.25]))
                        bands.append((part.tolist(), width))
                if bands:
                    slices.append((int(z_index), row_runs, bands))
            if not slices:
                continue
            # 先以完整跨层候选判定“前方薄突起”，不能因为某个真实端层较薄
            # 就把该层切换到后方附件。替代带本身也必须有跨层体量支撑。
            front_ratios = [len(bands[0][0]) * ap_lr_spacing_ratio / bands[0][1]
                            for _, _, bands in slices]
            front_volume = sum(len(bands[0][0]) * bands[0][1] for _, _, bands in slices)
            alternatives = [(z, runs, bands) for z, runs, bands in slices if len(bands) > 1]
            alternative_volume = sum(len(bands[1][0]) * bands[1][1]
                                     for _, _, bands in alternatives)
            skip_front = (float(np.median(front_ratios)) < 0.35
                          and len(alternatives) >= len(slices) * 0.5
                          and alternative_volume >= front_volume)
            relative_alternatives = [item for item in alternatives if relatively_small_front(item[2])]
            relative_alternative_volume = sum(
                len(bands[1][0]) * bands[1][1] for _, _, bands in relative_alternatives
            )
            relative_front_rows = {
                y for _, _, bands in relative_alternatives for y in bands[0][0]
            }
            tracked_front_volume = sum(
                len(bands[0][0]) * bands[0][1] for _, _, bands in slices
                if any(y in relative_front_rows for y in bands[0][0])
            )
            # 物理 AP/LR 比未必很扁的前突起，也可能远小于同块内真正
            # 主体。先取得跨层关系和至少两倍体量的证据，不能因一两张
            # 端层出现较大的后带，就把真主体切到后方附件。同一前带
            # 在其他层可能更厚，仍需按 AP 交叠跟踪并计入全部体量；
            # 与前突起无 AP 交叠的主体-only 端层不混入前突起评分。
            skip_relative_front = (
                len(relative_alternatives) >= len(slices) * 0.5
                and relative_alternative_volume >= tracked_front_volume * 2
            )
            for z_index, row_runs, bands in slices:
                replace_front = (
                    len(bands) > 1
                    and len(bands[1][0]) * ap_lr_spacing_ratio >= bands[1][1] * 0.35
                    and (
                        (len(bands[0][0]) * ap_lr_spacing_ratio < bands[0][1] * 0.35
                         and (skip_front or len(bands[0][0]) * bands[0][1]
                              < len(bands[1][0]) * bands[1][1] * 0.1))
                        or (skip_relative_front and relatively_small_front(bands))
                    )
                )
                # 全局识别到薄前突起，也不能删除只有一个主体带的端层，
                # 或替换本层已有充分 AP 支撑的前方主体。
                rows, width = bands[1 if replace_front else 0]
                for y in rows:
                    x_run = LocalizationStage._clip_run_to_width(row_runs[y], center_x, width)
                    candidates[
                        z_index + minimum[0], y + minimum[1],
                        x_run[0] + minimum[2]:x_run[-1] + minimum[2] + 1,
                    ] = True
        return candidates

    @staticmethod
    def _select_vertebral_body_support(
        candidates: np.ndarray,
        *,
        center_x: float | None = None,
        spacing_zyx: tuple[float, float, float] | None = None,
        label_voxel_count: int | None = None,
        label_si_span: int | None = None,
    ) -> np.ndarray:
        """比较跨 SI 支撑与主体形态，避免把第一段小突起当作主体。"""

        if not np.any(candidates):
            return candidates
        spacing = LocalizationStage._normalized_voxel_spacing_zyx(spacing_zyx)
        ap_lr_spacing_ratio = float(spacing[1] / spacing[2])
        components = sitk.GetArrayFromImage(sitk.ConnectedComponent(
            sitk.GetImageFromArray(candidates.astype(np.uint8)), True,
        ))
        indices = np.argwhere(candidates)
        if label_voxel_count is None:
            label_voxel_count = len(indices)
        if label_si_span is None:
            label_si_span = int(np.ptp(indices[:, 0]) + 1)
        if center_x is None:
            center_x = float(np.median(indices[:, 2]))
        component_ids = components[tuple(indices.T)]
        order = np.argsort(component_ids)
        boundaries = np.flatnonzero(np.diff(component_ids[order])) + 1
        records = []
        for member_indices in np.split(order, boundaries):
            points = indices[member_indices]
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            projections, row_widths = np.unique(points[:, :2], axis=0, return_counts=True)
            _, ap_depths = np.unique(projections[:, 0], return_counts=True)
            slice_counts = np.bincount(points[:, 0] - minimum[0])
            supported_z = np.flatnonzero(slice_counts >= max(1.0, slice_counts.max() * 0.1))
            z_groups = np.split(supported_z, np.flatnonzero(np.diff(supported_z) > 1) + 1)
            records.append({
                "id": int(component_ids[member_indices[0]]),
                "volume": len(points),
                "si_support": max(len(group) for group in z_groups),
                "ap_depth": float(np.median(ap_depths)),
                "lr_width": float(np.median(row_widths)),
                "lr_center": float(np.median(points[:, 2])),
                "anterior": float(np.median(points[:, 1])),
                "minimum": minimum,
                "maximum": maximum,
            })

        # 完整主体应有稳定的 AP 面积；少数宽而薄的前方突起或后方横桥，
        # 即使与主体同高，也不能凭“最靠前”抢占主体。比例只用同平面的
        # AP/LR 物理长度比较，避免仅改变左右或前后采样就切换主体。
        # 同轴中心距离及同图组件的体量比例中，spacing 的共同因子抵消。
        central = [
            record for record in records
            if abs(record["lr_center"] - center_x) <= record["lr_width"] * 0.75
        ]
        aligned = central or records
        # 末端安全检查会丢弃的短小后方残块，不能先靠紧凑形状淘汰前体。
        # 复用原标签的体量/跨度判据，避免按大附件的体量误删可用小主体。
        dominant = max(aligned, key=lambda record: record["volume"])
        compact = [
            record for record in aligned
            if record["ap_depth"] * ap_lr_spacing_ratio >= record["lr_width"] * 0.35
            and not (
                record["anterior"] > dominant["anterior"]
                and record["si_support"] < dominant["si_support"] * 0.5
                and record["volume"]
                < label_voxel_count * _BODY_REMNANT_MAX_LABEL_VOLUME_FRACTION
                and record["maximum"][0] - record["minimum"][0] + 1
                < label_si_span * _BODY_REMNANT_MAX_LABEL_SI_SPAN_FRACTION
            )
        ]
        supported = compact or aligned
        maximum_volume = max(record["volume"] for record in supported)
        substantial = [
            record for record in supported
            if record["volume"] >= maximum_volume * 0.1
        ]
        maximum_area = max(record["volume"] / record["si_support"] for record in substantial)
        substantial = [record for record in substantial
                       if record["volume"] / record["si_support"] >= maximum_area * 0.2]
        reference = max(substantial, key=lambda record: record["volume"])
        credible = [record for record in substantial
                    if record["si_support"] >= reference["si_support"] * 0.5]
        selected = min(credible, key=lambda record: (
            record["anterior"], -record["volume"], -record["si_support"], record["id"],
        ))
        selected_ids = [selected["id"]]
        for record in records:
            if record["id"] == selected["id"] or record["volume"] < selected["volume"] * 0.1:
                continue
            if LocalizationStage._is_aligned_terminal_component_bbox(
                selected["minimum"], selected["maximum"],
                record["minimum"], record["maximum"], transverse_axes=(1, 2),
            ):
                selected_ids.append(record["id"])
        return np.isin(components, selected_ids)

    @staticmethod
    def _s1_body_candidate_mask(
        seg_array: np.ndarray,
        target: LabelTarget,
    ) -> np.ndarray | None:
        """提取 S1 主体，保留骶骨主体同时压掉骶翼和后方附件。"""

        mask = np.isin(seg_array, target.label_ids)
        if not np.any(mask):
            return None
        mask = LocalizationStage._discard_small_3d_components(mask)
        indices = np.argwhere(mask)
        if len(indices) < 4:
            return mask

        body_mask = LocalizationStage._s1_body_mask_from_projection_support(
            mask,
            indices,
            minimum_support_fraction=0.65,
        )
        if body_mask is None:
            return mask

        raw_voxel_count = int(np.count_nonzero(mask))
        body_voxel_count = int(np.count_nonzero(body_mask))
        if raw_voxel_count and body_voxel_count / raw_voxel_count < 0.68:
            relaxed_body_mask = (
                LocalizationStage._s1_body_mask_from_projection_support(
                    mask,
                    indices,
                    minimum_support_fraction=0.5,
                )
            )
            if (
                relaxed_body_mask is not None
                and int(np.count_nonzero(relaxed_body_mask)) > body_voxel_count
            ):
                body_mask = LocalizationStage._clip_s1_relaxed_body_to_strict_ap_extent(
                    relaxed_body_mask,
                    body_mask,
                )
        return body_mask

    @staticmethod
    def _clip_s1_relaxed_body_to_strict_ap_extent(
        relaxed_body_mask: np.ndarray,
        strict_body_mask: np.ndarray,
        posterior_margin_voxels: int = 4,
    ) -> np.ndarray:
        """限制 S1 fallback 的后方扩张，避免骶翼/后方附件进入主体。"""

        strict_indices = np.argwhere(strict_body_mask)
        if len(strict_indices) == 0:
            return relaxed_body_mask

        clipped = relaxed_body_mask.copy()
        posterior_limit = int(strict_indices[:, 1].max()) + int(posterior_margin_voxels)
        if posterior_limit + 1 < clipped.shape[1]:
            clipped[:, posterior_limit + 1:, :] = False
        return LocalizationStage._retain_largest_3d_component(clipped)

    @staticmethod
    def _s1_body_mask_from_projection_support(
        mask: np.ndarray,
        indices: np.ndarray,
        minimum_support_fraction: float,
    ) -> np.ndarray | None:
        """按左右支撑度提取 S1 主体，并补回相连的短上缘。"""

        # S1 的头脚轴和普通腰椎不同，不能用固定的“最前方 y 行”。
        # 以左右方向的投影支撑度选主体：骶体在多层左右切片中连续出现，
        # 骶翼/后方附件只在较少的左右位置出现。调用方先用较严格阈值，
        # 占比异常偏小时再放宽一次，避免歪斜 S1 回正后主体被过度缩小。
        retained = LocalizationStage._retained_sagittal_projection_mask_by_lr_support(
            indices,
            minimum_support_fraction=minimum_support_fraction,
        )
        if np.count_nonzero(retained) < 4:
            return None
        body_mask = np.zeros_like(mask, dtype=bool)
        retained_indices = indices[retained]
        body_mask[tuple(retained_indices.T)] = True
        body_mask = LocalizationStage._retain_largest_3d_component(body_mask)

        # 投影支撑阈值对 S1 上缘的小体积连续段尤其敏感。沿图像头侧
        # 连续补回最多一层；候选连通块必须接触当前上缘的一像素二维膨胀，
        # 避免把同一 label 的骶翼和后方孤立附件带入四棱柱。
        retained_indices = np.argwhere(body_mask)
        if len(retained_indices):
            superior_z = int(retained_indices[:, 0].max())
            superior_points = np.argwhere(body_mask[superior_z])
            y_min = int(superior_points[:, 0].min()) - 4
            y_max = int(superior_points[:, 0].max()) + 4
            x_min = int(superior_points[:, 1].min()) - 4
            x_max = int(superior_points[:, 1].max()) + 4
            for z_index in range(superior_z + 1, superior_z + 2):
                if z_index >= mask.shape[0]:
                    break
                candidate = mask[z_index].copy()
                candidate[:max(0, y_min), :] = False
                candidate[min(mask.shape[1], y_max + 1):, :] = False
                candidate[:, :max(0, x_min)] = False
                candidate[:, min(mask.shape[2], x_max + 1):] = False
                if not np.any(candidate):
                    break
                top_slice = body_mask[superior_z]
                padded_top = np.pad(top_slice, 1, mode="constant")
                dilated_top = np.zeros_like(top_slice, dtype=bool)
                for y_offset in range(3):
                    for x_offset in range(3):
                        dilated_top |= padded_top[
                            y_offset:y_offset + top_slice.shape[0],
                            x_offset:x_offset + top_slice.shape[1],
                        ]
                component_image = sitk.ConnectedComponent(
                    sitk.GetImageFromArray(candidate.astype(np.uint8)),
                    True,
                )
                component_array = sitk.GetArrayFromImage(component_image)
                touching_components = np.unique(
                    component_array[dilated_top & candidate],
                )
                touching_components = touching_components[touching_components > 0]
                if touching_components.size:
                    body_mask[z_index] |= np.isin(
                        component_array,
                        touching_components,
                    )
        return LocalizationStage._retain_largest_3d_component(body_mask)

    @staticmethod
    def _retain_largest_3d_component(mask: np.ndarray) -> np.ndarray:
        """保留主体候选中最大的三维连通块，去除外侧孤立残渣。"""

        component_image = sitk.ConnectedComponent(
            sitk.GetImageFromArray(mask.astype(np.uint8)),
            True,
        )
        component_array = sitk.GetArrayFromImage(component_image)
        component_sizes = np.bincount(component_array.ravel())
        if component_sizes.size <= 1:
            return mask
        largest_component = int(np.argmax(component_sizes[1:]) + 1)
        return component_array == largest_component

    @staticmethod
    def _is_aligned_terminal_component_bbox(
        main_minimum: np.ndarray,
        main_maximum: np.ndarray,
        component_minimum: np.ndarray,
        component_maximum: np.ndarray,
        *,
        transverse_axes: tuple[int, ...],
        maximum_si_gap: int = _TERMINAL_EXTENSION_MAXIMUM_SI_GAP,
        minimum_transverse_overlap_fraction: float = (
            _TERMINAL_EXTENSION_MINIMUM_TRANSVERSE_OVERLAP_FRACTION
        ),
        minimum_si_span: int = _TERMINAL_EXTENSION_MINIMUM_SI_SPAN,
    ) -> bool:
        """判断候选包围盒是否为紧邻主体头脚端且横向对齐的延伸段。"""

        main_minimum = np.asarray(main_minimum, dtype=int)
        main_maximum = np.asarray(main_maximum, dtype=int)
        component_minimum = np.asarray(component_minimum, dtype=int)
        component_maximum = np.asarray(component_maximum, dtype=int)
        if not (
            main_minimum.shape
            == main_maximum.shape
            == component_minimum.shape
            == component_maximum.shape
        ):
            return False

        component_si_span = int(
            component_maximum[0] - component_minimum[0] + 1
        )
        if component_si_span < max(1, int(minimum_si_span)):
            return False
        if component_maximum[0] < main_minimum[0]:
            si_gap = int(main_minimum[0] - component_maximum[0] - 1)
        elif component_minimum[0] > main_maximum[0]:
            si_gap = int(component_minimum[0] - main_maximum[0] - 1)
        else:
            return False
        if si_gap > max(0, int(maximum_si_gap)):
            return False

        minimum_overlap = min(
            1.0,
            max(0.0, float(minimum_transverse_overlap_fraction)),
        )
        for axis in transverse_axes:
            if axis <= 0 or axis >= len(main_minimum):
                return False
            overlap = max(
                0,
                int(
                    min(main_maximum[axis], component_maximum[axis])
                    - max(main_minimum[axis], component_minimum[axis])
                    + 1
                ),
            )
            main_span = int(main_maximum[axis] - main_minimum[axis] + 1)
            component_span = int(
                component_maximum[axis] - component_minimum[axis] + 1
            )
            if min(main_span, component_span) <= 0:
                return False
            if overlap / float(min(main_span, component_span)) < minimum_overlap:
                return False
        return bool(transverse_axes)

    @staticmethod
    def _retain_main_3d_component_with_terminal_extensions(
        mask: np.ndarray,
        minimum_component_fraction: float = (
            _TERMINAL_EXTENSION_MINIMUM_COMPONENT_FRACTION
        ),
        maximum_si_gap: int = _TERMINAL_EXTENSION_MAXIMUM_SI_GAP,
        minimum_transverse_overlap_fraction: float = (
            _TERMINAL_EXTENSION_MINIMUM_TRANSVERSE_OVERLAP_FRACTION
        ),
        minimum_si_span: int = _TERMINAL_EXTENSION_MINIMUM_SI_SPAN,
    ) -> np.ndarray:
        """保留主体及可信的头脚端断连段，避免薄断层截短椎体。"""

        component_image = sitk.ConnectedComponent(
            sitk.GetImageFromArray(mask.astype(np.uint8)),
            True,
        )
        component_array = sitk.GetArrayFromImage(component_image)
        component_sizes = np.bincount(component_array.ravel())
        if component_sizes.size <= 1:
            return mask

        largest_component = int(np.argmax(component_sizes[1:]) + 1)
        largest_size = int(component_sizes[largest_component])
        main_indices = np.argwhere(component_array == largest_component)
        if main_indices.size == 0:
            return mask
        main_minimum = main_indices.min(axis=0)
        main_maximum = main_indices.max(axis=0)

        retained_components = [largest_component]
        minimum_fraction = max(0.0, float(minimum_component_fraction))

        for component_id in range(1, int(component_sizes.size)):
            if component_id == largest_component:
                continue
            component_size = int(component_sizes[component_id])
            if component_size < largest_size * minimum_fraction:
                continue

            component_indices = np.argwhere(component_array == component_id)
            component_minimum = component_indices.min(axis=0)
            component_maximum = component_indices.max(axis=0)
            if not LocalizationStage._is_aligned_terminal_component_bbox(
                main_minimum,
                main_maximum,
                component_minimum,
                component_maximum,
                transverse_axes=(1, 2),
                maximum_si_gap=maximum_si_gap,
                minimum_transverse_overlap_fraction=(
                    minimum_transverse_overlap_fraction
                ),
                minimum_si_span=minimum_si_span,
            ):
                continue
            retained_components.append(component_id)

        return np.isin(component_array, retained_components)

    @staticmethod
    def _trim_thin_terminal_si_slices(
        mask: np.ndarray,
        minimum_support_fraction: float = 0.10,
        minimum_terminal_slice_count: int = 2,
    ) -> np.ndarray:
        """裁掉轴向两端连续的低支撑薄片，避免撑大主体外包络。"""

        if mask.ndim != 3:
            return mask
        occupied = np.flatnonzero(mask.any(axis=(1, 2)))
        if occupied.size < 4:
            return mask
        slice_counts = mask.sum(axis=(1, 2)).astype(float)
        maximum_support = float(np.max(slice_counts[occupied]))
        if maximum_support <= 0.0:
            return mask
        support_fraction = min(
            1.0,
            max(0.0, float(minimum_support_fraction)),
        )
        minimum_support = max(
            2.0,
            maximum_support * support_fraction,
        )
        trimmed = mask.copy()
        for direction in (1, -1):
            ordered = occupied if direction == 1 else occupied[::-1]
            low_support_indices: list[int] = []
            for z_index in ordered:
                if slice_counts[int(z_index)] >= minimum_support:
                    break
                low_support_indices.append(int(z_index))
            if len(low_support_indices) < int(minimum_terminal_slice_count):
                continue
            trimmed[low_support_indices, :, :] = False
        return trimmed

    @staticmethod
    def _discard_small_3d_components(
        mask: np.ndarray,
        minimum_largest_fraction: float = 0.2,
    ) -> np.ndarray:
        """在主体规则运行前丢弃小离群块，保留较大附件供规则判别。"""

        component_image = sitk.ConnectedComponent(
            sitk.GetImageFromArray(mask.astype(np.uint8)),
            True,
        )
        component_array = sitk.GetArrayFromImage(component_image)
        component_sizes = np.bincount(component_array.ravel())
        if component_sizes.size <= 2:
            return mask

        foreground_sizes = component_sizes[1:]
        largest_size = int(np.max(foreground_sizes))
        minimum_size = max(
            1,
            int(math.ceil(largest_size * minimum_largest_fraction)),
        )
        retained_components = np.flatnonzero(foreground_sizes >= minimum_size) + 1
        return np.isin(component_array, retained_components)

    @staticmethod
    def _trim_lr_edge_columns_by_si_height(mask: np.ndarray) -> np.ndarray:
        """按横向位置的 SI 高度裁掉 AP/LR 边缘断崖式低高度残留。"""

        trimmed = LocalizationStage._trim_transverse_edge_by_si_height(
            mask,
            horizontal_axis=1,
        )
        return LocalizationStage._trim_transverse_edge_by_si_height(
            trimmed,
            horizontal_axis=2,
        )

    @staticmethod
    def _si_pillar_heights(mask: np.ndarray) -> np.ndarray:
        """批量计算每个 y/x 柱的前景 z 闭区间高度。"""

        if mask.ndim != 3:
            raise ValueError("SI 柱高计算要求三维 mask")
        if mask.shape[0] == 0:
            return np.zeros(mask.shape[1:], dtype=np.intp)

        occupied = np.any(mask, axis=0)
        first = np.argmax(mask, axis=0)
        last = mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0)
        return np.where(occupied, last - first + 1, 0).astype(
            np.intp,
            copy=False,
        )

    @staticmethod
    def _trim_transverse_edge_by_si_height(
        mask: np.ndarray,
        horizontal_axis: int,
    ) -> np.ndarray:
        """沿 AP 或 LR 方向裁掉边缘 SI 高度明显不足的横向位置。"""

        if horizontal_axis not in (1, 2):
            return mask

        reduce_axis = 2 if horizontal_axis == 1 else 1
        positions = np.flatnonzero(mask.any(axis=(0, reduce_axis)))
        if positions.size < 4:
            return mask

        heights = np.zeros(mask.shape[horizontal_axis], dtype=float)
        pillar_heights = LocalizationStage._si_pillar_heights(mask)
        for position in positions:
            position_heights = (
                pillar_heights[int(position), :]
                if horizontal_axis == 1
                else pillar_heights[:, int(position)]
            )
            position_heights = position_heights[position_heights > 0]
            if position_heights.size == 0:
                continue
            heights[int(position)] = float(np.percentile(position_heights, 75))

        supported_heights = heights[positions]
        stable_height = float(np.percentile(supported_heights, 75))
        if stable_height < 3.0:
            return mask
        minimum_edge_height = max(2, int(math.ceil(stable_height * 0.6)))

        left = 0
        while (
            left < positions.size - 1
            and heights[int(positions[left])] < minimum_edge_height
        ):
            left += 1

        right = positions.size - 1
        while (
            right > left
            and heights[int(positions[right])] < minimum_edge_height
        ):
            right -= 1

        trimmed_minimum = int(positions[left])
        trimmed_maximum = int(positions[right])
        if (
            trimmed_minimum == int(positions[0])
            and trimmed_maximum == int(positions[-1])
        ):
            return mask

        trimmed = mask.copy()
        if horizontal_axis == 1:
            trimmed[:, :trimmed_minimum, :] = False
            trimmed[:, trimmed_maximum + 1:, :] = False
        else:
            trimmed[:, :, :trimmed_minimum] = False
            trimmed[:, :, trimmed_maximum + 1:] = False
        return trimmed

    @staticmethod
    def _trim_transverse_edges_by_density(mask: np.ndarray) -> np.ndarray:
        """从稳定主体向外验证连续占据密度，只裁除严重空壳外缘。"""

        if mask.ndim != 3 or not np.any(mask):
            return mask
        spans = LocalizationStage._si_pillar_heights(mask)
        continuous = si_pillar_continuous_support(mask)
        occupied_counts = np.count_nonzero(mask, axis=0)
        # 几何跨度仍保留正常孔洞、倾斜端缘和分段主体。只有跨度内大半
        # 为空、最长可容短缺口链也不足三分之一时，才把首尾残片降权。
        hollow = (continuous * 3 < spans) & (occupied_counts * 2 < spans)
        if not np.any(hollow):
            return mask
        supported = np.where(hollow, continuous, spans)
        bounds = {}
        minimum_core_height = 0
        for axis in (1, 2):
            reduce_axis = 2 if axis == 1 else 1
            positions = np.flatnonzero(mask.any(axis=(0, reduce_axis)))
            if len(positions) < 4:
                continue
            reference = np.zeros(len(positions), dtype=float)
            verified = np.zeros_like(reference)
            for index, position in enumerate(positions):
                original_column = spans[position, :] if axis == 1 else spans[:, position]
                verified_column = (supported[position, :] if axis == 1
                                   else supported[:, position])
                occupied = original_column > 0
                reference[index] = np.percentile(original_column[occupied], 75)
                verified[index] = np.percentile(verified_column[occupied], 75)
            # 参考来自未经降权的跨度，不能因空壳多就降低整个主体门槛。
            stable_height = float(np.percentile(reference, 75))
            if stable_height < 3.0:
                continue
            minimum_height = max(2, int(math.ceil(stable_height * 0.6)))
            if not np.any(verified >= minimum_height):
                # 找不到可信核心时不凭密度挑出任意一行，保留上游结果。
                continue
            minimum_core_height = max(minimum_core_height, minimum_height)
            unsupported = (verified < minimum_height) & (verified < reference)
            left, right = 0, len(positions) - 1
            while left < right and unsupported[left]:
                left += 1
            while right > left and unsupported[right]:
                right -= 1
            if left > 0 or right < len(positions) - 1:
                bounds[axis] = (int(positions[left]), int(positions[right]))
        if not bounds:
            return mask
        # 两个方向分别可靠的区域可能互不重叠，交集也必须保留真正的核心。
        core_slice = tuple(
            slice(bounds[axis][0], bounds[axis][1] + 1)
            if axis in bounds else slice(None)
            for axis in (1, 2)
        )
        if not np.any(
            (supported[core_slice] >= minimum_core_height) & ~hollow[core_slice]
        ):
            return mask
        trimmed = mask.copy()
        for axis, (minimum, maximum) in bounds.items():
            before = [slice(None)] * 3
            after = [slice(None)] * 3
            before[axis] = slice(None, minimum)
            after[axis] = slice(maximum + 1, None)
            trimmed[tuple(before)] = False
            trimmed[tuple(after)] = False
        return trimmed

    @staticmethod
    def _slice_row_widths(slice_mask: np.ndarray) -> dict[int, int]:
        """统计单个横状位层面每个 AP 行的左右跨度宽度。"""

        row_widths: dict[int, int] = {}
        for y_index in np.unique(np.argwhere(slice_mask)[:, 0]):
            x_indices = np.flatnonzero(slice_mask[int(y_index), :])
            if x_indices.size == 0:
                continue
            row_widths[int(y_index)] = int(x_indices.max() - x_indices.min() + 1)
        return row_widths

    @staticmethod
    def _slice_row_run_counts(slice_mask: np.ndarray) -> dict[int, int]:
        """统计单个横状位层面每个 AP 行的左右连续段数量。"""

        row_run_counts: dict[int, int] = {}
        for y_index in np.unique(np.argwhere(slice_mask)[:, 0]):
            x_indices = np.flatnonzero(slice_mask[int(y_index), :])
            if x_indices.size == 0:
                continue
            row_run_counts[int(y_index)] = int(
                1 + np.count_nonzero(np.diff(x_indices) > 1)
            )
        return row_run_counts

    @staticmethod
    def _first_anterior_body_row_run(
        row_widths: dict[int, int],
        row_run_counts: dict[int, int] | None = None,
    ) -> list[int]:
        """从前向后选择第一段足够宽的连续 AP 行，并裁掉末端分裂附件。"""

        if not row_widths:
            return []
        max_width = max(row_widths.values())
        width_threshold = max(2, int(math.ceil(max_width * 0.35)))
        wide_rows = [
            y_index
            for y_index, width in sorted(row_widths.items())
            if width >= width_threshold
        ]
        if not wide_rows:
            return []

        current_run = [wide_rows[0]]
        for y_index in wide_rows[1:]:
            if y_index != current_run[-1] + 1:
                return current_run
            current_run.append(y_index)

        if row_run_counts:
            while len(current_run) > 1 and row_run_counts.get(current_run[-1], 1) > 1:
                current_run.pop()
        return current_run

    @staticmethod
    def _stable_central_run_width(
        slice_mask: np.ndarray,
        row_run: list[int],
        center_x: float,
    ) -> int | None:
        """估计同一轴向层主体 AP 行的稳定 LR 宽度，避开少数侧向凸出。"""

        run_lengths = []
        for y_index in row_run:
            x_run = LocalizationStage._central_x_run(
                slice_mask[int(y_index), :],
                center_x,
            )
            if x_run:
                run_lengths.append(len(x_run))
        if not run_lengths:
            return None
        median_width = float(np.median(run_lengths))
        core_widths = [
            width
            for width in run_lengths
            if float(width) <= median_width * 1.25
        ]
        if not core_widths:
            return int(round(median_width))
        return max(1, int(max(core_widths)))

    @staticmethod
    def _clip_run_to_width(
        x_run: list[int],
        center_x: float,
        target_width: int | None,
    ) -> list[int]:
        """把异常变宽的连续行裁成最靠近左右中心的稳定宽度。"""

        if target_width is None or len(x_run) <= target_width:
            return x_run
        target_width = max(1, int(target_width))
        best_start_index = min(
            range(0, len(x_run) - target_width + 1),
            key=lambda start_index: abs(
                (
                    x_run[start_index]
                    + x_run[start_index + target_width - 1]
                ) / 2.0
                - center_x
            ),
        )
        return x_run[best_start_index:best_start_index + target_width]

    @staticmethod
    def _central_x_run(row_mask: np.ndarray, center_x: float) -> list[int]:
        """选择单个 AP 行上最靠近左右中心的连续 x 段。"""

        x_indices = np.flatnonzero(row_mask)
        if x_indices.size == 0:
            return []

        runs: list[list[int]] = []
        current_run = [int(x_indices[0])]
        for x_index in x_indices[1:]:
            if int(x_index) != current_run[-1] + 1:
                runs.append(current_run)
                current_run = []
            current_run.append(int(x_index))
        runs.append(current_run)

        return min(
            runs,
            key=lambda run: (
                abs(((min(run) + max(run)) / 2.0) - center_x),
                -len(run),
            ),
        )

class LocalizationStage(LocalizationExecutionMixin, LocalizationOperations):
    """第四阶段：在校正阶段选定的空间中计算 label 质心和 bbox。"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, stage_name="localization")
