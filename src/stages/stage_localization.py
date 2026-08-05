#!/src/services/stage_localization.py
# -*- coding: utf-8 -*-

"""
阶段三：定位与横状位 XY 回正候选
椎体/目标 label 定位、目标 bbox 输出，以及不切换主流程输入的回正候选诊断
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import math
import os

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import (
    AnatomicalGeometryResult,
    LabelCoordinate,
    LabelTarget,
    LocalizationResult,
    LocalizationStageResult,
    NiftiImageInfo,
    VoxelBBox,
    VoxelPoint,
)
from src.models.enums import PlaneTypeEnum
from src.anatomy.lung import estimate_lung_frame_with_diagnostics
from src.anatomy.hip import estimate_hip_frame_with_diagnostics
from src.anatomy.spine import estimate_spine_frame_with_diagnostics
from src.utils import DebugLog, is_debug_log_file_enabled


@dataclass(frozen=True)
class _LabelVoxelRegion:
    """同一分割空间内可复用的 label 体素索引和包围盒。"""

    indices_zyx: np.ndarray
    bbox: VoxelBBox | None


class LocalizationStage(BaseStage):
    """定位与回正占位阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化定位配置、实验性回正阈值和 nnUNet 数据集路径。"""

        super().__init__(name="localization", config=config)
        # 这些阈值暂时只给实验开关使用。默认主流程不会自动回正，但保留
        # 阈值字段，方便后续真正设计回正算法时继续沿用 metadata 结构。
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
        # 阶段三先写候选回正图和诊断信息；候选 raw/seg 双文件可用时，
        # 后续定位几何、阶段四和阶段五统一切到候选图像空间。
        # 旧实验开关仍只服务 `_correct_images_if_needed` 兼容入口。
        self.correction_requested = bool(
            self.config.get('correction_requested', True)
        )
        self.enable_global_correction = bool(
            self.config.get('enable_experimental_global_correction', False)
        )
        self._load_config()
        self._load_nnunet_config()
        self._apply_injected_nnunet_config()

    def _apply_injected_nnunet_config(self):
        """用 Pipeline 注入的统一配置覆盖旧的阶段内文件配置。"""

        project_root = Path(__file__).resolve().parents[2]
        for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
            if key in self.config:
                self._nnunet_paths[key] = self._resolve_config_path(
                    self.config.get(key, ''),
                    project_root,
                )

    def _load_config(self):
        """
        加载定位/切割范围配置。

        配置只描述“某个部位在某种切割方式下，需要哪些定位范围”。
        层厚、层间距、切几张等采样参数不放在这里，后续由外部请求传入。
        """
        config_paths = [
            Path.cwd() / "config" / "localization_config.json",
            Path.cwd() / "localization_config.json",
            Path(__file__).parent.parent / "config" / "localization_config.json",
        ]

        self._localization_config = {}
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._localization_config = json.load(f)
                DebugLog(f"[定位] 加载配置文件: {config_path}")
                return

        DebugLog("[定位] 未找到 localization_config.json，使用空配置")

    def _load_nnunet_config(self):
        """
        加载 nnUNet 路径配置，用于根据模型 ID 查找 dataset.json。

        定位阶段不重新匹配模型，只复用 segmentation 阶段得到的 model_id。
        这里读取路径配置，是为了把 label 名称转换成真实 label id。
        """
        project_root = Path(__file__).resolve().parents[2]
        injected_keys = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
        if any(key in self.config for key in injected_keys):
            self._nnunet_paths = {
                key: self._resolve_config_path(self.config.get(key, ''), project_root)
                for key in injected_keys
            }
            return

        config_paths = [
            Path.cwd() / "config" / "nnuNet_config.json",
            Path.cwd() / "nnuNet_config.json",
            Path(__file__).parent.parent / "config" / "nnuNet_config.json",
        ]

        self._nnunet_paths = {}
        for config_path in config_paths:
            if not config_path.exists():
                continue
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._nnunet_paths = {
                'nnUNet_results': self._resolve_config_path(config.get('nnUNet_results', ''), project_root),
                'nnUNet_preprocessed': self._resolve_config_path(config.get('nnUNet_preprocessed', ''), project_root),
                'nnUNet_raw': self._resolve_config_path(config.get('nnUNet_raw', ''), project_root),
            }
            return

    @staticmethod
    def _resolve_config_path(value: str, project_root: Path) -> str:
        """展开环境变量和用户目录，并按项目根目录解析相对配置路径。"""

        if not value:
            return ""
        expanded = os.path.expanduser(os.path.expandvars(str(value)))
        path = Path(expanded)
        if not path.is_absolute():
            path = project_root / path
        return str(path)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        定位请求目标并生成阶段三结果。

        本阶段解析目标标签、提取质心和包围盒，并为已适配部位计算解剖坐标；
        当前回正仍为安全占位，阶段四和阶段五默认继续使用原始图像空间。
        """

        request = context.request
        prev_result = context.get_result('segmentation')
        preprocessing_result = context.get_result('preprocessing') or {}
        
        if not prev_result:
            raise ValueError("前置阶段 'segmentation' 结果不存在")
        
        seg_path = prev_result.get('seg_path')
        raw_path = (
            preprocessing_result.get('nifti_path')
            or prev_result.get('input_nifti')
            or str(Path(request.output_dir) / 'nifti' / 'raw_0000.nii.gz')
        )
        model_info = prev_result.get('model') or {}
        if not seg_path:
            raise ValueError("分割结果路径不存在")
        
        DebugLog("=" * 60)
        DebugLog("【阶段三】定位与回正占位 - 目标Label定位")
        DebugLog(f"  输入原图: {raw_path}")
        DebugLog(f"  输入分割: {seg_path}")
        DebugLog(f"  部位: {request.body_part.value}")
        cut_mode = self._resolve_request_cut_mode(request)
        # 同一次请求可能同时包含腰椎横状位区间切和矢/冠整切，因此这里
        # 保留当前部位的全部切法配置。第三步仍只用 cut_config 定位本轮
        # 所需标签，第四步则按每个计划自己的 cut_mode 读取 cut_configs。
        cut_configs = self._resolve_cut_configs(request)
        cut_config = cut_configs.get(cut_mode)
        if not cut_config:
            raise ValueError(
                f"未找到切割方式配置: body_part={request.body_part.value}, "
                f"cut_mode={cut_mode}"
            )
        resolved_cut_config = self._resolve_label_aliases(cut_config)
        DebugLog(f"  定位范围模式: {cut_mode}")
        DebugLog(f"  切割配置: {cut_config}")
        DebugLog(f"  标签配置: {resolved_cut_config}")
        DebugLog("=" * 60)
        
        output_path = Path(request.output_dir) / 'metadata'
        if is_debug_log_file_enabled():
            output_path.mkdir(parents=True, exist_ok=True)
        
        # [1] resolve_labels: 获取Label列表
        DebugLog("  [1] resolve_labels: 获取目标Label列表...")
        dataset_json_path = self._find_dataset_json(model_info, seg_path=seg_path)
        label_map = self._load_label_map(dataset_json_path)
        label_targets = self._resolve_label_targets(resolved_cut_config, label_map)
        DebugLog(f"    dataset.json: {dataset_json_path}")
        DebugLog(f"    目标Label: {label_targets}")
        
        # [2] extract_label_coordinates: 提取目标坐标
        DebugLog("  [2] extract_label_coordinates: 提取目标质心+包围盒...")
        image = sitk.ReadImage(str(seg_path))
        seg_array = sitk.GetArrayFromImage(image)
        label_regions = self._build_label_voxel_regions(seg_array, label_targets)
        label_coordinates = self._extract_label_coordinates(
            seg_array,
            label_targets,
            label_regions=label_regions,
        )
        self._validate_found_labels(label_coordinates)
        target_mask = self._build_target_mask(seg_array, label_targets)
        correction_scope_bbox = self._build_target_bbox(target_mask)
        DebugLog(f"    已提取 {len(label_coordinates)} 个Label坐标")
        DebugLog(f"    目标整体bbox: {correction_scope_bbox.to_dict()}")
        
        # [3] detect_skew: 估计横状位 XY 回正候选。
        DebugLog("  [3] detect_skew: 估计横状位 XY 回正候选...")
        xy_correction_candidate = self._build_xy_correction_candidate(
            seg_array,
            label_targets,
            body_part=request.body_part.value,
            label_regions=label_regions,
        )
        DebugLog(
            "    横状位 XY 候选角: "
            f"{xy_correction_candidate['angle_degrees']:.3f}°, "
            f"status={xy_correction_candidate['correction_status']}"
        )

        # [4] correct_skew: 写出候选回正图；候选文件可用时提升为本轮主流程输入。
        nifti_path = Path(request.output_dir) / 'nifti'
        DebugLog("  [4] correct_skew: 写出候选回正图，并选择后续工作输入...")
        correction_result = self._write_xy_correction_candidate_images(
            raw_path=str(raw_path),
            seg_path=str(seg_path),
            output_dir=nifti_path,
            candidate=xy_correction_candidate,
            correction_center_bbox=correction_scope_bbox,
        )
        correction_result = self._promote_xy_correction_candidate_if_available(
            correction_result,
        )

        if correction_result['correction_applied']:
            DebugLog(
                "    已启用回正候选，阶段三几何和后续阶段使用回正 raw/seg"
            )
            image = sitk.ReadImage(str(correction_result['seg_for_reconstruction_path']))
            seg_array = sitk.GetArrayFromImage(image)
            label_regions = self._build_label_voxel_regions(
                seg_array,
                label_targets,
            )
            label_coordinates = self._extract_label_coordinates(
                seg_array,
                label_targets,
                label_regions=label_regions,
            )
            self._validate_found_labels(label_coordinates)
            target_mask = self._build_target_mask(seg_array, label_targets)
            correction_scope_bbox = self._build_target_bbox(target_mask)
            DebugLog(
                "    回正空间目标整体bbox: "
                f"{correction_scope_bbox.to_dict()}"
            )
        else:
            DebugLog("    未启用主流程回正，后续阶段继续使用原始 raw/seg")

        label_masks = {}
        if request.body_part.value in {"颈椎", "胸椎", "髋骨"}:
            for label_name, region in label_regions.items():
                label_mask = np.zeros(seg_array.shape, dtype=bool)
                if region.indices_zyx.size:
                    label_mask[tuple(region.indices_zyx.T)] = True
                label_masks[label_name] = label_mask
        start_labels = resolved_cut_config.get('start_labels') or []
        end_labels = resolved_cut_config.get('end_labels') or []
        DebugLog("    [5] build_anatomical_geometry: 开始")
        anatomical_geometry, anatomical_diagnostics = self._build_anatomical_geometry(
            body_part=request.body_part.value,
            target_mask=target_mask,
            image=image,
            label_masks=label_masks,
            start_label=start_labels[0] if start_labels else None,
            end_label=end_labels[0] if end_labels else None,
        )
        if anatomical_geometry is not None:
            frame = anatomical_geometry.global_frame
            DebugLog(
                "    部位解剖坐标: "
                f"method={frame.method}, confidence={frame.confidence:.3f}, "
                f"applied={frame.applied}"
            )
            for warning in frame.warnings:
                DebugLog(f"    解剖坐标警告: {warning}")

        # 脊柱专用几何派生：保留完整 label 的 coords/bbox，同时输出过滤标注
        # 计算出的标注质心。非脊柱部位不生成这份结果。
        DebugLog("    [6] build_spine_body_geometry: 开始")
        spine_geometry = self._build_spine_body_geometry(
            body_part=request.body_part.value,
            image=image,
            seg_array=seg_array,
            label_targets=label_targets,
            label_regions=label_regions,
        )
        DebugLog("    [6] build_spine_body_geometry: 完成")
        if spine_geometry is not None:
            DebugLog(
                "    脊柱过滤标注与标注质心: "
                f"{len(spine_geometry['filtered_annotation_centroids'])} 个标注质心，"
                "全量 coords/bbox 保持不变"
            )

        # 研发期调试输出：定位框和目标二值掩膜信息已可由标签掩膜覆盖，
        # 为缩短 debug 轮耗时，不再默认写出这两类重复产物。
        debug_label_mask_overlay_path = output_path / '标签掩膜.nii.gz'
        if is_debug_log_file_enabled():
            self._write_debug_label_mask_overlay(
            reference_image=image,
            seg_array=seg_array,
            label_targets=label_targets,
            output_path=debug_label_mask_overlay_path,
        )
            DebugLog(f"    调试分label mask overlay: {debug_label_mask_overlay_path}")
        debug_vertebral_body_mask_overlay_path = output_path / '椎体主体掩膜.nii.gz'
        if is_debug_log_file_enabled():
            self._write_debug_vertebral_body_mask_overlay(
            reference_image=image,
            seg_array=seg_array,
            label_targets=label_targets,
            output_path=debug_vertebral_body_mask_overlay_path,
        )
            DebugLog(f"    调试椎体主体 mask overlay: {debug_vertebral_body_mask_overlay_path}")
        debug_vertebral_body_prism_mask_overlay_path = (
            output_path / '椎体四棱柱掩膜.nii.gz'
        )
        if is_debug_log_file_enabled():
            self._write_debug_vertebral_body_prism_mask_overlay(
                reference_image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=debug_vertebral_body_prism_mask_overlay_path,
            )
            DebugLog(
                "    调试椎体四棱柱 mask overlay: "
                f"{debug_vertebral_body_prism_mask_overlay_path}"
            )
        debug_xy_correction_crosshair_overlay_path = (
            output_path / '横状位XY回正十字线.nii.gz'
        )
        if is_debug_log_file_enabled():
            self._write_xy_correction_crosshair_overlay(
                reference_image=image,
                output_path=debug_xy_correction_crosshair_overlay_path,
                correction_center_bbox=correction_scope_bbox,
                candidate=xy_correction_candidate,
            )
            DebugLog(
                "    调试横状位 XY 回正十字线 overlay: "
                f"{debug_xy_correction_crosshair_overlay_path}"
            )

        localization_result_path = output_path / 'localization_result.json'
        if not is_debug_log_file_enabled():
            localization_result_path = ""
        image_info = NiftiImageInfo(
            size_xyz=list(image.GetSize()),
            spacing_xyz=list(image.GetSpacing()),
            origin_xyz=list(image.GetOrigin()),
            direction=list(image.GetDirection()),
        ).to_dict()
        result = {
            'coords': self._coordinates_to_dict(label_coordinates),
            'cut_mode': cut_mode,
            'cut_config': cut_config,
            'cut_configs': cut_configs,
            'resolved_cut_config': resolved_cut_config,
            'spine_cpr_config': self._resolve_spine_cpr_config(
                request.body_part.value,
            ),
            'label_targets': self._label_targets_to_dict(label_targets),
            'dataset_json_path': str(dataset_json_path),
            'localization_result_path': str(localization_result_path),
            # 阶段四需要图像空间信息把第三步的 z/y/x 体素坐标换算成 x/y/z
            # 物理坐标；这里只写 JSON 文件不够，必须同步放入 context 结果。
            'image': image_info,
            'raw_for_reconstruction_path': correction_result['raw_for_reconstruction_path'],
            'seg_for_reconstruction_path': correction_result['seg_for_reconstruction_path'],
            'correction_applied': correction_result['correction_applied'],
            'correction_angle_degrees': xy_correction_candidate['angle_degrees'],
            'correction_scope_bbox': correction_scope_bbox.to_dict(),
            'correction_params': correction_result['correction_params'],
            'corrected_raw_path': correction_result['corrected_raw_path'],
            'corrected_seg_path': correction_result['corrected_seg_path'],
            'candidate_raw_path': correction_result['candidate_raw_path'],
            'candidate_seg_path': correction_result['candidate_seg_path'],
            'is_skewed': correction_result['correction_applied'],
            'skew_params': correction_result['correction_params'],
            'spine_geometry': spine_geometry,
            'debug_bbox_overlay_path': '',
            'debug_target_mask_overlay_path': '',
            'debug_label_mask_overlay_path': str(debug_label_mask_overlay_path) if is_debug_log_file_enabled() else '',
            'debug_vertebral_body_mask_overlay_path': str(debug_vertebral_body_mask_overlay_path) if is_debug_log_file_enabled() else '',
            'debug_vertebral_body_prism_mask_overlay_path': str(debug_vertebral_body_prism_mask_overlay_path) if is_debug_log_file_enabled() else '',
            'debug_xy_correction_crosshair_overlay_path': str(debug_xy_correction_crosshair_overlay_path) if is_debug_log_file_enabled() else '',
        }
        if anatomical_geometry is not None:
            result['anatomical_geometry'] = anatomical_geometry.to_dict()
            result['anatomical_geometry_diagnostics'] = anatomical_diagnostics

        if is_debug_log_file_enabled():
            self._write_localization_result(
            localization_result_path,
            request=request,
            model_info=model_info,
            cut_config=cut_config,
            cut_configs=cut_configs,
            resolved_cut_config=resolved_cut_config,
            label_targets=label_targets,
            label_coordinates=label_coordinates,
            dataset_json_path=dataset_json_path,
            image=image,
            seg_path=result['seg_for_reconstruction_path'],
            raw_for_reconstruction_path=result['raw_for_reconstruction_path'],
            seg_for_reconstruction_path=result['seg_for_reconstruction_path'],
            correction_applied=result['correction_applied'],
            correction_angle_degrees=result['correction_angle_degrees'],
            correction_scope_bbox=correction_scope_bbox,
            correction_params=result['correction_params'],
            spine_geometry=result['spine_geometry'],
            corrected_raw_path=result['corrected_raw_path'],
            corrected_seg_path=result['corrected_seg_path'],
            candidate_raw_path=result['candidate_raw_path'],
            candidate_seg_path=result['candidate_seg_path'],
            debug_bbox_overlay_path=result['debug_bbox_overlay_path'],
            debug_target_mask_overlay_path=result['debug_target_mask_overlay_path'],
            debug_label_mask_overlay_path=result['debug_label_mask_overlay_path'],
            debug_vertebral_body_mask_overlay_path=result['debug_vertebral_body_mask_overlay_path'],
            debug_vertebral_body_prism_mask_overlay_path=result[
                'debug_vertebral_body_prism_mask_overlay_path'
            ],
            debug_xy_correction_crosshair_overlay_path=(
                result['debug_xy_correction_crosshair_overlay_path']
            ),
            is_skewed=result['is_skewed'],
            skew_params=result['skew_params'],
            anatomical_geometry=anatomical_geometry,
            spine_cpr_config=result['spine_cpr_config'],
        )
        
        context.set_stage_result(LocalizationStageResult(payload=result))
        
        DebugLog("  ✅ 阶段三完成")
        DebugLog("=" * 60)

        return context

    @staticmethod
    def _build_anatomical_geometry(
        body_part: str,
        target_mask: np.ndarray,
        image: sitk.Image,
        label_masks: Dict[str, np.ndarray] | None = None,
        start_label: str | None = None,
        end_label: str | None = None,
    ) -> tuple[AnatomicalGeometryResult | None, Dict[str, Any]]:
        """只为已经适配的部位计算统一解剖坐标，其余部位保持旧行为。"""

        if body_part == "肺部":
            return estimate_lung_frame_with_diagnostics(target_mask, image)
        if body_part in {"颈椎", "胸椎"}:
            if not label_masks or not start_label or not end_label:
                raise ValueError(f"{body_part} 解剖坐标缺少首尾椎体标签信息")
            return estimate_spine_frame_with_diagnostics(
                label_masks=label_masks,
                start_label=start_label,
                end_label=end_label,
                image=image,
            )
        if body_part == "髋骨":
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

    def _collect_label_names(
        self,
        resolved_cut_config: Dict[str, Any],
        label_map: Dict[str, list[int]] = None
    ) -> list[str]:
        """
        收集本次定位需要的 label 名称。

        普通目标仍然收集 start/end。脊柱整切的 start/end 是范围边界，
        不是只取两个端点；比如 C1 到 C7 会展开为 C1-C7 全部椎体。
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
        保持原来的端点语义，适配 skull、lung 这类整体目标。
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
        也可能反过来。这里先按 body_part + cut_mode 解析出的 label 范围
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
    ) -> Dict[str, Any]:
        """
        根据目标类型选择横状位角度检测方法。

        脊柱类目标优先用单个椎体的横状位主轴做稳健聚合；头骨、肺部等
        非脊柱目标没有天然的椎体顺序，候选入口会用更高阈值约束整体
        mask PCA，避免小角度形态偏差生成误导性候选。

        默认流程只把这里的角度写入候选诊断和候选 NIfTI，不把它作为
        阶段四/五重建输入。
        """
        if self._is_vertebrae_targets(label_targets):
            vertebrae_info = self._detect_vertebrae_axial_skew(
                seg_array,
                label_targets,
                label_regions=label_regions,
            )
            if vertebrae_info.get('enough_points'):
                return vertebrae_info

            fallback_mask = self._build_target_mask(seg_array, label_targets)
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

        这里故意不计算候选角度，避免把旧 PCA 角度继续写入结果后被误解为
        可执行的回正角。阶段三当前只承诺输出定位坐标和目标 bbox。
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
                '回正逻辑待实现，阶段三当前只保留定位坐标和目标 bbox'
            ),
        }

    def _build_xy_correction_candidate(
        self,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        body_part: str | None = None,
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
    ) -> Dict[str, Any]:
        """估计横状位 XY 回正候选角，但不执行主流程回正。

        脊柱目标使用逐椎体角度聚合；非脊柱目标保留整体 PCA，但必须超过
        更高阈值才生成候选，避免正位头骨等被 3-4 度形态偏差误触发。
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
                'body_part': body_part,
                'reason': reason,
            }
        skew_info = self._detect_target_axial_skew(
            seg_array,
            label_targets,
            label_regions=label_regions,
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

        for label_name in label_targets:
            if self._vertebra_key(label_name) == "C1":
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

        mean_raw_angle = self._weighted_axis_angle_mean([
            (item['raw_pca_angle_degrees'], item['point_count'])
            for item in label_angles
        ])
        deviation_angle = self._deviation_from_nearest_axial_axis(mean_raw_angle)
        abs_angle = abs(deviation_angle)
        return {
            **base_info,
            'angle_spread_degrees': float(angle_spread),
            'median_label_angle_degrees': median_angle,
            'raw_pca_angle_degrees': float(mean_raw_angle),
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
                '回正逻辑待实现，阶段三当前只输出定位坐标，不改写 raw/seg'
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

        阶段四/五只消费 `raw_for_reconstruction_path`、
        `seg_for_reconstruction_path` 以及阶段三几何结果。这里集中完成路径
        提升，后续 CPR、椎间切和重建不需要各自再判断候选文件。
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
        corrected_raw = self._rotate_image_xy(
            image=raw_image,
            rotation_degrees=angle_degrees,
            interpolator=sitk.sitkLinear,
            default_value=self._image_min_value(raw_image),
            center_index_xyz=center_index_xyz,
        )
        corrected_seg = self._rotate_image_xy(
            image=seg_image,
            rotation_degrees=angle_degrees,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
            center_index_xyz=center_index_xyz,
        )
        corrected_seg = sitk.Cast(corrected_seg, seg_image.GetPixelID())

        sitk.WriteImage(corrected_raw, str(candidate_raw_path))
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
    def _draw_xy_line(
        overlay: np.ndarray,
        z_index: int,
        center_yx: tuple[float, float],
        angle_degrees: float,
        half_length: float,
        label_id: int,
    ) -> None:
        """在单个横状位层面画一条指定角度的诊断线。"""

        radians = math.radians(angle_degrees)
        center_y, center_x = center_yx
        sample_count = int(max(2.0, half_length * 2.0))
        for distance in np.linspace(-half_length, half_length, sample_count):
            x_index = int(round(center_x + distance * math.cos(radians)))
            y_index = int(round(center_y + distance * math.sin(radians)))
            if (
                0 <= z_index < overlay.shape[0]
                and 0 <= y_index < overlay.shape[1]
                and 0 <= x_index < overlay.shape[2]
            ):
                overlay[z_index, y_index, x_index] = label_id

    def _write_xy_correction_crosshair_overlay(
        self,
        reference_image: sitk.Image,
        output_path: Path,
        correction_center_bbox: VoxelBBox,
        candidate: Dict[str, Any],
    ) -> str:
        """在原始网格上写出原始 x/y 轴和候选目标轴。"""

        overlay = np.zeros(tuple(reversed(reference_image.GetSize())), dtype=np.uint8)
        z_index = int(round(
            (correction_center_bbox.z_min + correction_center_bbox.z_max) / 2.0
        ))
        center_y = (
            correction_center_bbox.y_min + correction_center_bbox.y_max
        ) / 2.0
        center_x = (
            correction_center_bbox.x_min + correction_center_bbox.x_max
        ) / 2.0
        half_length = max(
            8.0,
            min(
                float(correction_center_bbox.x_max - correction_center_bbox.x_min),
                float(correction_center_bbox.y_max - correction_center_bbox.y_min),
            ) / 2.0,
        )
        self._draw_xy_line(overlay, z_index, (center_y, center_x), 0.0, half_length, 1)
        self._draw_xy_line(overlay, z_index, (center_y, center_x), 90.0, half_length, 2)
        self._draw_xy_line(
            overlay,
            z_index,
            (center_y, center_x),
            float(candidate.get('angle_degrees', 0.0)),
            half_length,
            3,
        )

        overlay_image = sitk.GetImageFromArray(overlay)
        overlay_image.CopyInformation(reference_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

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

        transform = sitk.Euler3DTransform()
        transform.SetCenter(center_point)
        transform.SetRotation(0.0, 0.0, math.radians(rotation_degrees))

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

        优先使用本次定位目标的 bbox 中心；没有 bbox 时退回整图中心，兼容
        单元测试或外部直接调用 helper 的旧方式。
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
    def _write_debug_bbox_overlay(
        reference_image: sitk.Image,
        label_coordinates: Dict[str, LabelCoordinate],
        correction_scope_bbox: VoxelBBox,
        output_path: Path,
        include_label_bboxes: bool = False,
    ) -> str:
        """
        写出研发期 bbox 可视化 overlay，方便在 ITK-SNAP 中直接看定位框。

        这个文件只用于人工验证定位，不参与后续 pipeline。overlay 与
        reference_image 使用同一套 size/spacing/origin/direction。数值约定：
        1 表示本次目标整体 bbox。默认不画每个 label 的 bbox，避免 ITK-SNAP
        里一屏叠太多框；如果临时需要细看单椎体范围，可把
        include_label_bboxes 打开，单个 label 的 bbox 会使用该 label 的第一个
        label id。因为只是研发辅助输出，后续稳定后删除本方法和调用点即可。
        """
        reference_array = sitk.GetArrayFromImage(reference_image)
        overlay_array = np.zeros(reference_array.shape, dtype=np.uint16)
        output_path = Path(output_path)

        def paint_bbox_edges(bbox: VoxelBBox, value: int) -> None:
            """将指定体素包围盒的六个边界面写入调试 overlay。"""

            z_size, y_size, x_size = overlay_array.shape
            z_min = max(0, min(int(bbox.z_min), z_size - 1))
            z_max = max(0, min(int(bbox.z_max), z_size - 1))
            y_min = max(0, min(int(bbox.y_min), y_size - 1))
            y_max = max(0, min(int(bbox.y_max), y_size - 1))
            x_min = max(0, min(int(bbox.x_min), x_size - 1))
            x_max = max(0, min(int(bbox.x_max), x_size - 1))
            if z_min > z_max or y_min > y_max or x_min > x_max:
                return

            # 只画六个面的一层边框，不填满盒子。这样在 ITK-SNAP 里能看到
            # 定位范围，但不会遮住内部真实分割。
            overlay_array[z_min, y_min:y_max + 1, x_min:x_max + 1] = value
            overlay_array[z_max, y_min:y_max + 1, x_min:x_max + 1] = value
            overlay_array[z_min:z_max + 1, y_min, x_min:x_max + 1] = value
            overlay_array[z_min:z_max + 1, y_max, x_min:x_max + 1] = value
            overlay_array[z_min:z_max + 1, y_min:y_max + 1, x_min] = value
            overlay_array[z_min:z_max + 1, y_min:y_max + 1, x_max] = value

        paint_bbox_edges(correction_scope_bbox, 1)
        if include_label_bboxes:
            for coord in label_coordinates.values():
                if not coord.found or not coord.bbox or not coord.label_ids:
                    continue
                paint_bbox_edges(coord.bbox, int(coord.label_ids[0]))

        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    @staticmethod
    def _write_debug_target_mask_overlay(
        reference_image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        output_path: Path,
    ) -> str:
        """
        写出研发期真实目标 mask overlay，方便在 ITK-SNAP 中核对定位是否偏。

        bbox overlay 画的是轴对齐外接盒，在矢状位/冠状位上会被少量极值体素
        或斜着的目标形态撑大。这个 overlay 只保留本次请求关心的 label 体素，
        数值统一写成 1，用来回答“目标本身在哪里”，不表达外接盒范围。
        """
        target_mask = LocalizationStage._build_target_mask(seg_array, label_targets)
        overlay_array = target_mask.astype(np.uint8)
        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

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

        for target in label_targets.values():
            vertebra_key = LocalizationStage._vertebra_key(target.label_name)
            if vertebra_key == "C1":
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
    ) -> str:
        """写出按主体四边形和左右边界填满的三维四棱柱 Mask。"""

        overlay_array = np.zeros(seg_array.shape, dtype=np.uint16)
        _, prism_regions = (
            LocalizationStage._build_spine_body_prism_regions(
                seg_array,
                label_targets,
                reference_image,
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
        for label_name, target in label_targets.items():
            vertebra_key = cls._vertebra_key(label_name)
            if vertebra_key is None or vertebra_key == "C1":
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
            if vertebra_key == "S1":
                body_mask = cls._s1_body_candidate_mask(
                    local_seg_array,
                    target,
                )
            else:
                body_mask = cls._vertebral_body_candidate_mask(
                    local_seg_array,
                    target,
                )
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
            cls._resolve_non_overlapping_prism_regions(prism_regions),
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
        region_slices = tuple(
            slice(
                int(start - mask_start),
                int(start - mask_start) + int(size),
            )
            for start, mask_start, size in zip(
                minimum,
                mask_minimum,
                local_prism_mask.shape,
            )
        )
        clipped = local_prism_mask & mask[region_slices]
        if not np.any(clipped):
            return None
        return minimum, clipped

    @classmethod
    def _resolve_non_overlapping_prism_regions(
        cls,
        prism_regions: dict[
            str,
            tuple[np.ndarray, np.ndarray] | None,
        ],
    ) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
        """只比较相交局部包围盒；碰撞涉及的棱柱全部失效。"""

        resolved_regions = dict(prism_regions)
        valid_labels = [
            label_name
            for label_name, prism_region in prism_regions.items()
            if prism_region is not None
        ]
        collided_labels: set[str] = set()
        for left_index, left_label in enumerate(valid_labels):
            left_minimum, left_mask = prism_regions[left_label]
            left_maximum = left_minimum + np.asarray(left_mask.shape)
            for right_label in valid_labels[left_index + 1:]:
                right_minimum, right_mask = prism_regions[right_label]
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
                if np.any(
                    left_mask[left_slices] & right_mask[right_slices]
                ):
                    collided_labels.update((left_label, right_label))
        for label_name in collided_labels:
            resolved_regions[label_name] = None
        return resolved_regions

    @classmethod
    def _build_spine_body_geometry(
        cls,
        body_part: str,
        image: sitk.Image,
        seg_array: np.ndarray,
        label_targets: Dict[str, LabelTarget],
        label_regions: Dict[str, _LabelVoxelRegion] | None = None,
    ) -> Dict[str, Any] | None:
        """
        在阶段三生成脊柱专用过滤标注和标注质心，供阶段四 CPR 使用。

        这里不修改原始 label 的质心和 bbox。完整 label 负责定位及覆盖范围，
        过滤标注负责计算标注质心；颈椎 C1 保留在完整定位中，但不进入 CPR
        标注质心。物理坐标直接由当前最终定位图像转换，保证与阶段四一致。
        """

        if body_part not in {"颈椎", "胸椎", "腰椎"}:
            return None

        sagittal_geometries: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        ] = {}
        body_indices, prism_regions = cls._build_spine_body_prism_regions(
            seg_array,
            label_targets,
            image,
            sagittal_geometries=sagittal_geometries,
            label_regions=label_regions,
        )

        filtered_annotation_centroids: Dict[str, Dict[str, Any]] = {}
        for label_name, target in label_targets.items():
            vertebra_key = cls._vertebra_key(label_name)
            if vertebra_key is None or vertebra_key == "C1":
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

        return {
            "filtered_annotation": {
                "method": "vertebral_body_candidate_v1",
                "source": "full_nnunet_label",
                "target": "vertebral_body",
            },
            "stage": "stage3_localization",
            "filtered_annotation_centroids": filtered_annotation_centroids,
            "excluded_from_annotation_centroids": ["vertebrae_C1"],
            "full_label_geometry": "coords",
            "full_label_bbox": "coords.*.bbox",
        }

    @staticmethod
    def _bbox_from_indices(indices_zyx: np.ndarray) -> dict[str, int]:
        """把 zyx 体素索引转成统一 bbox 字典。"""

        minimum = indices_zyx.min(axis=0)
        maximum = indices_zyx.max(axis=0)
        return {
            "z_min": int(minimum[0]),
            "z_max": int(maximum[0]),
            "y_min": int(minimum[1]),
            "y_max": int(maximum[1]),
            "x_min": int(minimum[2]),
            "x_max": int(maximum[2]),
        }

    @staticmethod
    def _bbox_from_region(
        minimum_zyx: np.ndarray,
        local_mask: np.ndarray,
    ) -> dict[str, int] | None:
        """把局部 mask 区域转成全局 bbox 字典。"""

        local_indices = np.argwhere(local_mask)
        if len(local_indices) == 0:
            return None
        return LocalizationStage._bbox_from_indices(local_indices + minimum_zyx)

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
                )
            )
            if np.count_nonzero(retained_projection_mask) >= 4:
                physical_points = physical_points[retained_projection_mask]
        center = physical_points.mean(axis=0)
        sagittal_axes = direction[:, [1, 2]]
        sagittal_points = (physical_points - center) @ sagittal_axes
        unique_points = np.unique(np.round(sagittal_points, 3), axis=0)
        hull = LocalizationStage._convex_hull_2d(unique_points)
        quad = LocalizationStage._simplify_hull_to_quadrilateral(hull)
        if quad is None:
            return None
        return center, sagittal_axes, quad

    @staticmethod
    def _retained_sagittal_projection_mask_by_lr_support(
        indices_zyx: np.ndarray,
        minimum_support_fraction: float = 0.35,
    ) -> np.ndarray:
        """只让 LR 支撑足够的 Y/Z 投影点参与矢状四边形拟合。"""

        if len(indices_zyx) == 0:
            return np.zeros(0, dtype=bool)

        projection_points, inverse, support_counts = np.unique(
            indices_zyx[:, [0, 1]],
            axis=0,
            return_inverse=True,
            return_counts=True,
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

        selected_projection_points = np.zeros(
            len(projection_points),
            dtype=bool,
        )
        selected_projection_points[
            supported_indices[selected_members]
        ] = True
        retained_mask = selected_projection_points[inverse]
        return retained_mask

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
    def _points_inside_convex_polygon(
        points: np.ndarray,
        polygon: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        """判断二维点是否位于有序凸多边形内部，并包含边界。"""

        edges = np.roll(polygon, -1, axis=0) - polygon
        deltas = points[:, None, :] - polygon[None, :, :]
        crosses = (
            edges[None, :, 0] * deltas[:, :, 1]
            - edges[None, :, 1] * deltas[:, :, 0]
        )
        edge_lengths = np.linalg.norm(edges, axis=1)
        if np.any(edge_lengths <= 1e-12):
            return np.zeros(len(points), dtype=bool)
        signed_distances = crosses / edge_lengths[None, :]
        if LocalizationStage._signed_area_2d(polygon) > 0.0:
            return np.all(signed_distances >= -tolerance, axis=1)
        return np.all(signed_distances <= tolerance, axis=1)

    @staticmethod
    def _vertebral_body_prism_mask(
        body_mask: np.ndarray,
        image: sitk.Image,
    ) -> np.ndarray | None:
        """兼容接口：把局部棱柱还原为与输入同尺寸的布尔 mask。"""

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
        """在主体包围盒内栅格化棱柱，返回最小 zyx 索引和局部 mask。"""

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

        minimum = body_indices.min(axis=0)
        maximum = body_indices.max(axis=0)
        grids = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1),
            np.arange(minimum[2], maximum[2] + 1),
            indexing="ij",
        )
        candidates_zyx = np.stack(grids, axis=-1).reshape(-1, 3)
        candidates_xyz = candidates_zyx[:, [2, 1, 0]].astype(float)
        candidate_points = origin + (candidates_xyz * spacing) @ direction.T

        body_xyz = body_indices[:, [2, 1, 0]].astype(float)
        body_points = origin + (body_xyz * spacing) @ direction.T
        left_right_axis = direction[:, 0]
        body_lr = (body_points - center) @ left_right_axis
        candidate_relative = candidate_points - center
        candidate_lr = candidate_relative @ left_right_axis
        candidate_sagittal = candidate_relative @ sagittal_axes
        tolerance = max(
            float(min(spacing[1], spacing[2])) * 1e-3,
            1e-3,
        )
        selected = (
            (candidate_lr >= float(np.min(body_lr)) - tolerance)
            & (candidate_lr <= float(np.max(body_lr)) + tolerance)
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
        return minimum, local_mask

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
        """返回二维点集的凸包顶点，按轮廓顺序排列。"""

        if len(points) < 4:
            return points
        ordered = sorted((float(point[0]), float(point[1])) for point in points)
        if len(ordered) < 4:
            return np.asarray(ordered, dtype=float)

        def cross(
            origin: tuple[float, float],
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> float:
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        lower: list[tuple[float, float]] = []
        for point in ordered:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper: list[tuple[float, float]] = []
        for point in reversed(ordered):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        hull = lower[:-1] + upper[:-1]
        return np.asarray(hull, dtype=float)

    @staticmethod
    def _simplify_hull_to_quadrilateral(hull: np.ndarray) -> np.ndarray | None:
        """把轮廓凸包压缩成四个主角点。"""

        if len(hull) < 4:
            return None
        vertices = [np.asarray(point, dtype=float) for point in hull]
        while len(vertices) > 4:
            areas = []
            for index, point in enumerate(vertices):
                previous = vertices[index - 1]
                next_point = vertices[(index + 1) % len(vertices)]
                previous_delta = previous - point
                next_delta = next_point - point
                area = abs(
                    previous_delta[0] * next_delta[1]
                    - previous_delta[1] * next_delta[0]
                ) / 2.0
                areas.append(float(area))
            remove_index = int(np.argmin(areas))
            vertices.pop(remove_index)
        quad = np.asarray(vertices, dtype=float)
        if abs(LocalizationStage._signed_area_2d(quad)) <= 1e-6:
            return None
        return quad

    @staticmethod
    def _signed_area_2d(points: np.ndarray) -> float:
        """计算二维多边形有向面积。"""

        if len(points) < 3:
            return 0.0
        x_values = points[:, 0]
        y_values = points[:, 1]
        return float(
            0.5 * np.sum(
                x_values * np.roll(y_values, -1)
                - y_values * np.roll(x_values, -1)
            )
        )

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
    ) -> np.ndarray | None:
        """从完整椎体 label 中估计前方宽主体的体素 mask。"""

        mask = np.isin(seg_array, target.label_ids)
        if not np.any(mask):
            return None
        mask = LocalizationStage._discard_small_3d_components(mask)
        indices = np.argwhere(mask)
        if indices.size == 0:
            return None

        center_x = float(np.median(indices[:, 2]))
        anterior_y_limit = LocalizationStage._estimate_anterior_y_limit(mask)
        body_mask = np.zeros_like(mask, dtype=bool)
        for z_index in np.unique(indices[:, 0]):
            slice_mask = mask[int(z_index), :, :]
            y_indices = np.flatnonzero(slice_mask.any(axis=1))
            if (
                anterior_y_limit is not None
                and y_indices.size > 0
                and float(y_indices.min()) > anterior_y_limit
            ):
                continue
            row_widths = LocalizationStage._slice_row_widths(slice_mask)
            row_run_counts = LocalizationStage._slice_row_run_counts(slice_mask)
            if not row_widths:
                continue
            row_run = LocalizationStage._first_anterior_body_row_run(
                row_widths,
                row_run_counts,
            )
            if not row_run:
                continue
            y_min = min(row_run)
            y_max = max(row_run)
            stable_width = LocalizationStage._stable_central_run_width(
                slice_mask,
                row_run,
                center_x,
            )
            for y_index in range(y_min, y_max + 1):
                x_run = LocalizationStage._central_x_run(
                    slice_mask[int(y_index), :],
                    center_x,
                )
                if not x_run:
                    continue
                x_run = LocalizationStage._clip_run_to_width(
                    x_run,
                    center_x,
                    stable_width,
                )
                x_min = min(x_run)
                x_max = max(x_run)
                body_mask[int(z_index), y_index, x_min:x_max + 1] = slice_mask[
                    int(y_index),
                    x_min:x_max + 1,
                ]

        if not np.any(body_mask):
            return body_mask
        body_mask = LocalizationStage._retain_largest_3d_component(body_mask)
        return LocalizationStage._trim_lr_edge_columns_by_si_height(body_mask)

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
    def _estimate_anterior_y_limit(mask: np.ndarray) -> float | None:
        """从各轴向层最前端位置的明显分簇估计主体层的 AP 上限。"""

        slice_y_minima: list[int] = []
        for z_index in np.unique(np.argwhere(mask)[:, 0]):
            y_indices = np.flatnonzero(mask[int(z_index), :, :].any(axis=1))
            if y_indices.size > 0:
                slice_y_minima.append(int(y_indices.min()))
        if len(slice_y_minima) < 4:
            return None

        sorted_minima = np.sort(np.asarray(slice_y_minima, dtype=int))
        gaps = np.diff(sorted_minima)
        if gaps.size == 0:
            return None
        split_index = int(np.argmax(gaps))
        if int(gaps[split_index]) < 4:
            return None
        return float(
            (sorted_minima[split_index] + sorted_minima[split_index + 1]) / 2.0
        )

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

    @staticmethod
    def _write_localization_result(
        output_path: Path,
        request,
        model_info: Dict[str, Any],
        cut_config: Dict[str, Any],
        cut_configs: Dict[str, Dict[str, Any]],
        resolved_cut_config: Dict[str, Any],
        label_targets: Dict[str, LabelTarget],
        label_coordinates: Dict[str, LabelCoordinate],
        dataset_json_path: Path,
        image: sitk.Image,
        seg_path: str,
        raw_for_reconstruction_path: str,
        seg_for_reconstruction_path: str,
        correction_applied: bool,
        correction_angle_degrees: float,
        correction_scope_bbox: VoxelBBox,
        correction_params: Dict[str, Any],
        corrected_raw_path: str,
        corrected_seg_path: str,
        candidate_raw_path: str,
        candidate_seg_path: str,
        debug_bbox_overlay_path: str,
        debug_target_mask_overlay_path: str,
        debug_label_mask_overlay_path: str,
        debug_vertebral_body_mask_overlay_path: str,
        debug_vertebral_body_prism_mask_overlay_path: str,
        debug_xy_correction_crosshair_overlay_path: str,
        spine_geometry: Dict[str, Any] | None,
        is_skewed: bool,
        skew_params: Dict[str, Any],
        anatomical_geometry: AnatomicalGeometryResult | None = None,
        spine_cpr_config: Dict[str, Any] | None = None,
    ) -> None:
        """写出定位结果，供后续映射、MPR 和调试使用。"""
        localization_result = LocalizationResult(
            body_part=request.body_part.value,
            cut_mode=LocalizationStage._resolve_request_cut_mode(request),
            seg_path=str(seg_path),
            model=model_info,
            dataset_json_path=str(dataset_json_path),
            image=NiftiImageInfo(
                size_xyz=list(image.GetSize()),
                spacing_xyz=list(image.GetSpacing()),
                origin_xyz=list(image.GetOrigin()),
                direction=list(image.GetDirection()),
            ),
            cut_config=cut_config,
            cut_configs=cut_configs,
            resolved_cut_config=resolved_cut_config,
            label_targets=label_targets,
            coords=label_coordinates,
            raw_for_reconstruction_path=raw_for_reconstruction_path,
            seg_for_reconstruction_path=seg_for_reconstruction_path,
            correction_applied=correction_applied,
            correction_angle_degrees=correction_angle_degrees,
            correction_scope_bbox=correction_scope_bbox,
            correction_params=correction_params,
            corrected_raw_path=corrected_raw_path,
            corrected_seg_path=corrected_seg_path,
            candidate_raw_path=candidate_raw_path,
            candidate_seg_path=candidate_seg_path,
            debug_bbox_overlay_path=debug_bbox_overlay_path,
            debug_target_mask_overlay_path=debug_target_mask_overlay_path,
            debug_label_mask_overlay_path=debug_label_mask_overlay_path,
            debug_vertebral_body_mask_overlay_path=debug_vertebral_body_mask_overlay_path,
            debug_vertebral_body_prism_mask_overlay_path=(
                debug_vertebral_body_prism_mask_overlay_path
            ),
            debug_xy_correction_crosshair_overlay_path=(
                debug_xy_correction_crosshair_overlay_path
            ),
            spine_geometry=spine_geometry,
            spine_cpr_config=spine_cpr_config or {},
            is_skewed=is_skewed,
            skew_params=skew_params,
            anatomical_geometry=anatomical_geometry,
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(localization_result.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _label_targets_to_dict(label_targets: Dict[str, LabelTarget]) -> Dict[str, Dict[str, Any]]:
        """把按名称索引的标签目标实体转换为可写 JSON 的字典。"""

        return {
            name: target.to_dict()
            for name, target in label_targets.items()
        }

    @staticmethod
    def _coordinates_to_dict(label_coordinates: Dict[str, LabelCoordinate]) -> Dict[str, Dict[str, Any]]:
        """把按名称索引的标签定位结果转换为可写 JSON 的字典。"""

        return {
            name: coord.to_dict()
            for name, coord in label_coordinates.items()
        }

    @staticmethod
    def _resolve_request_cut_mode(request) -> str:
        """根据请求部位和全部状位推导阶段三需要加载的定位范围模式。"""

        from src.planning.cut_policy import localization_cut_mode

        return localization_cut_mode(
            request.body_part,
            (config.plane for config in request.plane_configs),
        )

    def _resolve_cut_config(self, request, cut_mode: str | None = None) -> Dict[str, Any]:
        """
        根据请求部位和自动推导的定位范围模式查找配置。

        具体范围统一从 localization_config.json 读取。
        whole_range 使用 start/end；interspace 的每个 interval 也使用
        start/end，避免同一配置里混用 labels、数组等多种形态。
        """
        cut_mode = cut_mode or self._resolve_request_cut_mode(request)
        body_part = getattr(request.body_part, 'value', request.body_part)
        cut_configs = self._resolve_cut_configs(request)
        cut_config = cut_configs.get(cut_mode)
        if not cut_config:
            raise ValueError(
                f"未找到切割方式配置: body_part={body_part}, cut_mode={cut_mode}"
            )
        return cut_config

    def _resolve_spine_cpr_config(self, body_part: str) -> Dict[str, Any]:
        """从当前脊柱部位配置中读取 CPR 配置，非脊柱部位返回空配置。"""

        target_configs = self._localization_config.get('targets', {})
        target_config = (
            target_configs.get(body_part, {})
            if isinstance(target_configs, dict)
            else {}
        )
        if not isinstance(target_config, dict):
            return {}
        config = target_config.get('spine_cpr')
        if not isinstance(config, dict):
            return {}
        full_label = config.get('full_label')
        return {'full_label': full_label if isinstance(full_label, bool) else True}

    def _resolve_cut_configs(self, request) -> Dict[str, Dict[str, Any]]:
        """返回当前部位的全部切法配置，并统一校验其中的范围扩展设置。

        阶段三选择标签时仍只使用当前定位模式；完整映射专门提供给第四步，
        让一次混合状位请求中的整切和区间切都能读取各自配置。
        """

        body_part = getattr(request.body_part, 'value', request.body_part)
        target_configs = self._localization_config.get('targets', {})
        target_config = target_configs.get(body_part)
        if not target_config:
            raise ValueError(f"未找到部位定位配置: {body_part}")

        cut_configs: Dict[str, Dict[str, Any]] = {}
        for configured_mode in ('whole_range', 'interspace'):
            cut_config = target_config.get(configured_mode)
            if not cut_config:
                continue
            self._validate_cut_config(configured_mode, cut_config)
            cut_configs[configured_mode] = cut_config
        return cut_configs

    def _resolve_label_aliases(self, cut_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        把业务短名解析成模型/数据集中的标准 label 名。

        例如配置中可以写 L1、C7、skull；这里会根据 label_aliases
        转成 vertebrae_L1、vertebrae_C7、skull。后续再用这些标准名
        去 dataset.json 中查 label id。
        """
        if 'intervals' in cut_config:
            return {
                'intervals': [
                    {
                        'start': interval['start'],
                        'end': interval['end'],
                        'start_labels': self._label_alias(interval['start']),
                        'end_labels': self._label_alias(interval['end'])
                    }
                    for interval in cut_config['intervals']
                ]
            }

        return {
            'start': cut_config['start'],
            'end': cut_config['end'],
            'start_labels': self._label_alias(cut_config['start']),
            'end_labels': self._label_alias(cut_config['end'])
        }

    def _label_alias(self, name: str) -> list[str]:
        """
        获取一个定位名称对应的标准 label 名列表。

        alias 可以是字符串，也可以是字符串数组。肺部这类由左右肺共同
        构成的整体范围，可以用一个业务名 lung 映射到多个 label。
        """
        aliases = self._localization_config.get('label_aliases', {})
        value = aliases.get(name, name)
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _validate_cut_config(cut_mode: str, cut_config: Dict[str, Any]) -> None:
        """
        校验切割配置结构。

        配置统一使用 start/end 表达范围：
            whole_range: {"start": "L1", "end": "S1"}
            interspace: {"intervals": [{"start": "L1", "end": "L2"}]}
        """
        LocalizationStage._validate_slice_range_extension(cut_config)

        if cut_mode == 'whole_range':
            if not cut_config.get('start') or not cut_config.get('end'):
                raise ValueError(f"whole_range 配置必须包含 start/end: {cut_config}")
            return

        if cut_mode == 'interspace':
            intervals = cut_config.get('intervals')
            if not intervals:
                raise ValueError(f"interspace 配置必须包含 intervals: {cut_config}")
            for interval in intervals:
                if not interval.get('start') or not interval.get('end'):
                    raise ValueError(f"interspace interval 必须包含 start/end: {interval}")
            return

        raise ValueError(f"未知切割方式: {cut_mode}")

    @staticmethod
    def _validate_slice_range_extension(cut_config: Dict[str, Any]) -> None:
        """校验按状位配置的切片范围扩展距离。

        JSON 中 `start`、`end` 的单位固定为毫米。字段可以省略，省略端
        在第四步按 0mm 处理；显式值必须是非负有限数字，布尔值不视为
        数字，避免 JSON 的 true/false 被 Python 当成 1/0。
        """

        extensions = cut_config.get('slice_range_extension')
        if extensions is None:
            return
        if not isinstance(extensions, dict):
            raise ValueError("slice_range_extension 必须是对象")

        supported_planes = {plane.value for plane in PlaneTypeEnum}
        for plane_name, endpoint_config in extensions.items():
            if plane_name not in supported_planes:
                raise ValueError(
                    f"slice_range_extension 不支持的状位: {plane_name}"
                )
            if not isinstance(endpoint_config, dict):
                raise ValueError(
                    f"slice_range_extension.{plane_name} 必须是对象"
                )
            unknown_fields = set(endpoint_config) - {'start', 'end'}
            if unknown_fields:
                raise ValueError(
                    f"slice_range_extension.{plane_name} 仅支持 start/end"
                )
            for endpoint_name, value in endpoint_config.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"slice_range_extension.{plane_name}.{endpoint_name} "
                        "必须是有限数字"
                    )
                if float(value) < 0:
                    raise ValueError(
                        f"slice_range_extension.{plane_name}.{endpoint_name} "
                        "不能小于 0"
                    )
