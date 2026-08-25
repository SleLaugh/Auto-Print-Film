"""第三阶段：为后续定位与重建选定统一图像空间。"""

from pathlib import Path
from typing import Any, Dict

import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import SpatialCorrectionStageResult
from src.utils import DebugLog

from .stage_localization import (
    LocalizationOperations,
    _LOCALIZATION_WORKSPACE_KEY,
)


class SpatialCorrectionStage(LocalizationOperations):
    """估计 XY 校正候选，并唯一决定后续使用的 raw/seg。"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, stage_name="spatial_correction")

    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """从分割结果生成校正候选，并保存可序列化的空间契约。"""

        request = context.request
        segmentation_result = context.get_result("segmentation")
        preprocessing_result = context.get_result("preprocessing") or {}
        if not segmentation_result:
            raise ValueError("前置阶段 'segmentation' 结果不存在")

        seg_path = segmentation_result.get("seg_path")
        raw_path = (
            preprocessing_result.get("nifti_path")
            or segmentation_result.get("input_nifti")
            or str(Path(request.output_dir) / "nifti" / "raw_0000.nii.gz")
        )
        model_info = segmentation_result.get("model") or {}
        if not seg_path:
            raise ValueError("分割结果路径不存在")

        DebugLog("=" * 60)
        DebugLog("【第三阶段】空间校正 - 选择后续工作图像")
        DebugLog(f"  输入原图: {raw_path}")
        DebugLog(f"  输入分割: {seg_path}")
        DebugLog(f"  部位: {request.study_part.value}")

        localization_profile = self._resolve_request_localization_profile(request)
        cut_configs = self._resolve_cut_configs(request)
        slice_rules = self._resolve_slice_rules(request)
        cut_config = cut_configs.get(localization_profile)
        if not cut_config:
            raise ValueError(
                f"未找到定位范围配置: study_part={request.study_part.value}, "
                f"profile={localization_profile}"
            )
        resolved_cut_configs = {
            profile: self._resolve_label_aliases(config)
            for profile, config in cut_configs.items()
        }
        resolved_cut_config = resolved_cut_configs[localization_profile]

        dataset_json_path = self._find_dataset_json(model_info, seg_path=seg_path)
        label_map = self._load_label_map(dataset_json_path)
        label_targets = self._resolve_combined_label_targets(
            resolved_cut_configs,
            label_map,
        )

        image = sitk.ReadImage(str(seg_path))
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

        candidate = self._build_xy_correction_candidate(
            seg_array,
            label_targets,
            study_part=request.study_part.value,
            label_regions=label_regions,
        )
        DebugLog(
            "  横状位 XY 候选角: "
            f"{candidate['angle_degrees']:.3f}°, "
            f"status={candidate['correction_status']}"
        )

        correction_result = self._write_xy_correction_candidate_images(
            raw_path=str(raw_path),
            seg_path=str(seg_path),
            output_dir=Path(request.output_dir) / "nifti",
            candidate=candidate,
            correction_center_bbox=correction_scope_bbox,
        )
        correction_result = self._promote_xy_correction_candidate_if_available(
            correction_result,
        )

        if correction_result["correction_applied"]:
            image = sitk.ReadImage(
                str(correction_result["seg_for_reconstruction_path"])
            )
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
            DebugLog("  已启用校正候选，后续阶段统一使用候选空间")
        else:
            DebugLog("  未启用校正，后续阶段使用原始空间")

        # 快照只在本次进程内复用，不进入 manifest 或阶段 JSON。
        context.metadata[_LOCALIZATION_WORKSPACE_KEY] = {
            "seg_path": correction_result["seg_for_reconstruction_path"],
            "image": image,
            "seg_array": seg_array,
            "label_regions": label_regions,
            "label_targets": label_targets,
            "label_coordinates": label_coordinates,
            "target_mask": target_mask,
            "correction_scope_bbox": correction_scope_bbox,
            "dataset_json_path": dataset_json_path,
            "model_info": model_info,
            "localization_profile": localization_profile,
            "cut_config": cut_config,
            "cut_configs": cut_configs,
            "resolved_cut_config": resolved_cut_config,
            "resolved_cut_configs": resolved_cut_configs,
            "slice_rules": slice_rules,
            "candidate": candidate,
        }

        payload = {
            "raw_input_path": str(raw_path),
            "seg_input_path": str(seg_path),
            "model": model_info,
            "localization_profile": localization_profile,
            "cut_config": cut_config,
            "cut_configs": cut_configs,
            "resolved_cut_config": resolved_cut_config,
            "resolved_cut_configs": resolved_cut_configs,
            "slice_rules": slice_rules,
            "label_targets": self._label_targets_to_dict(label_targets),
            "dataset_json_path": str(dataset_json_path),
            "raw_for_reconstruction_path": correction_result[
                "raw_for_reconstruction_path"
            ],
            "seg_for_reconstruction_path": correction_result[
                "seg_for_reconstruction_path"
            ],
            "correction_applied": correction_result["correction_applied"],
            "correction_angle_degrees": candidate["angle_degrees"],
            "correction_scope_bbox": correction_scope_bbox.to_dict(),
            "correction_params": correction_result["correction_params"],
            "corrected_raw_path": correction_result["corrected_raw_path"],
            "corrected_seg_path": correction_result["corrected_seg_path"],
            "candidate_raw_path": correction_result["candidate_raw_path"],
            "candidate_seg_path": correction_result["candidate_seg_path"],
            "is_skewed": correction_result["correction_applied"],
            "skew_params": correction_result["correction_params"],
        }
        context.set_stage_result(SpatialCorrectionStageResult(payload=payload))

        DebugLog("  第三阶段完成")
        DebugLog("=" * 60)
        return context
