"""第五阶段：按部位生成解剖坐标系和专用几何。"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import (
    AnatomicalAnalysisStageResult,
    LabelTarget,
)
from src.utils import DebugLog, is_debug_log_file_enabled

from .stage_localization import (
    LocalizationOperations,
    _LOCALIZATION_WORKSPACE_KEY,
)


class AnatomicalAnalysisStage(LocalizationOperations):
    """计算通用解剖坐标，并为脊柱生成四棱柱等专用几何。"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, stage_name="anatomical_analysis")

    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """消费纯定位结果，不在本阶段决定切片数量或方向。"""

        localization_result = context.get_result("localization")
        if not localization_result:
            raise ValueError("前置阶段 'localization' 结果不存在")
        correction_result = context.get_result("spatial_correction") or {}

        DebugLog("=" * 60)
        DebugLog("【第五阶段】解剖分析 - 坐标系与专用几何")

        workspace = context.metadata.get(_LOCALIZATION_WORKSPACE_KEY)
        working_seg_path = localization_result["seg_for_reconstruction_path"]
        if not workspace or workspace.get("seg_path") != working_seg_path:
            label_targets = {
                name: LabelTarget(
                    label_name=value["label_name"],
                    label_ids=[int(item) for item in value["label_ids"]],
                )
                for name, value in localization_result["label_targets"].items()
            }
            image = sitk.ReadImage(str(working_seg_path))
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
            workspace = {
                "seg_path": working_seg_path,
                "image": image,
                "seg_array": seg_array,
                "label_regions": label_regions,
                "label_targets": label_targets,
                "label_coordinates": label_coordinates,
                "target_mask": target_mask,
                "correction_scope_bbox": self._build_target_bbox(target_mask),
                "dataset_json_path": Path(
                    localization_result["dataset_json_path"]
                ),
                "model_info": correction_result.get("model") or {},
                "cut_mode": localization_result["cut_mode"],
                "cut_config": localization_result["cut_config"],
                "cut_configs": localization_result["cut_configs"],
                "resolved_cut_config": localization_result[
                    "resolved_cut_config"
                ],
            }
            context.metadata[_LOCALIZATION_WORKSPACE_KEY] = workspace

        image = workspace["image"]
        seg_array = workspace["seg_array"]
        label_regions = workspace["label_regions"]
        label_targets = workspace["label_targets"]
        label_coordinates = workspace["label_coordinates"]
        target_mask = workspace["target_mask"]
        correction_scope_bbox = workspace["correction_scope_bbox"]

        body_part = context.request.body_part.value
        label_masks: Dict[str, np.ndarray] = {}
        if body_part in {"颈椎", "胸椎", "髋骨"}:
            for label_name, region in label_regions.items():
                label_mask = np.zeros(seg_array.shape, dtype=bool)
                if region.indices_zyx.size:
                    label_mask[tuple(region.indices_zyx.T)] = True
                label_masks[label_name] = label_mask

        resolved_cut_config = localization_result["resolved_cut_config"]
        start_labels = resolved_cut_config.get("start_labels") or []
        end_labels = resolved_cut_config.get("end_labels") or []
        anatomical_geometry, anatomical_diagnostics = (
            self._build_anatomical_geometry(
                body_part=body_part,
                target_mask=target_mask,
                image=image,
                label_masks=label_masks,
                start_label=start_labels[0] if start_labels else None,
                end_label=end_labels[0] if end_labels else None,
            )
        )
        spine_geometry = self._build_spine_body_geometry(
            body_part=body_part,
            image=image,
            seg_array=seg_array,
            label_targets=label_targets,
            label_regions=label_regions,
        )

        debug_label_path = ""
        debug_body_path = ""
        debug_prism_path = ""
        if is_debug_log_file_enabled():
            metadata_dir = Path(context.request.output_dir) / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            label_path = metadata_dir / "标签掩膜.nii.gz"
            self._write_debug_label_mask_overlay(
                reference_image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=label_path,
            )
            debug_label_path = str(label_path)

            body_path = metadata_dir / "椎体主体掩膜.nii.gz"
            self._write_debug_vertebral_body_mask_overlay(
                reference_image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=body_path,
            )
            debug_body_path = str(body_path)

            prism_path = metadata_dir / "椎体四棱柱掩膜.nii.gz"
            self._write_debug_vertebral_body_prism_mask_overlay(
                reference_image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=prism_path,
            )
            debug_prism_path = str(prism_path)

            self._write_localization_result(
                Path(localization_result["localization_result_path"]),
                request=context.request,
                model_info=workspace.get("model_info") or {},
                cut_config=localization_result["cut_config"],
                cut_configs=localization_result["cut_configs"],
                resolved_cut_config=resolved_cut_config,
                label_targets=label_targets,
                label_coordinates=label_coordinates,
                dataset_json_path=Path(localization_result["dataset_json_path"]),
                image=image,
                seg_path=working_seg_path,
                raw_for_reconstruction_path=localization_result[
                    "raw_for_reconstruction_path"
                ],
                seg_for_reconstruction_path=working_seg_path,
                correction_applied=localization_result["correction_applied"],
                correction_angle_degrees=localization_result[
                    "correction_angle_degrees"
                ],
                correction_scope_bbox=correction_scope_bbox,
                correction_params=localization_result["correction_params"],
                corrected_raw_path=localization_result["corrected_raw_path"],
                corrected_seg_path=localization_result["corrected_seg_path"],
                candidate_raw_path=localization_result["candidate_raw_path"],
                candidate_seg_path=localization_result["candidate_seg_path"],
                debug_bbox_overlay_path="",
                debug_target_mask_overlay_path="",
                debug_label_mask_overlay_path=debug_label_path,
                debug_vertebral_body_mask_overlay_path=debug_body_path,
                debug_vertebral_body_prism_mask_overlay_path=debug_prism_path,
                debug_xy_correction_crosshair_overlay_path=localization_result.get(
                    "debug_xy_correction_crosshair_overlay_path",
                    "",
                ),
                spine_geometry=spine_geometry,
                is_skewed=localization_result["is_skewed"],
                skew_params=localization_result["skew_params"],
                anatomical_geometry=anatomical_geometry,
                spine_cpr_config=localization_result["spine_cpr_config"],
            )

        payload = {
            "spine_geometry": spine_geometry,
            "anatomical_geometry_diagnostics": anatomical_diagnostics,
            "debug_label_mask_overlay_path": debug_label_path,
            "debug_vertebral_body_mask_overlay_path": debug_body_path,
            "debug_vertebral_body_prism_mask_overlay_path": debug_prism_path,
            "localization_result_path": localization_result[
                "localization_result_path"
            ],
        }
        if anatomical_geometry is not None:
            payload["anatomical_geometry"] = anatomical_geometry.to_dict()

        context.set_stage_result(AnatomicalAnalysisStageResult(payload=payload))

        DebugLog(
            "  解剖坐标: "
            f"{'applied' if anatomical_geometry is not None else 'skipped'}"
        )
        DebugLog(
            "  脊柱专用几何: "
            f"{'applied' if spine_geometry is not None else 'skipped'}"
        )
        if spine_geometry is not None:
            DebugLog("  矢状投影去重: 一维整数键（稳定路径）")
        DebugLog("  第五阶段完成")
        DebugLog("=" * 60)
        return context
