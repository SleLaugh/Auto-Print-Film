"""定位阶段的执行编排。

该 mixin 只负责把既有定位步骤串联起来，医学几何和配置逻辑仍由
LocalizationOperations 提供；保留 LocalizationStage.execute 入口兼容现有调用。
"""

from pathlib import Path

import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import LabelTarget, LocalizationStageResult, NiftiImageInfo
from src.utils import DebugLog, is_debug_file_enabled

_LOCALIZATION_WORKSPACE_KEY = "localization_workspace_v1"


class LocalizationExecutionMixin:
    """定位阶段的执行编排。"""

    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """生成纯标签定位结果，不在本阶段计算解剖几何。"""

        correction_result = context.get_result("spatial_correction")
        if not correction_result:
            raise ValueError("前置阶段 'spatial_correction' 结果不存在")

        DebugLog("=" * 60)
        DebugLog("【第四阶段】标签定位 - 质心与bbox")

        workspace = context.metadata.get(_LOCALIZATION_WORKSPACE_KEY)
        working_seg_path = correction_result["seg_for_reconstruction_path"]
        if not workspace or workspace.get("seg_path") != working_seg_path:
            label_targets = {
                name: LabelTarget(
                    label_name=value["label_name"],
                    label_ids=[int(item) for item in value["label_ids"]],
                )
                for name, value in correction_result["label_targets"].items()
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
            correction_scope_bbox = self._build_target_bbox(target_mask)
            workspace = {
                "seg_path": working_seg_path,
                "image": image,
                "seg_array": seg_array,
                "label_regions": label_regions,
                "label_targets": label_targets,
                "label_coordinates": label_coordinates,
                "target_mask": target_mask,
                "correction_scope_bbox": correction_scope_bbox,
                "dataset_json_path": Path(
                    correction_result["dataset_json_path"]
                ),
                "model_info": correction_result.get("model") or {},
                "localization_profile": correction_result[
                    "localization_profile"
                ],
                "cut_config": correction_result["cut_config"],
                "cut_configs": correction_result["cut_configs"],
                "resolved_cut_config": correction_result[
                    "resolved_cut_config"
                ],
                "resolved_cut_configs": correction_result[
                    "resolved_cut_configs"
                ],
                "slice_rules": correction_result["slice_rules"],
            }
            context.metadata[_LOCALIZATION_WORKSPACE_KEY] = workspace

        image = workspace["image"]
        label_coordinates = workspace["label_coordinates"]
        label_targets = workspace["label_targets"]
        correction_scope_bbox = workspace["correction_scope_bbox"]
        result_path = ""
        if is_debug_file_enabled():
            result_path = str(
                Path(context.request.output_dir)
                / "metadata"
                / "localization_result.json"
            )

        image_info = NiftiImageInfo(
            size_xyz=list(image.GetSize()),
            spacing_xyz=list(image.GetSpacing()),
            origin_xyz=list(image.GetOrigin()),
            direction=list(image.GetDirection()),
        ).to_dict()
        payload = {
            "coords": self._coordinates_to_dict(label_coordinates),
            "localization_profile": correction_result[
                "localization_profile"
            ],
            "cut_config": correction_result["cut_config"],
            "cut_configs": correction_result["cut_configs"],
            "resolved_cut_config": correction_result["resolved_cut_config"],
            "resolved_cut_configs": correction_result[
                "resolved_cut_configs"
            ],
            "slice_rules": correction_result["slice_rules"],
            "spine_cpr_config": self._resolve_spine_cpr_config(
                context.request.slice_part.value,
            ),
            "label_targets": self._label_targets_to_dict(label_targets),
            "dataset_json_path": correction_result["dataset_json_path"],
            "localization_result_path": result_path,
            "image": image_info,
            "raw_for_reconstruction_path": correction_result[
                "raw_for_reconstruction_path"
            ],
            "seg_for_reconstruction_path": working_seg_path,
            "correction_applied": correction_result["correction_applied"],
            "correction_angle_degrees": correction_result[
                "correction_angle_degrees"
            ],
            "correction_scope_bbox": correction_scope_bbox.to_dict(),
            "correction_params": correction_result["correction_params"],
            "corrected_raw_path": correction_result["corrected_raw_path"],
            "corrected_seg_path": correction_result["corrected_seg_path"],
            "candidate_raw_path": correction_result["candidate_raw_path"],
            "candidate_seg_path": correction_result["candidate_seg_path"],
            "is_skewed": correction_result["is_skewed"],
            "skew_params": correction_result["skew_params"],
        }
        if correction_result.get("spine_label_sequence") is not None:
            payload["spine_label_sequence"] = correction_result["spine_label_sequence"]
        context.set_stage_result(LocalizationStageResult(payload=payload))

        DebugLog(f"  已定位 {len(label_coordinates)} 个 label")
        DebugLog("  第四阶段完成")
        DebugLog("=" * 60)
        return context
