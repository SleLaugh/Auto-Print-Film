"""第三阶段：为后续定位与重建选定统一图像空间。"""

from pathlib import Path
from typing import Any, Dict

import SimpleITK as sitk
import numpy as np

from src.core.context import ProcessingContext
from src.core.resources import resource_operation
from src.models import SpatialCorrectionStageResult
from src.utils import DebugLog

from .stage_localization import (
    LocalizationOperations,
    _LOCALIZATION_WORKSPACE_KEY,
)
from .spine_label_sequence import repair_lumbar_body_sequence
from .spine_mixed_labels import repair_mixed_spine_body_labels


class SpatialCorrectionStage(LocalizationOperations):
    """估计 XY 校正候选，并唯一决定后续使用的 raw/seg。"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, stage_name="spatial_correction")

    def _repair_spine_body_sequence(
        self, *, image, seg_array, raw_path, seg_path, output_dir,
        slice_part, label_map, label_targets,
    ):
        """仅在 CT 和主体序列共同支持时修正前方椎体的语义编号。"""

        if slice_part not in {"颈椎", "腰椎", "胸腰椎", "全脊柱"}:
            return image, seg_array, str(seg_path), None
        if slice_part == "颈椎" and not any(
            name.startswith("vertebrae_C") for name in label_targets
        ):
            return image, seg_array, str(seg_path), None

        def load_aligned_raw():
            with resource_operation(
                "spatial_correction", "sequence_read_raw", path=str(raw_path),
            ) as details:
                try:
                    raw_image = sitk.ReadImage(str(raw_path))
                except RuntimeError as error:
                    raise ValueError("无法读取原图核实椎体序列") from error
                details.update(self._image_resource_details(raw_image))
                if raw_image.GetSize() != image.GetSize() or any(
                    not np.allclose(actual, expected, rtol=0.0, atol=1e-5)
                    for actual, expected in (
                        (raw_image.GetSpacing(), image.GetSpacing()),
                        (raw_image.GetOrigin(), image.GetOrigin()),
                        (raw_image.GetDirection(), image.GetDirection()),
                    )
                ):
                    raise ValueError("原图与分割空间不一致，不能核实椎体序列")
                return sitk.GetArrayFromImage(raw_image)

        def extract_body(array, target):
            with resource_operation(
                "spatial_correction", "sequence_extract_body",
                label=target.label_name, shape_zyx=array.shape,
            ):
                if target.label_name.endswith("_S1"):
                    return self._s1_body_candidate_mask(array, target)
                return self._vertebral_body_candidate_mask(
                    array, target, spacing_zyx=tuple(reversed(image.GetSpacing())),
                )

        repaired, diagnostic = None, None
        if slice_part in {"腰椎", "胸腰椎", "全脊柱"}:
            repaired, diagnostic = repair_lumbar_body_sequence(
                seg_array, load_aligned_raw, image, label_map,
                label_targets=label_targets, body_extractor=extract_body,
            )
        if repaired is None:
            mixed_repaired, mixed_diagnostic = repair_mixed_spine_body_labels(
                seg_array, load_aligned_raw, image, label_map,
                label_targets=label_targets,
                support_extractor=self._vertebral_body_support_candidates,
            )
            if mixed_repaired is not None:
                repaired, diagnostic = mixed_repaired, {
                    **mixed_diagnostic, "empty_slot_check": diagnostic,
                }
            elif diagnostic is not None:
                diagnostic["mixed_body_labels"] = mixed_diagnostic
            else:
                diagnostic = mixed_diagnostic
        diagnostic = {
            **diagnostic,
            "source_seg_path": str(seg_path),
            "working_seg_path": str(seg_path),
        }
        if repaired is None:
            return image, seg_array, str(seg_path), diagnostic
        if repaired.shape != seg_array.shape or not np.array_equal(
            repaired != 0, seg_array != 0,
        ):
            raise ValueError("椎体序列修正改变了分割前景范围")

        repaired_image = sitk.GetImageFromArray(repaired)
        repaired_image.CopyInformation(image)
        output_path = Path(output_dir) / "nifti" / "seg_spine_sequence_corrected.nii.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(repaired_image, str(output_path))
        diagnostic["working_seg_path"] = str(output_path)
        DebugLog(
            "  已按连续椎体主体与 CT 椎间隙证据修正前方标签序列: "
            f"{diagnostic.get('changed_voxels', diagnostic.get('changed_voxel_count', 0))} 体素"
        )
        return repaired_image, repaired, str(output_path), diagnostic

    @staticmethod
    def _image_resource_details(image: sitk.Image) -> dict[str, Any]:
        """Describe the loaded image and effective ITK default without changing it."""

        return {
            "size_xyz": image.GetSize(),
            "pixel_type": image.GetPixelIDTypeAsString(),
            "itk_threads": sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
        }

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
        DebugLog(f"  切割部位: {request.slice_part.value}")

        localization_profile = self._resolve_request_localization_profile(request)
        cut_configs = self._resolve_cut_configs(request)
        slice_rules = self._resolve_slice_rules(request)
        cut_config = cut_configs.get(localization_profile)
        if not cut_config:
            raise ValueError(
                f"未找到定位范围配置: slice_part={request.slice_part.value}, "
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

        with resource_operation(
            "spatial_correction", "read_segmentation", path=str(seg_path),
        ) as details:
            image = sitk.ReadImage(str(seg_path))
            details.update(self._image_resource_details(image))
            seg_array = sitk.GetArrayFromImage(image)
        with resource_operation("spatial_correction", "verify_spine_sequence") as details:
            image, seg_array, working_seg_path, sequence_diagnostic = self._repair_spine_body_sequence(
                image=image,
                seg_array=seg_array,
                raw_path=raw_path,
                seg_path=seg_path,
                output_dir=request.output_dir,
                slice_part=request.slice_part.value,
                label_map=label_map,
                label_targets=label_targets,
            )
            if sequence_diagnostic is not None:
                details["status"] = sequence_diagnostic.get("status")
        with resource_operation(
            "spatial_correction", "index_labels", label_count=len(label_targets),
        ):
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
        with resource_operation("spatial_correction", "target_bbox"):
            target_mask = self._build_target_mask(seg_array, label_targets)
            correction_scope_bbox = self._build_target_bbox(target_mask)

        with resource_operation("spatial_correction", "estimate_xy_candidate") as details:
            candidate = self._build_xy_correction_candidate(
                seg_array,
                label_targets,
                slice_part=request.slice_part.value,
                label_regions=label_regions,
                image=image,
            )
            details.update(
                angle_degrees=candidate["angle_degrees"],
                status=candidate["correction_status"],
            )
        DebugLog(
            "  横状位 XY 候选角: "
            f"{candidate['angle_degrees']:.3f}°, "
            f"status={candidate['correction_status']}"
        )

        with resource_operation(
            "spatial_correction", "write_xy_candidate", status=candidate["correction_status"],
        ):
            correction_result = self._write_xy_correction_candidate_images(
                raw_path=str(raw_path),
                seg_path=working_seg_path,
                output_dir=Path(request.output_dir) / "nifti",
                candidate=candidate,
                correction_center_bbox=correction_scope_bbox,
            )
        correction_result = self._promote_xy_correction_candidate_if_available(
            correction_result,
        )

        if correction_result["correction_applied"]:
            with resource_operation(
                "spatial_correction", "read_corrected_segmentation",
                path=str(correction_result["seg_for_reconstruction_path"]),
            ) as details:
                image = sitk.ReadImage(
                    str(correction_result["seg_for_reconstruction_path"])
                )
                details.update(self._image_resource_details(image))
                seg_array = sitk.GetArrayFromImage(image)
            with resource_operation(
                "spatial_correction", "index_corrected_labels", label_count=len(label_targets),
            ):
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
            with resource_operation("spatial_correction", "corrected_target_bbox"):
                target_mask = self._build_target_mask(seg_array, label_targets)
                correction_scope_bbox = self._build_target_bbox(target_mask)
            DebugLog("  已启用校正候选，后续阶段统一使用候选空间")
        else:
            DebugLog("  未启用校正，后续阶段使用原始空间")

        if sequence_diagnostic is not None:
            sequence_diagnostic["sequence_seg_path"] = working_seg_path
            sequence_diagnostic["working_seg_path"] = correction_result[
                "seg_for_reconstruction_path"
            ]

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
            "spine_label_sequence": sequence_diagnostic,
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
        if sequence_diagnostic is not None:
            payload["spine_label_sequence"] = sequence_diagnostic
        context.set_stage_result(SpatialCorrectionStageResult(payload=payload))

        DebugLog("  第三阶段完成")
        DebugLog("=" * 60)
        return context
