"""第五阶段：按部位生成解剖坐标系和专用几何。"""

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.core.resources import resource_operation
from src.models import (
    AnatomicalAnalysisStageResult,
    LabelTarget,
)
from src.utils import DebugLog, is_debug_file_enabled

from .stage_localization import (
    LocalizationOperations,
    _LOCALIZATION_WORKSPACE_KEY,
)


class AnatomicalAnalysisStage(LocalizationOperations):
    """计算通用解剖坐标，并为脊柱生成四棱柱等专用几何。"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, stage_name="anatomical_analysis")

    @classmethod
    def _brain_framing_support(
        cls,
        seg_array: np.ndarray,
        dataset_json_path: str | None,
    ) -> tuple[tuple[int, ...], tuple[str, ...], dict]:
        """仅解析同次分割中的骨性取景支持，不扩大脑区切片范围。

        模型的 ``skull`` 是实际解剖标签名，不是请求部位名。只保留语义明确、
        体素存在的正数标签；第七阶段按脑区 SI 及计划允许的颅顶余量筛选，排除范围外的
        面部、下颌等骨体素。这里不生成骨坐标或全卷脑骨并集。
        """

        diagnostic = {"status": "missing", "source_labels": [], "voxel_count": 0}
        if not dataset_json_path:
            return (), (), diagnostic
        label_map = cls._load_label_map(Path(dataset_json_path))
        candidates = {
            name: tuple(sorted({value for value in ids if value > 0}))
            for name, ids in label_map.items() if name.strip().casefold() == "skull"
        }
        candidates = {name: ids for name, ids in candidates.items() if ids}
        if not candidates:
            return (), (), diagnostic
        active_ids, active_names = set(), []
        for name, ids in candidates.items():
            count = sum(int(np.count_nonzero(np.isin(section, ids))) for section in seg_array)
            if count:
                active_ids.update(ids)
                active_names.append(name)
        if not active_ids:
            diagnostic["status"] = "empty"
            return (), (), diagnostic
        # 计数为模型中可用骨体素数；真正用于取景的范围由第七阶段再约束。
        count = sum(int(np.count_nonzero(np.isin(section, list(active_ids)))) for section in seg_array)
        diagnostic.update(status="available", source_labels=active_names, voxel_count=count)
        return tuple(sorted(active_ids)), tuple(active_names), diagnostic

    @classmethod
    def _lung_pose_support(
        cls,
        target_mask: np.ndarray,
        seg_array: np.ndarray,
        dataset_json_path: str | None,
    ) -> tuple[np.ndarray | None, tuple[str, ...]]:
        """从本次模型语义标签补充胸膜腔姿态证据，不扩展肺切割目标。

        仅识别明确的 pleural effusion 标签；不按模型号、数字标签或全部前景
        猜测支持范围，也不纳入心包标签。缺少该可选标签时保持原肺流程。
        """

        if not dataset_json_path:
            return None, ()
        label_map = cls._load_label_map(Path(dataset_json_path))
        support_labels = tuple(
            name for name in label_map
            if " ".join(name.replace("_", " ").casefold().split()) == "pleural effusion"
        )
        support_ids = {
            label_id for name in support_labels for label_id in label_map[name] if label_id > 0
        }
        if not support_ids:
            return None, ()
        support = np.isin(seg_array, list(support_ids))
        if not np.any(support & ~target_mask):
            return None, ()
        support |= target_mask
        return support, support_labels

    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """消费纯定位结果，并在第五阶段结束时释放运行时快照。"""

        try:
            return self._execute_with_workspace(context)
        finally:
            # 第六阶段只消费已序列化的定位和解剖结果。无论本阶段成功
            # 还是失败，都不应让完整分割、逐标签体素和 mask 继续驻留。
            if context.metadata.pop(_LOCALIZATION_WORKSPACE_KEY, None) is not None:
                DebugLog("  已释放第三至第五阶段运行时 label 快照")

    def _execute_with_workspace(
        self,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """复用或重建运行时快照，完成解剖坐标和专用几何计算。"""

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
            with resource_operation("anatomical_analysis", "read_working_segmentation") as details:
                image = sitk.ReadImage(str(working_seg_path))
                details.update(size_xyz=image.GetSize(), pixel_type=image.GetPixelIDTypeAsString())
                seg_array = sitk.GetArrayFromImage(image)
            with resource_operation(
                "anatomical_analysis", "rebuild_label_snapshot", label_count=len(label_targets),
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
                "localization_profile": localization_result[
                    "localization_profile"
                ],
                "cut_config": localization_result["cut_config"],
                "cut_configs": localization_result["cut_configs"],
                "resolved_cut_config": localization_result[
                    "resolved_cut_config"
                ],
                "resolved_cut_configs": localization_result[
                    "resolved_cut_configs"
                ],
                "slice_rules": localization_result["slice_rules"],
            }
            context.metadata[_LOCALIZATION_WORKSPACE_KEY] = workspace

        image = workspace["image"]
        seg_array = workspace["seg_array"]
        label_regions = workspace["label_regions"]
        label_targets = workspace["label_targets"]
        label_coordinates = workspace["label_coordinates"]
        target_mask = workspace["target_mask"]
        correction_scope_bbox = workspace["correction_scope_bbox"]

        slice_part = context.request.slice_part.value
        excluded_spine_labels = localization_result.get(
            "spine_cpr_config",
            {},
        ).get("excluded_labels", [])
        interspace_regions = localization_result.get(
            "resolved_cut_configs",
            {},
        ).get("interspace", {}).get("intervals", [])
        label_masks: Dict[str, np.ndarray] = {}
        # 脊柱直接复用已有索引，避免每个椎体额外常驻一份整卷 bool mask。
        if slice_part == "髋骨":
            with resource_operation(
                "anatomical_analysis", "build_hip_label_masks",
                label_count=len(label_regions), shape_zyx=seg_array.shape,
            ):
                for label_name, region in label_regions.items():
                    label_mask = np.zeros(seg_array.shape, dtype=bool)
                    if region.indices_zyx.size:
                        label_mask[tuple(region.indices_zyx.T)] = True
                    label_masks[label_name] = label_mask

        resolved_cut_config = localization_result["resolved_cut_config"]
        anatomical_cut_config = localization_result[
            "resolved_cut_configs"
        ].get("whole_range", resolved_cut_config)
        start_labels = anatomical_cut_config.get("start_labels") or []
        end_labels = anatomical_cut_config.get("end_labels") or []
        start_label = start_labels[0] if start_labels else None
        end_label = end_labels[0] if end_labels else None
        if slice_part in {"颈椎", "胸椎", "腰椎", "胸腰椎", "全脊柱"}:
            # 范围配置使用标准椎体名，实际模型可能以 sacrum 表示 S1。
            # 仅解析本次已验证的标签，不改写持久配置或缺标签校验政策。
            start_label = self._resolve_spine_endpoint_label(start_label, label_targets)
            end_label = self._resolve_spine_endpoint_label(end_label, label_targets)
        def load_spine_raw_array():
            with resource_operation("anatomical_analysis", "read_spine_raw") as details:
                raw_image = sitk.ReadImage(localization_result["raw_for_reconstruction_path"])
                details.update(size_xyz=raw_image.GetSize(), pixel_type=raw_image.GetPixelIDTypeAsString())
                if raw_image.GetSize() != image.GetSize() or any(
                    not np.allclose(actual, expected, rtol=0.0, atol=1e-5)
                    for actual, expected in (
                        (raw_image.GetSpacing(), image.GetSpacing()),
                        (raw_image.GetOrigin(), image.GetOrigin()),
                        (raw_image.GetDirection(), image.GetDirection()),
                    )
                ):
                    raise ValueError("端面密度验证的原图与工作分割空间不一致")
                return sitk.GetArrayFromImage(raw_image)

        with resource_operation(
            "anatomical_analysis", "build_spine_body_geometry",
            slice_part=slice_part, size_xyz=image.GetSize(), label_count=len(label_targets),
        ):
            spine_geometry = self._build_spine_body_geometry(
                slice_part=slice_part,
                image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                label_regions=label_regions,
                excluded_labels=excluded_spine_labels,
                interspace_regions=interspace_regions,
                raw_array_loader=load_spine_raw_array,
                pose_cache=workspace.get("spine_pose_cache"),
                validate_interspaces=any(
                    config.slice_type.value == "椎间隙"
                    for config in context.request.plane_configs
                ),
            )
        if spine_geometry is not None and localization_result.get("spine_label_sequence") is not None:
            spine_geometry["label_sequence"] = localization_result["spine_label_sequence"]
        lung_pose_support_mask, lung_pose_support_labels = (None, ())
        if slice_part == "肺部":
            with resource_operation("anatomical_analysis", "build_lung_pose_support"):
                lung_pose_support_mask, lung_pose_support_labels = self._lung_pose_support(
                    target_mask, seg_array, localization_result.get("dataset_json_path"),
                )
        framing_label_ids, framing_source_labels = (), ()
        coverage_diagnostic = None
        if slice_part == "颅脑":
            with resource_operation("anatomical_analysis", "build_brain_framing_support"):
                framing_label_ids, framing_source_labels, coverage_diagnostic = self._brain_framing_support(
                    seg_array, localization_result.get("dataset_json_path"),
                )
        # 脑区独占姿态与三轴基础范围；规划另加配置余量，骨只作独立取景支持。
        with resource_operation(
            "anatomical_analysis", "build_anatomical_frame",
            slice_part=slice_part, size_xyz=image.GetSize(), label_count=len(label_targets),
        ):
            anatomical_geometry, anatomical_diagnostics = self._build_anatomical_geometry(
                slice_part=slice_part,
                target_mask=target_mask,
                image=image,
                label_masks=label_masks,
                start_label=start_label,
                end_label=end_label,
                label_pose_annotations=(spine_geometry or {}).get("filtered_annotation_centroids"),
                label_indices_zyx=(
                    {name: region.indices_zyx for name, region in label_regions.items()}
                    if slice_part in {"颈椎", "胸椎", "腰椎", "胸腰椎", "全脊柱"} else None
                ),
                lung_pose_support_mask=lung_pose_support_mask,
                lung_pose_support_labels=lung_pose_support_labels,
            )
        del lung_pose_support_mask
        if coverage_diagnostic is not None:
            anatomical_diagnostics["bone_coverage"] = coverage_diagnostic
            if coverage_diagnostic["status"] != "available":
                anatomical_geometry.global_frame.warnings.append(
                    "本次分割缺少可用骨性取景支持，仅按脑区居中，不能保证脑部周边骨轮廓完整"
                )

        debug_label_path = ""
        debug_body_path = ""
        debug_prism_path = ""
        if is_debug_file_enabled():
            metadata_dir = Path(context.request.output_dir) / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            label_path = metadata_dir / "标签掩膜.nii.gz"
            debug_file_start = time.perf_counter()
            self._write_debug_label_mask_overlay(
                reference_image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=label_path,
            )
            DebugLog(
                "  [调试文件性能] 已生成: "
                f"{label_path}, "
                f"耗时={time.perf_counter() - debug_file_start:.3f}s"
            )
            debug_label_path = str(label_path)

            body_path = metadata_dir / "椎体主体掩膜.nii.gz"
            debug_file_start = time.perf_counter()
            self._write_debug_vertebral_body_mask_overlay(
                reference_image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=body_path,
                excluded_labels=excluded_spine_labels,
            )
            DebugLog(
                "  [调试文件性能] 已生成: "
                f"{body_path}, "
                f"耗时={time.perf_counter() - debug_file_start:.3f}s"
            )
            debug_body_path = str(body_path)

            prism_path = metadata_dir / "椎体四棱柱掩膜.nii.gz"
            debug_file_start = time.perf_counter()
            self._write_debug_vertebral_body_prism_mask_overlay(
                reference_image=image,
                seg_array=seg_array,
                label_targets=label_targets,
                output_path=prism_path,
                excluded_labels=excluded_spine_labels,
            )
            DebugLog(
                "  [调试文件性能] 已生成: "
                f"{prism_path}, "
                f"耗时={time.perf_counter() - debug_file_start:.3f}s"
            )
            debug_prism_path = str(prism_path)

            self._write_localization_result(
                Path(localization_result["localization_result_path"]),
                request=context.request,
                model_info=workspace.get("model_info") or {},
                cut_config=localization_result["cut_config"],
                cut_configs=localization_result["cut_configs"],
                resolved_cut_config=resolved_cut_config,
                resolved_cut_configs=localization_result[
                    "resolved_cut_configs"
                ],
                slice_rules=localization_result["slice_rules"],
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
                debug_label_mask_overlay_path=debug_label_path,
                debug_vertebral_body_mask_overlay_path=debug_body_path,
                debug_vertebral_body_prism_mask_overlay_path=debug_prism_path,
                spine_geometry=spine_geometry,
                is_skewed=localization_result["is_skewed"],
                skew_params=localization_result["skew_params"],
                anatomical_geometry=anatomical_geometry,
                spine_cpr_config=localization_result["spine_cpr_config"],
                anatomical_geometry_diagnostics=anatomical_diagnostics,
                framing_source_labels=list(framing_source_labels) if slice_part == "颅脑" else None,
                framing_label_ids=list(framing_label_ids) if slice_part == "颅脑" else None,
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
        if slice_part == "颅脑":
            payload.update(
                framing_source_labels=list(framing_source_labels),
                framing_label_ids=list(framing_label_ids),
            )

        context.set_stage_result(AnatomicalAnalysisStageResult(
            payload=payload,
            warnings=list(anatomical_geometry.global_frame.warnings) if anatomical_geometry else [],
        ))

        if anatomical_geometry is not None:
            frame = anatomical_geometry.global_frame
            DebugLog(f"  解剖坐标: {frame.method}, pose_applied={frame.applied}")
            if slice_part == "颅脑":
                pitch = anatomical_diagnostics.get("pitch", {})
                DebugLog(
                    f"  颅脑形态俯仰: status={pitch.get('status')}, "
                    f"候选角={pitch.get('candidate_angle_degrees')}°, "
                    f"采用角={pitch.get('angle_degrees')}°, 原因={pitch.get('reason')}"
                )
            for warning in frame.warnings:
                DebugLog(f"  解剖坐标告警: {warning}")
        else:
            DebugLog("  解剖坐标: skipped")
        DebugLog(
            "  脊柱专用几何: "
            f"{'applied' if spine_geometry is not None else 'skipped'}"
        )
        if spine_geometry is not None:
            DebugLog("  矢状投影去重: 一维整数键（稳定路径）")
        DebugLog("  第五阶段完成")
        DebugLog("=" * 60)
        return context
