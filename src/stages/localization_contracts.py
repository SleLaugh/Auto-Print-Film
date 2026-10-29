"""定位阶段的配置解析与结果序列化契约。

这些方法不负责医学几何计算，只把请求配置转换为第三至第五阶段共享的
内部契约，并将定位结果写成 JSON。它们通过 mixin 注入
``LocalizationOperations``，保留原有方法名和调用方式。
"""

from pathlib import Path
from typing import Any, Dict
import json

import SimpleITK as sitk

from src.config import localization_for, localization_profiles_for, slice_rules_for
from src.models import (
    AnatomicalGeometryResult,
    CutConfig,
    LabelCoordinate,
    LabelTarget,
    LocalizationResult,
    NiftiImageInfo,
    VoxelBBox,
)
from src.planning.slice_policy import localization_profile_for_request


class LocalizationContractMixin:
    """定位配置解析和 ``LocalizationResult`` 序列化的共享实现。"""

    @staticmethod
    def _write_localization_result(
        output_path: Path,
        request,
        model_info: Dict[str, Any],
        cut_config: CutConfig,
        cut_configs: Dict[str, CutConfig],
        resolved_cut_config: Dict[str, Any],
        resolved_cut_configs: Dict[str, Dict[str, Any]],
        slice_rules: Dict[str, Any],
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
        debug_label_mask_overlay_path: str,
        debug_vertebral_body_mask_overlay_path: str,
        debug_vertebral_body_prism_mask_overlay_path: str,
        spine_geometry: Dict[str, Any] | None,
        is_skewed: bool,
        skew_params: Dict[str, Any],
        anatomical_geometry: AnatomicalGeometryResult | None = None,
        spine_cpr_config: Dict[str, Any] | None = None,
        anatomical_geometry_diagnostics: Dict[str, Any] | None = None,
        framing_source_labels: list[str] | None = None,
        framing_label_ids: list[int] | None = None,
    ) -> None:
        """写出定位结果，供后续规划、重建和调试使用。"""

        localization_result = LocalizationResult(
            slice_part=request.slice_part.value,
            localization_profile=(
                localization_profile_for_request(
                    config.slice_type for config in request.plane_configs
                )
            ),
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
            resolved_cut_configs=resolved_cut_configs,
            slice_rules=slice_rules,
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
            debug_label_mask_overlay_path=debug_label_mask_overlay_path,
            debug_vertebral_body_mask_overlay_path=debug_vertebral_body_mask_overlay_path,
            debug_vertebral_body_prism_mask_overlay_path=(
                debug_vertebral_body_prism_mask_overlay_path
            ),
            spine_geometry=spine_geometry,
            spine_cpr_config=spine_cpr_config or {},
            is_skewed=is_skewed,
            skew_params=skew_params,
            anatomical_geometry=anatomical_geometry,
        )
        data = localization_result.to_dict()
        if anatomical_geometry_diagnostics is not None:
            data["anatomical_geometry_diagnostics"] = anatomical_geometry_diagnostics
        if framing_source_labels is not None:
            data["framing_source_labels"] = framing_source_labels
        if framing_label_ids is not None:
            data["framing_label_ids"] = framing_label_ids
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _label_targets_to_dict(
        label_targets: Dict[str, LabelTarget],
    ) -> Dict[str, Dict[str, Any]]:
        """把按名称索引的标签目标实体转换为可写 JSON 的字典。"""

        return {
            name: target.to_dict()
            for name, target in label_targets.items()
        }

    @staticmethod
    def _coordinates_to_dict(
        label_coordinates: Dict[str, LabelCoordinate],
    ) -> Dict[str, Dict[str, Any]]:
        """把按名称索引的标签定位结果转换为可写 JSON 的字典。"""

        return {
            name: coord.to_dict()
            for name, coord in label_coordinates.items()
        }

    @staticmethod
    def _resolve_request_localization_profile(request) -> str:
        """根据全部 SliceType 选择第三阶段需要加载的定位范围方案。"""
        return localization_profile_for_request(
            config.slice_type for config in request.plane_configs
        )

    def _resolve_cut_config(
        self,
        request,
        localization_profile: str | None = None,
    ) -> CutConfig:
        """根据请求中的 SliceType 查找对应的定位范围配置。"""

        localization_profile = (
            localization_profile
            or self._resolve_request_localization_profile(request)
        )
        slice_part = getattr(request.slice_part, "value", request.slice_part)
        cut_configs = self._resolve_cut_configs(request)
        cut_config = cut_configs.get(localization_profile)
        if cut_config is None:
            raise ValueError(
                f"未找到定位范围配置: slice_part={slice_part}, "
                f"profile={localization_profile}"
            )
        return cut_config

    def _resolve_spine_cpr_config(self, slice_part: str) -> Dict[str, Any]:
        """从当前脊柱部位配置中读取 CPR 配置，非脊柱部位返回空配置。"""

        localization = localization_for(self._slice_part_config, slice_part)
        config = localization.get("spine_cpr")
        if not isinstance(config, dict):
            return {}
        excluded_labels = []
        for label_name in config.get("excluded_labels", []):
            excluded_labels.extend(self._label_alias(label_name))
        result = {
            "full_label": config["full_label"],
            "excluded_labels": excluded_labels,
        }
        extension = config.get("height_extension_mm")
        if isinstance(extension, dict):
            result["height_extension_mm"] = {
                "top_mm": float(extension.get("top_mm", 0.0)),
                "bottom_mm": float(extension.get("bottom_mm", 0.0)),
            }
        return result

    def _resolve_cut_configs(self, request) -> Dict[str, CutConfig]:
        """返回当前部位的全部定位范围配置。"""

        slice_part = getattr(request.slice_part, "value", request.slice_part)
        return localization_profiles_for(self._slice_part_config, slice_part)

    def _resolve_slice_rules(self, request) -> Dict[str, Any]:
        """返回当前部位经统一校验的状位与切法规则。"""

        slice_part = getattr(request.slice_part, "value", request.slice_part)
        return slice_rules_for(self._slice_part_config, slice_part)

    def _resolve_label_aliases(self, cut_config: CutConfig) -> Dict[str, Any]:
        """把业务短名解析成模型/数据集中的标准 label 名。"""

        if isinstance(cut_config, list):
            return {
                "intervals": [
                    {
                        "region": interval["region"],
                        "start": interval["start"],
                        "end": interval["end"],
                        "start_labels": self._label_alias(interval["start"]),
                        "end_labels": self._label_alias(interval["end"]),
                    }
                    for interval in cut_config
                ]
            }

        return {
            "region": cut_config["region"],
            "start": cut_config["start"],
            "end": cut_config["end"],
            "start_labels": self._label_alias(cut_config["start"]),
            "end_labels": self._label_alias(cut_config["end"]),
        }

    def _label_alias(self, name: str) -> list[str]:
        """获取一个定位名称对应的标准 label 名列表。"""

        aliases = self._slice_part_config.get("label_aliases", {})
        value = aliases.get(name, name)
        if isinstance(value, list):
            return value
        return [value]
