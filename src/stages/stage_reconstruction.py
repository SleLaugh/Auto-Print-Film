#!/src/services/stage_reconstruction.py
# -*- coding: utf-8 -*-

"""
阶段五：重建与输出
MPR多平面重建与DICOM生成
按阶段四 SlicePlan 执行直线 plane_stack 重建
"""

from pathlib import Path
from typing import Dict, Any, List

import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.reconstruction import PlaneStackResampler
from src.reconstruction.dicom import (
    new_uid,
    read_source_dicom_metadata,
    write_ct_dicom,
)
from src.reconstruction.windowing import apply_window_to_image, unique_window_presets
from src.utils import DebugLog


class ReconstructionStage(BaseStage):
    """重建与输出阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="reconstruction", config=config)
        self.parallel = config.get('parallel', True) if config else True
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        
        loc_result = context.get_result('localization')
        map_result = context.get_result('mapping')
        
        if not loc_result:
            raise ValueError("前置阶段 'localization' 结果不存在")
        if not map_result:
            raise ValueError("前置阶段 'mapping' 结果不存在")
        
        # 阶段三现在会明确告诉后续阶段“实际该用哪张 raw 图像”。
        # 未回正时这里是原始 raw_0000.nii.gz；发生回正时才是 raw_corrected.nii.gz。
        # corrected_raw_path 只作为旧结果兼容字段保留。
        image_path = (
            loc_result.get('raw_for_reconstruction_path')
            or loc_result.get('corrected_raw_path')
        )
        if not image_path:
            raise ValueError("定位结果缺少 raw_for_reconstruction_path/corrected_raw_path")

        plans = map_result.get("plans", [])
        if not isinstance(plans, list):
            raise ValueError("阶段四输出 plans 必须是列表")
        for plan in plans:
            self._validate_plan(plan)
        
        DebugLog("=" * 60)
        DebugLog("【阶段五】重建与输出 - MPR重建与DICOM生成")
        DebugLog(f"  输入图像: {image_path}")
        DebugLog(f"  输出目录: {request.output_dir}")
        DebugLog(f"  规划组数: {len(plans)}")
        DebugLog("=" * 60)

        output_root = Path(request.output_dir)
        if not plans:
            result = self._build_plan_summary(plans, output_root)
            context.set_result(self.name, result)
            DebugLog("  没有可执行的切片计划")
            return context

        image = sitk.ReadImage(str(image_path))
        preprocessing_result = context.get_result("preprocessing") or {}
        source_metadata = dict(preprocessing_result.get("source_dicom_metadata") or {})
        if not source_metadata:
            source_metadata = read_source_dicom_metadata(request.dicom_dir)
        study_uid = source_metadata.get("0020|000d") or new_uid()
        plan_results = []
        for plan in plans:
            DebugLog(f"  执行 plan: {plan['plan_id']}")
            plan_results.append(
                self._execute_plan(
                    image=image,
                    plan=plan,
                    output_root=output_root,
                    study_uid=study_uid,
                    source_metadata=source_metadata,
                )
            )

        result = self._build_plan_summary(plans, output_root)
        result["outputs"] = plan_results
        
        context.set_result(self.name, result)
        
        DebugLog("  ✅ 阶段五完成")
        DebugLog("=" * 60)
        
        return context

    def _execute_plan(
        self,
        image: sitk.Image,
        plan: Dict[str, Any],
        output_root: Path,
        study_uid: str,
        source_metadata: Dict[str, str],
    ) -> Dict[str, Any]:
        resampler = PlaneStackResampler()
        windows = unique_window_presets(plan["window_presets"])
        series_uids = {window.name: new_uid() for window in windows}
        group_name = self._plan_group_name(plan)
        output_dirs = {
            window.name: output_root
            / str(plan["plane"])
            / window.name
            / group_name
            if group_name
            else output_root / str(plan["plane"]) / window.name
            for window in windows
        }
        for output_dir in output_dirs.values():
            output_dir.mkdir(parents=True, exist_ok=True)

        for planned_slice in plan["slices"]:
            sampled_image = resampler.sample_slice(image, plan, planned_slice)
            for window in windows:
                windowed_image = apply_window_to_image(sampled_image, window)
                output_path = output_dirs[window.name] / (
                    f"切片_{int(planned_slice['index']):04d}.dcm"
                )
                write_ct_dicom(
                    image=windowed_image,
                    output_path=output_path,
                    plan=plan,
                    planned_slice=planned_slice,
                    window=window,
                    series_uid=series_uids[window.name],
                    study_uid=study_uid,
                    total_slices=len(plan["slices"]),
                    source_metadata=source_metadata,
                )

        return {
            "plan_id": plan["plan_id"],
            "plane": plan["plane"],
            "group_name": plan.get("group_name", ""),
            "slice_count": len(plan["slices"]),
            "image_count": len(plan["slices"]) * len(windows),
            "window_outputs": {
                name: str(path)
                for name, path in output_dirs.items()
            },
        }

    @staticmethod
    def _plan_group_name(plan: Dict[str, Any]) -> str:
        if plan.get("cut_mode") in {"whole_range", "整切"}:
            return ""
        group_name = str(plan.get("group_name", ""))
        if group_name in {"", "whole_range"}:
            return "区间切"
        if group_name in {"interspace", "区间切"}:
            return "区间切"
        return group_name.replace("/", "_").replace("\\", "_")

    @staticmethod
    def _validate_plan(plan: Dict[str, Any]) -> None:
        plan_id = plan.get("plan_id", "<unknown>") if isinstance(plan, dict) else "<unknown>"
        if not isinstance(plan, dict):
            raise ValueError(f"plan_id={plan_id} 必须是对象")
        if not plan.get("plan_id"):
            raise ValueError("plan_id 不能为空")
        if plan.get("geometry_type") != "plane_stack":
            raise ValueError(
                f"plan_id={plan_id} geometry_type={plan.get('geometry_type')} "
                "当前阶段五仅支持 plane_stack"
            )
        for field_name in (
            "plane",
            "slices",
            "extent_mm",
            "output_spacing_mm",
            "slice_thickness",
            "slice_spacing",
            "window_presets",
        ):
            if field_name not in plan:
                raise ValueError(f"plan_id={plan_id} 缺少字段: {field_name}")
        if not isinstance(plan["slices"], list):
            raise ValueError(f"plan_id={plan_id} slices 必须是列表")
        if not plan["slices"]:
            raise ValueError(f"plan_id={plan_id} slices 不能为空")
        if not isinstance(plan["extent_mm"], dict):
            raise ValueError(f"plan_id={plan_id} extent_mm 必须是对象")
        if not isinstance(plan["output_spacing_mm"], dict):
            raise ValueError(f"plan_id={plan_id} output_spacing_mm 必须是对象")
        if float(plan["slice_thickness"]) <= 0:
            raise ValueError(f"plan_id={plan_id} slice_thickness 必须大于 0")
        if float(plan["slice_spacing"]) <= 0:
            raise ValueError(f"plan_id={plan_id} slice_spacing 必须大于 0")
        if not isinstance(plan["window_presets"], list) or not plan["window_presets"]:
            raise ValueError(f"plan_id={plan_id} window_presets 不能为空")
        for field_name in ("width", "height"):
            if float(plan["extent_mm"].get(field_name, 0.0)) < 0:
                raise ValueError(f"plan_id={plan_id} extent_mm.{field_name} 不能小于 0")
        for field_name in ("x", "y"):
            if float(plan["output_spacing_mm"].get(field_name, 0.0)) <= 0:
                raise ValueError(
                    f"plan_id={plan_id} output_spacing_mm.{field_name} 必须大于 0"
                )
        for planned_slice in plan["slices"]:
            for field_name in ("index", "center", "normal", "row_dir", "col_dir"):
                if field_name not in planned_slice:
                    raise ValueError(
                        f"plan_id={plan_id} slice 缺少字段: {field_name}"
                    )

    @staticmethod
    def _build_plan_summary(
        plans: List[Dict[str, Any]],
        output_root: Path,
    ) -> Dict[str, Any]:
        plane_summary = {}
        for plan in plans:
            plane = plan["plane"]
            window_count = len(dict.fromkeys(plan["window_presets"]))
            summary = plane_summary.setdefault(
                plane,
                {
                    "plan_count": 0,
                    "count": 0,
                    "slice_count": 0,
                    "image_count": 0,
                    "output_dir": str(output_root / plane),
                },
            )
            slice_count = len(plan["slices"])
            summary["plan_count"] += 1
            summary["count"] += slice_count
            summary["slice_count"] += slice_count
            summary["image_count"] += slice_count * window_count

        slice_count = sum(summary["slice_count"] for summary in plane_summary.values())
        image_count = sum(summary["image_count"] for summary in plane_summary.values())
        result = {
            "plan_count": len(plans),
            "slice_count": slice_count,
            "image_count": image_count,
            "planes": plane_summary,
        }
        result.update({
            "axial": plane_summary.get("横状位", {
                "plan_count": 0,
                "count": 0,
                "slice_count": 0,
                "image_count": 0,
                "output_dir": str(output_root / "横状位"),
            }),
            "sagittal": plane_summary.get("矢状位", {
                "plan_count": 0,
                "count": 0,
                "slice_count": 0,
                "image_count": 0,
                "output_dir": str(output_root / "矢状位"),
            }),
            "coronal": plane_summary.get("冠状位", {
                "plan_count": 0,
                "count": 0,
                "slice_count": 0,
                "image_count": 0,
                "output_dir": str(output_root / "冠状位"),
            }),
        })
        return result
