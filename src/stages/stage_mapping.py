#!/src/services/stage_mapping.py
# -*- coding: utf-8 -*-

"""
阶段四：切片规划

本阶段负责回答“要切哪些图”。它会根据第三步定位结果、部位、切割方式和
状位配置生成 SlicePlan[]，并把每一张待切图的中心点和方向提前展开好。
第五步后续只负责执行这些计划，不再重新决定切片数量和切片位置。
"""

import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import Point3D
from src.planning.geometry import normalize_vector, subtract_points, vector_length
from src.planning import select_planner
from src.utils import DebugLog, is_debug_log_file_enabled


class MappingStage(BaseStage):
    """切片规划阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="mapping", config=config)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        prev_result = context.get_result('localization')
        
        if not prev_result:
            raise ValueError("前置阶段 'localization' 结果不存在")
        
        request = context.request
        coords = prev_result.get('coords', {})
        
        DebugLog("=" * 60)
        DebugLog("【阶段四】切片规划 - 生成第五步执行计划")
        DebugLog(f"  部位: {request.body_part.value}")
        DebugLog(f"  切割方式: {request.cut_mode.value}")
        DebugLog(f"  定位坐标数量: {len(coords)}")
        DebugLog("=" * 60)
        
        # 第四步内部拆成 planner + expander。planner 负责找区间和路径；
        # expander 已在 planner 内部调用，把路径展开成每张图的 slices[]。
        DebugLog("  [1] select_planner: 选择部位/切法 planner...")
        planner = select_planner(
            request.body_part.value,
            request.cut_mode.value,
        )
        DebugLog(f"    planner: {planner.planner_name}")

        DebugLog("  [2] build_plans: 生成并展开切片计划...")
        planning_result = planner.build_plans(request, prev_result)
        plans = [plan.to_dict() for plan in planning_result.plans]
        diagnostics = self._build_plan_diagnostics(plans)

        debug_overlay_path = ""
        debug_overview_overlay_path = ""
        debug_native_overview_overlay_path = ""
        metadata_path = ""
        if is_debug_log_file_enabled():
            DebugLog("  [3] write_debug_overlay: 写出切割线 overlay...")
            debug_overlay_path = self._write_slice_plan_overlay(
            context=context,
            localization_result=prev_result,
            plans=plans,
        )
            (
                debug_overview_overlay_path,
                debug_native_overview_overlay_path,
            ) = self._write_slice_plan_overview_overlay(
                context=context,
                localization_result=prev_result,
                plans=plans,
            )
            DebugLog("  [4] write_metadata: 写出 slice_plans.json...")
            metadata_path = self._write_slice_plans(
                context,
                planner_name=planning_result.planner_name,
                plans=plans,
                diagnostics=diagnostics,
                warnings=planning_result.warnings,
                debug_overlay_path=debug_overlay_path,
                debug_overview_overlay_path=debug_overview_overlay_path,
                debug_native_overview_overlay_path=debug_native_overview_overlay_path,
            )

        result = {
            'planner': planning_result.planner_name,
            'plans': plans,
            'warnings': planning_result.warnings,
            'diagnostics': diagnostics,
            'plan_count': len(plans),
            'metadata_path': str(metadata_path),
            'debug_slice_plan_overlay_path': debug_overlay_path,
            'debug_slice_plan_overview_overlay_path': debug_overview_overlay_path,
            'debug_slice_plan_native_overview_overlay_path': debug_native_overview_overlay_path,
        }
        
        context.set_result(self.name, result)
        
        DebugLog(f"    切片计划数量: {len(plans)}")
        for diagnostic in diagnostics:
            DebugLog(
                "    "
                f"{diagnostic['plan_id']}: "
                f"{diagnostic['slice_count']} 张, "
                f"区间 {diagnostic['range_length_mm']}mm, "
                f"层厚 {diagnostic['slice_thickness']}mm, "
                f"层间距 {diagnostic['slice_spacing']}mm"
            )
        if debug_overlay_path:
            DebugLog(f"    切割线 overlay: {debug_overlay_path}")
        if debug_overview_overlay_path:
            DebugLog(f"    切割线总览 overlay: {debug_overview_overlay_path}")
        if planning_result.warnings:
            DebugLog(f"    warnings: {planning_result.warnings}")
        DebugLog("  ✅ 阶段四完成")
        DebugLog("=" * 60)
        
        return context

    def _write_slice_plans(
        self,
        context: ProcessingContext,
        planner_name: str,
        plans: list[Dict[str, Any]],
        diagnostics: list[Dict[str, Any]],
        warnings: list[str],
        debug_overlay_path: str,
        debug_overview_overlay_path: str,
        debug_native_overview_overlay_path: str,
    ) -> Path:
        """把阶段四输出写入 metadata/slice_plans.json，方便第五步和人工检查。"""

        metadata_dir = Path(context.request.output_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "slice_plans.json"
        data = {
            "schema_version": "slice_plan_v1",
            "planner": planner_name,
            "plan_count": len(plans),
            "plans": plans,
            "diagnostics": diagnostics,
            "warnings": warnings,
            "debug_slice_plan_overlay_path": debug_overlay_path,
            "debug_slice_plan_overview_overlay_path": debug_overview_overlay_path,
            "debug_slice_plan_native_overview_overlay_path": debug_native_overview_overlay_path,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return metadata_path

    def _build_plan_diagnostics(self, plans: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        为每个 plan 生成可读诊断信息。

        这部分不参与第五步执行，只用于排查“到底是计划张数少，还是 overlay
        看起来少”。尤其当层间距大于层厚时，输出图之间本来就会有物理空档，
        需要在 JSON 里直观看出来。
        """

        diagnostics = []
        for plan in plans:
            range_start = plan.get("range_start", {})
            range_end = plan.get("range_end", {})
            slice_spacing = float(plan.get("slice_spacing", 0.0))
            slice_thickness = float(plan.get("slice_thickness", 0.0))
            range_length = 0.0
            if range_start and range_end:
                range_length = vector_length(subtract_points(
                    end=self._dict_to_point(range_end),
                    start=self._dict_to_point(range_start),
                ))

            slice_count = len(plan.get("slices", []))
            slice_centers = [
                self._dict_to_point(planned_slice["center"])
                for planned_slice in plan.get("slices", [])
                if planned_slice.get("center")
            ]
            actual_spacings = [
                vector_length(subtract_points(end=current, start=previous))
                for previous, current in zip(slice_centers, slice_centers[1:])
            ]
            actual_spacing_min = min(actual_spacings, default=0.0)
            actual_spacing_max = max(actual_spacings, default=0.0)
            has_nonuniform_spacing = any(
                abs(actual_spacing - slice_spacing) > 1e-6
                for actual_spacing in actual_spacings
            )
            spacing_minus_thickness = slice_spacing - slice_thickness
            regular_center_count = (
                int(math.floor(range_length / slice_spacing)) + 1
                if slice_spacing > 0
                else 0
            )
            has_extra_endpoint_slice = False
            leading_uncovered = 0.0
            trailing_uncovered = 0.0
            target_start_data = plan.get("target_start")
            target_end_data = plan.get("target_end")
            target_length = 0.0
            first_coverage_start = None
            last_coverage_end = None
            target_shorter_than_thickness = False
            if target_start_data and target_end_data:
                target_start = self._dict_to_point(target_start_data)
                target_end = self._dict_to_point(target_end_data)
                target_vector = subtract_points(end=target_end, start=target_start)
                target_length = vector_length(target_vector)
                target_shorter_than_thickness = target_length < slice_thickness
            else:
                target_start = self._dict_to_point(range_start) if range_start else None
                target_end = self._dict_to_point(range_end) if range_end else None

            if slice_centers and target_start is not None and target_end is not None:
                target_vector = subtract_points(end=target_end, start=target_start)
                if vector_length(target_vector) > 0:
                    coverage_direction = normalize_vector(target_vector)
                else:
                    coverage_direction = Point3D(0.0, 0.0, 1.0)
                half_thickness = slice_thickness / 2.0
                first_coverage_start = Point3D(
                    slice_centers[0].x - coverage_direction.x * half_thickness,
                    slice_centers[0].y - coverage_direction.y * half_thickness,
                    slice_centers[0].z - coverage_direction.z * half_thickness,
                )
                last_coverage_end = Point3D(
                    slice_centers[-1].x + coverage_direction.x * half_thickness,
                    slice_centers[-1].y + coverage_direction.y * half_thickness,
                    slice_centers[-1].z + coverage_direction.z * half_thickness,
                )
                leading_delta = subtract_points(
                    end=first_coverage_start,
                    start=target_start,
                )
                trailing_delta = subtract_points(
                    end=target_end,
                    start=last_coverage_end,
                )
                leading_uncovered = max(
                    leading_delta.x * coverage_direction.x
                    + leading_delta.y * coverage_direction.y
                    + leading_delta.z * coverage_direction.z,
                    0.0,
                )
                trailing_uncovered = max(
                    trailing_delta.x * coverage_direction.x
                    + trailing_delta.y * coverage_direction.y
                    + trailing_delta.z * coverage_direction.z,
                    0.0,
                )
            calculation_note = (
                "阶段四按层间距从 range_start 到 range_end 展开每张输出图中心；"
                "区间末端不足一个完整层间距时不追加非等距中心。"
            )
            if spacing_minus_thickness > 0:
                calculation_note += (
                    f" 当前层间距比层厚大 {spacing_minus_thickness:.3f}mm，"
                    "厚层图之间会有未覆盖间隔，看起来会少线；"
                    "需要更多输出图时应减小 slice_spacing。"
                )

            diagnostics.append({
                "plan_id": plan.get("plan_id", ""),
                "plane": plan.get("plane", ""),
                "range_length_mm": round(range_length, 3),
                "slice_count": slice_count,
                "regular_center_count": regular_center_count,
                "has_extra_endpoint_slice": has_extra_endpoint_slice,
                "actual_spacing_min_mm": round(actual_spacing_min, 3),
                "actual_spacing_max_mm": round(actual_spacing_max, 3),
                "has_nonuniform_spacing": has_nonuniform_spacing,
                "target_start": target_start_data,
                "target_end": target_end_data,
                "target_length_mm": round(target_length, 3),
                "target_shorter_than_thickness": target_shorter_than_thickness,
                "first_coverage_start": (
                    first_coverage_start.to_dict() if first_coverage_start else None
                ),
                "last_coverage_end": (
                    last_coverage_end.to_dict() if last_coverage_end else None
                ),
                "leading_uncovered_mm": round(leading_uncovered, 3),
                "trailing_uncovered_mm": round(trailing_uncovered, 3),
                "slice_thickness": slice_thickness,
                "slice_spacing": slice_spacing,
                "spacing_minus_thickness": round(spacing_minus_thickness, 3),
                "calculation_note": calculation_note,
            })
        return diagnostics

    @staticmethod
    def _dict_to_point(point: Dict[str, Any]):
        """把 JSON 中的 x/y/z 字典转换成 Point3D，用于诊断距离计算。"""

        from src.models import Point3D

        return Point3D(
            x=float(point["x"]),
            y=float(point["y"]),
            z=float(point["z"]),
        )

    def _write_slice_plan_overlay(
        self,
        context: ProcessingContext,
        localization_result: Dict[str, Any],
        plans: list[Dict[str, Any]],
    ) -> tuple[str, str]:
        """
        写出第四步切割线调试 overlay。

        overlay 与第三步使用同一套 NIfTI 调试思路：参考图来自第三步声明的
        seg_for_reconstruction_path，输出图与参考图同 size/spacing/origin/direction。
        数值约定为：
        - plan_index * 10 + 1：该 plan 的中心路径线。
        - plan_index * 10 + 2：该 plan 已展开切片的中心点。
        - plan_index * 10 + 3：每张切片在该平面内的十字引导线。

        这个文件只用于人工核对“线画得对不对”，不作为第五步执行输入。
        """

        reference_path = localization_result.get("seg_for_reconstruction_path")
        if not reference_path:
            return ""

        reference_path = Path(reference_path)
        if not reference_path.exists():
            return ""

        reference_image = sitk.ReadImage(str(reference_path))
        reference_array = sitk.GetArrayFromImage(reference_image)
        overlay_array = np.zeros(reference_array.shape, dtype=np.uint16)

        for plan_index, plan in enumerate(plans, start=1):
            path_value = plan_index * 10 + 1
            center_value = plan_index * 10 + 2
            slice_guide_value = plan_index * 10 + 3
            path_points = plan.get("path_points", [])
            for start_point, end_point in zip(path_points, path_points[1:]):
                self._paint_physical_line(
                    reference_image=reference_image,
                    overlay_array=overlay_array,
                    start_point=start_point,
                    end_point=end_point,
                    value=path_value,
                )

            for planned_slice in plan.get("slices", []):
                self._paint_planned_slice_guides(
                    reference_image=reference_image,
                    overlay_array=overlay_array,
                    planned_slice=planned_slice,
                    extent_mm=plan.get("extent_mm", {}),
                    value=slice_guide_value,
                )
                self._paint_physical_marker(
                    reference_image=reference_image,
                    overlay_array=overlay_array,
                    point=planned_slice["center"],
                    value=center_value,
                )

        output_path = Path(context.request.output_dir) / "metadata" / "切片线.nii.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(reference_image)
        sitk.WriteImage(overlay_image, str(output_path))
        return str(output_path)

    def _write_slice_plan_overview_overlay(
        self,
        context: ProcessingContext,
        localization_result: Dict[str, Any],
        plans: list[Dict[str, Any]],
    ) -> tuple[str, str]:
        """
        写出单层总览 overlay，方便人工检查整切范围。

        真实 3D overlay 会把每张横状位 slice 画在各自的 z 层上，临床语义
        正确但人工检查时会显得“东一块西一块”。总览 overlay 会选一个固定
        x 层，把同一组横状位 slice 的切割线都投到这个矢状位层上。
        """

        reference_path = localization_result.get("seg_for_reconstruction_path")
        if not reference_path:
            return "", ""

        reference_path = Path(reference_path)
        if not reference_path.exists():
            return "", ""

        reference_image = sitk.ReadImage(str(reference_path))
        native_overlay_array = np.zeros(
            tuple(reversed(reference_image.GetSize())),
            dtype=np.uint16,
        )
        uniform_reference = self._build_uniform_overview_reference(reference_image)
        uniform_overlay_array = np.zeros(
            tuple(reversed(uniform_reference.GetSize())),
            dtype=np.uint16,
        )

        for plan_index, plan in enumerate(plans, start=1):
            # 第一版先服务腰椎整切的横状位总览。其他状位后续可以按同一入口扩展。
            if plan.get("plane") != "横状位":
                continue
            overview_value = plan_index * 10 + 4
            thickness_value = plan_index * 10 + 5
            self._paint_axial_plan_overview(
                reference_image=reference_image,
                overlay_array=native_overlay_array,
                plan=plan,
                value=overview_value,
                thickness_value=thickness_value,
            )
            self._paint_axial_plan_overview(
                reference_image=uniform_reference,
                overlay_array=uniform_overlay_array,
                plan=plan,
                value=overview_value,
                thickness_value=thickness_value,
            )

        metadata_dir = Path(context.request.output_dir) / "metadata"
        output_path = metadata_dir / "切片总览.nii.gz"
        native_output_path = metadata_dir / "切片原始网格总览.nii.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_image = sitk.GetImageFromArray(uniform_overlay_array)
        overlay_image.CopyInformation(uniform_reference)
        sitk.WriteImage(overlay_image, str(output_path))
        native_overlay_image = sitk.GetImageFromArray(native_overlay_array)
        native_overlay_image.CopyInformation(reference_image)
        sitk.WriteImage(native_overlay_image, str(native_output_path))
        return str(output_path), str(native_output_path)

    @staticmethod
    def _build_uniform_overview_reference(reference_image: sitk.Image) -> sitk.Image:
        """创建 z=0.5mm 的同物理空间检查网格。"""

        spacing = reference_image.GetSpacing()
        size = reference_image.GetSize()
        uniform_z_spacing = 0.5
        physical_z_extent = (size[2] - 1) * spacing[2]
        uniform_z_size = int(math.ceil(physical_z_extent / uniform_z_spacing)) + 1
        uniform_reference = sitk.Image(
            int(size[0]),
            int(size[1]),
            uniform_z_size,
            sitk.sitkUInt16,
        )
        uniform_reference.SetSpacing((spacing[0], spacing[1], uniform_z_spacing))
        uniform_reference.SetOrigin(reference_image.GetOrigin())
        uniform_reference.SetDirection(reference_image.GetDirection())
        return uniform_reference

    def _paint_axial_plan_overview(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        plan: Dict[str, Any],
        value: int,
        thickness_value: int,
    ) -> None:
        """
        把一组横状位 slice 投到固定 x 层上，形成单层矢状位总览。

        总览图用不同标签画厚度覆盖带和输出切片中心线。
        """

        slices = plan.get("slices", [])
        if not slices:
            return

        extent_mm = plan.get("extent_mm", {})
        half_height = float(extent_mm.get("height", 0.0)) / 2.0
        if half_height <= 0:
            return

        center_indices = [
            reference_image.TransformPhysicalPointToContinuousIndex((
                float(planned_slice["center"]["x"]),
                float(planned_slice["center"]["y"]),
                float(planned_slice["center"]["z"]),
            ))
            for planned_slice in slices
        ]
        overview_x_index = int(round(float(np.median([
            index_xyz[0] for index_xyz in center_indices
        ]))))
        half_height_voxels = half_height / reference_image.GetSpacing()[1]
        half_thickness_voxels = (
            float(plan.get("slice_thickness", 0.0))
            / (2.0 * reference_image.GetSpacing()[2])
        )

        slice_thickness = float(plan.get("slice_thickness", 0.0))
        center_tick_half_voxels = min(
            half_height_voxels * 0.25,
            5.0 / reference_image.GetSpacing()[1],
        )
        for center_index in center_indices:
            if slice_thickness > 0:
                self._paint_axial_overview_index_band(
                    overlay_array=overlay_array,
                    overview_x_index=overview_x_index,
                    center_index=center_index,
                    half_height_voxels=half_height_voxels,
                    half_thickness_voxels=half_thickness_voxels,
                    value=thickness_value,
                )

        # 中心线最后画，避免被同一张或相邻切片的厚度带覆盖。
        for center_index in center_indices:
            self._paint_axial_overview_index_line(
                overlay_array=overlay_array,
                overview_x_index=overview_x_index,
                center_y_index=center_index[1],
                z_index=int(round(center_index[2])),
                half_height_voxels=center_tick_half_voxels,
                value=value,
            )

    def _paint_axial_overview_index_band(
        self,
        overlay_array: np.ndarray,
        overview_x_index: int,
        center_index: tuple[float, float, float],
        half_height_voxels: float,
        half_thickness_voxels: float,
        value: int,
    ) -> None:
        """在图像轴索引空间按物理层厚填充总览覆盖带。"""

        z_start = int(math.ceil(center_index[2] - half_thickness_voxels - 1e-9))
        z_end = int(math.ceil(center_index[2] + half_thickness_voxels - 1e-9)) - 1
        for z_index in range(z_start, z_end + 1):
            self._paint_axial_overview_index_line(
                overlay_array=overlay_array,
                overview_x_index=overview_x_index,
                center_y_index=center_index[1],
                z_index=z_index,
                half_height_voxels=half_height_voxels,
                value=value,
            )

    def _paint_axial_overview_index_line(
        self,
        overlay_array: np.ndarray,
        overview_x_index: int,
        center_y_index: float,
        z_index: int,
        half_height_voxels: float,
        value: int,
    ) -> None:
        """在固定图像 x 索引层上画一条横状位总览线。"""

        y_start = int(round(center_y_index - half_height_voxels))
        y_end = int(round(center_y_index + half_height_voxels))
        if not 0 <= z_index < overlay_array.shape[0]:
            return
        if not 0 <= overview_x_index < overlay_array.shape[2]:
            return
        y_start = max(y_start, 0)
        y_end = min(y_end, overlay_array.shape[1] - 1)
        if y_start <= y_end:
            overlay_array[z_index, y_start:y_end + 1, overview_x_index] = value

    @staticmethod
    def _physical_point_to_array_index(
        reference_image: sitk.Image,
        point: Dict[str, float],
    ) -> tuple[int, int, int]:
        """把 x/y/z 物理坐标转换成 numpy 数组使用的 z/y/x 索引。"""

        index_xyz = reference_image.TransformPhysicalPointToIndex((
            float(point["x"]),
            float(point["y"]),
            float(point["z"]),
        ))
        return int(index_xyz[2]), int(index_xyz[1]), int(index_xyz[0])

    @staticmethod
    def _is_inside_array(array: np.ndarray, z: int, y: int, x: int) -> bool:
        """判断 z/y/x 是否落在 overlay 数组内。"""

        z_size, y_size, x_size = array.shape
        return 0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size

    def _paint_physical_line(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        start_point: Dict[str, float],
        end_point: Dict[str, float],
        value: int,
    ) -> None:
        """在 overlay 上画一条连接两个物理坐标点的体素线。"""

        start_zyx = self._physical_point_to_array_index(reference_image, start_point)
        end_zyx = self._physical_point_to_array_index(reference_image, end_point)
        dz = end_zyx[0] - start_zyx[0]
        dy = end_zyx[1] - start_zyx[1]
        dx = end_zyx[2] - start_zyx[2]
        steps = max(abs(dz), abs(dy), abs(dx), 1)

        for step in range(steps + 1):
            z = int(round(start_zyx[0] + dz * step / steps))
            y = int(round(start_zyx[1] + dy * step / steps))
            x = int(round(start_zyx[2] + dx * step / steps))
            if self._is_inside_array(overlay_array, z, y, x):
                overlay_array[z, y, x] = value

    def _paint_planned_slice_guides(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        planned_slice: Dict[str, Any],
        extent_mm: Dict[str, float],
        value: int,
    ) -> None:
        """
        在单张 slice 所在平面内画十字引导线。

        中心路径线在横状位上常常只像一个点，不利于人工检查。这里用
        row_dir/col_dir 和 extent_mm 画两条穿过中心的平面内线，打开
        overlay 时能直接看到“每张图切在哪里”。
        """

        center = planned_slice["center"]
        row_dir = planned_slice["row_dir"]
        col_dir = planned_slice["col_dir"]
        half_width = float(extent_mm.get("width", 0.0)) / 2.0
        half_height = float(extent_mm.get("height", 0.0)) / 2.0

        if half_width > 0:
            self._paint_direction_line(
                reference_image=reference_image,
                overlay_array=overlay_array,
                center=center,
                direction=row_dir,
                half_length=half_width,
                value=value,
            )
        if half_height > 0:
            self._paint_direction_line(
                reference_image=reference_image,
                overlay_array=overlay_array,
                center=center,
                direction=col_dir,
                half_length=half_height,
                value=value,
            )

    def _paint_direction_line(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        center: Dict[str, float],
        direction: Dict[str, float],
        half_length: float,
        value: int,
    ) -> None:
        """沿某个方向从中心点向两侧画一条物理长度线段。"""

        start_point = {
            "x": center["x"] - direction["x"] * half_length,
            "y": center["y"] - direction["y"] * half_length,
            "z": center["z"] - direction["z"] * half_length,
        }
        end_point = {
            "x": center["x"] + direction["x"] * half_length,
            "y": center["y"] + direction["y"] * half_length,
            "z": center["z"] + direction["z"] * half_length,
        }
        self._paint_physical_line(
            reference_image=reference_image,
            overlay_array=overlay_array,
            start_point=start_point,
            end_point=end_point,
            value=value,
        )

    def _paint_physical_marker(
        self,
        reference_image: sitk.Image,
        overlay_array: np.ndarray,
        point: Dict[str, float],
        value: int,
    ) -> None:
        """在 overlay 上标出一张 slice 的中心点，画成小十字方便查看。"""

        center_z, center_y, center_x = self._physical_point_to_array_index(
            reference_image,
            point,
        )
        offsets = [
            (0, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        for dz, dy, dx in offsets:
            z = center_z + dz
            y = center_y + dy
            x = center_x + dx
            if self._is_inside_array(overlay_array, z, y, x):
                overlay_array[z, y, x] = value
