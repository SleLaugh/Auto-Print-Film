#!/src/services/stage_reconstruction.py
# -*- coding: utf-8 -*-

"""
第七阶段：重建与输出
MPR多平面重建与DICOM生成
按第六阶段 SlicePlan 执行直线 plane_stack 重建

每个 plan 分成两个明确步骤：先串行完成全部 MPR HU 图像采样，再按窗位并行完成
按窗显示的切片和参数输出任务。不同窗位共享同一份 MPR 结果，不会重复重采样；GDCM 原生
写盘由 DICOM 模块统一串行保护，避免多线程写入触发原生内存竞态。
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import ReconstructionStageResult, WindowPreset
from src.reconstruction import CurvedStackResampler, PlaneStackResampler
from src.reconstruction.dicom import (
    new_uid,
    read_source_dicom_metadata,
    write_ct_dicom,
    write_localizer_dicom,
)
from src.reconstruction.windowing import apply_window_to_image, unique_window_presets
from src.reconstruction.framing_context import FramingContext, load_framing_context
from src.reconstruction.symmetry_framing import resolve_symmetry_framing_plane
from src.reconstruction.spine_localizer import (
    SpineLocalizerSurface,
    build_spine_localizer_surface,
)
from src.reconstruction.framing import (
    TargetFramer,
    frame_curved_slice,
    curved_group_bounds,
    prepare_interspace_framing,
)
from src.utils import DebugLog, is_debug_file_enabled
from src.runtime import window_worker_count


@dataclass(frozen=True)
class _LocalizerRender:
    """保存定位图原始背景及与其逐像素对齐的线条覆盖率。"""

    background_image: sitk.Image
    line_alpha: np.ndarray
    legacy_line_value: float
    background_kind: str = "orthogonal_plane"


class ReconstructionStage(BaseStage):
    """执行切片计划，并按窗位输出派生 CT DICOM 序列。"""

    _LOCALIZER_DRAW_AXES = {
        "横状位": (2, 1, 0),
        "矢状位": (0, 2, 1),
        "冠状位": (1, 2, 0),
    }
    _LOCALIZER_FIXED_AXES = {
        "矢状位": 0,
        "冠状位": 1,
        "横状位": 2,
    }
    _LOCALIZER_SUBPIXEL_SAMPLES_PER_PIXEL = 4
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化重建阶段并读取窗位并行配置。

        Args:
            config: 阶段配置。`parallel` 控制是否并行处理不同窗位；
                `max_window_workers` 限制同一 plan 的最大窗位任务数，优先于
                AUTOPRINTFILM_WINDOW_WORKERS，未配置时默认为 2。
        """
        super().__init__(name="reconstruction", config=config)
        self.parallel = config.get('parallel', True) if config else True
        """是否为同一 plan 的不同窗位启用线程池。"""
        self.max_window_workers = window_worker_count(config)
        """窗位线程池上限；worker 数不超过窗位数量，不包含原生辅助线程。"""
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        执行第六阶段的全部平面切片计划并输出分窗位 DICOM 序列。

        状位与计划串行执行，同一计划内的不同窗位可并行；每个重建区域都会生成
        自己的数字编号切片和 Localizer 定位图。
        """

        request = context.request
        
        loc_result = context.get_result('localization')
        plan_result = context.get_result('planning')
        
        if not loc_result:
            raise ValueError("前置阶段 'localization' 结果不存在")
        if not plan_result:
            raise ValueError("前置阶段 'planning' 结果不存在")
        
        # 第三阶段明确选定后续使用的 raw 图像，第四阶段把稳定路径转交给重建。
        # 未回正时是原始 raw_0000.nii.gz；发生回正时才指向校正候选。
        # corrected_raw_path 是调试 JSON 中保留的同义摘要字段。
        image_path = (
            loc_result.get('raw_for_reconstruction_path')
            or loc_result.get('corrected_raw_path')
        )
        if not image_path:
            raise ValueError("定位结果缺少 raw_for_reconstruction_path/corrected_raw_path")

        plans = plan_result.get("plans", [])
        if not isinstance(plans, list):
            raise ValueError("第六阶段输出 plans 必须是列表")
        for plan in plans:
            self._validate_plan(plan)
        
        DebugLog("=" * 60)
        DebugLog("【第七阶段】重建与输出 - MPR重建与DICOM生成")
        DebugLog(f"  输入图像: {image_path}")
        DebugLog(f"  输出目录: {request.output_dir}")
        DebugLog(f"  规划组数: {len(plans)}")
        DebugLog("=" * 60)

        output_root = Path(request.output_dir)
        if not plans:
            result = self._build_plan_summary(plans, output_root)
            context.set_stage_result(ReconstructionStageResult(payload=result))
            DebugLog("  没有可执行的切片计划")
            return context

        image = sitk.ReadImage(str(image_path))
        segmentation_path = (
            loc_result.get("seg_for_reconstruction_path")
            or loc_result.get("corrected_seg_path")
            or loc_result.get("seg_path")
        )
        if not segmentation_path:
            raise ValueError("整组居中需要 seg_for_reconstruction_path")
        segmentation = sitk.ReadImage(str(segmentation_path))
        anatomical_analysis = context.get_result("anatomical_analysis") or {}
        # 与第六阶段消费相同的展平结果。脑区仍是独立切割目标，骨性支持
        # 只能在最终脑 SI 范围及计划允许的颅顶余量内取景，不通过坐标表扩展层组。
        loc_result = {**loc_result, **anatomical_analysis}
        framing_context = load_framing_context(
            loc_result.get("dataset_json_path"), loc_result.get("coords", {}),
        ) if any(plan.get("fill_percent") for plan in plans) else None
        merge_framing_support = False
        if request.slice_part.value == "颅脑" and anatomical_analysis.get("framing_label_ids"):
            frame = anatomical_analysis["anatomical_geometry"]["global_frame"]
            framing_context = FramingContext(
                label_ids=tuple(anatomical_analysis["framing_label_ids"]),
                axis_xyz=tuple(frame["superior_inferior_axis"][axis] for axis in "xyz"),
            )
            merge_framing_support = True
        symmetry_planes = {
            str(plan.get("plan_id", "")): plane
            for plan in plans
            if (plane := resolve_symmetry_framing_plane(plan, anatomical_analysis)) is not None
        }
        preprocessing_result = context.get_result("preprocessing") or {}
        source_metadata = dict(preprocessing_result.get("source_dicom_metadata") or {})
        if not source_metadata:
            source_metadata = read_source_dicom_metadata(request.dicom_dir)
        study_uid = source_metadata.get("0020|000d") or new_uid()
        plan_results = []
        # 状位之间严格串行：先完成某个状位下的所有计划及窗位写出，再进入
        # 下一个状位。这样未来增加三平面后不会同时保留多套状位 MPR 图像。
        plans_by_plane = self._group_plans_by_plane(plans)
        for plane, plane_plans in plans_by_plane.items():
            DebugLog(f"  开始处理状位: {plane}, 共 {len(plane_plans)} 个 plan")
            interspace_groups = {}
            for plan in plane_plans:
                key = self._interspace_framing_key(plan)
                if key is not None:
                    interspace_groups.setdefault(key, []).append(plan)
            shared_geometries = {}
            for key, group in interspace_groups.items():
                geometry = prepare_interspace_framing(
                    segmentation, loc_result.get("coords", {}), group,
                    context=framing_context,
                    symmetry_planes=symmetry_planes,
                )
                shared_geometries[key] = geometry
                DebugLog(
                    f"  [椎间隙共同取景] 配置={key}, 间隙数={len(group)}, "
                    f"矩阵={geometry.size_xy}, 像素间距mm={geometry.spacing_xy}, "
                    f"统一倍率={geometry.scale:.6f}"
                )
            for plan in plane_plans:
                DebugLog(f"  执行 plan: {plan['plan_id']}")
                plan_results.append(
                    self._execute_plan(
                        image=image,
                        plan=plan,
                        output_root=output_root,
                        study_uid=study_uid,
                        source_metadata=source_metadata,
                        custom_window_presets=getattr(request, "window_presets", None),
                        target_framer=TargetFramer(
                            segmentation,
                            loc_result.get("coords", {}),
                            plan,
                            context=(
                                FramingContext(
                                    label_ids=framing_context.label_ids,
                                    axis_xyz=framing_context.axis_xyz,
                                    upper_extension_mm=plan.get("framing_superior_extension_mm", 0.0),
                                ) if merge_framing_support else framing_context
                            ),
                            merge_context_into_target=merge_framing_support,
                            symmetry_plane=symmetry_planes.get(str(plan.get("plan_id", ""))),
                            shared_geometry=shared_geometries.get(
                                self._interspace_framing_key(plan),
                            ),
                        ),
                        localization_result=loc_result,
                        anatomical_analysis=anatomical_analysis,
                    )
                )
            DebugLog(f"  状位处理完成: {plane}")

        result = self._build_plan_summary(plan_results, output_root)
        result["outputs"] = plan_results
        
        context.set_stage_result(ReconstructionStageResult(payload=result))
        
        DebugLog("  第七阶段完成")
        DebugLog("=" * 60)
        
        return context

    def _execute_plan(
        self,
        image: sitk.Image,
        plan: Dict[str, Any],
        output_root: Path,
        study_uid: str,
        source_metadata: Dict[str, str],
        localizer_image: sitk.Image | None = None,
        custom_window_presets: list[WindowPreset] | None = None,
        target_framer: TargetFramer | None = None,
        localization_result: Dict[str, Any] | None = None,
        anatomical_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        执行一个切片计划。

        窗位按请求级覆盖列表优先、默认配置回退解析。第一步把计划中的所有位置
        采样成 HU 图像；第二步再按窗位处理整套采样结果。先完整采样能保证窗位
        线程不触碰重采样器，也符合“先采集、后显示处理”的流程。
        """
        localizer_render: _LocalizerRender | None = None
        if localizer_image is not None:
            # 保留内部测试或调用方传入预构建定位图的兼容入口。该图可能已经烧录
            # 线条，因此这里只把它作为无额外覆盖掩膜的完整背景继续写出。
            localizer_render = _LocalizerRender(
                background_image=localizer_image,
                line_alpha=np.zeros(
                    self._localizer_array_2d(localizer_image).shape,
                    dtype=np.float32,
                ),
                legacy_line_value=float(np.max(
                    self._localizer_array_2d(localizer_image)
                )),
            )

        windows = unique_window_presets(
            plan["window_presets"],
            custom_presets=custom_window_presets,
        )
        series_uids = {window.name: new_uid() for window in windows}
        localizer_series_uids = {window.name: new_uid() for window in windows}
        region_name = self._plan_region_name(plan)
        output_plane = str(plan.get("output_plane") or plan["plane"])
        output_dirs = {
            window.name: output_root
            / output_plane
            / window.name
            / region_name
            if region_name
            else output_root / output_plane / window.name
            for window in windows
        }
        for output_dir in output_dirs.values():
            output_dir.mkdir(parents=True, exist_ok=True)

        resampler = self._resampler_for_plan(plan)

        # 第一步：按计划完成 HU 厚层采样与取景，保留未窗化的 float32 图像。
        # 所有采样完成之前不会开始任何窗宽窗位映射。
        DebugLog(f"    [1] MPR采样: 共 {len(plan['slices'])} 张")
        sampling_start = time.perf_counter()
        sampled_slices = []
        curved_masks = []
        slowest_slice_index = None
        slowest_slice_elapsed = 0.0
        for planned_slice in plan["slices"]:
            single_start = time.perf_counter()
            sampling_plan = plan
            if target_framer is not None:
                if plan.get("geometry_type") == "curved_stack":
                    curved_masks.append(resampler.sample_slice(
                        target_framer.mask, plan, planned_slice, is_mask=True,
                    ))
                else:
                    sampling_plan, planned_slice = target_framer.plane_geometry(plan, planned_slice)
            sampled_image = resampler.sample_slice(image, sampling_plan, planned_slice)
            single_elapsed = time.perf_counter() - single_start
            if single_elapsed > slowest_slice_elapsed:
                slowest_slice_elapsed = single_elapsed
                slowest_slice_index = planned_slice.get("index")
            sampled_slices.append((planned_slice, sampled_image))
        if curved_masks:
            group_bounds = curved_group_bounds(curved_masks)
            framed_slices = []
            for (planned_slice, sampled_image), mask in zip(sampled_slices, curved_masks):
                sampled_image, framing = frame_curved_slice(
                    sampled_image, mask, plan.get("fill_percent"), group_bounds,
                )
                framing["context_label_ids"] = list(target_framer.context_label_ids)
                framed_slices.append(({**planned_slice, "framing": framing}, sampled_image))
            sampled_slices = framed_slices
        if target_framer is not None and sampled_slices:
            framing = sampled_slices[0][0]["framing"]
            scale_target_size = framing.get("shared_target_size_mm")
            if scale_target_size is None:
                scale_target_size = framing.get("framing_target_size_mm", framing["target_size_mm"])
            DebugLog(
                f"    [整组取景] plan={plan['plan_id']}, FillPercent={plan.get('fill_percent')}, "
                f"目标范围mm={framing['target_size_mm']}, "
                f"倍率依据mm={scale_target_size}, "
                f"上下文标签={framing['context_label_ids']}, 统一倍率={framing['scale']:.6f}"
            )
        sampling_elapsed = time.perf_counter() - sampling_start
        average_elapsed = (
            sampling_elapsed / len(sampled_slices)
            if sampled_slices
            else 0.0
        )
        DebugLog(
            "    [重建性能] MPR采样完成: "
            f"plan={plan.get('plan_id', '')}, "
            f"geometry={plan.get('geometry_type', 'plane_stack')}, "
            f"数量={len(sampled_slices)}, "
            f"层厚={float(plan.get('slice_thickness', 0.0)):.3f}mm, "
            f"层间距={float(plan.get('slice_spacing', 0.0)):.3f}mm, "
            f"总耗时={sampling_elapsed:.3f}s, "
            f"平均={average_elapsed:.3f}s/张, "
            f"最慢slice={slowest_slice_index}, "
            f"最慢耗时={slowest_slice_elapsed:.3f}s"
        )
        DebugLog("    [1] MPR采样完成")

        if localizer_image is None and isinstance(image, sitk.Image):
            localizer_start = time.perf_counter()
            localizer_render = self._build_plan_localizer_render(
                image, plan,
                localization_result=localization_result,
                anatomical_analysis=anatomical_analysis,
            )
            DebugLog(
                "    [重建性能] Localizer生成完成: "
                f"plan={plan.get('plan_id', '')}, "
                f"has_localizer={localizer_render is not None}, "
                f"耗时={time.perf_counter() - localizer_start:.3f}s"
            )

        self._write_plan_windows(
            sampled_slices=sampled_slices,
            windows=windows,
            output_dirs=output_dirs,
            plan=plan,
            series_uids=series_uids,
            study_uid=study_uid,
            source_metadata=source_metadata,
            localizer_render=localizer_render,
            localizer_series_uids=localizer_series_uids,
        )

        return {
            "plan_id": plan["plan_id"],
            "config_id": plan.get("config_id", ""),
            "plane": output_plane,
            "source_plane": plan["plane"],
            "slice_type": plan["slice_type"],
            "group_name": plan.get("group_name", ""),
            "geometry_type": plan.get("geometry_type", "plane_stack"),
            "slice_thickness": float(plan["slice_thickness"]),
            "slice_spacing": float(plan["slice_spacing"]),
            "slice_count": len(plan["slices"]),
            "slice_image_count": len(plan["slices"]),
            "sampling_details": dict(plan.get("sampling_details") or {}),
            "framing": [
                {"index": planned_slice["index"], **planned_slice["framing"]}
                for planned_slice, _ in sampled_slices if "framing" in planned_slice
            ],
            "image_count": len(plan["slices"]) * len(windows),
            "window_outputs": {
                name: str(path)
                for name, path in output_dirs.items()
            },
            "localizer_only": False,
            "localizer_background": (
                localizer_render.background_kind if localizer_render is not None else None
            ),
        }

    def _write_plan_windows(
        self,
        *,
        sampled_slices: list[tuple[dict[str, Any], sitk.Image]],
        windows: list[Any],
        output_dirs: dict[str, Path],
        plan: Dict[str, Any],
        series_uids: dict[str, str],
        study_uid: str,
        source_metadata: Dict[str, str],
        localizer_render: _LocalizerRender | None,
        localizer_series_uids: dict[str, str],
    ) -> None:
        """
        并行写正式切片，再按同一份冻结几何串行写各窗 Localizer。

        每个任务写独立目录并使用独立序列 UID，因此没有文件名或序列状态冲突；
        底层 GDCM 写盘仍通过 DICOM 模块的进程内锁串行执行，线程池只并行安全的
        窗位计算和正式切片输出准备。Localizer 共享同一张带三维标签的参考图，
        必须在主线程冻结标签并串行写出，避免 SimpleITK 元数据在多线程复制时
        偶发退回二维默认方向。
        """

        localizer_geometry_metadata: dict[str, str] | None = None
        if localizer_render is not None:
            localizer_geometry_metadata = self._localizer_geometry_metadata(
                localizer_render.background_image,
            )

        if self.parallel and len(windows) > 1:
            worker_count = min(len(windows), self.max_window_workers)
            DebugLog(
                f"    [2] 窗位并行处理: {len(windows)} 个窗位, "
                f"{worker_count} 个线程"
            )
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="mpr-window",
            ) as executor:
                futures = [
                    executor.submit(
                        self._write_window_series,
                        sampled_slices=sampled_slices,
                        window=window,
                        output_dir=output_dirs[window.name],
                        plan=plan,
                        series_uid=series_uids[window.name],
                        study_uid=study_uid,
                        source_metadata=source_metadata,
                    )
                    for window in windows
                ]
                # 显式读取每个 Future，确保线程异常回到重建阶段并让 Pipeline 失败。
                for future in futures:
                    future.result()
        else:
            DebugLog(f"    [2] 窗位串行处理: {len(windows)} 个窗位")
            for window in windows:
                self._write_window_series(
                    sampled_slices=sampled_slices,
                    window=window,
                    output_dir=output_dirs[window.name],
                    plan=plan,
                    series_uid=series_uids[window.name],
                    study_uid=study_uid,
                    source_metadata=source_metadata,
                )

        if localizer_render is None or localizer_geometry_metadata is None:
            return

        DebugLog(
            "    [3] Localizer冻结几何并串行写出: "
            f"plan={plan.get('plan_id', '')}, "
            f"windows={len(windows)}, "
            f"position={localizer_geometry_metadata['0020|0032']}, "
            f"orientation={localizer_geometry_metadata['0020|0037']}"
        )
        for window in windows:
            self._write_window_localizer(
                localizer_render=localizer_render,
                geometry_metadata=localizer_geometry_metadata,
                window=window,
                output_dir=output_dirs[window.name],
                series_uid=localizer_series_uids[window.name],
                study_uid=study_uid,
                source_metadata=source_metadata,
                plan=plan,
            )

    @classmethod
    def _build_plan_localizer(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
        *,
        localization_result: Dict[str, Any] | None = None,
        anatomical_analysis: Dict[str, Any] | None = None,
    ) -> sitk.Image | None:
        """返回兼容既有检查入口的、已烧录定位线的 HU 图像。"""

        render = cls._build_plan_localizer_render(
            image, plan,
            localization_result=localization_result,
            anatomical_analysis=anatomical_analysis,
        )
        if render is None:
            return None
        return cls._compose_legacy_localizer_image(render)

    @classmethod
    def _build_plan_localizer_render(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
        *,
        localization_result: Dict[str, Any] | None = None,
        anatomical_analysis: Dict[str, Any] | None = None,
    ) -> _LocalizerRender | None:
        """生成定位图背景与亚像素线掩膜，不参与正式切片重建。"""

        surface = build_spine_localizer_surface(
            image, plan, localization_result, anatomical_analysis,
        )
        if plan.get("geometry_type") == "curved_stack":
            return cls._build_curved_plan_localizer_render(
                image, plan, localizer_surface=surface,
            )
        return cls._build_plane_plan_localizer_render(
            image, plan, localizer_surface=surface,
        )

    @classmethod
    def _build_plane_plan_localizer_render(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
        *,
        localizer_surface: SpineLocalizerSurface | None = None,
    ) -> _LocalizerRender | None:
        """在固定原图网格上绘制切片平面交线，仅按背景边界裁线。

        定位线表示平面位置与倾角，不表示正式切片的有限取景矩形。
        背景参考层、物理大小和线长均不能随 FillPercent 改变。
        """

        axes = cls._LOCALIZER_DRAW_AXES.get(str(plan.get("plane", "")))
        if axes is None:
            return None
        stack_axis, line_axis, fixed_axis = cls._resolve_localizer_axes(
            image,
            plan,
            axes,
        )
        center_indices = [
            image.TransformPhysicalPointToContinuousIndex((
                float(planned_slice["center"]["x"]),
                float(planned_slice["center"]["y"]),
                float(planned_slice["center"]["z"]),
            ))
            for planned_slice in plan.get("slices", [])
        ]
        if not center_indices:
            return None
        localizer_reference = plan.get("localizer_reference")
        if localizer_reference:
            reference_index = image.TransformPhysicalPointToContinuousIndex((
                float(localizer_reference["x"]),
                float(localizer_reference["y"]),
                float(localizer_reference["z"]),
            ))
            fixed_index = int(round(float(reference_index[fixed_axis])))
        else:
            # 没有目标参考点时，使用切片中心中位数作为确定性回退。
            fixed_index = int(round(float(np.median([
                index_xyz[fixed_axis] for index_xyz in center_indices
            ]))))
        fixed_index = min(max(fixed_index, 0), image.GetSize()[fixed_axis] - 1)
        if localizer_surface is not None:
            representative_slice = localizer_surface.sample(image)
        else:
            roi_index = [0, 0, 0]
            roi_size = list(image.GetSize())
            roi_index[fixed_axis] = fixed_index
            roi_size[fixed_axis] = 1
            representative_slice = sitk.RegionOfInterest(image, roi_size, roi_index)
            representative_slice = sitk.Cast(representative_slice, sitk.sitkFloat32)
        line_alpha_array = np.zeros(
            sitk.GetArrayViewFromImage(representative_slice).shape,
            dtype=np.float32,
        )
        source_array = sitk.GetArrayViewFromImage(image)
        source_min = float(np.min(source_array))
        source_max = float(np.max(source_array))
        intensity_span = max(source_max - source_min, 1.0)
        line_value = source_max + intensity_span

        for planned_slice in plan.get("slices", []):
            if localizer_surface is not None:
                # 曲面背景的前后深度随行变化；斜切 MPR 不能继续使用旧平面交线。
                for indices in localizer_surface.plane_intersections(image, planned_slice):
                    cls._paint_antialiased_polyline(
                        line_alpha_array, indices,
                        display_axes=(stack_axis, line_axis), fixed_axis=fixed_axis,
                    )
                continue
            segment = cls._slice_localizer_intersection(
                planned_slice=planned_slice,
                background_image=representative_slice,
                fixed_axis=fixed_axis,
            )
            if segment is None:
                continue
            start_point, end_point = segment
            start_xyz = np.asarray(
                image.TransformPhysicalPointToContinuousIndex(tuple(start_point)),
                dtype=float,
            )
            end_xyz = np.asarray(
                image.TransformPhysicalPointToContinuousIndex(tuple(end_point)),
                dtype=float,
            )
            cls._paint_antialiased_polyline(
                line_alpha_array,
                np.stack([start_xyz, end_xyz]),
                display_axes=(stack_axis, line_axis),
                fixed_axis=fixed_axis,
            )

        return cls._finalize_localizer_render(
            image=image,
            representative_slice=representative_slice,
            line_alpha_array=line_alpha_array,
            fixed_axis=fixed_axis,
            legacy_line_value=line_value,
            localizer_surface=localizer_surface,
        )

    @classmethod
    def _build_curved_plan_localizer(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
    ) -> sitk.Image | None:
        """返回兼容既有检查入口的、已烧录 CPR 定位线的 HU 图像。"""

        render = cls._build_curved_plan_localizer_render(
            image, plan, localizer_surface=build_spine_localizer_surface(image, plan),
        )
        if render is None:
            return None
        return cls._compose_legacy_localizer_image(render)

    @classmethod
    def _build_curved_plan_localizer_render(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
        *,
        localizer_surface: SpineLocalizerSurface | None = None,
    ) -> _LocalizerRender | None:
        """投影实际 CPR 中心曲线，并生成亚像素覆盖掩膜。"""

        curve_points = plan.get("curve_points") or []
        if len(curve_points) < 2:
            return None
        localizer_plane = str(plan.get("localizer_plane") or plan.get("plane"))
        try:
            fixed_axis = cls._LOCALIZER_FIXED_AXES[localizer_plane]
        except KeyError as error:
            raise ValueError(f"不支持的定位图参考状位: {localizer_plane}") from error
        display_axes = [axis for axis in range(3) if axis != fixed_axis]

        fixed_index = cls._curved_localizer_fixed_index(image, plan, fixed_axis)
        if localizer_surface is not None:
            representative_slice = localizer_surface.sample(image)
        else:
            roi_index = [0, 0, 0]
            roi_size = list(image.GetSize())
            roi_index[fixed_axis] = fixed_index
            roi_size[fixed_axis] = 1
            representative_slice = sitk.RegionOfInterest(image, roi_size, roi_index)
            representative_slice = sitk.Cast(representative_slice, sitk.sitkFloat32)
        line_alpha_array = np.zeros(
            sitk.GetArrayViewFromImage(representative_slice).shape,
            dtype=np.float32,
        )
        source_array = sitk.GetArrayViewFromImage(image)
        source_min = float(np.min(source_array))
        source_max = float(np.max(source_array))
        line_value = source_max + max(source_max - source_min, 1.0)

        projected_curves = cls._curved_localizer_projected_curves(
            plan=plan,
            fixed_axis=fixed_axis,
        )
        coordinate_transform_start = time.perf_counter()
        for projected_points in projected_curves.values():
            projected_indices = cls._physical_points_to_continuous_indices(
                image,
                projected_points,
            )
            if len(projected_indices) < 2:
                # 定位线属于显示附属物；退化投影保留干净背景即可，不能让它
                # 阻断正式 CPR 切片的重采样和写出。
                continue
            cls._paint_antialiased_polyline(
                line_alpha_array,
                projected_indices,
                display_axes=(display_axes[0], display_axes[1]),
                fixed_axis=fixed_axis,
            )
        DebugLog(
            "    [CPR定位图性能] "
            f"plan={plan.get('plan_id', '')}, "
            f"curves={len(projected_curves)}, "
            f"points={sum(len(points) for points in projected_curves.values())}, "
            "coordinate_transform=batch_affine, "
            "rasterizer=subpixel_bilinear, "
            f"耗时={time.perf_counter() - coordinate_transform_start:.3f}s"
        )

        return cls._finalize_localizer_render(
            image=image,
            representative_slice=representative_slice,
            line_alpha_array=line_alpha_array,
            fixed_axis=fixed_axis,
            legacy_line_value=line_value,
            localizer_surface=localizer_surface,
        )

    @classmethod
    def _paint_antialiased_segment(
        cls,
        line_alpha_array: np.ndarray,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        *,
        display_axes: tuple[int, int],
        fixed_axis: int,
    ) -> None:
        """绘制单段抗锯齿线；主要供独立几何测试和短交线使用。"""

        cls._paint_antialiased_polyline(
            line_alpha_array,
            np.stack([start_xyz, end_xyz]),
            display_axes=display_axes,
            fixed_axis=fixed_axis,
        )

    @classmethod
    def _paint_antialiased_polyline(
        cls,
        line_alpha_array: np.ndarray,
        points_xyz: np.ndarray,
        *,
        display_axes: tuple[int, int],
        fixed_axis: int,
    ) -> None:
        """整条曲线一次批量生成亚像素采样，再按双线性覆盖率落点。"""

        points = np.asarray(points_xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("定位线必须至少包含两个三维连续索引")
        if not np.all(np.isfinite(points)):
            raise ValueError("定位线索引必须是有限数值")

        first_axis, second_axis = display_axes
        sampled_segments: list[np.ndarray] = []
        for start, end in zip(points, points[1:]):
            difference = end - start
            displayed_length = max(
                abs(float(difference[first_axis])),
                abs(float(difference[second_axis])),
            )
            intervals = max(
                1,
                int(math.ceil(
                    displayed_length
                    * cls._LOCALIZER_SUBPIXEL_SAMPLES_PER_PIXEL
                )),
            )
            # 每段不重复加入终点；曲线最终端点在循环后统一补入。
            ratios = np.arange(intervals, dtype=float) / intervals
            sampled_segments.append(
                start[np.newaxis, :] + ratios[:, np.newaxis] * difference
            )
        sampled_segments.append(points[-1:])
        samples = np.concatenate(sampled_segments, axis=0)
        first_values = samples[:, first_axis]
        second_values = samples[:, second_axis]
        first_floor = np.floor(first_values).astype(np.int64)
        second_floor = np.floor(second_values).astype(np.int64)
        first_fraction = first_values - first_floor
        second_fraction = second_values - second_floor
        size_xyz = np.asarray(line_alpha_array.shape[::-1], dtype=np.int64)

        for first_offset, first_weight in (
            (0, 1.0 - first_fraction),
            (1, first_fraction),
        ):
            for second_offset, second_weight in (
                (0, 1.0 - second_fraction),
                (1, second_fraction),
            ):
                coverage = first_weight * second_weight
                indices_xyz = np.zeros((len(samples), 3), dtype=np.int64)
                indices_xyz[:, first_axis] = first_floor + first_offset
                indices_xyz[:, second_axis] = second_floor + second_offset
                indices_xyz[:, fixed_axis] = 0
                valid = (
                    (coverage > 0.0)
                    & np.all(indices_xyz >= 0, axis=1)
                    & np.all(indices_xyz < size_xyz, axis=1)
                )
                if not np.any(valid):
                    continue
                valid_indices = indices_xyz[valid]
                np.maximum.at(
                    line_alpha_array,
                    (
                        valid_indices[:, 2],
                        valid_indices[:, 1],
                        valid_indices[:, 0],
                    ),
                    coverage[valid].astype(np.float32, copy=False),
                )

    @classmethod
    def _finalize_localizer_render(
        cls,
        *,
        image: sitk.Image,
        representative_slice: sitk.Image,
        line_alpha_array: np.ndarray,
        fixed_axis: int,
        legacy_line_value: float,
        localizer_surface: SpineLocalizerSurface | None = None,
    ) -> _LocalizerRender:
        """统一定位图头脚翻转、二维提取和患者空间标签。"""

        line_alpha_slice = sitk.GetImageFromArray(line_alpha_array)
        line_alpha_slice.CopyInformation(representative_slice)
        background_slice = representative_slice
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        if float(direction[2, 2]) > 0.0:
            # Localizer 降为二维后不能再依赖三维 z 轴确定头脚顺序，背景和
            # 覆盖掩膜必须执行完全相同的翻转，防止显示线条错位。
            background_slice = sitk.Flip(
                background_slice,
                [False, False, True],
                flipAboutOrigin=False,
            )
            line_alpha_slice = sitk.Flip(
                line_alpha_slice,
                [False, False, True],
                flipAboutOrigin=False,
            )
        localizer = cls._extract_projection_axis(background_slice, fixed_axis)
        line_alpha_image = cls._extract_projection_axis(
            line_alpha_slice,
            fixed_axis,
        )
        remaining_axes = [axis for axis in range(3) if axis != fixed_axis]
        localizer_direction = np.asarray(
            background_slice.GetDirection(),
            dtype=float,
        ).reshape(3, 3)
        row_direction = localizer_direction[:, remaining_axes[0]]
        column_direction = localizer_direction[:, remaining_axes[1]]
        first_pixel = background_slice.TransformIndexToPhysicalPoint((0, 0, 0))
        localizer.SetMetaData(
            "0020|0037",
            "\\".join(
                f"{float(value):.12g}"
                for value in (*row_direction, *column_direction)
            ),
        )
        localizer.SetMetaData(
            "0020|0032",
            "\\".join(f"{float(value):.12g}" for value in first_pixel),
        )
        if localizer_surface is not None:
            # x/z 仍是原图参考网格，但每行取样的 y 不同，不能冒充单一冠状切面。
            localizer.SetMetaData(
                "0008|2111",
                "Spine-following coronal localizer: AP sampling depth varies by row. "
                "Position/orientation describe the x/z reference grid, not a single "
                "physical coronal plane. MPR lines intersect the sampling surface; "
                "CPR lines project the sampled surface center paths.",
            )
        line_alpha = np.clip(
            sitk.GetArrayFromImage(line_alpha_image).astype(np.float32),
            0.0,
            1.0,
        )
        return _LocalizerRender(
            background_image=localizer,
            line_alpha=line_alpha,
            legacy_line_value=float(legacy_line_value),
            background_kind=(
                "spine_following_coronal" if localizer_surface is not None else "orthogonal_plane"
            ),
        )

    @classmethod
    def _compose_legacy_localizer_image(
        cls,
        render: _LocalizerRender,
    ) -> sitk.Image:
        """为既有内部检查生成带线 HU 图，不用于分窗 DICOM 显示。"""

        background = cls._localizer_array_2d(render.background_image).astype(
            np.float32,
        )
        composed = background + render.line_alpha * (
            render.legacy_line_value - background
        )
        return cls._image_from_array_like(
            render.background_image,
            composed.astype(np.float32, copy=False),
        )

    @staticmethod
    def _physical_points_to_continuous_indices(
        image: sitk.Image,
        physical_points: Any,
    ) -> np.ndarray:
        """按图像仿射几何批量把物理点转换为连续索引。"""

        points = np.asarray(physical_points, dtype=float)
        dimension = int(image.GetDimension())
        if points.size == 0:
            return np.empty((0, dimension), dtype=float)
        if points.ndim != 2 or points.shape[1] != dimension:
            raise ValueError(
                "physical_points 必须是形状为 (N, image_dimension) 的二维数组"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("physical_points 必须是有限数值")

        origin = np.asarray(image.GetOrigin(), dtype=float)
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(
            dimension,
            dimension,
        )
        if (
            np.any(~np.isfinite(origin))
            or np.any(~np.isfinite(spacing))
            or np.any(spacing <= 0.0)
            or np.any(~np.isfinite(direction))
        ):
            raise ValueError("图像空间几何必须是有限且有效的")

        # SimpleITK 的物理坐标关系为 p = origin + D @ (index * spacing)。
        # 一次批量求解可避免长 CPR 定位曲线逐点跨 Python/C++ 边界。
        scaled_indices = np.linalg.solve(
            direction,
            (points - origin).T,
        ).T
        return scaled_indices / spacing

    @classmethod
    def _curved_localizer_fixed_index(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
        fixed_axis: int,
    ) -> int:
        """选择经过目标曲线的原始正交定位层。"""

        localizer_reference = plan.get("localizer_reference")
        if localizer_reference:
            reference_index = image.TransformPhysicalPointToContinuousIndex((
                float(localizer_reference["x"]),
                float(localizer_reference["y"]),
                float(localizer_reference["z"]),
            ))
            fixed_index = int(round(float(reference_index[fixed_axis])))
        else:
            curve_indices = [
                image.TransformPhysicalPointToContinuousIndex((
                    float(point["x"]),
                    float(point["y"]),
                    float(point["z"]),
                ))
                for point in plan.get("curve_points", [])
            ]
            fixed_index = int(round(float(np.median([
                index_xyz[fixed_axis] for index_xyz in curve_indices
            ]))))
        return min(max(fixed_index, 0), image.GetSize()[fixed_axis] - 1)

    @staticmethod
    def _curved_localizer_offsets(plan: Dict[str, Any]) -> List[float]:
        """返回 CPR 定位图需要投影的全部 offset 中心线。"""

        raw_offsets = plan.get("offsets_mm")
        offsets: list[float] = []
        if isinstance(raw_offsets, list):
            for value in raw_offsets:
                try:
                    offset = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(offset):
                    offsets.append(offset)
        if not offsets:
            offsets = [0.0]
        if not any(abs(offset) <= 1e-6 for offset in offsets):
            offsets.append(0.0)

        return sorted({round(offset, 6) for offset in offsets})

    @classmethod
    def _curved_localizer_projected_curves(
        cls,
        *,
        plan: Dict[str, Any],
        fixed_axis: int,
    ) -> Dict[float, List[tuple[float, float, float]]]:
        """按实际 CPR 采样曲线和 offset 生成定位曲线。"""

        curve = np.asarray([
            cls._point_array(point)
            for point in (plan.get("curve_points") or [])
        ], dtype=float)
        if len(curve) < 2:
            return {}
        offsets = cls._curved_localizer_offsets(plan)
        rows = cls._curved_localizer_sample_count(plan, len(curve))
        curve = CurvedStackResampler._resample_curve(curve, rows)
        offset_axis = cls._point_array(plan.get("offset_axis") or {})
        offset_length = float(np.linalg.norm(offset_axis))
        if not np.isfinite(offset_length) or offset_length <= 1e-8:
            return {}
        offset_axis /= offset_length
        try:
            offset_curves = CurvedStackResampler._offset_curve_samples(
                curve,
                offset_axis,
                offsets,
                support_mm=CurvedStackResampler._maximum_offset_extent_mm(plan),
            )
        except ValueError:
            return {}

        projected_curves: Dict[float, List[tuple[float, float, float]]] = {}
        for offset_mm in offsets:
            projected = offset_curves[offset_mm]
            projected_curves[offset_mm] = [
                tuple(float(value) for value in point)
                for point in projected
            ]
        return projected_curves

    @staticmethod
    def _curved_localizer_sample_count(plan: Dict[str, Any], fallback: int) -> int:
        """复用 CPR 重建器的纵向输出行数。"""

        extent = plan.get("extent_mm") or {}
        spacing = plan.get("output_spacing_mm") or {}
        try:
            height = float(extent["height"])
            spacing_y = float(spacing["y"])
        except (KeyError, TypeError, ValueError):
            return fallback
        if not math.isfinite(height) or not math.isfinite(spacing_y) or spacing_y <= 0.0:
            return fallback
        # 定位线至少保留首尾两个采样点；即使测试计划或极短范围只有一行
        # 正式切片，也不能让纯显示用的曲线退化后阻断重建流程。
        return max(2, int(round(height / spacing_y)) + 1)

    @staticmethod
    def _point_array(point: Dict[str, Any]) -> np.ndarray:
        """读取物理点或方向字典。"""

        return np.asarray([
            float(point["x"]),
            float(point["y"]),
            float(point["z"]),
        ], dtype=float)

    @classmethod
    def _resolve_localizer_axes(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
        default_axes: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """按计划显式状位或切片斜率选择原图正交定位面。

        第六阶段可用 ``localizer_plane`` 明确指定背景状位；缺少该字段时根据
        当前计划几何自动选择。
        """

        stack_axis, _, default_fixed_axis = default_axes
        localizer_plane = plan.get("localizer_plane")
        if localizer_plane is not None:
            try:
                fixed_axis = cls._LOCALIZER_FIXED_AXES[str(localizer_plane)]
            except KeyError as error:
                raise ValueError(
                    f"不支持的定位图参考状位: {localizer_plane}"
                ) from error
            if fixed_axis == stack_axis:
                raise ValueError(
                    f"定位图参考状位不能与切片堆叠轴相同: {localizer_plane}"
                )
            line_axis = next(
                axis for axis in range(3)
                if axis not in {stack_axis, fixed_axis}
            )
            return stack_axis, line_axis, fixed_axis

        slices = plan.get("slices", [])
        if not slices:
            return default_axes
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        normal = cls._vector(slices[0].get("normal", {}))
        stack_direction = direction[:, stack_axis]
        # 零角度计划保持原有定位面，避免已有输出方向发生不必要变化。
        if abs(abs(float(np.dot(normal, stack_direction))) - 1.0) <= 1e-6:
            return default_axes

        candidate_fixed_axes = [axis for axis in range(3) if axis != stack_axis]
        fixed_axis = max(
            candidate_fixed_axes,
            key=lambda axis: abs(float(np.dot(
                np.cross(normal, direction[:, axis]),
                stack_direction,
            ))),
        )
        line_axis = next(
            axis for axis in range(3)
            if axis not in {stack_axis, fixed_axis}
        )
        if fixed_axis == default_fixed_axis:
            return stack_axis, line_axis, fixed_axis
        return stack_axis, line_axis, fixed_axis

    @classmethod
    def _slice_localizer_intersection(
        cls,
        *,
        planned_slice: Dict[str, Any],
        background_image: sitk.Image,
        fixed_axis: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """计算切片平面与定位底图的交线，仅裁入底图的完整像素网格。"""

        center = cls._vector(planned_slice.get("center", {}), normalize=False)
        normal = cls._vector(planned_slice.get("normal", {}))
        direction = np.asarray(background_image.GetDirection(), dtype=float).reshape(3, 3)
        fixed_plane_normal = cls._normalize_array(direction[:, fixed_axis])
        fixed_plane_point = np.asarray(background_image.GetOrigin(), dtype=float)

        in_slice_normal = (
            fixed_plane_normal
            - float(np.dot(fixed_plane_normal, normal)) * normal
        )
        denominator = float(np.dot(fixed_plane_normal, in_slice_normal))
        if abs(denominator) <= 1e-8:
            return None
        distance = float(np.dot(center - fixed_plane_point, fixed_plane_normal))
        intersection_center = center - in_slice_normal * distance / denominator
        line_direction = cls._normalize_array(np.cross(normal, fixed_plane_normal))

        index_origin = np.asarray(background_image.TransformPhysicalPointToContinuousIndex(
            tuple(intersection_center),
        ))
        voxel_axes = direction * np.asarray(background_image.GetSpacing())
        index_step = np.linalg.solve(voxel_axes, line_direction)
        # 在连续索引中裁到底图边界，兼容非等距、斜向和反向存储。
        # 不使用成片 extent 或面内居中位置，也不虚构巨大 FOV 来延长线。
        lower_t, upper_t = -math.inf, math.inf
        for offset, component, size in zip(index_origin, index_step, background_image.GetSize()):
            if abs(component) <= 1e-10:
                if offset < -1e-7 or offset > size - 1 + 1e-7:
                    return None
                continue
            first_t = -offset / component
            second_t = (size - 1 - offset) / component
            lower_t = max(lower_t, min(first_t, second_t))
            upper_t = min(upper_t, max(first_t, second_t))
            if lower_t > upper_t:
                return None
        return (
            intersection_center + line_direction * lower_t,
            intersection_center + line_direction * upper_t,
        )

    @staticmethod
    def _vector(value: Dict[str, Any], *, normalize: bool = True) -> np.ndarray:
        """读取 x/y/z 字典；方向默认归一化，物理点可保留原值。"""

        vector = np.asarray([
            float(value["x"]),
            float(value["y"]),
            float(value["z"]),
        ])
        return ReconstructionStage._normalize_array(vector) if normalize else vector

    @staticmethod
    def _normalize_array(vector: np.ndarray) -> np.ndarray:
        """将 NumPy 三维向量归一化，并拒绝零长度或非有限方向。"""

        length = float(np.linalg.norm(vector))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("定位图方向向量长度无效")
        return vector / length

    @staticmethod
    def _extract_projection_axis(
        projection: sitk.Image,
        projection_axis: int,
    ) -> sitk.Image:
        """移除定位背景或线掩膜的单体素轴，得到二维显示图。"""

        size = list(projection.GetSize())
        size[projection_axis] = 0
        return sitk.Extract(projection, size, [0, 0, 0])

    @staticmethod
    def _write_window_series(
        *,
        sampled_slices: List[Tuple[Dict[str, Any], sitk.Image]],
        window: Any,
        output_dir: Path,
        plan: Dict[str, Any],
        series_uid: str,
        study_uid: str,
        source_metadata: Dict[str, str],
    ) -> None:
        """
        把一套已完成 MPR 的 HU 图像处理成一个窗位 DICOM 序列。

        本方法不会修改共享的 sampled_image，写盘时直接按当前窗位窗化浮点 HU；
        output_dir 和 series_uid 也只属于当前窗位，因此可以在线程池中执行。实际
        GDCM 写盘由 ``src.reconstruction.dicom`` 内的统一锁保护。

        Args:
            sampled_slices: `(切片计划, HU 图像)` 列表，顺序与第六阶段一致。
            window: 当前窗宽窗位预设实体。
            output_dir: 当前窗位独占的输出目录。
            plan: 当前第六阶段切片计划。
            series_uid: 当前窗位 DICOM 序列 UID。
            study_uid: 所有派生序列共同使用的检查 UID。
            source_metadata: 从原始 DICOM 继承的患者和检查级标签。
        """
        total_slices = len(sampled_slices)
        write_start = time.perf_counter()
        for planned_slice, sampled_image in sampled_slices:
            output_path = output_dir / (
                f"{int(planned_slice['index'])}.dcm"
            )
            write_ct_dicom(
                image=sampled_image,
                output_path=output_path,
                plan=plan,
                planned_slice=planned_slice,
                window=window,
                series_uid=series_uid,
                study_uid=study_uid,
                total_slices=total_slices,
                source_metadata=source_metadata,
            )
        DebugLog(
            "    [重建性能] 窗位切片写出完成: "
            f"plan={plan.get('plan_id', '')}, "
            f"window={getattr(window, 'name', window)}, "
            f"slices={total_slices}, "
            f"耗时={time.perf_counter() - write_start:.3f}s"
        )

    @staticmethod
    def _localizer_geometry_metadata(image: sitk.Image) -> dict[str, str]:
        """在进入窗位线程前冻结定位图三维位置和行列方向。"""

        required_tags = ("0020|0032", "0020|0037")
        missing_tags = [
            tag
            for tag in required_tags
            if not image.HasMetaDataKey(tag)
            or not image.GetMetaData(tag).strip()
        ]
        if missing_tags:
            raise ValueError(
                "Localizer 缺少必要空间标签: " + ", ".join(missing_tags)
            )
        optional_tags = ("0008|2111",)
        return {
            tag: image.GetMetaData(tag)
            for tag in (*required_tags, *optional_tags)
            if image.HasMetaDataKey(tag)
        }

    @staticmethod
    def _write_window_localizer(
        *,
        localizer_render: _LocalizerRender,
        geometry_metadata: dict[str, str],
        window: Any,
        output_dir: Path,
        series_uid: str,
        study_uid: str,
        source_metadata: Dict[str, str],
        plan: Dict[str, Any],
    ) -> None:
        """窗位化原始背景，再叠加共享的亚像素定位线并写出。"""

        write_start = time.perf_counter()
        windowed_background = apply_window_to_image(
            sitk.Image(localizer_render.background_image),
            window,
        )
        windowed_localizer = ReconstructionStage._blend_localizer_line(
            windowed_background,
            localizer_render.line_alpha,
        )
        for tag, value in geometry_metadata.items():
            windowed_localizer.SetMetaData(tag, value)
        write_localizer_dicom(
            image=windowed_localizer,
            output_path=output_dir / "Localizer.dcm",
            window=window,
            series_uid=series_uid,
            study_uid=study_uid,
            source_metadata=source_metadata,
            geometry_metadata=geometry_metadata,
        )
        if is_debug_file_enabled():
            ReconstructionStage._write_debug_localizer_png(
                windowed_background=windowed_background,
                line_alpha=localizer_render.line_alpha,
                output_path=output_dir / "Localizer.png",
            )
        DebugLog(
            "    [重建性能] Localizer写出完成: "
            f"plan={plan.get('plan_id', '')}, "
            f"window={getattr(window, 'name', window)}, "
            f"耗时={time.perf_counter() - write_start:.3f}s"
        )

    @classmethod
    def _blend_localizer_line(
        cls,
        windowed_background: sitk.Image,
        line_alpha: np.ndarray,
    ) -> sitk.Image:
        """按覆盖率把白色抗锯齿线叠加到 uint16 窗位背景。"""

        background = cls._localizer_array_2d(windowed_background).astype(
            np.float32,
        )
        alpha = cls._validated_localizer_line_alpha(line_alpha, background.shape)
        blended = np.rint(
            background + alpha * (65535.0 - background)
        ).astype(np.uint16)
        return cls._image_from_array_like(windowed_background, blended)

    @classmethod
    def _write_debug_localizer_png(
        cls,
        *,
        windowed_background: sitk.Image,
        line_alpha: np.ndarray,
        output_path: Path,
    ) -> None:
        """把相同覆盖掩膜以红色抗锯齿线叠加到调试 PNG。"""

        background = cls._localizer_array_2d(windowed_background)
        background_u8 = np.clip(
            background.astype(np.float32) / 65535.0 * 180.0,
            0.0,
            180.0,
        )
        alpha = cls._validated_localizer_line_alpha(
            line_alpha,
            background_u8.shape,
        )
        inverse_alpha = 1.0 - alpha
        red = np.rint(background_u8 * inverse_alpha + 255.0 * alpha)
        green = np.rint(background_u8 * inverse_alpha)
        blue = np.rint(background_u8 * inverse_alpha)
        rgb = np.clip(
            np.stack([red, green, blue], axis=-1),
            0.0,
            255.0,
        ).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(
            sitk.GetImageFromArray(rgb, isVector=True),
            str(output_path),
        )

    @staticmethod
    def _localizer_array_2d(image: sitk.Image) -> np.ndarray:
        """读取定位图为二维数组，同时支持二维和单层三维图像。"""

        array = sitk.GetArrayFromImage(image)
        if array.ndim == 2:
            return array
        singleton_axes = [
            axis
            for axis, size in enumerate(array.shape)
            if size == 1
        ]
        if array.ndim == 3 and singleton_axes:
            return np.squeeze(array, axis=singleton_axes[0])
        raise ValueError("定位图 PNG 只支持二维或单层三维图像")

    @staticmethod
    def _validated_localizer_line_alpha(
        line_alpha: np.ndarray,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        """校验并限制定位线覆盖率，避免背景与线条发生错位。"""

        alpha = np.asarray(line_alpha, dtype=np.float32)
        if alpha.shape != expected_shape:
            raise ValueError(
                "定位线掩膜尺寸与定位图背景不一致: "
                f"alpha={alpha.shape}, background={expected_shape}"
            )
        if not np.all(np.isfinite(alpha)):
            raise ValueError("定位线掩膜包含非有限数值")
        return np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _image_from_array_like(
        reference: sitk.Image,
        array: np.ndarray,
    ) -> sitk.Image:
        """以新像素数组创建图像，同时完整保留参考图像的几何和元数据。"""

        output = sitk.GetImageFromArray(array)
        output.CopyInformation(reference)
        for key in reference.GetMetaDataKeys():
            output.SetMetaData(key, reference.GetMetaData(key))
        return output

    @staticmethod
    def _interspace_framing_key(plan: Dict[str, Any]) -> tuple[str, str, str] | None:
        """同部位同配置的间隙共用矩阵及比例，重复状位配置各自隔离。"""

        if (
            plan.get("slice_type") != "椎间隙"
            or plan.get("geometry_type") != "plane_stack"
        ):
            return None
        return (
            str(plan.get("slice_part", "")),
            str(plan.get("config_id") or ""),
            str(plan.get("output_plane") or plan["plane"]),
        )

    @staticmethod
    def _group_plans_by_plane(
        plans: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按状位聚合计划，同时保留状位首次出现和状位内计划的原始顺序。

        第六阶段可能交错返回多个状位的计划。这里统一分组，保证第七阶段始终以
        “状位串行、状位内 plan 串行、plan 内窗位并行”的固定层级执行。

        Args:
            plans: 第六阶段生成的全部切片计划。

        Returns:
            以状位名称为键、该状位计划列表为值的有序字典。Python 字典会保留
            首次插入顺序，因此不需要额外排序或硬编码三平面顺序。
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for plan in plans:
            output_plane = str(plan.get("output_plane") or plan["plane"])
            grouped.setdefault(output_plane, []).append(plan)
        return grouped

    @staticmethod
    def _plan_region_name(plan: Dict[str, Any]) -> str:
        """返回用于输出目录的安全重建区域名称；整切位于窗位目录。"""

        if plan.get("geometry_type") == "curved_stack":
            return ""
        if plan.get("slice_type") == "MPR":
            return ""
        group_name = str(plan.get("group_name", ""))
        if group_name in {"", "whole_range"}:
            group_name = "区间切"
        if group_name in {"interspace", "区间切"}:
            group_name = "区间切"
        return (
            group_name
            .replace("-", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    @staticmethod
    def _validate_plan(plan: Dict[str, Any]) -> None:
        """校验第六阶段计划包含平面重建所需字段及合法的尺寸、间距。"""

        plan_id = plan.get("plan_id", "<unknown>") if isinstance(plan, dict) else "<unknown>"
        if not isinstance(plan, dict):
            raise ValueError(f"plan_id={plan_id} 必须是对象")
        if not plan.get("plan_id"):
            raise ValueError("plan_id 不能为空")
        geometry_type = plan.get("geometry_type")
        if geometry_type not in {"plane_stack", "curved_stack"}:
            raise ValueError(
                f"plan_id={plan_id} geometry_type={geometry_type} "
                "当前第七阶段仅支持 plane_stack/curved_stack"
            )
        for field_name in (
            "plane",
            "slice_type",
            "slices",
            "extent_mm",
            "output_spacing_mm",
            "slice_thickness",
            "slice_spacing",
            "window_presets",
        ):
            if field_name not in plan:
                raise ValueError(f"plan_id={plan_id} 缺少字段: {field_name}")
        slice_type = str(plan["slice_type"])
        if slice_type not in {"MPR", "CPR", "椎间隙"}:
            raise ValueError(
                f"plan_id={plan_id} SliceType={slice_type} 不受支持"
            )
        if slice_type == "CPR" and geometry_type != "curved_stack":
            raise ValueError(f"plan_id={plan_id} CPR 必须使用 curved_stack")
        if slice_type != "CPR" and geometry_type != "plane_stack":
            raise ValueError(
                f"plan_id={plan_id} {slice_type} 必须使用 plane_stack"
            )
        if not isinstance(plan["slices"], list):
            raise ValueError(f"plan_id={plan_id} slices 必须是列表")
        if not plan["slices"]:
            raise ValueError(f"plan_id={plan_id} slices 不能为空")
        declared_count = plan.get("slice_image_count")
        if declared_count is not None and declared_count != len(plan["slices"]):
            raise ValueError(
                f"plan_id={plan_id} 数量={declared_count} 与实际计划数量="
                f"{len(plan['slices'])} 不一致"
            )
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
            if geometry_type == "curved_stack" and "offset_mm" not in planned_slice:
                raise ValueError(f"plan_id={plan_id} CPR slice 缺少字段: offset_mm")
        if geometry_type == "curved_stack":
            for field_name in (
                "curve_points",
                "curve_spacing_mm",
                "offset_axis",
                "offsets_mm",
            ):
                if field_name not in plan:
                    raise ValueError(f"plan_id={plan_id} 缺少字段: {field_name}")
            if not isinstance(plan["curve_points"], list) or len(plan["curve_points"]) < 2:
                raise ValueError(f"plan_id={plan_id} curve_points 至少需要 2 个点")
            if float(plan["curve_spacing_mm"]) <= 0:
                raise ValueError(f"plan_id={plan_id} curve_spacing_mm 必须大于 0")
            if not isinstance(plan["offsets_mm"], list) or not plan["offsets_mm"]:
                raise ValueError(f"plan_id={plan_id} offsets_mm 不能为空")

    @staticmethod
    def _resampler_for_plan(plan: Dict[str, Any]) -> Any:
        """按第六阶段计划的几何类型选择专用重采样器。"""

        if plan.get("geometry_type") == "curved_stack":
            return CurvedStackResampler()
        return PlaneStackResampler()

    @staticmethod
    def _build_plan_summary(
        plan_results: List[Dict[str, Any]],
        output_root: Path,
    ) -> Dict[str, Any]:
        """按实际输出汇总计划数、切片数、窗位展开图像数和输出目录。"""

        plane_summary = {}
        for plan_result in plan_results:
            plane = plan_result["plane"]
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
            slice_count = int(plan_result.get("slice_count", 0))
            image_count = int(plan_result.get("image_count", 0))
            summary["plan_count"] += 1
            summary["count"] += slice_count
            summary["slice_count"] += slice_count
            summary["image_count"] += image_count

        slice_count = sum(summary["slice_count"] for summary in plane_summary.values())
        image_count = sum(summary["image_count"] for summary in plane_summary.values())
        result = {
            "plan_count": len(plan_results),
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
