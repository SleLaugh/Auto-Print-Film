#!/src/services/stage_reconstruction.py
# -*- coding: utf-8 -*-

"""
阶段五：重建与输出
MPR多平面重建与DICOM生成
按阶段四 SlicePlan 执行直线 plane_stack 重建

每个 plan 分成两个明确步骤：先串行完成全部 MPR HU 图像采样，再按窗位并行完成
灰度映射和 DICOM 写出。这样不同窗位共享同一份 MPR 结果，不会重复重采样。
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import ReconstructionStageResult
from src.reconstruction import PlaneStackResampler
from src.reconstruction.dicom import (
    new_uid,
    read_source_dicom_metadata,
    write_ct_dicom,
    write_localizer_dicom,
)
from src.reconstruction.windowing import apply_window_to_image, unique_window_presets
from src.utils import DebugLog


class ReconstructionStage(BaseStage):
    """执行阶段四计划，并按窗位输出派生 CT DICOM 序列。"""

    _LOCALIZER_DRAW_AXES = {
        "横状位": (2, 1, 0),
        "矢状位": (0, 2, 1),
        "冠状位": (1, 2, 0),
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化重建阶段并读取窗位并行配置。

        Args:
            config: 阶段配置。`parallel` 控制是否并行处理不同窗位；
                `max_window_workers` 限制同一 plan 的最大窗位线程数。
        """
        super().__init__(name="reconstruction", config=config)
        self.parallel = config.get('parallel', True) if config else True
        """是否为同一 plan 的不同窗位启用线程池。"""
        self.max_window_workers = max(
            1,
            int(config.get('max_window_workers', 4) if config else 4),
        )
        """窗位线程池上限；实际线程数不会超过当前窗位数量。"""
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        执行阶段四的全部平面切片计划并输出分窗位 DICOM 序列。

        状位与计划串行执行，同一计划内的不同窗位可并行；每个切割组都会生成
        自己的数字编号切片和 Localizer 定位图。
        """

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
            context.set_stage_result(ReconstructionStageResult(payload=result))
            DebugLog("  没有可执行的切片计划")
            return context

        image = sitk.ReadImage(str(image_path))
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
            for plan in plane_plans:
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
            DebugLog(f"  状位处理完成: {plane}")

        result = self._build_plan_summary(plans, output_root)
        result["outputs"] = plan_results
        
        context.set_stage_result(ReconstructionStageResult(payload=result))
        
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
        localizer_image: sitk.Image | None = None,
    ) -> Dict[str, Any]:
        """
        执行一个切片计划。

        第一步把计划中的所有位置采样成 HU 图像；第二步再按窗位处理整套采样结果。
        先完整采样能保证窗位线程不触碰重采样器，也符合“先采集、后显示处理”的流程。
        """
        if localizer_image is None and isinstance(image, sitk.Image):
            localizer_image = self._build_plan_localizer(image, plan)

        resampler = PlaneStackResampler()
        windows = unique_window_presets(plan["window_presets"])
        series_uids = {window.name: new_uid() for window in windows}
        localizer_series_uids = {window.name: new_uid() for window in windows}
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

        # 第一步：只做 MPR 和层厚采样，保留原始 float32 HU 图像。
        # 所有采样完成之前不会开始任何窗宽窗位映射。
        DebugLog(f"    [1] MPR采样: 共 {len(plan['slices'])} 张")
        sampled_slices = [
            (
                planned_slice,
                resampler.sample_slice(image, plan, planned_slice),
            )
            for planned_slice in plan["slices"]
        ]
        DebugLog("    [1] MPR采样完成")

        # 第二步：一个窗位任务负责处理整套已采样切片。每个任务写独立目录并使用
        # 独立 Series UID，因此线程之间没有文件名或 DICOM 序列状态冲突。
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
                        localizer_image=localizer_image,
                        localizer_series_uid=localizer_series_uids[window.name],
                    )
                    for window in windows
                ]
                # 显式读取每个 Future，确保线程异常回到阶段五并让 Pipeline 失败。
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
                    localizer_image=localizer_image,
                    localizer_series_uid=localizer_series_uids[window.name],
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

    @classmethod
    def _build_plan_localizer(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
    ) -> sitk.Image | None:
        """从代表性原 CT 正交层烧录每张计划切片与定位面的真实交线。"""

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
            # 兼容没有目标参考点的旧计划，继续使用切片中心中位数。
            fixed_index = int(round(float(np.median([
                index_xyz[fixed_axis] for index_xyz in center_indices
            ]))))
        fixed_index = min(max(fixed_index, 0), image.GetSize()[fixed_axis] - 1)
        roi_index = [0, 0, 0]
        roi_size = list(image.GetSize())
        roi_index[fixed_axis] = fixed_index
        roi_size[fixed_axis] = 1
        representative_slice = sitk.RegionOfInterest(image, roi_size, roi_index)
        representative_slice = sitk.Cast(representative_slice, sitk.sitkFloat32)
        localizer_array = sitk.GetArrayFromImage(representative_slice)
        source_array = sitk.GetArrayViewFromImage(image)
        source_min = float(np.min(source_array))
        source_max = float(np.max(source_array))
        intensity_span = max(source_max - source_min, 1.0)
        center_value = source_max + intensity_span
        size_xyz = representative_slice.GetSize()

        def paint(start_xyz: np.ndarray, end_xyz: np.ndarray, value: float) -> None:
            """把物理交线两端转换后的连续索引烧录到代表性定位层。"""

            difference = end_xyz - start_xyz
            steps = max(
                int(np.ceil(abs(difference[stack_axis]))),
                int(np.ceil(abs(difference[line_axis]))),
                1,
            )
            for step in range(steps + 1):
                index = start_xyz + difference * step / steps
                index_xyz = [0, 0, 0]
                index_xyz[stack_axis] = int(round(float(index[stack_axis])))
                index_xyz[line_axis] = int(round(float(index[line_axis])))
                index_xyz[fixed_axis] = 0
                if not all(
                    0 <= index_xyz[axis] < size_xyz[axis]
                    for axis in range(3)
                ):
                    continue
                localizer_array[
                    index_xyz[2],
                    index_xyz[1],
                    index_xyz[0],
                ] = value

        fixed_plane_index = [0, 0, 0]
        fixed_plane_index[fixed_axis] = fixed_index
        fixed_plane_point = np.asarray(
            image.TransformIndexToPhysicalPoint(fixed_plane_index),
            dtype=float,
        )
        fixed_plane_normal = np.asarray(
            image.GetDirection(),
            dtype=float,
        ).reshape(3, 3)[:, fixed_axis]
        for planned_slice in plan.get("slices", []):
            segment = cls._slice_localizer_intersection(
                planned_slice=planned_slice,
                extent_mm=plan.get("extent_mm", {}),
                fixed_plane_point=fixed_plane_point,
                fixed_plane_normal=fixed_plane_normal,
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
            paint(start_xyz, end_xyz, center_value)

        painted_slice = sitk.GetImageFromArray(localizer_array)
        painted_slice.CopyInformation(representative_slice)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        if float(direction[2, 2]) > 0.0:
            # Localizer 降为二维 DICOM 后不能再依赖三维 z 轴
            # 来决定头脚显示顺序。当原图 +z 指向物理头侧时，
            # 把高 z 放到像素顶部；-z 指向头侧时则保持原序。
            painted_slice = sitk.Flip(
                painted_slice,
                [False, False, True],
                flipAboutOrigin=False,
            )
        localizer = cls._extract_projection_axis(painted_slice, fixed_axis)
        remaining_axes = [axis for axis in range(3) if axis != fixed_axis]
        localizer_direction = np.asarray(
            painted_slice.GetDirection(),
            dtype=float,
        ).reshape(3, 3)
        row_direction = localizer_direction[:, remaining_axes[0]]
        column_direction = localizer_direction[:, remaining_axes[1]]
        first_pixel = painted_slice.TransformIndexToPhysicalPoint((0, 0, 0))
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
        return localizer

    @classmethod
    def _resolve_localizer_axes(
        cls,
        image: sitk.Image,
        plan: Dict[str, Any],
        default_axes: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """为斜切计划选择最能显示倾斜角度的原图正交定位面。"""

        stack_axis, _, default_fixed_axis = default_axes
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
        extent_mm: Dict[str, Any],
        fixed_plane_point: np.ndarray,
        fixed_plane_normal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """计算有限切片矩形与原图正交定位面的物理交线端点。"""

        center = cls._vector(planned_slice.get("center", {}), normalize=False)
        normal = cls._vector(planned_slice.get("normal", {}))
        row_dir = cls._vector(planned_slice.get("row_dir", {}))
        col_dir = cls._vector(planned_slice.get("col_dir", {}))
        fixed_plane_normal = cls._normalize_array(fixed_plane_normal)

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

        half_width = max(0.0, float(extent_mm.get("width", 0.0)) / 2.0)
        half_height = max(0.0, float(extent_mm.get("height", 0.0)) / 2.0)
        limits = []
        row_component = abs(float(np.dot(line_direction, row_dir)))
        col_component = abs(float(np.dot(line_direction, col_dir)))
        if row_component > 1e-8:
            limits.append(half_width / row_component)
        if col_component > 1e-8:
            limits.append(half_height / col_component)
        half_length = min(limits) if limits else 0.0
        return (
            intersection_center - line_direction * half_length,
            intersection_center + line_direction * half_length,
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
        """移除最大强度投影后的单体素轴，得到可写定位图的二维图像。"""

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
        localizer_image: sitk.Image | None,
        localizer_series_uid: str,
    ) -> None:
        """
        把一套已完成 MPR 的 HU 图像处理成一个窗位 DICOM 序列。

        本方法不会修改共享的 sampled_image，只会为当前窗位创建新的 uint16 图像；
        output_dir 和 series_uid 也只属于当前窗位，因此可以在线程池中安全执行。

        Args:
            sampled_slices: `(切片计划, HU 图像)` 列表，顺序与阶段四一致。
            window: 当前窗宽窗位预设实体。
            output_dir: 当前窗位独占的输出目录。
            plan: 当前阶段四切片计划。
            series_uid: 当前窗位 DICOM 序列 UID。
            study_uid: 所有派生序列共同使用的检查 UID。
            source_metadata: 从原始 DICOM 继承的患者和检查级标签。
        """
        total_slices = len(sampled_slices)
        for planned_slice, sampled_image in sampled_slices:
            windowed_image = apply_window_to_image(sampled_image, window)
            output_path = output_dir / (
                f"{int(planned_slice['index'])}.dcm"
            )
            write_ct_dicom(
                image=windowed_image,
                output_path=output_path,
                plan=plan,
                planned_slice=planned_slice,
                window=window,
                series_uid=series_uid,
                study_uid=study_uid,
                total_slices=total_slices,
                source_metadata=source_metadata,
            )
        if localizer_image is not None:
            windowed_localizer = apply_window_to_image(
                sitk.Image(localizer_image),
                window,
            )
            for geometry_tag in ("0020|0032", "0020|0037"):
                if localizer_image.HasMetaDataKey(geometry_tag):
                    windowed_localizer.SetMetaData(
                        geometry_tag,
                        localizer_image.GetMetaData(geometry_tag),
                    )
            write_localizer_dicom(
                image=windowed_localizer,
                output_path=output_dir / "Localizer.dcm",
                window=window,
                series_uid=localizer_series_uid,
                study_uid=study_uid,
                source_metadata=source_metadata,
            )

    @staticmethod
    def _group_plans_by_plane(
        plans: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按状位聚合计划，同时保留状位首次出现和状位内计划的原始顺序。

        阶段四未来可能交错返回多个状位的计划。这里统一分组，保证阶段五始终以
        “状位串行、状位内 plan 串行、plan 内窗位并行”的固定层级执行。

        Args:
            plans: 阶段四生成的全部切片计划。

        Returns:
            以状位名称为键、该状位计划列表为值的有序字典。Python 字典会保留
            首次插入顺序，因此不需要额外排序或硬编码三平面顺序。
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for plan in plans:
            grouped.setdefault(str(plan["plane"]), []).append(plan)
        return grouped

    @staticmethod
    def _plan_group_name(plan: Dict[str, Any]) -> str:
        """返回用于输出目录的安全切割组名称；整切直接位于窗位目录。"""

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
        """校验阶段四计划包含平面重建所需字段及合法的尺寸、间距。"""

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
        """按状位汇总计划数、切片数、窗位展开图像数和输出目录。"""

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
