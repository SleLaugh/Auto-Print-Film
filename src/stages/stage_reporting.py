#!/src/stages/stage_reporting.py
# # -*- coding: utf-8 -*-

"""
第八阶段：校验与结果汇总
生成报告与输出整理
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, Any

import pydicom
from pydicom.errors import InvalidDicomError

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.core.delivery_report import (
    build_status_report,
    write_failure_report,
    write_report,
)
from src.core.errors import OutputValidationError
from src.models import ReportingStageResult
from src.utils import DebugLog, is_debug_file_enabled


class ReportingStage(BaseStage):
    """结果汇总阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化第八阶段校验与结果汇总组件。"""

        super().__init__(name="reporting", config=config)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """生成正常交付报告，并在 debug 模式下输出完整诊断和 DICOM 校验结果。"""

        request = context.request
        
        DebugLog("=" * 60)
        DebugLog("【第八阶段】校验与结果汇总 - 校验后发布报告")
        DebugLog(f"  输出目录: {request.output_dir}")
        DebugLog("=" * 60)
        
        # 先校验再发布交付报告，避免调用方把不完整或几何不一致的序列当成成功结果。
        DebugLog("  [1] validate_output: 校验 DICOM 输出契约...")
        validation = self._validate_reconstruction(
            context.get_result('reconstruction') or {}
        )
        debug_report_path = ""
        if is_debug_file_enabled():
            DebugLog("  [2] generate_debug_report: 生成调试报告...")
            debug_report_path = str(self._write_debug_report(
                context,
                self._build_debug_report(context, validation),
            ))

        if not validation['valid']:
            error_count = len(validation['errors'])
            DebugLog(f"  [校验失败] 共发现 {error_count} 项 DICOM 输出错误")
            for error in validation['errors']:
                DebugLog(f"    - {error}")
            error_preview = '；'.join(validation['errors'][:3])
            if error_count > 3:
                error_preview += f'；另有 {error_count - 3} 项详见错误详情'
            error = OutputValidationError(
                f"第八阶段输出校验失败，共 {error_count} 项：{error_preview}",
                stage=self.name,
                retryable=False,
                details={
                    'error_count': error_count,
                    'errors': list(validation['errors']),
                },
            )
            write_failure_report(
                request.output_dir,
                str(error),
                request=request,
            )
            raise error

        DebugLog("  [3] generate_report: 生成处理报告...")
        report_data = self._build_delivery_report(context)
        report_path = self._write_report(context, report_data)
        
        # [4] organize_output: 确认重建阶段目录保持原样
        DebugLog("  [4] organize_output: 确认输出目录结构...")
        self._organize_output(context, report_data)

        reconstruction = context.get_result('reconstruction') or {}
        image_count = reconstruction.get('image_count')
        if image_count is None:
            image_count = sum(
                int(output.get('slice_count', 0))
                * len(output.get('window_outputs') or {})
                for output in reconstruction.get('outputs', [])
            )

        result = {
            'report_path': str(report_path),
            'debug_report_path': debug_report_path,
            'total_time': context.get_total_time(),
            'validation': validation,
            'image_count': int(image_count),
        }
        
        context.set_stage_result(ReportingStageResult(payload=result))
        
        DebugLog(f"  第八阶段完成，总耗时: {context.get_total_time():.2f}s")
        DebugLog("=" * 60)
        
        return context
    
    def _generate_report(self, context: ProcessingContext) -> Path:
        """生成简洁、稳定的正常交付报告。"""

        return self._write_report(context, self._build_delivery_report(context))

    def _build_delivery_report(self, context: ProcessingContext) -> Dict[str, Any]:
        """按状位配置、窗位和重建区域汇总请求值及实际输出。"""

        reconstruction = context.get_result('reconstruction') or {}
        outputs = list(reconstruction.get('outputs', []))

        plane_list = []
        plane_ordinals: dict[str, int] = {}
        for ordinal, plane_config in enumerate(
            context.request.plane_configs,
            start=1,
        ):
            config = plane_config.to_dict()
            plane = str(config['Plane'])
            plane_ordinal = plane_ordinals.get(plane, 0)
            output_plane = (
                plane_config.output_plane
                or (plane if plane_ordinal == 0 else f'{plane}{plane_ordinal}')
            )
            plane_ordinals[plane] = plane_ordinal + 1
            config_id = (
                plane_config.config_id
                or f"配置{ordinal:02d}"
            )
            matching_outputs = [
                output
                for output in outputs
                if self._output_matches_config(
                    output,
                    config_id=config_id,
                    plane=plane,
                    output_plane=output_plane,
                    slice_type=str(config['SliceType']),
                )
            ]
            actual_slice_types = {
                str(output['slice_type'])
                for output in matching_outputs
                if output.get('slice_type')
            }
            item = {
                'ConfigId': config_id,
                # Plane 表示医学状位；带序号的 output_plane 只用于隔离输出目录。
                'Plane': plane,
                'SliceType': self._single_actual_or_requested_value(
                    actual_slice_types,
                    config['SliceType'],
                ),
                'SliceThickness': self._optional_float(
                    plane_config.requested_slice_thickness,
                ),
                'SliceSpacing': self._optional_float(
                    plane_config.requested_slice_spacing,
                ),
                'SliceImageCount': self._optional_int(
                    plane_config.requested_slice_image_count,
                ),
                'WindowPresetList': self._delivery_windows(
                    config['WindowPresets'],
                    matching_outputs,
                ),
            }
            if 'FillPercent' in config:
                item['FillPercent'] = config['FillPercent']
            plane_list.append(item)

        return build_status_report(
            success=True,
            message='',
            request=context.request,
            plane_list=plane_list,
        )

    @staticmethod
    def _output_matches_config(
        output: Dict[str, Any],
        *,
        config_id: str,
        plane: str,
        output_plane: str,
        slice_type: str,
    ) -> bool:
        """优先按配置标识匹配；无标识时按状位和切割方式匹配。"""

        output_config_id = str(output.get('config_id', ''))
        if output_config_id:
            return output_config_id == config_id
        output_source_plane = str(
            output.get('source_plane')
            or output.get('plane', '')
        )
        return (
            output_source_plane in {plane, output_plane}
            and str(output.get('slice_type', '')) == slice_type
        )

    @classmethod
    def _delivery_windows(
        cls,
        requested_windows: list[str],
        outputs: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """按实际目录层级整理窗位及其重建区域。"""

        return [
            {
                'Name': window_name,
                'RegionList': [
                    cls._delivery_region(output, window_name)
                    for output in outputs
                    if window_name in (output.get('window_outputs') or {})
                ],
            }
            for window_name in dict.fromkeys(
                str(name) for name in requested_windows
            )
        ]

    @staticmethod
    def _delivery_region(
        output: Dict[str, Any],
        window_name: str,
    ) -> Dict[str, Any]:
        """整理一个窗位下实际重建区域的参数、数量和目录。"""

        region = ReportingStage._resolve_region(output)
        window_outputs = output.get('window_outputs') or {}
        return {
            'Region': region,
            'ActualSliceThickness': float(
                output.get('slice_thickness', 0.0)
            ),
            'ActualSliceSpacing': float(output.get('slice_spacing', 0.0)),
            'ActualSliceImageCount': int(output.get('slice_count', 0)),
            'Path': str(window_outputs[window_name]),
        }

    @staticmethod
    def _resolve_region(output: Dict[str, Any]) -> str:
        """返回稳定区域名称，整切为 default，椎间隙使用下划线。"""

        group_name = str(output.get('group_name') or '').strip()
        if group_name in {'', 'whole_range', '整切', 'CPR'}:
            return 'default'
        return (
            group_name
            .replace('-', '_')
            .replace('/', '_')
            .replace('\\', '_')
        )

    @staticmethod
    def _single_actual_or_requested_value(
        actual_values: set[Any],
        requested_value: Any,
    ) -> Any:
        """优先返回唯一实际值；没有输出时回退请求值。"""

        if len(actual_values) == 1:
            return next(iter(actual_values))
        if not actual_values:
            return requested_value
        raise ValueError(f"同一配置产生了多个实际切割方式: {actual_values}")

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        """把可选请求值转换成稳定 JSON 小数。"""

        return None if value is None else float(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        """把可选请求数量转换成稳定 JSON 整数。"""

        return None if value is None else int(value)

    def _build_debug_report(
        self,
        context: ProcessingContext,
        validation: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """汇总完整阶段结果，供 debug 模式排查问题。"""

        output_dir = Path(context.request.output_dir)
        planning = context.get_result('planning') or {}
        reconstruction = context.get_result('reconstruction') or {}
        validation = validation or self._validate_reconstruction(reconstruction)

        planning_plans = planning.get('plans', [])
        planning_summary = {
            'planner': planning.get('planner', ''),
            'plan_count': planning.get('plan_count', len(planning_plans)),
            'warnings': list(planning.get('warnings', [])),
            'diagnostics': list(planning.get('diagnostics', [])),
            'metadata_path': planning.get('metadata_path', ''),
            'plans': [self._plan_summary(plan) for plan in planning_plans],
        }
        reconstruction_summary = self._reconstruction_summary(reconstruction)
        warnings = list(planning_summary['warnings'])
        for plan in planning_summary['plans']:
            warnings.extend(plan['warnings'])
        warnings.extend(validation['warnings'])
        warnings = list(dict.fromkeys(str(warning) for warning in warnings))

        return {
            'schema_version': 'report_v2',
            'success': not context.has_errors() and validation['valid'],
            # 请求参数统一由 Request.to_dict() 输出。这样 plane_configs 这类新增字段
            # 只需要在 Request 中维护一次，报告不会因为手写字段列表而遗漏。
            'request': context.request.to_dict(),
            'timings': context.timings,
            'total_time': context.get_total_time(),
            'errors': [
                {'stage': e.stage, 'message': e.message}
                for e in context.errors
            ],
            'warnings': warnings,
            'planning': planning_summary,
            'reconstruction': reconstruction_summary,
            'validation': validation,
            'outputs': {
                'report': str(output_dir / 'report.json'),
                'nifti': str(output_dir / 'nifti'),
                'metadata': planning.get('metadata_path', ''),
                'axial': str(output_dir / '横状位'),
                'sagittal': str(output_dir / '矢状位'),
                'coronal': str(output_dir / '冠状位'),
                'planes': {
                    plane: summary.get('output_dir', '')
                    for plane, summary in reconstruction_summary['planes'].items()
                },
            },
        }

    @staticmethod
    def _write_report(context: ProcessingContext, report_data: Dict[str, Any]) -> Path:
        """把精简交付结果写入输出根目录的 report.json。"""

        report_path = write_report(context.request.output_dir, report_data)
        
        DebugLog(f"    报告已生成: {report_path}")
        return report_path

    @staticmethod
    def _write_debug_report(
        context: ProcessingContext,
        report_data: Dict[str, Any],
    ) -> Path:
        """把完整阶段诊断写入 metadata 目录下的调试报告。"""

        metadata_dir = Path(context.request.output_dir) / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)
        report_path = metadata_dir / '处理报告_调试.json'
        with open(report_path, 'w', encoding='utf-8') as file:
            json.dump(report_data, file, ensure_ascii=False, indent=2)
        DebugLog(f"    调试报告已生成: {report_path}")
        return report_path

    @staticmethod
    def _plan_summary(plan: Dict[str, Any]) -> Dict[str, Any]:
        """提取单个第六阶段计划中适合写入调试报告的摘要字段。"""

        return {
            'plan_id': plan.get('plan_id', ''),
            'slice_part': plan.get('slice_part', ''),
            'slice_type': plan.get('slice_type', ''),
            'plane': plan.get('plane', ''),
            'group_name': plan.get('group_name', ''),
            'geometry_type': plan.get('geometry_type', ''),
            'slice_count': len(plan.get('slices', [])),
            'slice_thickness': plan.get('slice_thickness'),
            'slice_spacing': plan.get('slice_spacing'),
            'window_presets': list(plan.get('window_presets', [])),
            'confidence': plan.get('confidence'),
            'source_labels': list(plan.get('source_labels', [])),
            'warnings': list(plan.get('warnings', [])),
        }

    @classmethod
    def _reconstruction_summary(cls, reconstruction: Dict[str, Any]) -> Dict[str, Any]:
        """整理第七阶段的计划、切片、图像及各状位输出统计。"""

        outputs = reconstruction.get('outputs', [])
        return {
            'plan_count': reconstruction.get('plan_count', len(outputs)),
            'slice_count': reconstruction.get('slice_count', 0),
            'image_count': reconstruction.get('image_count', 0),
            'planes': cls._json_safe(reconstruction.get('planes', {})),
            'outputs': cls._json_safe(outputs),
        }

    @classmethod
    def _validate_reconstruction(
        cls,
        reconstruction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """逐计划校验正式切片、定位图及同计划各窗位的共享几何。"""

        errors: list[str] = []
        warnings: list[str] = []
        series_results = []
        outputs = reconstruction.get('outputs', [])
        if (
            reconstruction.get('plan_count') is not None
            and int(reconstruction['plan_count']) != len(outputs)
        ):
            errors.append(
                '第七阶段计划汇总数量不一致，'
                f"声明 {int(reconstruction['plan_count'])} 个，实际 {len(outputs)} 个"
            )

        declared_plan_ids: set[str] = set()
        output_directory_owners: dict[str, str] = {}
        for output in outputs:
            plan_id = str(output.get('plan_id', ''))
            expected_count = int(output.get('slice_count', 0))
            if not plan_id:
                errors.append('第七阶段存在缺少 plan_id 的输出方案')
            elif plan_id in declared_plan_ids:
                errors.append(f'第七阶段存在重复 plan_id: {plan_id}')
            declared_plan_ids.add(plan_id)
            if expected_count <= 0:
                errors.append(f'{plan_id} 正式切片数量必须大于 0，实际为 {expected_count}')
            expected_thickness = cls._optional_finite_float(
                output.get('slice_thickness')
            )
            expected_spacing = cls._optional_finite_float(
                output.get('slice_spacing')
            )
            if output.get('slice_thickness') is None:
                errors.append(f'{plan_id} 第七阶段输出摘要缺少 slice_thickness')
            elif expected_thickness is None:
                errors.append(f'{plan_id} 第七阶段输出摘要中的 slice_thickness 无效')
            if output.get('slice_spacing') is None:
                errors.append(f'{plan_id} 第七阶段输出摘要缺少 slice_spacing')
            elif expected_spacing is None:
                errors.append(f'{plan_id} 第七阶段输出摘要中的 slice_spacing 无效')

            window_outputs = output.get('window_outputs') or {}
            if not window_outputs:
                errors.append(f'{plan_id} 没有任何窗位输出目录')
                continue
            expected_image_count = expected_count * len(window_outputs)
            if (
                output.get('image_count') is not None
                and int(output['image_count']) != expected_image_count
            ):
                errors.append(
                    f'{plan_id} 图像汇总数量不一致，'
                    f"声明 {int(output['image_count'])} 张，"
                    f'按 {expected_count} 张切片和 {len(window_outputs)} 个窗位'
                    f'应为 {expected_image_count} 张'
                )

            plan_series = []
            for window_name, directory in window_outputs.items():
                directory_path = Path(directory)
                directory_key = str(directory_path.resolve())
                owner = f'{plan_id}/{window_name}'
                previous_owner = output_directory_owners.get(directory_key)
                if previous_owner is not None:
                    errors.append(
                        f'{owner} 与 {previous_owner} 共用同一输出目录'
                    )
                else:
                    output_directory_owners[directory_key] = owner
                result = cls._validate_dicom_series(
                    plan_id=plan_id,
                    window_name=str(window_name),
                    directory=directory_path,
                    expected_count=expected_count,
                    expected_thickness=expected_thickness,
                    expected_spacing=expected_spacing,
                )
                plan_series.append(result)
                series_results.append(result)
                errors.extend(result['errors'])
                warnings.extend(result['warnings'])

            complete_series = [
                result for result in plan_series
                if result.get('_geometry_complete')
            ]
            if complete_series:
                baseline = complete_series[0]
                for current in complete_series[1:]:
                    comparison_errors = cls._compare_window_geometry(
                        plan_id=plan_id,
                        baseline=baseline,
                        current=current,
                    )
                    current['errors'].extend(comparison_errors)
                    current['valid'] = not current['errors']
                    errors.extend(comparison_errors)

        if reconstruction.get('slice_count', 0) and not outputs:
            errors.append('第七阶段报告了切片数量，但没有实际输出方案')
        if outputs and reconstruction.get('slice_count') is not None:
            actual_slice_count = sum(
                int(output.get('slice_count', 0)) for output in outputs
            )
            if int(reconstruction['slice_count']) != actual_slice_count:
                errors.append(
                    '第七阶段切片汇总数量不一致，'
                    f"声明 {int(reconstruction['slice_count'])} 张，"
                    f'各计划合计 {actual_slice_count} 张'
                )
        if outputs and reconstruction.get('image_count') is not None:
            actual_image_count = sum(
                int(output.get('image_count', 0)) for output in outputs
            )
            if int(reconstruction['image_count']) != actual_image_count:
                errors.append(
                    '第七阶段图像汇总数量不一致，'
                    f"声明 {int(reconstruction['image_count'])} 张，"
                    f'各计划合计 {actual_image_count} 张'
                )

        identity_errors = cls._validate_global_dicom_identity(series_results)
        errors.extend(identity_errors)
        for result in series_results:
            result.pop('_slice_geometry', None)
            result.pop('_localizer_geometry', None)
            result.pop('_geometry_complete', None)
            result.pop('_identity', None)

        errors = list(dict.fromkeys(errors))
        warnings = list(dict.fromkeys(warnings))

        return {
            'valid': not errors,
            'errors': errors,
            'warnings': warnings,
            'series': series_results,
        }

    @classmethod
    def _validate_dicom_series(
        cls,
        *,
        plan_id: str,
        window_name: str,
        directory: Path,
        expected_count: int,
        expected_thickness: float | None = None,
        expected_spacing: float | None = None,
    ) -> Dict[str, Any]:
        """校验一个窗位的正式切片和独立 Localizer DICOM 契约。"""

        errors: list[str] = []
        warnings: list[str] = []
        numbered_files: list[tuple[int, Path]] = []
        if directory.exists():
            for file_path in sorted(directory.glob('*.dcm')):
                if file_path.name == 'Localizer.dcm':
                    continue
                match = re.fullmatch(r'(\d+)\.dcm', file_path.name)
                if match:
                    numbered_files.append((int(match.group(1)), file_path))
                else:
                    errors.append(
                        f'{plan_id}/{window_name} 正式切片文件名格式错误: '
                        f'{file_path.name}'
                    )
        numbered_files.sort(key=lambda item: item[0])
        files = [file_path for _, file_path in numbered_files]
        actual_count = len(files)
        if not directory.exists():
            errors.append(f'{plan_id}/{window_name} 输出目录不存在')
        if actual_count != expected_count:
            errors.append(
                f'{plan_id}/{window_name} 期望 {expected_count} 张，实际 {actual_count} 张'
            )

        indices = []
        series_uids = set()
        study_uids = set()
        sop_instance_uids = []
        instance_numbers = []
        slice_geometry = []
        thickness_mismatches = []
        spacing_mismatches = []
        for file_path in files:
            match = re.fullmatch(r'(\d+)\.dcm', file_path.name)
            if not match:
                errors.append(f'{plan_id}/{window_name} 文件名格式错误: {file_path.name}')
                continue
            indices.append(int(match.group(1)))
            try:
                dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                series_uid = str(getattr(dataset, 'SeriesInstanceUID', '')).strip()
                study_uid = str(getattr(dataset, 'StudyInstanceUID', '')).strip()
                sop_uid = str(getattr(dataset, 'SOPInstanceUID', '')).strip()
                if series_uid:
                    series_uids.add(series_uid)
                if study_uid:
                    study_uids.add(study_uid)
                if sop_uid:
                    sop_instance_uids.append(sop_uid)
                instance_numbers.append(int(dataset.InstanceNumber))

                geometry = cls._dicom_geometry(dataset, include_slice_metrics=True)
                geometry['index'] = int(match.group(1))
                slice_geometry.append(geometry)
                if (
                    expected_thickness is not None
                    and not cls._numbers_close(
                        geometry['SliceThickness'], expected_thickness
                    )
                ):
                    thickness_mismatches.append(
                        f"{file_path.name}={geometry['SliceThickness']:g}"
                    )
                if (
                    expected_spacing is not None
                    and not cls._numbers_close(
                        geometry['SpacingBetweenSlices'], expected_spacing
                    )
                ):
                    spacing_mismatches.append(
                        f"{file_path.name}={geometry['SpacingBetweenSlices']:g}"
                    )
            except (InvalidDicomError, OSError, TypeError, ValueError, AttributeError) as exc:
                errors.append(f'{plan_id}/{window_name} DICOM 无法读取 {file_path.name}: {exc}')

        expected_indices = list(range(1, expected_count + 1))
        if indices != expected_indices:
            errors.append(
                f'{plan_id}/{window_name} 切片编号不连续，实际为 {indices}'
            )
        if len(series_uids) > 1:
            errors.append(f'{plan_id}/{window_name} 存在多个 SeriesInstanceUID')
        if not series_uids and files:
            errors.append(f'{plan_id}/{window_name} 缺少 SeriesInstanceUID')
        if len(study_uids) > 1:
            errors.append(f'{plan_id}/{window_name} 存在多个 StudyInstanceUID')
        if not study_uids and files:
            errors.append(f'{plan_id}/{window_name} 缺少 StudyInstanceUID')
        if len(sop_instance_uids) != actual_count:
            errors.append(f'{plan_id}/{window_name} 存在缺少 SOPInstanceUID 的正式切片')
        elif len(set(sop_instance_uids)) != actual_count:
            errors.append(f'{plan_id}/{window_name} 正式切片的 SOPInstanceUID 不唯一')
        if len(instance_numbers) != actual_count:
            errors.append(f'{plan_id}/{window_name} 存在缺少 InstanceNumber 的 DICOM')
        elif instance_numbers != expected_indices:
            errors.append(
                f'{plan_id}/{window_name} InstanceNumber 不连续，实际为 {instance_numbers}'
            )
        if thickness_mismatches:
            errors.append(
                f'{plan_id}/{window_name} SliceThickness 与计划 '
                f'{expected_thickness:g}mm 不一致: '
                + ', '.join(thickness_mismatches[:10])
            )
        if spacing_mismatches:
            errors.append(
                f'{plan_id}/{window_name} SpacingBetweenSlices 与计划 '
                f'{expected_spacing:g}mm 不一致: '
                + ', '.join(spacing_mismatches[:10])
            )

        localizer_geometry = None
        localizer_series_uid = ''
        localizer_study_uid = ''
        localizer_sop_uid = ''
        localizer_path = directory / 'Localizer.dcm'
        if not localizer_path.is_file():
            errors.append(f'{plan_id}/{window_name} 缺少定位图 Localizer.dcm')
        else:
            try:
                localizer_dataset = pydicom.dcmread(
                    str(localizer_path), stop_before_pixels=True
                )
                localizer_geometry = cls._dicom_geometry(
                    localizer_dataset,
                    include_slice_metrics=False,
                )
                localizer_series_uid = str(
                    getattr(localizer_dataset, 'SeriesInstanceUID', '')
                ).strip()
                localizer_study_uid = str(
                    getattr(localizer_dataset, 'StudyInstanceUID', '')
                ).strip()
                localizer_sop_uid = str(
                    getattr(localizer_dataset, 'SOPInstanceUID', '')
                ).strip()
                raw_image_type = getattr(localizer_dataset, 'ImageType', [])
                if isinstance(raw_image_type, str):
                    raw_image_type = raw_image_type.split('\\')
                image_type = {
                    str(value).upper()
                    for value in raw_image_type
                }
                if 'LOCALIZER' not in image_type:
                    errors.append(
                        f'{plan_id}/{window_name} Localizer.dcm 的 ImageType '
                        '缺少 LOCALIZER'
                    )
                if not localizer_series_uid:
                    errors.append(
                        f'{plan_id}/{window_name} Localizer.dcm 缺少 SeriesInstanceUID'
                    )
                elif localizer_series_uid in series_uids:
                    errors.append(
                        f'{plan_id}/{window_name} Localizer.dcm 与正式切片共用 '
                        'SeriesInstanceUID'
                    )
                if not localizer_study_uid:
                    errors.append(
                        f'{plan_id}/{window_name} Localizer.dcm 缺少 StudyInstanceUID'
                    )
                elif study_uids and localizer_study_uid not in study_uids:
                    errors.append(
                        f'{plan_id}/{window_name} Localizer.dcm 与正式切片的 '
                        'StudyInstanceUID 不一致'
                    )
                if not localizer_sop_uid:
                    errors.append(
                        f'{plan_id}/{window_name} Localizer.dcm 缺少 SOPInstanceUID'
                    )
                elif localizer_sop_uid in sop_instance_uids:
                    errors.append(
                        f'{plan_id}/{window_name} Localizer.dcm 的 SOPInstanceUID '
                        '与正式切片重复'
                    )
            except (InvalidDicomError, OSError, TypeError, ValueError, AttributeError) as exc:
                errors.append(
                    f'{plan_id}/{window_name} 定位图无法读取 Localizer.dcm: {exc}'
                )

        geometry_complete = (
            len(slice_geometry) == actual_count
            and actual_count == expected_count
            and localizer_geometry is not None
        )

        return {
            'plan_id': plan_id,
            'window': window_name,
            'directory': str(directory),
            'expected_count': expected_count,
            'actual_count': actual_count,
            'series_instance_uid': next(iter(series_uids), ''),
            'localizer_series_instance_uid': localizer_series_uid,
            'instance_numbers': instance_numbers,
            'valid': not errors,
            'errors': errors,
            'warnings': warnings,
            'geometry_summary': {
                'formal_slice_count': len(slice_geometry),
                'first_slice': slice_geometry[0] if slice_geometry else None,
                'last_slice': slice_geometry[-1] if slice_geometry else None,
                'localizer': localizer_geometry,
            },
            '_slice_geometry': slice_geometry,
            '_localizer_geometry': localizer_geometry,
            '_geometry_complete': geometry_complete,
            '_identity': {
                'formal_series_uids': sorted(series_uids),
                'formal_sop_uids': list(sop_instance_uids),
                'study_uids': sorted(study_uids),
                'localizer_series_uid': localizer_series_uid,
                'localizer_sop_uid': localizer_sop_uid,
                'localizer_study_uid': localizer_study_uid,
            },
        }

    @staticmethod
    def _validate_global_dicom_identity(
        series_results: list[Dict[str, Any]],
    ) -> list[str]:
        """校验一次任务内各 DICOM 序列和实例 UID 全局唯一且同属一个检查。"""

        errors: list[str] = []
        series_owners: dict[str, str] = {}
        sop_owners: dict[str, str] = {}
        study_owners: dict[str, str] = {}

        def register_unique(
            registry: dict[str, str],
            uid: str,
            owner: str,
            identity_name: str,
        ) -> None:
            if not uid:
                return
            previous_owner = registry.get(uid)
            if previous_owner is not None and previous_owner != owner:
                errors.append(
                    f'{owner} 与 {previous_owner} 重复使用 {identity_name}'
                )
            else:
                registry[uid] = owner

        for result in series_results:
            owner = f"{result['plan_id']}/{result['window']}"
            identity = result.get('_identity') or {}
            for uid in identity.get('formal_series_uids', []):
                register_unique(
                    series_owners, str(uid), f'{owner}/正式切片',
                    'SeriesInstanceUID',
                )
            register_unique(
                series_owners,
                str(identity.get('localizer_series_uid', '')),
                f'{owner}/Localizer.dcm',
                'SeriesInstanceUID',
            )
            for index, uid in enumerate(identity.get('formal_sop_uids', []), 1):
                register_unique(
                    sop_owners, str(uid), f'{owner}/{index}.dcm',
                    'SOPInstanceUID',
                )
            register_unique(
                sop_owners,
                str(identity.get('localizer_sop_uid', '')),
                f'{owner}/Localizer.dcm',
                'SOPInstanceUID',
            )
            for uid in identity.get('study_uids', []):
                study_owners.setdefault(str(uid), owner)
            localizer_study_uid = str(
                identity.get('localizer_study_uid', '')
            )
            if localizer_study_uid:
                study_owners.setdefault(localizer_study_uid, owner)

        if len(study_owners) > 1:
            errors.append(
                '同一任务内存在多个 StudyInstanceUID，涉及: '
                + ', '.join(sorted(set(study_owners.values())))
            )
        return errors

    @classmethod
    def _compare_window_geometry(
        cls,
        *,
        plan_id: str,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> list[str]:
        """确认同一计划各窗位只改变灰度映射，不改变空间几何。"""

        errors: list[str] = []
        baseline_window = str(baseline['window'])
        current_window = str(current['window'])
        localizer_fields = cls._geometry_difference_fields(
            baseline['_localizer_geometry'],
            current['_localizer_geometry'],
        )
        if localizer_fields:
            errors.append(
                f'{plan_id}/{current_window} 的 Localizer.dcm 与基准窗位 '
                f'{baseline_window} 几何不一致: {", ".join(localizer_fields)}'
            )

        mismatches = []
        for baseline_slice, current_slice in zip(
            baseline['_slice_geometry'],
            current['_slice_geometry'],
        ):
            fields = cls._geometry_difference_fields(
                baseline_slice,
                current_slice,
                ignored_fields={'index'},
            )
            if fields:
                mismatches.append(
                    f"{current_slice['index']}.dcm({', '.join(fields)})"
                )
        if mismatches:
            errors.append(
                f'{plan_id}/{current_window} 的正式切片与基准窗位 '
                f'{baseline_window} 几何不一致: {", ".join(mismatches[:10])}'
            )
        return errors

    @classmethod
    def _dicom_geometry(
        cls,
        dataset: Any,
        *,
        include_slice_metrics: bool,
    ) -> Dict[str, Any]:
        """读取参与输出契约比较的标准 DICOM 几何字段。"""

        rows = cls._required_positive_int(dataset, 'Rows')
        columns = cls._required_positive_int(dataset, 'Columns')
        geometry = {
            'Rows': rows,
            'Columns': columns,
            'PixelSpacing': cls._required_float_sequence(
                dataset, 'PixelSpacing', 2
            ),
            'ImagePositionPatient': cls._required_float_sequence(
                dataset, 'ImagePositionPatient', 3
            ),
            'ImageOrientationPatient': cls._required_float_sequence(
                dataset, 'ImageOrientationPatient', 6
            ),
        }
        if any(value <= 0.0 for value in geometry['PixelSpacing']):
            raise ValueError('PixelSpacing 必须全部大于 0')
        row_direction = geometry['ImageOrientationPatient'][:3]
        column_direction = geometry['ImageOrientationPatient'][3:]
        row_norm = math.sqrt(sum(value * value for value in row_direction))
        column_norm = math.sqrt(
            sum(value * value for value in column_direction)
        )
        direction_dot = sum(
            left * right
            for left, right in zip(row_direction, column_direction)
        )
        if (
            not math.isclose(row_norm, 1.0, rel_tol=1e-4, abs_tol=1e-4)
            or not math.isclose(
                column_norm, 1.0, rel_tol=1e-4, abs_tol=1e-4
            )
            or not math.isclose(
                direction_dot, 0.0, rel_tol=1e-4, abs_tol=1e-4
            )
        ):
            raise ValueError(
                'ImageOrientationPatient 的行列方向必须为正交单位向量'
            )
        if include_slice_metrics:
            geometry.update({
                'SliceThickness': cls._required_positive_float(
                    dataset, 'SliceThickness'
                ),
                'SpacingBetweenSlices': cls._required_positive_float(
                    dataset, 'SpacingBetweenSlices'
                ),
            })
        return geometry

    @staticmethod
    def _required_positive_int(dataset: Any, attribute: str) -> int:
        """读取大于零的整数字段。"""

        if not hasattr(dataset, attribute):
            raise ValueError(f'缺少 {attribute}')
        value = int(getattr(dataset, attribute))
        if value <= 0:
            raise ValueError(f'{attribute} 必须大于 0，实际为 {value}')
        return value

    @staticmethod
    def _required_finite_float(dataset: Any, attribute: str) -> float:
        """读取有限浮点字段。"""

        if not hasattr(dataset, attribute):
            raise ValueError(f'缺少 {attribute}')
        value = float(getattr(dataset, attribute))
        if not math.isfinite(value):
            raise ValueError(f'{attribute} 不是有限数值')
        return value

    @classmethod
    def _required_positive_float(cls, dataset: Any, attribute: str) -> float:
        """读取大于零的有限浮点字段。"""

        value = cls._required_finite_float(dataset, attribute)
        if value <= 0.0:
            raise ValueError(f'{attribute} 必须大于 0，实际为 {value:g}')
        return value

    @classmethod
    def _required_float_sequence(
        cls,
        dataset: Any,
        attribute: str,
        expected_length: int,
    ) -> list[float]:
        """读取长度固定且全部有限的 DICOM 数值序列。"""

        if not hasattr(dataset, attribute):
            raise ValueError(f'缺少 {attribute}')
        raw = getattr(dataset, attribute)
        values = str(raw).split('\\') if isinstance(raw, str) else list(raw)
        if len(values) != expected_length:
            raise ValueError(
                f'{attribute} 应有 {expected_length} 项，实际为 {len(values)} 项'
            )
        numbers = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f'{attribute} 含非有限数值')
        return numbers

    @staticmethod
    def _optional_finite_float(value: Any) -> float | None:
        """把可选计划数值转换成有限浮点数。"""

        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0.0 else None

    @staticmethod
    def _numbers_close(first: float, second: float) -> bool:
        """以 DICOM DS 舍入可接受的精度比较两个几何数值。"""

        return math.isclose(float(first), float(second), rel_tol=1e-6, abs_tol=1e-6)

    @classmethod
    def _geometry_difference_fields(
        cls,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        *,
        ignored_fields: set[str] | None = None,
    ) -> list[str]:
        """返回两个几何记录中不一致的字段名。"""

        ignored_fields = ignored_fields or set()
        differences = []
        for field in baseline:
            if field in ignored_fields:
                continue
            first = baseline[field]
            second = current.get(field)
            if isinstance(first, list):
                if (
                    not isinstance(second, list)
                    or len(first) != len(second)
                    or any(
                        not cls._numbers_close(left, right)
                        for left, right in zip(first, second)
                    )
                ):
                    differences.append(field)
            elif isinstance(first, float):
                if second is None or not cls._numbers_close(first, second):
                    differences.append(field)
            elif first != second:
                differences.append(field)
        return differences

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """递归把 Path 等调试结果转换成 JSON 可序列化的基础类型。"""

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): ReportingStage._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ReportingStage._json_safe(item) for item in value]
        return value

    @staticmethod
    def _organize_output(context: ProcessingContext, report_data: Dict[str, Any]) -> None:
        """确认第七阶段目录保持原样；不搬动、不重命名最终 DICOM。"""

        DebugLog(
            "    输出目录已由第七阶段按状位/窗位/重建区域创建，"
            "第八阶段已完成硬性校验并生成交付报告"
        )
