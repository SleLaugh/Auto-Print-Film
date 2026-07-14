#!/src/stages/stage_reporting.py
# # -*- coding: utf-8 -*-

"""
阶段六：结果汇总
生成报告与输出整理
"""

import json
import re
from pathlib import Path
from typing import Dict, Any

import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.models import ReportingStageResult
from src.utils import DebugLog, is_debug_log_file_enabled


class ReportingStage(BaseStage):
    """结果汇总阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="reporting", config=config)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        
        DebugLog("=" * 60)
        DebugLog("【阶段六】结果汇总 - 生成报告")
        DebugLog(f"  输出目录: {request.output_dir}")
        DebugLog("=" * 60)
        
        # [1] generate_report: 生成报告
        DebugLog("  [1] generate_report: 生成处理报告...")
        report_data = self._build_delivery_report(context)
        report_path = self._write_report(context, report_data)

        validation = self._validate_reconstruction(
            context.get_result('reconstruction') or {}
        )
        debug_report_path = ""
        if is_debug_log_file_enabled():
            DebugLog("  [2] generate_debug_report: 生成调试报告...")
            debug_report_path = str(self._write_debug_report(
                context,
                self._build_debug_report(context, validation),
            ))
        
        # [3] organize_output: 确认第五步目录保持原样
        DebugLog("  [3] organize_output: 确认输出目录结构...")
        self._organize_output(context, report_data)

        result = {
            'report_path': str(report_path),
            'debug_report_path': debug_report_path,
            'total_time': context.get_total_time(),
            'validation': validation,
            'image_count': sum(
                int(config['image_count'])
                for config in report_data['plane_configs']
            ),
        }
        
        context.set_stage_result(ReportingStageResult(payload=result))
        
        DebugLog(f"  ✅ 阶段六完成，总耗时: {context.get_total_time():.2f}s")
        DebugLog("=" * 60)
        
        return context
    
    def _generate_report(self, context: ProcessingContext) -> Path:
        """生成简洁、稳定的正常交付报告。"""

        return self._write_report(context, self._build_delivery_report(context))

    def _build_delivery_report(self, context: ProcessingContext) -> Dict[str, Any]:
        """只保留调用方交付时需要的部位、检查类型和各状位结果。"""

        request_data = context.request.to_dict()
        reconstruction = context.get_result('reconstruction') or {}
        plane_summaries = reconstruction.get('planes', {})
        actual_windows = self._actual_window_names_by_plane(reconstruction)
        output_root = Path(context.request.output_dir)

        plane_configs = []
        for config in request_data.get('plane_configs', []):
            plane = str(config['plane'])
            summary = plane_summaries.get(plane, {})
            image_count = int(summary.get('slice_count', 0))
            window_presets = []
            for window_name in config.get('window_presets', []):
                has_output = window_name in actual_windows.get(plane, set())
                window_presets.append({
                    'name': window_name,
                    'path': str(output_root / plane / window_name) if has_output else '',
                })

            plane_configs.append({
                'plane': plane,
                'slice_thickness': float(config['slice_thickness']),
                'slice_spacing': float(config['slice_spacing']),
                'line_extent_mode': config['line_extent_mode'],
                # 表示实际切片张数；多个窗位共享同一批切片，不重复累计。
                'image_count': image_count,
                'window_presets': window_presets,
            })

        return {
            'body_part': request_data['body_part'],
            'check_type': request_data['check_type'],
            'plane_configs': plane_configs,
        }

    @staticmethod
    def _actual_window_names_by_plane(
        reconstruction: Dict[str, Any],
    ) -> Dict[str, set[str]]:
        result: Dict[str, set[str]] = {}
        for output in reconstruction.get('outputs', []):
            plane = str(output.get('plane', ''))
            result.setdefault(plane, set()).update(
                str(name)
                for name in (output.get('window_outputs') or {})
            )
        return result

    def _build_debug_report(
        self,
        context: ProcessingContext,
        validation: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """汇总完整阶段结果，供 debug 模式排查问题。"""

        output_dir = Path(context.request.output_dir)
        mapping = context.get_result('mapping') or {}
        reconstruction = context.get_result('reconstruction') or {}
        validation = validation or self._validate_reconstruction(reconstruction)

        mapping_plans = mapping.get('plans', [])
        mapping_summary = {
            'planner': mapping.get('planner', ''),
            'plan_count': mapping.get('plan_count', len(mapping_plans)),
            'warnings': list(mapping.get('warnings', [])),
            'diagnostics': list(mapping.get('diagnostics', [])),
            'metadata_path': mapping.get('metadata_path', ''),
            'plans': [self._plan_summary(plan) for plan in mapping_plans],
        }
        reconstruction_summary = self._reconstruction_summary(reconstruction)
        warnings = list(mapping_summary['warnings'])
        for plan in mapping_summary['plans']:
            warnings.extend(plan['warnings'])
        warnings.extend(validation['warnings'])

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
            'mapping': mapping_summary,
            'reconstruction': reconstruction_summary,
            'validation': validation,
            'outputs': {
                'report': str(output_dir / 'report.json'),
                'nifti': str(output_dir / 'nifti'),
                'metadata': mapping.get('metadata_path', ''),
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
        output_dir = Path(context.request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        DebugLog(f"    报告已生成: {report_path}")
        return report_path

    @staticmethod
    def _write_debug_report(
        context: ProcessingContext,
        report_data: Dict[str, Any],
    ) -> Path:
        metadata_dir = Path(context.request.output_dir) / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)
        report_path = metadata_dir / '处理报告_调试.json'
        with open(report_path, 'w', encoding='utf-8') as file:
            json.dump(report_data, file, ensure_ascii=False, indent=2)
        DebugLog(f"    调试报告已生成: {report_path}")
        return report_path

    @staticmethod
    def _plan_summary(plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'plan_id': plan.get('plan_id', ''),
            'body_part': plan.get('body_part', ''),
            'cut_mode': plan.get('cut_mode', ''),
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
        errors: list[str] = []
        warnings: list[str] = []
        series_results = []
        outputs = reconstruction.get('outputs', [])
        for output in outputs:
            expected_count = int(output.get('slice_count', 0))
            for window_name, directory in (output.get('window_outputs') or {}).items():
                result = cls._validate_dicom_series(
                    plan_id=str(output.get('plan_id', '')),
                    window_name=str(window_name),
                    directory=Path(directory),
                    expected_count=expected_count,
                )
                series_results.append(result)
                errors.extend(result['errors'])
                warnings.extend(result['warnings'])

        if reconstruction.get('slice_count', 0) and not outputs:
            errors.append('第五步报告了切片数量，但没有实际输出方案')

        return {
            'valid': not errors,
            'errors': errors,
            'warnings': warnings,
            'series': series_results,
        }

    @staticmethod
    def _validate_dicom_series(
        *,
        plan_id: str,
        window_name: str,
        directory: Path,
        expected_count: int,
    ) -> Dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        numbered_files: list[tuple[int, Path]] = []
        if directory.exists():
            for file_path in directory.glob('*.dcm'):
                match = re.fullmatch(r'(\d+)\.dcm', file_path.name)
                if match:
                    numbered_files.append((int(match.group(1)), file_path))
        numbered_files.sort(key=lambda item: item[0])
        files = [file_path for _, file_path in numbered_files]
        actual_count = len(files)
        if not directory.exists():
            errors.append(f'{plan_id}/{window_name} 输出目录不存在: {directory}')
        if actual_count != expected_count:
            errors.append(
                f'{plan_id}/{window_name} 期望 {expected_count} 张，实际 {actual_count} 张'
            )

        indices = []
        series_uids = set()
        instance_numbers = []
        for file_path in files:
            match = re.fullmatch(r'(\d+)\.dcm', file_path.name)
            if not match:
                errors.append(f'{plan_id}/{window_name} 文件名格式错误: {file_path.name}')
                continue
            indices.append(int(match.group(1)))
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(file_path))
                reader.ReadImageInformation()
                if reader.HasMetaDataKey('0020|000e'):
                    series_uids.add(reader.GetMetaData('0020|000e').strip())
                if reader.HasMetaDataKey('0020|0013'):
                    instance_numbers.append(int(reader.GetMetaData('0020|0013').strip()))
            except (RuntimeError, ValueError) as exc:
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
        if len(instance_numbers) != actual_count:
            errors.append(f'{plan_id}/{window_name} 存在缺少 InstanceNumber 的 DICOM')
        elif instance_numbers != expected_indices:
            errors.append(
                f'{plan_id}/{window_name} InstanceNumber 不连续，实际为 {instance_numbers}'
            )

        return {
            'plan_id': plan_id,
            'window': window_name,
            'directory': str(directory),
            'expected_count': expected_count,
            'actual_count': actual_count,
            'series_instance_uid': next(iter(series_uids), ''),
            'instance_numbers': instance_numbers,
            'valid': not errors,
            'errors': errors,
            'warnings': warnings,
        }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): ReportingStage._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ReportingStage._json_safe(item) for item in value]
        return value

    @staticmethod
    def _organize_output(context: ProcessingContext, report_data: Dict[str, Any]) -> None:
        """确认第五步目录保持原样；不搬动、不重命名最终 DICOM。"""

        DebugLog(
            "    输出目录已由阶段五按状位/窗位/切割组创建，"
            "阶段六仅生成报告和校验清单"
        )
