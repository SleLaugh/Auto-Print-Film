"""
第二阶段：推理分割
模型匹配与推理
"""

from pathlib import Path
from typing import Dict, Any, Mapping
import json
import math
import re
import shlex
import shutil
import time
from tempfile import NamedTemporaryFile
from uuid import uuid4

from src.core.base_stage import BaseStage
from src.core.cache_lock import cache_lock
from src.core.cache_workspace import cache_workspace, cleanup_cache_workspaces
from src.capacity import capacity_enabled, capacity_inference, record_capacity_details
from src.core.context import ProcessingContext
from src.core.errors import ExternalProcessError
from src.core.inference_process import InferenceProcessAborted, run_inference_process
from src.core.nnunet_command import EXPORT_THREADS_ENV, nnunet_inference_launch
from src.config import (
    load_app_config,
    load_slice_part_config,
    model_mappings_for,
    validate_slice_part_config,
)
from src.models import SegmentationResult
from src.runtime import config_candidates, external_process_environment, resolve_application_path
from src.utils import DebugLog, ErrorLog, RunLog


class SegmentationStage(BaseStage):
    """推理分割阶段；在 ``-c/<模型ID>/segmentation`` 持久化分割结果。"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化分割阶段并加载统一切割部位配置及 nnUNet 运行配置。"""

        super().__init__(name="segmentation", config=config)
        self._load_config()
    
    def _load_config(self):
        """优先使用注入配置；未注入时读取候选 JSON，并用内置默认值补齐。"""
        # 默认配置
        default_config = {
            "_configuration": "3d_fullres",  # 模型配置：使用哪种网络架构配置
            "_fold": "0",  # 训练折数（fold）：使用哪个折的模型权重
            "_continue": True,  # 外部 nnUNet 跳过已有输出；阶段缓存另行判断
            "disable_tta": False,  # 是否关闭推理时的镜像增强
            "step_size": 0.5,  # 滑窗步长比例，范围为 (0, 1]
            "num_processes_preprocessing": 1,  # 限制预处理后台进程数
            "num_processes_segmentation_export": 1,  # 限制导出后台进程数
            "export_resampling_threads": 1,  # 每个导出进程的通道并行数；1 使用原命令
            "inference_timeout_seconds": 1800.0,
            "inference_timeout_seconds_by_model": {},
            "inference_max_retries": 1,
            # nnUNet 环境变量。为空时沿用系统环境变量；相对路径按应用根目录解析。
            "nnUNet_raw": "",
            "nnUNet_preprocessed": "",
            "nnUNet_results": "",
            "nnUNet_predict_command": "nnUNetv2_predict",
            "nnUNet_python": "",
        }
        
        if self.config:
            # Pipeline 已通过统一 ConfigLoader 形成生效配置，不再重复读取 JSON。
            default_config.update(self.config)
        else:
            # 未通过 Pipeline 注入配置时，直接构造 Stage 仍从标准路径加载配置。
            config_paths = config_candidates("nnuNet_config.json")
            config_loaded = False
            for config_path in config_paths:
                if not config_path.exists():
                    continue
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        default_config.update(json.load(f))
                    config_loaded = True
                    print(f"[推理与分割] 加载配置文件: {config_path}")
                    break
                except Exception as e:
                    print(f"[警告] 加载配置文件失败: {config_path}, 错误: {e}")
            if not config_loaded:
                print("[推理与分割] 未找到外部配置文件，使用默认配置")
        
        # 应用配置
        self._configuration = default_config.get('_configuration', "3d_fullres")
        self._fold = str(default_config.get('_fold', "0"))
        self._continue = default_config.get('_continue', True)
        self._predict_command = default_config.get('nnUNet_predict_command', 'nnUNetv2_predict')
        self._predict_python = default_config.get('nnUNet_python', '')
        # 仅校验新增推理选项，避免解析独立 Stage 调用方的其他应用配置。
        inference_keys = (
            "disable_tta", "step_size",
            "num_processes_preprocessing", "num_processes_segmentation_export",
            "export_resampling_threads",
            "inference_timeout_seconds", "inference_timeout_seconds_by_model",
            "inference_max_retries",
        )
        inference_options = {key: default_config[key] for key in inference_keys}
        nested_config = default_config.get("nnunet")
        if isinstance(nested_config, Mapping):
            inference_options["nnunet"] = {
                key: nested_config[key] for key in inference_keys if key in nested_config
            }
        inference_config = load_app_config(defaults=inference_options, environ={}).nnunet
        self._disable_tta = inference_config.disable_tta
        self._step_size = inference_config.step_size
        self._num_processes_preprocessing = inference_config.num_processes_preprocessing
        self._num_processes_segmentation_export = inference_config.num_processes_segmentation_export
        self._export_resampling_threads = inference_config.export_resampling_threads
        self._inference_timeout_seconds = inference_config.inference_timeout_seconds
        self._inference_timeout_seconds_by_model = (
            inference_config.inference_timeout_seconds_by_model
        )
        self._inference_max_retries = inference_config.inference_max_retries
        self._slice_part_config = (
            default_config.get('slice_part_config')
            or load_slice_part_config()
        )
        validate_slice_part_config(self._slice_part_config)
        
        # 设置nnUNet环境变量
        self._nnunet_env_vars = {
            key: resolve_application_path(default_config.get(key, ''))
            for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
        }

    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """匹配请求对应的 nnUNet 模型，执行推理并登记标准化分割输出。"""

        request = context.request
        prev_result = context.get_result('preprocessing')
        
        if not prev_result:
            raise ValueError("前置阶段 'preprocessing' 结果不存在")
        
        nifti_path = prev_result.get('nifti_path')
        if not nifti_path:
            raise ValueError("NIfTI文件路径不存在")
        
        DebugLog("=" * 60)
        DebugLog("【第二阶段】推理分割 - 模型匹配与推理")
        DebugLog(f"  输入NIfTI: {nifti_path}")
        slice_part = self._enum_value(request.slice_part)
        study_type = self._enum_value(request.study_type)
        scan_thickness = self._normalize_scan_thickness(request.scan_thickness)

        DebugLog(f"  切割部位: {slice_part}")
        DebugLog(f"  检查类型: {study_type}")
        DebugLog(f"  扫描厚度: {scan_thickness:.3f}mm")
        
        input_path = Path(nifti_path)

        # [1] match_model: 匹配模型（获取对应的 nnUNet 模型ID）
        DebugLog("  [1] match_model: 匹配模型...")
        model_info = self._match_model(
            slice_part=slice_part,
            study_type=study_type,
            scan_thickness=scan_thickness,
        )
        dataset_id = model_info['model_id']
        configuration = model_info.get('configuration', self._configuration)
        fold = model_info.get('fold', self._fold)
        DebugLog(
            f"    切割部位: {slice_part}, 检查类型: {study_type}, "
            f"扫描厚度: {scan_thickness:.3f}mm "
            f"-> 模型ID: {dataset_id}"
        )

        if request.cache_dir:
            output_dir = (
                Path(request.cache_dir)
                / str(dataset_id)
                / "segmentation"
            )
        else:
            # 未传 -c 时保留原有行为，分割结果写入 -o/segmentation。
            output_dir = Path(request.output_dir) / "segmentation"
        output_path = output_dir / "seg.nii.gz"
        record_capacity_details(model_id=str(dataset_id), segmentation_mode="unknown")

        cleanup_cache_workspaces(output_dir.parent, ".segmentation-")
        if self._can_reuse_segmentation(output_dir, output_path, dataset_id):
            record_capacity_details(segmentation_mode="cache_hit", cache_wait_seconds=0.0)
            # 已发布的缓存保持只读，多个切割部位可以同时复用。
            DebugLog(f"    已找到现成分割结果，跳过推理: {output_path}")
        else:
            lock_root = Path(request.cache_dir or request.output_dir) / ".locks"
            wait_started = time.monotonic() if capacity_enabled() else None
            with cache_lock(lock_root / f"segmentation-{dataset_id}.lock"):
                if wait_started is not None:
                    record_capacity_details(
                        cache_wait_seconds=round(time.monotonic() - wait_started, 6),
                    )
                # 等待锁的任务可能已生成缓存，必须复查，避免重复推理。
                if not self._can_reuse_segmentation(output_dir, output_path, dataset_id):
                    record_capacity_details(segmentation_mode="direct_inference")
                    self._generate_segmentation(
                        output_dir, input_path, dataset_id, configuration, fold,
                        inference_log_dir=Path(request.output_dir) / "log",
                    )
                else:
                    record_capacity_details(segmentation_mode="wait_reuse")
                    DebugLog(f"    复用其他任务已完成的分割结果: {output_path}")

        result = {
            'seg_path': str(output_path),
            'input_nifti': nifti_path,
            'model': model_info
        }
        
        context.set_stage_result(SegmentationResult(**result))
        context.metadata["model_id"] = str(dataset_id)
        
        DebugLog(f"  分割结果保存至: {output_path}")
        DebugLog("  第二阶段完成")
        DebugLog("=" * 60)
        
        return context

    def _generate_segmentation(
        self,
        output_dir: Path,
        input_path: Path,
        dataset_id: Any,
        configuration: str,
        fold: str,
        inference_log_dir: Path | None = None,
    ) -> None:
        """锁内生成独立推理结果，最后发布 seg，避免半成品被当作缓存。"""

        for attempt in range(1, self._inference_max_retries + 2):
            try:
                self._generate_segmentation_attempt(
                    output_dir, input_path, dataset_id, configuration, fold,
                    inference_log_dir=inference_log_dir, attempt=attempt,
                )
                return
            except ExternalProcessError as error:
                recoverable = error.details.get("reason") in {
                    "inference_timeout", "native_worker_crash",
                    "inference_signal", "background_worker_failed",
                }
                if (not recoverable or not error.details.get("cleanup_succeeded")
                        or attempt > self._inference_max_retries):
                    raise
                RunLog(
                    f"[nnUNet] 模型 {dataset_id} 第 {attempt} 次推理异常，"
                    "旧推理进程已清理，使用新的临时目录重试一次。"
                )

    def _generate_segmentation_attempt(
        self, output_dir: Path, input_path: Path, dataset_id: Any,
        configuration: str, fold: str, *,
        inference_log_dir: Path | None, attempt: int,
    ) -> None:
        """每次尝试使用新工作目录，避免继续模式复用上一次的半成品。"""

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with cache_workspace(output_dir.parent, ".segmentation-") as workspace:
            work_path = workspace.path
            temporary_seg = work_path / "seg.nii.gz"
            DebugLog("  [2] run_nnunet_inference: 执行nnUNet推理...")
            self._run_nnunet_inference(
                input_dir=input_path.parent,
                input_path=input_path,
                output_path=temporary_seg,
                dataset_id=dataset_id,
                configuration=configuration,
                fold=fold,
                continue_mode=self._continue,
                cache_pass_fds=workspace.pass_fds,
                inference_log_dir=inference_log_dir,
                attempt=attempt,
            )
            if not temporary_seg.is_file():
                raise FileNotFoundError(f"未找到nnUNet输出文件: {temporary_seg}")
            self._write_model_marker(work_path, dataset_id)

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "seg.nii.gz"
            # 此处只处理已确认失效的缓存。先移除旧seg，再发布标记和完整seg；
            # 即使发布中途退出，也不会把旧seg与新标记配对后误判为命中。
            output_path.unlink(missing_ok=True)
            # 新命令可能只输出seg，旧dataset.json等附属文件不能留给新分割使用。
            for child in output_dir.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            for child in work_path.iterdir():
                if (child.is_file() and child != temporary_seg
                        and not child.name.startswith(".cache-workspace.")):
                    child.replace(output_dir / child.name)
            temporary_seg.replace(output_path)

    @staticmethod
    def _can_reuse_segmentation(
        output_dir: Path,
        output_path: Path,
        dataset_id: Any,
    ) -> bool:
        """按模型目录中的分割文件和标记判断第二阶段缓存是否可复用。"""

        if not output_path.is_file():
            return False
        marker = output_dir / "model.json"
        if not marker.is_file():
            # 兼容没有模型标记的历史缓存：文件存在即可复用。
            return output_path.is_file()
        try:
            with marker.open("r", encoding="utf-8") as file:
                matches_model = str(json.load(file).get("model_id")) == str(dataset_id)
            # 读取标记期间失效缓存可能正在被替换，最终seg必须仍已发布。
            return matches_model and output_path.is_file()
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _write_model_marker(output_dir: Path, dataset_id: Any) -> None:
        """记录模型缓存对应的模型，避免模型切换时误复用。"""

        marker = output_dir / "model.json"
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir,
                prefix=".model.", suffix=".tmp", delete=False,
            ) as file:
                temporary = Path(file.name)
                json.dump({"model_id": str(dataset_id)}, file, ensure_ascii=False)
            temporary.replace(marker)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    
    def _match_model(
        self,
        slice_part: str,
        study_type: str,
        scan_thickness: float,
    ) -> Dict[str, Any]:
        """
        根据切割部位、检查类型和扫描厚度区间匹配对应模型。

        模型表从统一切割部位配置读取，再按扫描厚度闭区间匹配 model_id。
        
        Args:
            slice_part: 切割部位名称（如 "腰椎"、"颅脑"、"肺部"）
            study_type: 检查类型（CT、MR）
            scan_thickness: 当前原始扫描层厚，单位 mm
            
        Returns:
            dict: 模型信息
        """
        normalized_study_type = (study_type or "").strip()
        normalized_scan_thickness = self._normalize_scan_thickness(scan_thickness)
        mappings = model_mappings_for(
            self._slice_part_config,
            slice_part,
            normalized_study_type,
        )
        model_mapping = self._match_model_mapping(
            mappings,
            normalized_scan_thickness,
            slice_part=slice_part,
            study_type=normalized_study_type,
        )
        thickness_range = model_mapping['scan_thickness_range_mm']
        model_id = model_mapping['model_id']
        return {
            'model_id': model_id,
            'dataset_id': model_id,
            'scan_thickness': normalized_scan_thickness,
            'min_scan_thickness': float(thickness_range['min']),
            'max_scan_thickness': float(thickness_range['max']),
            'configuration': model_mapping.get(
                'configuration',
                self._configuration,
            ),
            'fold': str(model_mapping.get('fold', self._fold)),
            'matched_by': 'slice_part_config',
        }

    @staticmethod
    def _match_model_mapping(
        mappings,
        scan_thickness: float,
        *,
        slice_part: str = "",
        study_type: str = "",
    ) -> Dict[str, Any]:
        """
        在单条模型规则中按闭区间匹配扫描厚度对应的模型。

        模型配置示例：
            [
                {
                    "scan_thickness_range_mm": {"min": 0.0, "max": 1.5},
                    "model_id": 292
                }
            ]

        上下限均包含在匹配范围内。配置重叠会导致选模不确定，因此直接报错。
        """
        if not mappings:
            raise ValueError("模型规则不能为空")

        parsed_mappings = []
        for item in mappings:
            if 'model_id' not in item:
                raise ValueError(f"模型规则缺少 model_id: {item}")
            thickness_range = item.get('scan_thickness_range_mm')
            if not isinstance(thickness_range, dict):
                raise ValueError(
                    f"模型规则缺少 scan_thickness_range_mm: {item}"
                )
            minimum = SegmentationStage._parse_model_bound(
                thickness_range,
                'min',
            )
            maximum = SegmentationStage._parse_model_bound(
                thickness_range,
                'max',
            )
            if minimum > maximum:
                raise ValueError(
                    "模型规则扫描厚度下限不能大于上限: "
                    f"{item}"
                )
            parsed_mappings.append({
                **item,
                'scan_thickness_range_mm': {
                    'min': minimum,
                    'max': maximum,
                },
            })

        for index, current in enumerate(parsed_mappings):
            for previous in parsed_mappings[:index]:
                if max(
                    current['scan_thickness_range_mm']['min'],
                    previous['scan_thickness_range_mm']['min'],
                ) <= min(
                    current['scan_thickness_range_mm']['max'],
                    previous['scan_thickness_range_mm']['max'],
                ):
                    raise ValueError(
                        "模型规则扫描厚度区间重叠: "
                        f"{previous} 与 {current}"
                    )

        matches = [
            item
            for item in parsed_mappings
            if item['scan_thickness_range_mm']['min']
            <= scan_thickness
            <= item['scan_thickness_range_mm']['max']
        ]
        if matches:
            return matches[0]

        raise ValueError(
            "未找到匹配的扫描厚度模型: "
            f"slice_part={slice_part}, study_type={study_type}, "
            f"ScanThickness={scan_thickness}mm"
        )

    @staticmethod
    def _parse_model_bound(item: Dict[str, Any], field_name: str) -> float:
        """读取并校验模型映射中的扫描厚度边界。"""

        if field_name not in item or isinstance(item[field_name], bool):
            raise ValueError(f"模型规则缺少有效 {field_name}: {item}")
        try:
            value = float(item[field_name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"模型规则的 {field_name} 必须是非负有限数值: {item}"
            ) from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"模型规则的 {field_name} 必须是非负有限数值: {item}"
            )
        return value

    @staticmethod
    def _normalize_scan_thickness(scan_thickness: Any) -> float:
        """把请求中的当前扫描层厚规范为非负有限毫米数值。"""

        if isinstance(scan_thickness, bool):
            raise ValueError("ScanThickness 必须是非负有限数值")
        try:
            value = float(scan_thickness)
        except (TypeError, ValueError) as error:
            raise ValueError("ScanThickness 必须是非负有限数值") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError("ScanThickness 必须是非负有限数值")
        return value

    @staticmethod
    def _enum_value(value: Any) -> str:
        """
        兼容枚举和普通字符串。

        CLI 会构造枚举类型；单元测试或外部调用有时可能直接传字符串。
        统一转换后，后续匹配逻辑只处理字符串，代码会简单一些。
        """
        return str(getattr(value, 'value', value))
    
    def _run_nnunet_inference(self, 
                              input_dir: Path, 
                              input_path: Path,
                              output_path: Path, 
                              dataset_id: str, 
                              configuration: str, 
                              fold: str, 
                              continue_mode: bool,
                              cache_pass_fds: tuple[int, ...] = (),
                              inference_log_dir: Path | None = None,
                              attempt: int = 1):
        """
        执行nnUNet推理命令
        
        Args:
            input_dir: 输入目录（包含NIfTI文件）
            input_path: 输入 NIfTI 文件路径，文件名应为 {case_id}_0000.nii.gz
            output_path: 统一整理后的输出文件路径
            dataset_id: 数据集ID
            configuration: 模型配置（如 3d_fullres）
            fold: 训练折数
            continue_mode: 是否启用继续模式
            cache_pass_fds: POSIX预测子进程继承工作目录租约，父进程退出后仍保护写入目录
            inference_log_dir: 独立于临时缓存的原始推理日志目录，失败及重试日志均保留
            attempt: 当前尝试序号
        """
        # 构建命令
        prefix, export_adapter_enabled, export_adapter_reason = nnunet_inference_launch(
            self._predict_command, self._predict_python, self._export_resampling_threads,
        )
        if self._export_resampling_threads > 1:
            RunLog(
                f"[nnUNet] 导出通道并行适配器已启用，线程数 {self._export_resampling_threads}"
                if export_adapter_enabled else f"[nnUNet] 导出通道并行未启用：{export_adapter_reason}"
            )
        cmd = prefix + [
            "-i", str(input_dir),
            "-o", str(output_path.parent),  # 输出目录
            "-d", str(dataset_id),
            "-c", configuration,
            "-f", fold,
            "-step_size", str(self._step_size),
            "-npp", str(self._num_processes_preprocessing),
            "-nps", str(self._num_processes_segmentation_export),
        ]
        if self._disable_tta:
            cmd.append("--disable_tta")
        
        # 只控制 nnUNet 的已有输出；本阶段的分割缓存由 execute() 独立判断。
        if continue_mode:
            cmd.append("--continue_prediction")
        
        # 设置环境变量（合并当前环境变量和nnUNet环境变量）
        env = external_process_environment()
        # 所有 spawn worker 都继承诊断开关；原生崩溃及时打印 Python 调用栈。
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if export_adapter_enabled:
            env[EXPORT_THREADS_ENV] = str(self._export_resampling_threads)
        for key, value in self._nnunet_env_vars.items():
            if value:  # 如果值不为空
                env[key] = value
                DebugLog(f"    设置环境变量: {key}={value}")
        
        command_text = shlex.join(cmd)
        DebugLog(f"    执行命令: {command_text}")
        timeout_seconds = self._inference_timeout_seconds_by_model.get(
            str(dataset_id), self._inference_timeout_seconds,
        )
        log_dir = Path(inference_log_dir or output_path.parent / "log")
        log_name = f"nnunet-{dataset_id}-{uuid4().hex}-attempt-{attempt}"
        stdout_path = log_dir / f"{log_name}.stdout.log"
        stderr_path = log_dir / f"{log_name}.stderr.log"
        RunLog(
            f"[nnUNet] 模型 {dataset_id} 第 {attempt} 次推理开始，"
            f"单次超时 {timeout_seconds:g}s；"
            f"标准输出: {stdout_path}；标准错误: {stderr_path}"
        )
        
        try:
            # 执行命令
            with capacity_inference(model_id=dataset_id, attempt=attempt) as capacity_result:
                result = run_inference_process(
                    cmd,
                    env=env,
                    timeout_seconds=timeout_seconds,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    pass_fds=cache_pass_fds,
                )
                capacity_result["outcome"] = "succeeded" if result.returncode == 0 else "failed"
            
            # 检查执行结果
            if result.returncode != 0:
                # 报告只携带可读摘要，完整输出保存在独立日志，错误日志保留末尾。
                ErrorLog(
                    f"nnUNet 推理失败：模型 {dataset_id}，退出码 {result.returncode}\n"
                    f"执行命令: {command_text}\n"
                    f"标准错误日志: {stderr_path}\n标准输出日志: {stdout_path}\n"
                    f"标准错误末尾:\n{result.stderr or ''}\n"
                    f"标准输出末尾:\n{result.stdout or ''}"
                )
                error = self._describe_inference_failure(
                    dataset_id, result.returncode, result.stderr or '', env,
                    stdout=result.stdout or '',
                )
                error.details.update(
                    cleanup_succeeded=getattr(result, "cleanup_succeeded", True), attempt=attempt,
                    stdout_log=stdout_path.name, stderr_log=stderr_path.name,
                )
                if result.returncode < 0:
                    error.details["reason"] = "inference_signal"
                raise error
            
            DebugLog("    ✅ nnUNet推理完成")
            RunLog(f"[nnUNet] 模型 {dataset_id} 第 {attempt} 次推理完成。")
            
            case_id = self._get_case_id(input_path)
            expected_output = output_path.parent / f"{case_id}.nii.gz"
            self._normalize_output_file(expected_output, output_path)
            
        except InferenceProcessAborted as error:
            reasons = {
                "timeout": ("inference_timeout", f"超过单次推理时限 {timeout_seconds:g}s"),
                "native_worker_crash": ("native_worker_crash", "推理进程发生原生崩溃"),
                "cleanup_failed": ("process_cleanup_failed", "推理进程未能完全清理"),
            }
            reason, description = reasons.get(
                error.reason, ("inference_failed", "外部推理中止"),
            )
            cleanup = (
                "已清理本次推理进程。" if error.cleanup_succeeded
                else "未确认本次推理进程全部退出，停止自动重试。"
            )
            message = f"nnUNet 模型 {dataset_id} {description}；{cleanup}详细输出见任务 log 目录。"
            ErrorLog(
                f"{message}\n执行命令: {command_text}\n"
                f"标准错误日志: {stderr_path}\n标准输出日志: {stdout_path}\n"
                f"标准错误末尾:\n{error.stderr or ''}\n标准输出末尾:\n{error.stdout or ''}"
            )
            raise ExternalProcessError(
                message, stage="segmentation", details={
                    "model_id": int(dataset_id), "reason": reason,
                    "timeout_seconds": timeout_seconds, "attempt": attempt,
                    "cleanup_succeeded": error.cleanup_succeeded,
                    "stdout_log": stdout_path.name, "stderr_log": stderr_path.name,
                },
            ) from error
        except ExternalProcessError:
            # 失败进程的原始输出已记录，避免再把业务摘要和堆栈重复写入日志。
            raise
        except Exception as e:
            ErrorLog("    ❌ nnUNet推理过程中发生错误", e)
            raise

    @staticmethod
    def _describe_inference_failure(
        dataset_id: str | int,
        returncode: int,
        stderr: str,
        env: Dict[str, str],
        stdout: str = '',
    ) -> ExternalProcessError:
        """根据 nnUNet 明确的失败原因生成报告消息；不硬编码模型号或本机路径。"""

        model_id = int(dataset_id)
        directories = {
            key: env.get(key) or "未配置"
            for key in ("nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed")
        }
        details = {"model_id": model_id, "returncode": returncode}
        output = '\n'.join((stderr, stdout))
        missing = re.search(r"Could not find a dataset with the ID (\d+)\b", output)
        if missing and int(missing.group(1)) == model_id:
            # 只翻译 nnUNet 已确认的当前数据集缺失，不把权重缺失或其他异常混为一谈。
            locations = "；".join(f"{key}={value}" for key, value in directories.items())
            message = (
                f"nnUNet 推理失败：未找到模型 {model_id}"
                f"（未找到 Dataset{model_id:03d}* 数据集目录）。"
                f"查找目录：{locations}。"
                "请检查 nnUNet 路径配置，或部署该模型对应的完整训练结果。"
            )
            details.update(reason="dataset_not_found", directories=directories)
        else:
            stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            worker_failed = 'Background workers died' in output
            # worker 退出只是外层通知；优先保留两个输出流中更早的具体异常。
            exceptions = [
                line for line in lines
                if re.match(r'^(?:\w+\.)*\w*(?:Error|Exception)(?::|$)', line)
                and 'Background workers died' not in line
            ]
            # 管道断开通常是 worker 退出后的次生错误；另一个输出流若有
            # 文件读取、内存分配等具体异常，应优先显示它。
            primary_exceptions = [
                line for line in exceptions
                if line.split(':', 1)[0].rsplit('.', 1)[-1] not in {
                    'EOFError', 'BrokenPipeError', 'ConnectionResetError',
                }
            ]
            if worker_failed and not primary_exceptions:
                message = (
                    f"nnUNet 推理失败（模型 {model_id}，退出码 {returncode}）："
                    "推理后台进程异常退出，可能是进程内部异常或内存不足导致被系统终止。"
                    "完整输出及更早的异常请查看任务 log 目录中的 "
                    "nnunet-*.stdout.log 和 nnunet-*.stderr.log。"
                )
                details['reason'] = 'background_worker_failed'
                return ExternalProcessError(message, stage='segmentation', details=details)
            candidates = primary_exceptions or exceptions
            if candidates:
                stderr_candidates = [line for line in stderr_lines if line in candidates]
                reason = candidates[0] if worker_failed else (stderr_candidates or candidates)[-1]
            else:
                # 普通 stdout 进度不能覆盖 stderr 中的失败说明。
                reason = (stderr_lines or lines)[-1] if lines else "外部进程未提供错误说明"
            if len(reason) > 300:
                reason = reason[:300] + "…"
            message = (
                f"nnUNet 推理失败（模型 {model_id}，退出码 {returncode}）：{reason}。"
                "详细输出请查看任务 log 目录中的 "
                "nnunet-*.stdout.log 和 nnunet-*.stderr.log。"
            )
            details["reason"] = "inference_failed"
        return ExternalProcessError(message, stage="segmentation", details=details)

    @staticmethod
    def _get_case_id(input_path: Path) -> str:
        """
        从 nnUNet 输入文件名中还原 case_id。

        nnUNet 输入要求单通道文件名形如 {case_id}_0000.nii.gz；
        推理输出则通常是 {case_id}.nii.gz。比如 raw_0000.nii.gz
        对应输出 raw.nii.gz。
        """
        filename = input_path.name
        if filename.endswith(".nii.gz"):
            filename = filename[:-7]
        else:
            filename = input_path.stem
        if filename.endswith("_0000"):
            filename = filename[:-5]
        return filename

    @staticmethod
    def _normalize_output_file(expected_output: Path, output_path: Path) -> None:
        """
        将 nnUNet 输出整理成固定文件名。

        第二阶段统一在 -c/<模型ID>/segmentation/seg.nii.gz 生成或复用结果。
        nnUNet 原始输出通常是 {case_id}.nii.gz，所以这里先找预期输出；如果
        不同版本或不同参数导致文件名不一致，再在输出目录中寻找唯一的 nii.gz 结果。
        """
        if expected_output.exists():
            if expected_output != output_path:
                if output_path.exists():
                    output_path.unlink()
                shutil.move(str(expected_output), str(output_path))
            return

        if output_path.exists():
            return

        possible_outputs = [
            path for path in output_path.parent.glob("*.nii.gz")
            if path != output_path
        ]
        if len(possible_outputs) == 1:
            found_file = possible_outputs[0]
            shutil.move(str(found_file), str(output_path))
            return

        if len(possible_outputs) > 1:
            raise RuntimeError(f"nnUNet 输出目录包含多个候选结果，无法确定当前病例: {output_path.parent}")

        raise FileNotFoundError(f"未找到nnUNet输出文件: {output_path.parent}")
