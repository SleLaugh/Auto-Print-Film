"""
把默认文件、环境变量和显式参数合并成唯一应用配置。

加载器只负责默认文件、环境变量和显式参数的合并，不检查目录是否存在；真实环境检查由
PreflightValidator 完成，确保配置读取和业务验证职责分离。
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.runtime import config_path

from .settings import AppConfig, LoggingConfig, NnUNetConfig, PipelineConfig
from .slice_parts import validate_slice_part_config


def _first_nonempty(*values: Any, fallback: Any = "") -> Any:
    """按传入顺序选择第一个非 None、非空字符串的配置值。"""
    for value in values:
        if value is not None and value != "":
            return value
    return fallback


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """安全读取嵌套配置段，非映射值按空段处理。"""
    value = data.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _coerce_bool(value: Any, field_name: str) -> bool:
    """严格解析配置布尔值，避免 ``bool('false')`` 被误判为 True。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field_name} 必须是布尔值 true/false")


def _coerce_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是命令或解释器路径字符串")
    return value


def _coerce_float(value: Any, field_name: str) -> float:
    """接受 JSON 数值或环境变量中的数值文本，拒绝布尔值。"""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是数值")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} 必须是数值") from error


def _coerce_int(value: Any, field_name: str) -> int:
    """接受整数及整数文本，拒绝布尔值、浮点数和带小数点的文本。"""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
        try:
            return int(value.strip())
        except ValueError as error:
            raise ValueError(f"{field_name} 必须是正整数") from error
    raise ValueError(f"{field_name} 必须是正整数")


def _coerce_timeout_map(value: Any) -> Dict[str, float]:
    """环境变量支持 JSON 对象文本；其他来源也可直接传入映射。"""
    field_name = "nnunet.inference_timeout_seconds_by_model"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field_name} 必须是有效 JSON 对象") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} 必须是模型 ID 到秒数的对象")
    return {
        key: _coerce_float(timeout, f"{field_name}.{key}")
        for key, timeout in value.items()
    }


def load_app_config(
    *,
    explicit: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    default_path: Optional[Path] = None,
) -> AppConfig:
    """
    按显式参数、环境变量、默认文件的优先级加载不可变配置。

    Args:
        explicit: Python 调用方显式传入的最高优先级配置。
        environ: 环境变量映射；测试可注入，生产默认使用 os.environ。
        defaults: 已解析的默认配置；提供后不会读取磁盘。
        default_path: nnUNet 默认配置路径，日志配置读取同目录文件。
    """

    explicit_data = dict(explicit or {})
    if "study_part_config" in explicit_data:
        raise ValueError("应用配置包含已移除字段: study_part_config")
    environment = os.environ if environ is None else environ
    default_data = dict(
        _read_defaults(default_path) if defaults is None else defaults
    )
    if "study_part_config" in default_data:
        raise ValueError("默认配置包含已移除字段: study_part_config")
    explicit_nnunet = _section(explicit_data, "nnunet")
    default_nnunet = _section(default_data, "nnunet")

    def nn_value(name: str, flat_name: str, *, fallback: Any = "") -> Any:
        """合并嵌套调用参数、环境变量和当前扁平 JSON 字段。"""
        return _first_nonempty(
            explicit_nnunet.get(name),
            explicit_data.get(flat_name),
            environment.get(flat_name),
            default_nnunet.get(name),
            default_data.get(flat_name),
            fallback=fallback,
        )

    explicit_logging = _section(explicit_data, "logging")
    default_logging = _section(default_data, "logging")
    explicit_pipeline = _section(explicit_data, "pipeline")
    default_pipeline = _section(default_data, "pipeline")

    nnunet = NnUNetConfig(
        raw=str(nn_value("raw", "nnUNet_raw")),
        preprocessed=str(nn_value("preprocessed", "nnUNet_preprocessed")),
        results=str(nn_value("results", "nnUNet_results")),
        configuration=str(nn_value(
            "configuration", "_configuration", fallback="3d_fullres"
        )),
        fold=str(nn_value("fold", "_fold", fallback="0")),
        continue_mode=_coerce_bool(nn_value(
            "continue_mode", "_continue", fallback=True
        ), "nnunet.continue_mode"),
        predict_command=_coerce_path(nn_value(
            "predict_command", "nnUNet_predict_command", fallback="nnUNetv2_predict"
        ), "nnunet.predict_command"),
        python_executable=_coerce_path(nn_value(
            "python_executable", "nnUNet_python"
        ), "nnunet.python_executable"),
        disable_tta=_coerce_bool(nn_value(
            "disable_tta", "disable_tta", fallback=False
        ), "nnunet.disable_tta"),
        step_size=_coerce_float(nn_value(
            "step_size", "step_size", fallback=0.5
        ), "nnunet.step_size"),
        num_processes_preprocessing=_coerce_int(nn_value(
            "num_processes_preprocessing", "num_processes_preprocessing", fallback=1
        ), "nnunet.num_processes_preprocessing"),
        num_processes_segmentation_export=_coerce_int(nn_value(
            "num_processes_segmentation_export", "num_processes_segmentation_export", fallback=1
        ), "nnunet.num_processes_segmentation_export"),
        export_resampling_threads=_coerce_int(nn_value(
            "export_resampling_threads", "export_resampling_threads", fallback=1
        ), "nnunet.export_resampling_threads"),
        inference_timeout_seconds=_coerce_float(nn_value(
            "inference_timeout_seconds", "inference_timeout_seconds", fallback=1800.0
        ), "nnunet.inference_timeout_seconds"),
        inference_timeout_seconds_by_model=_coerce_timeout_map(nn_value(
            "inference_timeout_seconds_by_model", "inference_timeout_seconds_by_model", fallback={}
        )),
        inference_max_retries=_coerce_int(nn_value(
            "inference_max_retries", "inference_max_retries", fallback=1
        ), "nnunet.inference_max_retries"),
    )
    logging_values = {
        key: explicit_logging.get(key, default_logging.get(key, default))
        for key, default in {
            "debug_file_enabled": False,
            "console_enabled": True,
            "run_log_file_enabled": False,
            "debug_log_file_enabled": False,
            "error_log_file_enabled": True,
            "base_dir": "Log",
        }.items()
    }
    for key in (
        "debug_file_enabled",
        "console_enabled",
        "run_log_file_enabled",
        "debug_log_file_enabled",
        "error_log_file_enabled",
    ):
        logging_values[key] = _coerce_bool(logging_values[key], f"logging.{key}")
    logging = LoggingConfig(**logging_values)
    pipeline_values = {
        key: explicit_pipeline.get(key, default_pipeline.get(key, default))
        for key, default in {
            "fail_fast": True,
            "keep_intermediate_on_success": False,
            "keep_intermediate_on_failure": True,
            "run_preflight": True,
        }.items()
    }
    for key in (
        "fail_fast",
        "keep_intermediate_on_success",
        "keep_intermediate_on_failure",
        "run_preflight",
    ):
        pipeline_values[key] = _coerce_bool(pipeline_values[key], f"pipeline.{key}")
    pipeline = PipelineConfig(**pipeline_values)
    slice_part_config = explicit_data.get(
        "slice_part_config",
        default_data.get("slice_part_config", {}),
    )
    if slice_part_config:
        validate_slice_part_config(slice_part_config)
    return AppConfig(
        nnunet=nnunet,
        logging=logging,
        pipeline=pipeline,
        slice_part_config=dict(slice_part_config),
    )


def _read_defaults(default_path: Optional[Path]) -> Dict[str, Any]:
    """读取同一目录中的 nnUNet、日志和统一切割部位配置并合并。"""
    path = default_path or config_path("nnuNet_config.json")
    data: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            data.update(json.load(file))
    logger_path = path.with_name("logger_config.json")
    if logger_path.exists():
        with logger_path.open("r", encoding="utf-8") as file:
            data["logging"] = json.load(file)
    slice_part_path = path.with_name("slice_part_config.json")
    if slice_part_path.exists():
        with slice_part_path.open("r", encoding="utf-8") as file:
            data["slice_part_config"] = json.load(file)
    return data
