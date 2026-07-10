"""
把默认文件、环境变量和显式参数合并成唯一应用配置。

加载器只负责来源合并和旧字段归一化，不检查目录是否存在；真实环境检查由
PreflightValidator 完成，确保配置读取和业务验证职责分离。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .settings import AppConfig, LoggingConfig, NnUNetConfig, PipelineConfig


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
    environment = os.environ if environ is None else environ
    default_data = dict(defaults or _read_defaults(default_path))
    explicit_nnunet = _section(explicit_data, "nnunet")
    default_nnunet = _section(default_data, "nnunet")

    def nn_value(name: str, legacy_name: str, *, fallback: Any = "") -> Any:
        """兼容新嵌套字段和旧 JSON 字段名，并应用统一优先级。"""
        return _first_nonempty(
            explicit_nnunet.get(name),
            explicit_data.get(legacy_name),
            environment.get(legacy_name),
            default_nnunet.get(name),
            default_data.get(legacy_name),
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
        continue_mode=bool(nn_value(
            "continue_mode", "_continue", fallback=True
        )),
        model_rules=tuple(nn_value(
            "model_rules", "_model_rules", fallback=[]
        )),
    )
    logging = LoggingConfig(**{
        key: explicit_logging.get(key, default_logging.get(key, default))
        for key, default in {
            "console_enabled": True,
            "run_log_file_enabled": False,
            "debug_log_file_enabled": False,
            "error_log_file_enabled": True,
            "base_dir": "Log",
        }.items()
    })
    pipeline = PipelineConfig(**{
        key: explicit_pipeline.get(key, default_pipeline.get(key, default))
        for key, default in {
            "fail_fast": True,
            "keep_intermediate_on_success": False,
            "keep_intermediate_on_failure": True,
            "run_preflight": True,
        }.items()
    })
    return AppConfig(nnunet=nnunet, logging=logging, pipeline=pipeline)


def _read_defaults(default_path: Optional[Path]) -> Dict[str, Any]:
    """读取同一配置目录中的 nnUNet 和日志默认文件并合并。"""
    path = default_path or Path(__file__).with_name("nnuNet_config.json")
    data: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            data.update(json.load(file))
    logger_path = path.with_name("logger_config.json")
    if logger_path.exists():
        with logger_path.open("r", encoding="utf-8") as file:
            data["logging"] = json.load(file)
    return data
