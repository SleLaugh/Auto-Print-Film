"""
Pipeline 执行前的无副作用环境检查。

所有检查必须发生在创建输出目录之前。这样请求、能力、外部命令或模型配置有问题时，
不会留下半成品，更不会触发旧输出清理。
"""

import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from src.config import AppConfig
from src.runtime import resolve_application_path

from .capabilities import CapabilityRegistry
from .errors import DependencyError, RequestValidationError
from .nnunet_command import validate_nnunet_command


class PreflightValidator:
    """集中检查请求能力、路径关系、可执行命令和 nnUNet 模型目录。"""

    def __init__(
        self,
        *,
        registry: Optional[CapabilityRegistry] = None,
        executable_finder: Callable[[str], Optional[str]] = shutil.which,
        check_model_directories: bool = True,
    ):
        """
        初始化预检器。

        executable_finder 和 check_model_directories 可在纯单元测试中替换；生产入口使用
        `shutil.which` 并检查真实模型目录。
        """
        self.registry = registry
        """用于判断当前版本是否实现请求组合的能力表。"""
        self.executable_finder = executable_finder
        """外部命令查找函数，默认使用系统 PATH。"""
        self.check_model_directories = check_model_directories
        """是否校验真实 nnUNet 目录；假 Stage 测试可关闭。"""

    def validate(self, request, config: AppConfig) -> None:
        """按从请求到运行环境的顺序完成只读检查，失败时抛出标准错误。"""
        valid, message = request.validate()
        if not valid:
            raise RequestValidationError(message or "请求参数无效")

        registry = self.registry or CapabilityRegistry.current(
            config.slice_part_config or None
        )
        registry.validate(request)
        input_path = Path(request.dicom_dir).expanduser().resolve()
        output_path = Path(request.output_dir).expanduser().resolve()
        if not input_path.exists() or not input_path.is_dir():
            raise RequestValidationError(
                f"输入目录不存在或不是目录: {input_path}"
            )
        if not os.access(input_path, os.R_OK):
            raise RequestValidationError(f"输入目录不可读取: {input_path}")
        if input_path == output_path or input_path.is_relative_to(output_path):
            raise RequestValidationError("输入目录不能等于或位于输出目录内")
        if output_path.is_relative_to(input_path):
            raise RequestValidationError("输出目录不能位于输入目录内")

        cache_dir = getattr(request, "cache_dir", "")
        if cache_dir:
            cache_path = Path(cache_dir).expanduser().resolve()
            if cache_path.is_relative_to(output_path):
                raise RequestValidationError(
                    "缓存目录不能位于输出目录内，否则会被 -o 清理"
                )

        parent = output_path.parent
        existing_parent = parent
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        if not existing_parent.is_dir() or not os.access(existing_parent, os.W_OK):
            raise RequestValidationError(f"输出目录父级不可写: {existing_parent}")

        validate_nnunet_command(
            self.executable_finder,
            predict_command=config.nnunet.predict_command,
            python_executable=config.nnunet.python_executable,
        )

        if self.check_model_directories:
            self._validate_model_directories(config)

    @staticmethod
    def _validate_model_directories(config: AppConfig) -> None:
        """检查 nnUNet 三个标准目录均已配置且在当前机器存在。"""
        values = {
            "nnUNet_raw": config.nnunet.raw,
            "nnUNet_preprocessed": config.nnunet.preprocessed,
            "nnUNet_results": config.nnunet.results,
        }
        for name, value in values.items():
            if not value:
                raise DependencyError(f"未配置 {name}")
            if not Path(resolve_application_path(value)).is_dir():
                raise DependencyError(
                    f"{name} 目录不存在",
                    details={"setting": name},
                )
