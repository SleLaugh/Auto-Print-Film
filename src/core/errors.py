"""
框架层标准错误。

本模块把配置、请求、依赖、阶段输入和产物等失败统一成稳定结构。调用方可以依赖
错误码做程序判断，同时仍通过 Python 异常链保留原始原因，避免只能解析中文文本。
"""

from typing import Any, Dict, Optional


class FrameworkError(Exception):
    """所有可预期框架错误的基类，可安全写入 Response 和 manifest。"""

    code = "FRAMEWORK_ERROR"
    """稳定机器错误码；子类必须覆盖，不能把类名直接当成外部协议。"""

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化标准错误。

        Args:
            message: 面向使用者的中文错误说明。
            stage: 发生错误的阶段；预检等阶段外错误允许为空。
            retryable: 上层是否可以在不改请求的情况下重试。
            details: 脱敏后的结构化诊断信息，不允许放患者隐私。
        """
        super().__init__(message)
        self.message = message
        """用户可读错误消息。"""
        self.stage = stage
        """错误所属阶段名称。"""
        self.retryable = retryable
        """是否建议上层重试。"""
        self.details = dict(details or {})
        """供日志和调用方定位问题的脱敏详情。"""

    def to_dict(self) -> Dict[str, Any]:
        """转换成 JSON 可序列化的固定错误协议。"""
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "retryable": self.retryable,
            "details": self.details,
        }


class ConfigurationError(FrameworkError):
    """配置文件、环境变量或显式配置无法形成有效配置。"""
    code = "CONFIGURATION_ERROR"


class RequestValidationError(FrameworkError):
    """请求字段、路径或输入输出关系不合法。"""
    code = "REQUEST_VALIDATION_ERROR"


class CapabilityError(FrameworkError):
    """请求组合合法，但当前版本尚未注册对应处理能力。"""
    code = "CAPABILITY_ERROR"


class DependencyError(FrameworkError):
    """外部命令、模型目录或运行环境依赖缺失。"""
    code = "DEPENDENCY_ERROR"


class StageInputError(FrameworkError):
    """阶段需要的前置结果缺失或结果类型不符合契约。"""
    code = "STAGE_INPUT_ERROR"


class ExternalProcessError(FrameworkError):
    """nnUNet 等外部进程启动或执行失败。"""
    code = "EXTERNAL_PROCESS_ERROR"


class ArtifactError(FrameworkError):
    """输出目录所有权、产物路径或 manifest 操作失败。"""
    code = "ARTIFACT_ERROR"


class OutputValidationError(FrameworkError):
    """阶段已输出文件，但数量、格式或几何校验不通过。"""
    code = "OUTPUT_VALIDATION_ERROR"
