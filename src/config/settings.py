"""应用级不可变配置模型，负责描述一次 Pipeline 运行期间不会变化的设置。"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class NnUNetConfig:
    """nnUNet 目录、推理参数和模型匹配规则。"""

    raw: str = ""
    """nnUNet_raw 目录；空值表示未配置。"""
    preprocessed: str = ""
    """nnUNet_preprocessed 目录；空值表示未配置。"""
    results: str = ""
    """nnUNet_results 目录；定位和推理共同使用。"""
    configuration: str = "3d_fullres"
    """传给 nnUNetv2_predict 的网络配置名称。"""
    fold: str = "0"
    """推理使用的训练 fold。"""
    continue_mode: bool = True
    """是否向 nnUNet 推理命令传递 continue 行为。"""
    model_rules: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    """按检查类型、部位和扫描厚度匹配模型 ID 的不可变规则序列。"""


@dataclass(frozen=True)
class LoggingConfig:
    """控制控制台日志、三类文件日志和日志根目录。"""

    console_enabled: bool = True
    """是否向当前终端输出运行信息。"""
    run_log_file_enabled: bool = False
    """是否写入普通运行日志文件。"""
    debug_log_file_enabled: bool = False
    """是否写调试日志并保留调试影像产物。"""
    error_log_file_enabled: bool = True
    """是否把错误信息写入独立错误日志。"""
    base_dir: str = "Log"
    """日志根目录，可使用相对项目工作目录的路径。"""


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline 的失败策略、产物保留和预检开关。"""

    fail_fast: bool = True
    """阶段失败后是否立即停止；当前版本固定只支持 True。"""
    keep_intermediate_on_success: bool = False
    """任务成功后是否保留中间 NIfTI、分割和调试元数据。"""
    keep_intermediate_on_failure: bool = True
    """任务失败后是否保留中间现场用于诊断。"""
    run_preflight: bool = True
    """是否执行真实依赖预检；仅假 Stage 单元测试允许关闭。"""


@dataclass(frozen=True)
class AppConfig:
    """传入 Pipeline 的应用总配置，也是运行期间唯一配置事实来源。"""

    nnunet: NnUNetConfig = field(default_factory=NnUNetConfig)
    """nnUNet 相关配置。"""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """日志与调试产物配置。"""
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """流水线框架行为配置。"""

    def summary(self) -> Dict[str, Any]:
        """返回不包含机器绝对路径和环境变量的可复现摘要。"""

        return {
            "nnunet": {
                "configuration": self.nnunet.configuration,
                "fold": self.nnunet.fold,
                "continue_mode": self.nnunet.continue_mode,
                "model_rule_count": len(self.nnunet.model_rules),
                "raw_configured": bool(self.nnunet.raw),
                "preprocessed_configured": bool(self.nnunet.preprocessed),
                "results_configured": bool(self.nnunet.results),
            },
            "pipeline": {
                "fail_fast": self.pipeline.fail_fast,
                "keep_intermediate_on_success": (
                    self.pipeline.keep_intermediate_on_success
                ),
                "keep_intermediate_on_failure": (
                    self.pipeline.keep_intermediate_on_failure
                ),
            },
            "logging": {
                # manifest 只记录开关，不记录日志绝对路径。
                "debug_log_file_enabled": self.logging.debug_log_file_enabled,
            },
        }
