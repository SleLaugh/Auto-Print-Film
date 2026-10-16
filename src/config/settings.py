"""应用级不可变配置模型，负责描述一次 Pipeline 运行期间不会变化的设置。"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class NnUNetConfig:
    """nnUNet 目录和全局推理参数。"""

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
    """是否传递 --continue_prediction；不控制应用自身的阶段缓存。"""
    predict_command: str = "nnUNetv2_predict"
    """外部 nnUNet 命令；默认从 PATH 查找，也可配置可执行文件或 Python 脚本路径。"""
    python_executable: str = ""
    """可选外部推理 Python；为空时直接执行 predict_command。"""
    disable_tta: bool = False
    """是否关闭推理时的镜像增强；默认保留 nnUNet 的增强行为。"""
    step_size: float = 0.5
    """滑窗步长比例，范围为 0 < step_size <= 1；增大可能影响分割精度。"""
    num_processes_preprocessing: int = 1
    """预处理进程数，对应 -npp；默认单进程以控制主机内存占用。"""
    num_processes_segmentation_export: int = 1
    """分割导出进程数，对应 -nps；默认单进程以控制主机内存占用。"""

    def __post_init__(self) -> None:
        """在构造配置时拒绝非法推理选项，避免启动外部推理后才失败。"""
        if not isinstance(self.disable_tta, bool):
            raise ValueError("nnunet.disable_tta 必须是布尔值 true/false")
        if (
            isinstance(self.step_size, bool)
            or not isinstance(self.step_size, (int, float))
            or not 0 < self.step_size <= 1
            or not math.isfinite(self.step_size)
        ):
            raise ValueError("nnunet.step_size 必须是有限数值，范围为 0 < step_size <= 1")
        for name in (
            "num_processes_preprocessing",
            "num_processes_segmentation_export",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"nnunet.{name} 必须是正整数")


@dataclass(frozen=True)
class LoggingConfig:
    """控制调试产物、控制台日志、三类文件日志和日志根目录。"""

    debug_file_enabled: bool = False
    """是否生成并保留 manifest、调试影像和调试元数据。"""
    console_enabled: bool = True
    """是否向当前终端输出运行信息。"""
    run_log_file_enabled: bool = False
    """是否写入普通运行日志文件。"""
    debug_log_file_enabled: bool = False
    """是否写入 DebugLog.txt；不控制任何调试影像或元数据。"""
    error_log_file_enabled: bool = True
    """是否把错误信息写入独立错误日志。"""
    base_dir: str = "Log"
    """未绑定任务输出目录时使用的日志兜底路径。"""


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
    slice_part_config: Dict[str, Any] = field(default_factory=dict)
    """模型选择、定位区域及切法能力共用的统一切割部位配置。"""

    def summary(self) -> Dict[str, Any]:
        """返回不包含机器绝对路径和环境变量的可复现摘要。"""

        return {
            "nnunet": {
                "configuration": self.nnunet.configuration,
                "fold": self.nnunet.fold,
                "continue_mode": self.nnunet.continue_mode,
                "disable_tta": self.nnunet.disable_tta,
                "step_size": self.nnunet.step_size,
                "num_processes_preprocessing": self.nnunet.num_processes_preprocessing,
                "num_processes_segmentation_export": (
                    self.nnunet.num_processes_segmentation_export
                ),
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
                "debug_file_enabled": self.logging.debug_file_enabled,
                "debug_log_file_enabled": self.logging.debug_log_file_enabled,
            },
            "slice_parts": {
                "schema_version": self.slice_part_config.get("schema_version"),
                "enabled_count": sum(
                    1
                    for part in self.slice_part_config.get("slice_parts", {}).values()
                    if part.get("enabled") is True
                ),
            },
        }
