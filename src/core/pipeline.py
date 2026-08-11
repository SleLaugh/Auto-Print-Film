"""
八阶段 Pipeline 编排器。

Pipeline 只负责框架职责：预检、阶段顺序、状态转换、错误收集、产物保留和 manifest
更新。具体医学处理仍由各 Stage 实现，避免编排器逐渐变成业务代码集合。
"""

from datetime import datetime
from pathlib import Path
import shutil
import time
from typing import Any, Callable, List, Optional

from src.config import AppConfig, load_app_config
from src.utils import ErrorLog, RunLog

from .artifacts import ArtifactManager, MANIFEST_SCHEMA
from .base_stage import BaseStage
from .context import ProcessingContext
from .errors import FrameworkError
from .lifecycle import JobStatus, StageStatus
from .preflight import PreflightValidator
from .response import Response


_STAGE_LOG_LABELS = {
    "preprocessing": "预处理（DICOM转NIfTI）",
    "segmentation": "推理分割",
    "spatial_correction": "空间校正",
    "localization": "标签定位",
    "anatomical_analysis": "解剖分析",
    "planning": "切片规划",
    "reconstruction": "重建与输出",
    "reporting": "校验与结果汇总",
}
"""稳定阶段名对应的日志可读职责；自定义测试阶段统一显示为“自定义阶段”。"""


class Pipeline:
    """
    按顺序执行阶段，并统一维护任务状态、产物和 manifest。

    一个 Pipeline 实例保存阶段定义和不可变配置；每次 `run` 都创建新的 Context，
    因此任务状态不会在多次调用之间复用。
    """

    def __init__(
        self,
        stages: Optional[List[BaseStage]] = None,
        config: Optional[AppConfig] = None,
        preflight: Optional[PreflightValidator] = None,
        artifact_manager_factory: Optional[Callable[..., ArtifactManager]] = None,
    ):
        """
        初始化流水线及其框架依赖。

        Args:
            stages: 自定义阶段列表；为空时构造正式八阶段主链。
            config: 已由 ConfigLoader 合并完成的应用配置。
            preflight: 可替换的预检器，方便纯框架测试使用假依赖。
            artifact_manager_factory: 产物管理器工厂，方便隔离文件系统测试。
        """
        self.config = config or load_app_config()
        """本次 Pipeline 使用的唯一生效配置。"""
        self.stages = stages or self._default_stages(self.config)
        """按执行顺序排列的阶段实例。"""
        self.preflight = preflight or PreflightValidator()
        """任何输出变更前运行的无副作用预检器。"""
        self.artifact_manager_factory = artifact_manager_factory or ArtifactManager
        """按请求输出目录创建产物管理器的工厂。"""

    @staticmethod
    def _default_stages(config: AppConfig) -> List[BaseStage]:
        """
        构造正式八阶段主链。

        nnUNet 配置由统一 AppConfig 注入分割以及校正/定位/解剖分析三个
        共享定位配置的阶段，避免运行期间读到不同的模型路径。
        """
        from src.stages import (
            AnatomicalAnalysisStage,
            LocalizationStage,
            PlanningStage,
            PreprocessingStage,
            ReconstructionStage,
            ReportingStage,
            SegmentationStage,
            SpatialCorrectionStage,
        )

        nnunet_config = {
            "_configuration": config.nnunet.configuration,
            "_fold": config.nnunet.fold,
            "_continue": config.nnunet.continue_mode,
            "_model_rules": list(config.nnunet.model_rules),
            "nnUNet_raw": config.nnunet.raw,
            "nnUNet_preprocessed": config.nnunet.preprocessed,
            "nnUNet_results": config.nnunet.results,
        }
        return [
            PreprocessingStage(),
            SegmentationStage(nnunet_config),
            SpatialCorrectionStage(nnunet_config),
            LocalizationStage(nnunet_config),
            AnatomicalAnalysisStage(nnunet_config),
            PlanningStage(),
            ReconstructionStage(),
            ReportingStage(),
        ]

    def run(self, request: Any) -> Response:
        """
        执行一次完整任务并返回统一响应。

        执行顺序固定为：只读预检、准备受保护输出目录、创建任务记录、逐阶段执行、
        按保留策略清理中间产物、写最终 manifest。任一必需阶段失败后执行 fail-fast，
        后续阶段统一标记 skipped。
        """
        RunLog(f"Pipeline 开始执行，共 {len(self.stages)} 个阶段")
        if self.config.pipeline.run_preflight:
            self.preflight.validate(request, self.config)

        artifacts = self.artifact_manager_factory(
            request.output_dir,
            keep_intermediate_on_success=(
                self.config.pipeline.keep_intermediate_on_success
                # debug 开启时，metadata、NIfTI 和 segmentation 都是排查所需
                # 现场，成功任务也不能按普通生产策略清理。
                or self.config.logging.debug_log_file_enabled
            ),
            keep_intermediate_on_failure=(
                self.config.pipeline.keep_intermediate_on_failure
            ),
        )
        artifacts.prepare()

        context = ProcessingContext(
            request,
            stage_names=[stage.name for stage in self.stages],
        )
        context.effective_config_summary = self.config.summary()
        context.job_status = JobStatus.RUNNING
        context.started_at = self._now()
        pipeline_start = time.monotonic()
        self._write_manifest(context, artifacts)

        for index, stage in enumerate(self.stages):
            record = context.stage_records[stage.name]
            stage_label = _STAGE_LOG_LABELS.get(stage.name, "自定义阶段")
            try:
                record.start()
                RunLog(
                    f"[Pipeline] 阶段 {index + 1}/{len(self.stages)} 开始: "
                    f"{stage.name}（{stage_label}）"
                )
                context.set_current_stage(stage.name)
                self._write_manifest(context, artifacts)
                context = stage.execute(context)
                typed_result = context.stage_results.get(stage.name)
                record.succeed(getattr(typed_result, "schema_version", None))
                RunLog(
                    f"[Pipeline] 阶段 {index + 1}/{len(self.stages)} 完成: "
                    f"{stage.name}（{stage_label}），"
                    f"耗时 {record.duration_seconds or 0.0:.3f}s"
                )
                self._write_manifest(context, artifacts)
            except Exception as error:
                ErrorLog(
                    f"[Pipeline] 阶段 {index + 1}/{len(self.stages)} 执行失败\n"
                    f"阶段标识: {stage.name}\n"
                    f"阶段职责: {stage_label}\n"
                    f"错误原因: {error or type(error).__name__}",
                    error,
                )
                context.add_error(stage.name, error)
                record.fail(self._serialize_error(error, stage.name))
                for remaining in self.stages[index + 1:]:
                    pending = context.stage_records[remaining.name]
                    if pending.status is StageStatus.PENDING:
                        pending.skip()
                        remaining_label = _STAGE_LOG_LABELS.get(
                            remaining.name,
                            "自定义阶段",
                        )
                        RunLog(
                            "[Pipeline] 跳过后续阶段: "
                            f"{remaining.name}（{remaining_label}）"
                        )
                self._write_manifest(context, artifacts)
                break

        self._register_legacy_intermediate_artifacts(request, artifacts)
        success = not context.has_errors()
        context.warnings.extend(
            {"code": "CLEANUP_WARNING", "message": message}
            for message in artifacts.cleanup(success=success)
        )
        context.job_status = (
            JobStatus.SUCCEEDED if success else JobStatus.FAILED
        )
        context.finished_at = self._now()
        elapsed = time.monotonic() - pipeline_start
        response = Response(
            success=success,
            request=request,
            context=context,
            total_time=elapsed,
        )
        self._write_manifest(context, artifacts)
        RunLog(f"Pipeline 执行完成: {context.job_status.value}")
        return response

    @staticmethod
    def _now() -> str:
        """生成带本地时区的 ISO-8601 任务时间。"""
        return datetime.now().astimezone().isoformat()

    @staticmethod
    def _serialize_error(error: Exception, stage: str) -> dict:
        """保留已知标准错误；把未知异常包装成稳定的 UNEXPECTED_ERROR。"""
        if isinstance(error, FrameworkError):
            data = error.to_dict()
            data["stage"] = data["stage"] or stage
            return data
        return {
            "code": "UNEXPECTED_ERROR",
            "message": str(error),
            "stage": stage,
            "retryable": False,
            "details": {"exception": type(error).__name__},
        }

    @staticmethod
    def _manifest_data(context: ProcessingContext) -> dict:
        """从 Context 生成不包含患者 DICOM 元数据的任务清单快照。"""
        return {
            "schema_version": MANIFEST_SCHEMA,
            "job_id": context.job_id,
            "status": context.job_status.value,
            "created_at": context.created_at,
            "started_at": context.started_at,
            "finished_at": context.finished_at,
            "software_version": context.software_version,
            "request": context.request.to_dict(),
            "effective_config": context.effective_config_summary,
            "stages": [
                record.to_dict()
                for record in context.stage_records.values()
            ],
            "artifacts": [],
            "warnings": list(context.warnings),
            "errors": [
                record.error
                for record in context.stage_records.values()
                if record.error is not None
            ],
        }

    def _write_manifest(
        self,
        context: ProcessingContext,
        artifacts: ArtifactManager,
    ) -> None:
        """
        调试模式下通过 ArtifactManager 原子更新任务清单。

        manifest 记录完整阶段状态和诊断信息，只用于开发排查；正常交付模式不创建，
        避免在业务输出目录中增加调用方不需要的框架文件。
        """
        if not self.config.logging.debug_log_file_enabled:
            return
        artifacts.write_manifest(self._manifest_data(context))

    @staticmethod
    def _register_legacy_intermediate_artifacts(
        request: Any,
        artifacts: ArtifactManager,
    ) -> None:
        """
        登记仍由业务阶段直接创建的三个中间目录。

        在这些阶段改为向 ArtifactManager 申请路径前，由清理策略统一接管目录，
        同时保持业务输出位置不变。
        """
        root = Path(request.output_dir)
        for name in ("metadata", "nifti", "segmentation"):
            path = root / name
            if path.exists():
                artifacts.register(name, path, "intermediate")

    @staticmethod
    def _cleanup_debug_artifacts(request: Any) -> None:
        """直接清理调试中间目录的独立 helper；Pipeline 主流程不调用。"""

        root = Path(getattr(request, "output_dir", ""))
        if not root:
            return
        for name in ("metadata", "nifti", "segmentation"):
            path = root / name
            if path.exists():
                shutil.rmtree(path)
