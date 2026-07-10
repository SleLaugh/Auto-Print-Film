"""
六阶段 Pipeline 编排器。

Pipeline 只负责框架职责：预检、阶段顺序、状态转换、错误收集、产物保留和 manifest
更新。具体医学处理仍由各 Stage 实现，避免编排器逐渐变成业务代码集合。
"""

from datetime import datetime
from pathlib import Path
import shutil
import time
from typing import Any, Callable, List, Optional

from src.config import AppConfig, load_app_config
from src.utils import DebugLog, ErrorLog

from .artifacts import ArtifactManager, MANIFEST_SCHEMA
from .base_stage import BaseStage
from .context import ProcessingContext
from .errors import FrameworkError
from .lifecycle import JobStatus, StageStatus
from .preflight import PreflightValidator
from .response import Response


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
            stages: 自定义阶段列表；为空时构造正式六阶段主链。
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
        构造正式六阶段主链。

        nnUNet 配置由统一 AppConfig 注入分割和定位阶段，避免 Pipeline 运行期间各阶段
        再次读取不同配置文件。其他阶段暂时没有应用级配置依赖。
        """
        from src.stages import (
            LocalizationStage,
            MappingStage,
            PreprocessingStage,
            ReconstructionStage,
            ReportingStage,
            SegmentationStage,
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
            LocalizationStage(nnunet_config),
            MappingStage(),
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
        DebugLog("Pipeline 开始执行")
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
            try:
                record.start()
                context.set_current_stage(stage.name)
                self._write_manifest(context, artifacts)
                context = stage.execute(context)
                typed_result = context.stage_results.get(stage.name)
                record.succeed(getattr(typed_result, "schema_version", None))
                self._write_manifest(context, artifacts)
            except Exception as error:
                ErrorLog(f"阶段执行异常: {stage.name}", error)
                context.add_error(stage.name, error)
                record.fail(self._serialize_error(error, stage.name))
                for remaining in self.stages[index + 1:]:
                    pending = context.stage_records[remaining.name]
                    if pending.status is StageStatus.PENDING:
                        pending.skip()
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
        DebugLog(f"Pipeline 执行完成: {context.job_status.value}")
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
        迁移期登记旧阶段直接创建的三个中间目录。

        等各阶段全部改为向 ArtifactManager 申请路径后可以删除该兼容入口；现阶段它让
        新清理策略接管旧目录，同时不改变业务输出位置。
        """
        root = Path(request.output_dir)
        for name in ("metadata", "nifti", "segmentation"):
            path = root / name
            if path.exists():
                artifacts.register(name, path, "intermediate")

    @staticmethod
    def _cleanup_debug_artifacts(request: Any) -> None:
        """旧测试和旧调用的兼容 helper；新 Pipeline 不调用。"""

        root = Path(getattr(request, "output_dir", ""))
        if not root:
            return
        for name in ("metadata", "nifti", "segmentation"):
            path = root / name
            if path.exists():
                shutil.rmtree(path)
