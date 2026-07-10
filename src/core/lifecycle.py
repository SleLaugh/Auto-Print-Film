"""
任务与阶段生命周期模型。

Pipeline 是状态的唯一写入者；业务 Stage 只负责返回结果或抛出异常。集中维护状态可
保证 Response、日志和 manifest 对同一次任务给出一致结论。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


def _now() -> str:
    """返回带本地时区的 ISO-8601 时间，供 manifest 跨平台解析。"""
    return datetime.now().astimezone().isoformat()


class JobStatus(Enum):
    """整个处理任务的生命周期状态。"""

    PENDING = "pending"
    """任务对象已创建，但尚未开始执行。"""
    RUNNING = "running"
    """预检已通过，至少即将执行一个阶段。"""
    SUCCEEDED = "succeeded"
    """所有必需阶段均成功结束。"""
    FAILED = "failed"
    """至少一个必需阶段失败。"""
    CANCELLED = "cancelled"
    """任务被调用方取消；当前版本只保留协议值。"""


class StageStatus(Enum):
    """单个 Pipeline 阶段的生命周期状态。"""

    PENDING = "pending"
    """阶段尚未轮到执行。"""
    RUNNING = "running"
    """阶段已经开始且尚未返回。"""
    SUCCEEDED = "succeeded"
    """阶段已正常返回结果。"""
    FAILED = "failed"
    """阶段执行时抛出异常。"""
    SKIPPED = "skipped"
    """前置阶段失败，按 fail-fast 策略未执行。"""
    CANCELLED = "cancelled"
    """阶段被取消；当前版本只保留协议值。"""


@dataclass
class StageRecord:
    """记录一个阶段的状态、耗时、错误和结果契约版本。"""

    stage_name: str
    """阶段在 ProcessingContext 中使用的唯一名称。"""
    status: StageStatus = StageStatus.PENDING
    """当前阶段状态，只能通过本类状态方法改变。"""
    started_at: Optional[str] = None
    """阶段开始时间，ISO-8601 格式。"""
    finished_at: Optional[str] = None
    """阶段进入终态的时间，ISO-8601 格式。"""
    duration_seconds: Optional[float] = None
    """使用单调时钟计算的阶段耗时秒数。"""
    result_schema_version: Optional[str] = None
    """成功结果的契约版本；旧字典结果允许为空。"""
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    """本阶段产生的非致命警告。"""
    error: Optional[Dict[str, Any]] = None
    """阶段失败时的标准错误字典。"""
    _started_monotonic: Optional[float] = field(default=None, repr=False)
    """内部单调时钟起点，不写入 manifest。"""

    def start(self) -> None:
        """把阶段从 pending 转为 running，并开始计时。"""
        import time

        self._require(StageStatus.PENDING, StageStatus.RUNNING)
        self.status = StageStatus.RUNNING
        self.started_at = _now()
        self._started_monotonic = time.monotonic()

    def succeed(self, schema_version: Optional[str] = None) -> None:
        """把运行中的阶段标记成功，并记录结果契约版本。"""
        self._finish(StageStatus.SUCCEEDED)
        self.result_schema_version = schema_version

    def fail(self, error: Dict[str, Any]) -> None:
        """把运行中的阶段标记失败，并保存标准错误。"""
        self._finish(StageStatus.FAILED)
        self.error = dict(error)

    def skip(self) -> None:
        """把尚未开始的阶段标记为因前置失败而跳过。"""
        self._require(StageStatus.PENDING, StageStatus.SKIPPED)
        self.status = StageStatus.SKIPPED
        self.finished_at = _now()

    def cancel(self) -> None:
        """取消尚未开始或正在运行的阶段，供未来取消机制复用。"""
        if self.status not in (StageStatus.PENDING, StageStatus.RUNNING):
            raise ValueError(
                f"非法阶段状态转换: {self.status.value} -> cancelled"
            )
        if self.status is StageStatus.RUNNING:
            self._finish(StageStatus.CANCELLED)
        else:
            self.status = StageStatus.CANCELLED
            self.finished_at = _now()

    def _finish(self, target: StageStatus) -> None:
        """进入运行阶段的终态，并使用单调时钟计算耗时。"""
        import time

        self._require(StageStatus.RUNNING, target)
        self.status = target
        self.finished_at = _now()
        if self._started_monotonic is not None:
            self.duration_seconds = max(
                0.0,
                time.monotonic() - self._started_monotonic,
            )

    def _require(self, source: StageStatus, target: StageStatus) -> None:
        """阻止非法状态跳转，避免 manifest 出现自相矛盾的状态。"""
        if self.status is not source:
            raise ValueError(
                f"非法阶段状态转换: {self.status.value} -> {target.value}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换成 manifest 可直接写入的阶段记录。"""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "result_schema_version": self.result_schema_version,
            "warnings": list(self.warnings),
            "error": self.error,
        }
