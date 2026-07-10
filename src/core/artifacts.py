"""
任务输出目录所有权和产物保留管理。

本模块统一执行任务输出目录重建，并登记业务结果、中间文件和调试文件。调用方已经把
`output_dir` 定义为本次任务的专用目录，因此预检通过后会删除旧目录并重新创建。
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

from .errors import ArtifactError


MANIFEST_SCHEMA = "job_manifest_v1"
"""当前任务清单协议版本，同时也是输出目录所有权标记。"""


@dataclass(frozen=True)
class StageArtifact:
    """一项由框架创建并允许框架管理生命周期的文件或目录。"""

    artifact_type: str
    """产物用途类型，例如 nifti、segmentation、report。"""
    path: str
    """产物的规范化绝对路径，必须位于任务根目录内。"""
    retention: str
    """保留级别，目前使用 final、intermediate、debug 或 log。"""
    description: str = ""
    """供人工查看 manifest 时理解产物的中文说明。"""

    def to_dict(self) -> Dict[str, str]:
        """转换成 manifest 可写入的普通字典。"""
        return asdict(self)


class ArtifactManager:
    """重建任务输出目录，并集中登记、写清单和清理框架产物。"""

    def __init__(
        self,
        root: Path | str,
        *,
        keep_intermediate_on_success: bool = False,
        keep_intermediate_on_failure: bool = True,
    ):
        """
        初始化产物管理器，但此时不创建任何目录。

        Args:
            root: 本次任务输出根目录。
            keep_intermediate_on_success: 成功后是否保留阶段间文件。
            keep_intermediate_on_failure: 失败后是否保留现场用于排查。
        """
        self.root = Path(root).expanduser().resolve()
        """规范化后的任务输出根目录。"""
        self.keep_intermediate_on_success = keep_intermediate_on_success
        """成功任务的中间产物保留开关。"""
        self.keep_intermediate_on_failure = keep_intermediate_on_failure
        """失败任务的中间产物保留开关。"""
        self.artifacts: List[StageArtifact] = []
        """本次运行已登记且允许框架管理的产物列表。"""

    @property
    def manifest_path(self) -> Path:
        """返回当前任务根目录中的固定 manifest 路径。"""
        return self.root / "manifest.json"

    def prepare(self) -> None:
        """
        删除并重新创建本次任务输出目录。

        该方法只会在 Pipeline 预检成功后调用，因此依赖或请求错误不会误删旧结果。
        """
        if self.root.exists():
            if self.root.is_dir():
                shutil.rmtree(self.root)
            else:
                self.root.unlink()
        self.root.mkdir(parents=True)

    def register(
        self,
        artifact_type: str,
        path: Path | str,
        retention: str,
        description: str = "",
    ) -> StageArtifact:
        """登记一项框架产物，并验证它没有逃逸到任务根目录之外。"""
        resolved = Path(path).expanduser().resolve()
        self._require_inside_root(resolved)
        artifact = StageArtifact(
            artifact_type=artifact_type,
            path=str(resolved),
            retention=retention,
            description=description,
        )
        if artifact not in self.artifacts:
            self.artifacts.append(artifact)
        return artifact

    def build_manifest_stub(self, job_id: str = "") -> Dict[str, Any]:
        """构造测试和任务初始化可使用的最小 manifest。"""
        return {
            "schema_version": MANIFEST_SCHEMA,
            "job_id": job_id,
            "status": "pending",
            "artifacts": [item.to_dict() for item in self.artifacts],
            "stages": [],
            "warnings": [],
            "errors": [],
        }

    def write_manifest(self, data: Dict[str, Any]) -> Path:
        """在同一目录先写临时文件再替换，避免中断留下半个 JSON。"""
        if data.get("schema_version") != MANIFEST_SCHEMA:
            raise ArtifactError("manifest schema_version 无效")
        data = dict(data)
        data["artifacts"] = [item.to_dict() for item in self.artifacts]
        temporary = self.root / ".manifest.json.tmp"
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
        temporary.replace(self.manifest_path)
        return self.manifest_path

    def cleanup(self, *, success: bool) -> List[str]:
        """按任务结果清理已登记中间产物，并把清理失败降级为警告。"""
        keep = (
            self.keep_intermediate_on_success
            if success
            else self.keep_intermediate_on_failure
        )
        if keep:
            return []
        warnings: List[str] = []
        for artifact in self.artifacts:
            if artifact.retention != "intermediate":
                continue
            try:
                self._delete_path(Path(artifact.path))
            except (OSError, ArtifactError) as error:
                warnings.append(f"清理中间产物失败: {error}")
        return warnings

    def _delete_path(self, path: Path) -> None:
        """经过根目录边界检查后删除单个登记文件或目录。"""
        self._require_inside_root(path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _require_inside_root(self, path: Path) -> None:
        """拒绝任务根目录本身和所有根目录之外的路径。"""
        if path == self.root or not path.is_relative_to(self.root):
            raise ArtifactError(
                "产物路径必须位于任务输出目录内",
                details={"path": str(path)},
            )
