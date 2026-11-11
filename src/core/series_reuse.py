"""Optional exact-volume reuse with independent, immutable per-series snapshots.

Only successful, validated jobs publish entries. Discovery reads small manifests;
it never decodes every sibling volume. Paths in snapshots are portable tokens,
and consumers copy all artifacts before using them, so a producer can be rerun.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from src.runtime import external_process_environment, resolve_application_path
from src.utils import RunLog
from .cache_workspace import cache_workspace, cleanup_cache_workspaces
from .volume_identity import get_volume_identity


SCHEMA = "autoprintfilm-series-reuse-v1"
STAGES = ("segmentation", "spatial_correction", "localization", "anatomical_analysis", "planning")
_RAW = "@current-raw"
_FILE = "@bundle/"
_OUTPUT = "@current-output/"


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _generation(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]


def _hash_file(path):
    before = _generation(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if _generation(path) != before:
        raise ValueError(f"Cache artifact changed while reading: {path}")
    return digest.hexdigest()


def _safe_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Invalid cache artifact path")
    root = Path(root).resolve()
    result = root / relative
    if not result.resolve().is_relative_to(root) or any(
        part.is_symlink() for part in (result, *result.parents) if part != root.parent
        and part.is_relative_to(root)
    ):
        raise ValueError("Cache artifact escapes its snapshot")
    return result


def _code_identity():
    if getattr(sys, "frozen", False):
        return {"executable": str(Path(sys.executable).resolve()),
                "generation": _generation(sys.executable)}
    source_root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(source_root)): _hash_file(path)
            for path in sorted(source_root.rglob("*.py"))}


def _model_identity(config, model):
    """Include weight generations and label/plan contents without loading weights."""
    value = (resolve_application_path(config.nnunet.results)
             or external_process_environment().get("nnUNet_results", ""))
    if not value:
        raise ValueError("无法确定实际 nnUNet 模型目录")
    root = Path(value)
    files = {}
    datasets = list(root.glob(f"Dataset{int(model['model_id']):03d}_*"))
    if len(datasets) != 1:
        raise ValueError("无法唯一确定 nnUNet 数据集")
    plans = model.get("plans", "nnUNetPlans")
    folder = datasets[0] / f"nnUNetTrainer__{plans}__{model['configuration']}"
    selected = [folder / "dataset.json", folder / "plans.json",
                folder / f"fold_{model['fold']}" / "checkpoint_final.pth"]
    for path in selected:
        if not path.is_file():
            raise ValueError(f"模型来源文件缺失: {path}")
        files[str(path.resolve())] = (
            _hash_file(path) if path.suffix == ".json" else _generation(path)
        )
    return files


def _engine_identity(config):
    """Identify the external engine without importing torch or starting CUDA."""
    from .nnunet_command import (
        DEFAULT_PREDICT_COMMAND, _launcher_python, _official_launcher_source,
        gpu_adapter_path, gpu_adapter_sources, nnunet_command_prefix,
    )
    options = config.nnunet
    prefix = nnunet_command_prefix(options.predict_command, options.python_executable)
    launcher = {}
    environment = external_process_environment()
    gpu_resampling = None
    if options.python_executable and Path(prefix[-1]) == gpu_adapter_path():
        interpreter = [prefix[0]]
        launcher = {str(path): _hash_file(path) for path in gpu_adapter_sources()}
        gpu_resampling = environment.get("NNUNET_GPU_RESAMPLING", "1")
        if gpu_resampling not in {"0", "1"}:
            raise ValueError("NNUNET_GPU_RESAMPLING 必须为 0 或 1")
    elif options.predict_command == DEFAULT_PREDICT_COMMAND and options.python_executable:
        interpreter = [prefix[0]]
    else:
        command = prefix[-1]
        location = shutil.which(command) or command
        source = _official_launcher_source(location)
        if source is None:
            raise ValueError("自定义 nnUNet 入口无法确认来源，保留原调用")
        interpreter = [prefix[0]] if options.python_executable else _launcher_python(source)
        if not interpreter:
            raise ValueError("无法确定 nnUNet 推理解释器")
        launcher[str(Path(location).absolute())] = _hash_file(location)
    executable = shutil.which(interpreter[0]) or interpreter[0]
    script = (
        "import importlib.util,json; s=importlib.util.find_spec('nnunetv2'); "
        "print(json.dumps(list(s.submodule_search_locations) if s else []))"
    )
    result = subprocess.run(
        [executable, *interpreter[1:], "-c", script],
        env=environment, capture_output=True, text=True,
        timeout=10, check=True, encoding="utf-8",
    )
    locations = json.loads(result.stdout)
    if not isinstance(locations, list) or not locations:
        raise ValueError("无法确定实际 nnUNet 包来源")
    sources = {}
    for location in locations:
        root = Path(location).resolve()
        for path in sorted(root.rglob("*.py")):
            sources[str(path)] = _hash_file(path)
    if not sources:
        raise ValueError("nnUNet 源码不可读，保留原调用")
    identity = {"interpreter": [str(executable), *interpreter[1:]],
                "generation": _generation(executable), "launcher": launcher, "sources": sources}
    if gpu_resampling is not None:
        identity["gpu_resampling"] = gpu_resampling
    return identity


class SeriesReuseSession:
    """One job's private cache transaction; all optional failures disable reuse."""

    def __init__(self, context, config, segmentation_stage):
        self.disabled = False
        self.hit = False
        self.segmentation_hit = False
        self.context = context
        self.root = Path(context.request.cache_dir).expanduser().resolve()
        self.parent = Path(context.request.series_cache_parent).expanduser().resolve()
        self.raw_path = Path(context.get_result("preprocessing")["nifti_path"]).resolve()
        self.raw_generation = _generation(self.raw_path)
        self.identity = get_volume_identity(self.raw_path, self.root / "series_reuse" / "volume.json")
        request = context.request
        model = segmentation_stage._match_model(
            request.slice_part.value, request.study_type.value, request.scan_thickness,
        )
        options = asdict(config.nnunet)
        self._config = config
        self._model = model
        self._weights = _model_identity(config, model)
        self._code = _code_identity()
        self._engine = _engine_identity(config)
        self.segmentation_key = _digest({
            "schema": SCHEMA, "code": self._code, "nnunet": options,
            "weights": self._weights,
            "engine": self._engine,
            "model": {
                **{key: model[key] for key in ("model_id", "configuration", "fold")},
                "plans": model.get("plans", "nnUNetPlans"),
            },
        })
        planes = []
        for plane in request.plane_configs:
            item = plane.to_dict()
            item.pop("WindowPresets", None)
            planes.append(item)
        self.processing_key = _digest({
            "segmentation_key": self.segmentation_key,
            "slice_part": request.slice_part.value, "study_type": request.study_type.value,
            "scan_thickness": request.scan_thickness, "planes": planes,
            "slice_part_config": config.slice_part_config,
        })
        self.volume_root = Path("series_reuse") / "v1" / self.identity["digest"]
        work_root = self.root / "series_reuse" / ".work"
        cleanup_cache_workspaces(work_root, "reuse-")
        self._workspace = cache_workspace(work_root, "reuse-")
        self.cache_dir = self._workspace.__enter__().path / "snapshot"
        self.cache_dir.mkdir()
        self.snapshot = None
        self._local_entry = None
        self._discover()

    def local_segmentation_valid(self, output_dir):
        try:
            marker = json.loads((output_dir / "series_reuse.json").read_text("utf-8"))
            return marker == {"schema": SCHEMA, "volume": self.identity["digest"],
                              "segmentation_key": self.segmentation_key,
                              "generation": self._segmentation_generations(output_dir)}
        except (OSError, ValueError, TypeError):
            return False

    def record_local_segmentation(self, output_dir):
        if _model_identity(self._config, self._model) != self._weights:
            raise ValueError("Model files changed during inference")
        if _engine_identity(self._config) != self._engine:
            raise ValueError("nnUNet engine changed during inference")
        marker = {"schema": SCHEMA, "volume": self.identity["digest"],
                  "segmentation_key": self.segmentation_key,
                  "generation": self._segmentation_generations(output_dir)}
        temporary = output_dir / f".series-reuse-{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(_json_bytes(marker))
            temporary.replace(output_dir / "series_reuse.json")
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _segmentation_generations(output_dir):
        return {path.name: _generation(path) for path in sorted(output_dir.iterdir())
                if path.is_file() and path.name != "series_reuse.json"
                and not path.name.startswith(".")}

    def _materialize_segmentation(self):
        """Also populate the normal per-model cache for future ordinary jobs."""
        from .cache_lock import cache_lock
        model_id = self.snapshot["results"]["segmentation"]["model"]["model_id"]
        source = self.cache_dir / "segmentation"
        if not (source / "seg.nii.gz").is_file():
            raise ValueError("Snapshot lacks independent segmentation")
        output = self.root / str(model_id) / "segmentation"
        with cache_lock(self.root / ".locks" / f"segmentation-{model_id}.lock"):
            if self.local_segmentation_valid(output):
                return
            with cache_workspace(output.parent, ".segmentation-") as work:
                for child in source.iterdir():
                    if child.is_file() and not child.name.startswith("."):
                        shutil.copyfile(child, work.path / child.name)
                output.mkdir(parents=True, exist_ok=True)
                (output / "seg.nii.gz").unlink(missing_ok=True)
                for child in output.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                for child in work.path.iterdir():
                    if child.is_file() and not child.name.startswith(".") and child.name != "seg.nii.gz":
                        child.replace(output / child.name)
                (work.path / "seg.nii.gz").replace(output / "seg.nii.gz")
                self.record_local_segmentation(output)

    def disable(self, reason):
        self.disabled = True
        RunLog(f"[同图缓存] 已回退原流程: {reason}")

    def close(self):
        workspace = getattr(self, "_workspace", None)
        if workspace is not None:
            self._workspace = None
            workspace.__exit__(None, None, None)

    def _roots(self):
        # Study layout: <parent>/<series>/cache. Never recursively scan image files.
        roots = [self.root, self.parent / "cache"]
        roots.extend(child / "cache" for child in sorted(self.parent.iterdir())
                     if child.is_dir() and not child.is_symlink())
        seen = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen and resolved.is_relative_to(self.parent):
                seen.add(resolved)
                yield resolved

    def _read_entry(self, entry):
        ready = json.loads((entry / "ready.json").read_text(encoding="utf-8"))
        if (ready["schema"] != SCHEMA or ready["volume"] != self.identity
                or ready["segmentation_key"] != self.segmentation_key):
            return None
        if ready.get("integrity") != _digest({key: value for key, value in ready.items()
                                              if key != "integrity"}):
            return None
        return ready

    def _discover(self):
        candidates = []
        for root in self._roots():
            volume = root / self.volume_root
            exact = volume / self.processing_key
            candidates.extend((entry, True) for entry in sorted(exact.glob("*/ready.json")))
            candidates.extend((entry, False) for entry in sorted(volume.glob("*/*/ready.json"))
                              if entry.parent.parent != exact)
        # Prefer full reuse across all series before considering segmentation only.
        candidates.sort(key=lambda item: not item[1])
        for marker, full in candidates:
            entry = marker.parent
            try:
                ready = self._read_entry(entry)
                if ready is None or (full and ready["processing_key"] != self.processing_key):
                    continue
                self._copy_entry(entry, ready, full=full)
                self.snapshot = json.loads((self.cache_dir / "context.json").read_text("utf-8"))
                self._validate_snapshot(self.snapshot, full=full)
                self._prevalidate_results(full=full)
                if full:
                    from src.reconstruction.hu_cache import validate_hu_cache
                    if not validate_hu_cache(self.cache_dir / "hu"):
                        raise ValueError("Incomplete HU cache")
                self._materialize_segmentation()
                self.hit = full
                self.segmentation_hit = True
                if full and entry.is_relative_to(self.root):
                    self._local_entry = entry
                RunLog(f"[同图缓存] 命中{'完整切割' if full else '分割'}缓存，已复制独立快照")
                return
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                self.cache_dir.mkdir()
        RunLog("[同图缓存] 未命中可用快照，执行原流程并建立缓存")

    def _copy_entry(self, entry, ready, *, full):
        for relative, checksum in ready["files"].items():
            if not full and relative != "context.json" and not relative.startswith("segmentation/"):
                continue
            source = _safe_path(entry, relative)
            destination = _safe_path(self.cache_dir, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if _hash_file(destination) != checksum:
                raise ValueError("Cache checksum mismatch")

    @staticmethod
    def _validate_snapshot(snapshot, *, full):
        results = snapshot["results"]
        for stage in STAGES if full else ("segmentation",):
            if not isinstance(results[stage], dict):
                raise ValueError("Invalid stage snapshot")
        if full and not isinstance(results["planning"]["plans"], list):
            raise ValueError("Invalid planning snapshot")

    def _prevalidate_results(self, *, full):
        from src.models import SegmentationResult
        segmentation = SegmentationResult(**self._decode(self.snapshot["results"]["segmentation"]))
        if not Path(segmentation.seg_path).is_file() or not segmentation.model.get("model_id"):
            raise ValueError("Incomplete segmentation result")
        if full:
            from src.stages.stage_reconstruction import ReconstructionStage
            configs = self.context.request.plane_configs
            for plan in self.snapshot["results"]["planning"]["plans"]:
                ReconstructionStage._validate_plan(plan)
                config_id = str(plan.get("config_id") or "")
                if sum(config_id == f"配置{ordinal:02d}" for ordinal in range(1, len(configs) + 1)) != 1:
                    raise ValueError("Cached plan cannot match current request")
            for stage in STAGES:
                self._decode(self.snapshot["results"][stage])

    def can_restore(self, stage):
        return not self.disabled and (self.hit and stage in STAGES
                                     or self.segmentation_hit and stage == "segmentation")

    def restore_stage(self, context, stage):
        from src.models import (SegmentationResult, SpatialCorrectionStageResult,
                                LocalizationStageResult, AnatomicalAnalysisStageResult,
                                PlanningStageResult)
        types = dict(zip(STAGES, (SegmentationResult, SpatialCorrectionStageResult,
                                 LocalizationStageResult, AnatomicalAnalysisStageResult,
                                 PlanningStageResult)))
        payload = self._decode(self.snapshot["results"][stage])
        if stage == "segmentation":
            result = types[stage](**payload)
            context.metadata["model_id"] = str(payload["model"]["model_id"])
        else:
            result = types[stage](payload=payload)
        context.set_stage_result(result)
        if stage == "planning":
            from src.stages.stage_planning import PlanningStage
            PlanningStage._assign_plane_config_ids(context.request.plane_configs)
        return context

    def _decode(self, value):
        if isinstance(value, dict):
            return {key: self._decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._decode(item) for item in value]
        if value == _RAW:
            return str(self.raw_path)
        if isinstance(value, str) and value.startswith(_FILE):
            path = _safe_path(self.cache_dir, value[len(_FILE):])
            if not path.is_file():
                raise ValueError("Snapshot reference is missing")
            return str(path)
        if isinstance(value, str) and value.startswith(_OUTPUT):
            return str(_safe_path(Path(self.context.request.output_dir), value[len(_OUTPUT):]))
        return value

    def capture_context(self, context):
        """Persist parameters plus files, with no references to another series."""
        file_map = {}
        capture_prefix = f"files/{uuid.uuid4().hex}"
        segmentation = Path(context.get_result("segmentation")["seg_path"]).resolve()
        segmentation_root = self.cache_dir / "segmentation"
        segmentation_root.mkdir(exist_ok=True)
        for child in sorted(segmentation.parent.iterdir()):
            if child.is_file() and not child.name.startswith(".") and child.name != "series_reuse.json":
                destination = segmentation_root / child.name
                if child.resolve() != destination.resolve():
                    shutil.copyfile(child, destination)
                file_map[str(child.resolve())] = _FILE + "segmentation/" + child.name

        def encode(value):
            if isinstance(value, dict):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            if isinstance(value, Path):
                value = str(value)
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                return value
            path = Path(value).resolve()
            if path == self.raw_path:
                return _RAW
            if path.is_file():
                if str(path) not in file_map:
                    relative = f"{capture_prefix}/{len(file_map):04d}/{path.name}"
                    destination = _safe_path(self.cache_dir, relative)
                    # A hit may already refer to this private snapshot.
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if path != destination:
                        shutil.copyfile(path, destination)
                    file_map[str(path)] = _FILE + relative
                return file_map[str(path)]
            output = Path(context.request.output_dir).resolve()
            if path.is_relative_to(output) and path != output:
                return _OUTPUT + str(path.relative_to(output))
            # A missing debug file has no consumable content or stable source path.
            return ""

        # Normal captures start with an empty files directory. Hits retain their
        # already portable snapshot, avoiding file aliasing while copying onto itself.
        if self.hit:
            return
        payload = {"results": {stage: encode(context.get_result(stage)) for stage in STAGES}}
        self._validate_snapshot(payload, full=True)
        (self.cache_dir / "context.json").write_bytes(_json_bytes(payload))
        self.snapshot = payload

    def publish(self, context):
        if self.disabled:
            return
        from src.reconstruction.hu_cache import validate_hu_cache
        if _generation(self.raw_path) != self.raw_generation:
            raise ValueError("Input volume changed before cache publication")
        if _model_identity(self._config, self._model) != self._weights:
            raise ValueError("Model files changed before cache publication")
        if _code_identity() != self._code:
            raise ValueError("Algorithm code changed before cache publication")
        if _engine_identity(self._config) != self._engine:
            raise ValueError("nnUNet engine changed before cache publication")
        from src.stages.stage_preprocessing import PreprocessingStage
        if not PreprocessingStage._source_provenance_matches(
            Path(context.request.dicom_dir), self.raw_path,
            self.root / "metadata" / "dicom_metadata.json",
        ):
            raise ValueError("Source series changed before cache publication")
        if not validate_hu_cache(self.cache_dir / "hu"):
            raise ValueError("Incomplete HU cache")
        self.capture_context(context)
        if self.hit and self._local_entry is not None:
            self._adopt_entry(context, self._local_entry)
            return
        files = {str(path.relative_to(self.cache_dir)): _hash_file(path)
                 for path in sorted(self.cache_dir.rglob("*")) if path.is_file()}
        ready = {"schema": SCHEMA, "volume": self.identity,
                 "segmentation_key": self.segmentation_key,
                 "processing_key": self.processing_key, "files": files}
        request = context.request
        ready["summary"] = {
            "slice_part": request.slice_part.value,
            "study_type": request.study_type.value,
            "scan_thickness": request.scan_thickness,
            "source_size_xyz": self.identity["geometry"]["size"],
            "plane_configs": [plane.to_dict() for plane in request.plane_configs],
            "window_presets": [window.to_dict() for window in request.window_presets],
            "plan_count": len((context.get_result("planning") or {}).get("plans", [])),
        }
        ready["integrity"] = _digest(ready)
        (self.cache_dir / "ready.json").write_bytes(_json_bytes(ready))
        destination = self.root / self.volume_root / self.processing_key / uuid.uuid4().hex
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.replace(destination)
        self._adopt_entry(context, destination)
        RunLog("[同图缓存] 完整快照已发布到本序列缓存")

    def _adopt_entry(self, context, destination):
        # Context results must not point at the now-removed private workspace.
        old_root = self.cache_dir
        self.cache_dir = destination
        for stage in STAGES:
            data = context.get_result(stage)
            if data is not None:
                context.results[stage] = _rebase(data, old_root, destination)
                typed = context.stage_results.get(stage)
                if typed is not None:
                    if stage == "segmentation":
                        from src.models import SegmentationResult
                        context.stage_results[stage] = SegmentationResult(**context.results[stage])
                    else:
                        context.stage_results[stage] = type(typed)(payload=context.results[stage])


def _rebase(value, old, new):
    if isinstance(value, dict):
        return {key: _rebase(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase(item, old, new) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            return str(Path(new) / Path(value).relative_to(old))
        except ValueError:
            pass
    return value


def series_reuse_requested(request, config):
    if not (config.pipeline.series_reuse_enabled
            and getattr(request, "series_cache_parent", "")
            and getattr(request, "cache_dir", "")):
        return False
    try:
        parent = Path(request.series_cache_parent).expanduser().resolve()
        cache = Path(request.cache_dir).expanduser().resolve()
        output = Path(request.output_dir).expanduser().resolve()
        return parent.is_dir() and cache.is_relative_to(parent) and not cache.is_relative_to(output)
    except (OSError, RuntimeError):
        return False


def start_series_reuse(context, config, stages):
    """Return None without any cache I/O when the optional feature is disabled."""
    request = context.request
    if not series_reuse_requested(request, config):
        return None
    if not context.metadata.get("series_reuse_source_validated"):
        return None
    if tuple(stage.name for stage in stages) != ("preprocessing", *STAGES,
                                                "reconstruction", "reporting"):
        return None
    session = None
    try:
        # __new__ makes cleanup possible even if discovery fails in __init__.
        session = SeriesReuseSession.__new__(SeriesReuseSession)
        session.__init__(context, config, stages[1])
        return session
    except Exception as error:
        if session is not None:
            session.close()
        RunLog(f"[同图缓存] 初始化失败，执行原流程: {error}")
        return None
