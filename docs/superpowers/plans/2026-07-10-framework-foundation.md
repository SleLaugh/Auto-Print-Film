# AutoPrintFilm Framework Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变六阶段医学算法的前提下，增加统一配置、预检、任务状态、安全产物管理、标准错误和任务 manifest。

**Architecture:** 保留 `Request -> Pipeline -> Stage[] -> Response` 主链，在 Pipeline 外围注入不可变 `AppConfig`、`PreflightValidator` 和 `ArtifactManager`。Pipeline 独占生命周期变更，Context 保存有类型阶段结果和阶段记录，manifest 作为可恢复的运行事实记录。

**Tech Stack:** Python 3.10+、标准库 dataclasses/enum/json/pathlib/uuid、unittest、现有 SimpleITK/nnUNet 依赖。

## Global Constraints

- 不改变预处理、分割、定位、规划、重建和报告的医学计算规则。
- 保留 `Pipeline()`、现有 `Request` 字段、Context 旧字典接口和业务输出路径兼容性。
- 不增加运行时第三方依赖，不引入数据库、队列、API、并发、重试或断点续跑。
- 预检失败不得创建或修改输出目录。
- 所有只读预检通过后，递归删除并重建本次任务专用 `output_dir`；预检失败不得删除。
- 不记录患者敏感 DICOM 字段或完整环境变量。
- 不执行 `git commit`；完成后仅提供建议提交信息。

---

### Task 1: 标准错误与生命周期模型

**Files:**
- Create: `src/core/errors.py`
- Create: `src/core/lifecycle.py`
- Modify: `src/core/__init__.py`
- Test: `tests/test_framework_lifecycle.py`

**Interfaces:**
- Produces: `FrameworkError.to_dict() -> dict`
- Produces: `JobStatus`, `StageStatus`, `StageRecord`
- Consumes: only Python standard library

- [ ] **Step 1: Write failing error serialization and lifecycle tests**

```python
def test_framework_error_serializes_stable_fields(self):
    error = CapabilityError("不支持的能力", details={"body_part": "颈椎"})
    self.assertEqual(error.to_dict(), {
        "code": "CAPABILITY_ERROR",
        "message": "不支持的能力",
        "stage": None,
        "retryable": False,
        "details": {"body_part": "颈椎"},
    })

def test_stage_record_tracks_terminal_state(self):
    record = StageRecord("preprocessing")
    record.start()
    record.succeed("preprocessing_v1")
    self.assertIs(record.status, StageStatus.SUCCEEDED)
    self.assertIsNotNone(record.duration_seconds)
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run: `python3 -m unittest tests.test_framework_lifecycle -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.errors'`.

- [ ] **Step 3: Implement errors and lifecycle types**

Implement `FrameworkError` with `code`, `message`, `stage`, `retryable`, `details`, and subclasses `ConfigurationError`, `RequestValidationError`, `CapabilityError`, `DependencyError`, `StageInputError`, `ExternalProcessError`, `ArtifactError`, `OutputValidationError`. Implement enum values exactly as the design and guarded `StageRecord.start/succeed/fail/skip/cancel` transitions with ISO-8601 timestamps and `to_dict()`.

- [ ] **Step 4: Export public types and run focused tests**

Run: `python3 -m unittest tests.test_framework_lifecycle -v`
Expected: PASS.

---

### Task 2: 不可变配置与配置加载优先级

**Files:**
- Create: `src/config/settings.py`
- Create: `src/config/loader.py`
- Modify: `src/config/__init__.py`
- Modify: `src/config/nnuNet_config.json`
- Test: `tests/test_config_loader.py`

**Interfaces:**
- Produces: `load_app_config(explicit=None, environ=None, local_path=None) -> AppConfig`
- Produces: frozen `AppConfig`, `NnUNetConfig`, `LoggingConfig`, `PipelineConfig`
- Consumes: existing repository JSON files and nnUNet environment variables

- [ ] **Step 1: Write failing precedence and empty-path tests**

```python
def test_explicit_config_overrides_environment_and_file(self):
    config = load_app_config(
        explicit={"nnunet": {"results": "/explicit/results"}},
        environ={"nnUNet_results": "/env/results"},
        defaults={"nnunet": {"results": "/file/results"}},
    )
    self.assertEqual(config.nnunet.results, "/explicit/results")

def test_empty_repository_path_keeps_environment_value(self):
    config = load_app_config(
        environ={"nnUNet_results": "/env/results"},
        defaults={"nnunet": {"results": ""}},
    )
    self.assertEqual(config.nnunet.results, "/env/results")
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python3 -m unittest tests.test_config_loader -v`
Expected: FAIL because config types and loader do not exist.

- [ ] **Step 3: Implement frozen config models and pure merge loader**

The loader must normalize legacy JSON keys (`_configuration`, `_fold`, `_continue`, `nnUNet_*`, `_model_rules`) into typed fields, merge only non-empty path values, accept an injected environment mapping for tests, and never mutate `os.environ`.

- [ ] **Step 4: Remove developer absolute paths from committed defaults**

Set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to empty strings while preserving model rules.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest tests.test_config_loader -v`
Expected: PASS.

---

### Task 3: 能力注册与只读预检

**Files:**
- Create: `src/core/capabilities.py`
- Create: `src/core/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Produces: `CapabilityRegistry.validate(request) -> None`
- Produces: `PreflightValidator.validate(request, config) -> None`
- Consumes: `Request`, `AppConfig`, executable lookup injected as a callable

- [ ] **Step 1: Write failing supported/unsupported and no-output-mutation tests**

```python
def test_registry_accepts_current_lumbar_axial_whole_range(self):
    registry = CapabilityRegistry.current()
    registry.validate(make_request("腰椎", "whole_range", "横状位"))

def test_registry_rejects_unimplemented_cervical_interspace(self):
    with self.assertRaises(CapabilityError):
        CapabilityRegistry.current().validate(
            make_request("颈椎", "interspace", "横状位")
        )

def test_preflight_failure_does_not_create_output_directory(self):
    validator = PreflightValidator(executable_finder=lambda _: None)
    with self.assertRaises(DependencyError):
        validator.validate(request, config)
    self.assertFalse(Path(request.output_dir).exists())
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python3 -m unittest tests.test_preflight -v`
Expected: FAIL because registry and validator do not exist.

- [ ] **Step 3: Implement the current capability matrix**

Register only combinations already implemented by `select_planner`: cervical/thoracic/lumbar/lung/skull/hip whole-range axial and lumbar interspace axial. Only CT is registered. Validation must list the rejected combination in `CapabilityError.details`.

- [ ] **Step 4: Implement side-effect-free preflight checks**

Validate request structure, capabilities, normalized input/output relationship, input readability, output parent availability, executable discovery, and configured nnUNet directory existence. Provide switches for tests and custom pipelines so unit tests using fake stages do not require a real model.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest tests.test_preflight -v`
Expected: PASS.

---

### Task 4: 安全产物管理和原子 Manifest

**Files:**
- Create: `src/core/artifacts.py`
- Test: `tests/test_artifact_manager.py`

**Interfaces:**
- Produces: `ArtifactManager.prepare() -> None`
- Produces: `ArtifactManager.register(...) -> StageArtifact`
- Produces: `ArtifactManager.write_manifest(data) -> Path`
- Consumes: requested output root only

- [ ] **Step 1: Write failing directory ownership tests**

```python
def test_rejects_nonempty_foreign_directory(self):
    (root / "user-file.txt").write_text("keep", encoding="utf-8")
    with self.assertRaises(ArtifactError):
        ArtifactManager(root).prepare()
    self.assertEqual((root / "user-file.txt").read_text(), "keep")

def test_prepare_recreates_output_directory(self):
    manager = ArtifactManager(root)
    manager.prepare()
    owned = root / "intermediate" / "owned.txt"
    owned.parent.mkdir(parents=True)
    owned.write_text("owned", encoding="utf-8")
    foreign = root / "keep.txt"
    foreign.write_text("keep", encoding="utf-8")
    manager.register("intermediate", owned, "intermediate")
    manager.write_manifest(manager.build_manifest_stub())
    ArtifactManager(root).prepare()
    self.assertFalse(owned.exists())
    self.assertTrue(foreign.exists())
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python3 -m unittest tests.test_artifact_manager -v`
Expected: FAIL because `ArtifactManager` does not exist.

- [ ] **Step 3: Implement ownership checks, registration and atomic JSON replacement**

Treat `output_dir` as a task-owned directory: after preflight, recreate it completely. Write `.manifest.json.tmp` and replace `manifest.json` in the same directory.

- [ ] **Step 4: Implement retention cleanup**

Delete registered `intermediate` artifacts only after success when configured; preserve them on failure; never delete `final` or unknown files. Cleanup failure returns warnings rather than masking the task error.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest tests.test_artifact_manager -v`
Expected: PASS.

---

### Task 5: 有类型阶段结果和 Context 兼容层

**Files:**
- Create: `src/models/stage_results.py`
- Modify: `src/models/__init__.py`
- Modify: `src/core/context.py`
- Test: `tests/test_stage_result_contracts.py`

**Interfaces:**
- Produces: `StageResult` plus six named result subclasses
- Produces: `ProcessingContext.set_stage_result(result)`
- Produces: `ProcessingContext.require_stage_result(stage_name, expected_type)`
- Keeps: `set_result/get_result` behavior for old stages

- [ ] **Step 1: Write failing typed storage tests**

```python
def test_typed_result_is_available_through_new_and_legacy_interfaces(self):
    result = PreprocessingResult(nifti_path="/tmp/raw.nii.gz")
    context.set_stage_result(result)
    self.assertIs(context.require_stage_result("preprocessing", PreprocessingResult), result)
    self.assertEqual(context.get_result("preprocessing")["nifti_path"], "/tmp/raw.nii.gz")

def test_require_stage_result_rejects_wrong_type(self):
    context.set_stage_result(PreprocessingResult(nifti_path="/tmp/raw.nii.gz"))
    with self.assertRaises(StageInputError):
        context.require_stage_result("preprocessing", SegmentationResult)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python3 -m unittest tests.test_stage_result_contracts -v`
Expected: FAIL because typed results do not exist.

- [ ] **Step 3: Implement common result serialization and six result types**

Each result exposes `schema_version`, fixed `stage_name`, warnings, artifacts, diagnostics, stage-specific stable fields, `to_dict()`, and `from_legacy_dict()` where required. Do not move internal geometry structures out of their current models.

- [ ] **Step 4: Add Context typed storage without breaking legacy dictionaries**

Store typed objects separately and mirror their serialized dictionary into `results`. Legacy `set_result` continues timing behavior. Missing or wrong typed inputs raise `StageInputError`.

- [ ] **Step 5: Run focused and existing Context-dependent tests**

Run: `python3 -m unittest tests.test_stage_result_contracts tests.test_stage3_target_axial_correction -v`
Expected: PASS.

---

### Task 6: Pipeline 生命周期、Manifest 与安全输出集成

**Files:**
- Modify: `src/core/context.py`
- Modify: `src/core/pipeline.py`
- Modify: `src/core/response.py`
- Modify: `src/core/__init__.py`
- Test: `tests/test_pipeline_framework.py`
- Modify: `tests/test_debug_artifact_retention.py`

**Interfaces:**
- `Pipeline(stages=None, config=None, preflight=None, artifact_manager_factory=None)`
- `Pipeline.run(request) -> Response`
- `Response.to_dict()` adds `job_id`, `status`, `warnings` while retaining old fields

- [ ] **Step 1: Write failing fake-stage lifecycle tests**

```python
def test_successful_pipeline_writes_succeeded_manifest(self):
    response = make_pipeline([SuccessfulStage("one"), SuccessfulStage("two")]).run(request)
    manifest = json.loads((Path(request.output_dir) / "manifest.json").read_text())
    self.assertTrue(response.success)
    self.assertEqual(manifest["status"], "succeeded")
    self.assertEqual([item["status"] for item in manifest["stages"]], ["succeeded", "succeeded"])

def test_failed_stage_marks_remaining_stages_skipped(self):
    response = make_pipeline([FailingStage("one"), SuccessfulStage("two")]).run(request)
    self.assertFalse(response.success)
    self.assertEqual(response.status, "failed")
    self.assertEqual(response.context.stage_records["two"].status.value, "skipped")
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python3 -m unittest tests.test_pipeline_framework -v`
Expected: FAIL because Pipeline has no lifecycle or manifest integration.

- [ ] **Step 3: Extend Context with job fields and transitions**

Generate UUID job IDs, maintain job timestamps/status, create records for all configured stages, collect warnings, and serialize a non-sensitive manifest snapshot.

- [ ] **Step 4: Replace destructive output preparation with ArtifactManager**

Remove `_prepare_output_dir` recursive deletion. Run preflight before `ArtifactManager.prepare`. Preserve `_cleanup_debug_artifacts` only as a compatibility helper during migration; Pipeline cleanup must use registered artifacts.

- [ ] **Step 5: Integrate lifecycle and atomic manifest updates**

Write manifest at job start, stage start, every terminal stage state, and job completion. Wrap unknown exceptions into the standard serialized error while retaining the original exception chain in logs. Mark untouched stages skipped after fail-fast.

- [ ] **Step 6: Extend Response compatibly**

Keep `success`, `output_dir`, `total_time`, errors, timings, and three plane counts. Add job ID, job status, warnings, and stage status summary.

- [ ] **Step 7: Run focused pipeline tests**

Run: `python3 -m unittest tests.test_pipeline_framework tests.test_debug_artifact_retention -v`
Expected: PASS.

---

### Task 7: 入口集成、项目命令和文档

**Files:**
- Modify: `main.py`
- Create: `scripts/test.sh`
- Create: `scripts/test.ps1`
- Modify: `README.md`
- Test: `tests/test_main_preflight.py`

**Interfaces:**
- CLI keeps all current flags and exit codes
- Test scripts invoke the project virtual environment explicitly

- [ ] **Step 1: Write failing CLI framework-error test**

```python
def test_main_returns_one_for_preflight_error_without_creating_output(self):
    with mock.patch("main.Pipeline.run", side_effect=CapabilityError("不支持")):
        exit_code = main.main()
    self.assertEqual(exit_code, 1)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run: `python3 -m unittest tests.test_main_preflight -v`
Expected: FAIL until the entrypoint handles standard framework errors consistently.

- [ ] **Step 3: Load configuration once and pass it to Pipeline**

Keep argument parsing unchanged. Convert known framework failures into concise user messages and exit code 1; send detailed cause to error logging without dumping sensitive request metadata.

- [ ] **Step 4: Add deterministic test scripts**

Linux script executes `.venv/bin/python -m unittest discover -s tests -v`; PowerShell executes `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Both fail clearly when the virtual environment is missing.

- [ ] **Step 5: Update README**

Document configuration precedence, environment-variable model paths, current capability matrix, safe output ownership behavior, manifest fields, and the single recommended test command.

- [ ] **Step 6: Run the full regression suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: all original 75 tests plus new framework tests PASS.

Run: `python3 -m compileall -q main.py src tests`
Expected: exit code 0 with no output.

Run: `git diff --check`
Expected: exit code 0 with no output.

---

## Final Review Checklist

- [ ] Every unsupported request fails before output mutation.
- [ ] No developer absolute nnUNet paths remain in committed defaults.
- [ ] No Pipeline code recursively deletes a foreign output directory.
- [ ] Every terminal response matches the manifest job status.
- [ ] Existing Context dictionary consumers remain functional.
- [ ] No patient metadata enters logs, manifest or errors.
- [ ] Full tests, compile check and whitespace check pass.
