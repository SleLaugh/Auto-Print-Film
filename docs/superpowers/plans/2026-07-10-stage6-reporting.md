# Stage 6 Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 完善阶段六的报告、输出清单和第五步产物一致性检查，不改变任何重建图像。

**Architecture:** `ReportingStage` 只消费上下文中已有的 `mapping` 与 `reconstruction` 结果，以及输出目录中的最终 DICOM 文件。正常 `report.json` 只提供部位、检查类型和带实际张数/窗位路径的 `plane_configs`；完整计划与校验细节只在 debug 模式写入 `metadata/处理报告_调试.json`。

**Tech Stack:** Python、JSON、`pathlib`、SimpleITK（只读取 DICOM 元数据用于校验）。

## Global Constraints

- 第六步不得重新计算切割线、切片数量、层厚或层间距。
- 第五步生成的 DICOM 目录结构保持不变。
- `report.json` 是正常交付文件，不受 debug 文件清理影响。
- 校验失败只写入 debug 报告，不修改已经生成的 DICOM。

### Task 1: 简洁交付报告与调试报告

**Files:**
- Modify: `src/stages/stage_reporting.py`
- Test: `tests/test_reporting_plane_configs.py`

**Interfaces:**
- Consumes: `context.results["mapping"]` and `context.results["reconstruction"]`.
- Produces: 简洁 `report.json` 和 debug 模式下的 `metadata/处理报告_调试.json`。

- [ ] **Step 1: Write failing tests**

Add tests asserting that `report.json` only contains `body_part`, `check_type`, and enriched `plane_configs`. Assert each plane has physical `image_count`, and each requested window has `name` and actual `path`. Keep validation assertions on the debug report.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest -q tests/test_reporting_plane_configs.py`

Expected: FAIL because the current normal report still exposes the full debug structure.

- [ ] **Step 3: Implement report aggregation**

Add a delivery-report builder that enriches request `plane_configs` with physical slice count and per-window root path. Move the existing plan, UID, numbering, file-count, warning, and timing details into a debug-only report writer.

- [ ] **Step 4: Run focused tests and the full suite**

Run: `pytest -q tests/test_reporting_plane_configs.py && pytest -q`

Expected: all tests pass.

### Task 2: Pipeline handoff and final verification

**Files:**
- Modify: `tests/test_reporting_plane_configs.py`

**Interfaces:**
- Consumes: `ReportingStage.execute()` result.
- Produces: `report_path` and `validation` summary available in `context.results["reporting"]`.

- [ ] **Step 1: Add execute-level test**

Assert that `ReportingStage.execute()` stores the report path, validation summary, and total image count in the reporting result.

- [ ] **Step 2: Run verification**

Run: `pytest -q && python -m compileall -q src tests && git diff --check`

Expected: all tests pass and no whitespace errors are reported.
