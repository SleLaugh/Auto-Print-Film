# 切片中心完整覆盖修正实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有直线切片计划的首末成片中心覆盖目标边界，修复肺部 Mask 为 `z=103..522` 时输出中心仅到约 `105..520` 的问题。

**Architecture:** 保留阶段三原始目标边界和阶段五重建职责，仅修改阶段四通用 `cover_segment_range`。切片数量只由目标长度与中心间距决定，超出量首尾对称分配；层厚继续只影响第五步 slab 采样。

**Tech Stack:** Python、SimpleITK、NumPy、unittest。

## Global Constraints

- 所有部位、三个直线状位、整切与区间切共享同一中心覆盖规则。
- 相邻切片中心严格保持请求的 `slice_spacing`。
- 首末中心允许超出目标，不允许位于目标边界内侧。
- 不修改阶段三定位或阶段五 MPR 采样逻辑。
- 不修改或提交用户的 `.vscode/launch.json`，本次不执行 git commit。

---

### Task 1: 用真实肺部边界锁定中心覆盖语义

**Files:**
- Modify: `tests/test_planning_geometry.py`
- Modify: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Consumes: `cover_segment_range(...)` 和 `SingleLabelWholeRangePlanner.build_plans(...)`。
- Produces: 目标长度 `272.35mm`、间距 `5mm` 时生成 56 张且首末中心越过目标的回归契约。

- [x] **Step 1: Write failing tests**

将通用几何测试改为断言中心跨度 `275mm`、56 张、首尾中心各超出 `1.325mm`；增加真实肺部 spacing/origin/bounds 场景，断言输出连续索引最大值不小于 522、最小值不大于 103。

- [x] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_planning_geometry tests.test_stage4_slice_planning -v`

Expected: FAIL，旧实现的首末中心仍位于肺部目标边界内侧。

---

### Task 2: 修改第四步通用中心覆盖算法与诊断

**Files:**
- Modify: `src/planning/straight.py`
- Modify: `src/stages/stage_mapping.py`
- Modify: `src/planning/whole_range.py`

**Interfaces:**
- Consumes: `target_start`、`target_end`、`slice_spacing`，并保留 `slice_thickness` 供参数校验及后续计划使用。
- Produces: `range_start`、`range_end` 为首末切片中心，满足 `center_span >= target_length`。

- [x] **Step 1: Implement minimum center count**

使用 `step_count = ceil(target_length / slice_spacing)`，毫米级舍入误差靠近整数步时吸附到最近步数；`center_overflow = step_count * slice_spacing - target_length`，首尾各分配一半。

- [x] **Step 2: Extend diagnostics**

新增 `leading_center_uncovered_mm`、`trailing_center_uncovered_mm`、
`leading_center_overflow_mm` 和 `trailing_center_overflow_mm`；保留原有厚层边缘诊断，
更新说明文字以区分两种语义。

- [x] **Step 3: Run targeted tests to verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_planning_geometry tests.test_stage4_slice_planning -v`

Expected: PASS，所有中心间距仍严格等于配置值。

---

### Task 3: 更新说明

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-20-complete-segment-coverage-design.md`

**Interfaces:**
- Consumes: 已通过测试的最终行为。
- Produces: 清晰区分切片中心覆盖与厚层物理覆盖的项目说明。

- [x] **Step 1: Update documentation**

记录 `N = ceil(L / S) + 1`、层厚只影响 slab，以及真实肺部现在生成 56 张、
首末中心越过目标边界的原因。

---

### Task 4: 全量验证

**Files:**
- Verify: `src/`
- Verify: `tests/`

**Interfaces:**
- Consumes: 前三项已完成的代码、测试和说明。
- Produces: 全量测试、编译及差异格式检查结果。

- [x] **Step 1: Run full verification**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: 全量测试通过。

Run: `.venv/bin/python -m compileall -q src tests`

Expected: exit code 0。

Run: `git diff --check`

Expected: exit code 0，且 `.vscode/launch.json` 未被本次修改。
