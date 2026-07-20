# 髋骨倾斜直线切完整投影覆盖 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保持髋骨 AB 倾斜角度不变，并让横状位范围完整覆盖左右髋骨最大连通主体。

**Architecture:** 阶段三继续用左右外侧顶部区域中心 A、B 计算横状位三轴，但不再把 AB 平面作为头侧边界；三轴确定后，用全部保留点在最终头脚法向上的最小、最大投影生成 `AnatomicalFrame.bounds`。阶段四、Debug、Localizer 和阶段五继续消费统一计划，不增加髋骨业务分支。

**Tech Stack:** Python、NumPy、SimpleITK、unittest。

## Global Constraints

- AB 只决定角度，覆盖范围只能来自完整左右髋骨主体。
- 只修改髋骨横状位边界；AB、矢状位、冠状位方向保持不变。
- 孤立误分割仍由每侧最大连通域过滤，不得进入角度或范围。
- 配置的 `slice_range_extension` 在完整 Mask 边界之外继续生效。
- 第四步和第五步不得增加髋骨专属范围修补逻辑。
- 所有新增诊断字段和方法修改必须有中文注释。
- 本轮不执行 Git 提交，完成后保留差异等待用户审批。

---

### Task 1: 用完整主体投影确定髋骨上下界

**Files:**
- Modify: `tests/test_hip_anatomical_frame.py`
- Modify: `src/anatomy/hip.py`

**Interfaces:**
- Consumes: 左右最大连通主体物理点 `all_points` 和由 AB 得到的单位法向 `si_axis`。
- Produces: `frame.bounds.superior_inferior = AxisBounds(min_projection, max_projection)`，以及 AB 平面与完整覆盖边界诊断。

- [x] **Step 1: Write the failing projection-coverage test**

保留现有倾斜、正交和孤立点过滤断言；从测试主体 Mask 生成物理点并增加：

```python
retained_points = np.argwhere(retained_mask)[:, [2, 1, 0]].astype(float)
projections = retained_points @ si
self.assertAlmostEqual(frame.bounds.superior_inferior.minimum, projections.min())
self.assertAlmostEqual(frame.bounds.superior_inferior.maximum, projections.max())
```

同时根据诊断中的 A、B 重算 AB 单位方向，断言最终 `left_right_axis` 不变。

- [x] **Step 2: Run the test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_hip_anatomical_frame -v`

Expected: FAIL；当前上界等于 AB 平面投影，低于完整主体最大投影。

- [x] **Step 3: Implement the minimal boundary correction**

在最终 `si_axis` 确定后统一计算：

```python
si_bounds = _axis_bounds(all_points, si_axis)
ab_plane_projection = float(np.mean([
    np.dot(left_landmark, si_axis),
    np.dot(right_landmark, si_axis),
]))
```

将 `si_bounds` 直接写入 `AnatomicalBounds.superior_inferior`，并记录 AB 投影、完整上下界及二者差值；不要修改 AB 和三个单位轴。

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_hip_anatomical_frame tests.test_stage3_target_axial_correction tests.test_stage4_slice_planning -v`

Expected: 所有髋骨几何、阶段三和阶段四测试通过。

### Task 2: 文档一致性与全量验证

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-16-hip-straight-mpr-design.md`

**Interfaces:**
- Consumes: Task 1 已验证的最终行为。
- Produces: 与实现一致的髋骨角度和覆盖范围说明。

- [x] **Step 1: Verify Stage 4 and Stage 5 remain generic**

检查 `git diff -- src/planning src/stages/stage_reconstruction.py`，预期没有为了本修正新增髋骨分支；既有计划字段自动驱动 Debug、Localizer 和正式 DICOM。

- [x] **Step 2: Run full verification**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: 全量测试通过，编译和差异检查退出码均为 0；工作区仅保留本轮文件及用户已有 `.vscode/launch.json`，不生成 Git 提交。
