# 横矢冠三状位切片总览 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 调试模式下为横状位、矢状位、冠状位分别生成普通检查网格总览和原始网格总览。

**Architecture:** 阶段四先按状位分组，再用一套输入轴配置驱动通用索引绘制。每个状位独立创建两个 NIfTI；新路径字典承载全部文件，旧单路径字段继续指向横状位或第一个可用状位。

**Tech Stack:** Python 3、SimpleITK、NumPy、unittest、现有 `MappingStage`。

## Global Constraints

- 只支持当前零角度横、矢、冠直线计划的总览，不实现任意角度投影。
- 原始网格总览必须完整复制参考图像的 size、spacing、origin、direction。
- 普通总览只把当前状位法向输入轴细化为 `0.5 mm`。
- 保留现有旧路径字段，并新增按状位保存的路径字典。
- 新增类、字段、方法和关键计算必须有完整中文注释。
- 直接修改当前工作区，不创建分支，不执行 `git commit`。

---

### Task 1: 三状位文件与结果路径契约

**Files:**
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `src/stages/stage_mapping.py`

**Interfaces:**
- Consumes: `plans: list[dict]` 中的 `plane`、`cut_mode` 和 slices。
- Produces: `_write_slice_plan_overview_overlays(...) -> tuple[dict[str, str], dict[str, str]]`，以及结果字段 `debug_slice_plan_overview_paths`、`debug_slice_plan_native_overview_paths`。

- [ ] **Step 1: 写三状位六文件失败测试**

在 `Stage4SlicePlanningTest` 构造同时包含横、矢、冠的整切请求，执行 `MappingStage` 后断言：

```python
self.assertEqual(set(result["debug_slice_plan_overview_paths"]), {
    "横状位", "矢状位", "冠状位",
})
self.assertEqual(set(result["debug_slice_plan_native_overview_paths"]), {
    "横状位", "矢状位", "冠状位",
})
self.assertEqual(
    Path(result["debug_slice_plan_native_overview_paths"]["矢状位"]).name,
    "切片原始网格总览_矢_整.nii.gz",
)
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_mapping_stage_writes_separate_overviews_for_three_planes -v`

Expected: FAIL，结果中不存在按状位路径字典，且矢、冠文件未生成。

- [ ] **Step 3: 实现按状位分组和兼容字段**

在 `MappingStage` 中：

```python
overview_paths, native_overview_paths = (
    self._write_slice_plan_overview_overlays(
        context=context,
        localization_result=prev_result,
        plans=plans,
    )
)
debug_overview_overlay_path = self._legacy_overview_path(
    overview_paths,
    plans,
)
debug_native_overview_overlay_path = self._legacy_overview_path(
    native_overview_paths,
    plans,
)
```

`_write_slice_plan_overview_overlays` 只为实际存在的三个已知状位分组写文件；
`_legacy_overview_path` 优先返回横状位，否则按 plans 首次出现顺序返回路径。
阶段结果和 `slice_plans.json` 同时写入新旧字段。

- [ ] **Step 4: 运行目标测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_mapping_stage_writes_separate_overviews_for_three_planes -v`

Expected: PASS，六个路径均存在且文件名正确。

---

### Task 2: 通用三状位检查网格与绘制

**Files:**
- Modify: `src/stages/stage_mapping.py`
- Modify: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Consumes: plane 名称、参考图像、单状位 plans。
- Produces: `_OverviewAxes(stack_axis, fixed_axis, line_axis)`、`_build_uniform_overview_reference(reference_image, stack_axis)`、`_paint_plan_overview(...)`。

- [ ] **Step 1: 扩充失败测试中的图像几何断言**

测试使用非立方体参考图像和非等距 spacing，并断言：

```python
self.assertEqual(axial_image.GetSpacing(), (1.0, 2.0, 0.5))
self.assertEqual(sagittal_image.GetSpacing(), (0.5, 2.0, 3.0))
self.assertEqual(coronal_image.GetSpacing(), (1.0, 0.5, 3.0))
for path in result["debug_slice_plan_overview_paths"].values():
    self.assertTrue(np.any(sitk.GetArrayFromImage(sitk.ReadImage(path)) > 0))
```

并断言三个原始网格总览的 size、spacing、origin、direction 与参考图一致。

- [ ] **Step 2: 运行测试确认 RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_mapping_stage_writes_separate_overviews_for_three_planes -v`

Expected: FAIL，旧实现只细化 Z 且只绘制横状位。

- [ ] **Step 3: 实现带完整中文注释的轴配置和通用绘制**

在 `stage_mapping.py` 定义不可变 `_OverviewAxes`：

```python
@dataclass(frozen=True)
class _OverviewAxes:
    """一个默认直线状位在输入图像索引空间中的总览轴配置。"""
    stack_axis: int
    fixed_axis: int
    line_axis: int
```

固定映射为：横 `(2, 0, 1)`、矢 `(0, 1, 2)`、冠 `(1, 0, 2)`。

`_build_uniform_overview_reference` 复制输入 size/spacing，并只把 stack axis
改成 `0.5 mm`，用 `(size - 1) * spacing` 保持物理覆盖长度。

`_paint_plan_overview` 将物理中心转换为连续 x/y/z 索引，固定 fixed axis 的
中位数位置；沿 line axis 绘制 `extent_mm.height`，沿 stack axis 绘制真实
层厚覆盖带，最后覆盖绘制中心线。数组赋值前把 x/y/z 索引转换为 z/y/x。

- [ ] **Step 4: 运行阶段四测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning -v`

Expected: PASS；既有横状位 0.5 mm、中心线和厚度带断言继续成立。

---

### Task 3: 完整回归与交接

**Files:**
- Verify: `src/stages/stage_mapping.py`
- Verify: `tests/test_stage4_slice_planning.py`
- Verify: `docs/superpowers/specs/2026-07-13-three-plane-overview-overlay-design.md`

**Interfaces:**
- Consumes: Task 1、Task 2 的完整改动。
- Produces: 可测试的未提交工作区。

- [ ] **Step 1: 运行完整测试集**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: 所有测试 PASS，无 failure 和 error。

- [ ] **Step 2: 运行语法和差异检查**

Run: `.venv/bin/python -m compileall -q src tests`

Expected: exit code 0。

Run: `git diff --check`

Expected: exit code 0，无输出。

- [ ] **Step 3: 检查未提交文件并交接**

Run: `git status --short`

Expected: 显示本轮和此前工作区改动，不执行 `git add` 或 `git commit`。
