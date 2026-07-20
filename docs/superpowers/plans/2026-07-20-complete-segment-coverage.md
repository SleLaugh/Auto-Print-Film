# Complete Segment Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有直线整切和区间切以固定层间距让首末切片中心覆盖目标边界，并修正输出 DICOM 的层间距标签。

**Architecture:** 在 `src/planning/straight.py` 提供唯一的对称外扩计算器，普通直线计划和解剖坐标计划都调用它。第五步继续执行展开后的中心点；GDCM 与 pydicom 在临时文件内完成 DICOM 写出和标签校正后，再原子发布正式文件。

**Tech Stack:** Python、SimpleITK、NumPy、pydicom、unittest。

## Global Constraints

- 所有部位、三个直线状位、整切和区间切使用同一覆盖算法。
- 相邻切片中心距离必须严格等于请求层间距。
- 允许首尾对称超出目标，但首尾未覆盖量必须为零。
- 使用满足完整覆盖条件的最少切片数。
- 不修改 `.vscode/launch.json`，不主动提交 Git。

---

### Task 1: 通用对称外扩计算器

**Files:**
- Modify: `src/planning/straight.py`
- Test: `tests/test_planning_geometry.py`

**Interfaces:**
- Consumes: `target_start: Point3D`、`target_end: Point3D`、`slice_thickness: float`、`slice_spacing: float`。
- Produces: `cover_segment_range(...) -> tuple[Point3D, Point3D]`，返回第一张和最后一张切片中心。

- [x] **Step 1: Write failing coverage tests**

测试 272.35mm、5mm 层厚、5mm 间距，断言得到 56 张中心、中心跨度 275mm、首尾各超出 1.325mm；测试可整除范围、非零短区间、零长度及不超过 `0.001mm` 坐标容差的目标，以及层间距大于层厚。

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_planning_geometry -v`

Expected: FAIL，因为 `cover_segment_range` 尚不存在。

- [x] **Step 3: Implement minimal symmetric coverage**

实现等价公式，其中层厚只保留作参数校验和后续 slab 采样使用；距离最近合法
中心跨度不超过 `0.001mm` 时吸收坐标舍入误差：

```python
raw_step_count = target_length / spacing
nearest_step_count = round(raw_step_count)
if abs(nearest_step_count * spacing - target_length) <= 0.001:
    step_count = nearest_step_count
else:
    step_count = math.ceil(raw_step_count)
center_span = step_count * spacing
center_overflow = center_span - target_length
range_start = target_start - direction * (center_overflow / 2.0)
range_end = range_start + direction * step_count * spacing
```

零长度或长度不超过 `0.001mm` 坐标容差的目标返回中点两次以表示同一个中心；
超过容差的非零短区间仍生成两个首末中心。
非法厚度、间距必须抛出明确异常。

- [x] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_planning_geometry -v`

Expected: 全部通过。

---

### Task 2: 普通与解剖直线计划统一接入

**Files:**
- Modify: `src/planning/straight.py`
- Modify: `src/planning/whole_range.py`
- Modify: `src/stages/stage_mapping.py`
- Test: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Consumes: Task 1 的 `cover_segment_range`。
- Produces: 首尾完整覆盖、固定间距的既有 `SlicePlan`，协议字段不变。

- [x] **Step 1: Write failing integration tests**

为普通 bbox、倾斜区间及肺部解剖坐标分别断言：首末切片中心覆盖目标、所有间距统一、272.35mm 肺部生成 56 张。更新旧的“内缩后丢弃不足间距尾段”断言。

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_stage4_slice_planning -v`

Expected: FAIL，现有 `inset_segment_range` 仍会留下尾端未覆盖。

- [x] **Step 3: Route both builders through coverage helper**

`build_straight_slice_plan` 和 `_build_whole_range_plan_from_anatomical_frame` 同时传入层厚、层间距；保留原始 `target_start/target_end`，把对称外扩后的中心写入 `range_start/range_end`。Debug 诊断增加 `leading_center_overflow_mm`、`trailing_center_overflow_mm`、`leading_center_uncovered_mm` 和 `trailing_center_uncovered_mm`，并保持两个中心遗漏字段为零；原有厚层边缘诊断继续单独保留。

- [x] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_stage4_slice_planning -v`

Expected: 全部通过，且 Debug 断言显示无首尾遗漏。

---

### Task 3: DICOM 层间距标签保持计划值

**Files:**
- Modify: `requirements.txt`
- Modify: `src/reconstruction/dicom.py`
- Test: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: `plan["slice_thickness"]`、`plan["slice_spacing"]`。
- Produces: 可回读的 `(0018,0050) SliceThickness` 和 `(0018,0088) SpacingBetweenSlices`。

- [x] **Step 1: Write failing DICOM round-trip test**

写出层厚 5mm、层间距 5mm 的单张测试 DICOM，再用 pydicom 回读，断言两个标签均为 `5.0`。当前 GDCM 会把 `SpacingBetweenSlices` 覆盖为 `1`，测试应先失败。

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_stage5_reconstruction -v`

Expected: FAIL，回读 `(0018,0088)` 为 `1`。

- [x] **Step 3: Correct geometry tags after GDCM write**

让 GDCM 先写同目录临时文件，再用 pydicom 回读临时文件并设置：

```python
dataset.SliceThickness = str(plan["slice_thickness"])
dataset.SpacingBetweenSlices = str(plan["slice_spacing"])
dataset.save_as(temporary_path, enforce_file_format=True)
```

派生 DICOM 统一声明 `ISO_IR 192`，确保 GDCM 写出的 UTF-8 中文与字符集标签一致；
所有步骤成功后才替换正式路径。将 `pydicom>=3.0.0` 写入依赖文件，并为辅助方法
补充中文注释。

- [x] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_stage5_reconstruction -v`

Expected: 全部通过。

---

### Task 4: 文档和全量验证

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-20-complete-segment-coverage-design.md`

**Interfaces:**
- Consumes: 前三项已经验证的行为。
- Produces: 可供后续部位开发复用的边界规则说明。

- [x] **Step 1: Update documentation**

README 说明所有直线切片采用固定间距、最少张数和首尾对称外扩；设计文档记录最终函数名及诊断字段。

- [x] **Step 2: Run full verification**

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```

Expected: 全量测试通过，编译和差异检查退出码均为 0；Git 状态中 `.vscode/launch.json` 保持为用户原有未提交修改。
