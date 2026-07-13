# 通用直线整切全切 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有已支持整切的部位通过同一个 bbox 直线规划入口生成横状位、矢状位和冠状位 `plane_stack`。

**Architecture:** 部位 planner 只选择目标范围，`src/planning/whole_range.py` 统一维护三状位的堆叠轴、平面轴、方向和像素间距。阶段五继续只消费既有 `SlicePlan`，不增加部位判断；角度能力以后只替换方向来源。

**Tech Stack:** Python 3.11、dataclasses、SimpleITK、unittest。

## Global Constraints

- 本轮仅实现与输入图像轴对齐的直线整切，不实现回正、角度估计、曲线重建或部分切。
- 所有新增和修改的类、方法、字段使用中文注释。
- 保持 `SlicePlan` 和 `slice_plan_v1` 协议兼容。
- 不创建分支，不执行 `git commit`。

---

### Task 1: 通用三状位 bbox 整切几何

**Files:**
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `src/planning/whole_range.py`

**Interfaces:**
- Consumes: `build_plane_stack_slices(...)`、`image_axis_direction(...)`、`voxel_index_to_physical(...)`。
- Produces: `build_whole_range_plan_from_bbox(request, plane_config, localization_result, source_labels=None) -> SlicePlan`。
- Preserves: `build_axial_whole_range_plan_from_bbox(...)` 作为迁移期兼容别名，避免旧调用方突然失效。

- [ ] **Step 1: 写三状位轴映射失败测试**

在 `tests/test_stage4_slice_planning.py` 直接调用现有公共构造入口，分别传入横、矢、冠配置，断言方向、范围和输出像素间距。测试先使用现有函数名，等通用实现完成后再切换到新名称，这样 RED 来自当前明确的“只能处理横状位”限制：

```python
def test_whole_range_builder_builds_all_straight_planes(self):
    request = Request(
        dicom_dir="/tmp/dicom",
        output_dir="/tmp/output",
        body_part=BodyPartEnum.肺部,
        cut_mode=CutModeEnum.整切,
        plane_configs=[
            PlaneConfig(PlaneTypeEnum.横状位, 3.0, 5.0, [WindowPresetEnum.肺窗]),
            PlaneConfig(PlaneTypeEnum.矢状位, 3.0, 5.0, [WindowPresetEnum.肺窗]),
            PlaneConfig(PlaneTypeEnum.冠状位, 3.0, 5.0, [WindowPresetEnum.肺窗]),
        ],
    )
    localization_result = self._lumbar_localization_result()
    localization_result["image"]["spacing_xyz"] = [1.0, 2.0, 3.0]
    localization_result["image"]["size_xyz"] = [100, 120, 30]

    axial, sagittal, coronal = [
        build_axial_whole_range_plan_from_bbox(request, config, localization_result)
        for config in request.plane_configs
    ]
    self.assertEqual(axial.output_spacing_mm.to_dict(), {"x": 1.0, "y": 2.0, "z": 0.0})
    self.assertEqual(sagittal.output_spacing_mm.to_dict(), {"x": 2.0, "y": 3.0, "z": 0.0})
    self.assertEqual(coronal.output_spacing_mm.to_dict(), {"x": 1.0, "y": 3.0, "z": 0.0})
    self.assertEqual(sagittal.slices[0].normal.to_dict(), {"x": -1.0, "y": 0.0, "z": 0.0})
    self.assertEqual(coronal.slices[0].normal.to_dict(), {"x": 0.0, "y": 1.0, "z": 0.0})
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_whole_range_builder_builds_all_straight_planes -v`

Expected: FAIL with `ValueError: 横状位整切公共方法只能处理横状位`。

- [ ] **Step 3: 写 target_bbox 平面范围失败测试**

增加参数化子测试，分别断言三状位 `target_bbox` 的 width/height 使用 x/y、y/z、x/z 对应轴；同时断言 `full_plane` 使用对应 `size_xyz` 和 `spacing_xyz`。

```python
expected_target_extents = {
    PlaneTypeEnum.横状位: {"width": 10.0, "height": 20.0},
    PlaneTypeEnum.矢状位: {"width": 20.0, "height": 30.0},
    PlaneTypeEnum.冠状位: {"width": 10.0, "height": 30.0},
}
```

- [ ] **Step 4: 运行范围测试确认 RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_whole_range_target_extent_uses_plane_axes -v`

Expected: FAIL，因为现有 bbox 范围固定使用 x/y。

- [ ] **Step 5: 实现最小通用几何**

在 `src/planning/whole_range.py` 增加带完整中文类、字段和方法注释的不可变 `_PlaneAxes`，并建立固定映射：

```python
@dataclass(frozen=True)
class _PlaneAxes:
    """描述默认直线状位对应的输入图像轴和方向符号。"""

    stack_axis: int
    """切片中心依次移动的输入图像轴。"""
    stack_sign: float
    """切片法向相对 stack_axis 的方向符号。"""
    row_axis: int
    """输出图像横向对应的输入图像轴。"""
    row_sign: float
    """输出图像横向相对 row_axis 的方向符号。"""
    col_axis: int
    """输出图像纵向对应的输入图像轴。"""
    col_sign: float
    """输出图像纵向相对 col_axis 的方向符号。"""
```

映射使用：横状位 `(z+, x+, y+)`，矢状位 `(x-, y+, z-)`，冠状位 `(y+, x+, z-)`。该组合保证 `row_dir × col_dir == normal`，并让矢状位、冠状位的头侧显示在图像上方。

把 bbox 边界、向内缩半层厚、平面 extent 和输出 spacing 都改为根据 `_PlaneAxes` 计算；实现 `build_whole_range_plan_from_bbox(...)`，保留横状位旧函数作为仅转发的兼容入口。

- [ ] **Step 6: 运行新增和原横状位测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning -v`

Expected: 所有阶段四测试 PASS，原横状位范围和切片数不回归。

---

### Task 2: 部位 planner 与能力表接入通用入口

**Files:**
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `tests/test_preflight.py`
- Modify: `src/planning/single_label.py`
- Modify: `src/planning/lumbar.py`
- Modify: `src/core/capabilities.py`

**Interfaces:**
- Consumes: `build_whole_range_plan_from_bbox(...)`。
- Produces: 所有已登记整切部位的横、矢、冠计划；腰椎椎间切能力保持只支持横状位。

- [ ] **Step 1: 写 planner 复用与能力表失败测试**

把旧的“未来状位未实现”测试替换为三状位均生成计划的断言；增加腰椎整切三状位测试，以及能力表包含全部整切部位三状位组合的测试：

```python
def test_current_capabilities_register_three_planes_for_whole_range(self):
    combinations = set(CapabilityRegistry.current().combinations())
    for body_part in ("颈椎", "胸椎", "腰椎", "肺部", "头骨", "髋骨"):
        for plane in ("横状位", "矢状位", "冠状位"):
            self.assertIn((body_part, "whole_range", plane), combinations)
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning tests.test_preflight -v`

Expected: FAIL，因为两个整切 planner 仍过滤非横状位，能力表也只登记横状位。

- [ ] **Step 3: 接入通用构造器**

修改 `SingleLabelWholeRangePlanner` 和 `LumbarWholeRangePlanner`：遍历 `request.plane_configs` 时无条件调用 `build_whole_range_plan_from_bbox(...)`，删除整切状位 warning 分支，并同步中文类和方法注释。`LumbarInterspacePlanner` 不修改行为。

修改 `CapabilityRegistry.current()`，用整切部位和三个状位的笛卡尔积登记能力，保留 `("腰椎", "interspace", "横状位")`。

- [ ] **Step 4: 运行阶段四与预检测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning tests.test_preflight -v`

Expected: 全部 PASS。

---

### Task 3: 三状位计划到第五阶段的集成验证

**Files:**
- Modify: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: 三状位 `SlicePlan.to_dict()` 与 `ReconstructionStage._group_plans_by_plane(...)`。
- Produces: 三状位直线计划均可被既有 `PlaneStackResampler` 采样的回归保障。

- [ ] **Step 1: 写三状位重采样测试**

构造带明显 x/y/z 梯度的 3D float32 图像，分别执行横、矢、冠三个计划的第一张切片，断言输出尺寸由对应 `extent_mm/output_spacing_mm` 决定、输出方向等于计划中的 `row_dir/col_dir/normal`，且结果不是全默认值。

- [ ] **Step 2: 运行测试确认当前通用契约可执行**

Run: `.venv/bin/python -m unittest tests.test_stage5_reconstruction -v`

Expected: PASS；阶段五已经是通用 `plane_stack` 执行器，因此本任务只增加契约回归测试，不应修改生产代码。

- [ ] **Step 3: 运行完整测试集**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: 全部 PASS，无新增异常或 warning。

- [ ] **Step 4: 检查工作区边界**

Run: `git status --short`

Expected: 保留用户已有 `.vscode/launch.json`、`src/config/logger_config.json` 修改；只新增本设计/计划及本轮规划、能力、测试文件，不提交。

---

### Task 4: 分离横状位序列方向与平面法向

**Files:**
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `src/planning/whole_range.py`

- [x] **Step 1: 添加横状位从上到下的失败测试**

断言横状位 `range_start` 和第一张切片位于 z 轴高端，`range_end` 和最后一张位于 z 轴低端；同时断言行方向、列方向和法向保持现有右手系定义。

- [x] **Step 2: 运行单测确认 RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_axial_whole_range_orders_slices_from_superior_to_inferior -v`

Expected: 顺序断言 FAIL，证明测试覆盖的是当前缺失行为。

- [x] **Step 3: 拆分路径方向与法向符号**

将 `_PlaneAxes.stack_sign` 拆为带完整中文字段注释的 `path_sign` 和 `normal_sign`。bbox 起止点只使用 `path_sign`，切片法向只使用 `normal_sign`；横状位设置为路径反向、法向保持正向，矢状位和冠状位维持现状。

- [x] **Step 4: 运行针对性测试和完整测试**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning -v`

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: 全部 PASS，并通过 `git diff --check`。
