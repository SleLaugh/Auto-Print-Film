# Cervical and Thoracic Straight MPR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为颈椎和胸椎建立首尾椎体质心确定的完整三维坐标系，并生成与该轴垂直或平行的三状位直线 MPR、Debug 定位线和正式 DICOM。

**Architecture:** 新增独立脊柱解剖坐标计算器，阶段三按配置传入首尾标签和范围内全部椎体 Mask，输出既有 `AnatomicalGeometryResult`。阶段四把该坐标系映射为通用 `SlicePlan`；阶段五继续执行 `center`、`normal`、`row_dir`、`col_dir`，不增加部位分支。

**Tech Stack:** Python、NumPy、SimpleITK、unittest、现有 `AnatomicalFrame`、`SlicePlan` 和 `PlaneStackResampler`

## Global Constraints

- 颈椎范围读取配置 `C1 → C7`，胸椎范围读取配置 `C7 → L1`，不得在计算器中硬编码。
- 使用首尾椎体物理质心确定完整 XYZ 三维直线，全部椎体 Mask 只负责中心和范围。
- 横状切片垂直于脊柱轴；矢状位和临时冠状位平面包含脊柱轴。
- 每个状位内部切片严格平行且层间距统一。
- 冠状位本轮只做直线 MPR，不实现或命名为 CPR。
- 阶段五不增加颈椎、胸椎业务判断。
- 肺部、腰椎和头骨行为保持不变。
- 不主动提交 Git。

---

### Task 1: 三维脊柱解剖坐标计算器

**Files:**
- Create: `src/anatomy/spine.py`
- Modify: `src/anatomy/__init__.py`
- Test: `tests/test_spine_anatomical_frame.py`

**Interfaces:**
- Consumes: `label_masks: dict[str, np.ndarray]`、`start_label: str`、`end_label: str`、`image: sitk.Image`。
- Produces: `estimate_spine_frame_with_diagnostics(...) -> tuple[AnatomicalGeometryResult, dict[str, Any]]`。

- [ ] **Step 1: Write failing frame tests**

构造 C1、C4、C7 三个分离的立方 Mask，使 C1→C7 同时具有 x、y、z 分量。断言最终脊柱轴等于与 DICOM 头侧对齐后的质心连线，三轴单位正交，原点等于全部目标物理点均值，诊断信息记录首尾标签与质心。

增加测试：首标签缺失、末标签缺失、质心重合必须抛出明确 `ValueError`；中间标签缺失时继续并写入 `missing_labels`；翻转 DICOM 方向矩阵后最终三轴符号仍与患者方向一致。

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_spine_anatomical_frame -v`

Expected: FAIL，因为 `src.anatomy.spine` 尚不存在。

- [ ] **Step 3: Implement the estimator**

实现：

```python
def estimate_spine_frame_with_diagnostics(
    label_masks: dict[str, np.ndarray],
    start_label: str,
    end_label: str,
    image: sitk.Image,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]: ...
```

计算物理质心、首尾三维直线、与 DICOM 头侧对齐的 `spine_axis`、投影正交后的左右轴、叉积前后轴、全部有效 Mask 的物理中心与投影边界。成功时 `method="spine_endpoint_centroids"`、`applied=True`；诊断信息记录首尾质心、原始轴、最终轴、参与及缺失标签。

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_spine_anatomical_frame tests.test_anatomical_geometry_models -v`

Expected: 全部通过。

---

### Task 6: 胸椎横状位恢复默认水平横切

**Files:**
- Modify: `src/planning/whole_range.py`
- Test: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Consumes: 胸椎 `correction_scope_bbox`、输入图像方向和状位配置。
- Produces: 胸椎横状位默认 DICOM 横切计划；胸椎矢状位、冠状位继续使用解剖轴。

- [ ] **Step 1: Add a failing thoracic axial test**

使用倾斜胸椎解剖轴，断言横状位仍采用输入图像默认横切法向和头到脚 bbox
范围，同时矢状位继续采用胸椎解剖法向。

- [ ] **Step 2: Run focused test and confirm RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_thoracic_axial_uses_default_image_axis_but_sagittal_keeps_spine_axis -v`

Expected: FAIL，因为当前胸椎横状位仍读取解剖坐标。

- [ ] **Step 3: Bypass anatomical frame only for thoracic axial**

在整切入口中仅对“胸椎 + 横状位”使用既有 bbox 默认直线构造；其他胸椎
状位继续进入解剖坐标构造器。不修改阶段三和阶段五。

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
.venv/bin/python -m unittest tests.test_stage4_slice_planning -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: 全部通过。

---

### Task 2: 阶段三接入颈椎和胸椎

**Files:**
- Modify: `src/stages/stage_localization.py`
- Test: `tests/test_stage3_target_axial_correction.py`

**Interfaces:**
- Consumes: `resolved_cut_config["whole_range"]`、`label_targets` 和分割数组。
- Produces: 颈椎、胸椎 `anatomical_geometry` 及 `anatomical_geometry_diagnostics`。

- [ ] **Step 1: Write failing integration tests**

新增颈椎和胸椎测试，使用多个标签 Mask，断言阶段三分别使用配置中的 C1/C7、C7/L1 并输出 `method="spine_endpoint_centroids"`。断言肺部仍走 `lung_dicom_axes`，头骨仍不输出解剖坐标。

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_stage3_target_axial_correction -v`

Expected: FAIL，颈椎和胸椎当前返回 `None`。

- [ ] **Step 3: Integrate spine geometry**

扩展阶段三解剖坐标分发：肺部继续传整体 Mask；颈椎和胸椎根据配置顺序构造标准椎体名称到布尔 Mask 的映射，并调用脊柱计算器；其他部位返回空。日志改为“部位解剖坐标”，不再写死肺部。

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage3_target_axial_correction tests.test_spine_anatomical_frame -v`

Expected: 全部通过。

---

### Task 3: 阶段四三状位倾斜直线计划

**Files:**
- Modify: `src/planning/whole_range.py`
- Modify: `src/planning/straight.py`（仅在通用映射需要修正时修改）
- Test: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Consumes: 颈椎、胸椎 `anatomical_geometry.global_frame`。
- Produces: 既有 `SlicePlan`，不新增字段。

- [ ] **Step 1: Write failing planning tests**

为颈椎和胸椎提供具有 x、y、z 倾斜分量的 `AnatomicalFrame`，断言横状位 `normal` 等于脊柱轴，矢状位和冠状位平面包含脊柱轴。断言每个 plan 内法向一致、连续中心距离等于请求层间距、横状位按头侧到足侧排列、首尾中心按层厚内缩。肺部继续标准方向，未适配部位继续 bbox 默认方向。

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning -v`

Expected: FAIL，因为当前仅肺部读取 `anatomical_geometry`。

- [ ] **Step 3: Enable adapted whole-range parts**

把解剖坐标入口从肺部单一判断扩展为明确集合 `{肺部, 颈椎, 胸椎}`。复用 `anatomical_straight_frame` 和解剖坐标整体范围构造器；修正横状位范围方向，使正向头脚轴下从最大投影向最小投影展开，同时保持矢状、冠状显示符号。

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning tests.test_stage3_target_axial_correction -v`

Expected: 全部通过。

---

### Task 4: 定位图、正式 DICOM 与完整回归验证

**Files:**
- Modify: `tests/test_stage5_reconstruction.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: 阶段四生成的倾斜 `SlicePlan`。
- Produces: 与计划一致的 MPR、`Localizer.dcm` 和 DICOM 几何标签。

- [ ] **Step 1: Add stage-5 regression tests**

构造带完整三维法向的计划，断言 `PlaneStackResampler` 输出方向等于 `row_dir/col_dir`，`Localizer.dcm` 交线反映真实角度，并验证 `0020|0037` 与计划方向一致。测试只能扩展通用阶段五能力，不能在生产代码中新增部位判断。

- [ ] **Step 2: Run stage-5 tests**

Run: `.venv/bin/python -m unittest tests.test_stage5_reconstruction -v`

Expected: 现有通用实现通过；若暴露几何错误，只修复通用坐标转换。

- [ ] **Step 3: Update documentation**

README 明确颈椎和胸椎使用首尾椎体质心确定三维直线方向，横状位垂直、矢状位和临时冠状位平行；冠状位 CPR 尚未实现。

- [ ] **Step 4: Run complete verification**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: 所有测试通过，编译成功，无空白错误。

---

### Task 5: 矢状位画面按原图方向摆正

**Files:**
- Modify: `src/planning/straight.py`
- Modify: `src/planning/whole_range.py`
- Test: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Consumes: 阶段三脊柱坐标、输入图像 DICOM 方向和既有矢状位法向。
- Produces: 切割平面不变、但上下方向尽量贴近原图头脚轴的矢状位 `SlicePlan`。

- [ ] **Step 1: Add a failing sagittal display-direction test**

构造带前后倾斜的脊柱轴，断言新矢状位法向仍等于原计划法向，同时 `col_dir`
等于原图足侧方向在矢状平面内的单位投影，且三轴保持正交右手系。

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_cervical_sagittal_keeps_plane_but_uses_original_image_up_direction -v`

Expected: FAIL，因为当前矢状位 `col_dir` 直接使用负脊柱轴。

- [ ] **Step 3: Implement the minimum frame change**

扩展 `anatomical_straight_frame` 接收输入图像信息。仅在矢状位中，把负 DICOM
头脚轴正交投影到既有切片平面作为 `col_dir`，再计算与法向组成右手系的
`row_dir`；横状位、冠状位及 `normal` 不变。

- [ ] **Step 4: Verify focused and full regression tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_stage4_slice_planning tests.test_stage5_reconstruction -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: 全部通过。
