# 阶段五直线 MPR 重建实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让阶段五读取阶段四规范化的 `mapping.plans[]`，执行 `plane_stack` 直线 MPR、厚层采样、窗宽窗位输出和 DICOM 归档。

**Architecture:** 阶段五只消费 `SlicePlan[]`，不按部位或切割方式重新规划。每个 plan 的每个 `PlannedSlice` 先被重采样为 HU 图像，再按每个 `window_presets` 烘焙成固定 uint16 灰度并写出独立 DICOM series；未来 `curved_stack` 只保留几何类型入口，不在本计划实现。

**Tech Stack:** Python 3.10/3.11、SimpleITK/GDCM、NumPy、unittest/pytest。

## Global Constraints

- 阶段五切片数量严格等于 `sum(len(plan["slices"]) for plan in plans)`，不重新根据层间距计算数量。
- `slice_spacing` 只来自阶段四中心点；`slice_thickness` 用于厚层采样，不改变切片数量。
- 首版只支持 `geometry_type == "plane_stack"`；其它类型必须明确报错并带 `plan_id`。
- 窗宽窗位通过 `src.models.get_window_preset` 查询并烘焙进输出像素，阶段四只传递名称。
- 使用 SimpleITK/GDCM 写 DICOM，不新增 pydicom 依赖。
- 不改曲面 planner，不实现 `curved_stack` 采样。
- 不运行 `git commit`，完成后保留工作区修改。

---

### Task 1: 把阶段四计划接入第五步并建立执行结果契约

**Files:**
- Modify: `src/stages/stage_reconstruction.py`
- Test: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: `context.get_result("mapping")["plans"]`，以及定位结果中的 `raw_for_reconstruction_path`。
- Produces: `reconstruction` 结果中的 `plan_count`、`slice_count`、`image_count`、`planes` 和每个 plan/window 的输出目录。

- [x] **Step 1: Write the failing tests**

在 `tests/test_stage5_reconstruction.py` 创建最小的合成 raw 图像和一个 `plane_stack` plan，验证：

```python
def test_reconstruction_consumes_mapping_plans_and_reports_physical_slice_count(self):
    context = make_context_with_one_plan(slice_count=2, window_presets=["骨窗"])

    ReconstructionStage(config={"parallel": False}).execute(context)

    result = context.get_result("reconstruction")
    self.assertEqual(result["plan_count"], 1)
    self.assertEqual(result["slice_count"], 2)
    self.assertEqual(result["image_count"], 2)
    self.assertEqual(result["planes"]["横状位"]["slice_count"], 2)

def test_reconstruction_rejects_unsupported_geometry_type_with_plan_id(self):
    context = make_context_with_one_plan(geometry_type="curved_stack")

    with self.assertRaisesRegex(ValueError, "plan_id=.*geometry_type"):
        ReconstructionStage(config={"parallel": False}).execute(context)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_stage5_reconstruction.py`

Expected: FAIL because the current stage reads `axial_params/sagittal_params/coronal_params`, does not consume `mapping.plans`, and returns placeholder counts.

- [x] **Step 3: Implement the plan adapter and result counters**

在 `execute()` 中改为：

```python
plans = map_result.get("plans", [])
if not isinstance(plans, list):
    raise ValueError("阶段四输出 plans 必须是列表")

for plan in plans:
    self._validate_plan(plan)

slice_count = sum(len(plan["slices"]) for plan in plans)
```

按 `plan["plane"]` 聚合结果，保留 `axial/sagittal/coronal` 兼容键，但不再向 `_process_axial()` 等三个业务函数传空参数。验证器至少检查 `plan_id`、`geometry_type`、`slices`、`extent_mm`、`output_spacing_mm`、`slice_thickness`、`slice_spacing` 和 `window_presets`。

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_stage5_reconstruction.py`

Expected: PASS for plan consumption, counters and unsupported geometry error; actual pixel sampling tests remain pending.

---

### Task 2: 实现 `plane_stack` 的单张平面重采样

**Files:**
- Create: `src/reconstruction/__init__.py`
- Create: `src/reconstruction/plane_stack.py`
- Modify: `src/stages/stage_reconstruction.py`
- Test: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: 3D `sitk.Image`、`SlicePlan` 字典和一张 `slices[]` 字典。
- Produces: 保留 HU 的单层 3D `sitk.Image`（size=(cols, rows, 1)），其像素间距和方向由 plan/slice 决定。

- [x] **Step 1: Write the failing tests**

使用一个值为 `x + 10*y + 100*z` 的合成图像，验证横状位中心点采样值和输出尺寸：

```python
def test_plane_stack_resampler_uses_slice_center_and_in_plane_extent(self):
    image = make_ramp_image(size=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    plan = make_plan(
        extent_mm={"width": 4.0, "height": 4.0},
        output_spacing_mm={"x": 1.0, "y": 1.0, "z": 0.0},
    )

    output = PlaneStackResampler().sample_slice(image, plan, plan["slices"][0])

    self.assertEqual(output.GetSize(), (5, 5, 1))
    self.assertEqual(output.GetSpacing(), (1.0, 1.0, 1.0))
    self.assertAlmostEqual(float(sitk.GetArrayFromImage(output)[0, 0, 0]), expected_value)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_stage5_reconstruction.py -k plane_stack_resampler`

Expected: FAIL because `PlaneStackResampler` does not exist.

- [x] **Step 3: Implement the resampler**

在 `src/reconstruction/plane_stack.py` 定义：

```python
class PlaneStackResampler:
    def sample_slice(
        self,
        image: sitk.Image,
        plan: dict,
        planned_slice: dict,
    ) -> sitk.Image:
        ...
```

实现规则：

1. `cols = round(extent.width / spacing.x) + 1`，`rows = round(extent.height / spacing.y) + 1`。
2. 输出网格中心对应 `planned_slice["center"]`。
3. 输出网格两个方向使用 `row_dir` 和 `col_dir`，第三方向使用 `normal`。
4. 使用 `sitk.ResampleImageFilter` 的线性插值从输入 3D 图像采样一个 2D 厚度为 1 的平面。
5. 输出图像保持输入 HU 数值，不在该模块应用窗宽窗位。

- [x] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_stage5_reconstruction.py -k plane_stack_resampler`

Expected: PASS，且输出尺寸、spacing、方向和中心采样值符合合成图像计算结果。

---

### Task 3: 加入层厚 slab 平均，不改变第四步的切片数量

**Files:**
- Modify: `src/reconstruction/plane_stack.py`
- Modify: `src/stages/stage_reconstruction.py`
- Test: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: `plan["slice_thickness"]` 和输入图像 spacing。
- Produces: 一张中心位置不变、像素为厚层采样平均值的 HU 图像。

- [x] **Step 1: Write the failing test**

构造 z 方向值为 `[0, 10, 20]`、原始 z spacing 为 1mm 的图像，使用层厚 3mm，验证输出中心像素为 10；同时验证层间距 5mm 的两个中心不会被补成更多切片：

```python
def test_slab_thickness_averages_source_planes_without_adding_slices(self):
    image = make_z_ramp_image(values=[0, 10, 20], spacing_z=1.0)
    plan = make_plan(slice_thickness=3.0, slice_spacing=5.0, slice_count=2)

    result = ReconstructionStage(config={"parallel": False}).execute(
        make_context(image=image, plan=plan)
    )

    self.assertEqual(result.get_result("reconstruction")["slice_count"], 2)
    self.assertEqual(read_written_pixel(result, 0), 10)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_stage5_reconstruction.py -k slab_thickness`

Expected: FAIL because the placeholder stage writes no sampled output and no slab averaging exists.

- [x] **Step 3: Implement centered slab sampling**

在 `PlaneStackResampler.sample_slice()` 中根据最小输入 spacing 计算采样层数：

```python
sample_count = max(1, math.ceil(slice_thickness / min(input_spacing_xyz)))
sample_step = slice_thickness / sample_count
offsets = [
    -slice_thickness / 2 + sample_step / 2 + index * sample_step
    for index in range(sample_count)
]
```

对每个 offset 沿 `normal` 采样一个平面，使用 NumPy 对平面数组求平均，再转换回 HU 图像。`slice_spacing` 只来自已经存在的 `slices[]`，不在这里生成新中心点。

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_stage5_reconstruction.py -k "slab_thickness or plane_stack_resampler"`

Expected: PASS，厚层值正确，物理切片数量仍等于第四步 slices 数量。

---

### Task 4: 实现窗宽窗位分系列输出和目录结构

**Files:**
- Create: `src/reconstruction/windowing.py`
- Modify: `src/stages/stage_reconstruction.py`
- Test: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: `plan["window_presets"]` 中的中文名称。
- Produces: 每个唯一窗宽窗位一个输出目录，并复用同一张未窗化 HU 图像，避免重复重采样。

- [x] **Step 1: Write the failing tests**

```python
def test_reconstruction_writes_one_series_per_window_preset(self):
    context = make_context_with_one_plan(slice_count=2, window_presets=["骨窗", "脑窗"])

    ReconstructionStage(config={"parallel": False}).execute(context)

    root = Path(context.request.output_dir)
    self.assertEqual(len(list((root / "横状位" / "骨窗").rglob("*.dcm"))), 2)
    self.assertEqual(len(list((root / "横状位" / "脑窗").rglob("*.dcm"))), 2)
    self.assertEqual(context.get_result("reconstruction")["image_count"], 4)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_stage5_reconstruction.py -k window_preset`

Expected: FAIL because no DICOM output directories are currently generated.

- [x] **Step 3: Implement window processing and output paths**

新增 `unique_window_presets(names)` 和 `apply_window_to_image(image, preset)`，通过 `get_window_preset()` 查询固定窗宽窗位并按请求顺序去重，再把 HU 映射成固定 uint16 灰度。DICOM 的 Window Center/Width 使用全灰度范围，避免查看器对已经烘焙的像素再次套用原始窗口；选用的窗口名称、窗宽和窗位写入 SeriesDescription。目录使用：

```text
{output_dir}/{plane}/{window}/{group_name}/切片_0001.dcm
```

同一张 HU 平面只采样一次，然后循环写入多个 window series。重复的 window 名称去重并保持请求顺序。

- [x] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_stage5_reconstruction.py -k window_preset`

Expected: PASS，两个窗宽窗位目录各有完整切片序列，物理 slice_count 不乘窗位数量。

---

### Task 5: 写出带空间信息的 DICOM 并完善全量验证

**Files:**
- Create: `src/reconstruction/dicom.py`
- Modify: `src/stages/stage_reconstruction.py`
- Modify: `src/stages/stage_reporting.py` only if report needs actual reconstruction paths
- Test: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: HU 2D `sitk.Image`、plan、planned slice、window preset、输出路径。
- Produces: 可由 SimpleITK 读取的 CT DICOM 文件和完整重建结果统计。

- [x] **Step 1: Write the failing metadata tests**

```python
def test_written_dicom_contains_slice_geometry_and_window_tags(self):
    context = make_context_with_one_plan(slice_count=1, window_presets=["骨窗"])

    ReconstructionStage(config={"parallel": False}).execute(context)

    dcm_path = next(Path(context.request.output_dir).rglob("*.dcm"))
    image = sitk.ReadImage(str(dcm_path))
    self.assertEqual(image.GetMetaData("0008|0060"), "CT")
    self.assertEqual(image.GetMetaData("0018|0050"), "3.0")
    self.assertEqual(image.GetMetaData("0028|1051"), "2500")
    self.assertIn("\\", image.GetMetaData("0020|0037"))
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_stage5_reconstruction.py -k dicom`

Expected: FAIL because the current placeholder does not write DICOM metadata.

- [x] **Step 3: Implement the DICOM writer**

用 SimpleITK/GDCM 写出每张 2D 图像，并设置：

- `Modality=CT`、`ImageType=DERIVED\\SECONDARY`；
- 每个 plan/window 固定一个 SeriesInstanceUID，每张图唯一 SOPInstanceUID；
- `InstanceNumber` 使用 `planned_slice.index`；
- `PixelSpacing` 使用输出平面两个方向的 spacing；
- `SliceThickness` 使用 plan 层厚；
- `SpacingBetweenSlices` 使用 plan 层间距；
- `ImagePositionPatient` 使用首像素物理坐标；
- `ImageOrientationPatient` 使用 row_dir 和 col_dir；
- 像素已经烘焙成固定灰度，因此 `WindowCenter/WindowWidth` 使用全灰度范围；预设名称、窗宽和窗位写入 SeriesDescription；
- 缺少源患者/检查 UID 时生成合法 UID，并在 SeriesDescription 中写入状位、窗位和 group。

- [x] **Step 4: Run targeted and full tests**

Run: `pytest -q tests/test_stage5_reconstruction.py`

Expected: PASS。

Run: `pytest -q`

Expected: 原有测试和新增第五步测试全部通过。

- [x] **Step 5: Run static checks**

Run: `git diff --check`

Expected: no output and exit code 0。完成后不提交 git。

## Self-review checklist

- `mapping.plans[]` 是第五步唯一规划输入，没有回退到 `axial_params` 等旧占位字段。
- 第五步没有 body_part/cut_mode 分支；这些字段只用于目录、日志和元数据。
- `plane_stack` 的每张输出图都来自一个已有 `slices[]` 项。
- 厚层采样不会添加或删除切片中心。
- 多窗宽窗位只增加输出 series，不改变物理切片数量。
- `curved_stack` 仍然只做显式不支持校验，不会被误当成直线采样。
