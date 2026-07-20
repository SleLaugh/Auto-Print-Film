# 阶段五 GDCM 写盘串行保护 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留窗位灰度计算并行，同时禁止 SimpleITK/GDCM DICOM 写入器在同一进程内并发执行。

**Architecture:** 在 `src/reconstruction/dicom.py` 增加唯一的 GDCM 写盘入口和进程内互斥锁，锁住写入器从创建到销毁的完整生命周期；CT 成片和 Localizer 统一调用该入口。阶段五线程池、MPR 采样、pydicom 标签修复和目录结构保持不变。

**Tech Stack:** Python、threading、SimpleITK/GDCM、pydicom、unittest。

## Global Constraints

- 不修改髋骨解剖坐标、阶段四计划或 MPR 几何。
- 不关闭窗位线程池；仅串行保护 GDCM 原生写盘。
- CT 成片和 Localizer 必须共享同一把锁。
- 写盘异常、临时文件清理和原子替换行为保持不变。
- 所有新增方法和锁必须补充中文注释。
- 本轮不执行 Git 提交。

---

### Task 1: 锁定 GDCM 写盘并发契约

**Files:**
- Modify: `tests/test_stage5_reconstruction.py`
- Modify: `src/reconstruction/dicom.py`

**Interfaces:**
- Consumes: `write_ct_dicom(...)`、`write_localizer_dicom(...)` 已准备好的 SimpleITK 图像与输出路径。
- Produces: `_execute_gdcm_write(image: sitk.Image, output_path: Path) -> None`，保证同一进程最多一个 GDCM 写入器存活。

- [x] **Step 1: Write the failing concurrency test**

在四个线程开始调用 `write_ct_dicom` 前使用 `threading.Barrier(4)` 对齐时序；替换
`sitk.ImageFileWriter` 为记录活动数量并短暂等待的测试写入器。断言四个输出存在且
`max_active_writers == 1`。

- [x] **Step 2: Run the test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_stage5_reconstruction.Stage5ReconstructionTest.test_gdcm_writer_lifecycle_is_serialized_across_window_threads -v`

Expected: FAIL，当前四个线程可同时进入测试写入器的 `Execute`，最大活动数大于 1。

- [x] **Step 3: Implement the minimal serialized writer**

在 `src/reconstruction/dicom.py` 定义模块级 `threading.Lock`，并增加统一写入方法：

```python
def _execute_gdcm_write(image: sitk.Image, output_path: Path) -> None:
    with _GDCM_WRITE_LOCK:
        writer = sitk.ImageFileWriter()
        try:
            writer.SetFileName(str(output_path))
            writer.KeepOriginalImageUIDOn()
            writer.Execute(image)
        finally:
            del writer
```

`write_ct_dicom` 和 `write_localizer_dicom` 均改为调用该方法，不能在其他位置直接
创建 `ImageFileWriter`。

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage5_reconstruction -v`

Expected: 新增并发契约测试和既有阶段五测试全部通过。

### Task 2: 真实阶段五与全量验证

**Files:**
- Verify only: 当前任务已有的 `raw_0000.nii.gz` 和 `slice_plans.json`；若已被新任务清空，则使用内存合成的 41 张 512 级矩阵计划。
- Verify output: 项目所在 `fuseblk` 文件系统内的自动清理临时目录。

**Interfaces:**
- Consumes: 已生成的髋骨原图和阶段四计划，或等规模内存图像，不调用 nnUNet。
- Produces: 临时目录中的四窗位 DICOM，用于验证原生写盘和清理行为。

- [x] **Step 1: Run the real four-window Stage 5 probe**

执行 41 张横状位、4 个窗位的原生写盘，要求进程退出码为 0；每个窗位均生成
41 张编号成片和一个 `Localizer.dcm`，168 个文件均可由 pydicom 读取，目录中
不存在 `.tmp.dcm`。

- [x] **Step 2: Run full verification**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: 全量测试通过，编译和差异检查退出码均为 0；不产生 Git 提交。
