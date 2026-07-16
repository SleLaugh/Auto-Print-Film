# 髋骨三状位直线 MPR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现髋骨外侧顶部 AB 倾斜横切、默认矢冠切片和明确的定位图参考状位。

**Architecture:** 阶段三新增髋骨专属解剖计算器并输出统一坐标；阶段四按部位和
状位选择倾斜或默认 frame，同时写入通用定位图字段；阶段五只消费计划字段。

**Tech Stack:** Python、NumPy、SimpleITK、unittest。

## Global Constraints

- 只有髋骨横状位增加角度。
- 矢状位和冠状位保持默认 DICOM 方向。
- 第五步不得新增髋骨业务判断。

---

### Task 1: 髋骨外侧顶部坐标计算器

**Files:**
- Create: `src/anatomy/hip.py`
- Modify: `src/anatomy/__init__.py`
- Test: `tests/test_hip_anatomical_frame.py`

- [x] 写失败测试：左右顶部不等高、孤立毛刺、缺失 Mask 和翻转方向。
- [x] 运行测试确认失败。
- [x] 实现主体过滤、顶部 5 mm 外侧中心、AB 横切坐标和诊断。
- [x] 运行测试确认通过。

### Task 2: 阶段三与第四步接入

**Files:**
- Modify: `src/stages/stage_localization.py`
- Modify: `src/planning/whole_range.py`
- Modify: `src/models/reconstruction.py`
- Test: `tests/test_stage3_target_axial_correction.py`
- Test: `tests/test_stage4_slice_planning.py`

- [x] 写失败测试：阶段三髋骨分发；横状位使用 AB；矢冠方向不变。
- [x] 实现最小分发和三状位隔离。
- [x] 运行阶段三、四测试。

### Task 3: 通用定位图参考状位

**Files:**
- Modify: `src/stages/stage_reconstruction.py`
- Test: `tests/test_stage5_reconstruction.py`
- Modify: `README.md`

- [x] 写失败测试：计划显式选择冠状或矢状定位背景。
- [x] 通用消费 `localizer_plane`，无字段时保留旧回退。
- [x] 更新能力说明并运行全量测试、compileall 和 diff check。
