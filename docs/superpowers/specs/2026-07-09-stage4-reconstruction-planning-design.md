# 阶段四切割方案规划设计

## 背景

当前流水线已经有六个阶段：

1. 阶段一：DICOM 转 NIfTI。
2. 阶段二：nnUNet 分割。
3. 阶段三：根据 `body_part + cut_mode` 定位目标 label，输出每个目标的
   bbox、centroid、整体目标 bbox，以及调试 overlay。
4. 阶段四：当前仍是硬编码占位。
5. 阶段五：当前只创建三平面输出目录，MPR 和 DICOM 生成仍是占位。
6. 阶段六：生成报告。

阶段三已经回答了“目标在哪里”：例如 L1、L2、L3、L4、L5、S1 的坐标、
包围盒和整体范围。但它还没有回答“应该怎么切”：在哪个状位上画线，
从哪里切到哪里，是否沿曲线切，每隔多远切一张，每张切片多厚，每张
图像显示多大范围。

阶段四要成为后续重建的核心临床规则层。它只负责根据定位结果和请求参数
生成切割方案，不直接读取体数据做切片，也不直接写 DICOM。

## 核心判断

后半段流水线中，最容易变化、也最有业务含义的是阶段四。

- 阶段三负责定位目标。
- 阶段四负责规划切法。
- 阶段五负责按切割方案采样 3D volume。
- 阶段六负责报告、目录和 DICOM 等交付整理。

因此后续如果调整腰椎横断位、颈椎整切、鼻窦冠状位或眼眶矢状位的切法，
原则上应优先修改阶段四的 planner。阶段五和阶段六不应该反复写部位判断。

## 目标

阶段四输入第三步的定位结果和请求里的三平面配置，输出统一的
`ReconstructionSpec[]`。

每个 `ReconstructionSpec` 可以理解成一张“切片施工单”。它告诉第五步：

- 要切哪个状位。
- 这一组切片叫什么。
- 是普通平面切，还是沿曲线/曲面切。
- 切割线或曲线在哪里。
- 每张切片中心在哪里。
- 每张切片的方向怎么摆。
- 每张图像显示多大范围。
- 层厚、层间距、窗宽窗位是什么。

第五步只读取这张施工单并执行采样，不关心它来自腰椎、颈椎、肺部、鼻窦
还是眼眶。

## 非目标

本设计不实现以下内容：

- 不实现 MPR 采样算法。
- 不实现 DICOM 写出。
- 不实现自动回正。
- 不重新设计阶段三定位逻辑。
- 不在第一版覆盖眼眶、鼻窦、颞骨等需要更多关键结构 label 的复杂规则。

这些能力后续可以基于同一个 `ReconstructionSpec` 接口继续扩展。

## 阶段边界

### 阶段三：定位

阶段三输出的是“原料坐标”，即目标结构在 3D 体数据中的位置。

典型输出：

```text
coords
correction_scope_bbox
debug_target_mask_overlay_path
debug_label_mask_overlay_path
raw_for_reconstruction_path
seg_for_reconstruction_path
image.size / image.spacing / image.origin / image.direction
```

这些信息回答：

```text
L1 在哪里？
L2 在哪里？
整体腰椎范围在哪里？
目标 mask 是哪些体素？
后续重建应该读取哪张 raw 图像？
```

阶段三不回答：

```text
横状位应该沿哪条线切？
矢状位是否要做曲面重建？
L4-L5 区间切从哪里开始到哪里结束？
每张切片中心点在哪里？
```

### 阶段四：规划

阶段四把“目标坐标”翻译成“切割方案”。

它回答：

```text
这次要输出哪些状位？
每个状位有几组切片？
每组切片对应哪条直线或曲线？
根据层间距应该生成哪些切片中心？
每张切片怎么摆放？
每张图像需要显示多大范围？
```

阶段四不读取 raw 图像做插值采样，也不写 DICOM。

### 阶段五：重建

阶段五只执行切割方案。

它回答：

```text
给定 ReconstructionSpec，如何从 3D volume 采样出 2D 切片？
普通平面堆叠如何采样？
曲面堆叠如何采样？
层厚如何合成？
窗宽窗位如何应用到输出图像？
```

阶段五不应该根据 `body_part` 写分支，也不应该知道“腰椎整切”或“鼻窦冠状位”
的业务规则。

### 阶段六：交付

阶段六整理第五步产物。

它回答：

```text
报告怎么写？
输出目录怎么组织？
DICOM 文件如何按状位和窗宽窗位归档？
```

阶段六也不应该重新计算切割线。

## 坐标约定

阶段三当前 bbox 和 centroid 使用体素数组习惯，即 `z, y, x`。SimpleITK 图像
物理空间使用 `x, y, z`，单位通常是 mm。

阶段四内部应同时支持两种坐标：

- `voxel_zyx`：用于和阶段三结果、调试 overlay 对照。
- `physical_xyz`：用于规划、重建采样和后续 DICOM 空间信息。

第一版 `ReconstructionSpec` 的主坐标建议使用 `physical_xyz`。原因：

- 层厚和层间距是 mm 单位。
- 图像 spacing 在不同轴上可能不一致。
- 后续 DICOM 需要 ImagePositionPatient 和 ImageOrientationPatient。
- 斜切和曲面切都更适合在物理坐标中表达。

为了调试方便，spec 可以保留 `debug_voxel_zyx` 字段，但下游执行以
`physical_xyz` 为准。

## 统一输出结构

阶段四统一输出 `ReconstructionSpec[]`。每个 spec 代表“一组切片”。

### `spec_id`

唯一标识这一组切片。用于日志、报告和输出目录。

示例：

```text
腰椎_whole_range_横状位
腰椎_L4-L5_横状位
腰椎_whole_range_矢状位_curved
肺部_whole_range_横状位
```

### `body_part`

本次请求的部位。

示例：

```text
腰椎
颈椎
胸椎
肺部
头骨
```

### `cut_mode`

切割方式。

当前已有两类：

```text
whole_range
interspace
```

业务含义：

- `whole_range`：整切，覆盖一个完整目标范围。
- `interspace`：区间切，通常用于腰椎等相邻椎体/椎间隙。

### `plane`

输出状位。

当前使用已有枚举：

```text
横状位
矢状位
冠状位
```

### `group_name`

这一组切片的业务名称。

整切示例：

```text
whole_range
```

区间切示例：

```text
L1-L2
L2-L3
L3-L4
L4-L5
L5-S1
```

未来眼眶示例：

```text
left_eye
right_eye
```

### `geometry_type`

切割几何类型。第五步根据这个字段选择通用执行器。

第一版建议至少支持：

```text
plane_stack
curved_stack
```

含义：

- `plane_stack`：普通平面堆叠。可以是正交切，也可以是斜切。
- `curved_stack`：沿曲线或曲面展开，类似截图中的腰椎曲面切割。

### `path_points`

切割线或曲线上的点，使用物理坐标 `x, y, z`，单位 mm。

这是第四步最重要的结果之一，可以理解成“医生在定位像上画出来的线”。

直线至少两个点：

```json
[
  {"x": 10.0, "y": 20.0, "z": 30.0},
  {"x": 10.0, "y": 20.0, "z": 120.0}
]
```

曲线使用多个点：

```json
[
  {"x": 10.0, "y": 20.0, "z": 30.0},
  {"x": 11.5, "y": 21.0, "z": 45.0},
  {"x": 13.0, "y": 22.0, "z": 60.0}
]
```

普通平面切也可以使用 `path_points` 表达切片中心线。这样第五步不用关心这条线
是直线还是由部位规则生成的曲线。

### `slice_centers`

每张切片的中心点，使用物理坐标 `x, y, z`，单位 mm。

阶段四根据 `path_points` 和 `slice_spacing` 生成这些中心点。第五步逐张读取
这些中心点采样。

例如线长 100mm，层间距 5mm，则大约生成 21 个中心点。是否包含首尾点由
planner 明确决定。

### `orientations`

每张切片的方向信息。

每一项对应一个 `slice_center`，包括：

```text
normal
row_dir
col_dir
```

含义：

- `normal`：这一摞片子推进方向，也就是层间距方向。
- `row_dir`：输出 2D 图像从左到右的方向。
- `col_dir`：输出 2D 图像从上到下的方向。

普通平面堆叠里，每张切片的方向通常一致。曲面堆叠里，每个中心点附近的方向
可能不同。

方向向量必须是单位向量，并且尽量满足：

```text
row_dir ⟂ col_dir
row_dir ⟂ normal
col_dir ⟂ normal
```

如果无法稳定计算方向，planner 应输出 warning，而不是给出看似正常但不可信的
方向。

### `extent_mm`

每张输出图像需要覆盖的宽高范围，单位 mm。

示例：

```json
{
  "width": 160.0,
  "height": 160.0
}
```

业务含义：

- 横状位腰椎：应覆盖完整椎体及周围必要软组织。
- 矢状位腰椎：应覆盖 T12/L1 至 S1 的上下范围，以及左右方向足够范围。
- 肺部横断位：通常覆盖原始图像前后左右范围。

第一版可以从整体 bbox 加 margin 推导。后续不同部位可以单独调整 margin。

### `output_spacing_mm`

输出 2D 图像的像素间距，单位 mm。

第一版可以沿用原图较小的 in-plane spacing，或者由 planner 给定默认值。这个字段
和层间距不同：

- `slice_spacing`：相邻切片之间的距离。
- `output_spacing_mm`：一张 2D 图像内部每个像素之间的距离。

如果第一版暂不实现真实 MPR，可先写入规划值，供第五步后续使用。

### `slice_thickness`

层厚，来自请求中的 `PlaneConfig.slice_thickness`。

第五步使用它决定厚层采样方式。第一版如果不做厚层平均，也应把它写入 spec 和
报告，避免参数丢失。

### `slice_spacing`

层间距，来自请求中的 `PlaneConfig.slice_spacing`。

阶段四使用它生成 `slice_centers`。

### `window_presets`

窗宽窗位预设，来自请求中的 `PlaneConfig.window_presets`。

示例：

```text
骨窗
肺窗
纵隔窗
```

阶段四只传递这个字段，不负责应用窗宽窗位。

### `source_labels`

说明这一组切割方案基于哪些定位目标生成。

示例：

```text
vertebrae_L1
vertebrae_L2
vertebrae_L3
vertebrae_L4
vertebrae_L5
vertebrae_S1
```

如果某个 planner 使用整体 bbox 而不是单个 label，也应写明：

```text
correction_scope_bbox
```

这个字段用于调试和报告，方便判断切割线为什么长成现在这样。

### `confidence`

规划可信度，范围建议为 `0.0` 到 `1.0`。

第一版可以采用简单规则：

- 所需 label 全部找到：`1.0`
- 允许缺少非关键 label 并可插值补齐：`0.7`
- 关键 label 缺失但仍使用 bbox 兜底：`0.4`
- 无法生成可靠方案：不输出 spec 或输出 error。

### `warnings`

规划警告列表。

示例：

```text
缺少 vertebrae_S1，使用 sacrum 兜底
L3-L4 区间无法生成，因为 L4 未找到
曲线点数量不足，降级为普通平面切
```

警告不一定表示失败，但必须写入报告，便于人工排查。

## 规则组织

阶段四不要写成一个巨大的 `map_planes()`。推荐使用统一入口加多个 planner。

建议目录：

```text
src/planning/
    __init__.py
    base.py
    geometry.py
    spine_common.py
    lumbar.py
    cervical.py
    thoracic.py
    single_label.py
```

### `base.py`

定义 planner 基类和通用输入输出。

每个 planner 至少实现：

```text
build_specs(request, localization_result, plane_configs) -> list[ReconstructionSpec]
```

### `geometry.py`

只放公共几何方法，不写部位业务。

候选方法：

- 体素坐标转物理坐标。
- 物理坐标转体素坐标。
- 根据 bbox 加 margin。
- 根据点序列计算总长度。
- 按层间距对直线或曲线重采样，生成 `slice_centers`。
- 根据两个点计算方向向量。
- 根据中心线计算局部 tangent。
- 根据 `normal` 和参考方向生成 `row_dir` / `col_dir`。
- 校验方向向量是否接近正交。

### `spine_common.py`

脊柱通用规划方法，供颈椎、胸椎、腰椎复用。

候选方法：

- 从 `coords` 中按解剖顺序取出椎体中心点。
- 根据椎体中心点生成脊柱中心线。
- 根据相邻椎体中心点生成区间中点。
- 根据整体 bbox 推导横状位、矢状位、冠状位基础范围。
- 为脊柱整切生成普通平面堆叠 spec。

### `lumbar.py`

腰椎规则。

第一版建议包含：

```text
LumbarWholeRangePlanner
LumbarInterspacePlanner
```

`LumbarWholeRangePlanner`：

- 横状位：根据 L1-L5/S1 中心线生成斜横断位切片。
- 矢状位：第一版可生成普通 `plane_stack`；后续可生成 `curved_stack`，表达截图里沿腰椎曲线切割。
- 冠状位：如果请求包含冠状位，则根据整体 bbox 和中心线生成范围；如果业务暂不需要，可输出 warning 或不输出。

`LumbarInterspacePlanner`：

- 横状位：对 L1-L2、L2-L3、L3-L4、L4-L5、L5-S1 分别生成 spec。
- 每个区间以相邻椎体中心点推导局部方向和区间中心。
- 如果任一端椎体缺失，该区间不生成 spec，并写入 warning。

### `cervical.py`

颈椎规则。

第一版可以复用 `spine_common.py` 的整切逻辑，但保留独立 planner 名称：

```text
CervicalWholeRangePlanner
```

原因：

- 颈椎和腰椎都是连续椎体 label，但临床切割角度和范围可能不同。
- 独立类名便于后续单独调整，不会影响腰椎。

### `thoracic.py`

胸椎规则。

第一版可以复用 `spine_common.py` 的整切逻辑：

```text
ThoracicWholeRangePlanner
```

后续可以针对胸椎长范围、肋骨干扰、冠状位范围单独优化。

### `single_label.py`

单目标整切兜底规则。

适用于第一版中没有复杂专属 planner 的目标，例如：

```text
肺部
头骨
```

它根据目标 bbox 加 margin 生成基础平面堆叠，不做复杂曲线推导。

后续眼眶、鼻窦、颞骨如果有足够关键结构 label，应从这里拆出独立 planner。

## Planner 选择

阶段四入口根据 `body_part + cut_mode` 选择 planner。

建议注册表：

```text
(腰椎, whole_range) -> LumbarWholeRangePlanner
(腰椎, interspace) -> LumbarInterspacePlanner
(颈椎, whole_range) -> CervicalWholeRangePlanner
(胸椎, whole_range) -> ThoracicWholeRangePlanner
(肺部, whole_range) -> SingleLabelWholeRangePlanner
(头骨, whole_range) -> SingleLabelWholeRangePlanner
```

如果没有匹配 planner：

- 第一版应明确报错。
- 不要静默输出假参数。
- 报错信息需要包含 `body_part` 和 `cut_mode`。

## 状位处理

每个 planner 内部再根据 `request.plane_configs` 决定输出哪些状位。

例如：

```text
for plane_config in request.plane_configs:
    if plane_config.plane == 横状位:
        build_axial_specs(plane_config)
    elif plane_config.plane == 矢状位:
        build_sagittal_specs(plane_config)
    elif plane_config.plane == 冠状位:
        build_coronal_specs(plane_config)
```

这样可以支持：

- 同一个部位不同状位走不同切法。
- 同一个状位使用不同层厚、层间距和窗宽窗位。
- 某些部位暂不支持某个状位时输出 warning 或错误。

第一版建议：

- 请求的状位能够生成时，输出 spec。
- 请求的状位暂不支持时，不生成 spec，并在阶段四结果中写 warning。
- 如果所有请求状位都无法生成 spec，则阶段四失败。

## 普通平面切和曲面切

### `plane_stack`

`plane_stack` 是普通平面堆叠。

它可以表达：

- 标准横状位。
- 标准矢状位。
- 标准冠状位。
- 任意斜平面。
- 腰椎区间切中的一组局部斜横断位。

它的特点：

- 每张切片都是一个平面。
- 一组切片通常有相同或平滑变化的方向。
- 第五步可用统一 `PlaneStackReslicer` 执行。

### `curved_stack`

`curved_stack` 是沿曲线或曲面展开。

它可以表达截图中矢状位上的黄色曲线：

- 第四步先根据椎体中心点或目标结构拟合曲线。
- 再沿曲线生成采样点和局部方向。
- 第五步根据曲线采样网格生成曲面展开图像或曲面切片。

它的特点：

- 切割方向沿曲线变化。
- 普通 `normal` 可能每个点不同。
- 第五步需要统一 `CurvedStackReslicer` 执行。

第一版可以先定义 `curved_stack` 数据结构和 planner 输出形态，再决定是否立即实现第五步曲面采样。

## 腰椎整切示例

输入：

```text
body_part = 腰椎
cut_mode = whole_range
plane_configs = [
    横状位: slice_spacing=5, slice_thickness=5, window_presets=[骨窗],
    矢状位: slice_spacing=5, slice_thickness=5, window_presets=[骨窗]
]
```

阶段三定位结果包含：

```text
vertebrae_L1 centroid
vertebrae_L2 centroid
vertebrae_L3 centroid
vertebrae_L4 centroid
vertebrae_L5 centroid
vertebrae_S1 centroid
correction_scope_bbox
```

阶段四输出：

```text
spec: 腰椎_whole_range_横状位
    geometry_type = plane_stack
    path_points = 根据 L1-L5/S1 中心点生成的脊柱中心线
    slice_centers = 按 5mm 层间距沿中心线生成
    orientations = 根据中心线局部方向生成斜横断位方向
    extent_mm = 根据整体 bbox 加 margin
    slice_thickness = 5
    slice_spacing = 5

spec: 腰椎_whole_range_矢状位
    geometry_type = plane_stack 或 curved_stack
    path_points = 腰椎中心线或曲线
    slice_centers = 根据矢状位范围和层间距生成
    orientations = 根据矢状位规则生成
```

## 腰椎区间切示例

输入：

```text
body_part = 腰椎
cut_mode = interspace
plane_configs = [
    横状位: slice_spacing=3, slice_thickness=3, window_presets=[骨窗]
]
```

阶段四输出多组 spec：

```text
腰椎_L1-L2_横状位
腰椎_L2-L3_横状位
腰椎_L3-L4_横状位
腰椎_L4-L5_横状位
腰椎_L5-S1_横状位
```

每组 spec：

- 使用相邻椎体中心点推导区间中心。
- 使用相邻椎体中心点方向推导局部斜切方向。
- 使用层间距生成局部若干张切片中心。
- 使用层厚写入 spec，交给第五步执行厚层采样。

如果 L5 或 S1 缺失：

- 不生成 `L5-S1` spec。
- 在阶段四结果 warnings 中记录原因。

## 阶段四结果结构

阶段四写入 `context.set_result("mapping", result)`。

建议结果：

```json
{
  "specs": [],
  "warnings": [],
  "planner": "LumbarWholeRangePlanner",
  "spec_count": 0,
  "metadata_path": "/output/metadata/reconstruction_specs.json"
}
```

同时写出：

```text
metadata/reconstruction_specs.json
```

该文件用于：

- 人工检查第四步到底规划了哪些线。
- 第五步重建失败时回溯输入。
- 报告中记录切割方案。

## 调试产物

第一版阶段四建议输出 JSON 即可。

后续可以增加：

- 切割线 overlay。
- 每组 spec 的中心点点云 overlay。
- 曲线点 overlay。
- 在 ITK-SNAP 中可加载的 mask 或 markups。

调试 overlay 是研发期辅助产物，不应作为第五步执行的必要输入。第五步只依赖
`ReconstructionSpec[]`。

## 错误处理

阶段四应区分 warning 和 error。

### warning

不影响整个阶段继续执行，但需要写入结果。

示例：

- 某个非关键 label 缺失，已使用 bbox 兜底。
- 某个请求状位暂不支持，已跳过。
- 曲线点数量不足，已降级为普通平面切。

### error

无法生成任何有效 spec，阶段四失败。

示例：

- `localization` 结果不存在。
- 所需目标坐标全部缺失。
- `plane_configs` 为空。
- `body_part + cut_mode` 没有匹配 planner。
- 请求的所有状位都无法生成切割方案。

## 测试范围

第一版测试应覆盖：

- `ReconstructionSpec.to_dict()` 能输出 JSON 友好结构。
- `MappingStage` 能根据 `腰椎 + whole_range` 选择 `LumbarWholeRangePlanner`。
- 腰椎整切能根据 L1-L5/S1 centroid 生成横状位 `plane_stack` spec。
- 腰椎区间切能生成 L1-L2、L2-L3 等多个 spec。
- 缺少某个区间端点时，该区间被跳过并记录 warning。
- `SingleLabelWholeRangePlanner` 能根据整体 bbox 生成基础整切 spec。
- 不支持的 `body_part + cut_mode` 会给出明确错误。
- 阶段四会写出 `metadata/reconstruction_specs.json`。

暂不要求第一版测试真实 MPR 采样；那属于阶段五。

## 实现顺序建议

1. 新增 `src/models/reconstruction.py`，定义 `Point3D`、`Orientation`、
   `ReconstructionSpec` 等数据结构。
2. 新增 `src/planning/geometry.py`，实现坐标转换、点序列长度、按层间距生成
   `slice_centers` 等纯几何方法。
3. 新增 `src/planning/base.py` 和 planner 注册表。
4. 实现 `SingleLabelWholeRangePlanner`，作为简单兜底。
5. 实现 `spine_common.py`，提供脊柱中心线和区间点生成方法。
6. 实现 `LumbarWholeRangePlanner`。
7. 实现 `LumbarInterspacePlanner`。
8. 修改 `stage_mapping.py`，让它调用 planner 并写出 `reconstruction_specs.json`。
9. 补充阶段四 focused tests。

## 与第五步的接口约定

第五步后续只读取：

```text
map_result["specs"]
loc_result["raw_for_reconstruction_path"]
```

然后根据：

```text
spec.geometry_type
```

选择：

```text
PlaneStackReslicer
CurvedStackReslicer
```

第五步不根据 `body_part` 或 `cut_mode` 写重建分支。

## 与第六步的接口约定

第六步报告应记录：

- `metadata/reconstruction_specs.json` 路径。
- 每个 spec 的 `spec_id`、`plane`、`group_name`、`geometry_type`。
- warnings。
- 第五步实际输出的文件路径。

第六步不重新计算切割线，也不重新解释 `coords`。

## 后续扩展

后续可以逐步增加：

- 眼眶 planner：基于眼球、视神经等 label 生成左右眼斜切方案。
- 鼻窦 planner：基于硬腭、鼻甲、鼻腔等结构生成横状位/冠状位/矢状位方案。
- 颞骨 planner：基于耳蜗、听小骨等结构生成专属平面。
- 曲面 CPR planner：基于腰椎中心线或椎管中心线生成 `curved_stack`。
- 置信度算法：根据 label 完整度、曲线平滑度、bbox 合理性给出更细评分。

这些扩展都应优先改阶段四 planner，保持第五步和第六步稳定。
