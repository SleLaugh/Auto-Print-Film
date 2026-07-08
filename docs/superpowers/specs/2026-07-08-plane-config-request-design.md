# 三平面重建参数入参设计

## 背景

当前 `Request` 只包含输入路径、输出路径、部位、检查类型、扫描厚度和切割方式。后续要先做头部、腰椎、颈椎等“整切”闭环，重建阶段必须知道每个状位如何采样、用哪些窗宽窗位输出图像。

本设计只补齐请求参数结构，不实现 MPR 重建算法和 DICOM 生成逻辑。

## 目标

新增一个三平面配置数组，用来表达本次任务需要输出哪些状位，以及每个状位的层厚、层间距和窗宽窗位集合。

示例结构：

```json
{
  "plane_configs": [
    {
      "plane": "冠状位",
      "slice_thickness": 5,
      "slice_spacing": 10,
      "window_presets": ["骨窗", "肺窗", "纵隔窗"]
    },
    {
      "plane": "矢状位",
      "slice_thickness": 5,
      "slice_spacing": 10,
      "window_presets": ["骨窗"]
    },
    {
      "plane": "横状位",
      "slice_thickness": 5,
      "slice_spacing": 10,
      "window_presets": ["骨窗"]
    }
  ]
}
```

## 数据模型

新增 `WindowPresetEnum`：

- `骨窗`
- `肺窗`
- `纵隔窗`

复用已有 `PlaneTypeEnum`：

- `横状位`
- `矢状位`
- `冠状位`

新增 `PlaneConfig`：

- `plane: PlaneTypeEnum`
- `slice_thickness: float`
- `slice_spacing: float`
- `window_presets: list[WindowPresetEnum]`

扩展 `Request`：

- `plane_configs: list[PlaneConfig]`

## 默认值

如果调用方没有传 `plane_configs`，先给出保守默认值，保证老命令仍能运行到现有阶段：

```json
[
  {
    "plane": "横状位",
    "slice_thickness": 1,
    "slice_spacing": 1,
    "window_presets": ["骨窗"]
  },
  {
    "plane": "矢状位",
    "slice_thickness": 1,
    "slice_spacing": 1,
    "window_presets": ["骨窗"]
  },
  {
    "plane": "冠状位",
    "slice_thickness": 1,
    "slice_spacing": 1,
    "window_presets": ["骨窗"]
  }
]
```

## 校验规则

- `plane_configs` 必须是非空数组。
- `plane` 必须属于 `PlaneTypeEnum`。
- `slice_thickness` 必须大于 0。
- `slice_spacing` 必须大于 0。
- `window_presets` 必须是非空数组。
- `window_presets` 中的值必须属于 `WindowPresetEnum`。
- 同一个 `plane_configs` 中允许同一状位只出现一次；重复状位视为参数错误，避免后续输出目录冲突。

## 命令行形态

命令行先新增一个 JSON 字符串参数：

```bash
python main.py \
  -i /path/to/dicom \
  -o /path/to/output \
  --body_part 腰椎 \
  --cut_mode whole_range \
  --plane_configs '[{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗","肺窗"]}]'
```

同时支持连字符别名 `--plane-configs`。

后续如果命令行太长，再增加 `--plane_config_file` 读取 JSON 文件。本轮先不加文件参数，避免同时维护两个输入入口。

## 下游约定

`ReconstructionStage` 后续按 `request.plane_configs` 决定输出哪些目录。一个状位配置里有多个窗宽窗位时，语义是同一个状位输出多套图像，例如：

```text
冠状位/
  骨窗/
  肺窗/
  纵隔窗/
```

本轮只保证参数能被解析、校验、序列化并进入报告；具体目录创建、MPR 采样和 DICOM 写出在后续重建阶段实现。

## 测试范围

新增测试覆盖：

- `PlaneConfig.from_dict` 能把字符串转成枚举。
- `Request.from_dict` 能解析 `plane_configs`。
- `Request.to_dict` 能输出 JSON 友好的结构。
- `Request.validate` 能拒绝空数组、重复状位、非正层厚/层间距、空窗宽窗位。
- `main.py -h` 能显示新增参数。

## 暂不实现

- 不在本轮确定骨窗、肺窗、纵隔窗的具体窗位/窗宽数值。
- 不在本轮实现 MPR 重建。
- 不在本轮实现 DICOM 输出目录。
- 不在本轮扩展眼眶、鼻窦、颞骨等新部位。
