# 窗宽窗位预设设计

## 1. 背景

当前外部请求通过 `PlaneConfig.window_presets` 传入窗宽窗位名称，例如：

```json
{
  "plane": "横状位",
  "slice_thickness": 5,
  "slice_spacing": 5,
  "window_presets": ["骨窗", "软组织窗"]
}
```

现有 `WindowPresetEnum` 只包含骨窗、肺窗和纵隔窗，并且没有定义每种预设对应的窗宽与窗位。后续重建阶段如果直接在业务代码中判断名称并填写数值，会造成同一套预设散落在多个阶段，修改时容易遗漏。

本次需要在项目内部建立一套统一的窗宽窗位预设。调用方只传中文预设名称，具体窗宽与窗位由项目内部定义和查询。

## 2. 目标

本次实现需要达到以下目标：

1. 外部请求只传中文名称，例如 `脑窗`、`骨窗`。
2. 项目内部统一维护每个名称对应的窗宽和窗位。
3. 请求解析后仍然保存 `WindowPresetEnum`，保持现有强类型结构。
4. 后续重建或 DICOM 输出阶段通过同一个查询方法获取具体参数。
5. 请求报告和切片方案继续只记录中文预设名称，不重复写入窗宽和窗位。
6. 不支持通过英文名称查询预设。

## 3. 不在本次范围内

本次不处理以下内容：

- 不在阶段四应用窗宽窗位。
- 不生成实际窗宽窗位图像。
- 不修改阶段五的 MPR 采样逻辑。
- 不允许外部请求覆盖预设的窗宽或窗位。
- 不增加英文标识、用途说明或其他展示字段。
- 不把预设改成可动态编辑的 JSON 配置。

## 4. 数据定义

### 4.1 中文名称枚举

扩充 `WindowPresetEnum`，完整包含以下中文名称：

- 脑窗
- 骨窗
- 肺窗
- 纵隔窗
- 软组织窗
- 腹部窗
- 肝窗
- 肾窗
- 胰腺窗
- 骨盆窗
- 脊柱窗

枚举值仍然等于中文名称，例如：

```python
WindowPresetEnum.脑窗.value == "脑窗"
```

这样 `PlaneConfig.from_dict` 当前使用的 `WindowPresetEnum(value)` 解析方式不需要改变，旧请求中的骨窗、肺窗和纵隔窗也保持兼容。

### 4.2 预设参数模型

新增不可变数据模型 `WindowPreset`，只包含三个字段：

```python
@dataclass(frozen=True)
class WindowPreset:
    name: str
    window_width: int
    window_center: int
```

字段含义：

- `name`：中文预设名称。
- `window_width`：窗宽。
- `window_center`：窗位。

预设模型使用 `frozen=True`，避免后续阶段意外修改全局参数。

### 4.3 内部预设表

项目内部统一维护以下固定映射：

| 名称 | 窗宽 | 窗位 |
| --- | ---: | ---: |
| 脑窗 | 80 | 40 |
| 骨窗 | 2500 | 400 |
| 肺窗 | 1600 | -600 |
| 纵隔窗 | 350 | 40 |
| 软组织窗 | 400 | 50 |
| 腹部窗 | 350 | 50 |
| 肝窗 | 150 | 60 |
| 肾窗 | 350 | 40 |
| 胰腺窗 | 300 | 40 |
| 骨盆窗 | 400 | 40 |
| 脊柱窗 | 2000 | 300 |

映射使用 `WindowPresetEnum` 作为键，避免内部继续依赖未经校验的普通字符串。

## 5. 查询接口

新增统一查询方法：

```python
def get_window_preset(
    preset: WindowPresetEnum | str,
) -> WindowPreset:
    ...
```

支持以下两种内部调用方式：

```python
get_window_preset(WindowPresetEnum.脑窗)
get_window_preset("脑窗")
```

两种调用返回相同的 `WindowPreset`：

```python
WindowPreset(
    name="脑窗",
    window_width=80,
    window_center=40,
)
```

允许传入中文字符串，是为了让项目入口以外的简单调用也能直接按名称取值。字符串进入查询方法后必须先转换为 `WindowPresetEnum`，不能绕过枚举校验。

## 6. 外部请求数据流

完整数据流如下：

```text
外部请求中的 "脑窗"
    -> PlaneConfig.from_dict
    -> WindowPresetEnum.脑窗
    -> PlaneConfig.window_presets
    -> 阶段五调用 get_window_preset
    -> WindowPreset(name="脑窗", window_width=80, window_center=40)
    -> 应用到输出图像或 DICOM 标签
```

阶段四仍然只负责把 `WindowPresetEnum` 随切片方案传递下去，不负责把名称展开成数值，也不负责应用窗宽窗位。

## 7. 序列化约定

`PlaneConfig.to_dict()` 和 `Request.to_dict()` 保持现有输出格式：

```json
{
  "window_presets": ["脑窗", "骨窗"]
}
```

序列化结果不展开成下面这种结构：

```json
{
  "window_presets": [
    {"name": "脑窗", "window_width": 80, "window_center": 40}
  ]
}
```

原因是请求和切片方案表达的是“选择了哪些预设”，预设的固定数值属于项目内部定义。这样将来调整某个预设时，不需要改变外部请求格式。

## 8. 错误处理

### 8.1 不支持的中文名称

外部传入不存在的名称时，`WindowPresetEnum(value)` 继续抛出 `ValueError`，请求不会带着未知预设进入后续阶段。

### 8.2 英文名称

英文名称不属于外部接口。例如 `Brain`、`Bone`、`Lung` 都不能解析为预设，并按非法名称处理。

### 8.3 内部预设缺失

测试必须保证 `WindowPresetEnum` 中的每个成员都在内部预设表中存在对应值。这样新增枚举却忘记补参数时，会在测试阶段被发现。

## 9. 文件职责

文件职责确定如下：

- `src/models/enums.py`：维护 `WindowPresetEnum` 的中文名称集合。
- `src/models/window.py`：维护 `WindowPreset`、内部预设表和 `get_window_preset`。
- `src/models/__init__.py`：统一导出预设模型和查询方法。
- `tests/test_window_presets.py`：验证全部预设值和查询行为。
- `tests/test_request_plane_config.py`：验证新增中文名称能够通过外部请求解析和序列化。

## 10. 测试要求

实现前先编写失败测试，至少覆盖：

1. `WindowPresetEnum` 包含全部 11 个中文名称。
2. 每个中文名称对应的窗宽和窗位与预设表完全一致。
3. 查询方法接受 `WindowPresetEnum`。
4. 查询方法接受合法的中文字符串。
5. 英文名称不能作为预设输入。
6. 未知中文名称不能作为预设输入。
7. 每个枚举成员都存在内部参数定义。
8. `PlaneConfig.from_dict` 可以解析新增预设名称。
9. `PlaneConfig.to_dict` 仍然只序列化中文名称。

## 11. 验收标准

以下代码可以直接使用：

```python
config = PlaneConfig.from_dict({
    "plane": "横状位",
    "slice_thickness": 5,
    "slice_spacing": 5,
    "window_presets": ["脑窗", "骨窗"],
})

brain_window = get_window_preset(config.window_presets[0])

assert brain_window.name == "脑窗"
assert brain_window.window_width == 80
assert brain_window.window_center == 40
```

原有骨窗、肺窗、纵隔窗请求继续正常解析；项目全部自动化测试通过；不提交任何 Git commit，由用户检查后自行提交。
