# `slice_part_config.json` 配置说明

跨阶段职责、旧配置迁移及腹部/四肢的扩展边界见
[回正与区域配置分离设计](../../docs/superpowers/architecture/2026-11-18-回正与区域配置分离设计.md)。

`slice_part_config.json` 是切割部位相关配置的唯一事实来源。它同时控制：

- 原始扫描层厚对应的 nnUNet 分割模型；
- 标签别名、独立回正输入和定位区域的边界；
- 每个部位支持的状位、切法和定位图状位；
- 整体范围、椎间隙范围以及范围扩展；
- 脊柱 CPR 是否使用完整标签范围，以及需要排除的标签；
- 普通 MPR 在正数 FillPercent 下是否按所选部位标签连续取景。

默认文件为：

```text
src/config/slice_part_config.json
```

应用启动时会读取并严格校验该文件。发布包中的 `src/config/` 是外置配置目录，修改
JSON 后下一次任务启动即可生效，无需重新打包。
`dev build` 默认把源码目录中的全部 JSON 同步到发布目录和压缩包；仅显式使用
`dev build --keep-config` 时才沿用上次 `dist` 中的配置修改。

## 顶层结构

```json
{
  "schema_version": 1,
  "label_aliases": {},
  "slice_parts": {}
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | 整数 | 当前必须为 `1`。 |
| `label_aliases` | 对象 | 配置内部使用的标签别名到 nnUNet 标签名的映射。值可以是字符串，也可以是字符串数组。 |
| `slice_parts` | 对象 | 按切割部位名称组织的部位配置，至少包含一个部位。 |

### `label_aliases`

回正输入、区域边界和 CPR 排除项引用的是别名，例如 `L1`，而不是直接写
`vertebrae_L1`。别名的值是分割结果中的实际标签名；如果一个别名需要对应多个实际
标签，可以使用字符串数组。

同一个实际标签不能被多个别名重复引用。`pose.labels`、`pose.roles`、
`pose.fit_excluded_labels`、区域内的标签引用和 `spine_cpr.excluded_labels`，
都必须先在 `label_aliases` 中定义。别名数组中的顺序保留，不能重复填写同一别名。

当前配置包含以下标签组：

- 脊柱：`C1`–`C7`、`T1`–`T12`、`L1`–`L5`、`S1`；
- 单体或成对结构：`lung`、`brain`、`hip_left`、`hip_right`。

颅脑的配置目标仍是 `brain`，不增加听眦线、AC-PC、手工标志点或斜切开关。
第五阶段默认逐病例以脑形态估计前后倾角，不设绝对倾角上下限；分割或方向证据
不足时保留原 AP 投影，方向不由骨标签决定。
若当前模型 `dataset.json` 明确提供 `skull` 且分割中存在相应体素，运行时沿脑区最终
头脚轴的物理范围筛选骨体素，仅用于第七阶段取景、居中和防裁边，不要求新增 `skull`
别名或修改模型规则。方向和三状位基础范围只用 `brain`，切片层组再应用现有范围扩展。
当前仅横状位 MPR 默认 `top_mm=10.0`、`bottom_mm=0.0`，向头侧补 `10 mm`、下侧不补；
同一计划的骨取景上界同步增加该余量，下界仍为脑区，远处面部、下颌骨不延长层组。
缺骨仅用脑区取景并告警；脑区接边报告截断风险，配置余量不补出扫描外组织。
这属于形态估计，不保证与听眦线或 AC-PC 线平行，详见
[颅脑形态斜切设计](../../docs/superpowers/modules/reconstruction/2026-10-20-颅脑形态斜切设计.md)。

## 单个部位的结构

```json
{
  "enabled": true,
  "model_rules": {
    "CT": [
      {
        "scan_thickness_range_mm": { "min": 0.0, "max": 1.5 },
        "model_id": 292
      }
    ]
  },
  "localization": {
    "pose": {
      "method": "spine",
      "labels": ["L1", "L2", "L3", "L4", "L5", "S1"],
      "roles": {"start": "L1", "end": "S1"}
    },
    "regions": {
      "default": {
        "type": "whole_range",
        "target_labels": ["L1", "L2", "L3", "L4", "L5", "S1"],
        "bounds": {
          "upper": {
            "labels": ["L1", "L2", "L3", "L4", "L5", "S1"],
            "edge": "upper", "reduce": "max", "offset_mm": 0.0
          },
          "lower": {
            "labels": ["L1", "L2", "L3", "L4", "L5", "S1"],
            "edge": "lower", "reduce": "min", "offset_mm": 0.0
          }
        }
      }
    }
  },
  "slice_rules": {
    "横状位": {
      "MPR": {
        "regions": ["default"],
        "face_extension_mm": {
          "top_mm": 0.0, "bottom_mm": 0.0,
          "front_mm": 0.0, "back_mm": 0.0,
          "left_mm": 0.0, "right_mm": 0.0
        },
        "localizer_plane": "矢状位"
      }
    }
  }
}
```

### `enabled`

布尔值，表示该部位是否可以被任务使用。设为 `false` 后，查询接口会拒绝该部位，
但配置文件本身仍会经过结构校验。禁用部位可以保留模型规则，也可以使用空的定位和
切法规则；当前示例中的 `骶骨` 为禁用部位。

### `model_rules`

模型规则按检查类型组织：`model_rules.<检查类型>` 是非空数组，每项支持以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `scan_thickness_range_mm.min` | 非负有限数值 | 扫描层厚下限，单位 mm。 |
| `scan_thickness_range_mm.max` | 非负有限数值 | 扫描层厚上限，单位 mm，不能小于 `min`。 |
| `model_id` | 正整数 | nnUNet 数据集 ID。 |
| `configuration` | 可选 | 覆盖默认 nnUNet 配置，例如 `3d_fullres`。 |
| `fold` | 可选 | 覆盖默认训练折；值会传给 nnUNet 的 `-f` 参数。 |
| `plans` | 可选字符串 | 覆盖全局训练计划名称，对应 `-p`。 |
| `input_orientation` | 可选字符串，`RAS` / `LPS` | 第二阶段把临时推理输入的数组轴顺序/正负方向整理为模型训练约定；省略不调整。 |

`nnuNet_config.json` 提供全局默认值，命中的模型规则按字段覆盖：

| 模型规则中的字段 | 未填写时使用的全局字段 | 全局也未填写时 |
| --- | --- | --- |
| `configuration` | `_configuration` | `3d_fullres` |
| `fold` | `_fold` | `0` |
| `plans` | `_plans` | `nnUNetPlans` |

三个字段独立处理。例如只写 `configuration`，`fold` 和 `plans` 仍使用全局默认值；
只写 `model_id` 和层厚区间时，三个字段全部继承全局默认值。

`input_orientation` 独立于上述三个 nnUNet 参数，只在命中的模型规则中启用，不设置全局
默认。它通过轴置换和翻转重排临时输入，过程不插值、不改变原 CT 或其物理位置；预测结果
回到原 CT 物理网格后才发布分割缓存。该字段是模型输入的存储约定，不是第五阶段的解剖
回正，也不用于把真实斜扫描强行摆直。只接受完整大写 `RAS` / `LPS`，不接受 null 或其他类型。
启用时缓存必须具有匹配的方向标记；未标记或方向不同的历史结果不得直接复用。
移除该字段后，原带方向标记的缓存也须重新推理；完整命中条件见
[缓存设计](../../docs/superpowers/architecture/2026-10-08-预处理与分割缓存设计.md)。

层厚区间是闭区间，匹配条件为 `min <= ScanThickness <= max`。区间不能重叠；没有
任何区间命中时任务会明确失败，不会自动选择临近模型。例如 `0–1.5` 和 `1.6–4.5`
不会重叠，但 `1.55` 不属于任何一个区间。

当前配置中的 CT 模型选择如下，括号内为重采样设备配置：

| 部位 | `0–1.5 mm` | `1.6–4.5 mm` | `4.6–6.0 mm` |
| --- | --- | --- | --- |
| 颅脑 | `901`（GPU） | `297`（CPU） | `298`（CPU） |
| 髋骨 | `902`（GPU） | `297`（CPU） | `298`（CPU） |
| 颈椎、胸椎、腰椎、胸腰椎、全脊柱 | `292`（GPU） | `297`（CPU） | `298`（CPU） |
| 肺部 | `903`（GPU） | 无匹配规则 | 无匹配规则 |
| 上腹部、下腹部、全腹部 | `701`（CPU） | `701`（CPU） | `701`（CPU） |
| 骶骨 | 已禁用 | 已禁用 | 已禁用 |

上腹部、下腹部、全腹部实际使用单个连续 `0–6 mm` 闭区间，包括 `1.55`、`4.55` 等值；上表仅借用
其他部位的列宽展示。该范围是路由准入，不是所有层厚分割质量已验证的承诺。
701 规则显式固定 `configuration=3d_fullres`、`fold=0`，不继承全局这两项的改动，
并设置 `input_orientation=RAS`。本机模型的训练输入使用 RAS 数组存储约定，训练未采用
镜像增强，`NibabelIO` 也不自动归一数组方向；直接传入 LPS 存储的正式 CT 会造成输入约定
不一致。本设置只规范该模型的临时推理输入，不改权重、模型文件或原九部位配置。

当前启用的模型共七个：`292`、`297`、`298`、`701`、`901`、`902`、`903`。`315` 只保留在
重采样设备和超时配置中，没有启用的部位规则引用它；`294` 已不在当前模型规则中。

如需接入自定义模型，可在对应规则中填写 `configuration: "3d_lowres"`、`fold: "all"`
和实际的 `plans` 名称；其他未填写这些字段的规则仍使用全局默认。
`plans` 是训练计划名称，实际方案内容保存在 `plans.json`；`all` 对应单独训练的
`fold_all` 权重，不表示合并数字 fold。推理只需部署所用模型的完整训练产物，
无需复制训练数据。`nnuNet_config.json` 中的 `nnUNet_results` 是唯一模型根目录：

```text
nnUNet_results/Dataset<ID>_<名称>/
  nnUNetTrainer__<plans>__<configuration>/
    dataset.json
    plans.json
    fold_<fold>/checkpoint_final.pth
```

701 当前部署目录为 `Dataset701_AbdomenPelvis7/nnUNetTrainer__nnUNetPlans__3d_fullres/`，
使用 `fold_0`。其 `plans.json` 内部 `plans_name=nnUNetPlans24GB` 不改变推理目录查找规则；
当前沿用 `_plans=nnUNetPlans`，不要只因内部元数据不同就修改全局设置或模型文件。
实际部署到其他目录命名时，在 701 规则单独填写匹配的 `plans`。

当前项目入口和 CPU/GPU 映射如下；模型根目录和外部 Python 路径按部署机器调整：

```json
{
  "nnUNet_results": "/home/peng/03_data/00_nnUnet/model/nnUNet_results",
  "nnUNet_predict_command": "src/inference/nnunet_gpu_export.py",
  "nnUNet_python": "/home/peng/opt/fsl/bin/python",
  "acceleration_scheme": "scheme6",
  "export_resampling_threads": 4,
  "export_resampling_device_by_model": {
    "292": "gpu",
    "315": "gpu",
    "901": "gpu",
    "902": "gpu",
    "903": "gpu"
  }
}
```

`nnUNet_raw`、`nnUNet_preprocessed` 是训练数据目录，应用不再读取或检查，可从旧配置
删除。上述入口必须由已经安装 nnUNet、PyTorch 的外部 `nnUNet_python` 执行。
入口相对路径在源码运行时解析为项目脚本，在发布程序中解析为包内
`_internal/nnunet_adapter/nnunet_gpu_export.py`；旧发布包须重新构建并整体更新。
`export_resampling_device_by_model` 的键为任意正整数模型 ID 字符串，值只能为 `cpu`
或 `gpu`，未列出的模型默认 `cpu`，因此当前 `297`、`298`、`701` 使用 CPU。将某项改成
`cpu` 即可让该模型使用 CPU，例如 `"903": "cpu"`。应用按本次选中的模型 ID 精确匹配，
不依赖训练计划的数据集名称，也不会匹配其他编号的前缀。
`acceleration_scheme` 只能为 `scheme2` 或 `scheme6`：当前 `scheme6` 为 GPU 模型启用
GPU 输入 cubic 重采样、跳过推理后不用的临时掩膜重采样和类型转换、GPU 标签合并；
改为 `scheme2` 即恢复项目原有的 GPU 输出重采样方案。两种方案都保留 `gpu` / `cpu`
映射，且不改变网络预测设备。CPU 模型继续使用 CPU 重采样；真实分割输入或级联模型
保留原临时掩膜处理。方案 6 的 GPU 模型必须使用上述项目入口，切换到官方
`nnUNetv2_predict` 时还须将方案改为 `scheme2`。设备配置本身不代表已完成对应模型的病例验证。
当前肺部模型 `903` 使用全局单次超时 `1800` 秒；保留的 `"315": 180.0` 超时项只对
模型 `315` 生效，不会随部位切换自动转移到 `903`。
外部环境变量 `NNUNET_GPU_RESAMPLING=0` 可全局强制 CPU，默认 `1` 允许按模型映射选择。
`export_resampling_threads` 控制 CPU 路径的通道并行数，范围 `1` 至 `4`，不控制 GPU；
无需将当前的 `4` 改成 `1`。输出保留现有 float32 插值与类型转换顺序；方案 6 在 GPU
直接比较分数、保留最大分数的标签，输入 cubic 则以 float64 工作精度匹配原来的半像素
坐标和范围裁剪，不改变裁剪、临时掩膜创建和归一化。
方案 6 还需要外部 nnUNet Python 可导入 CuPy；本机 `/home/peng/opt/fsl/bin/python`
已安装 CuPy 13.6.0 和 fastrlock 0.8.3。也可用环境变量 `GPU_INPUT_CUPY_PATH` 指定独立
依赖目录。输入 GPU 失败会明确报错，标签合并不支持的情况会记录回退原因。
切换方案或设备只影响后续实际推理；已有本地分割缓存仍可能直接复用。需要确认新配置的
输出时，使用新的缓存和输出目录，并查看该次 `nnunet-*.stdout.log` 中的实际处理记录。
本机四个模型的配对运行已完成，分割及最终 DICOM 存在差异；验证范围与迁移限制见
[生产部署说明](../../docs/production.md#scheme6-validation)。
模型权重、`plans.json` 和已安装 nnUNet 的文件均不修改。现有 `disable_tta=true`、
`step_size=0.7` 及导出进程数 `1` 保留。更多部署和显存说明见
[生产部署说明](../../docs/production.md)。
程序只在 `nnUNet_results` 中按 `Dataset<ID>_` 精确匹配编号，并向 nnUNet 传递完整数据集名称，避免
不同长度的数据集编号被外部命令的前缀匹配混为同一编号。

### `localization`

定位分成两类独立输入：`pose` 负责回正方向，`regions` 负责切割区域。
普通 MPR 按“回正 → 选择当前切法方向 → 求区域边界 → 六方向外扩 → 切片”处理。
现有脑部形态倾角归入 `brain` 回正；颈椎横状位的附加方向处理保留在专用算法中，
不增加斜切角度、坐标系编号等公开配置。

本次直接改造现有配置，`schema_version` 仍为 `1`，不提供旧 `whole_range` 首尾端点
结构的兼容分支。旧字段会明确报错；部署时须一并更新程序和本文件。

#### `pose`：回正输入

| `method` | 必填输入 | 说明 |
| --- | --- | --- |
| `brain` | `labels` | 脑区标签，沿用脑部形态回正。 |
| `lung` | `labels` | 肺区标签，沿用肺部回正及既有可选辅助证据。 |
| `hip` | `roles.left`、`roles.right` | 左、右髋骨别名；无需在 `labels` 重复填写。 |
| `spine` | `labels`、`roles.start`、`roles.end` | `labels` 显式列出参与回正的椎体；端点角色用于方向参考，必须包含在 `labels` 中。 |

需要填写 `labels` 的方法必须提供非空别名数组；`hip` 仅通过左右角色提供标签，不接受 `labels`。
`roles` 中每个值是一个别名。
脊柱的 `start`、`end` 别名各自必须恰好对应一个实际椎体标签；髋骨每侧别名可以对应多个实际标签。
只有 `spine` 支持可选的 `fit_excluded_labels`，默认为空数组。它从脊柱回正拟合中
排除指定标签，但不删除该标签的端点角色、区域覆盖或其他用途；不能排除全部回正标签。
颈椎仍显式配置 `fit_excluded_labels: ["C1"]`，与 CPR 的 C1 排除分别管理。

回正输入不从区域上下界推断。仅修改 `bounds` 不会改变回正标签；只调整 CPR 范围也
不会修改普通 MPR 区域。更换回正标签本身则可能改变方向和图像中心，应单独验证。

#### `regions`：按切法定义区域

| `type` | 使用者 | 必填字段 |
| --- | --- | --- |
| `whole_range` | 普通 MPR | `target_labels`、`bounds.upper`、`bounds.lower` |
| `spine_range` | CPR | `start_label`、`end_label` |
| `interspace` | 椎间隙 | `start_label`、`end_label` |

一个部位可以定义多个普通 MPR 区域，每条 MPR 规则选择一个。CPR 当前使用一个独立的
`spine_range`；每条 CPR 规则选择该区域。椎间隙规则可以选择多个 `interspace`。
区域必须至少被一条切法规则引用，不能定义未使用区域。

`whole_range.target_labels` 是用于区域横向覆盖和取景的目标别名数组，必须非空。
它与回正标签、上下界参考标签分别配置；用某结构确定方向或边界，不会自动把它加入
取景目标。目标标签也不代表仅显示该标签内的 CT 像素，原图中的周围组织仍参与重建。

`whole_range.transverse_source` 为可选字符串，只接受 `labels` 或 `body`：

| 值 | 左右、前后范围来源 |
| --- | --- |
| `labels`，省略时默认 | 原目标标签投影范围，保留旧部位行为。 |
| `body` | 已求解头脚范围内的 CT 体表；目标标签用于躯干主体锚点，不把器官范围当作体表。 |

上腹部、下腹部、全腹部使用 `body`，仅补 LR/AP，不改变回正方向或上下边界。可靠分离的外围手臂可排除，
与躯干连接或无法可靠区分的组织保守保留。不可识别体表的源层回退该层原 CT 视野与
区域头脚范围的交集并告警；全段无源交集则失败。省略此字段不会额外提取体表。
不接受 `null`、数组、数字、布尔值或其他字符串；该字段只属于 `whole_range`。

上下边界 `bounds.upper`、`bounds.lower` 均使用以下结构：

| 字段 | 类型/取值 | 说明 |
| --- | --- | --- |
| `labels` | 非空别名数组 | 提供边界位置的标签。 |
| `edge` | `upper` / `lower` | 分别取每个实际标签在当前解剖头脚轴上的上界或下界，必填。 |
| `reduce` | `max` / `min` | 再从这些位置中取最大值或最小值；展开后只有一个实际标签时可省略。 |
| `offset_mm` | 有限数字 | 最后沿当前头脚轴增加偏移，正数向头侧，负数向脚侧；省略为 `0`。 |

边界名称与 `edge` 各自独立：区域下边界也可以引用某个标签的上界。
多个别名，或一个别名展开为多个实际标签时，必须明确 `reduce`。
不能直接按原始数组 Z 索引求上下界；物理位置在该切法实际使用的解剖方向上计算。
最终上界必须高于下界，否则任务报错，不自动交换。
`offset_mm` 是基础区域定义的一部分；`face_extension_mm` 在此基础上再外扩，二者相加。

原八部位迁移配置对全部 `target_labels` 取 `max` 上界和 `min` 下界，保持原有目标并集范围，
没有把脊柱范围缩成只看首尾椎体。原部位模型、六方向余量、定位图状位和取景参数保持原值。
新增腹部使用组合边界，见[腹部接入设计](../../docs/superpowers/architecture/2026-11-18-上下腹部接入设计.md)。

`spine_range` 的首尾标签按既有脊柱顺序展开，并由独立的 `spine_cpr` 决定 CPR 几何。
`interspace` 保留既有相邻椎体逻辑。MPR 边界偏移不会改变 CPR 中心线或椎间隙范围。
当前脊柱顺序支持 `C1` 至 `S1`；S1 在运行时优先使用 `vertebrae_S1`，缺少时回退到
模型已有的整个 `sacrum`，不能通过新增 `S2` 别名获得骶骨内部的 S2 边界。

定位阶段统一收集回正、边界、取景目标、CPR 和椎间隙所需标签，并检查实际分割体素。
缺失必需标签时明确报错；回正辅助证据仍按各算法既有规则处理。当前颈椎配置需要包含
至 `T1`，腰椎需要包含 `T12`；仅请求椎间隙时也会检查该部位配置所需标签。

#### `spine_cpr`：CPR 几何选项

```json
{
  "spine_cpr": {
    "full_label": true,
    "excluded_labels": ["C1"]
  }
}
```

- `full_label` 必填布尔值：`true` 使用完整标注范围，`false` 使用过滤后的主体范围；它不是启用开关。
- `excluded_labels` 是 CPR 排除别名数组，可省略，默认为空；不控制回正或普通 MPR 边界。

配置了 CPR 就必须同时配置 `spine_cpr`；没有 CPR 的部位不应配置它。
当前颈椎、胸椎、腰椎、胸腰椎、全脊柱的 `full_label` 均为 `true`，颈椎 CPR 另排除 C1。

### `slice_rules`

`slice_rules` 定义“状位 + 切法”的能力矩阵。支持的状位和切法只有：

- 状位：`横状位`、`冠状位`、`矢状位`；
- 切法：`MPR`、`CPR`、`椎间隙`。

每条规则包含：

| 字段 | 说明 |
| --- | --- |
| `regions` | 要使用的区域名称数组，必须引用当前部位 `localization.regions` 中已定义的区域。 |
| `face_extension_mm.top_mm` / `bottom_mm` | 沿患者头脚方向从当前区域上、下边界向外扩展的距离，单位 mm。 |
| `face_extension_mm.front_mm` / `back_mm` | 沿患者前后方向从当前区域前、后边界向外扩展的距离，单位 mm。 |
| `face_extension_mm.left_mm` / `right_mm` | 沿患者左右方向从当前区域左、右边界向外扩展的距离，单位 mm。 |
| `localizer_plane` | 定位图的参考显示状位，只能是三种状位之一，且不能与当前重建状位相同。 |
| `framing_policy` | 可选，仅普通 MPR 可配置；在正数 `FillPercent` 下按指定依据取景，详见下节。 |

六个范围扩展字段必须全部填写，值必须是非负有限数值；`0.0` 表示不扩展。`localizer_plane` 只决定定位图的
参考背景状位，不改变正式输出的重建状位。颈椎、胸椎、腰椎、胸腰椎、全脊柱的矢状位 MPR/CPR
选用冠状背景时，第七阶段可沿脊柱前后深度采样派生曲面；这不增加配置字段，也不
修改 `localizer_plane`、正式切片或范围扩展规则。曲面定位图不能视为原始冠状平面。
定位图保留原图网格和固定 `localizer_reference`，不随 FillPercent 缩放、裁切或移动；
MPR 直接按第六阶段真实平面求交，交线仅裁到底图边界或曲面显示网格，长度不随
正式成片取景变化。定位线不表示正式切片的有限视野，CPR 保持真实计划曲线路径。

`face_extension_mm` 是所有部位通用的六面扩展。它始终基于当前 `regions` 选中的区域，
按患者固定的上、下、前、后、左、右方向解释；换成其他状位时方向含义不变。普通 MPR
按六面扩展三维区域后再投影到当前平面；椎间隙仅把上下两项用于既有层组范围扩展，
前后左右四项用于最终成片面内视野；CPR 将前后/左右分别作用到对应 offset 和横向视野，
上下作用到纵向范围。六个值为 `0.0` 表示该方向不扩展。

椎间隙左右、前后分别可控，例如修改对应规则为：

```json
"face_extension_mm": {
  "top_mm": 0.0, "bottom_mm": 0.0,
  "front_mm": 5.0, "back_mm": 5.0,
  "left_mm": 10.0, "right_mm": 10.0
}
```

这使左右各增加 `10 mm`、前后各增加 `5 mm` 的患者方向余量；斜切时投影到实际 row/col
轴，各方向可能同时影响两个画面轴。四项不通过法向分量扩大层组，不改变切割角度、
法向范围、层厚、间距或数量。第六阶段将投影结果保存为内部计划字段
`framing_in_plane_extension_mm`（`row_min`、`row_max`、`col_min`、`col_max`），无需新增外部配置开关。

第七阶段先完成原 FillPercent 及跨间隙共同比例计算，再将各侧余量按 `ceil(mm / spacing)`
扩为像素，同配置各间隙取逐侧最大余量共用画布。PixelSpacing 和倍率不变；不对称扩展
仅使中心在面内平移，原采样网格嵌入新画布。正数、0、null 或省略 FillPercent 均支持。
前后左右全 0 或旧计划缺少上述内部字段时保留原行为，默认 JSON 数值不变。
余量不能补全源扫描外组织；完整无源支持行仍按原规则裁去，因此最终 Rows 可逐层不同。

胸腰椎和全脊柱的五条整体规则默认上下各增加 `50 mm`，前后和左右暂为 `0 mm`；
需要冠状位左右增加范围时，直接修改对应冠状位规则的 `left_mm`、`right_mm`。

颈椎、胸椎、腰椎、胸腰椎、全脊柱均可在任务的 `plane_configs` 中显式选择矢状位 CPR，例如：

```json
{
  "Plane": "矢状位",
  "SliceType": "CPR",
  "SliceThickness": 1.0,
  "SliceSpacing": 3.0,
  "WindowPresets": ["骨窗"]
}
```

冠状位 CPR 沿前后方向偏移，画面横向覆盖左右，默认使用矢状位定位图；
矢状位 CPR 沿左右方向偏移，画面横向覆盖前后，默认使用冠状位定位图。
冠状位 CPR 只展开 AP–SI 矢状投影，保留 LR–SI 左右侧弯，横向固定为第五阶段
`anatomical_geometry.global_frame` 的 LR；`curve_projection_axis` 取该 LR，
`offset_axis` 成对取同一坐标系 AP，无解剖几何时成对回退患者 LR/AP。
矢状位 CPR 仍以患者 AP 为 `curve_projection_axis`，展开 LR–SI 冠状投影，
保留 AP–SI 前后生理曲度。两状位纵向按对应投影弧长采样，真实三维
`curve_points` 保留为解剖轨迹；采样、offset 覆盖、基础横向 FOV 与定位线共用
对应投影几何，FOV 覆盖固定横向参考两侧最远标签外缘，避免偏心裁边。
本次只修正冠状位，矢状位沿用现有逻辑。上述均为内部计划语义，无需增加请求
参数或外部配置开关。旧计划缺少
`curve_projection_axis` 时保持原三维展开；需重新规划和重建才应用新语义。
`localizer_plane` 仍由各条规则显式配置。仅增加支持规则不会自动增加输出序列；省略
`SliceType` 时仍为 `MPR`，现有矢状位 MPR 和批量脚本的默认请求保持原有行为。

### `framing_policy`

普通MPR可在对应状位规则中配置：

```json
{
  "regions": ["default"],
  "face_extension_mm": {
    "top_mm": 0.0, "bottom_mm": 0.0,
    "front_mm": 0.0, "back_mm": 0.0,
    "left_mm": 0.0, "right_mm": 0.0
  },
  "localizer_plane": "矢状位",
  "framing_policy": { "basis": "selected_region", "margin_mm": 5.0, "canvas": "original" }
}
```

| 字段 | 约束 |
| --- | --- |
| `basis` | 必需，当前只能为`selected_region`。 |
| `margin_mm` | 必需，每侧余量，非负有限数字；允许0，拒绝bool、字符串、负数、非有限值。 |
| `centering` | 可选，`group`、`reference_slice`、`slice_target`；缺省`group`。 |
| `canvas` | 可选，`target`、`original`；缺省`target`，兼容旧紧致画布。 |

对象不接受其他字段、null、数组或非对象值，CPR和椎间隙不能配置。它不是请求字段，
第六阶段深拷贝为计划快照，第七阶段仅在正数FillPercent的普通plane_stack MPR使用。

当前肺部、髋骨三个状位的六条普通MPR规则都使用上述配置：`selected_region`、每侧5mm、
缺省group中心及`canvas=original`。整组原标签物理包络中点为固定中心，
包络宽高加两侧余量为共同目标尺寸T，小标签末层及空目标层同样保持组中心和倍率。
不再使用肺参考层或髋横/矢逐层目标中心，不以完整人体、皮肤或肩臂限制倍率。

上腹部、下腹部、全腹部三状位配置为 `selected_region`、`margin_mm=0`、`centering=group`、`canvas=target`。
正数占比的共同目标与中心来自第五阶段已解析的体表区域，不用某层局部肝肾或髋骨包围盒
缩窄腹部。`target` 采用目标画布，区别于肺/髋保留原 CT 画框比例的 `original`；
0/null/省略保持基础间距和已解析区域。`transverse_source` 决定区域本身，
`framing_policy` 决定正数占比如何显示该区域，两者职责不同。

`original`保留第六阶段原画框矩阵N和基础像素纵横比s，不把画布收紧到目标包络。
基础画框在原图尺寸可用时由完整源CT中心角点投影确定；保留的是原画框物理宽高比，
不复制原CT中心、不要求保全整个源范围。正数q=FillPercent/100时：

```text
f = max(T/(N*s))/q
output_spacing_mm = s*f
FOV = N*output_spacing_mm
```

100的FOV是在原画框比例内完整容纳T的矩形，不一定两轴都等于T；70、50保持组中心及矩阵，
宽高同时扩大为100的1/0.7、2倍，向四周扩展，80为1.25倍。各层、各窗同倍率；
两轴使用同一缩放因子，不拉伸。直接从完整原CT采样，不预裁外围图像再缩小补框。
小截面和不限制目标适配的一轴仍可留空气，冠状位100也不承诺零空气或胶片格四边满铺。

高占比允许包络外皮肤、肋骨、腹部、大腿或骶尾结构越框，低占比逐渐露出；
80不保证已包含全体表，扫描外仍为正常无数据背景。当前薄层髋骨模型902只有左右髋骨标签，
无独立sacrum/coccyx；
左右髋骨包络不等于所有骶尾结构保护；所选标签及余量完整保留，不补全扫描外数据。

诊断selected_region记录basis、margin_mm、centering、canvas及未加余量的
lower_in_plane_mm/upper_in_plane_mm整组边界（行列顺序）。默认肺/髋统一
center_source=group_selected_region，不记录参考层字段；framing_target_size_mm为组标签尺寸加余量，
slice_target_size_mm只记录每层诊断。实际间距字段为output_spacing_mm，不是spacing。
肺/髋当前正数策略不自动加入相邻标签/人体，也不输出source_preservation或skin_edge字段。

显式兼容配置可选择reference_slice参考层共同偏移、slice_target当前厚层中心或target紧致画布，
不是肺/髋当前六条生产默认；腹部默认显式采用target。0/null/省略忽略策略，肺/髋仍完整保源、基础间距、scale=1、整组中心，
必要时扩共同矩阵，不做上下ROI。其他部位、CPR、椎间隙、定位图、source_labels、方向、
法向范围、层厚、层间距、数量不变。

## 显示取景支持

正数肺/髋普通MPR按上节部位策略取景，保留共同画布；不自动提取人体或作100专属中心修正。
腹部使用显式 `body` 区域和上述取景策略。其余既有部位继续原有规则：脑区与有界skull、脊柱有界相邻椎体及可靠椎间镜像面保持不变。
其他适用平面与椎间隙在HU采样后、窗化前逐张以ROI裁去首尾完全无源支持行，包括0/null/省略。
裁后整幅图居中，Rows、垂直视野及中心可逐层不同；有效像素、Columns、PixelSpacing、
采样相位及同层跨窗几何保留。不按黑色或HU阈值裁剪，不取全组最短高度，不拉伸或补框。
CPR、Localizer不执行上下裁空，整张无支持层保留并标记`no_source_support`。

未配置策略的旧肺/髋计划及显式`body_support`、`trunk_framing.py`内部工具可继续使用旧人体
保护、肺参考层/髋逐层中心或`skin_edge`；这些不是默认正数肺/髋普通MPR规则。
所有`plane_stack`厚层只平均有效源采样，真实空气参与、越界填充值不参与，零支持保持空气；
不补扫描外组织、不改切割范围或参数，不影响CPR曲面厚层融合。
详见[整组取景设计](../../docs/superpowers/modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md)。

## 当前能力矩阵

| 部位 | 启用 | 定位范围 | 已配置的状位与切法 |
| --- | --- | --- | --- |
| 颅脑 | 是 | `brain`–`brain` 整体范围；同次骨标签仅辅助脑部范围内取景 | 横 MPR、冠 MPR、矢 MPR |
| 颈椎 | 是 | `C1`–`T1`；`C2_C3` 至 `C6_C7` 椎间隙 | 横 MPR/椎间隙、冠 MPR/CPR、矢 MPR/CPR |
| 胸椎 | 是 | `C7`–`L1` 整体范围 | 横 MPR、冠 MPR/CPR、矢 MPR/CPR |
| 腰椎 | 是 | `T12`–`S1`；`L1_L2` 至 `L5_S1` 椎间隙 | 横 MPR/椎间隙、冠 MPR/CPR、矢 MPR/CPR |
| 胸腰椎 | 是 | `C7`–`S1` 整体范围 | 横 MPR、冠 MPR/CPR、矢 MPR/CPR |
| 全脊柱 | 是 | `C2`–`S1` 整体范围 | 横 MPR、冠 MPR/CPR、矢 MPR/CPR |
| 肺部 | 是 | `lung`–`lung` 整体范围 | 横 MPR、冠 MPR、矢 MPR |
| 骶骨 | 否 | — | — |
| 髋骨 | 是 | `hip_left`–`hip_right` 整体范围 | 横 MPR、冠 MPR、矢 MPR |
| 上腹部 | 是 | 肝上界至双髋较高上界；横向使用区域内体表 | 横 MPR、冠 MPR、矢 MPR |
| 下腹部 | 是 | 双肾较高上界向头侧 30 mm 至双股骨较高上界；横向使用区域内体表 | 横 MPR、冠 MPR、矢 MPR |
| 全腹部 | 是 | 肝上界至双股骨较高上界；横向使用区域内体表 | 横 MPR、冠 MPR、矢 MPR |

上腹部、下腹部、全腹部复用 `hip` 方法和 `roles.left/right` 双髋回正，不宣称通用临床标准定位。
701 的股骨标签包含股骨头但无独立股骨头编号，当前以股骨上端近似头上界；
全腹部目标标签为肝、双髋和双股骨，上下边界偏移均为 `0`，不依赖双肾。
下腹部固定肾上方 `30 mm` 不能独立证明实际肾上腺完整覆盖。三个腹部部位六方向外扩均为零，
横、冠状位定位图为矢状位，矢状位定位图为冠状位；必需标签缺失或上下界反转时失败。

原独立切割部位 `脊柱` 更名为 `胸腰椎`，整体范围保持 `C7`–`S1`（19 个椎体）；
`全脊柱` 当前整体范围为 `C2`–`S1`（C2–C7、T1–T12、L1–L5、S1，共 24 个椎体）。
两者的模型规则、切割步骤和重建算法一致。三状位 MPR 使用 `default`，冠状位、矢状位
CPR 使用 `cpr_range`，当前二者的标签覆盖一致，不配置椎间隙。
CPR 使用完整标签范围，不排除椎体。扫描和分割结果需要覆盖所选范围；S1 标签缺失时
沿用 `sacrum` 回退。调用方使用 `"SlicePart": "胸腰椎"` 或 `"SlicePart": "全脊柱"`，
并在 `plane_configs` 中显式传入需要的五种组合；默认请求仍只生成横状位 MPR。

颈椎 `default` 覆盖 `C1`–`T1`，腰椎 `default` 覆盖 `T12`–`S1`，供三状位 MPR 使用；
各自独立的 `cpr_range` 保持相同首尾标签。颈椎 CPR 仍按排除项跳过 `C1`，
使用 `C2`–`T1`。整体端点扩展不新增椎间隙：颈椎仍为 `C2_C3` 至 `C6_C7` 五组，
腰椎仍为 `L1_L2` 至 `L5_S1` 五组，不生成 `C7_T1` 或 `T12_L1`。当前没有 `S2` 标签，
腰椎终点仍为 `S1`；实际模型只提供 `sacrum` 时沿用运行时标签回退。

既有部位的毫米余量以用户提供的线上配置为准。2026-11-19 核对的实际文件为
`/home/peng/03_data/demo/Project/发布/AutoSliceServer/src/config/slice_part_config.json`；
其 9 个部位、36 条重建规则的六方向余量已完整保留，配置结构迁移不改变这些数值。
所有含非零余量的规则如下，单位均为 mm：

| 部位 | 切法与状位 | 上 | 下 | 前 | 后 | 左 | 右 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 颅脑 | 横状位 MPR | 10 | 0 | 0 | 0 | 0 | 0 |
| 颈椎 | 横状位 MPR | 0 | 5 | 0 | 0 | 0 | 0 |
| 颈椎、腰椎 | 横状位椎间隙 | 0 | 0 | 0 | 0 | 20 | 20 |
| 胸腰椎、全脊柱 | 三状位 MPR；冠状位、矢状位 CPR | 50 | 50 | 0 | 0 | 0 | 0 |
| 肺部、髋骨 | 横状位 MPR | 3 | 3 | 0 | 0 | 0 | 0 |

表外规则的六方向余量均为零，包含新增上腹部、下腹部、全腹部的三状位 MPR。
下腹部肾上方 `30 mm` 属于基础区域的 `bounds.upper.offset_mm`，不属于此表的外扩。
颈椎横状位在 C1–T1 目标范围的底侧外扩 `5 mm`，不要求新增 T2 标签；椎间隙的
左右各 `20 mm` 用于成片视野；CPR 上下余量沿脊柱曲线首末段延长。
肺部和髋骨横状位的定位图仍使用冠状位，其余按 JSON 中的 `localizer_plane` 读取。
固定余量不补全源扫描外内容。来源指纹、完整比对范围和执行验证见
[线上配置迁移核对记录](../../docs/superpowers/archive/reviews/2026-11-19-线上配置迁移核对记录.md)。

`10 mm` 是当前默认工程余量，不保证所有病例颅顶完整，也不能恢复源扫描缺失部分。
范围校对应看首末切片厚层的外缘，Localizer 定位线表示中心面，第一条线不是覆盖起点。

## 修改规则与校验

修改配置时按以下顺序处理：

1. 新增标签时先在 `label_aliases` 中定义别名，再在定位区域中引用；
2. 为部位添加或修改 `model_rules`，确认层厚区间无重叠，且需要覆盖的层厚没有空洞；
3. 独立配置 `localization.pose`，在 `localization.regions` 定义区域，并在 `slice_rules` 中引用；
4. 根据需要添加 CPR 配置和范围扩展；
5. 启动任务前执行 JSON 语法和业务校验。

推荐的本地校验命令：

```bash
python -m json.tool src/config/slice_part_config.json >/dev/null
python -c "from src.config import load_slice_part_config; load_slice_part_config()"
```

校验会拒绝未知字段、非法状位或切法、重复标签、未定义标签、未使用区域、区域类型
不匹配、重叠层厚区间、非法范围扩展、同状位定位图，以及 CPR 配置不完整等情况。

修改后建议运行与配置直接相关的测试：

```bash
python -m unittest tests.test_config_loader tests.test_segmentation_model_matching
```
