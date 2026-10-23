# `slice_part_config.json` 配置说明

`slice_part_config.json` 是切割部位相关配置的唯一事实来源。它同时控制：

- 原始扫描层厚对应的 nnUNet 分割模型；
- 标签别名与定位区域的首尾端点；
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

区域端点和 CPR 排除项引用的是别名，例如 `L1`，而不是直接写
`vertebrae_L1`。别名的值是分割结果中的实际标签名；如果一个别名需要对应多个实际
标签，可以使用字符串数组。

同一个实际标签不能被多个别名重复引用。所有在 `regions.*.start_label`、
`regions.*.end_label` 或 `spine_cpr.excluded_labels` 中出现的名称，都必须先在
`label_aliases` 中定义。

当前配置包含以下标签组：

- 脊柱：`C1`–`C7`、`T1`–`T12`、`L1`–`L5`、`S1`；
- 单体或成对结构：`lung`、`brain`、`hip_left`、`hip_right`。

颅脑的配置目标仍是 `brain`，不增加听眦线、AC-PC、手工标志点或斜切开关。
第五阶段默认逐病例以脑形态估计前后倾角，不设绝对倾角上下限；分割或方向证据
不足时保留原 AP 投影，方向不由骨标签决定。
若当前模型 `dataset.json` 明确提供 `skull` 且分割中存在相应体素，运行时沿脑区最终
头脚轴的物理范围筛选骨体素，仅用于第七阶段取景、居中和防裁边，不要求新增 `skull`
别名或修改模型规则。方向和三状位基础范围只用 `brain`，切片层组再应用现有范围扩展。
当前仅横状位 MPR 默认 `start_mm=10.0`、`end_mm=0.0`，向头侧补 `10 mm`、下侧不补；
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
    "regions": {
      "default": {
        "type": "whole_range",
        "start_label": "L1",
        "end_label": "S1"
      }
    }
  },
  "slice_rules": {
    "横状位": {
      "MPR": {
        "regions": ["default"],
        "range_extension": { "start_mm": 0.0, "end_mm": 0.0 },
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

层厚区间是闭区间，匹配条件为 `min <= ScanThickness <= max`。区间不能重叠；没有
任何区间命中时任务会明确失败，不会自动选择临近模型。例如 `0–1.5` 和 `1.6–4.5`
不会重叠，但 `1.55` 不属于任何一个区间。

当前配置中的模型选择如下：

| 部位 | 检查类型与层厚区间 | 模型 |
| --- | --- | --- |
| 颅脑、髋骨 | CT：`0–1.5`、`1.6–4.5`、`4.6–6.0 mm` | `294`、`297`、`298` |
| 颈椎、胸椎、腰椎、脊柱 | CT：`0–1.5`、`1.6–4.5`、`4.6–6.0 mm` | `292`、`297`、`298` |
| 肺部 | CT：`0–1.5 mm` | `315` |
| 骶骨 | 已禁用 | — |

### `localization`

`localization.regions` 定义可以被切法引用的定位区域：

| 字段 | 可选值 | 说明 |
| --- | --- | --- |
| `type` | `whole_range`、`interspace` | 整体范围或椎间隙范围。 |
| `start_label` | 已定义别名 | 区域起点标签。 |
| `end_label` | 已定义别名 | 区域终点标签。 |

`MPR` 和 `CPR` 只能引用 `whole_range`，每条规则当前只能引用一个整体区域；
`椎间隙` 只能引用 `interspace`，可以一次引用多个椎间隙区域。区域必须至少被一条
切法规则引用，不能定义未使用区域。

同一部位当前只支持一个整体定位区域，MPR 和 CPR 共用它。例如将颈椎 `default` 的
`end_label` 从 `C7` 改为 `T1`，或将腰椎 `start_label` 从 `L1` 改为 `T12`，会同时
改变该部位整体切割的目标范围和 CPR 标签序列；椎间隙仍由各自的区域列表决定，
不会自动增加 `C7_T1` 或 `T12_L1`。

整体范围也参与空间校正、解剖姿态分析和定位图参考范围计算。因此端点扩展后，
除了新增椎体的覆盖范围，已有图像的中心、视野、切片数量及 CPR 中心线也可能变化；
即使椎间隙列表不变，其共享定位图和校正结果也可能随之调整。此次调整只更改默认
配置，沿用现有算法。

定位阶段会合并该部位的整体范围和椎间隙标签，并检查分割结果中的实际体素。
因此扩展后新增的标签必须在相应任务的分割结果中存在；当前颈椎默认终点为 `T1`，
需要包含至 `T1` 的目标标签，上述腰椎示例则需要 `T12`。仅请求椎间隙也会执行
这项检查，缺失时会报错。模型选择规则保持不变；已有分割缓存若包含这些标签即可
复用，生成过的图像需要重新运行任务才会采用新范围。

端点必须能由本次模型标签解析。当前脊柱顺序支持 `C1` 至 `S1`；S1 在运行时优先
使用 `vertebrae_S1`，缺少时回退到模型已有的整个 `sacrum`。这个回退不能表达
骶骨内部的 S2 边界，不能仅新增 `S2` 别名来实现切到 S2。

脊柱部位支持可选的 `localization.spine_cpr`：

```json
{
  "spine_cpr": {
    "full_label": true,
    "excluded_labels": ["C1"]
  }
}
```

- `full_label` 为布尔值。`true` 使用完整标注范围生成 CPR，`false` 使用过滤后的主体范围；
- `excluded_labels` 是别名数组，用于从 CPR 标签范围中排除指定标签，可以省略，默认为空数组。

只要某个部位配置了 `CPR`，就必须同时配置 `spine_cpr`；没有 `CPR` 的部位不应配置
`spine_cpr`。当前 `颈椎`、`胸椎`、`腰椎`、`脊柱`启用 CPR，`full_label` 均为 `true`，颈椎
另外排除 `C1`。

### `slice_rules`

`slice_rules` 定义“状位 + 切法”的能力矩阵。支持的状位和切法只有：

- 状位：`横状位`、`冠状位`、`矢状位`；
- 切法：`MPR`、`CPR`、`椎间隙`。

每条规则包含：

| 字段 | 说明 |
| --- | --- |
| `regions` | 要使用的区域名称数组，必须引用当前部位 `localization.regions` 中已定义的区域。 |
| `range_extension.start_mm` | 从定位起点向外扩展的距离，单位 mm。 |
| `range_extension.end_mm` | 从定位终点向外扩展的距离，单位 mm。 |
| `localizer_plane` | 定位图的参考显示状位，只能是三种状位之一，且不能与当前重建状位相同。 |
| `framing_policy` | 可选，仅普通 MPR 可配置；在正数 `FillPercent` 下按指定依据取景，详见下节。 |

范围扩展值必须是非负有限数值；`0.0` 表示不扩展。`localizer_plane` 只决定定位图的
参考背景状位，不改变正式输出的重建状位。颈椎、胸椎、腰椎、脊柱的矢状位 MPR/CPR
选用冠状背景时，第七阶段可沿脊柱前后深度采样派生曲面；这不增加配置字段，也不
修改 `localizer_plane`、正式切片或范围扩展规则。曲面定位图不能视为原始冠状平面。
定位图保留原图网格和固定 `localizer_reference`，不随 FillPercent 缩放、裁切或移动；
MPR 直接按第六阶段真实平面求交，交线仅裁到底图边界或曲面显示网格，长度不随
正式成片取景变化。定位线不表示正式切片的有限视野，CPR 保持真实计划曲线路径。

CPR 的 `range_extension` 沿当前输出状位的切片偏移轴扩展，分别增加负向和正向的
覆盖范围；冠状位和矢状位读取各自的 CPR 规则，不改变中心曲线的首尾椎体。
首尾椎体范围通过 `localization.regions` 调整。

颈椎、胸椎、腰椎、脊柱均可在任务的 `plane_configs` 中显式选择矢状位 CPR，例如：

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
矢状位 CPR 只展开冠状面的左右侧弯，保留矢状面前后生理曲度；纵向按冠状投影
弧长采样，横向固定为患者前后方向。该语义由规划器生成内部
`curve_projection_axis` 字段，无需增加请求参数或外部配置开关。旧计划缺少该字段时
保持原三维曲线展开行为；需要重新规划和重建才会采用新的矢状位 CPR。
`localizer_plane` 仍由各条规则显式配置。仅增加支持规则不会自动增加输出序列；省略
`SliceType` 时仍为 `MPR`，现有矢状位 MPR 和批量脚本的默认请求保持原有行为。

### `framing_policy`

普通 MPR 可在对应状位规则中配置以下对象；当前默认只在
`slice_parts.髋骨.slice_rules.冠状位.MPR` 中启用：

```json
{
  "regions": ["default"],
  "range_extension": { "start_mm": 0.0, "end_mm": 0.0 },
  "localizer_plane": "矢状位",
  "framing_policy": { "basis": "selected_region", "margin_mm": 5.0 }
}
```

| 字段 | 约束 |
| --- | --- |
| `basis` | 必需，当前只能为 `selected_region`。 |
| `margin_mm` | 必需，每侧余量，必须是非负有限数字，0 允许；拒绝 bool、字符串、负数及非有限值。 |

对象不接受其他字段，也不接受 null、数组或非对象值；CPR、椎间隙不能配置该对象。
它不是 `plane_configs` 请求字段，不增加部位专属倍率。第六阶段将当前状位规则深拷贝为
计划快照，第七阶段在所有正数 `FillPercent` 下使用；0、null或省略忽略该策略，保留各部位
原行为，默认髋骨冠状位仍保留完整源切面和基础间距。

`selected_region` 把本次所选原标签的整组物理包络投影到行、列轴，以包络宽高加
每侧 `margin_mm` 作为目标基准。任意正数占比 p 的视野为该基准除以 `p/100`，面内中心
始终取包络中点。所有层及各窗共用中心、倍率、PixelSpacing 和
矩阵，末层只剩小标签或空目标层时仍保持同一取景。该分支优先完整容纳所选标签，允许
腹部、外围皮肤、大腿及目标包络之外的股骨远端越框，不再以完整体表限制放大。
100优先完整显示所选部位；降低正数占比时直接从未预裁的原 CT 采样更大的外围范围，
皮肤、腹部和大腿逐渐露出，不保证80已包含全部体表；视野超出原扫描处保留正常无数据背景。
`source_labels`、切面、法向范围、层厚、层间距、张数及定位图均不改变。

重建诊断为 `center_source=group_selected_region`，`framing.selected_region` 记录
`basis`、`margin_mm` 及未加余量的 `lower_in_plane_mm`、`upper_in_plane_mm`（行、列轴顺序）。
`framing_target_size_mm` 记录整组原标签包络加两侧余量，`slice_target_size_mm` 仍为当前
厚层尺寸或空值。本分支不加入自动相邻标签或人体支持，不输出 `source_preservation`、
`skin_edge` 字段。

当前模型294没有独立 `sacrum`、`coccyx`，不能把左右髋骨/股骨包络等同于所有骶尾结构
的标签保护。本次两份真实源另核对了骶骨中线、髋臼和股骨头。配置可用于其他部位的普通
MPR；未启用该对象的计划继续使用原规则。肺部正式策略维持整组固定倍率与偏移，平滑
变倍率只作独立预览。公式及诊断见[整组取景设计](../../docs/superpowers/modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md)。

## 显示取景支持

以下完整人体保护、`skin_edge` 及髋骨逐层中心说明，均排除上节已启用 `selected_region` 的正数计划。

肺部正数 `FillPercent` 的 `plane_stack` MPR 对横、冠、矢三状位分别选择实际厚层内
选定目标与获准有界标签 bbox 面积（宽×高）最大的一张作参考层，面积相同取计划中
靠前的层。由参考 bbox 中点相对原计划中心的行、列偏移确定整组共同平移，所有层
和空目标层均使用同一平移，不逐层追随标签。共同取景尺寸按所有实际厚层相对该固定
平移后中心的两侧最大距离乘二逐轴取最大值，必要时将整组视野一次放宽以保留全部目标。
这仍是其他正数比例的取景规则和 100 的标签诊断基准；100 实际采用下述紧致人体公式。
全部实际厚层无目标时原目标中心回退整组。

髋骨正数普通 MPR 先按当前厚层联合 bbox 逐层部位居中，100 再作下述必要校正，以实际各厚层联合尺寸
逐轴最大值作为共同目标尺寸，不计跨层位置漂移。肺部、髋骨正数普通 MPR 同时保护
实际采样 CT 在各厚层的完整人体范围：每个源 CT 按 HU 提取一次人体支持，去掉外部
空气和可可靠分离的薄床板，保留肩臂、双腿、皮肤、肋骨及骶尾骨，边界加 `2 mm` 余量。
源几何与实际厚层采样决定各层支持范围。80 等其他正数比例围绕既定中心计算对称保护尺寸；
100 采用下述紧致视野及必要中心校正。人体不参与肺参考层或标签目标尺寸计算；无法识别人体的源层局部
回退完整源盒，与身体相连或不能可靠分离的设备保守保留，不发身体不可用告警。
显式 `body_support`、`trunk_framing.py` 保留内部工具语义，生产不启用排臂躯干取景。

两部位 `0<FillPercent<100` 普通 MPR 三状位的共同视野仍取“标签目标尺寸/请求占比”与
“既定中心下人体保护尺寸”的逐轴较大值。100 改用人体与目标并集的紧致包络：肺部对原
参考偏移追加整组同一中心校正，共同视野取所有层相对边界的包络宽高；髋骨共同视野取
各层并集尺寸最大值，仅将原中心超出完整容纳人体合法区间的坐标移入区间。100 不再与
旧标签对称视野取较大值，目标可以偏离绝对正中。两者再统一 PixelSpacing 并收紧共同画布。
像素数向上取整并保持原矩阵各轴奇偶性，不重新计算倍率。100 中心校正不保证原源图
网格相位；组内及各窗 Rows、Columns、PixelSpacing 一致；DICOM 原点按实际
矩阵和每层真实中心重算，`framing.output_size_xy` 记录实际列数、行数。这些计划随后
不执行逐张上下无源支持行 ROI，保持肺部整组共同偏移，髋骨 100 为最近合法中心、其他正数为原目标中心。
人体保护可统一限制放大，使实际目标占比低于请求值，100% 也不例外；不要求每张
小截面或任意胶片格宽高同时铺满。`framing_target_size_mm` 保留标签尺寸，
`source_preservation` 以 `mode=complete_body_slab` 另记录 `required_size_mm`、
`field_of_view_mm`、`zoom_limited`、`body_margin_mm=2` 和 `fallback_source_slice_count`
（回退完整源盒的源层数），`scale` 是实际倍率。100 另记录 `canvas_mode=skin_edge`、
`centering_policy`（肺 `group_offset`，髋 `nearest_body_fitting_center`）、`center_adjustment_mm`；
`center_source` 分别为 `group_skin_edge`、`slice_skin_edge`，原中心信息保存在
`target_center_source`、`target_center_before_skin_edge`。肺部 `reference_shift_mm` 为最终实际偏移，
100 新增 `reference_target_shift_mm` 保存原标签参考偏移。具体公式见
[整组取景设计](../../docs/superpowers/modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md)。

肺部、髋骨普通 MPR 在 0/null/省略时也保护完整源切面：保留基础 PixelSpacing、scale=1
和整组目标中心，必要时对称扩大共同矩阵，允许超过原矩阵并保持奇偶性，不执行上下 ROI，
诊断模式为 `complete_source_slab`，外部空气和床板可保留。
其他部位、CPR、椎间隙及 Localizer 保留各自规则，不增加请求开关。
脑区仍用脑区与有界 skull，脊柱仍用有界相邻椎体和可靠椎间镜像面。除上述肺部、髋骨
普通 MPR 外，平面成片在 HU 采样后、窗化前逐张以 ROI 裁去首尾完全无源支持行，
包括 0/null/省略。裁后整幅图居中，Rows、垂直视野及中心可逐层不同，有效像素、Columns、
PixelSpacing 和采样网格相位保留，同层各窗共用实际几何。不按黑色或 HU 阈值裁剪、
不取全组最短高度、不补框。CPR、Localizer 不执行上下裁空，整张无源支持层保留并标记 `no_source_support`。

所有 `plane_stack` 厚层只平均有效源采样：轴对齐按有效重叠厚度、斜切按有效子平面数归一，
真实空气参与、越界填充值不参与，零支持仍保留空气；不补出扫描外组织，不改变切割范围、
方向、层厚、层间距或张数。此规则不改变 CPR 的曲面厚层融合。
详见[整组取景设计](../../docs/superpowers/modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md)。

## 当前能力矩阵

| 部位 | 启用 | 定位范围 | 已配置的状位与切法 |
| --- | --- | --- | --- |
| 颅脑 | 是 | `brain`–`brain` 整体范围；同次骨标签仅辅助脑部范围内取景 | 横 MPR、冠 MPR、矢 MPR |
| 颈椎 | 是 | `C1`–`T1`；`C2_C3` 至 `C6_C7` 椎间隙 | 横 MPR/椎间隙、冠 MPR/CPR、矢 MPR/CPR |
| 胸椎 | 是 | `C7`–`L1` 整体范围 | 横 MPR、冠 MPR/CPR、矢 MPR/CPR |
| 腰椎 | 是 | `T12`–`S1`；`L1_L2` 至 `L5_S1` 椎间隙 | 横 MPR/椎间隙、冠 MPR/CPR、矢 MPR/CPR |
| 脊柱 | 是 | `C7`–`S1` 整体范围 | 横 MPR、冠 MPR/CPR、矢 MPR/CPR |
| 肺部 | 是 | `lung`–`lung` 整体范围 | 横 MPR、冠 MPR、矢 MPR |
| 骶骨 | 否 | — | — |
| 髋骨 | 是 | `hip_left`–`hip_right` 整体范围 | 横 MPR、冠 MPR、矢 MPR |

`脊柱` 是独立切割部位，三状位 MPR 与冠状位、矢状位 CPR 共用 `C7`–`S1`
整体范围，不配置椎间隙。CPR 使用完整标签范围，不排除椎体。扫描和分割结果需要
覆盖该范围；S1 标签缺失时沿用 `sacrum` 回退。调用方使用 `"SlicePart": "脊柱"`，
并在 `plane_configs` 中显式传入需要的五种组合；默认请求仍只生成横状位 MPR。

颈椎 `default` 的 `C1`–`T1`、腰椎 `default` 的 `T12`–`S1` 是三状位 MPR
及冠状位、矢状位 CPR 共用的整体目标范围；颈椎 CPR 仍按排除项跳过 `C1`，
使用 `C2`–`T1`。整体端点扩展不新增椎间隙：颈椎仍为 `C2_C3` 至 `C6_C7` 五组，
腰椎仍为 `L1_L2` 至 `L5_S1` 五组，不生成 `C7_T1` 或 `T12_L1`。当前没有 `S2` 标签，
腰椎终点仍为 `S1`；实际模型只提供 `sacrum` 时沿用运行时标签回退。

肺部和髋骨的横状位规则当前均设置 `start_mm=3.0`、`end_mm=3.0`；两者的横状位
定位图均使用冠状位。颅脑横状位 MPR 为 `start_mm=10.0`、`end_mm=0.0`，仅补头侧；
颈椎横状位 MPR 为 `start_mm=0.0`、`end_mm=5.0`，在 C1–T1 目标范围的终点侧外扩
`5 mm`，不要求新增 T2 标签。颈椎冠、矢状位 MPR 以及颅脑冠、矢状位和其他当前规则的
范围扩展为 `0.0 mm`。这项固定余量不补全源扫描外内容。定位图状位按 JSON 中的
`localizer_plane` 读取。颅脑头侧余量沿用既有字段，不新增 JSON 配置项。

`10 mm` 是当前默认工程余量，不保证所有病例颅顶完整，也不能恢复源扫描缺失部分。
范围校对应看首末切片厚层的外缘，Localizer 定位线表示中心面，第一条线不是覆盖起点。

## 修改规则与校验

修改配置时按以下顺序处理：

1. 新增标签时先在 `label_aliases` 中定义别名，再在定位区域中引用；
2. 为部位添加或修改 `model_rules`，确认层厚区间无重叠，且需要覆盖的层厚没有空洞；
3. 在 `localization.regions` 定义区域，并在 `slice_rules` 中引用；
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
