# SliceType 显式切片方式设计

2026-11-18 修订：外部 `SliceType` 不变，内部 MPR、CPR 和椎间隙分别使用独立范围契约；见
[回正与区域配置分离设计](../../architecture/2026-11-18-回正与区域配置分离设计.md)。

## 协议

每个 `plane_configs` 项使用 `SliceType` 指定切片方式，枚举顺序和协议值固定为：

1. `MPR`
2. `CPR`
3. `椎间隙`

省略 `SliceType` 时默认 `MPR`。旧切法值和自动推导逻辑不再兼容。
`SlicePlan.slice_type` 记录实际执行方式，底层 `geometry_type` 继续区分
`plane_stack` 和 `curved_stack` 重采样协议。

## 能力矩阵

- `MPR`：当前已注册部位的横状位、矢状位和冠状位。
- `CPR`：颈椎、胸椎、腰椎、胸腰椎、全脊柱的冠状位和矢状位。
- `椎间隙`：颈椎、腰椎横状位。

“胸腰椎”整体范围为 C7–S1，“全脊柱”整体范围为 C2–S1；两者使用相同模型和切割步骤，
均支持横状位、矢状位、冠状位 MPR 和冠、矢状位 CPR，
不注册椎间隙；三状位 MPR 使用完整整体标签，CPR 排除项不缩小 MPR 范围。

`slice_part_config.json` 是能力组合的唯一配置事实源，`CapabilityRegistry` 在启动时
从中生成能力表。调用方传入未注册组合时，预检在输出目录清理前失败；planner
不再自行猜测或替换切法。

## 执行规则

- MPR 使用普通直线 `plane_stack`。腰椎横状位 MPR 按整体解剖方向从上沿到下沿
  均匀切片，不使用椎间角度。
- 椎间隙为配置引用的每个有效脊柱间隙生成独立 `plane_stack`，方向来自相邻椎体
  几何。当前腰椎配置使用 `L1_L2` 至 `L5_S1`；颈椎因 C1 为环椎，配置只使用
  `C2_C3` 至 `C6_C7`，不生成 `C1_C2`。
- CPR 使用脊柱中心曲线生成 `curved_stack`。冠状位沿前后方向偏移，默认使用矢状位
  定位图；矢状位沿左右方向偏移，默认使用冠状位定位图。两者读取各自状位的
  `face_extension_mm`。有效质心不足或曲线无效时明确失败，
  不能把显式 CPR 请求静默降级成 MPR。

定位阶段把 MPR 映射到 `whole_range.regions` 具名区域、CPR 映射到独立
`spine_range`、椎间隙映射到 `interspace.intervals`；这些值描述内部标签范围契约，
不是新的外部切片方式。MPR 使用 `target_labels` 和上下边界，CPR 与椎间隙仍使用
各自首尾标签。回正方向另从 `localization.pose` 获取，不由 `SliceType` 选择的区域反推。

## 风险与约束

- `SliceType` 是全局协议枚举，不代表每个部位和状位都实现了全部方式。所有请求必须先
  通过能力矩阵校验，禁止在 planner 中散落部位判断来补救非法组合。
- 专用方式依赖更严格的分割和解剖几何。调用方显式请求 CPR 或椎间隙后，几何不足应
  明确失败并返回原因，禁止静默回退成 MPR，否则胶片内容会与模板配置不一致。
- 混合请求可能同时依赖 `whole_range`、`spine_range` 和 `interspace`。阶段契约同时保存
  `resolved_cut_configs` 中各自的解析结果；MPR、CPR 和椎间隙必须读取各自范围，不能共用
  一个可变的当前配置。
- `SliceType` 只决定切片算法。同一状位多份配置已经支持，通过 config_id 识别、
  output_plane 隔离目录；不能只按状位去重而丢掉 MPR 或椎间隙。
