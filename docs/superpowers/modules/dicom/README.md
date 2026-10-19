# DICOM 模块入口

本模块只放 DICOM 输出形态相关资料。写盘并发安全的总规则在 `../../architecture/`。

## 当前有效规则

- 重建切片文件名使用数字序号，如 `1.dcm`、`2.dcm`。
- `Localizer.dcm` 放在对应窗位 series 旁边，第八阶段计数时不应把它当作普通切片。
- `InstanceNumber` 仍表达 DICOM 内部序号，文件名变化不应改变重建几何。
- 定位图背景状位由各部位 `slice_rules.<重建状位>.<切法>.localizer_plane` 显式配置，
  不得与重建状位相同；FillPercent 不改变定位背景取样和真实切面位置，MPR 交线
  按最终成片视野裁剪，端点和长度可变；CPR 投影路径保持不变。
- 成片空间标签逐张使用实际取景几何；正数 FillPercent 按当前层目标居中，矩阵与
  PixelSpacing 共用，但相邻切片可有面内 ImagePositionPatient 平移，不能假定为固定
  面内原点的规则直角三维体。CPR 展开图不能视为原始解剖结构的规则三维体。
- 正式切片按实际窗宽窗位将未取整的重建 HU 映射为 uint16 灰度；GDCM 写盘后补入实际
  `WindowWidth`、`WindowCenter`、`LINEAR_EXACT` 以及对应 Rescale 标签，`RescaleType=HU`。
  这些派生显示片的窗外数值已截断，定量数据应读取原始 CT 序列。
- GDCM 写盘必须经由串行保护入口，具体跨模块约束读 `../../architecture/2026-07-20-GDCM写盘串行保护设计.md`。
- 同一计划各窗 Localizer 必须使用主线程冻结的同一份 `ImagePositionPatient` 和
  `ImageOrientationPatient` 串行写出；窗位仅改变定位图的显示像素、显示标签和序列标识。
- Localizer 定位线使用独立亚像素覆盖率掩膜，并在背景完成窗位映射后叠加；正式
  DICOM 使用白线，debug PNG 使用同一掩膜的红线。该显示处理不得进入正式切片
  重采样，也不得改变切片几何、层厚或层间距。
- Localizer 用 Rescale 标签将 uint16 像素换算到对应窗口的显示数值，并写入实际窗宽窗位；
  `RescaleType=US` 表示未指定单位。定位图属于合成显示数据，不能用于定量 HU 测量，
  调试 PNG 显示不变。三个脊柱部位矢状位 MPR/CPR 的冠状背景可沿脊柱前后深度
  采样派生曲面，几何不足回退单层；DICOM 的 `DerivationDescription` 注明非平面采样，空间标签只描述显示
  参考网格，不能将其当作普通原始冠状平面做精确三维联动。
- Localizer 缺少必要空间标签时必须失败，不能发布带二维默认方向的文件。
- 第八阶段是最终输出硬门禁：先校验正式切片、Localizer、UID、层厚层间距和
  同计划跨窗几何，再发布成功结果；任一硬错误必须使任务失败。
- 输出校验失败时抛出 `OUTPUT_VALIDATION_ERROR`，日志写明计划、窗位、文件和字段；
  debug 开启时保留 `metadata/处理报告_调试.json`，同时写入 `Success=false`、包含错误
  `Message` 且不含切片结果的交付 `report.json`。

## 推荐阅读

| 任务 | 先读文件 |
| --- | --- |
| 数字命名、定位图、第八阶段计数 | `2026-07-14-DICOM数字命名与定位图设计.md` |
| 定位图背景状位配置、参考点与取景隔离 | [定位图状位配置与取景隔离设计](2026-10-08-定位图状位配置与取景隔离设计.md) |
| 取景后的 PixelSpacing、方向和 CPR 限制 | [整组取景设计](../reconstruction/2026-10-08-整组居中与FillPercent取景设计.md) |
| GDCM 并发写盘保护 | `../../architecture/2026-07-20-GDCM写盘串行保护设计.md` |

## 少读原则

- 不涉及文件名、定位图、写盘和标签时，不需要读取本模块。
