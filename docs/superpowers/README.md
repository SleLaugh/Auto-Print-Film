# AutoPrintFilm 文档入口

这个目录用于让 Codex 和开发者快速定位当前有效设计。默认先读本文件，再读对应目录的 `README.md`，最后只打开 1 到 3 个相关设计文件。

## 快速阅读路线

| 需求类型 | 优先读取 | 备注 |
| --- | --- | --- |
| 框架、八阶段流水线、任务状态、预检、产物管理 | `architecture/README.md` | 改跨阶段契约时继续读八阶段拆分设计 |
| 请求参数、状位配置、外部接口契约 | `architecture/README.md` | 重点读状位配置请求设计 |
| 持久缓存、按模型复用分割、缓存失效边界 | `architecture/2026-10-08-预处理与分割缓存设计.md` | 缓存不是整任务断点续跑 |
| 交付报告、Python Response、重复状位、报告对比 | `architecture/2026-10-08-交付报告与响应契约设计.md` | 区分医学状位与输出目录名 |
| 批量脚本、并发病例、FillPercent 配置透传 | `architecture/2026-10-08-批量执行与配置透传设计.md` | 不读取 launch.json |
| 整组倍率、逐层居中、0/80/100、目标占图比例 | `modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md` | 正数逐层居中，0 保持整组中心；不逐张拉满 |
| GDCM、SimpleITK、DICOM 写盘并发安全 | `architecture/README.md` + `modules/dicom/README.md` | 写盘锁属于跨模块约束 |
| DICOM 文件名、定位图、InstanceNumber | `modules/dicom/README.md` | 不涉及 MPR 几何时不读 reconstruction |
| 第三至五阶段校正、label/bbox、解剖几何 | `modules/localization/README.md` | 只处理目标范围或四棱柱时从这里开始 |
| 第六/七阶段切片规划、MPR 和 CPR 重建 | `modules/reconstruction/README.md` | 规划、重采样或部位专属 MPR 优先读这里 |
| 切片中心覆盖、层间距、首尾扩展 | `modules/slicing/README.md` | 当前规则以后 2026-07-20 覆盖设计为准 |
| MPR、椎间隙、直线切割流程、SliceType 能力 | `modules/cutting/README.md` | 只改区间生成或切法选择时读这里 |
| 窗宽窗位、DICOM 定位图、CT 背景展示 | `modules/viewer/README.md` | 展示和定位图需求从这里开始 |
| 追溯历史实施或审阅过程 | `archive/plans/`、`archive/reviews/` | 默认不要读取 |

## 当前事实源

截至 2026-10-08，近期缺失设计和历史文档中的过时说明已进行补录核对。缺口与提交对应关系见
[近期变更文档补录审阅记录](archive/reviews/2026-10-08-近期变更文档补录审阅记录.md)。
新增设计均注明补录日期，不把补录冒充为开发前已存在的方案。
全部旧计划与旧规范的对应关系见[历史文档替代索引](archive/2026-10-08-历史文档替代索引.md)。

文档与实现发生冲突时，按以下顺序确认当前行为：

1. 运行时代码和实际配置；
2. 直接验证该行为的自动测试；
3. `architecture/`、`modules/*/README.md` 和模块当前设计；
4. `plans/` 中尚未完成的当前计划；
5. `archive/` 中的历史记录。

归档文件只解释当时的实施过程，不能覆盖当前代码。跨模块规则审阅方法见
`architecture/2026-07-29-文档与代码一致性审阅设计.md`。

## Token 使用规则

- 不要全量读取 `archive/`。
- 不要为了找当前规则同时读取同主题 plan 和 design；优先读模块 README 指定的当前设计。
- 如果需求只涉及一个模块，先不要读取其他模块。
- 如果 README 已经说明某文件是补充或历史，只在定位冲突、查旧证据、写迁移说明时读取。

## 新文档落位规则

后续新增设计或计划时，优先按本节选择路径，不再默认写入顶层 `specs/`。

| 新文档类型 | 写入路径 | 完成后处理 |
| --- | --- | --- |
| 系统级或跨阶段设计 | `architecture/YYYY-MM-DD-主题设计.md` | 保持为当前入口 |
| 空间校正、label、bbox、解剖几何 | `modules/localization/YYYY-MM-DD-主题设计.md` | 保持为当前入口 |
| 切片规划、MPR/CPR、重建执行 | `modules/reconstruction/YYYY-MM-DD-主题设计.md` | 保持为当前入口 |
| 切片中心、层间距、范围扩展 | `modules/slicing/YYYY-MM-DD-主题设计.md` | 保持为当前入口 |
| 直线切割、MPR、椎间隙、SliceType | `modules/cutting/YYYY-MM-DD-主题设计.md` | 保持为当前入口 |
| DICOM 命名、定位图、输出标签 | `modules/dicom/YYYY-MM-DD-主题设计.md` | 保持为当前入口 |
| 窗宽窗位、DICOM 定位图、展示图 | `modules/viewer/YYYY-MM-DD-主题设计.md` | 保持为当前入口 |
| 当前实施计划 | `plans/YYYY-MM-DD-主题.md` | 完成后移入 `archive/plans/` |
| 一次性代码或文档审阅记录 | `archive/reviews/YYYY-MM-DD-主题.md` | 保持历史属性，不作为当前规则 |
| 被新规则替代的旧设计 | `archive/specs_old/` | 在旧文件顶部注明当前替代文件 |

新增当前有效设计后，同步更新对应目录的 `README.md`，只写阅读优先级和当前规则摘要，不复制正文。

## 写作和注释规则

- 后续新增或修改的项目文档统一使用中文正文。
- 文档文件名沿用当前中文命名风格：`YYYY-MM-DD-中文主题.md` 或 `YYYY-MM-DD-中文主题设计.md`。
- 代码里的关键业务注释不要过少；涉及医学语义、坐标系、切片规则、DICOM 标签、异常保护和阶段输入输出时，应补充中文注释。
- 新增或调整重要方法时，应在文档、注释或 docstring 中说明方法职责、输入、输出和边界条件。
- 新增或调整关键变量时，应优先让变量名表达含义；业务含义不直观的变量需要用中文注释说明，尤其是方向向量、物理坐标、层厚、层间距、窗宽窗位和 label 映射。
- 技术名词、代码标识、路径、命令和 DICOM tag 保持原样，不强行翻译。

## 目录说明

- `architecture/`：系统级设计和跨模块约束。
- `modules/`：按功能模块拆分的当前有效知识。
- `plans/`：当前开发计划入口；没有当前任务时只保留说明文件。
- `specs/`：旧路径提示；当前规范不再直接放这里。
- `archive/`：历史计划、审阅记录和已被替代的旧设计；正文可能与当前代码不同，默认不要读取。

## 维护原则

- 当前有效设计放在 `architecture/` 或 `modules/`，并在对应 `README.md` 标明阅读优先级。
- 实施计划完成后移入 `archive/plans/`。
- 被新设计取代的规范移入 `archive/specs_old/`，并在旧文件内保留替代关系说明。
- 图片跟随对应设计目录，不单独散落在顶层。
- 变更请求字段、阶段输入输出、缓存、配置或可观察结果时，同轮更新对应设计与入口索引；
  只更新根 README 或提交说明不算补齐模块设计。审阅时逐项对照代码、测试与文档。
