# AutoPrintFilm

医学影像处理流水线：读取 DICOM 序列，转换为 NIfTI，调用 nnUNet 做分割，并继续执行空间校正、标签定位、解剖分析、切片规划、图像重建和结果校验。

默认 Pipeline 固定为八个阶段：

```text
1 前处理 -> 2 分割 -> 3 空间校正 -> 4 标签定位
-> 5 解剖分析 -> 6 切片规划 -> 7 图像重建 -> 8 校验与报告
```

## 环境要求

- Python 3.10 或 3.11
- pip
- 可选但推荐：CUDA + 对应版本的 PyTorch，用于 nnUNet GPU 推理
- nnUNet 模型文件：需要放在本机可访问的目录，并通过环境变量或配置文件指定

## Linux / macOS 安装依赖

在一台新电脑上下载项目后，进入项目目录运行：

```bash
git clone <你的仓库地址>
cd AutoPrintFilm
bash scripts/install_linux.sh
```

如果不使用安装脚本，也可以手动执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Windows 安装依赖

在 PowerShell 中运行：

```powershell
git clone <你的仓库地址>
cd AutoPrintFilm
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
```

如果不使用安装脚本，也可以手动执行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

如果系统没有 `py` 命令，把 `py -3` 换成 `python`。

## 配置 nnUNet 模型路径

分割阶段会调用 `nnUNetv2_predict`。依赖安装完成后，还需要准备 nnUNet 的模型目录。

默认配置文件是：

```text
src/config/nnuNet_config.json
```

默认这三个配置项为空：

```json
{
  "nnUNet_raw": "",
  "nnUNet_preprocessed": "",
  "nnUNet_results": ""
}
```

为空时，程序不会覆盖系统已有的 nnUNet 环境变量。也就是说，如果你已经在系统里设置好了 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results`，直接运行即可。

Linux / macOS 临时设置示例：

```bash
export nnUNet_raw=/data/nnUNet_raw
export nnUNet_preprocessed=/data/nnUNet_preprocessed
export nnUNet_results=/data/nnUNet_results
```

Windows PowerShell 临时设置示例：

```powershell
$env:nnUNet_raw = "D:/nnUNet/nnUNet_raw"
$env:nnUNet_preprocessed = "D:/nnUNet/nnUNet_preprocessed"
$env:nnUNet_results = "D:/nnUNet/nnUNet_results"
```

也可以把配置文件改成固定路径，例如 Linux：

```json
{
  "nnUNet_raw": "/data/nnUNet_raw",
  "nnUNet_preprocessed": "/data/nnUNet_preprocessed",
  "nnUNet_results": "/data/nnUNet_results"
}
```

Windows 示例：

```json
{
  "nnUNet_raw": "D:/nnUNet/nnUNet_raw",
  "nnUNet_preprocessed": "D:/nnUNet/nnUNet_preprocessed",
  "nnUNet_results": "D:/nnUNet/nnUNet_results"
}
```

同一部位可以在 `_model_rules[].model_mapper` 中按原始扫描层厚配置不同模型。每项
使用包含边界的毫米区间：

```json
{
  "min_scan_thickness": 0.0,
  "max_scan_thickness": 1.5,
  "model_id": 292
}
```

当前脊柱、头骨、骶骨和髋骨按 `0-1.5mm`、`1.6-4.5mm`、`4.6-6mm`
分别选择原薄层、3mm、6mm 模型。肺部目前只有 `0-1.5mm` 的模型。外部厚度未命中
任何区间或同时命中重叠区间时，分割阶段会明确失败，不会自动选择临近模型。

## 运行

先激活虚拟环境。

Linux / macOS：

```bash
source .venv/bin/activate
python main.py -i /path/to/dicom_dir -o /path/to/output_dir --study_part 腰椎 --study_type CT --scan_thickness 3.0
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py -i D:\data\dicom_dir -o D:\data\output_dir --study_part 腰椎 --study_type CT --scan_thickness 3.0
```

支持的 `--study_part`：

```text
头骨, 颈椎, 胸椎, 腰椎, 肺部, 骶骨, 髋骨
```

查看命令帮助：

```bash
python main.py -h
```

## 批量测试计划

仓库提供了固定病例清单脚本，用于批量执行本机测试影像。脚本读取：

```text
/home/peng/03_data/demo/Image/dicom
```

结果输出到：

```text
/home/peng/03_data/demo/Image/result
```

批量脚本会显式向 `main.py` 传入 `--plane_configs`，每个病例都会请求
`横状位`、`矢状位`、`冠状位` 三个状位，并为每个状位输出骨窗、肺窗、
纵隔窗和软组织窗。直接调用 `main.py` 且不传 `--plane_configs` 时，
程序按默认配置只输出横状位。

先只打印命令、检查路径和部位映射，不真正处理影像：

```bash
bash scripts/batch_test_plan.sh --dry-run
```

默认串行执行，每次只跑一个病例：

```bash
bash scripts/batch_test_plan.sh
```

需要异步并发时，用 `--jobs N` 控制最多同时运行的病例数：

```bash
bash scripts/batch_test_plan.sh --jobs 2
```

也可以使用简写：

```bash
bash scripts/batch_test_plan.sh -j 2
```

遇到第一条失败就停止继续启动后续病例：

```bash
bash scripts/batch_test_plan.sh --jobs 2 --stop-on-failure
```

并发任务的单例日志写入各病例输出目录下与 `report.json` 同级的 `log/`：

```text
/home/peng/03_data/demo/Image/result/<病例输出目录>/log
```

程序自身的 `RunLog`、`DebugLog`、`ErrorLog` 也统一保存在该目录中，原有日志正文、
分类和写入规则保持不变。每个病例直接使用固定文件名，不再按日期和小时增加子目录：

```text
log/RunLog.txt
log/DebugLog.txt
log/ErrorLog.txt
log/BatchLog.txt
```

单 GPU 环境建议先从 `--jobs 2` 试起；如果 nnUNet 显存不足或出现 OOM，
降回 `--jobs 1`。

### 当前能力范围

当前正式注册 CT 的横状位、矢状位和冠状位重建能力：

- 颈椎、胸椎、腰椎、肺部、头骨、髋骨三状位均支持基础 `MPR`；
- 每项 `plane_configs` 通过 `SliceType` 显式选择 `MPR`、`CPR` 或 `椎间隙`，
  未传时默认 `MPR`，系统不再按部位和状位自动改变切法；
- 颈椎、腰椎横状位额外支持 `椎间隙`；颈椎只生成 `C2_C3` 至 `C6_C7`，
  不生成包含环椎 C1 的 `C1_C2`；
- 颈椎、胸椎、腰椎冠状位额外支持 `CPR`；
- 不支持的部位、状位和 `SliceType` 组合会在清理输出目录前由预检拒绝。

每个 `plane_configs` 配置可传 `SliceThickness`（层厚）、`SliceSpacing`（层间距）
和 `SliceImageCount`（数量），三项必须传入其中两项或全部传入。第六阶段取得当前
部位在该状位上的有效范围后计算缺少的参数。总范围统一按
`层厚 + (数量 - 1) × 层间距` 计算，首张和末张影像的厚层外缘对应总范围边界。
三项中的任意一项传入 `0` 时与省略该字段含义相同，由另外两个非零参数计算；如果
处理 `0` 后只剩一个有效参数，请求校验失败。负数不表示未配置，仍属于非法输入。

- 只传层厚和层间距时，系统计算能够覆盖有效范围的最少数量；
- 只传层厚和数量时，系统计算层间距；
- 只传层间距和数量时，系统计算层厚；
- 三项全部传入且总范围大于有效范围时，首尾各外扩差值的一半；
- 三项全部传入且总范围小于有效范围时，保持数量不变，层厚和层间距都调整为
  `有效范围 ÷ 数量`。

`slice_range_extension` 配置的固定扩展先计入有效范围，再执行上述计算。椎间隙按
每个有效间隙独立应用同一份配置，例如数量为 20、识别出 5 个有效间隙时，会生成
5 个重建区域、共 100 张物理切片；某个间隙缺失时只跳过该区域，不把它的数量分配
给其他间隙。
同一请求允许存在多份相同状位但采样参数不同的配置。所有配置按请求顺序获得全局
唯一的 `配置01`、`配置02` 等 `ConfigId`。重复状位按出现顺序使用 `横状位`、
`横状位1`、`横状位2` 这样的唯一输出名称，直接隔离各自目录。

`report.json` 顶层返回 `StudyPart`、`StudyType`、数值类型的 `ScanThickness` 和
`PlaneList`。每个状位配置保留调用方原始传入的 `SliceThickness`、`SliceSpacing`、
`SliceImageCount`，包括用来表示未配置的 `0`。实际输出按目录顺序组织为
`WindowPresetList -> RegionList`；每个重建区域使用唯一的 `Region`，并返回实际层厚、
实际层间距、实际数量和路径。普通 MPR/CPR 的区域标识为 `default`，椎间隙统一使用
`C2_C3`、`L1_L2` 等下划线标识。
正式 DICOM 在临时文件中完成层厚、层间距校正后再原子发布，并统一使用
`ISO_IR 192` UTF-8 字符集，保证中文患者及检查信息可被标准读取器正确解码。

### 配置切片范围扩展

个别部位需要在分割边界之外额外保留影像时，可以在
`src/config/localization_config.json` 对应部位和切割方式下配置
`slice_range_extension`。状位名称直接使用现有的 `横状位`、`矢状位`、`冠状位`；
`start`、`end` 固定按毫米解释：

```json
"肺部": {
  "whole_range": {
    "start": "lung",
    "end": "lung",
    "slice_range_extension": {
      "横状位": {
        "start": 3.0,
        "end": 3.0
      }
    }
  }
}
```

`start` 从原始第一张切片所在边界向外扩展，`end` 从原始最后一张切片所在边界
向外扩展；方向跟随当前计划，因此同样适用于反向和斜向直线切片。整个配置、某个
状位或某一端没有填写时，对应扩展默认为 `0mm`，原有逻辑不变。配置距离先加入目标
有效范围，之后再根据层厚、层间距和数量计算最终总范围。

该能力不限制具体部位、切割方式或状位，可以在对应定位配置中按需增加。当前默认
为肺部和髋骨的横状位首尾各扩展 `3mm`；这只是当前配置，不是程序中的允许列表。

肺部已经接入第一版解剖坐标计算：第五阶段从一个整肺分割 Mask 中保留一至两个肺主体，
排除远端小块误分割，并计算肺部物理中心和三轴覆盖范围；第六阶段据此生成以肺部为中心、
完整包住肺部分割范围并按配置保留首尾余量的三状位直线切片。肺部仍沿用 DICOM 标准
方向，不根据肺部形态附加倾斜角度。

脊柱定位和重建以整体范围配置为范围事实源。第四阶段按配置首尾标签展开全部中间椎体；
胸椎默认配置为 `C7 → L1`，显式请求 CPR 时 C7、T1–T12、L1 均进入过滤标注和冠状位 CPR。
腰椎同样按实际配置范围使用。只有颈椎从过滤标注质心和 CPR 中排除 C1，C1 仍保留
在全量定位和切图覆盖范围。

颈椎、胸椎已接入三维直线 MPR：第五阶段使用配置首尾椎体建立整体脊柱轴；矢状位
继续生成直线 `plane_stack`，定位线画在冠状位背景上。胸椎横状位按临床规则保持
原图默认水平横切，颈椎横状位仍可使用整体脊柱轴。颈椎、胸椎和腰椎冠状位仅在
`SliceType=CPR` 时使用过滤标注/四棱柱质心生成 `curved_stack`；`SliceType=MPR`
时生成普通直线 `plane_stack`。CPR 首尾端点只在反弧时回正，并沿当前方向延伸到
原始首末标签的实际出口。

髋骨整切使用左右独立分割：第五阶段分别保留 `hip_left`、`hip_right` 的最大连通
主体，在各自顶部 5mm 内取远离身体中线的外侧区域中心，并用两点连线 AB 确定
横状位角度。AB 只负责方向；横状位沿最终法向使用完整髋骨主体的最大、最小投影
确定覆盖范围，再生成全部平行切片。矢状位和冠状位保持输入 DICOM 默认方向。
三个状位的 `Localizer.dcm` 定位图均选取经过髋骨整体中心的原始层，其中横状位和矢状位
使用冠状定位背景，冠状位使用矢状定位背景。

未传 `--plane_configs` 时默认请求横状位骨窗 MPR；需要矢状位、冠状位或专用
`SliceType` 时应显式传入 `--plane_configs`。MR 和骶骨会在处理输出前由预检明确拒绝。

腰椎横状位区间切优先从第五阶段四棱柱矢状四边形中选择相向端边，并在两条端边共同
切向覆盖区间内取得代表性边界点对；代表点连线作为每段斜横切法向，平面内覆盖仍使用
相邻全量 label bbox 的并集。这样既避免最近点落到角点形成锐角，也避免
横突、棘突等附件使全量 bbox 重叠而跳过 `L1_L2` 至 `L5_S1` 区间，并保持切线在真实
椎体间隙而不是整幅图像中心；输出平面会围绕该位置对称扩展，仍完整覆盖原选定范围。
S1 使用独立的左右支撑筛选，并在主体上缘局部补回一层与上缘相接的连通体素，不再套用
普通腰椎主体裁剪。四边形缺少时回退 faces/bbox；如果
回退路径中的 bbox 仍重叠，对应区间会跳过并写入 `slice_plans.json` 的 `warnings`。
腰椎矢状位整切及其他整切流程不受该回退限制影响。

当前椎体主体标注仍属于基于分割形态和固定支撑阈值的工程启发式。在椎体倾斜明显、
退变、骨赘、附件连接复杂或原始分割缺损的病例上，主体上缘/下缘仍可能出现少量欠分
或过分，并进一步传递到四棱柱边界和椎间切线位置。现阶段“相向端边代表点确定位置、
代表点连线确定法向”的总体方向已经可用，但主体标注稳定性仍需通过更多病例继续优化，
不应把当前输出视为已经完成临床级泛化验证。

第七阶段对每个切片计划先完成全部 MPR HU 图像采样，再按窗位使用线程池处理整套图像。
不同窗位不会重复执行 MPR，默认最多同时运行 4 个窗位线程；第七阶段会等待所有窗位
写出完成后再进入报告阶段。窗位灰度计算保持并行，SimpleITK/GDCM 原生写盘使用
进程内互斥锁串行执行，避免多线程写 DICOM 触发内存竞态。正式切片写出后会校正
DICOM 的 `SliceThickness` 和 `SpacingBetweenSlices`，确保查看器读取到的层厚、
层间距与第六阶段计划一致。

### 输出目录安全

请求和环境预检通过后，程序会清理输出目录中的旧业务结果，再执行本次任务，避免混入
上一次运行留下的结果。与 `report.json` 同级的 `log/` 会被保留，以免任务启动时已经
打开的日志被删除；请确保 `-o/--output` 指向本次任务专用目录。
只有 `debug_file_enabled=true` 时才生成并保留 `manifest.json`、调试 NIfTI、调试元数据、
`Localizer.png` 以及成功任务的中间现场；关闭后不生成这些额外文件，并按原策略清理
`metadata/`、`nifti/` 和 `segmentation/`。`debug_log_file_enabled` 只控制
`log/DebugLog.txt`，不再影响任何调试文件。metadata 中的调试 NIfTI 只保留第五阶段
生成的标签、椎体主体和椎体四棱柱三张掩膜，写出时会在 DebugLog 中逐文件记录完整
路径和生成耗时；第六阶段只写 `slice_plans.json`，不再生成规划 NIfTI。

Pipeline 会在 `RunLog` 中为每个阶段输出“当前序号/总阶段数、稳定阶段名、开始/完成状态
和耗时”；各阶段的输入路径、几何诊断和性能明细继续写入 `DebugLog`。
阶段失败时同时记录稳定标识、中文职责、异常类型、错误原因和调用堆栈，并逐项标明因
fail-fast 被跳过的后续阶段。默认主链中的
稳定阶段名依次为 `preprocessing`、`segmentation`、`spatial_correction`、`localization`、
`anatomical_analysis`、`planning`、`reconstruction`、`reporting`。
只要 `run_log_file_enabled=true`，`ErrorLog` 也会同步写入 `RunLog`，即使 DebugLog 和
独立 ErrorLog 文件均关闭，运行日志仍会保留失败原因。

## 配置优先级

框架配置按以下顺序生效：

```text
显式调用参数 > 环境变量 > 本机配置 > 仓库默认配置
```

仓库中的 nnUNet 路径默认为空，不绑定开发者电脑。推荐通过
`nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 环境变量配置模型目录。

程序启动后会先检查请求能力、输入目录、输出目录关系、nnUNet 命令和模型目录；
预检失败不会创建或清空输出目录。

## 运行测试

Linux / macOS：

```bash
bash scripts/test.sh
```

Windows PowerShell：

```powershell
.\scripts\test.ps1
```

测试脚本固定使用项目 `.venv`，避免系统 Python、FSL Python 或 Conda Python 混用。

## PyTorch / CUDA 说明

`requirements.txt` 会安装项目运行所需的 Python 包，包括 `nnunetv2`。如果需要 GPU 推理，建议先按照本机 CUDA 版本安装 PyTorch，再运行依赖安装命令。PyTorch 安装命令以官方选择器为准。

CPU 环境也可以安装依赖，但 nnUNet 推理速度会明显慢。
