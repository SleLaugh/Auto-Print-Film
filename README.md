# AutoPrintFilm

医学影像处理流水线：读取 DICOM 序列，转换为 NIfTI，调用 nnUNet 做分割，并继续执行空间校正、标签定位、解剖分析、切片规划、图像重建和结果校验。

默认 Pipeline 固定为八个阶段：

```text
1 前处理 -> 2 分割 -> 3 空间校正 -> 4 标签定位
-> 5 解剖分析 -> 6 切片规划 -> 7 图像重建 -> 8 校验与报告
```

## 环境要求

- 源码开发：Python 3.10 或 3.11、pip；发布程序已自带应用运行时
- 外部 nnUNet 推理环境；GPU 推理所需的 PyTorch / CUDA 在该外部环境中维护
- nnUNet 模型文件：需要放在本机可访问的目录，并通过环境变量或配置文件指定

## 生产发布

在项目根目录执行 `dev build`，用 PyInstaller onedir 生成带基础依赖的 Linux x86_64
可执行程序发布包：

```bash
dev build
```

直接入口为 `dist/AutoPrintFilm/AutoPrintFilm`，压缩包为
`dist/AutoPrintFilm-linux-x64.tar.gz`。整体移动应用目录，保留 `_internal/` 和符号链接；
生产端无需安装应用 Python 或运行 `main.py`，包内不交付应用源码脚本。
JSON 配置外置于产物的 `src/config/`，可直接编辑；重复构建默认保留已修改的发布配置。
nnUNet、PyTorch 和模型继续使用外部环境，通过 `nnuNet_config.json` 配置路径，不打入包。
新机器克隆项目后先运行一次 `./dev install-command`，或直接使用 `./dev build`。
支持 `ExecutablePath` 的 .NET Worker 直接启动程序，未设置时保留
`PythonExecutablePath` + `MainScriptPath` 兼容方式。当前服务调用路径为
`/home/peng/03_data/demo/Project/发布/AutoSliceServer/AutoPrintFilm`；移动目录后同步调整
服务配置中的 `ExecutablePath`。只有使用尚未支持该字段的旧 Worker 时，才需先改造并
重新发布 Worker。
解压、Worker 构建和配置步骤见 [发布部署手册](docs/production.md)。当前构建基线为
Ubuntu 22.04 Linux x86_64，其他平台需另行验证兼容性。
下文安装、开发测试与批量运行脚本仅提供于源码仓库。

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

独立转换工具 `src/OKT-convert2nii.py` 另需安装 `requirements-tools.txt` 中的依赖；
这些工具依赖不随主应用发布。

## 配置外部 nnUNet

分割阶段调用外部 `nnUNetv2_predict`。应用依赖安装不会安装 nnUNet，还需配置已有的
外部命令入口和模型目录。

默认配置文件是：

```text
src/config/nnuNet_config.json
```

此文件同时配置外部推理入口和模型目录。本机外部入口为：

```json
{
  "nnUNet_predict_command": "/home/peng/opt/fsl/bin/nnUNetv2_predict",
  "nnUNet_python": ""
}
```

`nnUNet_predict_command` 指向外部可执行命令。上面的入口自带外部 Python 路径，
所以 `nnUNet_python` 留空。`nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results`
只配置模型目录，不能代替命令路径。
服务启动时的 PATH 可能与开发终端不同，生产配置应填写命令的完整路径。
发布程序已读取这些 JSON 字段，修改后下一次任务启动生效，无需重新打包。

推理速度和后台进程数可以通过同一 JSON 的顶层字段调整。将以下字段合并到原 JSON，保留其他字段：

```json
{
  "disable_tta": false,
  "step_size": 0.5,
  "num_processes_preprocessing": 1,
  "num_processes_segmentation_export": 1
}
```

JSON 中 `disable_tta` 使用布尔值，默认 `false`，设为 `true` 会关闭镜像增强、减少重复推理。
`step_size` 使用数值，范围为 `0 < step_size <= 1`，默认 `0.5`；增大步长可以减少
滑窗重叠。`num_processes_preprocessing` 和 `num_processes_segmentation_export`
分别对应 nnUNet 的 `-npp`、`-nps`，默认均为 `1`，只接受正整数，拒绝布尔值和浮点数。
环境变量也可使用同名字段，分别接受 `true`/`false`、数值文本和整数文本；Python 显式配置
既支持这些顶层字段，也支持放在 `nnunet` 段中。
单病例的预处理原本实际已限制为 1 个进程；本次默认主要将导出进程池从 3 减为 1，
以减少主机内存和并发资源占用，不改变模型、TTA 或滑窗步长。
提速时先试 `disable_tta=true`、保持 `step_size=0.5`，再按效果测试 `0.75`。
这些调整可能影响分割及自动切割定位，应比较实际病例的耗时和切割结果。
分割缓存仍按模型 ID 复用；测试参数变化时使用新的 `-c` 目录，或清理对应的
`<cache>/<model_id>/segmentation/`，否则会跳过推理。

已有 PyInstaller 发布程序需要先重新执行 `dev build` 并更新完整应用包，才能支持新增的
上述字段。升级后直接修改外置 `src/config/nnuNet_config.json`，下一次任务启动生效，
无需再次打包；模型无需重新训练，外部 nnUNet 环境无需重新安装。

下面是模型目录配置项的空值示例；当前配置保留本机已有的路径，首次发布也会沿用：

```json
{
  "nnUNet_raw": "",
  "nnUNet_preprocessed": "",
  "nnUNet_results": ""
}
```

模型路径为空时，程序沿用系统已有的 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 环境变量。外部命令入口仍由 `nnUNet_predict_command` 指定。

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

切割部位相关配置统一维护在 `src/config/slice_part_config.json`。同一切割部位可以在
`slice_parts.<切割部位>.model_rules.<检查类型>` 中按原始扫描层厚配置不同模型。每项
使用包含边界的毫米区间：

```json
{
  "scan_thickness_range_mm": { "min": 0.0, "max": 1.5 },
  "model_id": 292
}
```

该文件同时管理切割部位是否启用、模型规则、定位区域、允许的状位与切法、范围扩展以及
定位图状位；
新增或调整切割部位时不再同步修改其他配置文件。`nnuNet_config.json` 只保存推理参数
和 nnUNet 路径。当前已启用的脊柱、头骨和髋骨按 `0-1.5mm`、`1.6-4.5mm`、`4.6-6mm`
分别选择原薄层、3mm、6mm 模型。肺部目前只有 `0-1.5mm` 的模型。外部厚度未命中
任何区间或同时命中重叠区间时，分割阶段会明确失败，不会自动选择临近模型。

统一切割部位配置在应用启动时严格校验。未知字段、重叠模型区间、未在 `label_aliases`
中定义的定位标签、未被切法引用的区域、错误的区域类型、非法范围扩展及非法定位图状位
都会直接阻止任务启动，不能依赖程序静默忽略配置拼写错误。

## 运行

先激活虚拟环境。

Linux / macOS：

```bash
source .venv/bin/activate
python main.py -i /path/to/dicom_dir -o /path/to/output_dir --slice_part 腰椎 --study_type CT --scan_thickness 3.0
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py -i D:\data\dicom_dir -o D:\data\output_dir --slice_part 腰椎 --study_type CT --scan_thickness 3.0
```

可以通过 `-c`（cache）指定持久化缓存根目录。第一阶段写入并复用 `-c/nifti/`，第二阶段
按模型 ID 写入并复用 `-c/<model_id>/segmentation/`。例如颈椎 CT 选中模型 `292` 时，
分割结果位于 `<cache>/292/segmentation/`；第三阶段及之后的所有产物（包括报告、调试/测试
文件和日志）仍按原逻辑写入 `-o/`：

```text
<output>/
└── ...                    # 第三阶段及之后的本次输出

<cache>/
├── nifti/                 # 第一阶段缓存，可复用
└── 292/
    └── segmentation/      # 第二阶段缓存，可复用
```

示例：

```bash
python main.py -i /path/to/dicom_dir -o /path/to/output_dir \
  -c /path/to/cache --slice_part 颈椎 \
  --study_type CT --scan_thickness 1.0
```

如需为本次任务覆盖窗宽窗位，可传入 `--window_presets` JSON 数组。列表中按
`Name` 匹配 `plane_configs` 使用的窗位名称；未覆盖的名称继续从
`src/config/window_presets.json` 查找。列表可以包含本次 `plane_configs` 未使用的
额外项；如果引用的名称在两个位置都找不到，任务失败并生成 `report.json`：

```bash
python main.py -i /path/to/dicom_dir -o /path/to/output_dir --scan_thickness 3.0 \
  --window_presets '[{"Name":"骨窗","windowWidth":50,"windowLevel":60}]'
```

命令行可识别的 `--slice_part` 值：

```text
头骨, 颈椎, 胸椎, 腰椎, 肺部, 骶骨, 髋骨
```

当前已启用：头骨、颈椎、胸椎、腰椎、肺部、髋骨；骶骨目前保留协议值但由预检拒绝。

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

默认 `error_log_file_enabled=true`，推理失败时 `log/ErrorLog.txt` 记录实际命令和完整
stdout/stderr，其余日志开关保持原值；显式关闭错误日志时仍遵守该配置。
报告摘要优先保留 stdout/stderr 中的具体异常。仅有 `Background workers died`
不能判断为内存不足，也可能是后台进程自身的 Python 异常；应先查看完整错误日志，
再结合系统的内存终止记录排查。将上述两个进程数设为 1 可减少并发资源占用，不能保证
解决所有后台进程退出。

对比两次批量执行的交付报告（会忽略不同结果目录造成的绝对路径差异）：

```bash
python scripts/compare_results.py \
  /home/peng/03_data/demo/Image/result/基准病例 \
  /home/peng/03_data/demo/Image/result/待比较病例
```

报告一致时退出码为 `0`；发现状态、部位、层厚、状位或切片数量差异时，输出差异路径并以
退出码 `1` 结束。

单 GPU 环境建议先从 `--jobs 2` 试起；如果 nnUNet 显存不足或出现 OOM，
降回 `--jobs 1`。

### 当前能力范围

每项 `plane_configs` 可选传入 `"FillPercent": 80`。同一计划的整组切片根据目标
标签整体范围统一居中；省略该字段、传 `null` 或 `0` 时仅整组居中，不放大也不缩小，
保留基础采样视野及像素间距。传入正数时整组等比例调整视野，让整体包围框的宽或高先达到指定
画面占比，另一方向不超过该占比。参照物是输出图像，不是屏幕，也不是查看器
的缩放百分比。合法范围为 `0 <= FillPercent <= 100`；0 表示不缩放，100 表示一方向贴边。
负数、布尔值、字符串、非有限值均无效。`line_extent_mode` 已移除，旧字段会报错。

例如：

```json
{"Plane":"横状位","SliceType":"椎间隙","SliceThickness":5,"SliceSpacing":5,"WindowPresets":["骨窗"],"FillPercent":80}
```

普通切片使用当前计划全部目标标签的投影包围范围，包含各层位置偏移，不仅是
面积最大的一层。椎间隙按每个间隙独立分组，使用相邻椎体完整标签（包含附件）；
CPR 使用该组全部曲面厚层标签在共同展开图坐标中的并集。组内各层及所有窗位
共用取景范围和缩放比例，不保证每张都精确居中或达到 80%。肺尖等末端小截面
保持自然较小，不会被单独放大。输出矩阵不变，两轴像素间距使用相同倍率；
平面组内 PixelSpacing、面内中心一致，只沿法向移动，层厚、层间距、数量和
切面方向不变。CPR 是展开图，不应视为原解剖结构的规则三维体。

无目标标签的厚层（例如椎间隙空白层、范围外扩层）仍沿用整组取景框，
重建结果 `framing` 标记为 `mode=group`、`source=group_target`。
整组没有可用目标标签时明确报错，不把背景当成目标。插值放大不会增加原始细节。
批量脚本默认显式传入 80%，不读取 `launch.json`；配置第六项填 `0` 或省略可关闭缩放。
取景处理不新增成片 PNG，仍输出 DICOM 与原有定位图（调试模式可输出定位图 PNG）；
旧文件需重新运行切割才会更新。

例如目标原本占图像 60%，设置 80 时放大至原尺寸的 `80/60` 倍；
原本占 95% 时设置 80 则会缩小。100 与 0 看起来接近，并不必然代表没有生效。
正式切片直接按实际选用的窗宽、窗位把重建 HU 映射为 uint16 灰度，不先取整 HU；
请求覆盖值同时决定像素显示和 DICOM 的 `WindowWidth`、`WindowCenter` 标签。
GDCM 写盘后设置 `RescaleSlope=窗宽/65535`、`RescaleIntercept=窗位-窗宽/2`、
`RescaleType=HU` 和 `VOILUTFunction=LINEAR_EXACT`，使查看器按相同窗口还原显示灰度。
同一计划各窗共用采样几何和未窗化 HU，分别生成各自灰度。
这是用于出片的派生显示 CT：窗内存在 16 位灰度量化，窗外数值已截断，不能恢复；
定量数据应读取原始 CT 序列。
`Localizer.dcm` 保留窗化背景和定位线，通过 Rescale 标签换算到对应窗口的显示数值，
同样写入实际窗宽窗位；其 `RescaleType=US`，属于合成显示数据，不能用于定量 HU 测量。
调试用 `Localizer.png` 保持原有显示效果。旧输出需重新运行切割才会使用这套窗位标签。
查看器若自动调对比度，对比时应固定为同一组实际窗宽窗位。
查看器的适配窗口和屏幕黑边不由 FillPercent 控制。

当前正式注册 CT 的横状位、矢状位和冠状位重建能力：

- 颈椎、胸椎、腰椎、肺部、头骨、髋骨三状位均支持基础 `MPR`；
- 每项 `plane_configs` 通过 `SliceType` 显式选择 `MPR`、`CPR` 或 `椎间隙`，
  未传时默认 `MPR`，系统不再按部位和状位自动改变切法；
- 颈椎、腰椎横状位额外支持 `椎间隙`；当前颈椎配置只生成 `C2_C3` 至
  `C6_C7`，不生成包含环椎 C1 的 `C1_C2`；
- 颈椎、胸椎、腰椎冠状位额外支持 `CPR`；
- 不支持的部位、状位和 `SliceType` 组合会在清理输出目录前由预检拒绝。

颈椎、腰椎椎间隙的画面左右、上下方向跟随原始横状位，保留各间隙的斜切角度。
切片仍按上位椎体到下位椎体排列；排列方向不决定画面上下。像素采样与 DICOM
方向、位置标签使用同一套显示坐标，已有反向成片需重新切割后更新。

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

`range_extension` 配置的固定扩展先计入有效范围，再执行上述计算。椎间隙按
每个有效间隙独立应用同一份配置，例如数量为 20、识别出 5 个有效间隙时，会生成
5 个重建区域、共 100 张物理切片；某个间隙缺失时只跳过该区域，不把它的数量分配
给其他间隙。
同一请求允许存在多份相同状位但采样参数不同的配置。所有配置按请求顺序获得全局
唯一的 `配置01`、`配置02` 等 `ConfigId`。重复状位按出现顺序使用 `横状位`、
`横状位1`、`横状位2` 这样的唯一输出名称，直接隔离各自目录；`report.json` 中各项
`Plane` 仍返回基础医学状位 `横状位`，具体目录由 `RegionList.Path` 返回。

`report.json` 是每次任务的固定完成标记，顶层返回布尔值 `Success`、文本 `Message`、
`SlicePart`、`StudyType`、数值类型的 `ScanThickness` 和 `PlaneList`。成功时
`Success=true`、`Message` 为空，并返回完整切片结果；参数解析、预检或任一处理阶段失败
时仍会原子写入该文件，`Success=false`、`Message` 为错误信息、`PlaneList` 为空。
每个成功状位配置保留调用方原始传入的 `SliceThickness`、`SliceSpacing`、
`SliceImageCount`，包括用来表示未配置的 `0`。实际输出按目录顺序组织为
`WindowPresetList -> RegionList`；每个重建区域使用唯一的 `Region`，并返回实际层厚、
实际层间距、实际数量和路径。普通 MPR/CPR 的区域标识为 `default`，椎间隙统一使用
`C2_C3`、`L1_L2` 等下划线标识。

上述保证覆盖 Python 能捕获的参数、预检、处理异常和主动中断。`kill -9`、原生库崩溃、
OOM 强制终止或整机掉电不会给进程执行写文件的机会；生产执行服务必须在进程外等待
Python 退出，发现 `report.json` 缺失或损坏时原子补写 `Success=false` 的兜底报告。
正式 DICOM 在临时文件中完成层厚、层间距校正后再原子发布，并统一使用
`ISO_IR 192` UTF-8 字符集，保证中文患者及检查信息可被标准读取器正确解码。

### 配置切片范围扩展和定位图状位

个别部位需要在分割边界之外额外保留影像时，可以在
`src/config/slice_part_config.json` 的 `slice_rules.<状位>.<切法>.range_extension`
中配置。`start_mm`、`end_mm` 固定按毫米解释：

```json
"横状位": {
  "MPR": {
    "regions": ["default"],
    "range_extension": { "start_mm": 3.0, "end_mm": 3.0 },
    "localizer_plane": "冠状位"
  }
}
```

`start_mm` 从原始第一张切片所在边界向外扩展，`end_mm` 从原始最后一张切片所在边界
向外扩展；方向跟随当前计划，因此同样适用于反向和斜向直线切片。两个字段必须显式
配置，使用 `0` 表示不扩展。配置距离先加入目标有效范围，之后再根据层厚、层间距和
数量计算最终总范围。椎间隙切法对 `regions` 中的每个间隙分别应用同一扩展值。

该能力不限制具体部位、切割方式或状位，可以在对应切法规则中按需设置。当前默认
为肺部和髋骨的横状位首尾各扩展 `3mm`；这只是当前配置，不是程序中的允许列表。

`localizer_plane` 为当前重建状位和切法指定定位图使用的原始正交状位，支持 `横状位`、
`矢状位`、`冠状位`，且不能与当前重建状位相同。每条切法规则都必须显式配置，MPR、
椎间隙和 CPR 统一读取该字段。当前肺部横状位使用冠状位定位图；其余默认值也都在
`slice_part_config.json` 中显式声明，不再由 planner 按部位写死。

肺部已经接入第一版解剖坐标计算：第五阶段从一个整肺分割 Mask 中保留一至两个肺主体，
排除远端小块误分割，并计算肺部物理中心和三轴覆盖范围；第六阶段据此生成以肺部为中心、
完整包住肺部分割范围并按配置保留首尾余量的三状位直线切片。肺部仍沿用 DICOM 标准
方向，不根据肺部形态附加倾斜角度。

脊柱定位和重建以整体范围配置为范围事实源。第四阶段按配置首尾标签展开全部中间椎体；
胸椎默认配置为 `C7 → L1`，显式请求 CPR 时 C7、T1–T12、L1 均进入过滤标注和冠状位 CPR。
腰椎同样按实际配置范围使用。过滤标注质心和 CPR 的排除项由
`spine_cpr.excluded_labels` 控制；当前颈椎配置排除 C1，C1 仍保留在全量定位和
切图覆盖范围。

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
写出完成后再进入报告阶段。正式切片按窗位分别生成灰度并写入实际显示标签，SimpleITK/GDCM 原生写盘使用
进程内互斥锁串行执行，避免多线程写 DICOM 触发内存竞态。正式切片写出后会校正
DICOM 的 `SliceThickness` 和 `SpacingBetweenSlices`，确保查看器读取到的层厚、
层间距与第六阶段计划一致。

### 输出目录安全

请求和环境预检通过后，程序会清理 `-o/` 中旧的业务结果，并重新执行第三阶段及后续阶段。
`-c/nifti/` 和 `-c/<model_id>/segmentation/` 属于持久化阶段缓存，不会被 `-o` 清理。
`report.json`、`manifest.json`（调试模式）、调试 NIfTI、调试元数据、`Localizer.png`、
测试文件和 `log/` 均继续写入 `-o/`。
只有 `debug_file_enabled=true` 时才生成并保留 `manifest.json`、调试 NIfTI、调试元数据、
`Localizer.png` 以及成功任务的中间现场。`debug_log_file_enabled` 只控制
`log/DebugLog.txt`，不再影响任何调试文件。`-c` 下的第一、二阶段缓存不会因输出清理删除。
metadata 中的调试 NIfTI 只保留第五阶段
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

当前 `nnuNet_config.json` 保留已有的模型目录。迁移到其他机器时按实际位置调整，
也可以将这些项设为空，再通过 `nnUNet_raw`、`nnUNet_preprocessed`、
`nnUNet_results` 环境变量配置模型目录。

程序启动后会先检查请求能力、输入目录、输出目录关系、nnUNet 命令和模型目录；
预检失败不会清理旧业务结果；入口仍会尽力写入失败 `report.json`，因此可能创建输出目录。

## 运行测试

Linux / macOS：

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
bash scripts/test.sh
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\test.ps1
```

测试脚本固定使用项目 `.venv`，统一运行 `pytest`，同时收集业务、外部命令与可执行发布的
回归测试。可在脚本后传入 pytest 参数，例如 `bash scripts/test.sh tests/test_frozen_runtime.py`。

## 外部 nnUNet 推理环境

`requirements.txt` 只安装本程序的基础影像处理依赖。nnUNet、PyTorch、CUDA 和模型由
现有外部推理环境提供，本程序通过子进程调用。外部命令或外部 Python 路径可在
`src/config/nnuNet_config.json` 中设置；详见 [发布部署手册](docs/production.md)。
开发和 CI 检查使用 `requirements-dev.txt`，其中额外包含 pytest 和 Ruff。
