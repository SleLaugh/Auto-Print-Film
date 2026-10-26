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
JSON 配置外置于产物的 `src/config/`，可直接编辑；每次构建默认同步当前源码的全部
JSON 配置，应用目录和压缩包都会包含。仅在需要沿用上次 `dist` 中的配置修改时使用
`dev build --keep-config`。
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

姿态回正使用基础依赖 `scipy>=1.10,<2`。已有虚拟环境也需重新执行
`python -m pip install -r requirements.txt`；使用发布程序时需重新执行 `dev build`
并更新完整应用包。已有分割缓存可继续复用，原有成片须重新切割才会采用新的方向。

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

Python 分割阶段会把每次推理的标准输出、标准错误实时保存到任务的
`log/nnunet-*.stdout.log`、`log/nnunet-*.stderr.log`，不依赖 DebugLog 开关。
原始进度不再同步转发到控制台或 DebugLog；监测循环直接读取原始日志检测致命崩溃，
并检查单次推理时限，避免控制台读取阻塞监测。需要实时进度时查看这两个日志文件。
外部 Python 及其 worker 启用 `PYTHONFAULTHANDLER=1`，发现段错误等致命崩溃日志后
中止本次推理；即使没有崩溃日志，也有按模型配置的单次推理超时兜底：

```json
{
  "inference_timeout_seconds": 1800,
  "inference_timeout_seconds_by_model": {"315": 180},
  "inference_max_retries": 1
}
```

时限单位为秒，必须为有限正数；模型覆盖项的键为模型 ID 字符串。仓库配置中肺部模型
`315` 每次最多 180 秒，其他模型默认 1800 秒，可根据实际数据调整。
Linux 下超时、原生崩溃或后台 worker 异常退出后，只有确认本次推理进程组已清理，
才会用全新的临时目录额外重试一次；`inference_max_retries=0` 可关闭重试。
缺少模型等明确配置错误直接失败，重试仍失败时写入失败报告。计时只覆盖单次外部推理，
不包括缓存锁等待和后续重建。C# 不参与超时或重试判断。
这些改动需要重新构建并更新应用包，之后调整 JSON 无需重新打包。

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
和 nnUNet 路径。当前已启用的脊柱、颅脑和髋骨按 `0-1.5mm`、`1.6-4.5mm`、`4.6-6mm`
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

同一原始序列的多个任务可共用一个 `-c`，同时写入各自不同的 `-o`。第一阶段缓存未命中时
由一个进程转换，其他进程等待并复查；已完成缓存只读复用。第二阶段按“缓存目录＋模型 ID”
单写：同模型首次推理只运行一次，等待者随后复用结果；不同模型使用独立锁，可分别推理。
第三至八阶段在各任务输出目录内并行执行。缓存锁文件和活动临时目录不要手动删除。

C# Worker 不再按原始序列限制整个任务串行，仍在同一临界区内检查全局
`MaxConcurrentCount`、检查规范化输出路径并启动登记进程。同一输出目录同时只允许一个
任务，不同输出目录的同序列任务可以并发；并发数不会自动根据 GPU 显存调整。
直接使用 CLI 时也须保证各任务的 `-o` 不重叠。推理时限和重试仍由 Python 负责，C# 不增加超时判断。

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
颅脑, 颈椎, 胸椎, 腰椎, 脊柱, 肺部, 骶骨, 髋骨
```

当前已启用：颅脑、颈椎、胸椎、腰椎、脊柱、肺部、髋骨；骶骨目前保留协议值但由预检拒绝。

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

默认 `error_log_file_enabled=true`，推理失败时 `log/ErrorLog.txt` 记录实际命令和
stdout/stderr 各自末尾最多 64 KiB，其余日志开关保持原值；显式关闭错误日志时仍遵守该配置。
完整原始输出实时保存在 `log/nnunet-*.stdout.log` 和 `log/nnunet-*.stderr.log`，
每次尝试使用独立文件名，失败与重试日志均保留，不依赖上述日志开关。
报告摘要优先保留 stdout/stderr 中的具体异常。仅有 `Background workers died`
不能判断为内存不足，也可能是后台进程自身的 Python 异常；应收集本次尝试的两份原始推理日志，
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

命中分割缓存后，空间校正、定位和重建仍会执行 CPU 运算。源码 CLI 和发布程序在
加载影像库前，默认将每次 ITK 滤镜运算的线程数限制为 `min(4, CPU 亲和性允许的逻辑 CPU 数)`，
OMP、OpenBLAS、MKL、NumExpr 默认各为 1；已有的对应环境变量会保留。
窗位线程池默认上限为 2。需要调整时，在启动任务前设置：

```bash
AUTOPRINTFILM_NATIVE_THREADS=4 AUTOPRINTFILM_WINDOW_WORKERS=1 \
  bash scripts/batch_test_plan.sh --jobs 3
```

`AUTOPRINTFILM_NATIVE_THREADS` 调整未显式配置的 ITK 线程数；例如已经设置
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` 时，仍以该原生环境变量为准。
`AUTOPRINTFILM_WINDOW_WORKERS` 调整窗位线程池上限；Python 显式构造
`ReconstructionStage` 时的 `max_window_workers` 优先。两项应用环境变量均须为正整数。
这些限制分别控制 ITK 运算和窗位任务，并非整个进程的总线程硬上限；窗位任务内仍可能
调用 ITK，主线程及库的辅助线程也会计入系统线程数。限制不能代替外部任务并发数和
内存限制，也不改变分割缓存。非法应用线程配置会终止任务，并在指定的 `-o` 下尽力写入
失败 `report.json`。无法读取 CPU 亲和性的平台使用系统逻辑 CPU 数计算默认值。
子进程会继承这些环境变量，因此未命中缓存时，外部 nnUNet 的 OMP/BLAS 等线程默认值
也会受影响，除非其启动器自行覆盖；nnUNet 的 `-npp`、`-nps` 进程数仍由原配置控制。
普通 `import main` 不修改调用方的线程环境；嵌入式调用应在加载 NumPy/SimpleITK 前
显式调用 `src.runtime.bootstrap_native_threads()`，或由宿主设置原生线程变量。

仓库日志配置已开启 `run_log_file_enabled`，将诊断信息保存到每个任务的
`log/RunLog.txt`；发布环境保留的日志开关仍以外置配置为准。
`RunLog` 的 `[资源]` 行记录每个阶段开始、完成或失败时的进程资源快照。Linux 下包含
PID、当前 RSS、进程整个运行期间的峰值 RSS、线程数、系统可用内存和已用 swap，
内存单位为 MiB。第三阶段另外记录带 `step=` 的读取分割、椎体序列核实、逐标签主体提取、
标签索引、目标 bbox、镜像候选估计，以及原图/分割旋转与写盘的开始、完成或失败。
完成行包含耗时；读取分割后记录图像尺寸、像素类型和实际 ITK 线程数。这些信息通过
RunLog 保存，不需要开启调试影像或 DebugLog。快照不是阶段内部的持续采样，不能仅凭
缺少 OOM 日志或一次低内存读数排除系统问题。已有发布包需重新构建后才包含这些改动。
桌面和服务一起卡住时，还应保留系统 journal 与 `/var/lib/systemd/pstore/` 中的内核转储，
详见[系统卡死排查](docs/production.md#系统卡死时保留内核证据)。

### 当前能力范围

每项 `plane_configs` 可选传入 `"FillPercent": 80`，合法范围为 `0 <= FillPercent <= 100`。
省略、null 或0时保留整组目标中心与基础像素间距，不缩放；肺/髋普通 MPR 必要时
扩大共同矩阵保护完整源切面。负数、布尔值、字符串和非有限值无效，旧 `line_extent_mode` 已移除。
占比的参照物是输出图像，不是屏幕或查看器缩放百分比，100也不表示原始倍率。

肺部、髋骨的三个普通MPR状位统一使用所选原标签整组包络中心，整组固定面内中心、
倍率与矩阵；末层小标签与空目标层也沿用，不选择肺参考层、不逐层追随髋部目标。
共同目标尺寸为整组标签物理投影包络加每侧 `5 mm`，正数占比不再以完整人体限制放大。
100完整容纳所选标签与余量，框外皮肤、肋骨、腹部、大腿或骶尾结构允许越框。

这六条MPR规则统一配置：

```json
{"basis":"selected_region","margin_mm":5.0,"canvas":"original"}
```

`centering`省略即`group`。`canvas=original`保留第六阶段原画框的Rows、Columns与基础
像素纵横比，按该画框的物理宽高比等比适配部位，不再按目标包络把输出画布收紧成窄条。
第六阶段在原图尺寸可用时已用完整源CT中心角点投影建立基础画框；这里保留的是该画框比例，
中心仍为部位整组包络中点，不复制原CT中心，也不以完整源范围限制放大。

设目标尺寸为T、基础矩阵为N、基础间距为s，正数占比q=FillPercent/100：
共同缩放因子为 `f=max(T/(N*s))/q`，实际间距为`s*f`，物理视野为`N*s*f`。
100的实际视野是在原画框比例内完整容纳T的矩形，不能把两轴视野都写成T/q。
70、50保持同一组中心与矩阵，宽高分别同时扩为100的1/0.7、2倍，向四周扩展；
80同样为1.25倍。所有层直接从完整原CT采样更大范围，不缩小预裁图或补黑框。

整组共同倍率保留各层真实尺寸，肺底少量肺组织不单独放大，髋部横状位也不逐层跳位。
冠状位等小截面或原画框比例与部位比例不同时仍可有空气，100不承诺零空气或任意胶片格满铺。
不拉伸解剖、不裁所选部位；视野外的外围组织可随占比降低重新显示，扫描外缺失组织不能恢复。

配置对象不是请求字段。可选`centering`仍接受`group`、`reference_slice`、`slice_target`；
可选`canvas`接受`target`、`original`，省略为旧`target`紧致画布。参考层/逐层目标中心及
紧致画布仅保留显式兼容方式，不属于这六条默认规则。0/null/省略、颅脑、颈椎、胸椎、
腰椎、脊柱、CPR及椎间隙保持原行为。无策略旧肺/髋计划仍可走人体保护/skin_edge兼容路径。

其余部位正数普通 MPR 仍以原整组投影范围结合厚层内原目标与有界显示上下文确定共同倍率，
逐层按联合 bbox 居中；切割范围仍由 `source_labels` 决定。脊柱上下文从本次模型标签表读取
同类椎体，限制在原目标患者 LPS 头脚物理外缘内，不混入无关器官或未显式选中的 `sacrum`；
0/null/省略不应用这一正数上下文。颅脑仍保留原有脑区及有界 skull 取景，各占比均生效。

椎间隙保持同一部位、同一项状位配置共用裁空前矩阵与物理比例。共同矩阵取末组之前
各组基础矩阵的逐轴中位数，单组用自身；共同尺寸覆盖原投影与逐层联合范围。
可靠时沿面内 LR 对齐对称面，预先容纳新中心到两侧边界的最大距离；证据不足时用 bbox 中心。
0/null/省略仍统一矩阵并保留基础间距。CPR沿原曲面展开有界上下文与目标，厚层掩膜并集
决定共同倍率、当前展开掩膜决定中心；空层回退组中心，路径、offset和厚层采样不变。
真实解剖大小差异保留，不把大小不同的椎体强行拉成一样大。

两轴像素间距始终使用同一倍率，不拉伸解剖。肺/髋正数普通MPR保留原共同矩阵和画框比例，
不做逐张上下 ROI。其余适用平面和椎间隙在 HU 采样后、窗化前，仅裁首尾完全无原 CT 支持的行，
包括0/null/省略；真实空气与有效内容保留，不按黑色裁剪、不取全组最短高度、不补框。
裁后 Rows和面内中心可逐层不同，Columns、PixelSpacing及采样相位保持，同层各窗几何相同。
CPR和Localizer不裁空。整张无源支持层保留并记录 `no_source_support`。

`plane_stack`厚层平均只纳入有效源采样：轴对齐按有效重叠厚度、斜切按有效子平面数归一。
真实空气参与，越界填充值不参与；零支持仍为 `-1024 HU`，不补全扫描外组织。
它不能消除斜切首末层图内完全超出原 CT 的局部黑区；完整横截面需要另行调整范围或方向。

`framing.selected_region`记录`basis`、`margin_mm`、`centering=group`、`canvas=original`及原标签未加余量边界。
肺/髋全部普通MPR的`center_source=group_selected_region`，包含小标签与空目标层；没有默认参考层字段。`framing_target_size_mm` 为共同标签覆盖尺寸加余量，
`slice_target_size_mm` 仅记录当前厚层尺寸；当前正数肺/髋策略不产生 `source_preservation`、
`skin_edge` 或自动身体支持字段。0/null/省略的肺/髋保源诊断仍为 `complete_source_slab`。
DICOM写出真实中心、矩阵、间距与方向；其他模式的逐层居中或上下裁空可引起面内位移，
读取方应尊重逐张几何。CPR是展开图，不能承诺恢复原解剖三维体或与原 CT 精确三维联动。

FillPercent只影响正式切片，Localizer保留固定参考点、方向和首末像素的物理范围。粗显示轴
在绘线前仅对二维背景做线性细化，再在新显示网格上重画约一像素宽的抗锯齿线：轴间距
大于最细显示轴的两倍或大于2mm时，目标间距为最细显示轴间距与1mm的较小值。
这不增加解剖细节，也不重采样整个三维体。MPR白线由第六阶段真实
无限平面求交，仅裁到底图边界或曲面显示网格，不用正式成片视野裁线；切换占比不改变
定位图矩阵、取样、参考点或交线端点与长度。CPR定位线仍投影实际offset中心曲线。

正式切片按未取整的重建HU与实际窗宽窗位映射uint16灰度，各窗共用HU与几何。
GDCM写盘后设置 `RescaleSlope=窗宽/65535`、`RescaleIntercept=窗位-窗宽/2`、
`RescaleType=HU`、`VOILUTFunction=LINEAR_EXACT`；请求覆盖值同时决定像素和窗位标签。
这是出片用派生显示CT，窗内有量化、窗外已截断，定量应读取原始CT。
`Localizer.dcm`是窗化背景加定位线的合成显示数据，`RescaleType=US`，不能用于定量HU测量。
查看器应固定相同实际窗位比较；fit设置和胶片格比例仍可产生屏幕留白，后端不保证任意格子四边铺满。

详细公式、字段和兼容工具见[整组取景设计](docs/superpowers/modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md)。
批量脚本默认显式传入80%，不读取 `launch.json`；第六项填0或省略可关闭缩放。
旧输出需要重新切割才会更新；不新增正式成片PNG，调试定位图PNG保留。

当前正式注册 CT 的横状位、矢状位和冠状位重建能力：

- 颈椎、胸椎、腰椎、脊柱、肺部、颅脑、髋骨三状位均支持基础 `MPR`；
- 每项 `plane_configs` 通过 `SliceType` 显式选择 `MPR`、`CPR` 或 `椎间隙`，
  未传时默认 `MPR`，系统不再按部位和状位自动改变切法；
- 颈椎、腰椎横状位额外支持 `椎间隙`；当前颈椎配置只生成 `C2_C3` 至
  `C6_C7`，不生成包含环椎 C1 的 `C1_C2`；
- 颈椎、胸椎、腰椎、脊柱的冠状位、矢状位额外支持 `CPR`；
- 不支持的部位、状位和 `SliceType` 组合会在清理输出目录前由预检拒绝。

`SlicePart=脊柱` 使用 `C7`–`S1` 的整体范围，包含 C7、T1–T12、L1–L5、S1，
三状位 MPR 与冠状位、矢状位 CPR 共用该范围，复用现有脊柱模型和重建算法。
它是独立切割部位，不改变颈椎、胸椎和腰椎各自的配置范围。

当前启用部位的整切 MPR 三状位共用第五阶段的解剖坐标。颅脑、髋骨和可靠双侧肺
使用左右镜像对称面，颈椎、胸椎、腰椎、脊柱汇聚可靠逐椎体左右方向。颅脑在左右轴确定后，
默认从完整 `brain` 的物理体素在 AP/SI 平面内估计形态主轴，只绕左右轴调整前后倾角，
三个状位共同使用更新后的坐标；这是脑形态斜切，不代表听眦线或 AC-PC 对齐。
倾角逐病例计算，不设绝对角度上下限；是否采用由脑区完整性、形态稳定性和方向一致性判断。
证据不足时保留原 AP 投影并记录回退原因。脊柱、肺部、髋骨的普通整切 MPR
保留输入前后倾角，不额外抬头、低头：
AP 取输入 AP 基准在 LR 正交平面内的投影，SI 由 LR 与 AP 叉积构造，
不通过首尾质心连线把脊柱整体摆竖直。
肺本体镜像仅因形态质量不足被拒绝时，可使用本次模型明确提供的 `pleural effusion`
标签补充胸膜腔轮廓，再执行相同的镜像与双侧质量检查；不纳入心包或未知标签，
不放宽门槛，切割中心和范围仍由原肺主体决定。缺少辅助标签或证据仍不足时继续回退。
质量不足时记录警告并保留患者参考三轴。回正采用刚性采样，不强行镜像或变形组织，
也不保证左右解剖完全对称。详见[统一姿态设计](docs/superpowers/modules/reconstruction/2026-10-16-统一解剖姿态与三状位回正设计.md)。

姿态参考由 `patient_reference_axes` 统一消除存储轴的正负歧义，头侧按患者 LPS
`+Z` 确定；冠状、矢状成片头侧向上，不随单个存储轴反向而倒置。该参考只校正
方向符号，保留已有斜扫描角度；体素到物理点的换算仍使用原始 `direction`、
`origin` 和 `spacing`。头脚符号校准本身不改变俯仰；颅脑形态斜切是第五阶段另行进行的方向估计。
所有正式矢状位 MPR、CPR 的阅读方向一致：左到右为患者前到后，上到下为头到脚；
脊柱图像中椎体在左、后方棘突在右。CPR 的显示方向不改变曲面堆叠位置和切片顺序。

颅脑的方向与三状位基础范围都由 `brain` 决定，骨标签不会延长层组或增加面部、下颌切片。
横状位 MPR 当前通过 `range_extension` 默认向头侧补 `10 mm`、下侧不补，改善脑区边界
与骨性颅顶之间的覆盖差；冠状位、矢状位没有这项默认余量。这是工程余量，不保证
所有病例均完整包含颅顶，也不能补出源扫描缺失的组织。
第七阶段可从同次模型 `dataset.json` 解析真实 `skull`，只保留沿最终头脚轴与脑区物理
支撑范围有正重叠的骨体素；横状位计划的取景上界同步增加头侧配置余量，下界仍停在
脑区范围，辅助脑部取景、居中和防裁边。缺少可用骨标签时
仅用脑区取景并告警；脑区接触源图像边界时报告截断风险，不补出扫描外组织。
定位图 CT 背景仍可能显示脸和下颌，定位线按实际切面绘制，不能缩短线条伪装切片范围。
Localizer 保留固定参考点、方向和首末像素物理范围；粗网格按通用定位图规则细化，
不随 FillPercent 缩放、裁切或移动。MPR 白线
按第六阶段真实平面横穿底图，仅裁到底图边界，长度不随正式切片取景变化。
定位线表示切面位置与方向，不表示正式切片的有限视野。
开启调试文件后，可在
`metadata/localization_result.json` 的 `anatomical_geometry_diagnostics.pitch` 查看采用状态、
角度和回退原因，并结合 `slice_plans.json`、`Localizer.dcm` 复核真实切面。
原有 `--slice_part 颅脑` 和状位参数可直接使用，无需新增配置；已有分割缓存可复用，
成片需重新切割。方法、门槛与限制见[颅脑形态斜切设计](docs/superpowers/modules/reconstruction/2026-10-20-颅脑形态斜切设计.md)。

镜像拟合还检查左右方向是否唯一：涉及 LR 的近重根子空间、圆柱类目标缺少角向
信息、许可角度内存在等价对称面时会拒绝姿态更新。仅 AP/SI 特征值相近而 LR
独立时，不因此拒绝可靠左右方向。

颈椎、腰椎椎间隙使用相邻椎体可靠的局部左右方向处理侧倾，同时保留各间隙的
斜切位置；质量不足时明确回退。切片仍按上位椎体到下位椎体排列；排列方向不决定
画面上下。像素采样与 DICOM 方向、位置标签使用同一套实际坐标。

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
`Success=true`，返回实际生成的切片结果；全部请求区间有效时 `Message` 为空，
部分脊柱椎间区间因标签或几何不足被跳过时，`Message` 说明跳过的区间及原因。
参数解析、预检或任一处理阶段失败
时仍会原子写入该文件，`Success=false`、`Message` 为错误信息、`PlaneList` 为空。
每个成功状位配置保留调用方原始传入的 `SliceThickness`、`SliceSpacing`、
`SliceImageCount`，包括用来表示未配置的 `0`。实际输出按目录顺序组织为
`WindowPresetList -> RegionList`；每个重建区域使用唯一的 `Region`，并返回实际层厚、
实际层间距、实际数量和路径。普通 MPR/CPR 的区域标识为 `default`，椎间隙统一使用
`C2_C3`、`L1_L2` 等下划线标识。
`ActualSliceThickness`、`ActualSliceSpacing` 以数值类型保留两位小数；请求值、内部
采样计算、调试记录及 DICOM 参数保留原精度。

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
为肺部和髋骨的横状位首尾各扩展 `3mm`；颅脑横状位 MPR 为 `start_mm=10.0`、
`end_mm=0.0`，沿当前计划头侧补 `10mm`，下侧不补。颅脑冠、矢状位不加这项余量；
这些是当前配置，不是程序中的允许列表。
覆盖应看首末厚层的外缘；Localizer 的定位线表示切片中心面，首线距离覆盖边界
通常仍有半层厚，不能仅凭中心线位置判断漏切。`10mm` 是当前工程余量，不保证所有
病例颅顶完整，也不补全源扫描截断。

`localizer_plane` 为当前重建状位和切法指定定位图的参考显示状位，支持 `横状位`、
`矢状位`、`冠状位`，且不能与当前重建状位相同。MPR、椎间隙和 CPR 的规则都必须显式
配置并统一读取该字段。
当前肺部横状位使用冠状位定位图；其余默认值也都在
`slice_part_config.json` 中显式声明，不再由 planner 按部位写死。

第五阶段从整肺分割 Mask 中保留一至两个肺主体并排除远端小块。两个连通主体只是
必要条件：先通过镜像质量检查，再确认两主体分居候选对称面两侧、沿 LR 充分分离，
且反射后的近邻主要跨主体对应，才采用左右姿态。单侧、连为一体、同侧碎块或上述
证据不足时警告并保留患者语义参考方向。中心和三轴覆盖范围始终来自保留的完整
主体，第六阶段在该范围之外继续应用配置的首尾余量。

脊柱定位和重建以整体范围配置为范围事实源。第四阶段按配置首尾标签展开全部中间椎体；
当前整体默认范围为颈椎 `C1 → T1`、胸椎 `C7 → L1`、腰椎 `T12 → S1`、脊柱 `C7 → S1`，
供三状位 MPR 和冠状位、矢状位 CPR 共用；颈椎 CPR 排除 C1 后使用 C2–T1。
颈椎仅横状位 MPR 通过 `range_extension` 在终点侧另向外扩 `5 mm`，起点不扩；
冠、矢状位 MPR 保持 `0/0 mm`，不把整体终点改为 T2，也不保证扩展范围超出源 CT 后仍有数据。
显式请求胸椎 CPR 时 C7、T1–T12、L1
均进入过滤标注和所请求状位的 CPR。整体范围扩展不新增椎间隙，颈椎仍为 `C2_C3`
至 `C6_C7` 五组，腰椎仍为 `L1_L2` 至 `L5_S1` 五组。过滤标注质心和 CPR 的排除项由
`spine_cpr.excluded_labels` 控制；当前颈椎配置排除 C1，C1 仍保留在全量定位和
切图覆盖范围。

S1 在运行时按本次实际 `label_targets` 解析，优先 `vertebrae_S1`，模型只提供
`sacrum` 时回退到它；不改写标准范围配置。整体坐标端点、诊断、椎间主体配对和
`L5_S1` 计划的 `source_labels` 使用实际标签名，没有这两种标签时沿用缺标签校验。

颈椎、胸椎、腰椎、脊柱第五阶段用可靠逐椎体 LR 法向与输入 AP 投影构造整体正交坐标，
三个普通整切状位都保留输入前后倾角。配置首尾质心仅用于范围有效性检查和诊断，
不再决定全局 SI；没有可靠 LR 时保留患者参考三轴并警告，`applied` 反映实际姿态变化。
局部椎间隙仍依真实相向端面及相邻 LR 确定斜切，CPR 的曲线几何不受该规则替换。
各脊柱部位冠状位与矢状位均按 `SliceType` 选择切法：省略或传入 `MPR` 时生成普通
直线 `plane_stack`；显式传入 `CPR` 时使用过滤标注/四棱柱质心生成 `curved_stack`。
冠状位 CPR 沿前后方向偏移、横向覆盖左右，使用矢状位定位图；矢状位 CPR 沿左右方向
偏移、横向覆盖前后，使用冠状位定位图。两者分别读取各自状位的 CPR 范围扩展规则。
矢状位 CPR 只在冠状面展开左右侧弯，保留矢状面原有的前后生理曲度；采样中心线
投影到整组固定的前后参考面，横轴保持患者前后方向，纵向使用冠状投影弧长。
因此保留前后弯曲幅度，但不承诺每个结构的原图头脚毫米位置完全不变。
内部计划用 `curve_projection_axis` 标记这一采样方式，真实三维 `curve_points` 仍保留给
定位背景；缺少该字段的旧计划保持原三维展开行为，需要重新规划、重建才能应用修复。
CPR 首尾端点只在反弧时回正，并沿当前方向延伸到原始首末标签的实际出口。
平滑后再用终端原始标签外缘兜底；矢状位按冠状投影切向计算所需覆盖距离，并
换算回真实三维切线延伸量，保证投影端点不漏裁。缺少原始标签时不按 bbox 补长。

各脊柱部位矢状位 MPR、CPR 使用冠状定位背景时，先在原始 x/z 显示网格按脊柱
前后深度 y(z) 逐行采样，再按通用定位图规则对粗显示轴做二维线性细化，保留物理路径映射。
这改善单层背景无法同时穿过弯曲椎体的问题；不使用 MIP。
CPR 深度来自真实计划曲线，MPR 来自第五阶段有效四棱柱/主体质心，几何不足时保留
原单层背景。MPR 定位线直接按第六阶段计划的真实平面与该曲面求交，仅裁到底图
显示网格边界；不使用正式成片的取景矩形。切换 FillPercent 不改变定位图背景的矩阵、
像素间距、参考点、取样或定位线长度；有效交线按真实方向横穿底图。CPR 定位线投影实际
采样的 offset 中心曲线。此优化只影响定位图；正式切片的范围、采样、偏移、层厚、间距和数量不变。
曲面定位图的 DICOM 会标明派生曲面采样；空间标签仅描述显示参考网格，不能将其
当作普通原始冠状平面做精确三维联动。其余定位背景及腰椎横位椎间隙共享参考层保持原规则。

髋骨整切使用左右独立分割：第五阶段分别保留 `hip_left`、`hip_right` 的最大连通
主体，以双侧主体边界的镜像对称面确定左右轴；顶部外侧两点 AB 仅保留诊断用途。
三个状位都使用同一解剖坐标，各方向覆盖边界重新投影完整主体确定，不以拟合采样点
或 AB 代替完整范围。左右重叠、严重单侧缺损或镜像证据不足时警告并保留输入方向。
三个状位的 `Localizer.dcm` 定位图均选取经过髋骨整体中心的原始层，其中横状位和矢状位
使用冠状定位背景，冠状位使用矢状定位背景。

未传 `--plane_configs` 时默认请求横状位骨窗 MPR；需要矢状位、冠状位或专用
`SliceType` 时应显式传入 `--plane_configs`。MR 和骶骨会在处理输出前由预检明确拒绝。

颅脑、髋骨、肺部在第三阶段不再做整卷 XY 回正，方向由第五阶段确定后在第七阶段
采样时应用，避免重复旋转。脊柱第三阶段使用可靠逐椎体镜像角汇聚生成 XY 候选；
候选提升后，定位和解剖分析均在新的统一工作空间重新计算。

腰椎横状位区间切优先从第五阶段四棱柱矢状四边形中选择相向端边，并在两条端边共同
切向覆盖区间内取得代表性边界点对；代表点连线作为每段斜横切法向，平面内覆盖仍使用
相邻全量 label bbox 的并集。这样既避免最近点落到角点形成锐角，也避免
横突、棘突等附件使全量 bbox 重叠而跳过 `L1_L2` 至 `L5_S1` 区间，并保持切线在真实
椎体间隙而不是整幅图像中心；输出平面会围绕该位置对称扩展，仍完整覆盖原选定范围。
S1 使用独立的左右支撑筛选，并在主体上缘局部补回一层与上缘相接的连通体素，不再套用
普通腰椎主体裁剪。四边形缺少时回退 faces/bbox；如果
回退路径中的 bbox 仍重叠，对应区间会跳过并写入 `slice_plans.json` 的 `warnings`。
若相邻主体体素量相差超过四倍、四棱柱前后范围无交集，且候选法向偏离输入头脚轴
超过 60°，则将其识别为小残块侧壁形成的伪间隙，直接跳过，不再用同一坏标签的
bbox 或质心补造切面。其他有效间隙继续输出，跳过原因写入交付报告的 `Message`；
这不恢复原分割中缺失的椎体主体。
腰椎矢状位整切及其他整切流程不受该回退限制影响。

颈、胸、腰椎及整段脊柱共用第五阶段的主体与四棱柱提取。普通椎体按中央连续 LR
宽度枚举候选，以 AP 深度、体量和 SI 连续支撑筛选主体，避免提前按原始连通块大小
删掉主体、后方分叉撑大宽度阈值或前方薄突起抢占主体。四边形外扩失败时，使用完整
凸包的最小面积包围矩形并保留原始边界；S1 继续使用独立的骶骨规则。
AP/LR 形态比例按物理尺寸比较，主体选择不应因同一结构的非等距采样而换到后方附件。
普通椎体还会在原有裁剪结束后校验连续占据密度：首末跨度很大、实际占据和连续
支撑都很少的外缘薄片不再当作全高主体。正常短缺口保留，空层不填充；密度验证
必须保留可信核心，也不会放宽原裁剪。这一步使用分割占据情况，不依赖 HU 恒定。

若上述主体裁剪明显漏掉原标签内的稳定三维核心，会按物理尺度的中央 LR 占据支撑
有限恢复主体；C2 齿突和 S1 仍走各自规则。恢复体及其相邻端面使用原标签多个 LR
切面的连续边界共识确定方向，覆盖四棱柱仍用于包围。端面不能来自恢复窗口裁出的
平面；还会对候选边界内外多个物理距离窗验证相对 CT 密度下降，排除落在骨内的
错误平台，不设固定 HU 门槛。端面证据不足时保留四边形回退；其他正常椎间配对
继续沿用原规划。CT 只在出现主体恢复时延迟读取一次。

腰椎若出现单个前方薄帽标签、其上下主体只相隔一个正常椎体步长，并且 CT 证实
只有一个椎间盘谷，第三阶段可按可靠 S1/L5/L4 锚点修正必要的前方主体编号。
原始缓存不改写，派生分割保存到输出 `nifti/seg_spine_sequence_corrected.nii.gz`，
后续定位、覆盖范围、CPR 和椎间切统一读取它。真实缺体、独立薄椎体、上方证据不足
或存在多个 CT 谷时不重编号；正常序列提前返回，不额外读取 CT。
可靠三维主体先固定编号，剩余边缘按三维物理距离关联，避免侧倾时的矢状投影重叠
导致椎体串号。薄帽必须与相邻椎体的前方主体有真实三维接触、归属明确，并完整处于
受限前体走廊内，才能整块归并；没有可信薄帽时整次编号修正回退，保留原分割。

当前椎体主体标注仍属于基于分割形态和支撑阈值的工程启发式。在椎体倾斜明显、
退变、骨赘、附件连接复杂或原始分割缺损的病例上，主体上缘/下缘仍可能出现少量欠分
或过分，并进一步传递到四棱柱边界和椎间切线位置。现阶段“相向端边代表点确定位置、
代表点连线确定法向”的总体方向已经可用，但主体标注稳定性仍需通过更多病例继续优化，
不应把当前输出视为已经完成临床级泛化验证。

第七阶段对每个切片计划先完成全部 MPR HU 图像采样，再按窗位使用线程池处理整套图像。
不同窗位不会重复执行 MPR，默认最多同时运行 2 个窗位线程；第七阶段会等待所有窗位
任务完成。窗位计算可能调用原生库，2 个 worker 不代表整个进程只有 2 个线程。
全部窗位写出完成后再进入报告阶段。正式切片按窗位分别生成灰度并写入实际显示标签，SimpleITK/GDCM 原生写盘使用
进程内互斥锁串行执行，避免多线程写 DICOM 触发内存竞态。正式切片写出后会校正
DICOM 的 `SliceThickness` 和 `SpacingBetweenSlices`，确保查看器读取到的层厚、
层间距与第六阶段计划一致。

### 输出目录安全

请求和环境预检通过后，程序会清理 `-o/` 中旧的业务结果，并重新执行第三阶段及后续阶段。
同一输出目录不可同时执行多个任务；C# Worker 按规范化后的输出路径互斥，CLI 调用方
须自行保证目录独占。同序列任务使用不同输出目录时可并行，共享的第一、二阶段缓存仍受各自写锁保护。
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
