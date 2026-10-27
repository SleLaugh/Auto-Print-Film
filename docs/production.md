# AutoPrintFilm 发布部署手册

肺部、髋骨三状位普通 MPR 默认按所选标签取景，每侧余量 `5 mm`。
肺部、髋骨全部普通MPR状位统一使用原标签整组包络中心，整组固定中心、倍率与矩阵。
保留第六阶段原CT投影画框及物理宽高比，100容纳所选标签+余量；70、50同心向四周扩展视野。
0/null/省略、其他部位、CPR、椎间隙与定位图保留原规则。详见[整组取景设计](superpowers/modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md)。

## 1. 构建可执行程序

在项目根目录执行：

```bash
dev build
```

本机已注册 `dev`。其他机器克隆项目后可用 `./dev build`；要省略 `./`，先注册一次：

```bash
./dev install-command
```

命令安装到当前用户的 `~/.local/bin/dev`，不覆盖其他已有的 `dev`；该目录须在 PATH 中。

构建工具在项目 `.build-cache/` 中准备固定 Python 和基础依赖，用 PyInstaller **onedir** 打包成可执行程序，验证后输出到 `dist/`。首次需要联网，后续复用缓存。生产端直接运行程序，不需要安装应用 Python 或创建 venv。

nnUNet、PyTorch、CUDA 和模型继续使用现有外部环境，不打入应用包，也不由 `dev build` 安装或修改。构建不修改服务数据库、生产目录、外部推理环境或病例数据。

## 2. 发布产物

```text
项目根目录/dist/
├── AutoPrintFilm/
│   ├── AutoPrintFilm               # 直接运行的 Linux 可执行程序
│   ├── _internal/                 # 程序依赖库，必须一同保留
│   ├── src/
│   │   └── config/                # 外置 JSON，可直接编辑
│   │       ├── nnuNet_config.json
│   │       ├── slice_part_config.json
│   │       ├── window_presets.json
│   │       └── logger_config.json
│   ├── PORTABLE.json              # 打包方式和构建平台信息
│   ├── RELEASE.json               # 发布文件校验记录
│   ├── requirements.lock.txt      # 实际构建依赖版本
│   └── docs/production.md
├── AutoPrintFilm-linux-x64.tar.gz
└── AutoPrintFilm-linux-x64.tar.gz.sha256
```

应用代码已打入程序，不再交付 `main.py`、`src/` 下的应用 `.py` 文件或独立 `python/` 目录。开发 `.venv`、模型和病例数据也不在包中。

当前目标为 **Linux x86_64，构建基线 Ubuntu 22.04**。其他 Linux 机器需验证系统库和 ABI 兼容性；这不是 Windows/macOS 包，也不承诺仅满足某个 glibc 最低版本就能运行全部影像处理库。

应用目录可整体移动，须保留可执行程序、`_internal/`、配置和相对符号链接。不要只复制可执行文件；复制目录用 `cp -a`，或使用本工具生成的 tar 包。[PyInstaller onedir 的符号链接要求](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#requirements-imposed-by-symbolic-links-in-frozen-application)

## 3. 解压到生产目录

将 `.tar.gz` 和 `.sha256` 一起复制到服务器，在它们所在目录执行：

```bash
sha256sum -c AutoPrintFilm-linux-x64.tar.gz.sha256
```

当前服务的程序与配置路径为：

```text
程序：/home/peng/03_data/demo/Project/发布/AutoSliceServer/AutoPrintFilm
配置：/home/peng/03_data/demo/Project/发布/AutoSliceServer/src/config/*.json
```

应用目录的位置可自行选择。以下使用 `/home/peng/03_data/production/AutoPrintFilm` 演示
首次部署，不要求将已经可用的服务移到该目录。确认校验输出 `OK` 后执行：

```bash
test ! -e /home/peng/03_data/production/AutoPrintFilm && \
mkdir -p /home/peng/03_data/production && \
tar -xzf AutoPrintFilm-linux-x64.tar.gz -C /home/peng/03_data/production && \
/home/peng/03_data/production/AutoPrintFilm/AutoPrintFilm --help
```

最后一条只查看帮助，不检查外部模型或运行病例。服务账号须能读取、执行完整程序目录，访问外部推理环境、模型和输入，并写入任务缓存、输出及日志。

已有生产版本时先备份程序与外置配置，等待在途任务结束后再替换整个目录；不要直接解压覆盖运行中的版本。升级时以新包的 `src/config/*.json` 为基础，合并生产所需的模型路径、推理命令等本机设置，核对后再运行。整份覆盖成旧配置会丢失新版部位名称、切割范围等更新。

## 4. 更新 Worker 并保存服务配置

支持 `ExecutablePath` 的 .NET Worker 在该字段非空时直接启动程序，**不添加 `main.py` 参数**；未设置时继续使用原来的 `PythonExecutablePath` + `MainScriptPath` 方式。

按实际程序位置合并或更新服务配置中的字段；当前服务使用：

```json
{
  "ExecutablePath": "/home/peng/03_data/demo/Project/发布/AutoSliceServer/AutoPrintFilm"
}
```

若按上一节示例安装到 `/home/peng/03_data/production/AutoPrintFilm`，则填
`/home/peng/03_data/production/AutoPrintFilm/AutoPrintFilm`。
当前服务已能直接运行程序，升级应用时不需要重复改造 Worker。只有旧 Worker 尚未支持
`ExecutablePath` 时，才需先更新源码并重新编译发布。在具备 **.NET 6 SDK** 和项目引用
依赖的构建机器中，进入服务端源码根目录执行：

```bash
dotnet --list-sdks
test ! -e dist/AISliceDicomService && \
dotnet publish \
  Apps/ElincloudAISliceServer.AISliceDicomService/ElincloudAISliceServer.AISliceDicomService.csproj \
  -c Release -r linux-x64 --self-contained false \
  -o dist/AISliceDicomService
```

按实际 Worker 的安装位置发布产物，保留旧二进制供回退。应用仓库中的 `服务端/` 是忽略的
本地参考副本；Worker 源码应在服务端源码仓库维护和提交。

Worker 从数据库 **`AIDicomSliceConfig.AIDicomSlice`** 读取完整配置 JSON。先备份原对象，再合并 `ExecutablePath`，保留 `WindowPresets`、`BodyPartMappings`、`SliceRuleConfigInfos`、并发配置及其他字段。**不能把上面只有一个字段的小 JSON 当作完整配置提交。** 旧 Python/主脚本字段可以保留；新 Worker 在 `ExecutablePath` 非空时优先使用直接程序模式。

保存后重新读取完整配置，确认新路径正确、原有规则仍在，再让实际 Worker 重启加载。若使用 systemd，替换为已确认的真实服务名：

```bash
WORKER_UNIT='<实际Worker服务名.service>'
test "$WORKER_UNIT" != '<实际Worker服务名.service>'
sudo systemctl restart "$WORKER_UNIT"
systemctl is-active "$WORKER_UNIT"
```

其他服务管理方式使用相应的重启命令。实际服务任务应启动上述 `AutoPrintFilm` 程序，参数与输出 `report.json` 协议保持原样。

Worker 允许同一原始序列的多个任务并发，但每个任务必须使用不同的输出目录。
全局 `MaxConcurrentCount` 与规范化输出路径互斥在同一临界区内检查，并与启动、进程登记
一起执行，避免多条派发线程同时越过名额限制。启动时先恢复各医院的在途进程，再派发新任务；
任务完成检查按任务自身 PID 和输出目录判断，不因同序列的其他任务仍在运行而延后完成。

同序列任务继续共用序列级 `cache_dir`。Python 第一阶段缺失缓存时跨进程单写，等待者
复查并复用；第二阶段按缓存目录和模型 ID 单写，同模型只推理一次，其他任务等待后复用，
不同模型推理可独立进行。完整缓存只读复用，第三至八阶段在各任务输出目录并行执行。
保留 `.locks` 文件和活动工作目录的租约保护，不要在任务运行时清理缓存。
并发数仍需按实际资源配置，不自动根据 GPU 显存调整；C# 不增加任务超时，Python 推理保护继续生效。
这项调度变更需要重新编译并更新 Worker，Python 代码变更也须重新打包更新应用。

## 5. 修改外置配置

直接编辑程序目录中的 `src/config/*.json`，下一次启动程序读取，**无需重新打包**。配置从可执行文件所在目录定位，Worker 的工作目录不同也不影响配置位置。[PyInstaller 运行路径说明](https://pyinstaller.org/en/stable/runtime-information.html)

- `nnuNet_config.json`：外部推理入口、模型根目录及推理参数。
- `slice_part_config.json`：部位、扫描层厚和切割规则。
- `window_presets.json`：窗宽窗位。
- `logger_config.json`：日志与调试设置。

三维回正涉及应用代码和 SciPy 本地计算依赖。源码环境执行
`python -m pip install -r requirements.txt`；已有发布程序须重新 `dev build` 并更新完整包。
分割缓存仍可复用，旧切片不会自动改变，需要重新执行切割。脊柱、肺部、髋骨的普通
整切 MPR 以可靠 LR 法向和输入 AP 投影构造三轴，保留输入前后倾角，不额外抬头、低头。
肺/髋正数普通MPR以所选标签覆盖范围加每侧5mm放大，不再以完整皮肤或肩臂限制倍率。
肺/髋三状位统一以原标签整组包络中心取景，所有层及占比保持固定中心与矩阵。
canvas=original保留第六阶段原CT投影画框及物理宽高比，等比适配标签加余量；
100容纳所选部位，70、50宽高同时向四周扩至1/0.7、2倍视野，80为1.25倍，
直接从完整原CT采样，目标框外的皮肤、肋骨或骶尾结构允许越框，不预裁原数据。
组内保持共同倍率和画布，末端小截面不单独拉满；不补全扫描外组织。
肺/髋0/null/省略仍保完整源切面、基础间距、scale=1和整组中心，必要时扩大共同矩阵，
不做上下ROI。其他部位、CPR和椎间隙保留原规则。新配置与程序一起更新；仅换程序而
保留旧无framing_policy配置会继续走旧人体保护兼容路径，需以新包配置为基础合并生产设置。
颅脑默认再按完整脑区在 AP/SI 平面内的形态主轴估计俯仰，只绕 LR 更新三状位共用
坐标；质量不足保留原 AP 投影并警告。它不是听眦线或 AC-PC 对齐，不新增模型或参数。
颅脑方向与三状位基础范围只用 `brain`；当前颅脑横状位 MPR 用既有 `range_extension`
默认向头侧补 `10 mm`、下侧不补，冠、矢状位不加这项余量。同次明确的 `skull` 仅在
脑区最终头脚轴物理支撑范围内辅助取景；横状位取景上界同步增加计划头侧余量，下界
仍为脑区，不把远处面部、下颌加入层组。`10 mm` 为工程余量，不保证所有病例颅顶完整。
覆盖检查以首末厚层外缘为准，定位线是中心面，不能仅凭第一条线判断漏切。没有可用骨标签
时仅用脑区取景并告警，脑区接边时报告截断风险，不补全扫描外组织。定位图背景可见
脸或下颌，定位线仍须反映真实切面。Localizer 保留固定参考点、方向及首末像素物理范围，
粗显示轴按[通用定位图规则](superpowers/modules/dicom/2026-10-08-定位图状位配置与取景隔离设计.md)
在绘线前做二维线性细化，不随 FillPercent 缩放、裁切或移动。
MPR 交线仅裁到底图边界，长度不随正式切片取景变化，
不表示成片的有限视野。开启调试文件后可在
`metadata/localization_result.json` 的 `anatomical_geometry_diagnostics.pitch` 复核角度与回退原因。
详见[颅脑形态斜切设计](superpowers/modules/reconstruction/2026-10-20-颅脑形态斜切设计.md)。
脊柱首尾质心仅用于范围有效性检查和诊断，不再决定全局 SI；无可靠 LR 时保持
患者参考三轴并告警，`applied` 反映实际姿态变化。局部椎间隙仍依真实端面及相邻
LR 斜切，CPR 曲线规则不变。回正只改变采样方向，不强制镜像真实结构。
存储轴反向由 `patient_reference_axes` 统一按患者语义取符号，SI 朝 LPS `+Z`，
整切冠状、矢状成片头侧向上；不改变原始物理坐标换算或已有斜扫描角度。
头脚符号校准本身不改变俯仰，颅脑形态俯仰由第五阶段另行估计。
镜像质量检查包含 LR 方向唯一性：涉及 LR 的近重根、圆柱角向信息不足或搜索范围内
等价对称面会拒绝更新，单纯 AP/SI 等谱不否决独立 LR。肺部还须在镜像质量通过后
验证两主体分居对称面两侧、沿 LR 分离且反射跨主体匹配，不能只凭连通域数量。
肺本体仅因镜像形态质量不足而回退时，按当前模型 `dataset.json` 的明确语义标签
`pleural effusion` 补充胸膜腔姿态轮廓，再使用同一质量与双侧门槛；不硬编码标签 ID，
不纳入心包标签，不改变原肺切割范围。原肺方向已可靠、方向歧义或其他拒绝原因不重试。
S1 使用实际 `label_targets`，优先 `vertebrae_S1`、只提供 `sacrum` 时回退；标准配置
不变，坐标端点、诊断及 `L5_S1` 的主体配对和输出计划均使用实际标签名。
详细方法见[统一姿态说明](superpowers/modules/reconstruction/2026-10-16-统一解剖姿态与三状位回正设计.md)。

`nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 只配置模型目录。外部推理命令需要单独指定；服务的 PATH 可能不包含开发环境的命令目录，生产应使用完整命令路径。本机已确认的配置为（合并到原 JSON，保留其他字段）：

```json
{
  "nnUNet_predict_command": "/home/peng/opt/fsl/bin/nnUNetv2_predict",
  "nnUNet_python": ""
}
```

这会直接执行已有 FSL 环境的 nnUNet 入口，该入口自带外部 Python 路径，所以 `nnUNet_python` 留空。当前运行目录的配置文件为 `/home/peng/03_data/demo/Project/发布/AutoSliceServer/src/config/nnuNet_config.json`。现有发布程序已经支持读取这两个字段，补充配置后下次任务启动即生效，无需重新打包或修改 Worker。

迁移机器时把命令路径调整为该机器上的外部入口；也可指定已有外部环境的 Python，例如：

```json
{
  "nnUNet_predict_command": "nnUNetv2_predict",
  "nnUNet_python": "/home/peng/opt/fsl/bin/python3"
}
```

此时由该外部 Python 调用 nnUNet 入口，其 nnUNet/PyTorch 安装和版本保持原样。两个字段均未配置时沿用原来的 PATH 命令调用。`nnUNet_python` 必须指向外部 Python，不能指向本包的 `AutoPrintFilm` 可执行程序。

`nnuNet_config.json` 还支持以下顶层推理速度及后台进程参数。将字段合并到原 JSON，保留其他字段：

```json
{
  "disable_tta": false,
  "step_size": 0.5,
  "num_processes_preprocessing": 1,
  "num_processes_segmentation_export": 1,
  "export_resampling_threads": 4
}
```

- `disable_tta`：JSON 使用布尔值，默认 `false`；设为 `true` 关闭镜像增强、减少重复推理。
- `step_size`：JSON 使用数值，范围为 `0 < step_size <= 1`，默认 `0.5`；增大后减少滑窗重叠。
- `num_processes_preprocessing` / `num_processes_segmentation_export`：分别传为 `-npp` / `-nps`，
  默认均为 `1`；只接受正整数，布尔值、浮点数和带小数点的文本均不合法。
- `export_resampling_threads`：单个导出进程内的类别通道并行数，接受整数 `1` 至 `4`。
  仓库配置为 `4`，未提供字段的旧配置为 `1`；设为 `1` 恢复原命令和原导出路径。
  不改变服务任务并发，也不改变模型、TTA、滑窗步长、原插值或标签判定规则。

环境变量也可使用同名字段，分别接受 `true`/`false`、数值文本和整数文本，优先于 JSON。
Python 显式配置支持顶层字段和 `nnunet` 嵌套段，优先于环境变量。
单病例的预处理原本实际已限制为 1 个进程；本次默认主要将导出进程池从 3 减为 1，
以减少主机内存和并发资源占用，不改变模型、TTA 或滑窗步长。

新增通道并行由项目自带入口实现，使用现有外部 nnUNet/PyTorch 环境，不修改已安装包。
只有能确认的官方预测入口才会切换：外部 Python 来自显式 `nnUNet_python`，或官方
Python 启动脚本的 shebang。无法确认的自定义命令保持原调用，RunLog 会说明未启用。
优化仅覆盖官方默认重采样器的多通道线性原尺寸恢复，其余路径保留原函数。
每次推理的 `nnunet-*.stdout.log` 会保存 `[AutoPrintFilm nnUNet]` 粗阶段耗时，
便于区分重采样、标签判定和写出。此配置不是增加 `-nps`；单病例的一个导出任务
不会因为增大进程池而自动拆成多个通道任务。

更新已有发布包时需重新构建完整目录，包含外部推理入口后再配置此字段。
建议用独立缓存比较原路径 `1` 与并行 `2`、`4`：核对分割标签、NIfTI 几何及最终切片，
同时记录时间、峰值内存和生产并发下的长尾。保留生产任务并发设置，不清理在途缓存。

确认模型前向占比值得优化后，可测试 `disable_tta=true`、`step_size=0.5`，
再按效果测试 `step_size=0.75`。
这些设置可能影响分割及自动切割定位，需要比较实际病例的耗时和切割结果。
分割缓存仍按模型 ID 复用，不区分这些参数；测试时使用新的 `-c` 目录，或清理对应的
`<cache>/<model_id>/segmentation/`，否则会复用旧分割、跳过推理。

首次加入上述字段的支持涉及应用代码变更，已有 PyInstaller 发布程序须重新执行
`dev build`，按第 3 节更新完整应用目录。升级后只改外置 JSON，下一次任务启动生效，
无需再次打包；模型无需重新训练，外部 nnUNet 环境无需重新安装，当前 Worker 无需改造。

Python 分割阶段的保护参数也放在 `nnuNet_config.json` 中，C# 不增加超时或重试判断：

```json
{
  "inference_timeout_seconds": 1800,
  "inference_timeout_seconds_by_model": {"315": 180},
  "inference_max_retries": 1
}
```

两个时限字段的单位均为秒，数值必须有限且大于 0；按模型覆盖项的键必须是无前导零的
正整数 ID 字符串。仓库配置为肺部模型 `315` 单次 180 秒，其他模型单次 1800 秒，
按实际影像规模调整即可。计时不包括缓存锁等待或后续重建。环境变量支持同名参数，
模型时限映射使用 JSON 对象文本；映射整体覆盖，`{}` 可清空模型覆盖项。
`inference_max_retries` 只允许 `0` 或 `1`，表示失败后的额外重试次数。

外部 Python 及其 spawn worker 继承 `PYTHONFAULTHANDLER=1`、`PYTHONUNBUFFERED=1`。
发现段错误等致命崩溃日志后立即中止本次推理，未输出崩溃信息时由单次时限兜底。
Linux 下会先终止本次推理进程组，必要时强制结束，并确认没有活动成员后才重试。
重试使用新临时目录；旧进程清理未确认、模型缺失或其他明确配置错误均不自动重试。
重试仍失败时生成失败 `report.json`，每次尝试的日志均保留。
Windows 源码运行保留正常推理和有时限的 `taskkill /T /F` 清理尝试；由于未使用
Windows Job Object 验证孤儿进程树，清理结果不会标记为已确认，因此不自动重试。

原始 stdout/stderr 实时写入任务的 `log/nnunet-*.stdout.log` 和
`log/nnunet-*.stderr.log`，不依赖 DebugLog 或调试影像开关；开始、结束和重试写入
RunLog。原始进度不再同步转发到控制台或 DebugLog，监测循环从原始日志检测致命崩溃并
检查单次时限，避免控制台读取阻塞监测；需要实时进度时可查看对应日志文件。
新增保护需要重新构建并更新完整应用包，之后改外置 JSON 无需重新打包。

遇到 `Background workers died` 时，先查看本次任务输出目录的 `log/ErrorLog.txt`：
其中记录实际命令和 stdout/stderr 各自末尾最多 64 KiB，报告摘要会优先显示更具体的异常。
排障时同时收集本次尝试对应的两份原始推理日志，完整早期异常可能不在 ErrorLog 的末尾摘要中。该提示本身
不能证明主机内存不足，后台进程也可能因 Python 异常退出；没有具体异常时再结合系统的
内存终止记录排查。两个进程数设为 1 只能减少并发资源占用，不能保证解决所有失败。
仓库默认 `logger_config.json` 开启 `error_log_file_enabled` 和 `run_log_file_enabled`，关闭 `debug_log_file_enabled`；
升级时保留的旧配置或显式配置仍优先，若旧配置关闭错误日志，须设为 `true` 后重跑以收集详情。

多个 Worker 任务同时运行时，应用默认限制每次 ITK 滤镜运算线程数为
`min(4, CPU 亲和性允许的逻辑 CPU 数)`；不支持亲和性查询的平台使用系统逻辑 CPU 数。
OMP/BLAS/NumExpr 默认各为 1，窗位线程池默认上限为 2；已有原生线程环境变量优先。
可在 Worker 的启动环境中设置 `AUTOPRINTFILM_NATIVE_THREADS=4` 和
`AUTOPRINTFILM_WINDOW_WORKERS=1`，由子进程继承。两项均须为正整数；前者只设置
未显式配置的 ITK 线程数。服务端最大同时任务数仍由 Worker 自己控制。
分割缓存命中后仍会进行校正和重建，因此这些上限也适用于缓存任务。
这些是 ITK 运算和窗位任务的上限，并非进程总线程硬上限；窗位任务仍可能调用 ITK，
原生库也可能有辅助线程。环境变量会传给子进程，未命中缓存时外部 nnUNet 的
OMP/BLAS 等线程默认值也会继承，除非其启动器自行覆盖；`-npp`、`-nps` 配置保持独立。
非法应用线程配置仍会在指定的输出目录下尽力生成失败 `report.json`。
阶段资源快照写入 `DebugLog` 的 `[资源]` 行；需要外置 `logger_config.json` 中的
`debug_log_file_enabled=true` 才启用，仓库默认关闭此项。Linux 下记录 RSS、进程累计峰值 RSS、
线程数、系统可用内存和 swap，内存单位 MiB。Linux 可读取 `/proc/thread-self/stat` 时，
`native_tid` 标识调用线程、`cpu` 是它最近运行的 CPU；线程可在步骤内部迁移，
这些字段不能当成 CPU 亲和性限制或故障核心的直接证明。
除阶段边界外，第三阶段记录带 `step=` 的
子步骤：读取分割、椎体序列核实及逐标签主体提取、标签索引、目标 bbox、镜像候选估计、
原图/分割旋转与写盘。第五阶段细分逐椎体主体、四棱柱、姿态与端面分析；
第七阶段细分计划及切片采样、CPR 掩膜、取景、定位图与窗位输出，包含计划、切片或窗位标识。
每步记录开始、完成或失败及耗时；读图后记录尺寸、像素类型和
实际 ITK 线程数。这些诊断统一保存在 DebugLog，不需要开启调试影像。
这部分涉及应用代码，已有程序须重新构建并更新完整应用包后生效。

关闭 DebugLog 明细时，新增步骤日志跳过资源采样和计时。低频栈采样使用独立开关，
在外置 `logger_config.json` 设置 `stack_dump_interval_seconds=60` 即每分钟采集一次；
默认 `0` 关闭，非零值须为至少 10 秒的有限数值，不必开启所有 DebugLog 明细。
CLI 为每次运行追加唯一运行标识、PID、boot ID、UTC/monotonic 时间和环境，
随后记录各 Python 线程的文件、函数和调用行，原生段错误/abort 也尽力记录。
这些记录统一追加到 `log/DebugLog.txt`，可能与普通日志交错。正常退出取消定时记录并
恢复标准错误处理；诊断不可用不会阻止任务。不因采集超时终止业务，也不强制同步磁盘。
生产常态资源监测可使用[独立主机采集器](host-diagnostics.md)，不要求开启应用 DebugLog。
普通文件日志逐条 flush，文件在应用内部重试时追加且保留；flush 仍不保证
内核崩溃或掉电时所有末尾数据都已写入磁盘。堆栈文件不能替代完整内核转储。
下次异常重启后，先复制相应病例的整个 `log/` 目录及 pstore 原始文件，再继续重试；
外部清理程序仍可能删除日志，应用内的保留规则不能约束服务端清理。

在 Linux 下，外部推理子进程会恢复 PyInstaller 启动前的动态库搜索路径；父进程仍使用自己的打包库。[PyInstaller 外部程序调用说明](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#launching-external-programs-from-the-frozen-application)

每次 `dev build` 默认把当前源码 `src/config/*.json` 同步到发布目录和压缩包，包括
`nnuNet_config.json`、`slice_part_config.json`、`window_presets.json`、`logger_config.json`。
上次 `dist/AutoPrintFilm/src/config/` 的 JSON 会被当前源码配置替换，确保部位名称、
切割范围等更新随程序发布；独立生产目录的配置不会被构建命令修改。
若本次明确需要沿用上次 `dist` 中的 JSON 修改：

```bash
dev build --keep-config
```

此选项会优先保留旧包同名 JSON，仅新增文件采用源码配置。旧命令
`dev build --reset-config` 仍可用，与默认 `dev build` 行为一致。

## 6. 缓存与构建验证

构建用 Python 的固定下载地址与 SHA256 在 `build-config.json`，基础依赖在 `requirements.txt`，构建工具在 `requirements-build.txt`，约束在 `constraints-build.txt`。这些只管理项目构建缓存，不管理外部 nnUNet/PyTorch 环境。

依赖组合变化后下次构建使用新的缓存；主动重新准备当前组合可执行：

```bash
dev build --refresh-runtime
```

构建失败保留上次完整产物，只清理本次临时构建文件。工具仅替换自身生成的应用目录及对应压缩包/校验文件，不清空整个 `dist`。构建过程中不要同时编辑产物配置或启动另一个构建。

发布前验证包括：

- 移动到不同目录后，从其他工作目录运行可执行程序 `--help`。
- NumPy 运算、SciPy 镜像面拟合（最近邻树及优化器）、SimpleITK 合成 NIfTI 写入/读取，以及 pydicom 合成 DICOM 写入/读取、像素和中文信息。
- 外置配置路径正确，修改外置日志配置后新启动的程序读取修改值，验证结束恢复原配置。
- 确认包内不含 torch、nnunetv2 或应用源码入口。

验证不使用患者数据，不检查外部模型，不运行 nnUNet 推理。构建日志位于 `.build-cache/last-build.log`。

### 系统卡死时保留内核证据

桌面和 Web 请求一起卡住时，除了病例日志，还要保存系统日志。TCP 端口可连接只说明
连接当时被接受，不能确认应用仍能完成请求。重启后先查看事故所在 boot：

```bash
journalctl --list-boots
journalctl -b -1 -k --no-pager
sudo ls -l /var/lib/systemd/pstore
```

`-1` 指上一次启动；多次重启后应改用 `--list-boots` 中对应的 boot ID。
有些内核崩溃来不及写入 journal，只留在 pstore。`systemd-pstore` 归档的
`/var/lib/systemd/pstore/<编号>/dmesg.txt` 可能需要本机 sudo 才能读取；保留原始文件供复核。
转储若明确包含 `Kernel panic`、`Fatal exception in interrupt`，说明发生了内核层故障。
应结合完整堆栈检查内核版本、驱动及硬件/固件稳定性，不能仅因当时正在切割或某驱动
出现在已加载模块列表中，就认定对应应用算法或驱动是根因。
不要用反复运行曾导致卡死的完整病例代替取证；应用步骤日志也不能替代内核转储。
