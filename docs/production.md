# AutoPrintFilm 发布部署手册

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

已有生产版本时先备份程序与外置配置，等待在途任务结束后再替换整个目录；不要直接解压覆盖运行中的版本。升级时保留实际生产 `src/config/*.json`，核对新旧配置格式后再运行。

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

## 5. 修改外置配置

直接编辑程序目录中的 `src/config/*.json`，下一次启动程序读取，**无需重新打包**。配置从可执行文件所在目录定位，Worker 的工作目录不同也不影响配置位置。[PyInstaller 运行路径说明](https://pyinstaller.org/en/stable/runtime-information.html)

- `nnuNet_config.json`：外部推理入口、模型根目录及推理参数。
- `slice_part_config.json`：部位、扫描层厚和切割规则。
- `window_presets.json`：窗宽窗位。
- `logger_config.json`：日志与调试设置。

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
  "num_processes_segmentation_export": 1
}
```

- `disable_tta`：JSON 使用布尔值，默认 `false`；设为 `true` 关闭镜像增强、减少重复推理。
- `step_size`：JSON 使用数值，范围为 `0 < step_size <= 1`，默认 `0.5`；增大后减少滑窗重叠。
- `num_processes_preprocessing` / `num_processes_segmentation_export`：分别传为 `-npp` / `-nps`，
  默认均为 `1`；只接受正整数，布尔值、浮点数和带小数点的文本均不合法。

环境变量也可使用同名字段，分别接受 `true`/`false`、数值文本和整数文本，优先于 JSON。
Python 显式配置支持顶层字段和 `nnunet` 嵌套段，优先于环境变量。
单病例的预处理原本实际已限制为 1 个进程；本次默认主要将导出进程池从 3 减为 1，
以减少主机内存和并发资源占用，不改变模型、TTA 或滑窗步长。

提速时先测试 `disable_tta=true`、`step_size=0.5`，再按效果测试 `step_size=0.75`。
这些设置可能影响分割及自动切割定位，需要比较实际病例的耗时和切割结果。
分割缓存仍按模型 ID 复用，不区分这些参数；测试时使用新的 `-c` 目录，或清理对应的
`<cache>/<model_id>/segmentation/`，否则会复用旧分割、跳过推理。

首次加入上述字段的支持涉及应用代码变更，已有 PyInstaller 发布程序须重新执行
`dev build`，按第 3 节更新完整应用目录。升级后只改外置 JSON，下一次任务启动生效，
无需再次打包；模型无需重新训练，外部 nnUNet 环境无需重新安装，当前 Worker 无需改造。

遇到 `Background workers died` 时，先查看本次任务输出目录的 `log/ErrorLog.txt`：
其中记录实际命令和完整 stdout/stderr，报告摘要会优先显示更具体的异常。该提示本身
不能证明主机内存不足，后台进程也可能因 Python 异常退出；没有具体异常时再结合系统的
内存终止记录排查。两个进程数设为 1 只能减少并发资源占用，不能保证解决所有失败。
仓库默认 `logger_config.json` 开启 `error_log_file_enabled`，其余日志开关保持原值；
升级时保留的旧配置或显式配置仍优先，若旧配置关闭错误日志，须设为 `true` 后重跑以收集详情。

在 Linux 下，外部推理子进程会恢复 PyInstaller 启动前的动态库搜索路径；父进程仍使用自己的打包库。[PyInstaller 外部程序调用说明](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#launching-external-programs-from-the-frozen-application)

首次构建沿用源码配置；重复 `dev build` 默认保留上次 `dist/AutoPrintFilm/src/config/` 中的 JSON 修改。独立生产目录的配置不会被构建命令修改。若明确要让构建产物改用当前源码配置：

```bash
dev build --reset-config
```

## 6. 缓存与构建验证

构建用 Python 的固定下载地址与 SHA256 在 `build-config.json`，基础依赖在 `requirements.txt`，构建工具在 `requirements-build.txt`，约束在 `constraints-build.txt`。这些只管理项目构建缓存，不管理外部 nnUNet/PyTorch 环境。

依赖组合变化后下次构建使用新的缓存；主动重新准备当前组合可执行：

```bash
dev build --refresh-runtime
```

构建失败保留上次完整产物，只清理本次临时构建文件。工具仅替换自身生成的应用目录及对应压缩包/校验文件，不清空整个 `dist`。构建过程中不要同时编辑产物配置或启动另一个构建。

发布前验证包括：

- 移动到不同目录后，从其他工作目录运行可执行程序 `--help`。
- NumPy 运算、SimpleITK 合成 NIfTI 写入/读取，以及 pydicom 合成 DICOM 写入/读取、像素和中文信息。
- 外置配置路径正确，修改外置日志配置后新启动的程序读取修改值，验证结束恢复原配置。
- 确认包内不含 torch、nnunetv2 或应用源码入口。

验证不使用患者数据，不检查外部模型，不运行 nnUNet 推理。构建日志位于 `.build-cache/last-build.log`。
