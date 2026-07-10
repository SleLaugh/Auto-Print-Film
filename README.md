# AutoPrintFilm

医学影像处理流水线：读取 DICOM 序列，转换为 NIfTI，调用 nnUNet 做分割，并继续执行定位、映射、重建和报告阶段。

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

## 运行

先激活虚拟环境。

Linux / macOS：

```bash
source .venv/bin/activate
python main.py -i /path/to/dicom_dir -o /path/to/output_dir --body_part 腰椎 --check_type CT --scan_thickness 3mm --cut_mode whole_range
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py -i D:\data\dicom_dir -o D:\data\output_dir --body_part 腰椎 --check_type CT --scan_thickness 3mm --cut_mode whole_range
```

支持的 `--body_part`：

```text
头骨, 颈椎, 胸椎, 腰椎, 肺部, 骶骨
```

查看命令帮助：

```bash
python main.py -h
```

### 当前能力范围

当前正式注册的能力只包含 CT 横状位：

- 颈椎、胸椎、腰椎、肺部、头骨、髋骨整切；
- 腰椎椎间切。

未传 `--plane_configs` 时默认请求横状位骨窗。MR、骶骨以及颈椎/胸椎椎间切会在
处理输出前由预检明确拒绝。矢状位和冠状位参数允许继续传入，便于后续逐步实现；
当前阶段四只执行已经支持的横状位计划，并在调试报告中记录未实现状位 warning。

### 输出目录安全

请求和环境预检通过后，程序会先删除整个输出目录，再创建全新的任务目录并执行阶段，
避免混入上一次运行留下的结果。请确保 `-o/--output` 指向本次任务专用目录。
只有 `debug_log_file_enabled=true` 时才生成 `manifest.json`，用于记录任务 ID、阶段状态、
错误和调试产物清单；非调试模式不生成该文件。

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
