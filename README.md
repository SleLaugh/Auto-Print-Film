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
python main.py -i /path/to/dicom_dir -o /path/to/output_dir --body-part 腰椎
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py -i D:\data\dicom_dir -o D:\data\output_dir --body-part 腰椎
```

支持的 `--body-part`：

```text
颅脑, 颈椎, 胸椎, 腰椎, 肺部, 骶骨, 髋骨
```

查看命令帮助：

```bash
python main.py -h
```

## PyTorch / CUDA 说明

`requirements.txt` 会安装项目运行所需的 Python 包，包括 `nnunetv2`。如果需要 GPU 推理，建议先按照本机 CUDA 版本安装 PyTorch，再运行依赖安装命令。PyTorch 安装命令以官方选择器为准。

CPU 环境也可以安装依赖，但 nnUNet 推理速度会明显慢。
