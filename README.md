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
python main.py -i /path/to/dicom_dir -o /path/to/output_dir --body_part 腰椎 --check_type CT --scan_thickness 3mm
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py -i D:\data\dicom_dir -o D:\data\output_dir --body_part 腰椎 --check_type CT --scan_thickness 3mm
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

当前正式注册 CT 的横状位、矢状位和冠状位直线重建能力：

- 颈椎、胸椎、腰椎、肺部、头骨、髋骨整切；
- 系统不再接收 `cut_mode` 参数，而是根据部位和状位自动选择切法；
- 腰椎横状位自动按椎间区间切；
- 腰椎矢状位、冠状位以及其他部位的所有状位自动整切。

所有直线整切和区间切统一采用“切片中心完整覆盖”规则：按请求层间距生成等距
切片中心，非零目标长度 `L` 和层间距 `S` 对应的最少张数为
`N = ceil(L / S) + 1`；零长度或长度不超过 `0.001mm` 坐标容差的目标只生成一张。
不可整除产生的中心超出量会对称
分配到两端，保证首张和末张切片的中心都到达或越过目标边界。
因此允许首尾略微切到目标外侧，但不会再舍弃不足一个层间距的肺尖、肺底或其他
部位边界；当层间距大于层厚时，只保留配置本身造成的中间物理空档。
斜向坐标在 `0.001mm` 内的存储舍入误差按同一边界处理，避免无意义地多生成一张。
层厚只决定每张成片的 slab 采样范围，不参与切片数量和中心范围计算。例如未配置
额外范围扩展时，肺部目标长度 `272.35mm`、层厚和层间距均为 `5mm`，会生成 56 张，
中心跨度为 `275mm`，首尾各外扩 `1.325mm`；对应连续原图索引约覆盖
`100.96..524.04`。
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
覆盖范围，之后仍按层间距生成严格等距中心，所以最终首末中心可能因无法整除层间距
而在配置距离之外再略微超出。

肺部已经接入第一版解剖坐标计算：阶段三从一个整肺分割 Mask 中保留一至两个肺主体，
排除远端小块误分割，并计算肺部物理中心和三轴覆盖范围；阶段四据此生成以肺部为中心、
完整包住肺部分割范围并按配置保留首尾余量的三状位直线切片。肺部仍沿用 DICOM 标准
方向，不根据肺部形态附加倾斜角度。

颈椎和胸椎已接入三维直线 MPR：阶段三读取配置范围内开始、结束椎体的物理质心，
以两点连线建立完整 XYZ 脊柱轴；阶段四生成与该轴垂直的横状位，以及与该轴平行的
矢状位和冠状位平面。阶段五、Debug 切片线和 `Localizer.dcm` 均直接使用同一套平面
方向。当前冠状位仍为直线 MPR，曲线 CPR 尚未实现。

髋骨整切使用左右独立分割：阶段三分别保留 `hip_left`、`hip_right` 的最大连通
主体，在各自顶部 5mm 内取远离身体中线的外侧区域中心，并用两点连线 AB 确定
横状位角度。横状位从 AB 起始面向髋骨下界生成全部平行切片；矢状位和冠状位
保持输入 DICOM 默认方向。三个状位的 `Localizer.dcm` 均选取经过髋骨整体中心的
原始层，其中横状位和矢状位使用冠状定位背景，冠状位使用矢状定位背景。

未传 `--plane_configs` 时默认请求横状位骨窗。MR 和骶骨会在处理输出前由预检明确拒绝。

已知限制：当前腰椎横状位区间切依赖相邻椎体分割 bbox 的相向边界。部分真实数据中，
椎体附件会导致相邻 bbox 在头脚方向重叠，此时对应的 `L1-L2` 至 `L5-S1` 区间会被
阶段四跳过并写入 `slice_plans.json` 的 `warnings`。腰椎矢状位整切及其他整切流程不受
该限制影响；重叠 bbox 的区间范围算法留待后续单独修正。

阶段五对每个切片计划先完成全部 MPR HU 图像采样，再按窗位使用线程池处理整套图像。
不同窗位不会重复执行 MPR，默认最多同时运行 4 个窗位线程；阶段五会等待所有窗位
写出完成后再进入报告阶段。正式切片写出后会校正 DICOM 的 `SliceThickness` 和
`SpacingBetweenSlices`，确保查看器读取到的层厚、层间距与阶段四计划一致。

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
