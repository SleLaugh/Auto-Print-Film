# 外部主机诊断采集

这套脚本独立于 AutoPrintFilm。手动 CLI 默认每 10 秒读取系统指标，容量服务示例采用
主机 5 秒、GPU 5 秒。主程序无需导入诊断模块，
不改变生产任务并发、线程数或业务服务配置，也不需要重启业务。
应用的 `debug_log_file_enabled` 默认关闭；排查具体算法位置时，可结合少量已有应用
阶段/错误日志与外部时间线。外部采集器本身不启用应用内定时栈或逐条同步刷盘。
需要具体 Python 调用行时，应用支持独立的低频栈配置：在外置 `logger_config.json`
设 `stack_dump_interval_seconds=60`，无需开启普通 DebugLog 明细；每分钟的栈写到
病例 `log/DebugLog.txt`，默认 `0` 关闭。此功能须先更新包含新代码的应用包。

## 容量日志开关与存放位置

容量指标独立写入 `/var/log/autoprintfilm-host/DebugCapacity/host-YYYY-MM-DD.jsonl`，
不写业务 `RunLog`。服务必须通过 `--logger-config` 指向**实际生产外置**
`logger_config.json`，跟随其中的 `debug_log_file_enabled` 开关。
示例中的 `/REPLACE_WITH_PRODUCTION_PATH/logger_config.json` 必须在安装时替换；
占位文件不存在时采集保持关闭，不会自行启用。

每轮先读取这份小配置，只有 JSON 布尔值 `true` 才采集；`false`、缺字段、缺文件、
非法 JSON、权限不足或配置超过 64 KiB 都关闭采集。关闭时不读取 `/proc`、`/sys`，
不查询 GPU，不创建 `DebugCapacity` 或写入指标文件，只每 5 秒检查一次配置。
systemd 仍可能创建日志根目录，这是服务的目录管理。读取小配置仍有少量开销。
开启后才创建采样器及输出；关闭时关闭文件并丢弃 CPU、磁盘、进程和 GPU 缓存。
再次开启的第一条增量为空，不把关闭期间算成连续采样。

`--samples N` 表示最多 N 轮检查，关闭状态也计数，便于有限测试按时结束。
不传 `--logger-config` 时保留独立手动采样行为，不检查应用开关；生产服务不要省略它。
不需要为开关切换重启业务或诊断服务。

## 采集内容与限制

`collect_host_diagnostics.py --output <目录>` 按 UTC 日期写入 `host-YYYY-MM-DD.jsonl`。
每条包含时间与 boot ID、CPU 增量、load、可用内存/swap、PSI、相关进程 PID/PPID、
RSS/线程数/CPU、磁盘可用空间、磁盘 I/O 增量、温度以及采集耗时/读取错误。
不保存完整命令行，进程识别按实际入口而非参数中的仓库路径；排除采集器与测试入口。
打包程序同参数父子仍保留资源行，外层标为 `role=launcher`，服务进程为 `service`，
普通 Python 也可能保留。它们只是资源分类，不能据进程数推断实际病例并发；
实际并发以应用容量事件为准。多进程 RSS 相加还可能重复计算共享内存。

`disk_io.devices` 保存整盘及 dm/md 层的读取/写入字节率、IOPS、平均请求耗时、
I/O 忙碌时间比例及加权平均队列长度，来自 `/proc/diskstats` 的相邻样本增量。
扇区计数按 512 字节换算；不记录 loop/ram 设备及可识别分区，避免重复和大量空记录。
首次、计数器重置或设备身份变化的增量为 `null`，不是零吞吐。
映射层和底层设备仍可能描述同一批 I/O，不应直接相加；NVMe 的忙碌比例也不能单独
作为饱和判据，要结合吞吐、请求耗时、队列和 PSI。字段依据
[Linux 内核磁盘统计文档](https://docs.kernel.org/admin-guide/iostats.html)。

手动 CLI 默认不执行外部命令；显式加 `--gpu` 后查询所有可见 NVIDIA **整卡**的
显存已用/总量（MiB）、利用率、功耗和温度，不能将整卡显存归因到某个 PID。
`--gpu-interval` 默认 30 秒，可设有限且不小于 1 的秒数；容量示例设为 5。
GPU 只在主机采样轮次中检查是否到期，因此实际周期也受 `--interval` 影响，
把 GPU 设 1 秒、主机仍设 5 秒不会得到每秒 GPU 数据。
其余轮次复用缓存，并记录 `age_s` 与 `sampled_monotonic_s`。
采样最大值只是**观测最大值**，不能当真实峰值；短于周期的 CPU、内存、显存峰值可能漏掉。

每次 `nvidia-smi` 超时 2 秒；缺命令、驱动报错、超时或格式错误写入 `gpu.error`，
`devices=null`，不解释为 0 显存；不支持的单字段也为 `null`。
失败结果同样缓存到下一次 GPU 周期，顶层 `errors=[]` 不代表 GPU 查询成功。
GPU 查询同步执行，可能使这一条指标延迟，正常开销应在本机有限采样测量。
`collection_duration_ms` 包含查询耗时，不包括 JSON 序列化、写盘及 flush。

`capture_kernel_evidence.py --output <目录>` 每次启动创建新的 UTC 时间戳目录：

- `boots.txt` 保存 `journalctl --list-boots`。
- `previous-boot-kernel.txt` 保存 `journalctl -b -1 -k --no-pager -o short-iso`。
- `pstore/<原编号>/dmesg.txt` 保存 `/var/lib/systemd/pstore/*/dmesg.txt` 中已有普通文件，
  保留原目录编号，按修改时间从新到旧复制，单次合计最多 32 MiB。
- `metadata.json` 记录命令退出码、超时、输出截断、pstore 文件清单和权限错误。
  两条 journal 命令各最多运行 8 秒，终止后的等待最多 1 秒；各自 stdout 最多
  32 MiB、stderr 最多 64 KiB。输出采用流式读取，避免把整份 journal 放入内存。

证据源缺失、无读取权限或 journal 命令失败会写入元数据，不阻止随后采集系统指标；
无法创建输出目录或写入元数据时返回非零退出码。

脚本不执行 sudo，不清除、更名或覆盖源 pstore，不处理其他 pstore 文件，不改内核设置。
普通用户无权限读取时只记录失败；示例服务使用 root 读取系统证据。
没有持久化 journal、pstore 未配置或硬件未能留下记录时，证据可能为空。
外部采集器不能获得 Python 当前具体行、所有线程栈、完整 vmcore；可配合上述应用内
低频调用栈与关键步骤日志，不能单凭一条 CPU/温度
指标判定硬件根因。硬锁死或断电也可能使最后若干样本丢失；每条 JSONL 会 flush，
但不逐条 fsync，因此不能保证已写入物理介质。

## 任务摘要与离线容量报告

### 分割内部耗时

更新后的推理适配器在任务 `log/nnunet-*-attempt-*.stdout.log` 中写入
`[AutoPrintFilm nnUNet timing]` JSON。`stage` 区分脚本导入、nnUNet 导入、
模型初始化、预处理等待、预测和导出；每条带 `pid`、`status`、`seconds` 与
单调时钟起止值。模型初始化包含权重读取及网络构造，不等于只读权重的时间。
`preprocessing_next` 是父进程等到下一份输入的时间，包含可能发生的 worker
启动、读文件、预处理和进程通信；方案 6 的 `preprocessing_compute` 才是
对应 worker 内的计算区间。

`RunLog.txt` 的 `[nnUNet 进程计时]` 记录 Popen、监测等待、进程组清理和
总区间。监测等待包含子进程启动、执行、退出及轮询延迟，不是独立排队时间。
这些事件只在实际推理时产生；分割缓存命中仍会跳过推理。

所有区间都是包含内部调用的 host wall time，未增加 CUDA 同步，也不是纯 GPU
kernel 时间。父进程等待和 worker 计算可能重叠，预处理、预测、导出也可能
包含在较大的总区间中，不能直接相加或相减作为独立阶段。脚本入口之前的
解释器/site 启动没有细分测量。查看慢任务时先按 PID 和起止区间定位，再决定
优化输入、模型初始化、预测或导出；不要把整个 segmentation 阶段当作网络前向。

### CPR 重建验证

CPR 保持原有层厚、子曲面密度、HU 三线性插值及平均、mask 最近邻及并集规则。
当前实现仅在同一计划内复用曲线方向和逐行坐标系；定位图需要相同曲线、行数、
方向及有效支持范围才复用方向场。采样以约 64K 点的行块限制临时数组大小，
每个像素仍按原先的子曲面顺序累计。

第三至第五阶段可复用同一分割快照的逐椎体对称性结果，包含未接受的候选。
缓存只活在本次 localization workspace，匹配索引对象、图像几何和采样参数；
空间校正、分割快照替换后不能复用，第五阶段完成或失败都会释放。

性能验证应固定同一份 raw/seg，使用不同输出目录，对比同一源码基线和修改版。
先核验 DICOM 数量、像素、位置、方向、间距和层厚，再比较关闭 profiler 后的
阶段耗时；UID、路径和运行时间等运行元数据单独处理。历史运行可能来自不同
源码或参数，不能仅凭历史 RunLog 就认定当前优化保持输出一致。

### 普通 MPR 与椎间隙取景验证

腰椎、颈椎、胸椎的椎间隙共同取景只提取一次标签点云，准备共同倍率时保留
每个原计划、原切面的投影边界和尺寸摘要，执行时复用。摘要不持有完整 mask
或点云，不跨任务保留；计划对象、切面方向和中心必须匹配。其他平面取景也
在当前方向内复用相同法向深度、相同层厚的目标边界，空层结果同样复用。
方向变化会清空边界缓存，不对浮点深度取整，也不合并邻层。

普通 MPR 的近轴厚层采样先求保守源层索引区间，区间内仍逐层使用原物理点、
重叠厚度和累加顺序。近轴偏差、反向轴、大物理原点均纳入区间估计；不能可靠
确定范围时退回全轴遍历。每个子平面直接设置重采样输出几何，避免额外分配
参考图和再次复制 float32 数组。层厚、采样密度、插值及源边界处理不变。

多部位回归应包含不同大小的腰椎、发生空间校正的颈椎，以及胸椎、肺、髋和脑，
固定与模型匹配的原图和分割结果，逐项比较 DICOM 像素、几何与规划结果。
此类基准只说明分割后处理的收益，不包含模型推理加速；计时期间应串行运行
病例，并避免同时运行测试或 profiler。资源操作存在嵌套，不能把各名称耗时
直接相加当作总耗时。

2026-11-13 的 MPR 增量验证包含 8 个任务：腰椎、颈椎各 2 个，胸椎、肺、髋和脑
各 1 个。基线由当前代码撤回本轮 3 个 MPR 文件的改动得到，保留前一轮 CPR 和
姿态缓存优化；原 `/tmp` 快照已丢失，这次使用明确标注的重建基线，并独立核对
源码差异。每个任务只跑一对，不清空系统页缓存，也不保证独占主机。

| 任务 | 重建阶段优化前 → 后（秒） |
| --- | ---: |
| 大腰椎 | 23.327 → 16.980 |
| 另一例腰椎 | 11.928 → 9.600 |
| 颈椎第一例 | 11.089 → 9.070 |
| 颈椎第二例 | 11.222 → 8.276 |
| 胸椎 | 7.741 → 7.592 |
| 肺 | 4.967 → 4.795 |
| 髋 | 7.021 → 6.897 |
| 脑 | 6.226 → 6.277 |

828 张 DICOM、268,785,521 个像素零差异；所检查的几何和显示标签、完整切割
计划与仅归一化输出路径的报告一致。主要收益出现在腰椎、颈椎；其他部位的小幅
变动不作为稳定提速结论，分割推理不包含在该对照中。原始数据哈希未变，记录
保存在本机 `outputs/diagnostics/mpr_performance_20261113/`，该目录不随 Git 提交。
代码、测试和文档随本次变更提交，模型、病例输入和生成的 DICOM 不进入仓库。

### 标签索引与取景扫描复用验证

同日的后续优化以提交 `8c61845` 为直接基线，包含上述 MPR 优化及其后的椎间隙
核验修复。第三阶段每个分割空间只扫描一次目标 ID 并集，再从稀疏集合分配各标签
索引；回正后仍重新建立索引。第七阶段从正式分割路径独立建立多计划共享索引，
共同倍率准备和取景构造使用同一份稀疏集合，单次构造也复用目标坐标。
索引不改变体素行序、坐标存储布局、质心、bbox 和后续物理坐标计算；不跨任务
复用，也不延长第三至第五阶段工作区生命周期。纯 CPR 继续使用原掩膜路径。

本次仍固定原图和分割，串行执行 8 对独立进程，不运行分割推理或 DICOM 转原图，
不启用 profiler；每个病例只测一对，不清空系统页缓存，不保证主机独占。下表为
`main` 的分割后流程耗时，不是单独重建时间，也不能与上一轮不同基线的降幅相加。

| 任务 | 本轮优化前 → 后（秒） | 耗时减少 |
| --- | ---: | ---: |
| 大腰椎 | 25.874 → 20.372 | 21.26% |
| 另一例腰椎 | 17.337 → 14.648 | 15.51% |
| 颈椎第一例 | 19.684 → 16.429 | 16.53% |
| 颈椎第二例 | 14.046 → 10.652 | 24.16% |
| 胸椎 | 32.009 → 26.135 | 18.35% |
| 肺 | 7.431 → 7.311 | 1.61% |
| 髋 | 9.998 → 9.268 | 7.30% |
| 脑 | 8.137 → 8.132 | 0.06% |

本组总耗时 `134.516 → 112.947s`，减少约 `16.03%`。腰椎和颈椎同时减少定位与
取景开销，胸椎收益主要来自第三阶段 `16.352 → 10.278s`；肺、脑未见明确速度收益。
大腰椎峰值 RSS `1213.3 → 1087.0 MiB`，全部病例未出现明显内存增加。
828 张 DICOM、268,785,521 个像素零差异，21 项几何/显示标签、完整计划及仅归一化
输出路径的报告完全一致，原始输入 SHA256 全部未变。当前 112 个源码文件与已验证
候选快照一致；源码、输入哈希、逐例分阶段耗时及 RSS 保存在本机
`outputs/diagnostics/label_index_performance_20261113/`。

Python 回归为 `2833 passed, 3 skipped, 1192 subtests passed`，跳过项均因可选外部
nnUNet 依赖缺失；按本轮范围未执行 C# 调度测试。直接只读代码审查结果为
`No findings.`，改动文件的静态检查通过。

### MPR 采样固定配置与定位图统计复用验证

本轮以提交 `cead3e1` 为精确基线，已包含上面的标签索引优化。普通 MPR 仅在单张
厚层切片内复用滤波器固定配置，仍逐偏移执行二维采样。原点算式、插值、有效支持、采样密度、浮点精度和
累加顺序不变。定位线灰度按需统计一次同请求只读原图的 min/max，多个计划只
共享标量；窗化去掉转换 float64 之前的原始类型数组复制。本轮 CPR 几何缓存尝试
经病例复测没有提速证据，最终撤回；`curved_stack.py` 与本轮基线完全一致。

精确对照覆盖近轴与斜切、反向轴、大物理原点、浮点源 NaN/Inf、多个偏移的默认值
复位、自定义采样接口及并发切片隔离。保留实现的完整 Python 回归为
`2913 passed, 3 skipped, 1 warning, 1192 subtests passed`；跳过项为可选 nnUNet
依赖缺失，不运行 C# 调度测试。记录见本机 `retained_python_tests.txt`。
一次 CPR 测试中记录了 `PyDataMem_UserFREE` / `PyCapsule_GetPointer` 底层释放警告；
对整个触发测试文件开启该警告即报错后复核，`43 passed`，未复现。该警告来源尚未
确定，不能认定已经修复；记录见 `native_warning_recheck.txt`。最终改动的直接只读
审查结果为 `No findings.`，静态检查和 diff 检查通过。

固定输入病例对照记录位于本机 `outputs/diagnostics/sampling_efficiency_20261113/`。
基线通过 Git 提取为独立快照，候选为已通过上述测试的冻结源码；串行使用独立进程，
保持 ITK 4 线程、窗口写出 2 线程、OMP/BLAS 1 线程。对照不运行分割推理或
DICOM 转原图，也不改写原始缓存。病例包括原 8 例 MPR，以及来自原成功请求的
胸腰椎冠状位、矢状位 CPR。比较全部 DICOM 像素、21 项几何/显示标签、完整计划和
仅归一化输出路径的报告，并核对原始输入 SHA256。主机非独占、不清空页缓存，
单对耗时只能作为该次测量，不能把不同基线的提速百分比相加。

最初候选（包含后来撤回的 CPR 几何缓存）的首轮 9 对结果如下，负数表示候选更慢：

| 任务 | 重建阶段基线 → 候选（秒） | 耗时减少 |
| --- | ---: | ---: |
| 大腰椎 | 12.627 → 12.369 | 2.04% |
| 另一例腰椎 | 7.711 → 7.625 | 1.12% |
| 颈椎第一例 | 6.698 → 6.478 | 3.28% |
| 颈椎第二例 | 6.119 → 5.989 | 2.12% |
| 胸椎 | 7.773 → 7.670 | 1.32% |
| 肺 | 4.809 → 4.929 | -2.50% |
| 髋 | 7.078 → 6.890 | 2.66% |
| 脑 | 6.544 → 6.569 | -0.37% |
| 胸腰椎双平面 CPR | 39.983 → 40.763 | -1.95% |

870 张 DICOM、295,046,537 个像素零差异，几何/显示标签、计划及报告全部一致，
原始输入 SHA256 未变。当前结果说明腰椎、颈椎等病例只有小幅改善；CPR 首对没有
提速证据，不能用合成采样微基准的改善代替病例结论。所有首对原始测量保存在
`benchmark_summary.json`、`benchmark_summary.txt` 及 `comparisons/`，包含分阶段耗时
和进程峰值 RSS。

随后对肺与胸腰椎 CPR 各补两对，第二对反转执行顺序，第三对恢复基线先执行。
全部重复输出仍精确一致，原始结果保留在 `repeat_summary.json/.txt`。肺的三次
重建中位数为 `5.027 → 4.929s`，减少 `1.94%`；CPR 则为 `40.069 → 41.146s`，
增加 `2.69%`，不据此宣称 CPR 提速。检查发现候选 HU 采样循环把返回数组绑定为
局部变量，导致前一块在下一次插值期间仍存活。试验修正版本恢复直接累计表达式，
释放时机与原实现一致，并通过弱引用测试覆盖原生路径与自定义采样接口。
这是已确认的生命周期差异，但不能仅凭代码检查把全部耗时差异归因于它。
修正后的 CPR 单独使用新快照验证，重建仍为 `42.392 → 42.751s`，主流程为
`64.036 → 64.559s`；单对约 `0.85%` 的变化不能证明稳定回退，也没有提速证据。
记录保留在 `final_cpr/final_summary.json/.txt`，该目录是修正尝试，并非最终保留版本。
因此最终撤回全部 CPR 几何缓存及其专用测试，只保留其余三项优化；所有试验源码
快照和测量仍保留。最终保留版本位于 `retained_changes/`，性能结论不沿用被撤回代码。
最终保留版本另跑一次胸腰椎双平面 CPR，与既有不可变基线作纯输出对照：42 张
DICOM、26,261,016 个像素零差异，21 项标签、完整计划和规范化报告一致，输入哈希
未变。最终 112 个运行时源码及配置文件与工作区匹配，仅 3 个生产源码文件有差异。
该对照非同期执行，不比较速度；结果见 `retained_changes/final_summary.txt`。
本轮代码、测试及文档留在工作区，未暂存或提交。

### 后处理 CPU 工作量与线程数验证

2026-11-16 的 CPU 对照以当时已经包含上述三项未提交优化的工作区为基线，
不是仅以 `cead3e1` 为基线。新候选只增加同次 CPR 规划的分割读取与终端标签
点云复用：正常完整规划的分割读取从 3 次减为 1 次，点云提取从 5 次减为 3 次。
缓存不跨规划调用；读取失败不缓存，保持后续重试。标签顺序、坐标计算、端点
选择和延伸规则不变。实现及边界测试见 CPR 端点延伸设计文档。

基准同时记录实际耗时和进程所有线程累计的 CPU 秒数。CPU 秒数减少才表示工作量
减少；降低线程数可能只是把相同工作摊到更长时间。Linux 按单核口径显示的 100%
约等于占满一个逻辑核，不能与全机利用率混用，也不能把已创建线程数视为忙核数。
本机现有默认值已经是 ITK 4 线程、窗化/写出最多 2 个 worker、OMP/BLAS 1 线程。
固定同一源码、同一输入，仅改变 ITK 默认线程数，得到以下探索性结果：

| 处理配置 | ITK 线程 | 实际耗时（秒） | CPU 工作量（秒） |
| --- | ---: | ---: | ---: |
| 大腰椎 | 4 | 20.492 | 30.547 |
| 大腰椎 | 2 | 24.103 | 29.305 |
| 大腰椎 | 1 | 27.834 | 27.567 |
| 胸椎 | 4 | 25.709 | 31.320 |
| 胸椎 | 2 | 28.850 | 31.896 |
| 胸椎 | 1 | 31.073 | 30.725 |

降到 2 线程，本次耗时分别增加约 17.6% 和 12.2%；降到 1 线程，增加约 35.8%
和 20.9%。CPU 工作量并未同比下降，因此保持现有默认线程设置。首个大腰椎
4 线程探索运行期间，采样器收到一次末端样本筛选修正；原记录保留，表中改用
最终冻结计量脚本重新执行的 `_02` 结果。主流程和阶段的 CPU 计时公式未改变。

CPR 规划复用使用相同线程设置测两对，第二对反转执行顺序，结果如下：

| 对照 | 规划耗时基线 → 候选（秒） | 主流程耗时基线 → 候选（秒） | 主流程 CPU 基线 → 候选（秒） |
| --- | ---: | ---: | ---: |
| 基线先执行 | 6.892 → 5.838 | 60.210 → 58.737 | 60.414 → 58.935 |
| 候选先执行 | 6.838 → 5.665 | 60.148 → 58.854 | 60.341 → 59.075 |

规划耗时减少约 15.3%～17.2%，主流程 CPU 工作量减少约 2.1%～2.4%。该病例平均
占用约一个逻辑核，重建仍占约 39 秒，是剩余主要开销。本改动没有明显降低采样
CPU 峰值，也不能据此声称所有部位都获得同样改善。两对峰值 RSS 分别为
`1107.8 → 1101.9 MiB`、`1099.3 → 1115.9 MiB`，未见一致增减趋势。
另做了保持浮点运算顺序的插值临时数组复用试验，36 组边界验证通过，但微基准
收益不稳定，随机访问存在回退，因此未加入生产代码；`curved_stack.py` 保持不变。

每对 CPR 比较 42 张 DICOM、26,261,016 个像素，全部零差异；21 项几何/显示标签、
完整计划和仅归一化输出路径的报告完全一致。线程试验输出也完全一致。所有运行
串行使用独立进程，只读既有 raw/seg，不含分割推理和 DICOM 转原图，不运行
profiler，不清空页缓存，不保证主机独占；进程及外部负载采样保留供核验。
本轮三种配置来自同一 CT 序列，不是三个独立病例；各运行区间外部 CPU 采样均值
约为单核口径的 71%～95%，因此不把规划之外的小幅变化当作稳定收益。
记录在本机 `outputs/diagnostics/cpu_efficiency_20261116/`，不随 Git 提交。
这轮基线已含既有优化，收益不可与前几轮百分比直接相加。

最终 Python 回归为 `2927 passed, 3 skipped, 1192 subtests passed`，本次无警告；
跳过项为可选 nnUNet 依赖缺失，按范围未执行 C# 调度测试。独立只读审查无发现，
改动文件 Ruff 和 diff 检查通过。23 个输入文件的最终 SHA256、1770 张原始 DICOM
的文件列表/大小/修改时间、112 个候选运行时文件均核验通过。HEAD 和暂存内容未变，
既有代码改动及 `.vscode/launch.json` 保留；本轮未暂存、未提交。

### 应用容量事件

新增任务容量事件沿用 `debug_log_file_enabled`，写入任务 `log/DebugLog.txt`，
不写 `RunLog.txt`，不依赖 `debug_file_enabled` 或定时栈开关。
`[Capacity]` 记录 Pipeline 的起止、阶段耗时、模型、分割缓存命中/等待后复用/
实际推理，以及每次真实推理子进程调用的边界。每次任务通常只增加两条摘要，
每次推理尝试另加两条；不在影像逐层循环增加采样。
应用在任务启动时读取配置，关闭 debug 对新任务生效；已运行任务沿用启动时配置。
独立采集器每轮重新检查配置，关闭后最迟下一轮停止资源采样和写入。

序列只记录规范化缓存路径的短哈希，不输出原路径或患者信息。
Python 摘要需要重新打包应用，仅更新外置配置不能让旧程序产生这些新事件。
主机采集脚本可独立启用，无需更新业务程序。容量分析只覆盖 Python 启动后的处理
区间，服务端排队时间没有采集；本次日志功能不需要修改或重新部署 Worker。

在需要分析时手动运行纯标准库脚本，指定本次观察覆盖的任务输出父目录：

```bash
python3 scripts/summarize_capacity.py --tasks-root <任务输出父目录> \
  --host-root outputs/diagnostics/DebugCapacity --output outputs/diagnostics/capacity-report.json
```

脚本只读限定的 `log/DebugLog.txt` 与主机 `host-*.jsonl`，不读取影像。
它按 boot ID 和实际时间区间统计已记录任务重叠、推理重叠、耗时分位数、失败数及
资源采样，GPU 缓存重复值不算新采样。旧日志缺少事件、未结束任务和无资源样本
均需按缺失处理；不把配置上限、任务总数或进程数量当作实际任务并发。
报告覆盖的是已记录的 Pipeline 区间，不包括 CLI 导入及未记录任务；观测到的重叠
和采样最大值不能证明服务器的稳定上限。比较并发收益还需相近模型、影像规模、
缓存冷热比例、足够业务负载和观察窗口。

## 可审查的安装步骤

以下命令供确认后手动执行；在仓库根目录运行。仅安装和启动新的诊断服务，
不会操作 AutoPrintFilm 的业务服务。

```bash
python3 scripts/collect_host_diagnostics.py --help
python3 scripts/capture_kernel_evidence.py --help
cat scripts/autoprintfilm-host-diagnostics.service.example

sudo install -d -o root -g root -m 0755 /opt/autoprintfilm-diagnostics
sudo install -o root -g root -m 0755 scripts/collect_host_diagnostics.py /opt/autoprintfilm-diagnostics/
sudo install -o root -g root -m 0755 scripts/capture_kernel_evidence.py /opt/autoprintfilm-diagnostics/
sudo install -o root -g root -m 0644 scripts/autoprintfilm-host-diagnostics.service.example /etc/systemd/system/autoprintfilm-host-diagnostics.service
sudoedit /etc/systemd/system/autoprintfilm-host-diagnostics.service
# 在 ExecStart 中把 logger-config 占位路径换成实际生产外置 logger_config.json。
sudo systemd-analyze verify /etc/systemd/system/autoprintfilm-host-diagnostics.service
sudo systemctl daemon-reload
sudo systemctl enable --now autoprintfilm-host-diagnostics.service
sudo systemctl status autoprintfilm-host-diagnostics.service --no-pager
```

示例安装位置是 `/opt/autoprintfilm-diagnostics`，输出位置是
`/var/log/autoprintfilm-host`，目录由 root 拥有，`adm` 组可读，其他用户不能访问。
脚本及安装目录由 root 拥有，普通业务用户不能修改 root 服务执行的脚本。
服务只轮询容量开关；不自动调用内核证据脚本，避免关闭 debug 时仍采集内核日志。
`ProtectHome=read-only` 允许读取位于 `/home` 的生产配置。
`Nice=10`、`CPUWeight=10`、`IOWeight=10` 降低竞争时的优先级；它们不是 CPU 或内存
硬上限。采集失败时按 `Restart=on-failure`、`RestartSec=10` 重试。
不需要 GPU 数据时，可先审核 unit，再移除 `ExecStart` 的 `--gpu`。

如只需短时观察，可手动采 6 条样本，不安装服务：

```bash
python3 scripts/collect_host_diagnostics.py --output /tmp/autoprintfilm-host-observation/DebugCapacity --interval 5 --gpu --gpu-interval 5 --samples 6 --logger-config /REPLACE_WITH_PRODUCTION_PATH/logger_config.json
```

替换上述配置路径后，只有该配置开启 debug 才写入样本，否则 6 轮检查后退出且不创建输出。
需要保存重启前内核证据时，独立手动运行以下命令；它不由容量服务自动触发：

```bash
python3 scripts/capture_kernel_evidence.py --output /tmp/autoprintfilm-kernel-evidence
```

## 本机用户级服务方案

不需要 root 的用户级服务也可按同一开关运行。先确认本机没有重复的采集器，
再使用以下本机实际路径；部署目录变化时同步修改。用户服务仅记录容量指标，
不会触发业务重启、内核采集或改变并发。

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/autoprintfilm-capacity-monitor.service <<'EOF'
[Unit]
Description=AutoPrintFilm gated capacity monitor

[Service]
Type=simple
Nice=10
CPUWeight=10
IOWeight=10
UMask=0027
NoNewPrivileges=true
ExecStart=/usr/bin/python3 /home/peng/03_data/demo/code/AutoPrintFilm/scripts/collect_host_diagnostics.py --output /home/peng/03_data/demo/code/AutoPrintFilm/outputs/diagnostics/DebugCapacity --interval 5 --gpu --gpu-interval 5 --logger-config /home/peng/03_data/demo/Project/发布/AutoSliceServer/src/config/logger_config.json
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now autoprintfilm-capacity-monitor.service
systemctl --user status autoprintfilm-capacity-monitor.service --no-pager
```

此方案的输出在仓库 `outputs/diagnostics/DebugCapacity`，跟随上面的生产外置配置；
debug 关闭时服务仍显示运行，但只低频检查配置，不写指标。用户服务依赖用户 systemd
管理器存活，登出或重启后的持续运行取决于本机用户管理器/linger 设置；示例不改该设置。
停用命令保留已有日志：

```bash
systemctl --user disable --now autoprintfilm-capacity-monitor.service
```

## 停用与后续移除

本轮诊断代码集中在独立提交“增加可开关的切割诊断与主机采集日志”，便于以后回退。
只需停用时，在实际生产外置 `logger_config.json` 中设置
`debug_log_file_enabled=false`、`stack_dump_interval_seconds=0`：前者关闭任务容量事件、
资源明细和跟随配置的主机采样，后者关闭独立定时栈。应用配置对新任务生效，已有任务
沿用启动配置；主机采集器在下一轮检查配置后停止采样。

完全移除前，先执行上面的用户服务停用命令，再删除
`~/.config/systemd/user/autoprintfilm-capacity-monitor.service` 并运行
`systemctl --user daemon-reload`。如果安装的是系统级示例，则停用并删除对应的
`autoprintfilm-host-diagnostics.service`，然后执行 `sudo systemctl daemon-reload`。
回退仓库提交不会自动删除这些仓库外的已安装服务、修改生产外置配置或删除历史日志。

移除源码时，优先对这笔独立提交执行 `git revert <提交号>`，并检查后续提交的依赖和
冲突；不要直接恢复整个代码目录。若选择手动清理，须连同 `main.py` 的栈诊断生命周期、
Pipeline/分割阶段的容量事件、定位/解剖分析/重建阶段的资源操作日志，以及对应配置、
测试和文档一起移除，不能只删除 `src/diagnostics.py`、`src/capacity.py` 或采集脚本。
历史 `DebugLog.txt` 和 `outputs/diagnostics` 可保留供追溯，删除前另行确认。

## 日志体积和维护

指标文件按日分开，不自动删除历史记录；持续开启时，5 秒周期每天约 17,280 条，
CLI 默认 10 秒约 8,640 条，具体大小取决于进程与块设备数量。关闭时不追加指标。
建议先观察一小时的实际增量，再按可用空间制定保留天数或目录配额。
每次手动运行的内核证据最多约 96 MiB 加元数据；容量服务重启不会创建内核副本。
上述限制是单次限制，整个历史目录没有总量上限，需要人工归档/删除或另配保留策略。
目录配额/外部保留任务不包含在此示例中。

采集主要是 `/proc`、`/sys` 的小文本读取和低频 JSONL 追加；实际开销随进程数量变化，
可以从 `collection_duration_ms` 和服务自身 CPU/RSS 判断。可用空间不足或写入失败时，
采集器退出，由 systemd 重试；此时先释放空间或调整保留策略。
停止采集只需 `sudo systemctl disable --now autoprintfilm-host-diagnostics.service`，
已有日志和业务进程均保留。
