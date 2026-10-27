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
