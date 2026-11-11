#!/usr/bin/env python3
"""Read Linux host counters into daily UTC JSONL files without importing the application.

Example: python3 scripts/collect_host_diagnostics.py --output /path/to/host-monitor
Defaults to one sample every 10 seconds until Ctrl-C. --samples limits the run.
--gpu enables a separate nvidia-smi query (default every 30 seconds, 2 s timeout).
Capacity example: --interval 5 --gpu --gpu-interval 5.
No root access is required. Unreadable optional counters are null/omitted.
CPU utilization is normalized to the host's capacity; process CPU can exceed
100% for a multithreaded process. First samples have no CPU utilization delta.
Only process names/categories are saved, never complete command lines.
Each JSONL record is flushed, without fsync; abrupt power loss can still lose
recent filesystem buffers. Run a single collector per output directory.
--logger-config gates all sampling on debug_log_file_enabled=true; an unreadable,
missing or invalid configuration disables sampling. --samples counts loop attempts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def error_info(source: str, exc: Exception) -> dict:
    return {"source": source, "type": type(exc).__name__, "message": str(exc)[:240]}


def debug_logging_enabled(path: Path) -> bool:
    """Fail closed before constructing a sampler or touching its output directory."""
    try:
        with path.open("rb") as stream:
            text = stream.read(65537)
        if len(text) > 65536:
            return False
        config = json.loads(text.decode("utf-8"))
        return isinstance(config, dict) and config.get("debug_log_file_enabled") is True
    except (OSError, UnicodeError, ValueError):
        return False


def process_kind(comm: str, command: bytes, *, cwd: str | None = None) -> str | None:
    """Classify entry points, not repository names found in arbitrary arguments."""
    argv = [arg.decode(errors="replace") for arg in command.split(b"\0") if arg]
    executable = Path(argv[0]).name.lower() if argv else comm.lower()
    arguments = argv[1:]
    if executable in {"dotnet", "dotnet.exe"}:
        services = {"autoprintfilmservice.dll", "elincloudaisliceserver.dll"}
        return "service" if any(Path(arg).name.lower() in services for arg in arguments) else None
    if executable in {"autoprintfilm", "autoprintfilm.exe"}:
        if "--multiprocessing-fork" in arguments:
            return "python"
        entry = executable
    elif "python" in executable or "python" in comm.lower():
        entry = ""
        index = 0
        while index < len(arguments):
            arg = arguments[index]
            if arg in {"-c", "-m"}:
                value = arguments[index + 1] if index + 1 < len(arguments) else ""
                if arg == "-m" and value in {"pytest", "unittest"}:
                    return None
                if (arg == "-m" and value.startswith("nnunetv2.inference.")) or (
                    arg == "-c"
                    and "from nnunetv2.inference.predict_from_raw_data import" in value
                    and "predict_entry_point" in value
                ):
                    return "nnunet"
                return "python"
            if arg in {"-W", "-X"}:
                index += 2
                continue
            if not arg.startswith("-") or arg == "-":
                entry = arg
                break
            index += 1
        name = Path(entry).name.lower()
        if name in {
            "collect_host_diagnostics.py", "capture_kernel_evidence.py", "pytest", "py.test",
        } or name.startswith("test_") and name.endswith(".py"):
            return None
    else:
        entry = executable

    name = Path(entry).name.lower()
    if name.startswith("nnunetv2_predict") or name in {"nnunet_export.py", "nnunet_gpu_export.py"}:
        return "nnunet"
    project_main = name == "main.py" and (
        Path(entry).parent.name.lower() == "autoprintfilm"
        or not Path(entry).is_absolute() and cwd is not None
        and Path(cwd).name.lower() == "autoprintfilm"
    )
    if name in {"autoprintfilm", "autoprintfilm.exe", "frozen_entry.py"} or project_main:
        options = {arg.split("=", 1)[0] for arg in arguments}
        if options & {"-i", "--input"} and options & {"-o", "--output"} and not (
            options & {"-h", "--help", "--_build-check"}
        ):
            return "autoprintfilm"
        return "auxiliary"
    if "python" in executable or "python" in comm.lower():
        return "python"
    return None


class HostSampler:
    def __init__(
        self,
        output: Path,
        *,
        proc_root: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
        gpu: bool = False,
        gpu_interval: float = 30,
    ):
        self.output = Path(output)
        self.proc = Path(proc_root)
        self.sys = Path(sys_root)
        self.clock_ticks = os.sysconf("SC_CLK_TCK")
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.kernel_release = os.uname().release
        self.gpu_enabled = gpu
        if not math.isfinite(gpu_interval) or gpu_interval < 1:
            raise ValueError("gpu_interval must be finite and >= 1 second")
        self.gpu_interval = gpu_interval
        try:
            self.boot_id = (self.proc / "sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            self.boot_id = None
        self.previous_cpu: dict[str, tuple[int, int]] = {}
        self.previous_processes: dict[int, tuple[int, int, float]] = {}
        self.previous_disk_io: dict[tuple[str, str, str], tuple[dict, float]] = {}
        self.gpu_cache: dict | None = None
        self.last_gpu_poll: float | None = None

    def cpu(self) -> dict:
        counters = {}
        previous = self.previous_cpu
        current = {}
        for line in (self.proc / "stat").read_text().splitlines():
            fields = line.split()
            if not fields or not (
                fields[0] == "cpu" or fields[0].startswith("cpu") and fields[0][3:].isdigit()
            ):
                continue
            if len(fields) < 5:
                raise ValueError("incomplete CPU counters")
            name = fields[0]
            # guest/guest_nice are included in user/nice; do not add twice.
            ticks = [int(value) for value in fields[1:9]]
            total = sum(ticks)
            idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
            utilization = None
            if name in previous:
                delta_total = total - previous[name][0]
                delta_idle = idle - previous[name][1]
                if delta_total > 0 and 0 <= delta_idle <= delta_total:
                    utilization = round(100.0 * (delta_total - delta_idle) / delta_total, 3)
            current[name] = (total, idle)
            counters[name] = {
                "utilization_percent": utilization,
                "total_ticks": total,
                "idle_ticks": idle,
            }
        if "cpu" not in counters:
            raise ValueError("aggregate CPU counters missing")
        self.previous_cpu = current
        aggregate = counters.pop("cpu")
        return {**aggregate, "per_cpu": counters}

    def memory(self) -> dict:
        values = {}
        for line in (self.proc / "meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            values[name] = int(value.split()[0]) * 1024
        result = {
            "total_bytes": values.get("MemTotal"),
            "available_bytes": values.get("MemAvailable"),
            "swap_total_bytes": values.get("SwapTotal"),
            "swap_free_bytes": values.get("SwapFree"),
            "dirty_bytes": values.get("Dirty"),
            "writeback_bytes": values.get("Writeback"),
        }
        result["swap_used_bytes"] = (
            values["SwapTotal"] - values["SwapFree"]
            if "SwapTotal" in values and "SwapFree" in values
            else None
        )
        return result

    def load(self) -> dict:
        fields = (self.proc / "loadavg").read_text().split()
        running, total = (int(value) for value in fields[3].split("/"))
        return {
            "1m": float(fields[0]),
            "5m": float(fields[1]),
            "15m": float(fields[2]),
            "runnable_tasks": running,
            "total_tasks": total,
        }

    def pressure(self) -> dict:
        result = {}
        for resource in ("cpu", "memory", "io"):
            try:
                text = (self.proc / "pressure" / resource).read_text()
            except OSError:
                continue
            entries = {}
            for line in text.splitlines():
                name, *fields = line.split()
                entries[name] = {
                    key: int(value) if key == "total" else float(value)
                    for key, value in (field.split("=", 1) for field in fields)
                }
            result[resource] = entries
        return result

    def processes(self, monotonic_s: float) -> list[dict]:
        result = []
        current = {}
        packaged_commands = {}
        with os.scandir(self.proc) as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    pid = int(entry.name)
                    if pid == os.getpid():
                        continue
                    directory = self.proc / entry.name
                    text = (directory / "stat").read_text()
                    left, right = text.index("("), text.rindex(")")
                    comm = text[left + 1 : right]
                    fields = text[right + 1 :].split()
                    # Read a bounded prefix for classification; never persist it.
                    try:
                        with (directory / "cmdline").open("rb") as command_file:
                            command = command_file.read(4096)
                    except OSError:
                        command = b""
                    cwd = None
                    if any(Path(arg.decode(errors="replace")).name == "main.py"
                           for arg in command.split(b"\0")):
                        try:
                            cwd = str((directory / "cwd").readlink())
                        except OSError:
                            pass
                    kind = process_kind(comm, command, cwd=cwd)
                    if kind is None:
                        continue
                    cpu_ticks = int(fields[11]) + int(fields[12])
                    start_ticks = int(fields[19])
                    previous = self.previous_processes.get(pid)
                    utilization = None
                    if previous and previous[0] == start_ticks:
                        elapsed = monotonic_s - previous[2]
                        delta_ticks = cpu_ticks - previous[1]
                        if elapsed > 0 and delta_ticks >= 0:
                            utilization = round(100.0 * delta_ticks / self.clock_ticks / elapsed, 3)
                    current[pid] = (start_ticks, cpu_ticks, monotonic_s)
                    result.append(
                        {
                            "pid": pid,
                            "ppid": int(fields[1]),
                            "comm": comm,
                            "kind": kind,
                            "role": "worker" if kind in {"autoprintfilm", "nnunet"} else kind,
                            "state": fields[0],
                            "start_time_ticks": start_ticks,
                            "rss_bytes": max(0, int(fields[21])) * self.page_size,
                            "threads": int(fields[17]),
                            "cpu_seconds": cpu_ticks / self.clock_ticks,
                            "cpu_percent": utilization,
                            "processor": int(fields[36]) if len(fields) > 36 else None,
                        }
                    )
                    executable = Path(command.split(b"\0", 1)[0].decode(errors="replace"))
                    if kind == "autoprintfilm" and executable.name.lower() in {
                        "autoprintfilm", "autoprintfilm.exe",
                    }:
                        packaged_commands[pid] = command
                except (OSError, ValueError, IndexError):
                    # Processes can disappear or become unreadable during a scan.
                    continue
        self.previous_processes = current
        by_pid = {record["pid"]: record for record in result}
        for record in result:
            parent = by_pid.get(record["ppid"])
            if record["state"] == "Z":
                record["role"] = "exited"
            elif parent and record["pid"] in packaged_commands and (
                packaged_commands.get(parent["pid"]) == packaged_commands[record["pid"]]
            ):
                # Keep the bootloader's actual resource row, but distinguish it
                # from its identically invoked application child.
                parent["role"] = "launcher"
        return sorted(result, key=lambda item: item["pid"])

    def disk(self) -> dict:
        result = {}
        for name, path in (("root", Path("/")), ("output", self.output)):
            usage = shutil.disk_usage(path)
            result[name] = {"total_bytes": usage.total, "available_bytes": usage.free}
        return result

    def disk_io(self, monotonic_s: float) -> dict:
        """Read whole-device Linux diskstats; sector counters always use 512 bytes."""
        result = []
        current = {}
        indices = {
            "read_ios": 0, "read_sectors": 2, "read_time_ms": 3,
            "write_ios": 4, "write_sectors": 6, "write_time_ms": 7,
            "io_time_ms": 9, "weighted_io_time_ms": 10,
        }
        for line in (self.proc / "diskstats").read_text().splitlines():
            fields = line.split()
            if len(fields) < 14:
                continue
            major, minor, name = fields[:3]
            if name.startswith(("loop", "ram")) or (
                self.sys / "dev/block" / f"{major}:{minor}" / "partition"
            ).exists():
                continue
            values = [int(value) for value in fields[3:14]]
            counters = {key: values[index] for key, index in indices.items()}
            identity = (major, minor, name)
            previous = self.previous_disk_io.get(identity)
            interval = monotonic_s - previous[1] if previous else None
            delta = None
            status = "initial"
            if previous and interval > 0:
                difference = {key: value - previous[0][key] for key, value in counters.items()}
                if all(value >= 0 for value in difference.values()):
                    delta = {
                        "read_bytes": difference["read_sectors"] * 512,
                        "write_bytes": difference["write_sectors"] * 512,
                        "io_time_ms": difference["io_time_ms"],
                        "weighted_io_time_ms": difference["weighted_io_time_ms"],
                        "read_bytes_per_second": round(difference["read_sectors"] * 512 / interval, 3),
                        "write_bytes_per_second": round(difference["write_sectors"] * 512 / interval, 3),
                        "read_iops": round(difference["read_ios"] / interval, 3),
                        "write_iops": round(difference["write_ios"] / interval, 3),
                        "read_await_ms": round(difference["read_time_ms"] / difference["read_ios"], 3)
                        if difference["read_ios"] else None,
                        "write_await_ms": round(difference["write_time_ms"] / difference["write_ios"], 3)
                        if difference["write_ios"] else None,
                        "io_busy_percent": round(difference["io_time_ms"] / (interval * 10), 3),
                        "average_queue_depth": round(difference["weighted_io_time_ms"] / (interval * 1000), 3),
                    }
                    status = "valid"
                else:
                    status = "counter_reset"
            elif previous:
                status = "nonpositive_interval"
            current[identity] = (counters, monotonic_s)
            result.append({
                "name": name, "major": int(major), "minor": int(minor),
                "counters": counters, "in_flight": values[8],
                "interval_seconds": interval, "delta_status": status, "delta": delta,
            })
        self.previous_disk_io = current
        return {"sector_bytes": 512, "devices": result}

    def temperatures(self) -> list[dict]:
        result = []
        paths = list((self.sys / "class/thermal").glob("thermal_zone*/temp"))
        paths.extend((self.sys / "class/hwmon").glob("hwmon*/temp*_input"))
        for path in paths:
            try:
                celsius = int(path.read_text().strip()) / 1000.0
                if not -50 <= celsius <= 200:
                    continue
                result.append({"sensor": str(path.relative_to(self.sys)), "celsius": celsius})
            except (OSError, ValueError):
                continue
        return result

    def gpu(self, monotonic_s: float) -> dict | None:
        if not self.gpu_enabled:
            return None
        if self.last_gpu_poll is None or monotonic_s - self.last_gpu_poll >= self.gpu_interval:
            self.last_gpu_poll = monotonic_s
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                        "power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(
                        f"nvidia-smi exit {completed.returncode}: {completed.stderr.strip()[:160]}"
                    )
                devices = []
                keys = (
                    "index",
                    "utilization_percent",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_watts",
                    "temperature_celsius",
                )
                for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
                    if len(row) != len(keys):
                        raise ValueError("unexpected nvidia-smi column count")
                    device = {}
                    for key, value in zip(keys, row):
                        try:
                            numeric = float(value)
                            device[key] = numeric if math.isfinite(numeric) else None
                        except ValueError:
                            device[key] = None
                    devices.append(device)
                self.gpu_cache = {"devices": devices}
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
                self.gpu_cache = {"devices": None, "error": error_info("gpu", exc)}
        return {
            **self.gpu_cache,
            "sampled_monotonic_s": self.last_gpu_poll,
            "poll_interval_seconds": self.gpu_interval,
            "age_s": round(monotonic_s - self.last_gpu_poll, 3),
        }

    def sample(self, monotonic_s: float, timestamp: datetime) -> dict:
        record = {
            "schema_version": 1,
            "timestamp_utc": timestamp.isoformat(),
            "monotonic_s": monotonic_s,
            "boot_id": self.boot_id,
            "kernel_release": self.kernel_release,
            "errors": [],
        }
        readers = {
            "cpu": self.cpu,
            "memory": self.memory,
            "loadavg": self.load,
            "pressure": self.pressure,
            "processes": lambda: self.processes(monotonic_s),
            "disk": self.disk,
            "disk_io": lambda: self.disk_io(monotonic_s),
            "temperatures": self.temperatures,
            "gpu": lambda: self.gpu(monotonic_s),
        }
        for name, reader in readers.items():
            try:
                record[name] = reader()
            except Exception as exc:
                record[name] = None
                record["errors"].append(error_info(name, exc))
        return record


class DailyJsonlWriter:
    def __init__(self, output: Path):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.day = None

    def write(self, record: dict) -> None:
        day = record["timestamp_utc"][:10]
        if day != self.day:
            self.close()
            self.file = (self.output / f"host-{day}.jsonl").open("a", encoding="utf-8")
            self.day = day
        self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
            self.day = None


def collect(
    output: Path,
    interval: float = 10,
    samples: int | None = None,
    *,
    gpu: bool = False,
    gpu_interval: float = 30,
    logger_config: Path | None = None,
    sampler=None,
    monotonic=time.monotonic,
    now=utc_now,
    sleep=time.sleep,
) -> None:
    writer = None
    active_sampler = None
    pending_sampler = sampler
    try:
        count = 0
        while samples is None or count < samples:
            started = monotonic()
            count += 1
            if logger_config is not None and not debug_logging_enabled(Path(logger_config)):
                if writer is not None:
                    writer.close()
                    writer = None
                active_sampler = None
                if samples is None or count < samples:
                    sleep(max(1.0, interval - (monotonic() - started)))
                continue
            if writer is None:
                writer = DailyJsonlWriter(output)
            if active_sampler is None:
                active_sampler = pending_sampler if pending_sampler is not None else HostSampler(
                    output, gpu=gpu, gpu_interval=gpu_interval,
                )
                pending_sampler = None
            timestamp = now()
            try:
                record = active_sampler.sample(started, timestamp)
            except Exception as exc:
                record = {
                    "schema_version": 1,
                    "timestamp_utc": timestamp.isoformat(),
                    "monotonic_s": started,
                    "boot_id": active_sampler.boot_id,
                    "kernel_release": active_sampler.kernel_release,
                    "errors": [error_info("sample", exc)],
                }
            record["collection_duration_ms"] = round((monotonic() - started) * 1000, 3)
            # Writing is intentionally outside the sampling exception handler.
            # Disk/permission failures must reach systemd as a failed process.
            writer.write(record)
            if samples is None or count < samples:
                sleep(max(0.0, interval - (monotonic() - started)))
    finally:
        if writer is not None:
            writer.close()


def positive_interval(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 1:
        raise argparse.ArgumentTypeError("interval must be finite and >= 1 second")
    return result


def positive_samples(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("samples must be >= 1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="JSONL output directory")
    parser.add_argument(
        "--interval", type=positive_interval, default=10, help="seconds (default 10)"
    )
    parser.add_argument("--samples", type=positive_samples,
                        help="stop after this many loop attempts, including disabled checks")
    parser.add_argument(
        "--gpu", action="store_true", help="enable global NVIDIA GPU counters"
    )
    parser.add_argument(
        "--gpu-interval", type=positive_interval, default=30,
        help="minimum seconds between GPU polls, limited by host interval (default 30)",
    )
    parser.add_argument(
        "--logger-config", type=Path,
        help="gate sampling on debug_log_file_enabled=true in this JSON file (fail closed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        collect(args.output, args.interval, args.samples, gpu=args.gpu,
                gpu_interval=args.gpu_interval, logger_config=args.logger_config)
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        print(f"host diagnostics output failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
