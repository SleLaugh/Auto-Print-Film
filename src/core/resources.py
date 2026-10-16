"""Low-cost resource snapshots for diagnosing concurrent CPU jobs."""

import os
from pathlib import Path
import sys


def _read_proc_fields(path: Path, keys: set[str]) -> dict[str, int]:
    """Read selected Linux counters; unavailable diagnostics must not fail a job."""

    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return {}
    fields = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key in keys:
            try:
                fields[key] = int(value.split()[0])
            except (ValueError, IndexError):
                continue
    return fields


def resource_usage_summary(proc_root: Path = Path("/proc")) -> str:
    """Report RSS, process-lifetime peak RSS, native threads and host memory.

    Linux memory fields use KiB. Other platforms still report the PID; if the
    standard resource module is available, include its process-lifetime peak.
    Missing counters are omitted rather than reported as zero.
    """

    values = [f"pid={os.getpid()}"]
    status = _read_proc_fields(proc_root / "self/status", {"VmRSS", "VmHWM", "Threads"})
    for key, name in (("VmRSS", "rss_mib"), ("VmHWM", "peak_rss_mib")):
        if key in status:
            values.append(f"{name}={status[key] / 1024:.1f}")
    if "VmHWM" not in status:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
            values.append(f"peak_rss_mib={peak / divisor:.1f}")
        except (ImportError, OSError, ValueError):
            pass
    if "Threads" in status:
        values.append(f"threads={status['Threads']}")
    memory = _read_proc_fields(
        proc_root / "meminfo", {"MemAvailable", "SwapTotal", "SwapFree"},
    )
    if "MemAvailable" in memory:
        values.append(f"system_available_mib={memory['MemAvailable'] / 1024:.1f}")
    if {"SwapTotal", "SwapFree"} <= memory.keys():
        used = memory["SwapTotal"] - memory["SwapFree"]
        values.append(f"system_swap_used_mib={used / 1024:.1f}")
    return " ".join(values)
