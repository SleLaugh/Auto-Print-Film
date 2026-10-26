"""Low-cost resource snapshots for diagnosing concurrent CPU jobs."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator


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


def _log_operation(stage: str, step: str, phase: str, details: dict[str, Any]) -> None:
    """Keep optional diagnostics from changing the operation's outcome."""

    try:
        # Import lazily: resources is also imported while the pipeline is loading.
        from src.utils import RunLog

        fields = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, default=str, separators=(',', ':'))}"
            for key, value in details.items()
        )
        RunLog(
            f"[资源] stage={stage} step={step} phase={phase} "
            f"{fields} {resource_usage_summary()}".strip()
        )
    except Exception:
        # A disabled/unavailable logger must never mask the processing exception.
        pass


@contextmanager
def resource_operation(stage: str, step: str, **details: Any) -> Iterator[dict[str, Any]]:
    """Log a bounded processing step, preserving its result and original exception.

    The yielded details can be extended with image dimensions or other metadata
    discovered by the operation. These additions appear in its completion log.
    """

    operation_details = dict(details)
    started = time.monotonic()
    _log_operation(stage, step, "start", operation_details)
    try:
        yield operation_details
    except BaseException as error:
        _log_operation(stage, step, "failed", {
            **operation_details,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error_type": type(error).__name__,
        })
        raise
    else:
        _log_operation(stage, step, "complete", {
            **operation_details,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
