"""Optional task stack diagnostics using only the Python standard library.

The CLI owns one diagnostics session per process. Stack dumps share DebugLog's
append-only file and can interleave with ordinary logging. No fsync or Python
sampling thread is added. These logs cannot replace kernel crash evidence or
guarantee that the output immediately before a panic reaches disk.
"""

from datetime import datetime, timezone
import faulthandler
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import TextIO
from uuid import uuid4


_THREAD_ENVIRONMENT_VARIABLES = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "AUTOPRINTFILM_NATIVE_THREADS",
    "AUTOPRINTFILM_WINDOW_WORKERS",
)
# If detaching an interpreter hook fails, its descriptor must stay valid rather
# than being closed and potentially reused by unrelated task output.
_RETAINED_STREAMS: list[TextIO] = []


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() or None
    except (OSError, UnicodeError):
        return None


def _marker(stream: TextIO, phase: str, run_id: str, **details) -> None:
    record = {
        "phase": phase,
        "run_id": run_id,
        "pid": os.getpid(),
        "wall_time": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        **details,
    }
    stream.write("\n[PythonDiagnostics] " + json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _restore_stderr_handler() -> bool:
    for stream in (sys.stderr, sys.__stderr__):
        if stream is None:
            continue
        try:
            faulthandler.enable(file=stream, all_threads=True)
            return True
        except Exception:
            pass
    try:
        # A closed/replaced stderr must not leave a handler targeting our fd.
        faulthandler.disable()
        return True
    except Exception:
        return False


class TaskDiagnostics:
    """Own the descriptor used by fatal-error and periodic stack dumps."""

    def __init__(self, path: Path, stream: TextIO, run_id: str) -> None:
        self.path = path
        self._stream = stream
        self._run_id = run_id
        self._closed = False

    def record_native_versions(self) -> None:
        """Record already-loaded version strings without importing native packages."""
        if self._closed:
            return
        try:
            versions = {}
            for name in ("numpy", "scipy", "SimpleITK", "pydicom"):
                module = sys.modules.get(name)
                if module is not None:
                    # Inspect the module dictionary directly; getattr could run
                    # a package's lazy __getattr__ and import additional code.
                    version = vars(module).get("__version__")
                    versions[name] = version if isinstance(version, str) else None
            _marker(self._stream, "native_libraries", self._run_id, versions=versions)
        except Exception:
            pass

    def close(self) -> None:
        """Cancel the C watchdog and detach its fd without masking task errors."""
        if self._closed:
            return
        self._closed = True
        cancelled = False
        try:
            faulthandler.cancel_dump_traceback_later()
            cancelled = True
        except Exception:
            pass
        restored = _restore_stderr_handler()
        if not cancelled or not restored:
            _RETAINED_STREAMS.append(self._stream)
            return
        try:
            _marker(self._stream, "close", self._run_id)
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass


def start_task_diagnostics(
    output_dir: Path | str, *, interval_seconds: float = 60.0,
) -> TaskDiagnostics | None:
    """Append task markers/stacks to DebugLog; unavailable diagnostics are optional."""
    stream = None
    diagnostics = None
    try:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            return None
        path = Path(output_dir) / "log" / "DebugLog.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8")
        run_id = uuid4().hex
        details = {
            "interval_seconds": interval_seconds,
            "ppid": os.getppid(),
            "executable": sys.executable,
            "python_version": sys.version,
            "kernel_release": platform.release(),
            "native_thread_env": {
                name: os.environ[name]
                for name in _THREAD_ENVIRONMENT_VARIABLES
                if name in os.environ
            },
        }
        try:
            details["affinity_cpus"] = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass
        boot_id = _boot_id()
        if boot_id is not None:
            details["boot_id"] = boot_id
        _marker(stream, "start", run_id, **details)
        diagnostics = TaskDiagnostics(path, stream, run_id)
        faulthandler.enable(file=stream, all_threads=True)
        faulthandler.dump_traceback_later(
            interval_seconds, repeat=True, file=stream, exit=False,
        )
        return diagnostics
    except Exception:
        if diagnostics is not None:
            diagnostics.close()
        elif stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        return None
