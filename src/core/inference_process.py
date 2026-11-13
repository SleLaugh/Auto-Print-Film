"""Bounded nnUNet execution, full logs, and verified POSIX group cleanup."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Mapping, Sequence


_PLATFORM = os.name
LOG_TAIL_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_POLL_SECONDS = 0.05
_TERMINATE_GRACE_SECONDS = 1.0
_KILL_GRACE_SECONDS = 2.0
_FATAL_PYTHON_ERROR = re.compile(
    r"Fatal Python error: (?:Segmentation fault|Aborted|Bus error|"
    r"Illegal instruction|Floating point exception)(?:\s|$)"
)


def _timing_clock():
    try:
        return time.perf_counter()
    except Exception:
        return None


def _record_timing(timings, name, started):
    ended = _timing_clock()
    if started is not None and ended is not None:
        timings[name] = {
            "seconds": round(ended - started, 6),
            "start_monotonic": started, "end_monotonic": ended,
        }
    return ended


class InferenceProcessAborted(RuntimeError):
    """An inference attempt was stopped; retry only after verified cleanup."""

    def __init__(
        self,
        reason: str,
        *,
        timeout_seconds: float,
        stdout: str,
        stderr: str,
        cleanup_succeeded: bool,
        timings: dict | None = None,
    ) -> None:
        self.reason = reason
        self.timeout_seconds = timeout_seconds
        self.stdout = stdout
        self.stderr = stderr
        self.cleanup_succeeded = cleanup_succeeded
        self.timings = timings or {}
        message = {
            "timeout": f"nnUNet inference exceeded {timeout_seconds:g} seconds",
            "native_worker_crash": "nnUNet Python worker reported a fatal native error",
            "cleanup_failed": "nnUNet process group could not be completely stopped",
        }.get(reason, f"nnUNet inference aborted: {reason}")
        if not cleanup_succeeded:
            message += "; process group cleanup could not be verified; do not retry"
        super().__init__(message)


class _LogFollower:
    def __init__(self, path: Path, *, detect_fatal: bool = False) -> None:
        self.path = path
        self.offset = 0
        self.detect_fatal = detect_fatal
        self.fatal_detected = False
        self._fatal_suffix = b""

    def _observe(self, data: bytes) -> None:
        if not self.detect_fatal or self.fatal_detected:
            return
        # tqdm may leave stderr without a newline before faulthandler writes
        # its fatal banner. Match across both progress text and read boundaries.
        scan = self._fatal_suffix + data
        self._fatal_suffix = scan[-256:]
        self.fatal_detected = bool(
            _FATAL_PYTHON_ERROR.search(scan.decode("utf-8", errors="replace"))
        )

    def follow(self) -> None:
        # Regular files avoid the pipe EOF deadlock caused by orphan workers.
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read(_READ_CHUNK_BYTES)
        self.offset += len(data)
        self._observe(data)

    def tail(self) -> str:
        with self.path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - LOG_TAIL_BYTES))
            data = stream.read(LOG_TAIL_BYTES)
        # Catch a fatal message written immediately before the root exited.
        self._observe(data)
        return data.decode("utf-8", errors="replace")


def _group_has_live_processes(process_group: int) -> bool:
    """On Linux, orphan zombies are dead even though killpg(0) finds them."""
    proc_root = Path("/proc")
    if proc_root.is_dir():
        try:
            entries = os.scandir(proc_root)
            with entries:
                for entry in entries:
                    if not entry.name.isdigit():
                        continue
                    try:
                        # comm can contain spaces and ')'; fields follow its
                        # final closing parenthesis: state, ppid, pgrp, ...
                        with open(f"/proc/{entry.name}/stat", "rb") as stat_file:
                            fields = stat_file.read(4096).rsplit(b")", 1)[1].split()
                        if int(fields[2]) == process_group and fields[0] not in (b"Z", b"X"):
                            return True
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    except (OSError, ValueError, IndexError):
                        # Cannot verify a potentially live member: fail closed
                        # using the portable group probe below.
                        break
                else:
                    return False
        except OSError:
            pass
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(process_group: int, signum: int) -> bool:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def _wait_for_cleanup(process: subprocess.Popen, deadline: float) -> bool:
    while True:
        process.poll()  # Reap our direct child without an unbounded wait.
        if not _group_has_live_processes(process.pid) and process.returncode is not None:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_POLL_SECONDS, remaining))


def _cleanup_process_group(process: subprocess.Popen) -> bool:
    process.poll()
    if not _group_has_live_processes(process.pid) and process.returncode is not None:
        return True
    _signal_group(process.pid, signal.SIGTERM)
    if _wait_for_cleanup(process, time.monotonic() + _TERMINATE_GRACE_SECONDS):
        return True
    _signal_group(process.pid, signal.SIGKILL)
    return _wait_for_cleanup(process, time.monotonic() + _KILL_GRACE_SECONDS)


def _cleanup_windows_process_tree(process: subprocess.Popen) -> bool:
    """Best-effort bounded stop; taskkill cannot prove orphan-tree cleanup."""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_KILL_GRACE_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass
    # A root that exited before taskkill can leave undetectable descendants.
    # Never claim it is safe to start a replacement inference on that evidence.
    return False


def _cleanup(process: subprocess.Popen) -> bool:
    if _PLATFORM == "nt":
        return _cleanup_windows_process_tree(process)
    return _cleanup_process_group(process)


def run_inference_process(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run inference with bounded waits and full logs on disk.

    Both raw output files are available while inference runs. The monitoring
    loop never mirrors output to a logger or console, which could block its
    deadline checks when a caller stops reading. Returned output contains at
    most ``LOG_TAIL_BYTES`` bytes per stream, decoded as UTF-8. A
    timeout covers the inference attempt; cleanup has its own bounded grace.
    ``cleanup_succeeded`` is required before retrying an aborted attempt.
    POSIX owns and verifies a process group. Windows preserves normal inference
    and attempts taskkill on abort, but explicitly disables automatic retries
    because it cannot verify an orphaned process tree without a Windows Job.
    """
    if _PLATFORM not in ("posix", "nt"):
        raise NotImplementedError("nnUNet watchdog supports POSIX and Windows")
    if _PLATFORM == "nt" and pass_fds:
        raise ValueError("inheriting cache-lock descriptors requires POSIX")
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite positive number")
    stdout_path, stderr_path = Path(stdout_path), Path(stderr_path)
    if stdout_path.resolve() == stderr_path.resolve():
        raise ValueError("stdout_path and stderr_path must be different files")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = _LogFollower(stdout_path)
    stderr = _LogFollower(stderr_path, detect_fatal=True)
    process = None
    cleanup_performed = False
    timings = {}
    process_started = None
    try:
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process_options = (
                {"start_new_session": True, "pass_fds": pass_fds}
                if _PLATFORM == "posix"
                else {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
            )
            process_started = _timing_clock()
            process = subprocess.Popen(
                command,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                **process_options,
            )
            spawn_finished = _record_timing(timings, "popen", process_started)
            deadline = time.monotonic() + timeout_seconds
            reason = None
            while True:
                stderr.follow()
                if stderr.fatal_detected:
                    reason = "native_worker_crash"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "timeout"
                    break
                if process.poll() is not None:
                    break
                time.sleep(min(_POLL_SECONDS, remaining))
            if _PLATFORM == "nt" and reason is None:
                stderr.tail()
                if stderr.fatal_detected:
                    reason = "native_worker_crash"
            wait_finished = _record_timing(timings, "wait_until_exit_or_abort", spawn_finished)
            # Also stop descendants after a successful main-process exit. They
            # may still own the GPU, log descriptors or an inherited cache lock.
            cleanup_succeeded = (
                _cleanup(process) if _PLATFORM == "posix" or reason is not None else False
            )
            _record_timing(timings, "process_group_cleanup", wait_finished)
            cleanup_performed = True
            stdout_tail, stderr_tail = stdout.tail(), stderr.tail()
            _record_timing(timings, "process_monitor_total", process_started)
            if reason is None and stderr.fatal_detected:
                reason = "native_worker_crash"
            if reason is not None or (_PLATFORM == "posix" and not cleanup_succeeded):
                raise InferenceProcessAborted(
                    reason or "cleanup_failed",
                    timeout_seconds=timeout_seconds,
                    stdout=stdout_tail,
                    stderr=stderr_tail,
                    cleanup_succeeded=cleanup_succeeded,
                    timings=timings,
                )
            result = subprocess.CompletedProcess(command, process.returncode, stdout_tail, stderr_tail)
            result.cleanup_succeeded = cleanup_succeeded
            result.timings = timings
            return result
    except BaseException as error:
        if process is not None and not cleanup_performed:
            cleanup_started = _timing_clock()
            cleanup_succeeded = _cleanup(process)
            _record_timing(timings, "process_group_cleanup", cleanup_started)
            _record_timing(timings, "process_monitor_total", process_started)
            if not cleanup_succeeded:
                raise InferenceProcessAborted(
                    "cleanup_failed",
                    timeout_seconds=timeout_seconds,
                    stdout=stdout.tail(),
                    stderr=stderr.tail(),
                    cleanup_succeeded=False,
                    timings=timings,
                ) from error
        raise
