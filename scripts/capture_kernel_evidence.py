#!/usr/bin/env python3
"""保存上次启动的内核证据；仅使用标准库，不修改 journal 或 pstore。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import stat
import subprocess
import sys
import time


COMMAND_TIMEOUT_SECONDS = 8.0
COMMAND_OUTPUT_LIMIT = 32 * 1024 * 1024
COMMAND_ERROR_LIMIT = 64 * 1024
PSTORE_COPY_LIMIT = 32 * 1024 * 1024
PSTORE_ROOT = Path("/var/lib/systemd/pstore")


def _utc_now():
    return datetime.now(timezone.utc)


def _error(error):
    return {"error_type": type(error).__name__, "error": str(error)}


def capture_command(directory, name, command, *, timeout=COMMAND_TIMEOUT_SECONDS,
                    output_limit=COMMAND_OUTPUT_LIMIT):
    """流式保存有大小限制的输出；超时杀掉直接子进程，不积攒输出到内存。"""
    result = {"command": command, "timeout_seconds": timeout, "status": "error"}
    started = time.monotonic()
    cleanup_deadline = started + timeout + 1.0
    process = None
    try:
        with (directory / f"{name}.txt").open("wb") as output, (
            directory / f"{name}.stderr.txt"
        ).open("wb") as errors:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
            )
            streams = {
                process.stdout: [output, output_limit, 0, False],
                process.stderr: [errors, COMMAND_ERROR_LIMIT, 0, False],
            }
            with selectors.DefaultSelector() as selector:
                for stream in streams:
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        result["timed_out"] = True
                        break
                    for key, _ in selector.select(min(remaining, 0.25)):
                        try:
                            chunk = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        state = streams[key.fileobj]
                        kept = chunk[:max(0, state[1] - state[2])]
                        state[0].write(kept)
                        state[2] += len(kept)
                        state[3] |= len(kept) < len(chunk)
            result.update(
                stdout_bytes=streams[process.stdout][2],
                stdout_truncated=streams[process.stdout][3],
                stderr_bytes=streams[process.stderr][2],
                stderr_truncated=streams[process.stderr][3],
            )
            if result.get("timed_out"):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            try:
                result["returncode"] = process.wait(
                    timeout=max(0.0, min(1.0, cleanup_deadline - time.monotonic())),
                )
            except subprocess.TimeoutExpired:
                process.kill()
                result["cleanup_timeout"] = True
            result["status"] = (
                "timeout" if result.get("timed_out") else
                "complete" if result.get("returncode") == 0 else "failed"
            )
    except (OSError, ValueError) as error:
        result.update(_error(error))
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(
                        timeout=max(0.0, min(1.0, cleanup_deadline - time.monotonic())),
                    )
                except subprocess.TimeoutExpired:
                    result["cleanup_timeout"] = True
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def capture_pstore(directory, *, source=PSTORE_ROOT, limit=PSTORE_COPY_LIMIT):
    """仅复制一级编号目录内的普通 dmesg.txt，按 mtime 从新到旧分配总预算。"""
    result = {"source": str(source), "copy_limit_bytes": limit, "copied_bytes": 0,
              "truncated": False, "files": []}
    candidates = []
    try:
        with os.scandir(source) as entries:
            for entry in entries:
                relative = Path(entry.name) / "dmesg.txt"
                item = {"source": str(relative)}
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    path = source / relative
                    info = path.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        item["status"] = "skipped_not_regular"
                        result["files"].append(item)
                        continue
                    item.update(source_bytes=info.st_size, mtime_ns=info.st_mtime_ns)
                    candidates.append((info.st_mtime_ns, str(relative), path, item))
                except FileNotFoundError:
                    continue
                except OSError as error:
                    item.update(status="error", **_error(error))
                    result["files"].append(item)
    except OSError as error:
        result.update(status="unavailable", **_error(error))
        return result

    copy_failed = False
    for _, relative, path, item in sorted(candidates, key=lambda row: (-row[0], row[1])):
        if copy_failed:
            item.update(status="skipped_after_copy_error", copied_bytes=0)
            result["files"].append(item)
            continue
        remaining = limit - result["copied_bytes"]
        if remaining <= 0:
            item.update(status="skipped_budget", copied_bytes=0, truncated=True)
            result["truncated"] = True
            result["files"].append(item)
            continue
        written = 0
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as original:
                if not stat.S_ISREG(os.fstat(original.fileno()).st_mode):
                    raise OSError("source changed to a non-regular file")
                destination = directory / "pstore" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as copied:
                    while written < remaining:
                        chunk = original.read(min(65536, remaining - written))
                        if not chunk:
                            break
                        copied.write(chunk)
                        written += len(chunk)
                    truncated = bool(original.read(1))
                item.update(status="truncated" if truncated else "complete",
                            destination=str(destination.relative_to(directory)),
                            copied_bytes=written, truncated=truncated)
                result["truncated"] |= truncated
        except OSError as error:
            item.update(status="error", copied_bytes=written, **_error(error))
            # A failed buffered write may have written only part of its last chunk.
            # Stop here so another file cannot exceed the aggregate copy budget.
            copy_failed = True
        result["copied_bytes"] += written
        result["files"].append(item)
    result["status"] = "complete" if not any(
        item["status"] == "error" for item in result["files"]
    ) else "partial"
    return result


def capture_evidence(output):
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    directory = output / f"kernel-{_utc_now():%Y%m%dT%H%M%S.%fZ}-{os.getpid()}"
    directory.mkdir(mode=0o750)
    metadata = {
        "captured_at_utc": _utc_now().isoformat(),
        "journal": [
            capture_command(directory, "boots", ["journalctl", "--list-boots", "--no-pager"]),
            capture_command(directory, "previous-boot-kernel", [
                "journalctl", "-b", "-1", "-k", "--no-pager", "-o", "short-iso",
            ]),
        ],
        "pstore": capture_pstore(directory),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="内核证据根目录")
    args = parser.parse_args()
    os.umask(0o027)
    try:
        directory = capture_evidence(args.output)
    except OSError as error:
        print(f"无法保存内核证据: {error}", file=sys.stderr)
        return 1
    print(f"内核证据已保存: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
