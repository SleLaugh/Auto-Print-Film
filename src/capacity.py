"""Small debug-only events for offline capacity analysis, without resource polling.

One session describes Pipeline.run, not the CLI's complete process lifetime.
All event I/O is optional: failures must preserve the processing result. The
host sampler supplies CPU/GPU data independently of these task boundaries.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4


_CURRENT = ContextVar("capacity_session", default=None)


def sequence_key(path) -> str:
    """Match the dispatcher's normalized cache identity without logging its path."""
    normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class CapacitySession:
    def __init__(self, request):
        self.stream = None
        self.run_id = uuid4().hex
        self.started = time.monotonic()
        self.status = "aborted"
        self.details = {}
        try:
            self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
        except (OSError, UnicodeError):
            self.boot_id = None
        try:
            self.identity = {
                "series_key": sequence_key(request.cache_dir or request.dicom_dir),
                "slice_part": getattr(request.slice_part, "value", request.slice_part),
                "study_type": getattr(request.study_type, "value", request.study_type),
            }
            path = Path(request.output_dir) / "log" / "DebugLog.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = path.open("a", encoding="utf-8")
            self.emit("task_start", scope="pipeline_run")
        except Exception:
            self.identity = {}

    def emit(self, event, **fields):
        if self.stream is None:
            return None
        try:
            record = {
                "schema_version": 1,
                "event": event,
                "run_id": self.run_id,
                "boot_id": self.boot_id,
                "pid": os.getpid(),
                "wall_time": datetime.now(timezone.utc).isoformat(),
                "monotonic_s": time.monotonic(),
                **self.identity,
                **fields,
            }
            self.stream.write("[Capacity] " + json.dumps(record, ensure_ascii=False) + "\n")
            self.stream.flush()
            return record
        except Exception:
            return None

    def close(self):
        try:
            self.emit(
                "task_end", status=self.status,
                duration_seconds=round(time.monotonic() - self.started, 6),
                **self.details,
            )
        finally:
            if self.stream is not None:
                try:
                    self.stream.close()
                except Exception:
                    pass


@contextmanager
def capacity_session(request, *, enabled):
    session = None
    if enabled:
        try:
            session = CapacitySession(request)
        except Exception:
            pass
    token = _CURRENT.set(session)
    try:
        yield session
    except BaseException as error:
        if session is not None:
            session.status = "failed" if isinstance(error, Exception) else "aborted"
            session.details["error_type"] = type(error).__name__
        raise
    finally:
        _CURRENT.reset(token)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def capacity_enabled():
    return _CURRENT.get() is not None


def record_capacity_details(**details):
    session = _CURRENT.get()
    if session is not None:
        session.details.update(details)


@contextmanager
def capacity_inference(*, model_id, attempt):
    """Mark an actual subprocess attempt, excluding the cache-lock wait."""
    session = _CURRENT.get()
    result = {}
    if session is None:
        yield result
        return
    started = time.monotonic()
    session.emit("inference_start", model_id=str(model_id), attempt=attempt)
    try:
        yield result
    except BaseException as error:
        result["outcome"] = "failed" if isinstance(error, Exception) else "aborted"
        result["error_type"] = type(error).__name__
        raise
    finally:
        session.emit(
            "inference_end", model_id=str(model_id), attempt=attempt,
            duration_seconds=round(time.monotonic() - started, 6),
            outcome=result.get("outcome", "unknown"),
            **({"error_type": result["error_type"]} if "error_type" in result else {}),
        )
