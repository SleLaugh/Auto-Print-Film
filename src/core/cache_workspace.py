"""Private cache workspaces with leases that survive an inherited POSIX fd.

Only directories created by this helper are reclaimed. Legacy directories without
its marker, symlinks, and active leases are left alone. Windows reclamation is
conservative: without POSIX descriptor inheritance, abandoned workspaces are retained.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Iterator
import uuid


_MARKER = ".cache-workspace.json"
_LEASE = ".cache-workspace.lease"
_SCHEMA = "autoprintfilm-cache-workspace-v1"


@dataclass(frozen=True)
class CacheWorkspace:
    path: Path
    pass_fds: tuple[int, ...]


def _try_lock(fd: int) -> bool:
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise


def _open_regular(path: Path, flags: int) -> int:
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise OSError(f"Cache workspace metadata is not a regular file: {path}")
    return fd


def _owned_workspace(path: Path, prefix: str) -> bool:
    if not re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", path.name):
        return False
    try:
        if not stat.S_ISDIR(path.lstat().st_mode):
            return False
        fd = _open_regular(path / _MARKER, os.O_RDONLY)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            marker = json.load(stream)
        return marker == {"schema": _SCHEMA, "workspace": path.name, "lease": _LEASE}
    except (OSError, ValueError, UnicodeError):
        return False


def _cleanup_workspace(path: Path, prefix: str) -> bool:
    if os.name != "posix" or not _owned_workspace(path, prefix):
        return False
    try:
        fd = _open_regular(path / _LEASE, os.O_RDWR)
    except OSError:
        return False
    try:
        if not _try_lock(fd):
            return False
        # Keep the independent lease locked throughout deletion. Concurrent
        # cleaners either skip it or discover that the directory is gone.
        if not _owned_workspace(path, prefix):
            return False
        shutil.rmtree(path)
        return True
    except OSError:
        # Reclamation is best effort and must not fail an otherwise valid cache hit.
        return False
    finally:
        os.close(fd)


def cleanup_cache_workspaces(parent: Path, prefix: str) -> int:
    """Nonblockingly reclaim abandoned, marked workspaces; return removal count."""

    if os.name != "posix":
        return 0
    parent = Path(parent)
    try:
        children = list(parent.iterdir())
    except OSError:
        return 0
    return sum(_cleanup_workspace(path, prefix) for path in children)


@contextmanager
def cache_workspace(parent: Path, prefix: str) -> Iterator[CacheWorkspace]:
    """Create a private workspace; pass ``pass_fds`` to its POSIX subprocess.

    Closing the owner's descriptor deliberately does not explicitly unlock it:
    an inheriting child can still be writing after its owner exits. Only a fresh,
    independently opened descriptor that acquires the lease may remove the data.
    The marker is written after acquiring the lease, so concurrent cleanup skips
    the incomplete setup window. No generated lock/marker enters nnUNet input.
    """

    if not prefix or Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("Cache workspace prefix must be a nonempty filename prefix")
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (prefix + uuid.uuid4().hex)
    path.mkdir(mode=0o700)
    fd = None
    marked = False
    try:
        fd = _open_regular(path / _LEASE, os.O_RDWR | os.O_CREAT | os.O_EXCL)
        if os.name == "posix" and not _try_lock(fd):
            raise RuntimeError("New cache workspace lease is already locked")
        marker = {"schema": _SCHEMA, "workspace": path.name, "lease": _LEASE}
        with (path / _MARKER).open("x", encoding="utf-8") as stream:
            json.dump(marker, stream)
        marked = True
        yield CacheWorkspace(path=path, pass_fds=(fd,) if os.name == "posix" else ())
    finally:
        if fd is not None:
            os.close(fd)
        if marked and os.name == "posix":
            _cleanup_workspace(path, prefix)
        elif marked:
            # Normal Windows completion retains TemporaryDirectory's cleanup;
            # only scanning other (possibly still active) contexts is disabled.
            shutil.rmtree(path, ignore_errors=True)
        else:
            # No subprocess can receive this workspace before its marker exists.
            shutil.rmtree(path, ignore_errors=True)
