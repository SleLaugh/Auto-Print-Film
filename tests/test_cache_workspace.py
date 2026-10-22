"""Lease reclamation tests use only temporary paths and lightweight processes."""

import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

from src.core import cache_workspace as workspace_module
from src.core.cache_workspace import cache_workspace, cleanup_cache_workspaces


def _owner(parent, prefix, connection, release, with_child=False):
    with cache_workspace(Path(parent), prefix) as work:
        (work.path / "partial.bin").write_bytes(b"unfinished")
        child = None
        if with_child:
            ready = Path(parent) / "child-ready"
            stop = Path(parent) / "child-stop"
            program = (
                "import os,sys,time; from pathlib import Path; "
                "os.fstat(int(sys.argv[1])); Path(sys.argv[2]).write_text('ready'); "
                "\nwhile not Path(sys.argv[3]).exists(): time.sleep(0.01)\n"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", program, str(work.pass_fds[0]), str(ready), str(stop)],
                pass_fds=work.pass_fds,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 10
            while not ready.exists():
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Synthetic inheriting child failed to start")
                time.sleep(0.01)
        connection.send({"path": str(work.path), "child_pid": child.pid if child else None})
        try:
            release.wait(30)
        finally:
            if child is not None:
                (Path(parent) / "child-stop").touch()
                child.wait(timeout=10)


def _cleaner(parent, prefix, barrier, connection):
    barrier.wait(timeout=15)
    connection.send(cleanup_cache_workspaces(Path(parent), prefix))


def _start_owner(parent, prefix=".segmentation-", with_child=False):
    mp = multiprocessing.get_context("spawn")
    receiver, sender = mp.Pipe(duplex=False)
    release = mp.Event()
    process = mp.Process(target=_owner, args=(str(parent), prefix, sender, release, with_child))
    process.start()
    sender.close()
    assert receiver.poll(15), "Owner did not create synthetic workspace"
    info = receiver.recv()
    receiver.close()
    return process, release, info


def _stop(process):
    if process.is_alive():
        process.kill()
    process.join(timeout=5)
    assert not process.is_alive()


@pytest.mark.skipif(os.name != "posix", reason="POSIX inheritable descriptor leases")
def test_active_workspace_is_not_reclaimed_and_normal_exit_cleans_it(tmp_path):
    with cache_workspace(tmp_path, ".segmentation-") as work:
        (work.path / "active.bin").write_bytes(b"still writing")
        assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 0
        assert (work.path / "active.bin").is_file()
        assert len(work.pass_fds) == 1
        os.fstat(work.pass_fds[0])
        path = work.path
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGKILL recovery")
def test_killed_owner_leaves_reclaimable_workspace(tmp_path):
    owner, _release, info = _start_owner(tmp_path)
    path = Path(info["path"])
    try:
        assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 0
        _stop(owner)
        assert path.is_dir()
        assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 1
        assert not path.exists()
        assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 0
    finally:
        _stop(owner)


@pytest.mark.skipif(os.name != "posix", reason="POSIX child descriptor inheritance")
def test_dead_owner_live_inheriting_child_keeps_lease_until_child_exits(tmp_path):
    owner, _release, info = _start_owner(tmp_path, with_child=True)
    path = Path(info["path"])
    try:
        _stop(owner)
        os.kill(info["child_pid"], 0)
        assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 0
        assert (path / "partial.bin").is_file()
        (tmp_path / "child-stop").touch()
        deadline = time.monotonic() + 10
        while path.exists() and time.monotonic() < deadline:
            cleanup_cache_workspaces(tmp_path, ".segmentation-")
            time.sleep(0.01)
        assert not path.exists(), "Child exit must release the inherited lease"
    finally:
        (tmp_path / "child-stop").touch()
        _stop(owner)
        try:
            os.kill(info["child_pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX child descriptor inheritance")
def test_context_close_does_not_explicitly_unlock_inherited_child_lease(tmp_path):
    child = None
    try:
        with cache_workspace(tmp_path, ".segmentation-") as work:
            path = work.path
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=work.pass_fds)
        assert path.is_dir(), "Closing the owner fd must retain a child's active workspace"
        assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 0
        child.kill()
        child.wait(timeout=5)
        assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 1
        assert not path.exists()
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX concurrent cleanup")
def test_concurrent_cleaners_remove_abandoned_once_and_keep_active_writer(tmp_path):
    owner, _release, info = _start_owner(tmp_path)
    _stop(owner)
    abandoned = Path(info["path"])
    mp = multiprocessing.get_context("spawn")
    children, receivers = [], []
    try:
        with cache_workspace(tmp_path, ".segmentation-") as active:
            barrier = mp.Barrier(3)
            for _ in range(3):
                receiver, sender = mp.Pipe(duplex=False)
                child = mp.Process(target=_cleaner, args=(str(tmp_path), ".segmentation-", barrier, sender))
                child.start()
                sender.close()
                children.append(child)
                receivers.append(receiver)
            removed = []
            for receiver in receivers:
                assert receiver.poll(15)
                removed.append(receiver.recv())
                receiver.close()
            for child in children:
                child.join(15)
                assert child.exitcode == 0
            assert sum(removed) == 1
            assert not abandoned.exists()
            assert active.path.is_dir()
    finally:
        for child in children:
            _stop(child)


def test_cleanup_preserves_legacy_unknown_metadata_and_symlinks(tmp_path):
    legacy = tmp_path / (".segmentation-" + uuid.uuid4().hex)
    legacy.mkdir()
    (legacy / "partial.bin").write_bytes(b"unknown writer")
    unknown = tmp_path / (".segmentation-" + uuid.uuid4().hex)
    unknown.mkdir()
    (unknown / ".cache-workspace.json").write_text(json.dumps({"schema": "unknown"}))
    (unknown / ".cache-workspace.lease").touch()
    target = tmp_path / "outside"
    target.mkdir()
    link = tmp_path / (".segmentation-" + uuid.uuid4().hex)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        link = None
    assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 0
    assert legacy.is_dir() and unknown.is_dir() and target.is_dir()
    if link is not None:
        assert link.is_symlink()


def test_non_posix_cleanup_is_conservative(tmp_path, monkeypatch):
    unknown = tmp_path / ".segmentation-legacy"
    unknown.mkdir()
    monkeypatch.setattr(workspace_module, "os", SimpleNamespace(name="nt"))
    assert cleanup_cache_workspaces(tmp_path, ".segmentation-") == 0
    assert unknown.is_dir()


def test_non_posix_normal_context_still_cleans_own_workspace(tmp_path, monkeypatch):
    class WindowsOS:
        name = "nt"

        def __getattr__(self, name):
            return getattr(os, name)

    monkeypatch.setattr(workspace_module, "os", WindowsOS())
    with cache_workspace(tmp_path, ".segmentation-") as work:
        assert work.pass_fds == ()
        path = work.path
        (path / "result.bin").write_bytes(b"complete")
    assert not path.exists()


def test_prefix_cannot_escape_workspace_parent(tmp_path):
    for prefix in ("", "../escape", ".", ".."):
        with pytest.raises(ValueError):
            with cache_workspace(tmp_path, prefix):
                pytest.fail("Invalid prefix was accepted")
