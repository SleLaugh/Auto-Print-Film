"""Small subprocesses exercise actual faulthandler hooks without native workloads."""

import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import textwrap
from unittest.mock import Mock

import pytest

from src import diagnostics


def _run(output, script):
    return subprocess.run(
        [sys.executable, "-u", "-c", textwrap.dedent(script), str(output)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def _markers(path):
    prefix = "[PythonDiagnostics] "
    return [json.loads(line[len(prefix):]) for line in path.read_text().splitlines()
            if line.startswith(prefix)]


def test_each_run_appends_its_own_identity_to_existing_debug_log(tmp_path):
    path = tmp_path / "log" / "DebugLog.txt"
    path.parent.mkdir()
    path.write_text("earlier application debug log\n")
    script = """
        import os, sys
        from src import diagnostics
        assert not {'numpy', 'scipy', 'SimpleITK', 'pydicom'} & sys.modules.keys()
        for name in diagnostics._THREAD_ENVIRONMENT_VARIABLES:
            os.environ[name] = '3'
        os.environ['PRIVATE_TEST_SECRET'] = 'must-not-appear-in-debug-log'
        session = diagnostics.start_task_diagnostics(sys.argv[1])
        assert session is not None
        print(session.path)
        session.close()
        session.close()
    """
    for _ in range(2):
        result = _run(tmp_path, script)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(path)
    assert path.read_text().startswith("earlier application debug log\n")
    records = _markers(path)
    assert [record["phase"] for record in records] == ["start", "close", "start", "close"]
    assert records[0]["run_id"] == records[1]["run_id"]
    assert records[2]["run_id"] == records[3]["run_id"]
    assert records[0]["run_id"] != records[2]["run_id"]
    for record in records[::2]:
        assert record["interval_seconds"] == 60
        assert record["pid"] > 0
        assert record["ppid"] == os.getpid()
        assert record["executable"] == sys.executable
        assert record["python_version"] == sys.version
        assert record["kernel_release"] == platform.release()
        assert record["native_thread_env"] == {
            name: "3" for name in diagnostics._THREAD_ENVIRONMENT_VARIABLES
        }
        if hasattr(os, "sched_getaffinity"):
            assert record["affinity_cpus"] == sorted(os.sched_getaffinity(0))
        assert record["monotonic_seconds"] > 0
        assert record["wall_time"].endswith("+00:00")
        if Path("/proc/sys/kernel/random/boot_id").exists():
            assert record["boot_id"]
    assert "must-not-appear-in-debug-log" not in path.read_text()
    assert "PRIVATE_TEST_SECRET" not in path.read_text()
    assert list(path.parent.iterdir()) == [path]


def test_native_versions_use_loaded_module_dicts_without_importing_packages(tmp_path):
    result = _run(tmp_path, """
        import sys, types
        from src.diagnostics import start_task_diagnostics
        names = ('numpy', 'scipy', 'SimpleITK', 'pydicom')
        assert not set(names) & sys.modules.keys()
        session = start_task_diagnostics(sys.argv[1])
        assert session is not None
        session.record_native_versions()
        assert not set(names) & sys.modules.keys()
        class LazyModule(types.ModuleType):
            def __getattr__(self, name):
                raise AssertionError('must not invoke lazy package attributes')
        for name in names:
            module = LazyModule(name)
            if name != 'SimpleITK':
                module.__version__ = 'loaded-' + name
            sys.modules[name] = module
        session.record_native_versions()
        session.close()
        saved = session.path.read_bytes()
        session.record_native_versions()
        assert session.path.read_bytes() == saved
    """)
    assert result.returncode == 0, result.stderr
    records = _markers(tmp_path / "log/DebugLog.txt")
    assert [record["phase"] for record in records] == [
        "start", "native_libraries", "native_libraries", "close",
    ]
    assert records[1]["versions"] == {}
    assert records[2]["versions"] == {
        "numpy": "loaded-numpy", "scipy": "loaded-scipy",
        "SimpleITK": None, "pydicom": "loaded-pydicom",
    }
    assert len({record["run_id"] for record in records}) == 1


def test_c_watchdog_repeats_stack_during_a_long_step_without_python_thread(tmp_path):
    result = _run(tmp_path, """
        import sys, threading, time
        from src import diagnostics
        before = threading.active_count()
        session = diagnostics.start_task_diagnostics(sys.argv[1], interval_seconds=0.05)
        assert session is not None
        assert threading.active_count() == before
        def long_step():
            time.sleep(0.3)
        long_step()
        session.close()
    """)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "log/DebugLog.txt").read_text()
    assert text.count("Timeout (") >= 2
    assert "in long_step" in text
    assert _markers(tmp_path / "log/DebugLog.txt")[-1]["phase"] == "close"


def test_close_cancels_timer_before_fd_close(tmp_path):
    result = _run(tmp_path, """
        import sys, time
        from src import diagnostics
        session = diagnostics.start_task_diagnostics(sys.argv[1], interval_seconds=0.05)
        assert session is not None
        session.close()
        saved = session.path.read_bytes()
        time.sleep(0.16)
        assert session.path.read_bytes() == saved
        assert session._stream.closed
        session.close()
    """)
    assert result.returncode == 0, result.stderr
    assert "Timeout (" not in (tmp_path / "log/DebugLog.txt").read_text()


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGABRT and core limits")
def test_sigabrt_writes_fatal_traceback_to_debug_log_without_a_core(tmp_path):
    result = _run(tmp_path, """
        import os, resource, sys
        from src.diagnostics import start_task_diagnostics
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        session = start_task_diagnostics(sys.argv[1])
        assert session is not None
        def abort_here():
            os.abort()
        abort_here()
    """)
    assert result.returncode == -signal.SIGABRT
    text = (tmp_path / "log/DebugLog.txt").read_text()
    assert "Fatal Python error: Aborted" in text
    assert "in abort_here" in text
    assert [record["phase"] for record in _markers(tmp_path / "log/DebugLog.txt")] == ["start"]
    assert not list(tmp_path.glob("core*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGABRT and core limits")
def test_sigabrt_after_close_uses_stderr_instead_of_the_closed_debug_fd(tmp_path):
    result = _run(tmp_path, """
        import os, resource, sys
        from src.diagnostics import start_task_diagnostics
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        session = start_task_diagnostics(sys.argv[1])
        assert session is not None
        session.close()
        def abort_after_close():
            os.abort()
        abort_after_close()
    """)
    assert result.returncode == -signal.SIGABRT
    assert "Fatal Python error: Aborted" in result.stderr
    assert "in abort_after_close" in result.stderr
    path = tmp_path / "log/DebugLog.txt"
    assert "Fatal Python error" not in path.read_text()
    assert [record["phase"] for record in _markers(path)] == ["start", "close"]


@pytest.mark.parametrize("failure", ["enable", "timer"])
def test_partial_setup_failure_restores_handler_and_closes_its_file(tmp_path, monkeypatch, failure):
    enable = Mock()
    timer = Mock()
    cancel = Mock()
    monkeypatch.setattr(diagnostics.faulthandler, "enable", enable)
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", timer)
    monkeypatch.setattr(diagnostics.faulthandler, "cancel_dump_traceback_later", cancel)
    if failure == "enable":
        enable.side_effect = [OSError("cannot enable"), None]
    else:
        timer.side_effect = OSError("cannot schedule")
    assert diagnostics.start_task_diagnostics(tmp_path) is None
    stream = enable.call_args_list[0].kwargs["file"]
    assert stream.closed
    assert enable.call_args_list[-1].kwargs == {"file": sys.stderr, "all_threads": True}
    cancel.assert_called_once_with()
    if failure == "timer":
        timer.assert_called_once_with(60.0, repeat=True, file=stream, exit=False)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), -float("inf"), True, "60", None])
def test_invalid_interval_does_not_create_logs_or_install_handlers(tmp_path, monkeypatch, interval):
    enable = Mock()
    timer = Mock()
    monkeypatch.setattr(diagnostics.faulthandler, "enable", enable)
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", timer)
    assert diagnostics.start_task_diagnostics(tmp_path, interval_seconds=interval) is None
    assert not (tmp_path / "log").exists()
    enable.assert_not_called()
    timer.assert_not_called()


def test_unwritable_output_does_not_change_handlers(tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    enable = Mock()
    timer = Mock()
    monkeypatch.setattr(diagnostics.faulthandler, "enable", enable)
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", timer)
    assert diagnostics.start_task_diagnostics(blocked / "output") is None
    enable.assert_not_called()
    timer.assert_not_called()


def test_optional_marker_failure_does_not_fail_task_or_sync_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.faulthandler, "enable", Mock())
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", Mock())
    monkeypatch.setattr(diagnostics.faulthandler, "cancel_dump_traceback_later", Mock())
    sync = Mock(side_effect=AssertionError("diagnostics must not fsync"))
    monkeypatch.setattr(diagnostics.os, "fsync", sync)
    session = diagnostics.start_task_diagnostics(tmp_path)
    assert session is not None
    with monkeypatch.context() as isolated:
        isolated.setattr(diagnostics, "_marker", Mock(side_effect=OSError("log unavailable")))
        session.record_native_versions()
        session.close()
    assert session._stream.closed
    sync.assert_not_called()


def test_close_failure_preserves_processing_exception_and_live_fd(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.faulthandler, "enable", Mock())
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", Mock())
    monkeypatch.setattr(
        diagnostics.faulthandler, "cancel_dump_traceback_later",
        Mock(side_effect=RuntimeError("cannot detach watchdog")),
    )
    session = diagnostics.start_task_diagnostics(tmp_path)
    error = ValueError("original processing error")
    assert session is not None
    try:
        with pytest.raises(ValueError) as caught:
            try:
                raise error
            finally:
                session.close()
        assert caught.value is error
        assert not session._stream.closed
        assert diagnostics._RETAINED_STREAMS[-1] is session._stream
        session.close()
    finally:
        diagnostics._RETAINED_STREAMS.remove(session._stream)
        session._stream.close()
