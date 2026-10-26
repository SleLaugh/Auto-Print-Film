"""Exercise watchdog failure modes with real CPU-only Python process groups."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from src.core import inference_process
from src.core.inference_process import InferenceProcessAborted, run_inference_process


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process-group watchdog")


@pytest.fixture
def pid_file(tmp_path):
    path = tmp_path / "children.pids"
    yield path
    # Keep the test suite clean even when a process-cleanup assertion fails.
    if path.exists():
        for value in path.read_text().splitlines():
            try:
                os.kill(int(value), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _run(tmp_path, script, *, timeout=5, pass_fds=(), arguments=()):
    return run_inference_process(
        [sys.executable, "-u", "-c", textwrap.dedent(script), *map(str, arguments)],
        env={**os.environ, "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"},
        timeout_seconds=timeout,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        pass_fds=pass_fds,
    )


def _assert_dead(pid_file, expected_count):
    pids = list(map(int, pid_file.read_text().splitlines()))
    assert len(pids) == expected_count
    for pid in pids:
        stat_file = Path(f"/proc/{pid}/stat")
        if stat_file.exists():
            fields = stat_file.read_bytes().rsplit(b")", 1)[1].split()
            assert fields[0] in (b"Z", b"X"), f"process {pid} is still running"


def test_success_preserves_full_logs_and_exit_status(tmp_path):
    result = _run(tmp_path, "import sys; print('inference complete'); print('details', file=sys.stderr)")
    assert result.returncode == 0
    assert result.stdout == "inference complete\n"
    assert result.stderr == "details\n"
    assert (tmp_path / "stdout.log").read_text() == result.stdout
    assert (tmp_path / "stderr.log").read_text() == result.stderr


def test_raw_logs_are_available_before_process_completes(tmp_path):
    gate = tmp_path / "progress-observed"
    script = """
        import pathlib, sys, time
        print('first step')
        print('stderr progress', file=sys.stderr)
        gate = pathlib.Path(sys.argv[1])
        deadline = time.monotonic() + 2
        while not gate.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gate.exists(), 'progress was buffered until exit'
        print('complete')
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run, tmp_path, script, arguments=(gate,))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stdout_path, stderr_path = tmp_path / "stdout.log", tmp_path / "stderr.log"
            if (stdout_path.exists() and stderr_path.exists()
                    and "first step" in stdout_path.read_text()
                    and "stderr progress" in stderr_path.read_text()):
                assert not future.done()
                gate.touch()
                break
            time.sleep(0.01)
        result = future.result(timeout=5)
    assert result.returncode == 0
    assert "first step" in result.stdout
    assert "stderr progress" in result.stderr


def test_monitor_delay_cannot_accept_success_after_deadline(tmp_path, monkeypatch):
    original_follow = inference_process._LogFollower.follow

    def delayed_follow(follower):
        time.sleep(0.3)
        original_follow(follower)

    monkeypatch.setattr(inference_process._LogFollower, "follow", delayed_follow)
    with pytest.raises(InferenceProcessAborted) as caught:
        _run(tmp_path, "print('finished before monitor resumes')", timeout=0.1)
    assert caught.value.reason == "timeout"
    assert caught.value.cleanup_succeeded is True


def test_unread_console_pipe_cannot_block_inference_timeout(tmp_path, pid_file):
    """Exercise the real stage with more output than its unread console can hold."""
    predictor = tmp_path / "fake_predict.py"
    predictor.write_text(
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()) + '\\n')\n"
        "sys.stdout.write('x' * 1024 * 1024); sys.stdout.flush()\n"
        "sys.stderr.write('y' * 1024 * 1024); sys.stderr.flush()\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    marker = tmp_path / "result.json"
    wrapper = textwrap.dedent("""
        import json, pathlib, sys
        import src.stages.stage_segmentation as module
        from src.core.errors import ExternalProcessError
        from src.utils.Logger_utils import _logger_manager
        # Keep the stage's real console DebugLog path. Error reporting happens
        # after cleanup; suppress its large tail so this test isolates whether
        # console backpressure can stop the active inference monitor.
        _logger_manager._write_to_files = lambda *args, **kwargs: None
        module.ErrorLog = lambda *args, **kwargs: None
        root, predictor, marker = map(pathlib.Path, sys.argv[1:])
        stage = module.SegmentationStage({
            'nnUNet_predict_command': str(predictor), 'nnUNet_python': sys.executable,
            'inference_timeout_seconds': 0.4, 'inference_max_retries': 0,
        })
        try:
            stage._run_nnunet_inference(
                root, root / 'raw_0000.nii.gz', root / 'seg.nii.gz',
                292, '3d_fullres', '0', True,
            )
        except ExternalProcessError as error:
            marker.write_text(json.dumps(error.details))
        else:
            raise AssertionError('hanging predictor unexpectedly succeeded')
    """)
    with subprocess.Popen(
        [sys.executable, "-u", "-c", wrapper, str(tmp_path), str(predictor), str(marker)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            # Do not drain stdout until the wrapper has finished: the old
            # synchronous mirroring blocked here before checking its deadline.
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    details = json.loads(marker.read_text())
    assert details["reason"] == "inference_timeout"
    assert details["cleanup_succeeded"] is True
    assert next((tmp_path / "log").glob("*.stdout.log")).stat().st_size >= 1024 * 1024
    _assert_dead(pid_file, 1)


def test_output_tails_are_bounded_while_disk_logs_are_complete(tmp_path):
    length = inference_process.LOG_TAIL_BYTES * 5
    result = _run(tmp_path, """
        import sys
        length = int(sys.argv[1])
        sys.stdout.write('x' * length + '\\nstdout end\\n')
        sys.stderr.write('y' * length + '\\nstderr end\\n')
    """, arguments=(length,))
    assert len(result.stdout.encode()) == inference_process.LOG_TAIL_BYTES
    assert len(result.stderr.encode()) == inference_process.LOG_TAIL_BYTES
    assert result.stdout.endswith("stdout end\n")
    assert result.stderr.endswith("stderr end\n")
    assert (tmp_path / "stdout.log").stat().st_size > length
    assert (tmp_path / "stderr.log").stat().st_size > length


@pytest.mark.parametrize("fatal", [
    "Segmentation fault", "Aborted", "Bus error", "Illegal instruction", "Floating point exception",
])
def test_fatal_worker_message_aborts_hanging_main_process(tmp_path, pid_file, fatal):
    started = time.monotonic()
    with pytest.raises(InferenceProcessAborted) as caught:
        _run(tmp_path, """
            import os, pathlib, subprocess, sys, time
            path = pathlib.Path(sys.argv[1])
            with path.open('a') as stream:
                stream.write(str(os.getpid()) + '\\n')
            worker = (
                "import os, pathlib, sys; "
                "path=pathlib.Path(sys.argv[1]); "
                "path.open('a').write(str(os.getpid()) + '\\\\n'); "
                "print('Fatal Python error: ' + sys.argv[2], file=sys.stderr, flush=True)"
            )
            subprocess.Popen([sys.executable, '-u', '-c', worker, str(path), sys.argv[2]])
            time.sleep(60)
        """, timeout=10, arguments=(pid_file, fatal))
    error = caught.value
    assert error.reason == "native_worker_crash"
    assert error.cleanup_succeeded is True
    assert "Fatal Python error: " + fatal in error.stderr
    assert time.monotonic() - started < 5
    _assert_dead(pid_file, 2)


@pytest.mark.parametrize("progress", ["", "inference\r50%", "x" * 1024 + "\r50%"])
def test_actual_worker_sigsegv_is_reported_by_faulthandler(tmp_path, pid_file, progress):
    worker = (
        "import os, pathlib, resource, signal, sys; "
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
        "sys.stderr.write(sys.argv[2]); sys.stderr.flush(); "
        "pathlib.Path(sys.argv[1]).open('a').write(str(os.getpid()) + '\\n'); "
        "os.kill(os.getpid(), signal.SIGSEGV)"
    )
    started = time.monotonic()
    with pytest.raises(InferenceProcessAborted) as caught:
        _run(tmp_path, """
            import os, pathlib, subprocess, sys, time
            path = pathlib.Path(sys.argv[1])
            with path.open('a') as stream:
                stream.write(str(os.getpid()) + '\\n')
            subprocess.Popen([sys.executable, '-u', '-c', sys.argv[2], str(path), sys.argv[3]])
            time.sleep(60)
        """, timeout=10, arguments=(pid_file, worker, progress))
    assert caught.value.reason == "native_worker_crash"
    assert caught.value.cleanup_succeeded is True
    assert "Fatal Python error: Segmentation fault" in caught.value.stderr
    assert time.monotonic() - started < 5
    _assert_dead(pid_file, 2)


def test_unrecognized_fatal_text_does_not_abort(tmp_path):
    result = _run(tmp_path, """
        import sys
        print('Fatal Python error: unknown exception', file=sys.stderr)
    """)
    assert result.returncode == 0


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux zombie-state check")
def test_timeout_kills_sigterm_ignoring_children_and_grandchildren(tmp_path, pid_file):
    grandchild = (
        "import os, pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).open('a').write(str(os.getpid()) + '\\n'); "
        "time.sleep(60)"
    )
    child = (
        "import os, pathlib, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).open('a').write(str(os.getpid()) + '\\n'); "
        f"subprocess.Popen([sys.executable, '-u', '-c', {grandchild!r}, sys.argv[1]]); "
        "time.sleep(60)"
    )
    started = time.monotonic()
    with pytest.raises(InferenceProcessAborted) as caught:
        _run(tmp_path, """
            import os, pathlib, signal, subprocess, sys, time
            path = pathlib.Path(sys.argv[1])
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            with path.open('a') as stream:
                stream.write(str(os.getpid()) + '\\n')
            subprocess.Popen([sys.executable, '-u', '-c', sys.argv[2], str(path)])
            print('waiting for workers')
            time.sleep(60)
        """, timeout=0.6, arguments=(pid_file, child))
    error = caught.value
    assert error.reason == "timeout"
    assert error.timeout_seconds == 0.6
    assert error.cleanup_succeeded is True
    assert "waiting for workers" in error.stdout
    assert time.monotonic() - started < 5
    _assert_dead(pid_file, 3)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux zombie-state check")
def test_parent_exit_cleans_descendant_holding_output_descriptors(tmp_path, pid_file):
    started = time.monotonic()
    result = _run(tmp_path, """
        import os, pathlib, subprocess, sys, time
        path = pathlib.Path(sys.argv[1])
        with path.open('a') as stream:
            stream.write(str(os.getpid()) + '\\n')
        worker = "import os, pathlib, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).open('a').write(str(os.getpid()) + '\\\\n'); print('worker ready'); time.sleep(60)"
        subprocess.Popen([sys.executable, '-u', '-c', worker, str(path)])
        while len(path.read_text().splitlines()) < 2:
            time.sleep(0.01)
        print('parent complete')
    """, timeout=10, arguments=(pid_file,))
    assert result.returncode == 0
    assert "parent complete" in result.stdout
    assert time.monotonic() - started < 5
    _assert_dead(pid_file, 2)


@pytest.mark.parametrize("exception_type", [LookupError, KeyboardInterrupt])
def test_monitor_exception_also_cleans_process_group(tmp_path, pid_file, exception_type, monkeypatch):
    original_follow = inference_process._LogFollower.follow

    def fail(follower):
        original_follow(follower)
        if pid_file.exists():
            raise exception_type("monitor failed")

    monkeypatch.setattr(inference_process._LogFollower, "follow", fail)
    with pytest.raises(exception_type, match="monitor failed"):
        _run(tmp_path, """
            import os, pathlib, sys, time
            pathlib.Path(sys.argv[1]).write_text(str(os.getpid()) + '\\n')
            time.sleep(60)
        """, arguments=(pid_file,))
    _assert_dead(pid_file, 1)


def test_nonzero_exit_is_returned_after_cleanup(tmp_path):
    result = _run(tmp_path, "import sys; print('failed', file=sys.stderr); sys.exit(7)")
    assert result.returncode == 7
    assert result.stderr == "failed\n"


def test_pass_fds_reaches_the_inference_child(tmp_path):
    shared = tmp_path / "inherited-descriptor"
    with shared.open("wb") as stream:
        result = _run(tmp_path, """
            import os, sys
            os.write(int(sys.argv[1]), b'inherited')
        """, arguments=(stream.fileno(),), pass_fds=(stream.fileno(),))
    assert result.returncode == 0
    assert shared.read_bytes() == b"inherited"


def test_cleanup_failure_prevents_success_or_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(inference_process, "_cleanup_process_group", lambda _process: False)
    with pytest.raises(InferenceProcessAborted) as caught:
        _run(tmp_path, "print('complete')")
    assert caught.value.reason == "cleanup_failed"
    assert caught.value.cleanup_succeeded is False
    assert "do not retry" in str(caught.value)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout_fails_before_spawning(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite positive"):
        _run(tmp_path, "raise AssertionError('must not run')", timeout=timeout)
    assert not (tmp_path / "stdout.log").exists()


def test_windows_preserves_success_but_disables_unverified_tree_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(inference_process, "_PLATFORM", "nt")
    process = Mock(pid=2345, returncode=0)
    process.poll.return_value = 0
    spawn = Mock(return_value=process)
    monkeypatch.setattr(inference_process.subprocess, "Popen", spawn)
    result = _run(tmp_path, "print('normal Windows inference')")
    assert result.returncode == 0
    assert result.cleanup_succeeded is False
    assert spawn.call_args.kwargs["creationflags"] == 0x200
    assert "start_new_session" not in spawn.call_args.kwargs
    assert "pass_fds" not in spawn.call_args.kwargs


def test_windows_tree_cleanup_is_bounded_and_never_claims_verification(monkeypatch):
    process = Mock(pid=2345)
    process.poll.return_value = None
    taskkill = Mock(side_effect=subprocess.TimeoutExpired(["taskkill"], 2))
    monkeypatch.setattr(inference_process.subprocess, "run", taskkill)
    assert inference_process._cleanup_windows_process_tree(process) is False
    assert taskkill.call_args.args[0] == ["taskkill", "/PID", "2345", "/T", "/F"]
    assert taskkill.call_args.kwargs["timeout"] == inference_process._KILL_GRACE_SECONDS
    process.kill.assert_called_once()
    process.wait.assert_called_once_with(timeout=inference_process._TERMINATE_GRACE_SECONDS)
