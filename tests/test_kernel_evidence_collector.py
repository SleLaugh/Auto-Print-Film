import importlib.util
import json
import os
from pathlib import Path
import sys


def _collector():
    path = Path(__file__).parents[1] / "scripts" / "capture_kernel_evidence.py"
    spec = importlib.util.spec_from_file_location("kernel_evidence_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_streaming_records_truncation_and_timeout(tmp_path):
    collector = _collector()
    result = collector.capture_command(
        tmp_path, "bounded", [sys.executable, "-c", "print('x' * 1000)"], output_limit=32,
    )
    assert result["status"] == "complete"
    assert result["stdout_truncated"] is True
    assert (tmp_path / "bounded.txt").stat().st_size == 32
    result = collector.capture_command(
        tmp_path, "timeout", [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05,
    )
    assert result["status"] == "timeout"
    assert result["returncode"] != 0
    assert result["elapsed_seconds"] < 2
    missing = collector.capture_command(tmp_path, "missing", [str(tmp_path / "no-command")])
    assert missing["status"] == "error"
    assert missing["error_type"] == "FileNotFoundError"


def test_pstore_budget_preserves_numbering_and_newest_first_without_source_changes(tmp_path):
    collector = _collector()
    source, destination = tmp_path / "source", tmp_path / "destination"
    destination.mkdir()
    for name, content, timestamp in (("0", b"older", 1), ("7", b"newest", 3), ("3", b"middle", 2)):
        directory = source / name
        directory.mkdir(parents=True)
        path = directory / "dmesg.txt"
        path.write_bytes(content)
        os.utime(path, ns=(timestamp, timestamp))
        (directory / "other.txt").write_bytes(b"ignored")
    (source / "link").mkdir()
    (source / "link" / "dmesg.txt").symlink_to(source / "7" / "dmesg.txt")
    result = collector.capture_pstore(destination, source=source, limit=8)
    assert result["copied_bytes"] == 8
    assert result["truncated"] is True
    assert (destination / "pstore/7/dmesg.txt").read_bytes() == b"newest"
    assert (destination / "pstore/3/dmesg.txt").read_bytes() == b"mi"
    assert not (destination / "pstore/0").exists()
    assert not list(destination.rglob("other.txt"))
    assert (source / "0/dmesg.txt").read_bytes() == b"older"
    assert (source / "3/dmesg.txt").read_bytes() == b"middle"
    assert (source / "7/dmesg.txt").read_bytes() == b"newest"
    assert {item["status"] for item in result["files"]} >= {
        "complete", "truncated", "skipped_budget", "skipped_not_regular",
    }


def test_missing_commands_and_permission_errors_are_saved_as_metadata(tmp_path, monkeypatch):
    collector = _collector()

    def denied(_):
        raise PermissionError("test pstore unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(collector.os, "scandir", denied)
        result = collector.capture_pstore(tmp_path, source=tmp_path / "private")
    assert result["status"] == "unavailable"
    assert result["error_type"] == "PermissionError"
    captured = []

    def missing_command(directory, name, command, **kwargs):
        captured.append(command)
        return {"status": "error", "error_type": "FileNotFoundError"}

    monkeypatch.setattr(collector, "capture_command", missing_command)
    monkeypatch.setattr(collector, "capture_pstore", lambda directory: result)
    first = collector.capture_evidence(tmp_path / "evidence")
    second = collector.capture_evidence(tmp_path / "evidence")
    assert first != second
    metadata = json.loads((first / "metadata.json").read_text())
    assert metadata["pstore"]["error_type"] == "PermissionError"
    assert all(item["status"] == "error" for item in metadata["journal"])
    assert captured[0] == ["journalctl", "--list-boots", "--no-pager"]
    assert captured[1] == ["journalctl", "-b", "-1", "-k", "--no-pager", "-o", "short-iso"]
