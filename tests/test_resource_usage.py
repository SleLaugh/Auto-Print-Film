from pathlib import Path
from unittest.mock import patch

from src.core.resources import resource_usage_summary


def test_linux_snapshot_reports_process_and_system_memory_in_mib(tmp_path):
    (tmp_path / "self").mkdir()
    (tmp_path / "self/status").write_text(
        "Name:\tpython\nVmRSS:\t2048 kB\nVmHWM:\t4096 kB\nThreads:\t12\n",
        encoding="ascii",
    )
    (tmp_path / "meminfo").write_text(
        "MemAvailable: 8192 kB\nSwapTotal: 4096 kB\nSwapFree: 3072 kB\n",
        encoding="ascii",
    )

    with patch("src.core.resources.os.getpid", return_value=123):
        summary = resource_usage_summary(tmp_path)

    assert summary == (
        "pid=123 rss_mib=2.0 peak_rss_mib=4.0 threads=12 "
        "system_available_mib=8.0 system_swap_used_mib=1.0"
    )


def test_unavailable_proc_counters_do_not_break_processing(tmp_path):
    summary = resource_usage_summary(tmp_path)

    assert summary.startswith("pid=")
    assert "rss_mib=" not in summary.replace("peak_rss_mib=", "peak=")
    assert "system_available_mib=" not in summary
    assert "threads=" not in summary


def test_malformed_counters_are_omitted(tmp_path):
    (tmp_path / "self").mkdir()
    (tmp_path / "self/status").write_text(
        "VmRSS: unknown\nVmHWM: 4096 kB\nThreads:\n", encoding="ascii",
    )

    summary = resource_usage_summary(Path(tmp_path))

    assert "peak_rss_mib=4.0" in summary
    assert "rss_mib=" not in summary.replace("peak_rss_mib=", "peak=")
    assert "threads=" not in summary
