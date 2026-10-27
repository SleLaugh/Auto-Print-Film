from src.core.resources import resource_usage_summary


def _write_thread_stat(tmp_path, *, name="mpr worker (window)", cpu="7"):
    directory = tmp_path / "thread-self"
    directory.mkdir()
    fields = ["S"] + ["0"] * 35 + [cpu] + ["0"] * 13
    (directory / "stat").write_text(f"456 ({name}) {' '.join(fields)}\n", encoding="ascii")


def test_resource_snapshot_identifies_calling_native_thread_and_its_last_cpu(tmp_path):
    _write_thread_stat(tmp_path)

    summary = resource_usage_summary(tmp_path)

    assert "native_tid=456 cpu=7" in summary


def test_malformed_thread_stat_does_not_break_resource_diagnostics(tmp_path):
    _write_thread_stat(tmp_path, cpu="unknown")

    summary = resource_usage_summary(tmp_path)

    assert "native_tid=" not in summary
    assert "cpu=" not in summary


def test_missing_thread_stat_does_not_report_a_guessed_cpu(tmp_path):
    summary = resource_usage_summary(tmp_path)

    assert "native_tid=" not in summary
    assert "cpu=" not in summary
