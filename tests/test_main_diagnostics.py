from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main
from src.config import AppConfig, LoggingConfig


@pytest.mark.parametrize("outcome", [
    "success", "failure", "exception", "diagnostics_unavailable", "diagnostics_disabled",
])
def test_cli_closes_stack_diagnostics_without_changing_task_result(tmp_path, monkeypatch, outcome):
    args = SimpleNamespace(
        input=str(tmp_path / "dicom"), output=str(tmp_path / "output"),
        slice_part="腰椎", study_type="CT", scan_thickness=1.0,
        plane_configs=None, window_presets=None, cache_dir=None,
    )
    diagnostic = Mock(path=tmp_path / "output/log/DebugLog.txt")
    start = Mock(return_value=None if outcome == "diagnostics_unavailable" else diagnostic)
    monkeypatch.setattr(main, "start_task_diagnostics", start)
    monkeypatch.setattr(main, "parse_arguments", lambda: args)
    monkeypatch.setattr(main, "load_app_config", lambda: AppConfig(logging=LoggingConfig(
        debug_log_file_enabled=False,
        stack_dump_interval_seconds=0 if outcome == "diagnostics_disabled" else 60,
    )))
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(main, "_try_write_failure_report", Mock())
    monkeypatch.setattr(main.Request, "validate", lambda self: (True, None))
    monkeypatch.setattr(main.faulthandler, "enable", Mock())
    for name in ("RunLog", "DebugLog", "ErrorLog"):
        monkeypatch.setattr(main, name, Mock())
    pipeline = Mock()
    if outcome == "exception":
        pipeline.run.side_effect = RuntimeError("processing failed")
    else:
        pipeline.run.return_value = SimpleNamespace(
            success=outcome != "failure", total_time=0.1,
            output_dir=args.output, errors=[{"stage": "test", "message": "failed"}],
        )
    monkeypatch.setattr(main, "Pipeline", Mock(return_value=pipeline))

    result = main.main()

    assert result == (1 if outcome in {"failure", "exception"} else 0)
    pipeline.run.assert_called_once()
    if outcome == "diagnostics_disabled":
        start.assert_not_called()
    else:
        start.assert_called_once_with(args.output, interval_seconds=60)
    if outcome in {"diagnostics_unavailable", "diagnostics_disabled"}:
        diagnostic.close.assert_not_called()
    else:
        diagnostic.record_native_versions.assert_called_once_with()
        diagnostic.close.assert_called_once_with()
