"""CLI 的终态耗时应留在任务日志末尾，同时保留原有报告和退出码。"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import main
from src.config import AppConfig, LoggingConfig
from src.utils import Logger_utils as logger_utils


@pytest.fixture
def cli(tmp_path, monkeypatch):
    args = SimpleNamespace(
        input=str(tmp_path / "dicom"), output=str(tmp_path / "output"),
        slice_part="腰椎", study_type="CT", scan_thickness=1.0,
        plane_configs=None, window_presets=None, cache_dir=None,
    )
    events = []
    real_parse_arguments = main.parse_arguments
    monkeypatch.setattr(main.sys, "argv", [
        "main.py", "-i", args.input, "-o", args.output,
        "--scan_thickness", "1.0",
    ])
    monkeypatch.setattr(main, "parse_arguments", Mock(return_value=args))
    monkeypatch.setattr(main, "load_app_config", Mock(return_value=AppConfig(
        logging=LoggingConfig(stack_dump_interval_seconds=60),
    )))
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(main.faulthandler, "enable", Mock())
    monkeypatch.setattr(main.Request, "validate", lambda self: (True, None))
    for name in ("RunLog", "DebugLog", "ErrorLog"):
        monkeypatch.setattr(main, name, Mock(
            side_effect=lambda *values, kind=name: events.append((kind, values)),
        ))
    diagnostics = Mock()

    def close_diagnostics():
        # The final elapsed-time reading must happen after diagnostic cleanup.
        assert clock.call_count == 1
        events.append(("close", ()))

    diagnostics.close.side_effect = close_diagnostics
    monkeypatch.setattr(main, "start_task_diagnostics", Mock(return_value=diagnostics))
    report = Mock(wraps=main._try_write_failure_report)
    monkeypatch.setattr(main, "_try_write_failure_report", report)
    pipeline = Mock()
    pipeline.run.return_value = SimpleNamespace(
        success=True, total_time=3.21, output_dir=args.output, errors=[],
    )
    monkeypatch.setattr(main, "Pipeline", Mock(return_value=pipeline))
    # Replace main's module binding rather than the process-wide time module.
    clock = Mock(side_effect=[100.0, 123.456])
    monkeypatch.setattr(main, "time", SimpleNamespace(monotonic=clock), raising=False)
    return SimpleNamespace(
        args=args, events=events, pipeline=pipeline, diagnostics=diagnostics,
        report=report, clock=clock, real_parse_arguments=real_parse_arguments,
        report_path=tmp_path / "output" / "report.json",
    )


def assert_footer(cli, status, *, diagnostics_closed=True):
    expected = f"{status}，总耗时: 23.46s"
    assert cli.events[-1] == ("RunLog", (expected,))
    duration_messages = [
        values[0] for kind, values in cli.events
        if kind == "RunLog" and "总耗时:" in values[0]
    ]
    assert duration_messages == [expected]
    if diagnostics_closed:
        assert cli.events.index(("close", ())) < len(cli.events) - 1
        cli.diagnostics.close.assert_called_once_with()
    else:
        cli.diagnostics.close.assert_not_called()
    assert cli.clock.call_count == 2


@pytest.mark.parametrize("success", [True, False], ids=["success", "failed-response"])
def test_complete_duration_is_last_and_response_and_report_are_preserved(cli, success):
    """成功和失败均统计完整入口耗时，不改变 Response 及交付报告的原有耗时。"""

    cli.pipeline.run.return_value.success = success
    cli.pipeline.run.return_value.errors = [
        {"stage": "reconstruction", "message": "no usable region"},
    ]
    cli.report_path.parent.mkdir(parents=True)
    pipeline_report = {
        "Success": success, "Message": "pipeline result", "PlaneList": [], "TotalTime": 3.21,
    }
    cli.report_path.write_text(json.dumps(pipeline_report), encoding="utf-8")

    assert main.main() == (0 if success else 1)

    assert json.loads(cli.report_path.read_text(encoding="utf-8")) == pipeline_report
    assert cli.pipeline.run.return_value.total_time == 3.21
    cli.report.assert_not_called()
    assert_footer(cli, "✅ 成功" if success else "❌ 处理失败")
    if success:
        assert ("DebugLog", (f"输出目录: {cli.args.output}",)) in cli.events[:-1]
    else:
        assert ("DebugLog", ("  错误: [reconstruction] no usable region",)) in cli.events[:-1]


@pytest.mark.parametrize("failure", [
    "pipeline-exception", "interrupted", "explicit-exit", "initialization",
    "preflight", "startup",
])
def test_configured_failures_end_with_elapsed_time_and_keep_report(cli, failure):
    """尚无 Response 的可捕获失败必须使用入口时钟，并保留失败原因。"""

    kwargs = {}
    exit_code = 1
    message = "processing failed"
    if failure == "pipeline-exception":
        cli.pipeline.run.side_effect = RuntimeError(message)
    elif failure == "interrupted":
        cli.pipeline.run.side_effect = KeyboardInterrupt()
        exit_code, message = 130, "任务被中断"
    elif failure == "explicit-exit":
        cli.pipeline.run.side_effect = SystemExit(7)
        exit_code, message = 7, "程序主动退出，退出码: 7"
    elif failure == "initialization":
        main.Pipeline.side_effect = RuntimeError(message)
    elif failure == "preflight":
        cli.args.plane_configs = "not-json"
        message = "plane_configs 不是合法 JSON"
    elif failure == "startup":
        kwargs["startup_error"] = ValueError(message)

    assert main.main(**kwargs) == exit_code

    assert_footer(cli, "❌ 处理失败")
    report = json.loads(cli.report_path.read_text(encoding="utf-8"))
    assert report["Success"] is False
    assert message in report["Message"]
    assert report["SlicePart"] == "腰椎"
    assert report["ScanThickness"] == 1.0
    assert report["PlaneList"] == []
    cli.report.assert_called_once()
    if failure in {"initialization", "preflight", "startup"}:
        cli.pipeline.run.assert_not_called()
    else:
        cli.pipeline.run.assert_called_once()
    if failure != "explicit-exit":
        assert any(kind == "ErrorLog" for kind, _ in cli.events[:-1])


@pytest.mark.parametrize("failure", ["arguments", "configuration", "formal-logging"])
def test_early_failures_log_elapsed_time_after_binding_current_directory(cli, failure):
    """先从原始参数绑定本次目录，参数或配置失败也能在本次任务日志收尾。"""

    exit_code = 1
    if failure == "arguments":
        main.parse_arguments.side_effect = main.ArgumentParseError("invalid arguments")
        exit_code = 2
    elif failure == "configuration":
        main.load_app_config.side_effect = ValueError("invalid configuration")
    else:
        main.configure_logging.side_effect = [None, OSError("cannot configure logging")]

    assert main.main() == exit_code

    assert_footer(cli, "❌ 处理失败", diagnostics_closed=False)
    assert main.configure_logging.call_args_list[0] == call(output_dir=cli.args.output)
    cli.report.assert_called_once()
    assert json.loads(cli.report_path.read_text(encoding="utf-8"))["Success"] is False
    cli.pipeline.run.assert_not_called()


@pytest.mark.parametrize("with_output", [True, False], ids=["task-help", "plain-help"])
def test_help_is_successful_and_logs_only_when_task_directory_is_available(
    cli, monkeypatch, capsys, with_output,
):
    """帮助带输出目录时写程序结束耗时，不带时不创建任务日志。"""

    argv = ["main.py", "--help"]
    if with_output:
        argv.extend(["-o", cli.args.output])
    monkeypatch.setattr(main.sys, "argv", argv)
    monkeypatch.setattr(main, "parse_arguments", cli.real_parse_arguments)

    assert main.main() == 0

    assert "SpineMPRProcessor" in capsys.readouterr().out
    if with_output:
        assert_footer(cli, "✅ 程序结束", diagnostics_closed=False)
        main.configure_logging.assert_called_once_with(output_dir=cli.args.output)
    else:
        assert cli.events == []
        main.configure_logging.assert_not_called()
    cli.report.assert_not_called()
    main.load_app_config.assert_not_called()
    main.start_task_diagnostics.assert_not_called()


@pytest.mark.parametrize("exit_code", [0, 1, 7, 130])
def test_footer_write_failure_does_not_replace_original_exit_code(cli, exit_code):
    """日志不可写时尽力收尾，不能把正常、失败或中断的退出码替换成日志异常。"""

    if exit_code == 1:
        cli.pipeline.run.return_value.success = False
    elif exit_code == 7:
        cli.pipeline.run.side_effect = SystemExit(7)
    elif exit_code == 130:
        cli.pipeline.run.side_effect = KeyboardInterrupt()

    def fail_only_on_footer(message):
        if "总耗时:" in message:
            raise OSError("log disk unavailable")

    main.RunLog.side_effect = fail_only_on_footer

    assert main.main() == exit_code

    assert any("总耗时:" in item.args[0] for item in main.RunLog.call_args_list)
    cli.diagnostics.close.assert_called_once_with()


def test_no_available_task_log_does_not_write_footer_to_previous_logger(cli, monkeypatch):
    """没有输出目录且参数失败时，不能借用之前任务的日志写收尾耗时。"""

    monkeypatch.setattr(main.sys, "argv", ["main.py", "--invalid"])
    main.parse_arguments.side_effect = main.ArgumentParseError("invalid arguments")

    assert main.main() == 2

    main.configure_logging.assert_not_called()
    main.RunLog.assert_not_called()
    cli.report.assert_not_called()


def test_real_logger_keeps_second_task_argument_failure_in_second_directory(
    cli, monkeypatch, tmp_path,
):
    """连续执行任务时，第二个任务解析失败也只能追加第二个目录的真实日志。"""

    manager = object.__new__(logger_utils.LoggerManager)
    manager._initialized = True
    manager.base_dir = tmp_path / "unused-logs"
    manager._file_loggers = {}
    manager._console_logger = None
    manager.log_encoding = "utf-8"
    manager.debug_file_enabled = False
    manager.console_enabled = False
    manager.run_log_file_enabled = True
    manager.debug_log_file_enabled = True
    manager.error_log_file_enabled = True

    # LoggerManager uses fixed logging names; preserve any handlers belonging to
    # the process's original manager while this isolated manager owns the names.
    saved_loggers = []
    for kind in ("RunLog", "DebugLog", "ErrorLog"):
        logger = logging.getLogger(f"FileLogger_{kind}")
        saved_loggers.append((logger, list(logger.handlers), logger.level, logger.propagate))
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    monkeypatch.setattr(logger_utils, "_logger_manager", manager)
    for name in ("RunLog", "DebugLog", "ErrorLog", "configure_logging"):
        monkeypatch.setattr(main, name, getattr(logger_utils, name))
    main.load_app_config.return_value = AppConfig(logging=LoggingConfig(
        console_enabled=False, run_log_file_enabled=True,
        debug_log_file_enabled=True, error_log_file_enabled=True,
    ))
    first = tmp_path / "first"
    second = tmp_path / "second"
    cli.args.output = str(first)
    monkeypatch.setattr(main.sys, "argv", [
        "main.py", "-i", cli.args.input, "-o", str(first), "--scan_thickness", "1.0",
    ])
    cli.clock.side_effect = [100.0, 123.456, 200.0, 223.456]
    try:
        assert main.main() == 0
        first_logs = {
            path.name: path.read_text(encoding="utf-8")
            for path in (first / "log").iterdir()
        }
        assert first_logs["RunLog.txt"].rstrip().endswith("✅ 成功，总耗时: 23.46s")
        assert first_logs["DebugLog.txt"].rstrip().endswith("✅ 成功，总耗时: 23.46s")

        monkeypatch.setattr(main, "parse_arguments", cli.real_parse_arguments)
        monkeypatch.setattr(main.sys, "argv", [
            "main.py", "-i", cli.args.input, "-o", str(second),
            "--scan_thickness", "invalid",
        ])

        assert main.main() == 2

        for filename, original in first_logs.items():
            assert (first / "log" / filename).read_text(encoding="utf-8") == original
        for filename in ("RunLog.txt", "DebugLog.txt"):
            text = (second / "log" / filename).read_text(encoding="utf-8")
            assert text.rstrip().endswith("❌ 处理失败，总耗时: 23.46s")
            assert text.count("总耗时:") == 1
            assert "✅ 成功" not in text
        report = json.loads((second / "report.json").read_text(encoding="utf-8"))
        assert report["Success"] is False
        assert "invalid float value" in report["Message"]
    finally:
        for logger, handlers, level, propagate in saved_loggers:
            for handler in list(logger.handlers):
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate
