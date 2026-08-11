import pytest

from src.utils.Logger_utils import _logger_manager


class _NullConsoleLogger:
    """吞掉测试期间的控制台日志，同时保留日志开关原值。"""

    @staticmethod
    def info(*args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def isolate_application_logs(monkeypatch):
    """禁止测试把模拟阶段和异常写入真实的项目日志目录。"""

    null_console = _NullConsoleLogger()
    monkeypatch.setattr(
        _logger_manager,
        "_get_console_logger",
        lambda: null_console,
    )
    monkeypatch.setattr(
        _logger_manager,
        "_write_to_files",
        lambda *args, **kwargs: None,
    )
