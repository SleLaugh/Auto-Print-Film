import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.config import LoggingConfig
from src.core.artifacts import ArtifactManager
from src.utils.Logger_utils import LoggerManager, configure_logging


class LoggerUtilsTest(unittest.TestCase):
    def test_configure_logging_places_files_under_output_log(self):
        """三类文件日志应统一绑定到本次任务的 output/log。"""

        manager = LoggerManager()
        original = {
            "base_dir": manager.base_dir,
            "debug_file_enabled": manager.debug_file_enabled,
            "console_enabled": manager.console_enabled,
            "run_log_file_enabled": manager.run_log_file_enabled,
            "debug_log_file_enabled": manager.debug_log_file_enabled,
            "error_log_file_enabled": manager.error_log_file_enabled,
            "log_encoding": manager.log_encoding,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "case-output"
                configure_logging(
                    LoggingConfig(
                        console_enabled=False,
                        debug_log_file_enabled=True,
                        error_log_file_enabled=False,
                    ),
                    output_dir=output_dir,
                )

                manager._write_to_files(
                    manager.LOG_TYPE_DEBUG,
                    "任务启动日志",
                )
                log_path = manager._get_log_file_path(manager.LOG_TYPE_DEBUG)
                ArtifactManager(output_dir).prepare()
                manager._write_to_files(
                    manager.LOG_TYPE_DEBUG,
                    "任务结束日志",
                )

                self.assertEqual(log_path, output_dir / "log" / "DebugLog.txt")
                self.assertEqual(
                    log_path.read_text(encoding="utf-8"),
                    "任务启动日志\n任务结束日志\n",
                )
        finally:
            manager.console_enabled = original["console_enabled"]
            manager.debug_file_enabled = original["debug_file_enabled"]
            manager.run_log_file_enabled = original["run_log_file_enabled"]
            manager.debug_log_file_enabled = original["debug_log_file_enabled"]
            manager.error_log_file_enabled = original["error_log_file_enabled"]
            manager.log_encoding = original["log_encoding"]
            manager.set_base_dir(original["base_dir"])

    def test_exception_log_includes_type_message_and_traceback(self):
        """错误日志应直接给出异常类型、原因和可定位代码行的调用堆栈。"""

        manager = LoggerManager()
        console_logger = Mock()
        with patch.object(manager, "console_enabled", True), patch.object(
            manager,
            "_get_console_logger",
            return_value=console_logger,
        ), patch.object(manager, "_write_to_files"):
            try:
                raise RuntimeError("boom")
            except RuntimeError as error:
                manager.error_log("阶段失败", error)

        content = console_logger.info.call_args.args[0]
        self.assertIn("阶段失败", content)
        self.assertIn("异常类型: RuntimeError", content)
        self.assertIn("异常信息: boom", content)
        self.assertIn("调用堆栈:", content)
        self.assertIn("raise RuntimeError(\"boom\")", content)

    def test_error_log_is_mirrored_to_enabled_run_log(self):
        """只开启 RunLog 时，真实失败原因也必须落入运行日志。"""

        manager = LoggerManager()
        run_logger = Mock()
        with patch.object(manager, "run_log_file_enabled", True), patch.object(
            manager,
            "debug_log_file_enabled",
            False,
        ), patch.object(
            manager,
            "error_log_file_enabled",
            False,
        ), patch.object(
            manager,
            "_get_file_logger",
            return_value=run_logger,
        ) as get_file_logger:
            LoggerManager._write_to_files(
                manager,
                manager.LOG_TYPE_ERROR,
                "structured failure",
            )

        get_file_logger.assert_called_once_with(manager.LOG_TYPE_RUN)
        run_logger.info.assert_called_once_with("structured failure")


if __name__ == "__main__":
    unittest.main()
