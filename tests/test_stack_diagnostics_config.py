import math

import pytest

from src.config import LoggingConfig, load_app_config


def test_stack_sampling_defaults_off_and_can_run_without_verbose_debug_logging():
    assert LoggingConfig().stack_dump_interval_seconds == 0
    config = load_app_config(environ={}, defaults={"logging": {
        "debug_log_file_enabled": False, "stack_dump_interval_seconds": "60",
    }})
    assert not config.logging.debug_log_file_enabled
    assert config.logging.stack_dump_interval_seconds == 60


@pytest.mark.parametrize("interval", [-1, 0.1, 9, True, math.nan, math.inf, "invalid"])
def test_invalid_or_excessively_frequent_stack_sampling_is_rejected(interval):
    with pytest.raises(ValueError, match="stack_dump_interval_seconds"):
        load_app_config(environ={}, defaults={"logging": {
            "stack_dump_interval_seconds": interval,
        }})


@pytest.mark.parametrize("interval", [0, 10, 60, 120.5])
def test_valid_stack_intervals_are_preserved(interval):
    assert LoggingConfig(stack_dump_interval_seconds=interval).stack_dump_interval_seconds == interval
