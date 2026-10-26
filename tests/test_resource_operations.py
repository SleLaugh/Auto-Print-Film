from unittest.mock import patch

import pytest

from src.core.resources import resource_operation
from src.utils.Logger_utils import LoggerManager


def test_operation_logs_resource_boundaries_and_discovered_image_details():
    messages = []
    with patch("src.utils.RunLog", side_effect=messages.append), patch(
        "src.core.resources.resource_usage_summary", return_value="pid=123 rss_mib=32.0"
    ), patch("src.core.resources.time.monotonic", side_effect=[10.0, 11.25]):
        with resource_operation(
            "spatial_correction", "read_segmentation", label="vertebrae_L5"
        ) as details:
            details.update(size_xyz=(32, 24, 12), pixel_type="8-bit unsigned integer")

    assert len(messages) == 2
    assert "stage=spatial_correction step=read_segmentation phase=start" in messages[0]
    assert 'label="vertebrae_L5"' in messages[0]
    assert "size_xyz=" not in messages[0]
    assert "phase=complete" in messages[1]
    assert "elapsed_seconds=1.25" in messages[1]
    assert "size_xyz=[32,24,12]" in messages[1]
    assert 'pixel_type="8-bit unsigned integer"' in messages[1]
    assert all("pid=123 rss_mib=32.0" in message for message in messages)


@pytest.mark.parametrize("error", [RuntimeError("sample failed"), KeyboardInterrupt()])
def test_operation_logs_failure_and_reraises_the_same_exception(error):
    messages = []
    with patch("src.utils.RunLog", side_effect=messages.append), patch(
        "src.core.resources.resource_usage_summary", return_value="pid=123"
    ), patch("src.core.resources.time.monotonic", side_effect=[20.0, 22.5]):
        with pytest.raises(type(error)) as caught:
            with resource_operation("spatial_correction", "estimate_xy_candidate"):
                raise error

    assert caught.value is error
    assert len(messages) == 2
    assert "phase=failed" in messages[1]
    assert "elapsed_seconds=2.5" in messages[1]
    assert f'error_type="{type(error).__name__}"' in messages[1]
    assert not any("phase=complete" in message for message in messages)


def test_disabled_run_logging_does_not_skip_processing_or_create_files(tmp_path, monkeypatch):
    manager = LoggerManager()
    monkeypatch.setattr(manager, "console_enabled", False)
    monkeypatch.setattr(manager, "run_log_file_enabled", False)
    monkeypatch.setattr(manager, "base_dir", tmp_path / "logs")
    processed = []

    with resource_operation("spatial_correction", "index_labels") as details:
        processed.append("indexed")
        details["label_count"] = len(processed)

    assert processed == ["indexed"]
    assert details["label_count"] == 1
    assert not (tmp_path / "logs").exists()


def test_logger_failure_cannot_mask_the_processing_exception():
    error = ValueError("original processing failure")
    with patch("src.utils.RunLog", side_effect=OSError("log directory unavailable")):
        with pytest.raises(ValueError) as caught:
            with resource_operation("spatial_correction", "read_segmentation"):
                raise error

    assert caught.value is error
