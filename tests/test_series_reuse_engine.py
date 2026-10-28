"""External entry point and engine changes must invalidate reuse."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.config import AppConfig, NnUNetConfig
from src.core import series_reuse
from test_series_reuse_end_to_end import _pipeline, _request, lung_reuse_case as lung_reuse_case


def test_external_engine_tracks_actual_package_and_interpreter(tmp_path, monkeypatch):
    package = tmp_path / "nnunetv2"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("version=1", encoding="utf-8")
    calls = []

    def locate(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps([str(package)]))

    monkeypatch.setattr(series_reuse.subprocess, "run", locate)
    config = AppConfig(nnunet=NnUNetConfig(python_executable=sys.executable))
    first = series_reuse._engine_identity(config)
    assert first["generation"] == series_reuse._generation(sys.executable)
    assert first["sources"] == {str(source): series_reuse._hash_file(source)}
    assert calls[0][0][0] == sys.executable
    assert "torch" not in calls[0][0][-1]
    source.write_text("version=2", encoding="utf-8")
    assert series_reuse._engine_identity(config) != first


def test_custom_wrapper_cannot_claim_official_engine_identity(tmp_path, monkeypatch):
    wrapper = tmp_path / "custom.py"
    wrapper.write_text("print('custom inference')", encoding="utf-8")
    config = AppConfig(nnunet=NnUNetConfig(
        python_executable=sys.executable, predict_command=str(wrapper),
    ))
    monkeypatch.setattr(series_reuse.subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "Unverified wrapper must not be probed or replaced",
    ))
    with pytest.raises(ValueError, match="自定义"):
        series_reuse._engine_identity(config)


def test_unavailable_external_package_rejects_identity(monkeypatch):
    monkeypatch.setattr(series_reuse.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="[]"))
    config = AppConfig(nnunet=NnUNetConfig(python_executable=sys.executable))
    with pytest.raises(ValueError, match="包来源"):
        series_reuse._engine_identity(config)


def test_engine_upgrade_misses_full_and_segmentation_cache(lung_reuse_case, monkeypatch):
    parent, inference_calls = lung_reuse_case
    engine = {"version": "v1"}
    monkeypatch.setattr(series_reuse, "_engine_identity", lambda *args: dict(engine))
    first = _pipeline().run(_request(parent, "A"))
    assert first.success, first.errors
    engine["version"] = "v2"
    second = _pipeline().run(_request(parent, "B"))
    assert second.success, second.errors
    assert len(inference_calls) == 2
    assert not second.context.metadata["series_reuse_session"].hit


def test_engine_update_during_reporting_does_not_publish(lung_reuse_case, monkeypatch):
    parent, _ = lung_reuse_case
    engine = {"version": "v1"}
    monkeypatch.setattr(series_reuse, "_engine_identity", lambda *args: dict(engine))
    pipeline = _pipeline()
    reporting = pipeline.stages[-1]
    original = reporting.execute

    def upgrade(context):
        result = original(context)
        engine["version"] = "v2"
        return result

    monkeypatch.setattr(reporting, "execute", upgrade)
    response = pipeline.run(_request(parent, "A"))
    assert response.success, response.errors
    assert not list((Path(parent) / "A" / "cache" / "series_reuse" / "v1").glob(
        "*/*/*/ready.json",
    ))
