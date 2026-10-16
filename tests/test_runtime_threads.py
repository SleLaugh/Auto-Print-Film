"""Thread defaults must be established before native libraries initialize."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from src import runtime
from src.stages.stage_reconstruction import ReconstructionStage


SOURCE_ROOT = Path(__file__).resolve().parents[1]
THREAD_SETTINGS = (
    *runtime.NATIVE_THREAD_VARIABLES,
    "AUTOPRINTFILM_NATIVE_THREADS",
    "AUTOPRINTFILM_WINDOW_WORKERS",
)


@pytest.mark.parametrize("cpu_count, expected", [(1, "1"), (2, "2"), (32, "4")])
def test_native_defaults_follow_available_cpus(cpu_count, expected):
    environment = {}
    with patch.object(runtime, "_available_cpu_count", return_value=cpu_count):
        effective = runtime.bootstrap_native_threads(environment)

    assert environment == effective
    assert effective["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == expected
    assert all(effective[name] == "1" for name in runtime.NATIVE_THREAD_VARIABLES[1:])


def test_available_cpus_follow_affinity_with_cpu_count_fallback():
    with patch.object(runtime.os, "sched_getaffinity", return_value={2, 4}, create=True):
        assert runtime._available_cpu_count() == 2
    with (
        patch.object(runtime.os, "sched_getaffinity", side_effect=OSError, create=True),
        patch.object(runtime.os, "cpu_count", return_value=6),
    ):
        assert runtime._available_cpu_count() == 6


def test_native_override_only_changes_the_default_itk_limit():
    environment = {"AUTOPRINTFILM_NATIVE_THREADS": "7"}
    effective = runtime.bootstrap_native_threads(environment)
    assert effective["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "7"
    assert effective["OMP_NUM_THREADS"] == "1"


def test_explicit_library_limits_are_preserved_and_bootstrap_is_idempotent():
    environment = {name: str(index + 2) for index, name in enumerate(runtime.NATIVE_THREAD_VARIABLES)}
    environment["AUTOPRINTFILM_NATIVE_THREADS"] = "9"
    original = dict(environment)

    first = runtime.bootstrap_native_threads(environment)
    second = runtime.bootstrap_native_threads(environment)

    assert environment == original
    assert first == second == {name: original[name] for name in runtime.NATIVE_THREAD_VARIABLES}


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "", "three", " 2", True])
def test_invalid_native_override_is_reported_without_partial_mutation(value):
    environment = {"AUTOPRINTFILM_NATIVE_THREADS": value}
    original = dict(environment)
    with pytest.raises(ValueError, match="AUTOPRINTFILM_NATIVE_THREADS"):
        runtime.bootstrap_native_threads(environment)
    assert environment == original


def test_native_library_specific_thread_syntax_is_preserved():
    environment = {"OMP_NUM_THREADS": "4,2", "OPENBLAS_NUM_THREADS": "0"}
    effective = runtime.bootstrap_native_threads(environment)
    assert effective["OMP_NUM_THREADS"] == "4,2"
    assert effective["OPENBLAS_NUM_THREADS"] == "0"


def test_window_workers_use_constructor_then_environment_then_default(monkeypatch):
    monkeypatch.delenv("AUTOPRINTFILM_WINDOW_WORKERS", raising=False)
    assert ReconstructionStage().max_window_workers == 2
    monkeypatch.setenv("AUTOPRINTFILM_WINDOW_WORKERS", "3")
    assert ReconstructionStage().max_window_workers == 3
    assert ReconstructionStage({"max_window_workers": 1}).max_window_workers == 1
    monkeypatch.setenv("AUTOPRINTFILM_WINDOW_WORKERS", "invalid")
    assert ReconstructionStage({"max_window_workers": 4}).max_window_workers == 4
    with pytest.raises(ValueError, match="AUTOPRINTFILM_WINDOW_WORKERS"):
        ReconstructionStage()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "invalid"])
def test_invalid_window_constructor_limit_is_rejected(value):
    with pytest.raises(ValueError, match="max_window_workers"):
        ReconstructionStage({"max_window_workers": value})


@pytest.mark.parametrize("value", ["0", "-1", "1.5", ""])
def test_invalid_window_environment_limit_is_rejected(value):
    with pytest.raises(ValueError, match="AUTOPRINTFILM_WINDOW_WORKERS"):
        runtime.window_worker_count(environ={"AUTOPRINTFILM_WINDOW_WORKERS": value})


def _isolated_environment():
    environment = dict(os.environ)
    for name in THREAD_SETTINGS:
        environment.pop(name, None)
    return environment


@pytest.mark.parametrize("entrypoint", ["main.py", "scripts/frozen_entry.py"])
def test_executable_limits_are_set_before_first_native_library_import(entrypoint, tmp_path):
    environment = _isolated_environment()
    environment["AUTOPRINTFILM_NATIVE_THREADS"] = "3"
    environment["OPENBLAS_NUM_THREADS"] = "2"
    observer = """
import importlib.abc
import json
import os
import runpy
import sys

class ImportObserver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"numpy", "SimpleITK"}:
            print("NATIVE_IMPORT " + json.dumps({
                "module": fullname,
                "limits": {name: os.environ.get(name) for name in (
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                )},
            }), flush=True)

sys.meta_path.insert(0, ImportObserver())
sys.argv = [sys.argv[1], "-i", sys.argv[2], "-o", sys.argv[3], "--scan_thickness", "1"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", observer, str(SOURCE_ROOT / entrypoint),
         str(tmp_path / "missing-dicom"), str(tmp_path / "output")],
        cwd=SOURCE_ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
    )
    # Import the real stages, then stop at read-only preflight: no image is read.
    assert result.returncode == 1, result.stderr
    observations = [
        json.loads(line.removeprefix("NATIVE_IMPORT "))
        for line in result.stdout.splitlines()
        if line.startswith("NATIVE_IMPORT ")
    ]
    assert {item["module"] for item in observations} == {"numpy", "SimpleITK"}
    for observation in observations:
        assert observation["limits"] == {
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "3",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }


def test_importing_main_does_not_bootstrap_or_change_thread_limits():
    environment = _isolated_environment()
    # A library import must not validate or apply CLI-only environment settings.
    environment["AUTOPRINTFILM_NATIVE_THREADS"] = "invalid"
    result = subprocess.run(
        [sys.executable, "-c", "import os; before = dict(os.environ); import main; "
         "assert dict(os.environ) == before"],
        cwd=SOURCE_ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("entrypoint", ["main.py", "scripts/frozen_entry.py"])
@pytest.mark.parametrize("setting", [
    "AUTOPRINTFILM_NATIVE_THREADS", "AUTOPRINTFILM_WINDOW_WORKERS",
])
def test_invalid_thread_configuration_writes_failure_report(entrypoint, setting, tmp_path):
    """Both executable entries must preserve the request's failure contract."""

    environment = _isolated_environment()
    environment.update({setting: "invalid", "PYTHONUTF8": "1"})
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    previous_result = output_dir / "previous-result.txt"
    previous_result.write_text("keep", encoding="utf-8")
    runner = (
        "import runpy,sys; sys.argv=sys.argv[1:]; "
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )

    result = subprocess.run(
        [sys.executable, "-c", runner, str(SOURCE_ROOT / entrypoint),
         "-i", str(tmp_path / "missing-input"), "-o", str(output_dir),
         "--slice_part", "胸椎", "--scan_thickness", "1.5"],
        cwd=SOURCE_ROOT,
        env=environment,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 1, result.stderr
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["Success"] is False
    assert setting in report["Message"]
    assert report["SlicePart"] == "胸椎"
    assert report["ScanThickness"] == 1.5
    assert report["PlaneList"] == []
    assert previous_result.read_text(encoding="utf-8") == "keep"
