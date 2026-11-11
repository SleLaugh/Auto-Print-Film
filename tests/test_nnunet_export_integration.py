"""Exercise external interpreter selection, compatibility and frozen packaging."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.core import nnunet_command
from src.stages.stage_segmentation import SegmentationStage


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    "import re\nimport sys\n"
    "from nnunetv2.inference.predict_from_raw_data import predict_entry_point\n"
    "if __name__ == '__main__':\n"
    "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
    "    sys.exit(predict_entry_point())\n"
)


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    script = tmp_path / "adapter path/nnunet_export.py"
    script.parent.mkdir()
    script.write_text("# external adapter fixture\n", encoding="utf-8")
    monkeypatch.setattr(nnunet_command, "_adapter_path", lambda: script)
    return script


def _launcher(tmp_path, shebang, *, source=LAUNCHER):
    command = tmp_path / "nnUNetv2_predict"
    command.write_text(shebang + "\n" + source, encoding="utf-8")
    command.chmod(0o755)
    return command


def test_one_thread_preserves_original_command_without_inspecting_launcher(tmp_path, monkeypatch):
    command = _launcher(tmp_path, "#!/bin/sh", source="exit 0\n")
    monkeypatch.setattr(nnunet_command, "_official_launcher_source", Mock(side_effect=AssertionError))
    assert nnunet_command.nnunet_inference_launch(str(command), "", 1) == ([str(command)], False, "")


def test_explicit_external_python_uses_official_entry_adapter(adapter, tmp_path):
    # Keep a venv symlink rather than canonicalizing it to the base interpreter.
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    prefix, enabled, reason = nnunet_command.nnunet_inference_launch(
        python_executable=str(python), export_resampling_threads=4,
    )
    assert enabled and not reason
    assert prefix == [str(python), str(adapter)]


def test_official_launcher_keeps_shebang_interpreter_symlink_and_flags(adapter, tmp_path):
    python = tmp_path / "venv/bin/python3"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    command = _launcher(tmp_path, f"#!{python} -u")
    prefix, enabled, _ = nnunet_command.nnunet_inference_launch(str(command), "", 3)
    assert enabled
    assert prefix == [str(python), "-u", str(adapter)]


@pytest.mark.parametrize("shebang", ["#!/usr/bin/env python3", "#!/usr/bin/env -S python3 -u"])
def test_env_python_shebang_uses_external_path(adapter, tmp_path, monkeypatch, shebang):
    python = tmp_path / "external-bin/python3"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(python.parent))
    command = _launcher(tmp_path, shebang)
    prefix, enabled, _ = nnunet_command.nnunet_inference_launch(str(command), "", 4)
    assert enabled
    assert prefix == [str(python), *(["-u"] if "-S" in shebang else []), str(adapter)]


def test_env_split_shebang_supports_absolute_python_with_spaces(adapter, tmp_path):
    python = tmp_path / "external venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    command = _launcher(tmp_path, f"#!/usr/bin/env -S '{python}' -u")
    prefix, enabled, _ = nnunet_command.nnunet_inference_launch(str(command), "", 4)
    assert enabled
    assert prefix == [str(python), "-u", str(adapter)]


def test_explicit_interpreter_can_override_official_script_without_shebang(adapter, tmp_path):
    command = _launcher(tmp_path, "# no shebang")
    prefix, enabled, _ = nnunet_command.nnunet_inference_launch(str(command), sys.executable, 4)
    assert enabled
    assert prefix == [sys.executable, str(adapter)]


@pytest.mark.parametrize("shebang,source", [
    ("#!/bin/sh", "exit 0\n"),
    ("#!/usr/bin/env SOMETHING=1 python3", LAUNCHER),
    ("#!/missing/python", LAUNCHER),
    ("#!/usr/bin/env -S python3 -m other_module", LAUNCHER),
    ("#!/usr/bin/env -S python3 -mother_module", LAUNCHER),
    ("#!/usr/bin/env -S python3 -x", LAUNCHER),
    ("#!/usr/bin/env python3", "import os\nos.environ['CUSTOM'] = '1'\n" + LAUNCHER),
    ("#!/usr/bin/env python3", LAUNCHER + "print('custom work')\n"),
])
def test_custom_or_unknown_launchers_remain_unchanged(adapter, tmp_path, shebang, source):
    command = _launcher(tmp_path, shebang, source=source)
    prefix, enabled, reason = nnunet_command.nnunet_inference_launch(str(command), "", 4)
    assert prefix == [str(command)]
    assert not enabled
    assert "nnUNet_python" in reason


def test_custom_python_script_preserves_explicit_interpreter(adapter, tmp_path):
    command = _launcher(tmp_path, "# custom Python", source="print('custom')\n")
    prefix, enabled, reason = nnunet_command.nnunet_inference_launch(str(command), sys.executable, 4)
    assert prefix == [sys.executable, str(command)]
    assert not enabled and reason


def test_missing_adapter_reports_fallback_reason(adapter):
    adapter.unlink()
    prefix, enabled, reason = nnunet_command.nnunet_inference_launch(
        python_executable=sys.executable, export_resampling_threads=4,
    )
    assert prefix == nnunet_command.nnunet_command_prefix(python_executable=sys.executable)
    assert not enabled and "部署包" in reason


def test_stage_passes_only_adapter_thread_environment_and_keeps_predict_arguments(adapter, tmp_path):
    model_paths = {"nnUNet_results": str(tmp_path / "nnUNet_results")}
    stage = SegmentationStage({
        "nnUNet_python": sys.executable, "export_resampling_threads": 3,
        "num_processes_segmentation_export": 2, **model_paths,
    })
    output = tmp_path / "seg.nii.gz"
    with (
        patch("src.stages.stage_segmentation.run_inference_process") as run,
        patch("src.stages.stage_segmentation.RunLog") as log,
        patch.object(stage, "_normalize_output_file"),
    ):
        run.return_value = subprocess.CompletedProcess([], 0, "", "")
        stage._run_nnunet_inference(tmp_path, tmp_path / "raw_0000.nii.gz", output, "294", "3d_fullres", "0", True)
    assert run.call_args.args[0] == [
        sys.executable, str(adapter), "-i", str(tmp_path), "-o", str(tmp_path),
        "-d", "294", "-c", "3d_fullres", "-f", "0", "-step_size", "0.5",
        "-npp", "1", "-nps", "2", "--continue_prediction",
    ]
    assert run.call_args.kwargs["env"][nnunet_command.EXPORT_THREADS_ENV] == "3"
    for key, expected in model_paths.items():
        assert run.call_args.kwargs["env"][key] == expected
    assert any("已启用" in item.args[0] for item in log.call_args_list)


def test_stage_logs_unsupported_command_fallback(adapter, tmp_path):
    command = _launcher(tmp_path, "#!/bin/sh", source="exit 0\n")
    stage = SegmentationStage({"nnUNet_predict_command": str(command), "export_resampling_threads": 4})
    with (
        patch("src.stages.stage_segmentation.run_inference_process") as run,
        patch("src.stages.stage_segmentation.RunLog") as log,
        patch.object(stage, "_normalize_output_file"),
        patch.dict(os.environ, {}, clear=True),
    ):
        run.return_value = subprocess.CompletedProcess([], 0, "", "")
        stage._run_nnunet_inference(tmp_path, tmp_path / "raw_0000.nii.gz", tmp_path / "seg.nii.gz", "294", "3d_fullres", "0", True)
    assert run.call_args.args[0][0] == str(command)
    assert nnunet_command.EXPORT_THREADS_ENV not in run.call_args.kwargs["env"]
    assert any("未启用" in item.args[0] and "nnUNet_python" in item.args[0] for item in log.call_args_list)


def test_direct_stage_default_and_validation_preserve_compatibility():
    assert SegmentationStage({"_fold": "1"})._export_resampling_threads == 1
    for value in (True, 0, -1, 5, 1.5, "1.5"):
        with pytest.raises(ValueError, match="export_resampling_threads"):
            SegmentationStage({"export_resampling_threads": value})


def test_source_and_frozen_adapter_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert nnunet_command._adapter_path() == ROOT / "src/inference/nnunet_export.py"
    monkeypatch.setattr(sys, "frozen", True)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "moved/_internal"), raising=False)
    assert nnunet_command._adapter_path() == tmp_path / "moved/_internal/nnunet_adapter/nnunet_export.py"
    monkeypatch.delattr(sys, "_MEIPASS")
    monkeypatch.setattr(nnunet_command, "application_root", lambda: tmp_path / "moved")
    assert nnunet_command._adapter_path() == tmp_path / "moved/_internal/nnunet_adapter/nnunet_export.py"


def test_frozen_spec_packages_external_script_without_inference_dependencies():
    analysis = Mock(return_value=SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[]))
    runpy.run_path(str(ROOT / "AutoPrintFilm.spec"), init_globals={
        "SPECPATH": str(ROOT), "Analysis": analysis, "PYZ": Mock(), "EXE": Mock(), "COLLECT": Mock(),
    })
    kwargs = analysis.call_args.kwargs
    assert kwargs["datas"] == [
        (str(ROOT / "src/inference" / name), "nnunet_adapter")
        for name in ("nnunet_export.py", "nnunet_gpu_export.py", "nnunet_gpu_resampling.py")
    ]
    assert {"torch", "nnunetv2"}.issubset(kwargs["excludes"])
