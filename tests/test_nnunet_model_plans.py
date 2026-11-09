"""Custom plans must select the intended weights throughout inference and retries."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import NnUNetConfig, load_app_config, load_slice_part_config, validate_slice_part_config
from src.core.context import ProcessingContext
from src.core.inference_process import InferenceProcessAborted
from src.core.pipeline import Pipeline
from src.models import Request
from src.stages.stage_segmentation import SegmentationStage


def _model_config(overrides):
    config = load_slice_part_config()
    config["slice_parts"]["腰椎"]["model_rules"]["CT"][0] = {
        "scan_thickness_range_mm": {"min": 0.0, "max": 1.5},
        "model_id": 29222,
        **overrides,
    }
    return config


def test_lumbar_custom_model_uses_own_plans_and_cache_on_retry(tmp_path):
    roots = {"nnUNet_results": str(tmp_path / "nnUNet_results")}
    for name in ("Dataset292_Liver", "Dataset29222_ctver"):
        (Path(roots["nnUNet_results"]) / name).mkdir(parents=True)
    stage = SegmentationStage({
        **roots, "inference_max_retries": 1,
        "slice_part_config": _model_config({
            "configuration": "3d_lowres", "fold": "all", "plans": "nnUNetPlans_Lowres3mm",
        }),
    })
    source = tmp_path / "nifti/raw_0000.nii.gz"
    source.parent.mkdir()
    source.touch()
    old_cache = tmp_path / "cache/292/segmentation/seg.nii.gz"
    old_cache.parent.mkdir(parents=True)
    old_cache.write_bytes(b"old model segmentation")
    context = ProcessingContext(Request(
        dicom_dir=str(tmp_path / "dicom"), output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"), scan_thickness=1.0,
    ))
    context.set_result("preprocessing", {"nifti_path": str(source)})
    commands = []

    def infer(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            raise InferenceProcessAborted(
                reason="timeout", timeout_seconds=1800, stdout="", stderr="",
                cleanup_succeeded=True,
            )
        output = Path(command[command.index("-o") + 1]) / "raw.nii.gz"
        output.write_bytes(b"new model segmentation")
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("src.stages.stage_segmentation.run_inference_process", side_effect=infer):
        stage.execute(context)

    assert len(commands) == 2
    for command in commands:
        for flag, value in (("-d", "Dataset29222_ctver"), ("-c", "3d_lowres"),
                            ("-f", "all"), ("-p", "nnUNetPlans_Lowres3mm")):
            assert command[command.index(flag) + 1] == value
    result = context.get_result("segmentation")
    assert Path(result["seg_path"]) == tmp_path / "cache/29222/segmentation/seg.nii.gz"
    assert Path(result["seg_path"]).read_bytes() == b"new model segmentation"
    assert result["model"]["plans"] == "nnUNetPlans_Lowres3mm"
    assert old_cache.read_bytes() == b"old model segmentation"
    old_model = stage._match_model("颈椎", "CT", 1.0)
    assert (old_model["configuration"], old_model["fold"], old_model["plans"]) == (
        "3d_fullres", "0", "nnUNetPlans",
    )


@pytest.mark.parametrize("plans", [None, 12, "", " ", " ../other", "../other", "a\\b", "a__b"])
def test_model_rules_reject_invalid_plans_names(plans):
    config = load_slice_part_config()
    config["slice_parts"]["腰椎"]["model_rules"]["CT"][0]["plans"] = plans
    with pytest.raises(ValueError, match="plans"):
        validate_slice_part_config(config)
    with pytest.raises(ValueError, match="plans"):
        NnUNetConfig(plans=plans)


@pytest.mark.parametrize("overrides,expected", [
    ({}, ("3d_fullres", "2", "GlobalPlans")),
    ({"configuration": "3d_lowres"}, ("3d_lowres", "2", "GlobalPlans")),
    ({"fold": "all"}, ("3d_fullres", "all", "GlobalPlans")),
    ({"fold": 0}, ("3d_fullres", "0", "GlobalPlans")),
    ({"plans": "CustomPlans"}, ("3d_fullres", "2", "CustomPlans")),
    ({"configuration": "3d_lowres", "fold": "all", "plans": "CustomPlans"},
     ("3d_lowres", "all", "CustomPlans")),
])
def test_model_fields_independently_override_file_defaults_in_prediction(tmp_path, overrides, expected):
    defaults = {
        "_configuration": "3d_fullres", "_fold": "2", "_plans": "GlobalPlans",
        "nnUNet_results": str(tmp_path / "nnUNet_results"),
    }
    path = tmp_path / "nnuNet_config.json"
    path.write_text(json.dumps(defaults))
    path.with_name("slice_part_config.json").write_text(json.dumps(_model_config(overrides)))
    config = load_app_config(default_path=path, environ={})
    assert config.summary()["nnunet"]["plans"] == "GlobalPlans"
    stage = Pipeline._default_stages(config)[1]
    source = tmp_path / "input/raw_0000.nii.gz"
    source.parent.mkdir()
    source.touch()
    context = ProcessingContext(Request(
        dicom_dir=str(tmp_path / "dicom"), output_dir=str(tmp_path / "output"),
        scan_thickness=1.0,
    ))
    context.set_result("preprocessing", {"nifti_path": str(source)})

    def infer(command, **kwargs):
        output = Path(command[command.index("-o") + 1]) / "raw.nii.gz"
        output.write_bytes(b"complete prediction")
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("src.stages.stage_segmentation.run_inference_process", side_effect=infer) as run:
        stage.execute(context)
    run.assert_called_once()
    command = run.call_args.args[0]
    assert tuple(command[command.index(flag) + 1] for flag in ("-c", "-f", "-p")) == expected
    model = context.get_result("segmentation")["model"]
    assert tuple(model[key] for key in ("configuration", "fold", "plans")) == expected


def test_old_global_config_without_plans_uses_official_default():
    config = load_app_config(environ={}, defaults={"slice_part_config": _model_config({})})
    stage = Pipeline._default_stages(config)[1]
    assert config.nnunet.plans == "nnUNetPlans"
    assert stage._match_model("腰椎", "CT", 1.0)["plans"] == "nnUNetPlans"
