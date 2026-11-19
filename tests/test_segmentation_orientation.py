"""模型存储方向适配保持体素物理位置，输出和缓存必须回到原始 CT 空间。"""

import itertools
import json
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.inference.orientation import (
    orient_image_for_inference,
    prepare_oriented_input,
    restore_segmentation_file,
    restore_segmentation_image,
)
from src.models import Request, SlicePartEnum
from src.stages.stage_segmentation import SegmentationStage


def _image(oblique=False):
    image = sitk.GetImageFromArray(np.arange(7 * 8 * 9, dtype=np.int16).reshape(7, 8, 9))
    image.SetSpacing((0.73456, 0.89123, 1.234567))
    image.SetOrigin((1234.5, -987.1234, 1134.56789))
    if oblique:
        yaw, tilt = np.deg2rad((13, 19))
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        image.SetDirection((rz @ rx).ravel())
    return image


def _geometry(image):
    return image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()


def _physical_locations_by_value(image):
    array = sitk.GetArrayFromImage(image)
    indices = np.indices(array.shape).reshape(3, -1).T[:, ::-1]
    physical = indices @ (np.asarray(image.GetDirection()).reshape(3, 3) * image.GetSpacing()).T + image.GetOrigin()
    return physical[np.argsort(array.ravel())]


@pytest.mark.parametrize("orientation", ("RAS", "LPS"))
@pytest.mark.parametrize("order", ((0, 1, 2), (2, 0, 1), (1, 2, 0)))
@pytest.mark.parametrize("flips", tuple(itertools.product((False, True), repeat=3)))
@pytest.mark.parametrize("oblique", (False, True))
def test_axis_permutation_flips_and_oblique_images_preserve_every_voxel_without_resampling(
    orientation, order, flips, oblique,
):
    original = sitk.Flip(sitk.PermuteAxes(_image(oblique), order), flips)
    with patch.object(sitk, "Resample", side_effect=AssertionError("不能插值")), patch.object(
        sitk, "ResampleImageFilter", side_effect=AssertionError("不能插值"),
    ):
        normalized, transform = orient_image_for_inference(original, orientation)
        assert sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(normalized.GetDirection()) == orientation
        np.testing.assert_allclose(_physical_locations_by_value(normalized), _physical_locations_by_value(original), atol=1e-10)
        segmentation = sitk.GetImageFromArray((sitk.GetArrayFromImage(normalized) % 8).astype(np.uint8))
        segmentation.CopyInformation(normalized)
        restored = restore_segmentation_image(segmentation, transform)
    assert restored.GetPixelID() == sitk.sitkUInt8
    assert _geometry(restored) == _geometry(original)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), sitk.GetArrayFromImage(original) % 8)


@pytest.mark.parametrize("orientation", ("RAS", "LPS"))
def test_nifti_header_rounding_does_not_reject_valid_oblique_large_origin_output(tmp_path, orientation):
    source = tmp_path / "raw_0000.nii.gz"
    sitk.WriteImage(sitk.Flip(sitk.PermuteAxes(_image(True), (2, 0, 1)), (True, False, True)), str(source))
    original = sitk.ReadImage(str(source))
    source_bytes = source.read_bytes()
    normalized_path, transform = prepare_oriented_input(source, tmp_path / "image_input", orientation)
    normalized = sitk.ReadImage(str(normalized_path))
    segmentation = sitk.GetImageFromArray((sitk.GetArrayFromImage(normalized) % 8).astype(np.uint8))
    segmentation.CopyInformation(normalized)
    output = tmp_path / "seg.nii.gz"
    sitk.WriteImage(segmentation, str(output))
    restore_segmentation_file(output, transform)
    actual = sitk.ReadImage(str(output))
    assert actual.GetSize() == original.GetSize()
    np.testing.assert_allclose(actual.GetSpacing(), original.GetSpacing(), rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual.GetOrigin(), original.GetOrigin(), rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual.GetDirection(), original.GetDirection(), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(original) % 8)
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("field", ("size", "origin", "spacing", "direction"))
def test_wrong_segmentation_geometry_is_rejected_before_inverse_transform(field):
    normalized, transform = orient_image_for_inference(_image(True), "RAS")
    segmentation = sitk.Cast(normalized > 20, sitk.sitkUInt8)
    if field == "size":
        segmentation = segmentation[:-1, :, :]
    elif field == "origin":
        segmentation.SetOrigin((segmentation.GetOrigin()[0] + 0.5, *segmentation.GetOrigin()[1:]))
    elif field == "spacing":
        segmentation.SetSpacing((1.0, *segmentation.GetSpacing()[1:]))
    else:
        segmentation.SetDirection(tuple(np.eye(3).ravel()))
    with pytest.raises(ValueError, match="nnUNet 输出.*空间不一致"):
        restore_segmentation_image(segmentation, transform)


@pytest.mark.parametrize("marker,expected", [
    (None, False), ({"model_id": "701"}, False),
    ({"model_id": "701", "input_orientation": "LPS"}, False),
    ({"model_id": "701", "input_orientation": "RAS"}, True),
    ({"model_id": "903", "input_orientation": "RAS"}, False),
])
def test_oriented_cache_requires_matching_orientation_marker(tmp_path, marker, expected):
    output = tmp_path / "seg.nii.gz"
    output.touch()
    if marker is not None:
        (tmp_path / "model.json").write_text(json.dumps(marker))
    assert SegmentationStage._can_reuse_segmentation(tmp_path, output, 701, input_orientation="RAS") is expected


def test_legacy_models_keep_cache_and_model_metadata_behavior(tmp_path):
    output = tmp_path / "seg.nii.gz"
    output.touch()
    assert SegmentationStage._can_reuse_segmentation(tmp_path, output, 292)
    SegmentationStage._write_model_marker(tmp_path, 292)
    assert json.loads((tmp_path / "model.json").read_text()) == {"model_id": "292"}
    stage = SegmentationStage({"inference_max_retries": 0})
    assert "input_orientation" not in stage._match_model("颈椎", "CT", 1.0)
    assert stage._match_model("上腹部", "CT", 1.0)["input_orientation"] == "RAS"


@pytest.mark.parametrize("marker", [{"model_id": "701", "input_orientation": "RAS"}, [], None, 701])
def test_removing_orientation_option_or_invalid_marker_cannot_reuse_oriented_cache(tmp_path, marker):
    output = tmp_path / "seg.nii.gz"
    output.touch()
    (tmp_path / "model.json").write_text(json.dumps(marker))
    assert not SegmentationStage._can_reuse_segmentation(tmp_path, output, 701)


def _context(root):
    source = root / "nifti" / "raw_0000.nii.gz"
    source.parent.mkdir()
    sitk.WriteImage(_image(True), str(source))
    request = Request(
        dicom_dir=str(root / "dicom"), output_dir=str(root / "output"),
        cache_dir=str(root / "cache"), slice_part=SlicePartEnum.上腹部, scan_thickness=1.0,
    )
    context = ProcessingContext(request)
    context.set_result("preprocessing", {"nifti_path": str(source)})
    return context, source


def test_stage_replaces_unmarked_cache_restores_geometry_and_never_publishes_normalized_ct(tmp_path):
    context, source = _context(tmp_path)
    source_bytes = source.read_bytes()
    cache = tmp_path / "cache" / "701" / "segmentation"
    cache.mkdir(parents=True)
    (cache / "seg.nii.gz").write_bytes(b"old incorrectly oriented segmentation")
    stage = SegmentationStage({"inference_max_retries": 0})
    calls = []

    def infer(**options):
        normalized_path = options["input_path"]
        assert normalized_path.name == source.name
        assert normalized_path.parent == options["input_dir"]
        assert normalized_path.parent.name == "image_input"
        assert list(options["input_dir"].iterdir()) == [normalized_path]
        assert normalized_path != source
        calls.append(normalized_path)
        normalized = sitk.ReadImage(str(normalized_path))
        assert sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(normalized.GetDirection()) == "RAS"
        segmentation = sitk.GetImageFromArray((sitk.GetArrayFromImage(normalized) % 8).astype(np.uint8))
        segmentation.CopyInformation(normalized)
        sitk.WriteImage(segmentation, str(options["output_path"]))
        (options["output_path"].parent / "dataset.json").write_text('{"labels":{"liver":1}}')

    with patch.object(stage, "_run_nnunet_inference", side_effect=infer):
        stage.execute(context)
        stage.execute(context)
    assert len(calls) == 1
    assert not calls[0].exists()
    assert {path.name for path in cache.iterdir()} == {"seg.nii.gz", "model.json", "dataset.json"}
    assert json.loads((cache / "model.json").read_text()) == {"model_id": "701", "input_orientation": "RAS"}
    actual, original = sitk.ReadImage(str(cache / "seg.nii.gz")), sitk.ReadImage(str(source))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(original) % 8)
    np.testing.assert_allclose(actual.GetDirection(), original.GetDirection(), rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual.GetOrigin(), original.GetOrigin(), rtol=0, atol=1e-7)
    assert actual.GetSize() == original.GetSize()
    assert source.read_bytes() == source_bytes
    assert context.get_result("segmentation")["model"]["input_orientation"] == "RAS"


def test_bad_output_geometry_does_not_publish_or_relabel_old_cache(tmp_path):
    context, _ = _context(tmp_path)
    cache = tmp_path / "cache" / "701" / "segmentation"
    cache.mkdir(parents=True)
    old_segmentation = b"old segmentation"
    (cache / "seg.nii.gz").write_bytes(old_segmentation)
    (cache / "model.json").write_text('{"model_id":"701"}')
    stage = SegmentationStage({"inference_max_retries": 0})

    def infer(**options):
        image = sitk.ReadImage(str(options["input_path"]))
        bad = sitk.Cast(image > 20, sitk.sitkUInt8)
        bad.SetOrigin((0., 0., 0.))
        sitk.WriteImage(bad, str(options["output_path"]))

    with patch.object(stage, "_run_nnunet_inference", side_effect=infer), pytest.raises(ValueError, match="空间不一致"):
        stage.execute(context)
    assert (cache / "seg.nii.gz").read_bytes() == old_segmentation
    assert json.loads((cache / "model.json").read_text()) == {"model_id": "701"}
    assert not stage._can_reuse_segmentation(cache, cache / "seg.nii.gz", 701, input_orientation="RAS")
    assert not list(cache.parent.glob(".segmentation-*"))
