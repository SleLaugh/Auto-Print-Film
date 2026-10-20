"""Localizer 始终使用原图背景和全长平面交线，不随 MPR 的 FillPercent 变动。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from src.reconstruction.framing import TargetFramer
from src.stages.stage_reconstruction import ReconstructionStage


def _case(angle_degrees):
    z, y, x = np.indices((28, 36, 44))
    raw = sitk.GetImageFromArray((x + 3 * y + 20 * z).astype(np.float32))
    raw.SetSpacing((0.8, 0.9, 1.2))
    raw.SetOrigin((10., -35., 80.))
    labels = np.zeros((28, 36, 44), dtype=np.uint8)
    # The framed target lies far from the fixed sagittal reference layer. A
    # small MPR FOV therefore cannot define the extent of its Localizer lines.
    labels[5:13, 10:17, 27:35] = 7
    labels[13:23, 19:26, 29:38] = 7
    segmentation = sitk.GetImageFromArray(labels)
    segmentation.CopyInformation(raw)
    theta = np.radians(angle_degrees)
    row = np.array([1., 0., 0.])
    column = np.array([0., np.cos(theta), np.sin(theta)])
    normal = np.cross(row, column)
    midpoint = np.array(raw.TransformContinuousIndexToPhysicalPoint((21.5, 17.5, 13.5)))
    slices = [
        {
            "index": index,
            "center": dict(zip("xyz", midpoint + normal * offset)),
            "row_dir": dict(zip("xyz", row)),
            "col_dir": dict(zip("xyz", column)),
            "normal": dict(zip("xyz", normal)),
        }
        for index, offset in enumerate((4., 0., -4.), 1)
    ]
    reference = dict(zip("xyz", raw.TransformIndexToPhysicalPoint((8, 17, 13))))
    plan = {
        "plan_id": "localizer_fill_regression",
        "slice_part": "颅脑", "slice_type": "MPR", "plane": "横状位",
        "geometry_type": "plane_stack", "group_name": "whole_range",
        "extent_mm": {"width": 43 * 0.8, "height": 35 * 0.9},
        "output_spacing_mm": {"x": 0.8, "y": 0.9, "z": 0.},
        "slice_thickness": 1., "slice_spacing": 4.,
        "window_presets": ["骨窗"], "source_labels": ["brain"],
        "localizer_plane": "矢状位", "localizer_reference": reference,
        "slices": slices,
    }
    return raw, segmentation, {"brain": {"label_ids": [7]}}, plan


def _geometry(image):
    return {
        "size": image.GetSize(), "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(), "direction": image.GetDirection(),
        "ipp": image.GetMetaData("0020|0032"),
        "iop": image.GetMetaData("0020|0037"),
    }


def _groups(indices):
    return 0 if not len(indices) else 1 + int(np.count_nonzero(np.diff(indices) > 1))


@pytest.mark.parametrize("angle_degrees", (0., 23.))
def test_real_mpr_fill_changes_leave_localizer_background_geometry_and_full_lines_identical(
    tmp_path, monkeypatch, angle_degrees,
):
    raw, segmentation, coords, original = _case(angle_degrees)
    original_render = ReconstructionStage._build_plan_localizer_render
    captures = []

    def record_render(cls, image, plan, **kwargs):
        # Execute the real renderer used by _execute_plan; record its result
        # before writing, without replacing or normalizing any geometry.
        render = original_render(image, plan, **kwargs)
        assert render is not None
        captures.append({
            "geometry": _geometry(render.background_image),
            "background": sitk.GetArrayFromImage(render.background_image),
            "line_alpha": render.line_alpha.copy(),
            "reference": deepcopy(plan.get("localizer_reference")),
        })
        return render

    monkeypatch.setattr(
        ReconstructionStage, "_build_plan_localizer_render", classmethod(record_render),
    )
    stage = ReconstructionStage({"parallel": False})
    localizers, mpr_spacings = [], []
    expected_background = sitk.GetArrayFromImage(raw)[:, :, 8][::-1]
    expected_ipp = raw.TransformIndexToPhysicalPoint((8, 0, raw.GetSize()[2] - 1))
    for fill in (0., 25., 50., 100.):
        plan = {**deepcopy(original), "fill_percent": fill}
        before = deepcopy(plan)
        result = stage._execute_plan(
            image=raw, plan=plan, output_root=tmp_path / str(int(fill)),
            study_uid="1.2.3.4", source_metadata={},
            target_framer=TargetFramer(segmentation, coords, plan),
        )
        assert plan == before
        capture = captures[-1]
        assert capture["reference"] == original["localizer_reference"]
        assert capture["geometry"]["size"] == (raw.GetSize()[1], raw.GetSize()[2])
        np.testing.assert_array_equal(capture["background"], expected_background)
        alpha = capture["line_alpha"]
        # All three distinct physical planes reach both source-image edges;
        # their endpoints cannot shrink to a rectangle around the target.
        assert _groups(np.flatnonzero(alpha[:, 0] > 1e-6)) == len(original["slices"])
        assert _groups(np.flatnonzero(alpha[:, -1] > 1e-6)) == len(original["slices"])
        directory = Path(result["window_outputs"]["骨窗"])
        localizer = pydicom.dcmread(str(directory / "Localizer.dcm"))
        localizers.append(localizer)
        assert (localizer.Rows, localizer.Columns) == expected_background.shape
        np.testing.assert_allclose(localizer.PixelSpacing, [1.2, 0.9], atol=1e-9)
        np.testing.assert_allclose(localizer.ImagePositionPatient, expected_ipp, atol=1e-9)
        np.testing.assert_allclose(
            localizer.ImageOrientationPatient, [0., 1., 0., 0., 0., -1.], atol=1e-9,
        )
        mpr = pydicom.dcmread(str(directory / "1.dcm"))
        mpr_spacings.append(np.array(mpr.PixelSpacing, dtype=float))
        row = np.array(mpr.ImageOrientationPatient[:3], dtype=float)
        col = np.array(mpr.ImageOrientationPatient[3:], dtype=float)
        center = (np.array(mpr.ImagePositionPatient, dtype=float)
                  + row * (mpr.Columns - 1) * float(mpr.PixelSpacing[1]) / 2
                  + col * (mpr.Rows - 1) * float(mpr.PixelSpacing[0]) / 2)
        normal = np.array(list(original["slices"][0]["normal"].values()))
        planned_center = np.array(list(original["slices"][0]["center"].values()))
        assert float((center - planned_center) @ normal) == pytest.approx(0., abs=1e-6)
    assert len(captures) == 4
    for capture in captures[1:]:
        assert capture["geometry"] == captures[0]["geometry"]
        np.testing.assert_array_equal(capture["background"], captures[0]["background"])
        np.testing.assert_array_equal(capture["line_alpha"], captures[0]["line_alpha"])
    for localizer in localizers[1:]:
        for field in ("Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient"):
            assert getattr(localizer, field) == getattr(localizers[0], field)
        np.testing.assert_array_equal(localizer.pixel_array, localizers[0].pixel_array)
    # Real Fill framing still scales the exported MPR while leaving the source
    # Localizer untouched; this is not a test of four identical mocked plans.
    assert len({tuple(spacing) for spacing in mpr_spacings}) == 4
    np.testing.assert_allclose(mpr_spacings[1], mpr_spacings[-1] * 4, atol=1e-8)
    np.testing.assert_allclose(mpr_spacings[2], mpr_spacings[-1] * 2, atol=1e-8)
