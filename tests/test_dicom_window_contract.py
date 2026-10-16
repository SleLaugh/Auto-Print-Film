"""回读实际成片，验证请求窗口标签与标准 DICOM 显示结果一致。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_windowing
import SimpleITK as sitk

from src.models import WindowPreset
from src.reconstruction.dicom import write_ct_dicom, write_localizer_dicom


class DicomWindowContractTest(unittest.TestCase):
    WINDOWS = (
        WindowPreset("纵隔窗", 350, 40),
        WindowPreset("骨窗", 2500, 400),
        WindowPreset("肺窗", 1600, -600),
        WindowPreset("小数窗", 350.5, 40.25),
        WindowPreset("小窗宽", 0.25, -0.125),
    )

    @staticmethod
    def _source_state(image):
        return (
            sitk.GetArrayFromImage(image),
            image.GetOrigin(),
            image.GetSpacing(),
            image.GetDirection(),
            {tag: image.GetMetaData(tag) for tag in image.GetMetaDataKeys()},
        )

    def _assert_source_unchanged(self, image, original):
        current = self._source_state(image)
        np.testing.assert_array_equal(current[0], original[0])
        self.assertEqual(current[1:], original[1:])

    def _assert_window_tags(self, dataset, window):
        self.assertAlmostEqual(float(dataset.WindowWidth), window.window_width)
        self.assertAlmostEqual(float(dataset.WindowCenter), window.window_center)
        self.assertEqual(dataset.VOILUTFunction, "LINEAR_EXACT")

    @staticmethod
    def _write_ct(image, path, window):
        write_ct_dicom(
            image=image,
            output_path=path,
            plan={
                "plan_id": "window-contract",
                "plane": "横状位",
                "group_name": "whole_range",
                "output_spacing_mm": {"x": 1.0, "y": 1.0},
                "slice_thickness": 1.0,
                "slice_spacing": 1.0,
            },
            planned_slice={
                "index": 1,
                "center": {"x": 0, "y": 0, "z": 0},
                "normal": {"x": 0, "y": 0, "z": 1},
                "row_dir": {"x": 1, "y": 0, "z": 0},
                "col_dir": {"x": 0, "y": 1, "z": 0},
            },
            window=window,
            series_uid="1.2.826.0.1.3680043.2.1125.991",
            study_uid="1.2.826.0.1.3680043.2.1125.992",
            total_slices=1,
        )
        return pydicom.dcmread(path)

    def _assert_ct_pixels_and_rescale(self, dataset, window):
        self.assertEqual(dataset.PixelRepresentation, 0)
        self.assertEqual(dataset.BitsAllocated, 16)
        self.assertEqual(dataset.BitsStored, 16)
        self.assertEqual(dataset.pixel_array.dtype, np.dtype(np.uint16))
        self.assertAlmostEqual(
            float(dataset.RescaleSlope),
            window.window_width / 65535.0,
            delta=window.window_width / 65535.0 * 1e-10,
        )
        self.assertAlmostEqual(
            float(dataset.RescaleIntercept),
            window.window_center - window.window_width / 2.0,
        )
        self.assertEqual(dataset.RescaleType, "HU")

    def _assert_ct_viewer_display(self, dataset, source_hu, window):
        modality_values = apply_modality_lut(dataset.pixel_array, dataset)
        viewed = apply_windowing(modality_values, dataset)
        # pydicom returns the rescaled legal range; normalize it to black=0,
        # white=1. This exercises a real reader rather than the writer helper.
        slope = float(dataset.RescaleSlope)
        intercept = float(dataset.RescaleIntercept)
        viewed_normalized = (viewed - intercept) / (65535.0 * slope)
        # Independent LINEAR_EXACT definition in centered coordinates. Keep
        # fractional HU here so a premature integer-HU conversion is observable.
        reference_display = np.clip(
            (source_hu.astype(np.float64) - window.window_center)
            / window.window_width + 0.5,
            0.0,
            1.0,
        )
        # The only intended loss is rounding to 16-bit display gray: half a
        # gray step, with a small allowance for floating-point calculations.
        np.testing.assert_allclose(
            viewed_normalized,
            reference_display,
            rtol=0,
            atol=0.5 / 65535.0 + 1e-7,
        )
        return viewed_normalized

    def test_ct_requested_window_controls_tags_and_viewer_brightness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, window in enumerate(self.WINDOWS):
                with self.subTest(window=window.name):
                    lower = window.window_center - window.window_width / 2.0
                    upper = window.window_center + window.window_width / 2.0
                    source = np.array(
                        [
                            [-1e9, lower - window.window_width, lower,
                             lower + window.window_width / 4.0,
                             window.window_center],
                            [window.window_center + window.window_width / 4.0,
                             upper, upper + window.window_width, 1e9,
                             window.window_center + window.window_width / 8.0],
                        ],
                        dtype=np.float64,
                    )
                    image = sitk.GetImageFromArray(source[np.newaxis, ...])
                    # Stale source tags must never override the requested window.
                    image.SetMetaData("0028|1050", "99999")
                    image.SetMetaData("0028|1051", "88888")
                    image.SetMetaData("0028|1056", "SIGMOID")
                    original = self._source_state(image)
                    dataset = self._write_ct(
                        image, Path(temp_dir) / f"ct-{index}.dcm", window,
                    )
                    self._assert_window_tags(dataset, window)
                    self._assert_ct_pixels_and_rescale(dataset, window)
                    viewed = self._assert_ct_viewer_display(dataset, source, window)
                    # Explicit anchors verify black/white clipping, center, and
                    # quarter brightness independently of the reference formula.
                    np.testing.assert_allclose(
                        viewed,
                        [[0, 0, 0, 0.25, 0.5], [0.75, 1, 1, 1, 0.625]],
                        rtol=0,
                        atol=0.5 / 65535.0 + 1e-7,
                    )
                    np.testing.assert_array_equal(
                        dataset.pixel_array[0, :3], [0, 0, 0],
                    )
                    np.testing.assert_array_equal(
                        dataset.pixel_array[1, 1:4], [65535, 65535, 65535],
                    )
                    self._assert_source_unchanged(image, original)

    def test_ct_retains_sub_hu_display_gradient_without_integer_hu_rounding(self):
        window = WindowPreset("亚HU梯度", 50, 60)
        source = np.array(
            [
                [59.6, 59.7, 59.8, 59.9, 60.0],
                [60.05, 60.1, 60.2, 60.3, 60.4],
            ],
            dtype=np.float64,
        )
        image = sitk.GetImageFromArray(source[np.newaxis, ...])
        original = self._source_state(image)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self._write_ct(image, Path(temp_dir) / "gradient.dcm", window)
            self._assert_window_tags(dataset, window)
            self._assert_ct_pixels_and_rescale(dataset, window)
            viewed = self._assert_ct_viewer_display(dataset, source, window)
            # Integer-HU rounding would collapse all ten samples to 60 HU.
            self.assertTrue(np.all(np.diff(viewed.ravel()) > 0))
            self.assertEqual(np.unique(dataset.pixel_array).size, source.size)
            self._assert_source_unchanged(image, original)

    def test_ct_window_arithmetic_preserves_requested_decimal_precision(self):
        windows = (
            WindowPreset("高窗位窄窗", 0.01, 1000.1),
            WindowPreset("非整齐小数窗", 350.1, 1000.12345),
        )
        fractions = np.linspace(-0.25, 1.25, 2048).reshape(32, 64)

        with tempfile.TemporaryDirectory() as temp_dir:
            for window_index, window in enumerate(windows):
                for dtype in (np.float32, np.float64):
                    with self.subTest(window=window.name, source_dtype=dtype):
                        source = (
                            window.window_center
                            + window.window_width * (fractions - 0.5)
                        ).astype(dtype)
                        image = sitk.GetImageFromArray(source[np.newaxis, ...])
                        original = self._source_state(image)
                        dataset = self._write_ct(
                            image,
                            Path(temp_dir) / f"decimal-{window_index}-{dtype.__name__}.dcm",
                            window,
                        )
                        self._assert_window_tags(dataset, window)
                        self._assert_ct_pixels_and_rescale(dataset, window)
                        modality_values = apply_modality_lut(dataset.pixel_array, dataset)
                        viewed = apply_windowing(modality_values, dataset)
                        viewed_normalized = (
                            (viewed - float(dataset.RescaleIntercept))
                            / (65535.0 * float(dataset.RescaleSlope))
                        )
                        # Compare with the actual source values in float64 so
                        # float32 input precision is separate from window arithmetic.
                        reference_display = np.clip(
                            (source.astype(np.float64) - window.window_center)
                            / window.window_width + 0.5,
                            0.0,
                            1.0,
                        )
                        np.testing.assert_allclose(
                            viewed_normalized,
                            reference_display,
                            rtol=0,
                            atol=0.5 / 65535.0,
                        )
                        self._assert_source_unchanged(image, original)

    def test_localizer_round_trip_preserves_black_midgray_and_white(self):
        source = np.array([[0, 32768, 65535], [65535, 32768, 0]], dtype=np.uint16)
        image = sitk.GetImageFromArray(source[np.newaxis, ...])
        image.SetMetaData("0020|0032", "0\\0\\0")
        image.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
        original = self._source_state(image)
        expected_display = source.astype(np.float64) / 65535.0

        with tempfile.TemporaryDirectory() as temp_dir:
            for index, window in enumerate(self.WINDOWS):
                with self.subTest(window=window.name):
                    path = Path(temp_dir) / f"localizer-{index}.dcm"
                    write_localizer_dicom(
                        image=image,
                        output_path=path,
                        window=window,
                        series_uid="1.2.826.0.1.3680043.2.1125.993",
                        study_uid="1.2.826.0.1.3680043.2.1125.992",
                    )
                    dataset = pydicom.dcmread(path)
                    self._assert_window_tags(dataset, window)
                    self.assertEqual(dataset.PixelRepresentation, 0)
                    self.assertEqual(dataset.pixel_array.dtype, np.dtype(np.uint16))
                    np.testing.assert_array_equal(dataset.pixel_array, source)
                    slope = float(dataset.RescaleSlope)
                    intercept = float(dataset.RescaleIntercept)
                    self.assertAlmostEqual(
                        slope,
                        window.window_width / 65535.0,
                        delta=window.window_width / 65535.0 * 1e-10,
                    )
                    self.assertAlmostEqual(
                        intercept, window.window_center - window.window_width / 2.0,
                    )
                    # Rescaled display intensities must not claim quantitative HU.
                    self.assertEqual(dataset.RescaleType, "US")
                    modality_values = apply_modality_lut(dataset.pixel_array, dataset)
                    standard_display = np.clip(
                        (modality_values - window.window_center)
                        / window.window_width + 0.5,
                        0.0,
                        1.0,
                    )
                    np.testing.assert_allclose(
                        standard_display, expected_display, rtol=0, atol=1e-9,
                    )
                    viewed = apply_windowing(modality_values, dataset)
                    # pydicom returns the rescaled legal range; normalize it to
                    # black=0 and white=1 before comparing display brightness.
                    viewed_normalized = (viewed - intercept) / (65535.0 * slope)
                    np.testing.assert_allclose(
                        viewed_normalized, expected_display, rtol=0, atol=1e-9,
                    )
                    self._assert_source_unchanged(image, original)


if __name__ == "__main__":
    unittest.main()
