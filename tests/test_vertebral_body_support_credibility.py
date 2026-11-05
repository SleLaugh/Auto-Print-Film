"""三维主体支撑不能被短而紧凑的附件候选替代。"""

import unittest

import numpy as np
from scipy import ndimage

from src.models import LabelTarget
from src.stages.stage_localization import LocalizationStage


class VertebralBodySupportCredibilityTest(unittest.TestCase):
    _BODY_REGION = (slice(4, 30), slice(10, 28), slice(10, 70))
    _SPACINGS = (None, (1.0, 0.392578125, 0.392578125))

    def _connected_body(self, with_posterior_slab):
        array = np.zeros((40, 160, 80), dtype=np.uint8)
        array[self._BODY_REGION] = 5
        # The narrow attachment connects the raw label, but has insufficient
        # transverse support to connect the independently extracted candidates.
        array[30, 26:154, 39:40] = 5
        array[30, 150:154, 10:70] = 5
        if with_posterior_slab:
            array[31:32, 60:133, 20:59] = 5
        return array

    def _expected_body(self, shape):
        expected = np.zeros(shape, dtype=bool)
        expected[self._BODY_REGION] = True
        return expected

    def test_single_slice_compact_support_cannot_replace_complete_body(self):
        expected = self._expected_body((40, 160, 80))
        candidates = expected.copy()
        candidates[31:32, 60:133, 20:59] = True
        original = candidates.copy()

        selected = LocalizationStage._select_vertebral_body_support(
            candidates, center_x=40.0,
        )

        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_array_equal(candidates, original)

    def test_connected_label_keeps_same_body_when_posterior_slab_appears(self):
        for label_name in ("vertebrae_C7", "vertebrae_T8", "vertebrae_L3"):
            for spacing in self._SPACINGS:
                baseline = None
                for with_slab in (False, True):
                    with self.subTest(label=label_name, spacing=spacing, slab=with_slab):
                        array = self._connected_body(with_slab)
                        original = array.copy()
                        _, component_count = ndimage.label(
                            array == 5, structure=np.ones((3, 3, 3), dtype=bool),
                        )
                        self.assertEqual(component_count, 1)

                        body = LocalizationStage._vertebral_body_candidate_mask(
                            array, LabelTarget(label_name, [5]),
                            spacing_zyx=spacing,
                        )

                        np.testing.assert_array_equal(body, self._expected_body(array.shape))
                        np.testing.assert_array_equal(array, original)
                        if baseline is None:
                            baseline = body.copy()
                        else:
                            np.testing.assert_array_equal(body, baseline)

    def test_short_multislice_compact_support_cannot_replace_complete_body(self):
        for spacing in self._SPACINGS:
            with self.subTest(spacing=spacing):
                expected = self._expected_body((40, 160, 80))
                candidates = expected.copy()
                # This fragment exceeds a tenth of the body's volume, but its
                # five supported layers still cannot represent that whole body.
                candidates[33:38, 50:71, 26:54] = True
                original = candidates.copy()

                selected = LocalizationStage._select_vertebral_body_support(
                    candidates, center_x=40.0, spacing_zyx=spacing,
                )

                np.testing.assert_array_equal(selected, expected)
                np.testing.assert_array_equal(candidates, original)

    def test_large_short_posterior_bridge_cannot_disqualify_stable_wide_body(self):
        # C7 can have a wide, shallow anterior body, while a four-layer rear
        # bridge passes the AP/LR compactness rule and exceeds the old 10%
        # remnant threshold. Repeat each axis independently to preserve the
        # physical evidence under anisotropic sampling.
        for factors in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)):
            with self.subTest(factors=factors):
                expected = np.zeros((40, 200, 130), dtype=bool)
                expected[8:25, 10:28, 30:100] = True
                candidates = expected.copy()
                candidates[26:30, 50:184, 49:80] = True
                for axis, factor in enumerate(factors):
                    candidates = np.repeat(candidates, factor, axis=axis)
                    expected = np.repeat(expected, factor, axis=axis)
                original = candidates.copy()
                spacing = tuple(value / factor for value, factor in zip(
                    (1.0, 0.349609375, 0.349609375), factors,
                ))

                selected = LocalizationStage._select_vertebral_body_support(
                    candidates, center_x=65.0 * factors[2],
                    spacing_zyx=spacing,
                    label_voxel_count=100000 * int(np.prod(factors)),
                    label_si_span=31 * factors[0],
                )

                np.testing.assert_array_equal(selected, expected)
                np.testing.assert_array_equal(candidates, original)

    def test_fewer_layers_alone_do_not_disqualify_physical_posterior_body(self):
        candidates = np.zeros((40, 200, 130), dtype=bool)
        candidates[8:25, 10:28, 30:100] = True
        candidates[26:30, 50:184, 49:80] = True
        expected = np.zeros_like(candidates)
        expected[26:30, 50:184, 49:80] = True
        # The same four-layer component has 12 mm of SI support here. It is
        # not a thin bridge, so the pre-existing compact-body choice stands.
        selected = LocalizationStage._select_vertebral_body_support(
            candidates, center_x=65.0,
            spacing_zyx=(3.0, 0.349609375, 0.349609375),
            label_voxel_count=100000, label_si_span=31,
        )

        np.testing.assert_array_equal(selected, expected)

    def test_candidate_mask_keeps_wide_body_before_large_short_bridge(self):
        array = np.zeros((40, 200, 130), dtype=np.uint8)
        array[8:25, 10:28, 30:100] = 5
        array[26:30, 50:184, 49:80] = 5
        expected = np.zeros_like(array, dtype=bool)
        expected[8:25, 10:28, 30:100] = True
        original = array.copy()

        body = LocalizationStage._vertebral_body_candidate_mask(
            array, LabelTarget("vertebrae_C7", [5]),
            spacing_zyx=(1.0, 0.349609375, 0.349609375),
        )

        np.testing.assert_array_equal(body, expected)
        np.testing.assert_array_equal(array, original)

    def test_c2_keeps_existing_selection_before_its_specialized_recovery(self):
        array = np.zeros((40, 200, 130), dtype=np.uint8)
        array[8:25, 10:28, 30:100] = 5
        array[26:30, 50:184, 49:80] = 5
        expected = np.zeros_like(array, dtype=bool)
        expected[26:30, 50:184, 49:80] = True

        body = LocalizationStage._vertebral_body_candidate_mask(
            array, LabelTarget("vertebrae_C2", [5]),
            spacing_zyx=(1.0, 0.349609375, 0.349609375),
        )

        np.testing.assert_array_equal(body, expected)

    def test_stable_small_anterior_body_survives_larger_posterior_support(self):
        array = np.zeros((36, 70, 70), dtype=np.uint8)
        body_region = (slice(8, 24), slice(10, 25), slice(24, 44))
        array[body_region] = 5
        array[6:30, 35:60, 8:56] = 5
        original = array.copy()

        body = LocalizationStage._vertebral_body_candidate_mask(
            array, LabelTarget("vertebrae_C5", [5]),
        )

        expected = np.zeros(array.shape, dtype=bool)
        expected[body_region] = True
        np.testing.assert_array_equal(body, expected)
        np.testing.assert_array_equal(array, original)

    def test_tall_thin_fragment_does_not_disqualify_short_complete_body(self):
        array = np.zeros((36, 40, 80), dtype=np.uint8)
        body_region = (slice(12, 20), slice(15, 30), slice(30, 50))
        array[body_region] = 5
        array[2:32, 5:8, 38:42] = 5
        original = array.copy()

        body = LocalizationStage._vertebral_body_candidate_mask(
            array, LabelTarget("vertebrae_T8", [5]),
        )

        expected = np.zeros(array.shape, dtype=bool)
        expected[body_region] = True
        np.testing.assert_array_equal(body, expected)
        np.testing.assert_array_equal(array, original)

    def test_large_noncompact_posterior_does_not_set_compact_body_area_threshold(self):
        # The real body can be below 20% of the posterior transverse area,
        # or even below 10% of its volume, while retaining normal SI support.
        for rear_start, rear_end in ((20, 110), (5, 125)):
            for spacing in self._SPACINGS:
                with self.subTest(rear=(rear_start, rear_end), spacing=spacing):
                    array = np.zeros((40, 100, 130), dtype=np.uint8)
                    array[10:26, 10:25, 50:80] = 5
                    array[8:30, 50:80, rear_start:rear_end] = 5
                    original = array.copy()
                    expected = np.zeros_like(array, dtype=bool)
                    expected[10:26, 10:25, 50:80] = True

                    body = LocalizationStage._vertebral_body_candidate_mask(
                        array, LabelTarget("vertebrae_C5", [5]),
                        spacing_zyx=spacing,
                    )

                    np.testing.assert_array_equal(body, expected)
                    np.testing.assert_array_equal(array, original)

    def test_short_complete_anterior_body_is_not_a_short_posterior_fragment(self):
        for spacing in self._SPACINGS:
            with self.subTest(spacing=spacing):
                array = np.zeros((40, 160, 130), dtype=np.uint8)
                array[12:20, 10:32, 50:80] = 5
                array[2:26, 50:70, 20:110] = 5
                original = array.copy()
                expected = np.zeros_like(array, dtype=bool)
                expected[12:20, 10:32, 50:80] = True

                body = LocalizationStage._vertebral_body_candidate_mask(
                    array, LabelTarget("vertebrae_C5", [5]), spacing_zyx=spacing,
                )

                np.testing.assert_array_equal(body, expected)
                np.testing.assert_array_equal(array, original)

    def test_large_thin_anterior_sheet_does_not_disqualify_complete_body(self):
        for spacing, body_height in ((None, 16), ((1.0, 0.392578125, 0.392578125), 16),
                                     ((1.0, 1.0, 1.0), 6)):
            with self.subTest(spacing=spacing, body_height=body_height):
                array = np.zeros((50, 90, 100), dtype=np.uint8)
                array[2:42, 5:13, 20:80] = 5
                array[16:16 + body_height, 25:40, 35:65] = 5
                original = array.copy()
                expected = np.zeros_like(array, dtype=bool)
                expected[16:16 + body_height, 25:40, 35:65] = True

                body = LocalizationStage._vertebral_body_candidate_mask(
                    array, LabelTarget("vertebrae_C5", [5]), spacing_zyx=spacing,
                )

                np.testing.assert_array_equal(body, expected)
                np.testing.assert_array_equal(array, original)


if __name__ == "__main__":
    unittest.main()
