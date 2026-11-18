import unittest

from src.planning.sampling import resolve_sampling


class SamplingResolutionTest(unittest.TestCase):
    """验证层厚、层间距、数量和物理范围的统一计算。"""

    def test_thickness_and_spacing_calculate_count(self):
        sampling = resolve_sampling(300.0, 5.0, 5.0, None)

        self.assertEqual(sampling.slice_image_count, 60)
        self.assertEqual(sampling.slice_thickness, 5.0)
        self.assertEqual(sampling.slice_spacing, 5.0)
        self.assertEqual(sampling.total_range_mm, 300.0)

    def test_thickness_and_count_calculate_spacing(self):
        sampling = resolve_sampling(300.0, 5.0, None, 60)

        self.assertEqual(sampling.slice_spacing, 5.0)
        self.assertEqual(sampling.total_range_mm, 300.0)

    def test_spacing_and_count_calculate_thickness(self):
        sampling = resolve_sampling(300.0, None, 5.0, 60)

        self.assertEqual(sampling.slice_thickness, 5.0)
        self.assertEqual(sampling.total_range_mm, 300.0)

    def test_larger_total_range_extends_both_ends_equally(self):
        sampling = resolve_sampling(300.0, 5.0, 5.0, 70)

        self.assertEqual(sampling.total_range_mm, 350.0)
        self.assertEqual(sampling.start_extension_mm, 25.0)
        self.assertEqual(sampling.end_extension_mm, 25.0)
        self.assertFalse(sampling.adjusted)

    def test_smaller_total_range_keeps_count_and_evenly_adjusts_values(self):
        sampling = resolve_sampling(300.0, 3.0, 4.0, 50)

        self.assertEqual(sampling.slice_image_count, 50)
        self.assertEqual(sampling.slice_thickness, 6.0)
        self.assertEqual(sampling.slice_spacing, 6.0)
        self.assertEqual(sampling.total_range_mm, 300.0)
        self.assertTrue(sampling.adjusted)
        self.assertIn("有效范围除以数量", sampling.adjustment_reason)

    def test_zero_sampling_parameter_is_calculated_from_other_two(self):
        cases = (
            ((0.0, 5.0, 60), "slice_thickness"),
            ((5.0, 0.0, 60), "slice_spacing"),
            ((5.0, 5.0, 0), "slice_image_count"),
        )
        for values, calculated_field in cases:
            with self.subTest(calculated_field=calculated_field):
                sampling = resolve_sampling(300.0, *values)
                self.assertEqual(sampling.slice_thickness, 5.0)
                self.assertEqual(sampling.slice_spacing, 5.0)
                self.assertEqual(sampling.slice_image_count, 60)

    def test_count_only_evenly_divides_range_into_requested_number_of_slices(self):
        for missing in (None, 0):
            for count in (1, 20, 60):
                with self.subTest(missing=missing, count=count):
                    sampling = resolve_sampling(302.0, missing, missing, count)

                    self.assertEqual(sampling.slice_image_count, count)
                    self.assertAlmostEqual(sampling.slice_thickness, 302.0 / count)
                    self.assertEqual(sampling.slice_thickness, sampling.slice_spacing)
                    self.assertAlmostEqual(sampling.total_range_mm, 302.0)
                    self.assertAlmostEqual(sampling.start_extension_mm, 0.0)
                    self.assertAlmostEqual(sampling.end_extension_mm, 0.0)
                    self.assertFalse(sampling.adjusted)

    def test_thickness_or_spacing_only_uses_contiguous_slices_and_rounds_count_up(self):
        for values in (
            (5.0, 0.0, 0),
            (0.0, 5.0, 0),
            (5.0, None, None),
            (None, 5.0, None),
        ):
            with self.subTest(values=values):
                sampling = resolve_sampling(302.0, *values)

                self.assertEqual(sampling.slice_thickness, 5.0)
                self.assertEqual(sampling.slice_spacing, 5.0)
                self.assertEqual(sampling.slice_image_count, 61)
                self.assertEqual(sampling.total_range_mm, 305.0)
                self.assertEqual(sampling.start_extension_mm, 1.5)
                self.assertEqual(sampling.end_extension_mm, 1.5)
                self.assertFalse(sampling.adjusted)

    def test_thickness_or_spacing_only_covers_short_range_with_one_slice(self):
        for values in ((5.0, None, None), (None, 5.0, None)):
            with self.subTest(values=values):
                sampling = resolve_sampling(2.0, *values)

                self.assertEqual(sampling.slice_image_count, 1)
                self.assertEqual(sampling.slice_thickness, 5.0)
                self.assertEqual(sampling.slice_spacing, 5.0)
                self.assertEqual(sampling.start_extension_mm, 1.5)
                self.assertEqual(sampling.end_extension_mm, 1.5)

    def test_count_only_rejects_zero_or_tolerance_sized_derived_thickness(self):
        for effective_range, count in ((0.0, 1), (0.01, 20), (0.02, 20)):
            with self.subTest(effective_range=effective_range, count=count):
                with self.assertRaisesRegex(ValueError, "无法计算出有效的层厚、层间距"):
                    resolve_sampling(effective_range, 0.0, 0.0, count)

    def test_count_only_accepts_derived_thickness_above_tolerance(self):
        sampling = resolve_sampling(0.0202, 0.0, 0.0, 20)

        self.assertEqual(sampling.slice_image_count, 20)
        self.assertAlmostEqual(sampling.slice_thickness, 0.00101)
        self.assertEqual(sampling.slice_thickness, sampling.slice_spacing)
        self.assertAlmostEqual(sampling.total_range_mm, 0.0202)

    def test_invalid_parameter_count_is_rejected(self):
        for values in (
            (0.0, 0.0, 0),
            (None, 0.0, None),
            (None, None, None),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "必须至少传入其中一个"):
                    resolve_sampling(300.0, *values)

    def test_impossible_derived_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "层厚"):
            resolve_sampling(10.0, None, 5.0, 4)
        with self.assertRaisesRegex(ValueError, "层间距"):
            resolve_sampling(10.0, 10.0, None, 2)

    def test_thickness_and_one_image_still_cannot_determine_spacing(self):
        with self.assertRaisesRegex(ValueError, "数量为 1"):
            resolve_sampling(10.0, 10.0, None, 1)


if __name__ == "__main__":
    unittest.main()
