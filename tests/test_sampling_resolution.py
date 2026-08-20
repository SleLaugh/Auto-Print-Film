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

    def test_two_zero_sampling_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "必须传入其中两个或三个"):
            resolve_sampling(300.0, 0.0, 0.0, 20)

    def test_invalid_parameter_count_is_rejected(self):
        for values in (
            (5.0, None, None),
            (None, None, None),
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "必须传入其中两个或三个"):
                    resolve_sampling(300.0, *values)

    def test_impossible_derived_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "层厚"):
            resolve_sampling(10.0, None, 5.0, 4)
        with self.assertRaisesRegex(ValueError, "层间距"):
            resolve_sampling(10.0, 10.0, None, 2)

    def test_one_image_requires_all_parameters(self):
        with self.assertRaisesRegex(ValueError, "数量为 1"):
            resolve_sampling(10.0, 10.0, None, 1)


if __name__ == "__main__":
    unittest.main()
