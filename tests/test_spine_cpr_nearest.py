"""CPR 精确最近点查询的结果和原始全量距离算法保持一致。"""

import unittest

import numpy as np

from src.planning.spine_cpr import _nearest_curve_indices


def _brute_force_nearest(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """小规模基准直接使用原始平方距离及 np.argmin。"""

    return np.argmin(np.sum((points[:, None, :] - curve[None, :, :]) ** 2, axis=2), axis=1)


class SpineCprNearestTest(unittest.TestCase):
    def test_random_points_match_original_distances(self):
        random = np.random.default_rng(20261022)
        points = random.normal(size=(3000, 3)) * (40.0, 20.0, 300.0)
        curve = random.normal(size=(513, 3)) * (40.0, 20.0, 300.0)

        np.testing.assert_array_equal(
            _nearest_curve_indices(points, curve),
            _brute_force_nearest(points, curve),
        )

    def test_projected_physical_points_match_original_distances(self):
        random = np.random.default_rng(73)
        depths = np.linspace(-200.0, 350.0, 601)
        curve = np.column_stack((10.0 * np.sin(depths / 80.0), depths / 10.0, depths))
        points = random.uniform((-40.0, -50.0, -210.0), (40.0, 50.0, 360.0), (4000, 3))
        axis = np.asarray((1.0, 0.17, -0.08))
        axis /= np.linalg.norm(axis)
        curve -= np.outer(curve @ axis, axis)
        points -= np.outer(points @ axis, axis)

        np.testing.assert_array_equal(
            _nearest_curve_indices(points, curve),
            _brute_force_nearest(points, curve),
        )

    def test_equidistant_candidates_choose_smallest_original_index(self):
        curve = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)))
        points = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 3.0), (0.5, 0.5, 0.0)))

        np.testing.assert_array_equal(_nearest_curve_indices(points, curve), (0, 0, 0))

    def test_nearly_equal_distances_match_original_rounding(self):
        curve = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)))
        points = np.asarray(((1e-16, 0.0, 0.0), (-1e-16, 0.0, 0.0),
                             (0.0, 1e-14, 0.0), (0.0, -1e-14, 0.0)))

        np.testing.assert_array_equal(
            _nearest_curve_indices(points, curve),
            _brute_force_nearest(points, curve),
        )

    def test_duplicate_curve_points_preserve_first_original_index(self):
        curve = np.asarray(((3.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0), (-3.0, 0.0, 0.0)))
        points = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (-3.0, 0.0, 0.0)))

        np.testing.assert_array_equal(_nearest_curve_indices(points, curve), (1, 1, 3))

    def test_one_unique_curve_point_supports_duplicate_coordinates(self):
        points = np.asarray(((0.0, 0.0, 0.0), (5.0, -10.0, 20.0)))
        for curve in (np.asarray(((1.0, 2.0, 3.0),)), np.tile((1.0, 2.0, 3.0), (8, 1))):
            with self.subTest(curve_points=len(curve)):
                np.testing.assert_array_equal(_nearest_curve_indices(points, curve), (0, 0))

    def test_empty_points_return_empty_integer_indices(self):
        result = _nearest_curve_indices(np.empty((0, 3)), np.empty((0, 3)))

        self.assertEqual(result.shape, (0,))
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_nonempty_points_without_curve_keep_argmin_error(self):
        with self.assertRaises(ValueError):
            _nearest_curve_indices(np.zeros((1, 3)), np.empty((0, 3)))

    def test_non_contiguous_points_across_query_chunks(self):
        random = np.random.default_rng(51)
        points = random.normal(size=(65539, 6))[:, ::2]
        curve = random.normal(size=(23, 6))[:, ::2]

        np.testing.assert_array_equal(
            _nearest_curve_indices(points, curve),
            _brute_force_nearest(points, curve),
        )

    def test_float32_distance_ties_keep_original_rounding(self):
        points = np.zeros((1, 3), dtype=np.float32)
        curve = np.asarray(((1.0, 1e-4, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32)

        np.testing.assert_array_equal(_nearest_curve_indices(points, curve), (0,))

    def test_non_finite_coordinates_keep_original_argmin(self):
        points = np.asarray(((0.0, 0.0, 0.0), (np.nan, 0.0, 0.0), (np.inf, 0.0, 0.0)))
        curve = np.asarray(((2.0, 0.0, 0.0), (np.nan, 0.0, 0.0), (1.0, 0.0, 0.0)))

        np.testing.assert_array_equal(
            _nearest_curve_indices(points, curve),
            _brute_force_nearest(points, curve),
        )


if __name__ == "__main__":
    unittest.main()
