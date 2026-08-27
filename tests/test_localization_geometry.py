"""定位阶段脊柱几何基础工具的独立契约测试。"""

import unittest

import numpy as np

from src.stages.localization_geometry import LocalizationSpineGeometryMixin


class LocalizationGeometryTest(unittest.TestCase):
    """验证几何基础工具的边界行为。"""

    def test_bbox_from_region_uses_global_zyx_coordinates(self):
        local_mask = np.asarray(
            [
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=bool,
        )

        self.assertEqual(
            LocalizationSpineGeometryMixin._bbox_from_region(
                np.asarray([10, 20, 30]),
                local_mask,
            ),
            {
                "z_min": 10,
                "z_max": 11,
                "y_min": 20,
                "y_max": 21,
                "x_min": 30,
                "x_max": 31,
            },
        )

    def test_convex_hull_and_quadrilateral_cover_points(self):
        points = np.asarray(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 2.0],
                [0.0, 2.0],
                [1.5, 0.8],
            ]
        )

        hull = LocalizationSpineGeometryMixin._convex_hull_2d(points)
        quad = LocalizationSpineGeometryMixin._simplify_hull_to_quadrilateral(hull)

        self.assertIsNotNone(quad)
        self.assertTrue(
            np.all(
                LocalizationSpineGeometryMixin._points_inside_convex_polygon(
                    points,
                    quad,
                    tolerance=1e-6,
                )
            )
        )

    def test_empty_region_returns_none(self):
        self.assertIsNone(
            LocalizationSpineGeometryMixin._bbox_from_region(
                np.zeros(3, dtype=int),
                np.zeros((2, 2, 2), dtype=bool),
            )
        )


if __name__ == "__main__":
    unittest.main()
