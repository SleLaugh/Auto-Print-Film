"""四棱柱备用包围几何必须保留主体边界及稳定方向。"""

import math
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.stages.localization_geometry import LocalizationSpineGeometryMixin as Geometry
from src.stages.stage_localization import LocalizationStage


class PrismEnclosingGeometryTest(unittest.TestCase):
    def assert_contains_points(self, quadrilateral, points):
        self.assertIsNotNone(quadrilateral)
        self.assertTrue(np.all(Geometry._points_inside_convex_polygon(
            points,
            quadrilateral,
            tolerance=1e-6,
        )))

    def test_fallback_accepts_complete_hull_and_preserves_body_tilt(self):
        polygon = np.asarray(((0, 0), (10, 0), (10, 5), (6, 10), (0, 10)), dtype=float)
        angle = math.radians(23.0)
        rotation = np.asarray(((math.cos(angle), -math.sin(angle)),
                               (math.sin(angle), math.cos(angle))))
        points = polygon @ rotation.T + (123.4, -77.8)
        hull = Geometry._convex_hull_2d(points)
        self.assertGreater(len(hull), 4)

        quadrilateral = Geometry._oriented_enclosing_quadrilateral(points, hull)

        self.assert_contains_points(quadrilateral, points)
        self.assertAlmostEqual(abs(Geometry._signed_area_2d(quadrilateral)), 100.0)
        edges = np.roll(quadrilateral, -1, axis=0) - quadrilateral
        local_edges = edges @ rotation
        self.assertTrue(np.all(np.min(np.abs(local_edges), axis=1) < 1e-8))

    def test_chamfer_does_not_rotate_whole_enclosing_rectangle(self):
        points = np.asarray(((0, 0), (10, 0), (10, 5), (6, 10), (0, 10)), dtype=float)
        reference = np.asarray(((0, 0), (10, 0), (10, 5), (0, 10)), dtype=float)

        quadrilateral = Geometry._oriented_enclosing_quadrilateral(points, reference)

        self.assert_contains_points(quadrilateral, points)
        # 原先偏爱斜边会得到面积 156 的倾斜包络，倒角不代表整块主体的方向。
        self.assertAlmostEqual(abs(Geometry._signed_area_2d(quadrilateral)), 100.0)
        np.testing.assert_allclose(quadrilateral.min(axis=0), (0, 0), atol=1e-9)
        np.testing.assert_allclose(quadrilateral.max(axis=0), (10, 10), atol=1e-9)

    def test_limits_use_original_points_before_hull_rounding(self):
        reference = np.asarray(((0, 0), (10, 0), (10, 5), (6, 10), (0, 10)), dtype=float)
        points = np.vstack((reference, (-0.0004, 5), (5, 10.0004)))

        quadrilateral = Geometry._oriented_enclosing_quadrilateral(points, reference)

        self.assert_contains_points(quadrilateral, points)
        self.assertLessEqual(float(quadrilateral[:, 0].min()), -0.0004)
        self.assertGreaterEqual(float(quadrilateral[:, 1].max()), 10.0004)

    def test_repeated_closing_vertex_is_not_a_degenerate_body(self):
        points = np.asarray(((0, 0), (5, 0), (5, 3), (0, 3)), dtype=float)
        reference = np.vstack((points, points[0]))

        quadrilateral = Geometry._oriented_enclosing_quadrilateral(points, reference)

        self.assert_contains_points(quadrilateral, points)
        self.assertAlmostEqual(abs(Geometry._signed_area_2d(quadrilateral)), 15.0)

    def test_large_physical_origin_does_not_erase_small_polygon_area(self):
        points = np.asarray(((0, 0), (0.01, 0), (0.01, 0.02), (0, 0.02))) + 1e9
        expected_area = float(np.prod(np.ptp(points, axis=0)))

        self.assertAlmostEqual(Geometry._signed_area_2d(points), expected_area, places=12)
        self.assertAlmostEqual(Geometry._signed_area_2d(points[::-1]), -expected_area, places=12)
        quadrilateral = Geometry._expand_quadrilateral_to_contain_points(points, points)
        self.assert_contains_points(quadrilateral, points)

    def test_invalid_or_collinear_projection_cannot_make_a_prism(self):
        reference = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=float)
        collinear = np.asarray(((0, 0), (1, 0), (2, 0)), dtype=float)
        for points, contour in (
            (np.empty((0, 2)), reference),
            (collinear, collinear),
            (np.asarray(((0, 0), (1, np.nan))), reference),
            (reference, np.asarray(((0, 0), (1, 0), (1, np.inf)))),
        ):
            with self.subTest(points=points, reference=contour):
                self.assertIsNone(Geometry._oriented_enclosing_quadrilateral(points, contour))

    def test_stage_fallback_with_six_vertex_hull_keeps_all_body_voxels(self):
        body_mask = np.zeros((12, 20, 12), dtype=bool)
        for z_index in range(2, 9):
            y_max = 7 if z_index in (2, 8) else 8
            body_mask[z_index, 3:y_max + 1, 3:8] = True
        image = sitk.GetImageFromArray(body_mask.astype(np.uint8))
        indices = np.argwhere(body_mask)
        with patch.object(
            LocalizationStage,
            "_expand_quadrilateral_to_contain_points",
            return_value=None,
        ):
            geometry = LocalizationStage._sagittal_body_quadrilateral(
                indices,
                image,
                retain_all_projection=True,
            )

        self.assertIsNotNone(geometry)
        region = LocalizationStage._vertebral_body_prism_region(
            indices,
            image,
            retain_all_projection=True,
            sagittal_geometry=geometry,
        )
        self.assertIsNotNone(region)
        minimum, mask = region
        self.assertTrue(np.all(mask[tuple((indices - minimum).T)]))


if __name__ == "__main__":
    unittest.main()
