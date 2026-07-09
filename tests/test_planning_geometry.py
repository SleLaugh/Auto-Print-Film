import unittest

from src.models import Point3D
from src.planning.geometry import (
    build_plane_stack_slices,
    interpolate_points,
    normalize_vector,
)


class PlanningGeometryTest(unittest.TestCase):
    def test_interpolate_points_generates_centers_from_start_to_end_by_spacing(self):
        points = interpolate_points(
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(0.0, 0.0, 10.0),
            spacing=5.0,
        )

        self.assertEqual([point.to_dict() for point in points], [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 5.0},
            {"x": 0.0, "y": 0.0, "z": 10.0},
        ])

    def test_normalize_vector_rejects_zero_vector(self):
        with self.assertRaisesRegex(ValueError, "方向向量长度不能为 0"):
            normalize_vector(Point3D(0.0, 0.0, 0.0))

    def test_build_plane_stack_slices_assigns_index_center_and_directions(self):
        slices = build_plane_stack_slices(
            path_points=[
                Point3D(0.0, 0.0, 0.0),
                Point3D(0.0, 0.0, 10.0),
            ],
            slice_spacing=5.0,
            normal=Point3D(0.0, 0.0, 1.0),
            row_dir=Point3D(1.0, 0.0, 0.0),
            col_dir=Point3D(0.0, 1.0, 0.0),
        )

        self.assertEqual(len(slices), 3)
        self.assertEqual(slices[1].index, 2)
        self.assertEqual(slices[1].center.to_dict(), {
            "x": 0.0,
            "y": 0.0,
            "z": 5.0,
        })
        self.assertEqual(slices[1].normal.to_dict(), {
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
        })


if __name__ == "__main__":
    unittest.main()
