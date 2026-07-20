import unittest

from src.models import Point3D
from src.planning.geometry import (
    build_plane_stack_slices,
    interpolate_points,
    normalize_vector,
)
from src.planning.straight import cover_segment_range


class PlanningGeometryTest(unittest.TestCase):
    def test_cover_segment_range_centers_symmetrically_cover_non_divisible_target(self):
        """目标长度不能整除间距时，首末切片中心应在目标两端之外。"""

        range_start, range_end = cover_segment_range(
            target_start=Point3D(0.0, 0.0, 0.0),
            target_end=Point3D(0.0, 0.0, 272.35),
            slice_thickness=5.0,
            slice_spacing=5.0,
        )
        centers = interpolate_points(range_start, range_end, spacing=5.0)

        self.assertEqual(len(centers), 56)
        self.assertAlmostEqual(range_end.z - range_start.z, 275.0, places=6)
        self.assertAlmostEqual(range_start.z, -1.325, places=6)
        self.assertAlmostEqual(range_end.z, 273.675, places=6)
        self.assertAlmostEqual(0.0 - range_start.z, 1.325, places=6)
        self.assertAlmostEqual(range_end.z - 272.35, 1.325, places=6)

    def test_cover_segment_range_does_not_add_slice_when_exactly_divisible(self):
        """覆盖长度恰好匹配目标时，不应因浮点误差多生成一张切片。"""

        range_start, range_end = cover_segment_range(
            target_start=Point3D(0.0, 0.0, 0.0),
            target_end=Point3D(0.0, 0.0, 10.0),
            slice_thickness=2.0,
            slice_spacing=2.0,
        )
        centers = interpolate_points(range_start, range_end, spacing=2.0)

        self.assertEqual(
            [point.z for point in centers],
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        )

    def test_cover_segment_range_absorbs_oblique_coordinate_rounding(self):
        """斜向坐标的微米级舍入误差不应让名义 10mm 区间多出一张。"""

        target_start = Point3D(0.0, 0.0, 0.0)
        target_end = Point3D(7.07107, 7.07107, 0.0)
        range_start, range_end = cover_segment_range(
            target_start=target_start,
            target_end=target_end,
            slice_thickness=5.0,
            slice_spacing=5.0,
        )
        centers = interpolate_points(range_start, range_end, spacing=5.0)

        self.assertEqual(len(centers), 3)
        self.assertAlmostEqual(
            (range_start.x + range_end.x) / 2.0,
            (target_start.x + target_end.x) / 2.0,
            places=9,
        )
        self.assertAlmostEqual(
            (range_start.y + range_end.y) / 2.0,
            (target_start.y + target_end.y) / 2.0,
            places=9,
        )

    def test_cover_segment_range_uses_one_center_only_for_zero_length(self):
        """只有零长度目标可以只生成一张位于目标点的切片。"""

        range_start, range_end = cover_segment_range(
            target_start=Point3D(1.0, 2.0, 3.0),
            target_end=Point3D(1.0, 2.0, 3.0),
            slice_thickness=4.0,
            slice_spacing=1.0,
        )

        self.assertEqual(range_start.to_dict(), {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(range_end, range_start)

    def test_cover_segment_range_centers_cover_short_nonzero_target(self):
        """非零目标即使短于层厚和间距，首末中心也必须越过目标两端。"""

        range_start, range_end = cover_segment_range(
            target_start=Point3D(0.0, 0.0, 0.0),
            target_end=Point3D(0.0, 0.0, 2.0),
            slice_thickness=5.0,
            slice_spacing=5.0,
        )
        centers = interpolate_points(range_start, range_end, spacing=5.0)

        self.assertEqual([point.z for point in centers], [-1.5, 3.5])
        self.assertLessEqual(range_start.z, 0.0)
        self.assertGreaterEqual(range_end.z, 2.0)

    def test_cover_segment_range_keeps_requested_spacing_when_spacing_exceeds_thickness(self):
        """层间距大于层厚时仍保持固定间距，仅保证整体首尾包住目标。"""

        range_start, range_end = cover_segment_range(
            target_start=Point3D(0.0, 0.0, 0.0),
            target_end=Point3D(0.0, 0.0, 10.0),
            slice_thickness=2.0,
            slice_spacing=5.0,
        )

        centers = interpolate_points(range_start, range_end, spacing=5.0)
        self.assertEqual([point.z for point in centers], [0.0, 5.0, 10.0])

    def test_cover_segment_range_center_span_does_not_depend_on_thickness(self):
        """目标长度不能整除间距时，改变层厚不得改变首末中心和张数。"""

        center_ranges = []
        for slice_thickness in (2.0, 8.0):
            range_start, range_end = cover_segment_range(
                target_start=Point3D(0.0, 0.0, 0.0),
                target_end=Point3D(0.0, 0.0, 11.0),
                slice_thickness=slice_thickness,
                slice_spacing=5.0,
            )
            center_ranges.append([
                point.z
                for point in interpolate_points(
                    range_start,
                    range_end,
                    spacing=5.0,
                )
            ])

        self.assertEqual(center_ranges, [
            [-2.0, 3.0, 8.0, 13.0],
            [-2.0, 3.0, 8.0, 13.0],
        ])

    def test_cover_segment_range_rejects_non_positive_parameters(self):
        """层厚和层间距都必须是正数。"""

        for thickness, spacing, message in (
            (0.0, 1.0, "层厚必须大于 0"),
            (1.0, 0.0, "层间距必须大于 0"),
        ):
            with self.subTest(thickness=thickness, spacing=spacing):
                with self.assertRaisesRegex(ValueError, message):
                    cover_segment_range(
                        target_start=Point3D(0.0, 0.0, 0.0),
                        target_end=Point3D(0.0, 0.0, 10.0),
                        slice_thickness=thickness,
                        slice_spacing=spacing,
                    )

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

    def test_interpolate_points_does_not_append_non_uniform_endpoint(self):
        points = interpolate_points(
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(0.0, 0.0, 12.6),
            spacing=5.0,
        )

        self.assertEqual([point.z for point in points], [0.0, 5.0, 10.0])

    def test_interpolate_points_keeps_theoretical_final_step_with_float_error(self):
        points = interpolate_points(
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(0.3, 0.4, 0.0),
            spacing=0.1,
        )

        self.assertEqual(len(points), 6)

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
