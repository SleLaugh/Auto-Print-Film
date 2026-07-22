import unittest

from src.models import ExtentMM, PlannedSlice, Point3D, SlicePlan


class SlicePlanModelsTest(unittest.TestCase):
    def test_slice_plan_to_dict_outputs_json_friendly_values(self):
        plan = SlicePlan(
            plan_id="腰椎_whole_range_横状位",
            body_part="腰椎",
            cut_mode="whole_range",
            plane="横状位",
            group_name="whole_range",
            geometry_type="plane_stack",
            range_start=Point3D(0.0, 0.0, 0.0),
            range_end=Point3D(0.0, 0.0, 10.0),
            path_points=[
                Point3D(0.0, 0.0, 0.0),
                Point3D(0.0, 0.0, 10.0),
            ],
            slices=[
                PlannedSlice(
                    index=1,
                    center=Point3D(0.0, 0.0, 0.0),
                    normal=Point3D(0.0, 0.0, 1.0),
                    row_dir=Point3D(1.0, 0.0, 0.0),
                    col_dir=Point3D(0.0, 1.0, 0.0),
                )
            ],
            extent_mm=ExtentMM(width=160.0, height=160.0),
            output_spacing_mm=Point3D(1.0, 1.0, 0.0),
            slice_thickness=5.0,
            slice_spacing=5.0,
            window_presets=["骨窗"],
            source_labels=["vertebrae_L1", "vertebrae_S1"],
            localizer_plane="冠状位",
            confidence=1.0,
            warnings=[],
        )

        data = plan.to_dict()

        self.assertEqual(data["slices"][0]["center"], {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        })
        self.assertEqual(data["slice_spacing"], 5.0)
        self.assertEqual(data["window_presets"], ["骨窗"])
        self.assertEqual(data["localizer_plane"], "冠状位")

    def test_slice_plan_to_dict_preserves_curved_stack_fields(self):
        plan = SlicePlan(
            plan_id="腰椎_CPR_矢状位",
            body_part="腰椎",
            cut_mode="cpr",
            plane="矢状位",
            group_name="CPR",
            geometry_type="curved_stack",
            range_start=Point3D(2.0, 3.0, 1.0),
            range_end=Point3D(2.0, 3.0, 5.0),
            path_points=[
                Point3D(2.0, 3.0, 1.0),
                Point3D(2.0, 3.0, 5.0),
            ],
            slices=[
                PlannedSlice(
                    index=1,
                    center=Point3D(-3.0, 3.0, 3.0),
                    normal=Point3D(1.0, 0.0, 0.0),
                    row_dir=Point3D(0.0, 1.0, 0.0),
                    col_dir=Point3D(0.0, 0.0, 1.0),
                    offset_mm=-5.0,
                ),
                PlannedSlice(
                    index=2,
                    center=Point3D(2.0, 3.0, 3.0),
                    normal=Point3D(1.0, 0.0, 0.0),
                    row_dir=Point3D(0.0, 1.0, 0.0),
                    col_dir=Point3D(0.0, 0.0, 1.0),
                    offset_mm=0.0,
                ),
            ],
            extent_mm=ExtentMM(width=120.0, height=4.0),
            output_spacing_mm=Point3D(1.0, 1.0, 0.0),
            slice_thickness=3.0,
            slice_spacing=5.0,
            window_presets=["骨窗"],
            source_labels=["vertebrae_L1", "vertebrae_L2", "vertebrae_L3"],
            curve_points=[
                Point3D(2.0, 3.0, 1.0),
                Point3D(2.0, 3.0, 3.0),
                Point3D(2.0, 3.0, 5.0),
            ],
            curve_spacing_mm=1.0,
            offset_axis=Point3D(1.0, 0.0, 0.0),
            offsets_mm=[-5.0, 0.0],
            confidence=1.0,
            warnings=[],
        )

        data = plan.to_dict()

        self.assertEqual(data["geometry_type"], "curved_stack")
        self.assertEqual(data["curve_points"][1], {
            "x": 2.0,
            "y": 3.0,
            "z": 3.0,
        })
        self.assertEqual(data["curve_spacing_mm"], 1.0)
        self.assertEqual(data["offset_axis"], {"x": 1.0, "y": 0.0, "z": 0.0})
        self.assertEqual(data["offsets_mm"], [-5.0, 0.0])
        self.assertEqual(data["slices"][0]["offset_mm"], -5.0)


if __name__ == "__main__":
    unittest.main()
