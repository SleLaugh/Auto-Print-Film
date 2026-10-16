"""椎间隙跨计划共同取景几何，不依赖窗位或解剖尺寸相同。"""

import copy
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.reconstruction.framing import TargetFramer, prepare_interspace_framing
from src.reconstruction.plane_stack import PlaneStackResampler


class InterspaceSharedFramingTest(unittest.TestCase):
    def setUp(self):
        labels = np.zeros((9, 50, 70), np.uint8)
        labels[1, 10:20, 10:30] = 1
        labels[7, 20:40, 35:45] = 2
        self.segmentation = sitk.GetImageFromArray(labels)
        self.segmentation.SetSpacing((0.5, 1.0, 2.0))
        self.coords = {"first": {"label_ids": [1]}, "last": {"label_ids": [2]}}
        self.plans = []
        for name, extent, z in (("first", (30.0, 40.0), 2.0), ("last", (20.0, 50.0), 14.0)):
            self.plans.append({
                "plan_id": name, "slice_type": "椎间隙", "geometry_type": "plane_stack",
                "source_labels": [name], "fill_percent": 80,
                "extent_mm": dict(zip(("width", "height"), extent)),
                "output_spacing_mm": {"x": 0.5, "y": 1.0, "z": 0.0},
                "slice_thickness": 1.0,
                "slices": [{
                    "index": 1, "center": {"x": 0.0, "y": 0.0, "z": z},
                    "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
                }],
            })

    def _prepare(self, plans=None):
        return prepare_interspace_framing(
            self.segmentation, self.coords, self.plans if plans is None else plans,
        )

    def test_shared_matrix_and_scale_keep_each_target_center(self):
        original = copy.deepcopy(self.plans)
        geometry = self._prepare()
        self.assertEqual(geometry.size_xy, (61, 41))
        self.assertEqual(geometry.target_size_mm, (10.0, 20.0))
        self.assertEqual(geometry.plan_ids, ("first", "last"))
        self.assertEqual(geometry.reference_plan_ids, ("first",))
        self.assertAlmostEqual(geometry.scale, 1.64)
        np.testing.assert_allclose(geometry.spacing_xy, (0.5 / 1.64, 1.0 / 1.64))
        expected_centers = ((9.75, 14.5, 2.0), (19.75, 29.5, 14.0))
        for plan, expected_center, target_size in zip(self.plans, expected_centers, ((10, 10), (5, 20))):
            framer = TargetFramer(self.segmentation, self.coords, plan, shared_geometry=geometry)
            framed_plan, framed_slice = framer.plane_geometry(plan, plan["slices"][0])
            details = framed_slice["framing"]
            np.testing.assert_allclose([framed_slice["center"][key] for key in "xyz"], expected_center)
            self.assertEqual(details["target_size_mm"], list(target_size))
            self.assertEqual(details["shared_target_size_mm"], [10, 20])
            self.assertEqual(details["output_size_xy"], [61, 41])
            self.assertEqual(details["scale_scope"], "interspace_config")
            self.assertEqual(details["shared_plan_ids"], ["first", "last"])
            self.assertEqual(details["canvas_reference_plan_ids"], ["first"])
            sample = PlaneStackResampler().sample_slice(self.segmentation, framed_plan, framed_slice)
            self.assertEqual(sample.GetSize(), (61, 41, 1))
            np.testing.assert_allclose(sample.GetSpacing()[:2], geometry.spacing_xy)
            for key in ("normal", "row_dir", "col_dir", "index"):
                self.assertEqual(framed_slice[key], plan["slices"][0][key])
        self.assertEqual(self.plans, original)

    def test_zero_none_and_omitted_percent_share_base_spacing(self):
        self.plans[0]["fill_percent"] = 0
        self.plans[1].pop("fill_percent")
        for value in (None, 0):
            with self.subTest(value=value):
                self.plans[0]["fill_percent"] = value
                geometry = self._prepare()
                self.assertEqual(geometry.spacing_xy, (0.5, 1.0))
                self.assertEqual(geometry.size_xy, (61, 41))
                self.assertEqual(geometry.scale, 1.0)

    def test_every_slice_projection_contributes_to_target_maximum(self):
        plan = self.plans[1]
        oblique = copy.deepcopy(plan["slices"][0])
        angle = np.deg2rad(60)
        oblique["col_dir"] = {"x": 0.0, "y": np.cos(angle), "z": np.sin(angle)}
        oblique["normal"] = {"x": 0.0, "y": -np.sin(angle), "z": np.cos(angle)}
        plan["slices"].insert(0, oblique)
        geometry = self._prepare()
        self.assertEqual(geometry.target_size_mm, (10.0, 20.0))
        framer = TargetFramer(self.segmentation, self.coords, plan, shared_geometry=geometry)
        spacings = []
        for item in plan["slices"]:
            framed_plan, framed = framer.plane_geometry(plan, item)
            normal = np.array([item["normal"][key] for key in "xyz"])
            delta = np.array([framed["center"][key] - item["center"][key] for key in "xyz"])
            self.assertAlmostEqual(float(delta @ normal), 0.0)
            spacings.append(framed_plan["output_spacing_mm"])
        self.assertEqual(spacings[0], spacings[1])

    def test_only_one_target_point_cloud_is_retained_during_preparation(self):
        point_clouds = []
        original_init = TargetFramer.__init__

        def tracking_init(framer, *args, **kwargs):
            self.assertFalse(any(reference() is not None for reference in point_clouds))
            original_init(framer, *args, **kwargs)
            point_clouds.append(weakref.ref(framer.points))

        with patch.object(TargetFramer, "__init__", tracking_init):
            self._prepare()
        self.assertEqual(len(point_clouds), 2)
        self.assertFalse(any(reference() is not None for reference in point_clouds))

    def test_incompatible_base_spacing_or_fill_is_rejected(self):
        for field, value in (("fill_percent", 60), ("output_spacing_mm", {"x": 1, "y": 1})):
            with self.subTest(field=field):
                plans = copy.deepcopy(self.plans)
                plans[1][field] = value
                with self.assertRaisesRegex(ValueError, "一致"):
                    self._prepare(plans)

    def test_empty_or_missing_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "至少"):
            self._prepare([])
        self.plans[1]["source_labels"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "label_ids"):
            self._prepare()

    def test_canvas_uses_preceding_median_and_ignores_extreme_terminal_matrix(self):
        plans = []
        for index, size in enumerate(((61, 41), (63, 42), (65, 43), (501, 401), (5, 5))):
            plan = copy.deepcopy(self.plans[index % 2])
            plan["plan_id"] = f"region_{index}"
            plan["extent_mm"] = {"width": (size[0] - 1) * 0.5, "height": size[1] - 1.0}
            plans.append(plan)
        first = self._prepare(plans)
        self.assertEqual(first.size_xy, (64, 42))
        self.assertEqual(first.reference_plan_ids, tuple(f"region_{index}" for index in range(4)))
        plans[-1]["extent_mm"] = {"width": 50000.0, "height": 100000.0}
        second = self._prepare(plans)
        self.assertEqual(second, first)

    def test_single_group_uses_its_own_canvas(self):
        geometry = self._prepare(self.plans[:1])
        self.assertEqual(geometry.size_xy, (61, 41))
        self.assertEqual(geometry.plan_ids, ("first",))
        self.assertEqual(geometry.reference_plan_ids, ("first",))


if __name__ == "__main__":
    unittest.main()
