"""末层附件与相邻主体共同取景，保持原切层范围及整个序列的比例。"""

import copy
import unittest

import numpy as np
import SimpleITK as sitk

from src.reconstruction.framing import TargetFramer, prepare_interspace_framing
from src.reconstruction.framing_context import FramingContext
from src.reconstruction.plane_stack import PlaneStackResampler


def _point(vector):
    return dict(zip("xyz", map(float, vector)))


class FramingContextTest(unittest.TestCase):
    def setUp(self):
        labels = np.zeros((10, 64, 64), np.uint8)
        labels[2, 20:40, 20:40] = 1
        labels[5, 42:46, 28:34] = 1  # 原范围末层只剩细小附件。
        labels[3, 20:40, 14:50] = 2
        labels[5, 20:40, 14:50] = 2  # 同层更大的相邻主体。
        labels[0, :60, :60] = 2  # 同类远处标签不能沿长轴扩大原取景范围。
        labels[9, :60, :60] = 2
        labels[5, :10, :10] = 3  # 未授权的同层标签不得拉偏中心。
        labels[7, 24:38, 22:46] = 4
        self.seg = sitk.GetImageFromArray(labels)
        self.labels = labels
        self.coords = {
            "source": {"label_ids": [1]}, "neighbor": {"label_ids": [2]},
            "unrelated": {"label_ids": [3]},
            "other_source": {"label_ids": [4]},
        }
        self.context = FramingContext(label_ids=(2,), axis_xyz=(0.0, 0.0, 1.0))
        self.plan = {
            "plan_id": "end_of_range", "slice_part": "胸椎", "slice_type": "MPR",
            "geometry_type": "plane_stack", "source_labels": ["source"], "fill_percent": 80,
            "slice_thickness": 1.0, "slice_spacing": 1.0,
            "extent_mm": {"width": 63.0, "height": 63.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slices": [{
                "index": i + 1, "center": _point([31.5, 31.5, z]),
                "row_dir": _point([1, 0, 0]), "col_dir": _point([0, 1, 0]),
                "normal": _point([0, 0, 1]),
            } for i, z in enumerate((2, 3, 4, 5))],
        }

    def test_end_slice_centers_and_retains_neighbor_body_and_source_appendage(self):
        raw_array = np.where(np.isin(self.labels, [1, 2]), 1500, -1024).astype(np.float32)
        raw = sitk.GetImageFromArray(raw_array)
        original = copy.deepcopy(self.plan)
        framer = TargetFramer(self.seg, self.coords, self.plan, context=self.context)
        spacings = []
        for item in self.plan["slices"]:
            framed_plan, framed = framer.plane_geometry(self.plan, item)
            spacings.append(framed_plan["output_spacing_mm"])
            self.assertEqual(framed["normal"], item["normal"])
            self.assertEqual(framed["center"]["z"], item["center"]["z"])
            self.assertEqual(framed_plan["slice_thickness"], self.plan["slice_thickness"])
            self.assertEqual(framed_plan["slice_spacing"], self.plan["slice_spacing"])
        np.testing.assert_allclose([spacings[0][axis] for axis in "xy"], [36 / 64 / 0.8] * 2)
        self.assertTrue(all(spacing == spacings[0] for spacing in spacings))
        self.assertEqual(framed["center"], _point([31.5, 32.5, 5]))
        image = PlaneStackResampler().sample_slice(raw, framed_plan, framed)
        visible = np.argwhere(sitk.GetArrayViewFromImage(image)[0] > 0)
        lower, upper = visible.min(axis=0), visible.max(axis=0)
        np.testing.assert_allclose((lower + upper) / 2, [31.5, 31.5], atol=0.5)
        self.assertTrue(np.all(lower > 0))
        self.assertTrue(np.all(upper < 63))
        # 检查真实像素完整覆盖大主体和附件，而非只验证 framing 元数据。
        self.assertGreaterEqual(upper[1] - lower[1] + 1, 50)
        self.assertGreaterEqual(upper[0] - lower[0] + 1, 36)
        self.assertEqual(self.plan, original)

    def test_context_only_slice_is_centered_and_truly_empty_slice_uses_group_fallback(self):
        framer = TargetFramer(self.seg, self.coords, self.plan, context=self.context)
        _, context_only = framer.plane_geometry(self.plan, self.plan["slices"][1])
        self.assertEqual(context_only["center"], _point([31.5, 29.5, 3]))
        self.assertEqual(context_only["framing"]["center_source"], "slice_target")
        self.assertEqual(context_only["framing"]["slice_target_size_mm"], [36, 20])
        _, empty = framer.plane_geometry(self.plan, self.plan["slices"][2])
        self.assertEqual(empty["center"], _point([29.5, 32.5, 4]))
        self.assertEqual(empty["framing"]["center_source"], "group_fallback")

    def test_nonpositive_and_omitted_fill_ignore_context_for_plan_and_curved_mask(self):
        for percent in (0, None, "omitted"):
            with self.subTest(fill_percent=percent):
                plan = copy.deepcopy(self.plan)
                if percent == "omitted":
                    plan.pop("fill_percent")
                else:
                    plan["fill_percent"] = percent
                baseline = TargetFramer(self.seg, self.coords, plan)
                actual = TargetFramer(self.seg, self.coords, plan, context=self.context)
                for item in plan["slices"]:
                    self.assertEqual(actual.plane_geometry(plan, item), baseline.plane_geometry(plan, item))
                plan["geometry_type"] = "curved_stack"
                baseline = TargetFramer(self.seg, self.coords, plan)
                actual = TargetFramer(self.seg, self.coords, plan, context=self.context)
                np.testing.assert_array_equal(
                    sitk.GetArrayFromImage(actual.mask), sitk.GetArrayFromImage(baseline.mask),
                )

    def test_context_axis_crop_is_physical_and_invariant_to_axis_sign(self):
        angle = np.deg2rad(37)
        rotation = np.array([
            [np.cos(angle), 0, np.sin(angle)], [0, -1, 0],
            [np.sin(angle), 0, -np.cos(angle)],
        ])
        origin = np.array([15.0, -26.0, 31.0])
        self.seg.SetDirection(tuple(rotation.ravel()))
        self.seg.SetOrigin(tuple(origin))
        plan = copy.deepcopy(self.plan)
        for item in plan["slices"]:
            for name in ("center", "row_dir", "col_dir", "normal"):
                transformed = rotation @ np.array(list(item[name].values()))
                if name == "center":
                    transformed += origin
                item[name] = _point(transformed)
        for sign in (1, -1):
            with self.subTest(axis_sign=sign):
                context = FramingContext(label_ids=(2,), axis_xyz=tuple(sign * rotation[:, 2]))
                framer = TargetFramer(self.seg, self.coords, plan, context=context)
                np.testing.assert_allclose(framer.framing_target_size(plan, plan["slices"][-1]), [36, 26])
                _, framed = framer.plane_geometry(plan, plan["slices"][-1])
                expected = origin + rotation @ [31.5, 32.5, 5]
                np.testing.assert_allclose(list(framed["center"].values()), expected)

    def test_context_enlarges_the_shared_interspace_spacing_for_every_plan(self):
        first = copy.deepcopy(self.plan)
        first["slice_type"] = "椎间隙"
        second = copy.deepcopy(first)
        second["plan_id"] = "other_interspace"
        second["source_labels"] = ["other_source"]
        second["slices"] = [{**copy.deepcopy(first["slices"][0]), "center": _point([31.5, 31.5, 7])}]
        second["extent_mm"]["width"] = 31.0
        plans = [first, second]
        baseline = prepare_interspace_framing(self.seg, self.coords, plans)
        geometry = prepare_interspace_framing(self.seg, self.coords, plans, context=self.context)
        self.assertEqual(geometry.size_xy, (64, 64))
        self.assertEqual(geometry.target_size_mm, (36, 26))
        self.assertGreater(geometry.spacing_xy[0], baseline.spacing_xy[0])
        for plan in plans:
            framer = TargetFramer(self.seg, self.coords, plan, context=self.context, shared_geometry=geometry)
            for item in plan["slices"]:
                framed_plan, framed = framer.plane_geometry(plan, item)
                self.assertEqual(framed["normal"], item["normal"])
                self.assertEqual(framed["center"]["z"], item["center"]["z"])
                np.testing.assert_allclose(
                    [framed_plan["output_spacing_mm"][axis] for axis in "xy"], geometry.spacing_xy,
                )

    def test_curved_mask_includes_only_authorized_context_within_original_source_depth(self):
        plan = {**self.plan, "geometry_type": "curved_stack"}
        framer = TargetFramer(self.seg, self.coords, plan, context=self.context)
        expected = self.labels == 1
        expected[2:6] |= self.labels[2:6] == 2
        np.testing.assert_array_equal(sitk.GetArrayFromImage(framer.mask), expected)
        self.assertEqual(framer.mask.GetOrigin(), self.seg.GetOrigin())
        self.assertEqual(framer.mask.GetSpacing(), self.seg.GetSpacing())
        self.assertEqual(framer.mask.GetDirection(), self.seg.GetDirection())


if __name__ == "__main__":
    unittest.main()
