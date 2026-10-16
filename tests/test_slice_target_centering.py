"""弯曲目标在相同物理比例下逐层居中，且不能改变原切面。"""

import copy
import unittest

import numpy as np
import SimpleITK as sitk

from src.reconstruction.framing import TargetFramer, curved_group_bounds, frame_curved_slice
from src.reconstruction.plane_stack import PlaneStackResampler


def _point(vector):
    return dict(zip("xyz", map(float, vector)))


def _plan(slices, spacing=(1, 1), thickness=1):
    return {
        "plan_id": "bent_target", "slice_type": "MPR", "geometry_type": "plane_stack",
        "source_labels": ["target"], "fill_percent": 100, "slice_thickness": thickness,
        "slice_spacing": 12, "output_spacing_mm": {"x": spacing[0], "y": spacing[1], "z": 0},
        "extent_mm": {"width": 63 * spacing[0], "height": 63 * spacing[1]}, "slices": slices,
    }


class SliceTargetCenteringTest(unittest.TestCase):
    coords = {"target": {"label_ids": [1]}}

    def test_bent_asymmetric_target_is_centered_in_all_planes_with_rotated_source_axes(self):
        angle = np.deg2rad(28)
        tilt = np.deg2rad(17)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0], [0, 0, 1],
        ]) @ np.array([
            [1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt), np.cos(tilt)],
        ])
        source_spacing = np.array([0.8, 1.0, 1.3])
        origin = np.array([-25.0, 14.0, 7.0])
        cases = (
            (0, 1, 2, "肺部", ("lung_left", "lung_right")),
            (1, 2, 0, "髋骨", ("hip_left", "hip_right")),
            (2, 0, 1, "头骨", ("skull_left", "skull_right")),
        )
        for row_axis, col_axis, normal_axis, part, names in cases:
            with self.subTest(part=part, normal_axis=normal_axis):
                labels = np.zeros((64, 64, 64), np.uint8)
                slices, expected = [], []
                # 相邻层目标偏移数十个像素，且目标的重心不等于包围框中心。
                for depth, x, y in ((12, 10, 40), (30, 28, 15), (48, 47, 45)):
                    for i in range(10):
                        for j in range(8):
                            if i < 5 and j < 4:
                                continue
                            index = np.zeros(3, int)
                            index[row_axis], index[col_axis], index[normal_axis] = x + i, y + j, depth
                            labels[tuple(index[::-1])] = 1 if i < 5 else 2
                    index = np.full(3, 31.5)
                    index[normal_axis] = depth
                    center = origin + rotation @ (index * source_spacing)
                    target_index = index.copy()
                    target_index[row_axis], target_index[col_axis] = x + 4.5, y + 3.5
                    expected.append(origin + rotation @ (target_index * source_spacing))
                    slices.append({
                        "index": len(slices) + 1, "center": _point(center),
                        "row_dir": _point(rotation[:, row_axis]),
                        "col_dir": _point(rotation[:, col_axis]),
                        "normal": _point(rotation[:, normal_axis]),
                    })
                segmentation = sitk.GetImageFromArray(labels)
                segmentation.SetDirection(tuple(rotation.ravel()))
                segmentation.SetSpacing(tuple(source_spacing))
                segmentation.SetOrigin(tuple(origin))
                raw = sitk.GetImageFromArray(np.where(labels > 0, 1500, -1024).astype(np.float32))
                raw.CopyInformation(segmentation)
                plan = _plan(slices, source_spacing[[row_axis, col_axis]], thickness=0.5)
                plan["slice_part"] = part
                plan["source_labels"] = list(names)
                coords = {name: {"label_ids": [i]} for i, name in enumerate(names, 1)}
                original = copy.deepcopy(plan)
                framer = TargetFramer(segmentation, coords, plan)
                output_spacings = []
                for item, expected_center in zip(slices, expected):
                    framed_plan, framed_slice = framer.plane_geometry(plan, item)
                    image = PlaneStackResampler().sample_slice(raw, framed_plan, framed_slice)
                    actual_center = image.TransformContinuousIndexToPhysicalPoint((31.5, 31.5, 0))
                    np.testing.assert_allclose(actual_center, expected_center, atol=1e-8)
                    shift = np.array(actual_center) - list(item["center"].values())
                    self.assertAlmostEqual(np.dot(shift, rotation[:, normal_axis]), 0, places=9)
                    for field in ("normal", "row_dir", "col_dir"):
                        self.assertEqual(framed_slice[field], item[field])
                    self.assertEqual(framed_plan["slice_thickness"], plan["slice_thickness"])
                    self.assertEqual(framed_plan["slice_spacing"], plan["slice_spacing"])
                    output_spacings.append(image.GetSpacing())
                    visible = np.argwhere(sitk.GetArrayViewFromImage(image)[0] > 0)
                    self.assertGreater(len(visible), 0)
                    lower, upper = visible.min(axis=0), visible.max(axis=0)
                    np.testing.assert_allclose((lower + upper) / 2, [31.5, 31.5], atol=1)
                    self.assertTrue(np.all(lower > 0))
                    self.assertTrue(np.all(upper < 63))
                self.assertEqual(output_spacings[0], output_spacings[1])
                self.assertEqual(output_spacings[1], output_spacings[2])
                self.assertEqual(plan, original)

    def test_thick_slab_includes_target_voxel_support_even_when_its_center_is_outside(self):
        labels = np.zeros((9, 64, 64), np.uint8)
        labels[2, 8:16, 4:14] = 1
        labels[6, 40:48, 44:54] = 1
        segmentation = sitk.GetImageFromArray(labels)
        segmentation.SetSpacing((1, 1, 2))
        item = {
            "index": 1, "center": _point([31.5, 31.5, 5.5]),
            "row_dir": _point([1, 0, 0]), "col_dir": _point([0, 1, 0]),
            "normal": _point([0, 0, 1]),
        }
        plan = _plan([item], thickness=2)
        _, framed = TargetFramer(segmentation, self.coords, plan).plane_geometry(plan, item)
        self.assertEqual(framed["center"], _point([8.5, 11.5, 5.5]))
        self.assertEqual(framed["framing"]["center_source"], "slice_target")
        self.assertEqual(framed["framing"]["slice_target_size_mm"], [10, 8])

    def test_oblique_slab_uses_projected_voxel_half_extent_for_target_overlap(self):
        labels = np.zeros((12, 64, 64), np.uint8)
        labels[5, 15, 10] = 1
        labels[5, 45, 50] = 1
        segmentation = sitk.GetImageFromArray(labels)
        angle = np.deg2rad(45)
        row, col, normal = np.array([
            [np.cos(angle), 0, -np.sin(angle)], [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ])
        target = np.array([10, 15, 5])
        center = target + normal * 0.7
        item = {
            "index": 1, "center": _point(center), "row_dir": _point(row),
            "col_dir": _point(col), "normal": _point(normal),
        }
        plan = _plan([item], thickness=0.2)
        _, framed = TargetFramer(segmentation, self.coords, plan).plane_geometry(plan, item)
        actual_center = np.array(list(framed["center"].values()))
        np.testing.assert_allclose(actual_center, center, atol=1e-10)
        self.assertEqual(framed["framing"]["center_source"], "slice_target")
        np.testing.assert_allclose(framed["framing"]["slice_target_size_mm"], [np.sqrt(2), 1])

    def test_curved_positive_fill_centers_each_asymmetric_target_without_changing_group_scale(self):
        images, masks = [], []
        for x, y in ((7, 36), (40, 9)):
            array = np.zeros((1, 64, 64), np.uint8)
            array[:, y:y+8, x:x+10] = 1
            array[:, y:y+4, x:x+5] = 0
            mask = sitk.GetImageFromArray(array)
            image = sitk.GetImageFromArray(np.where(array, 1000, -1024).astype(np.float32))
            images.append(image)
            masks.append(mask)
        bounds = curved_group_bounds(masks)
        spacings = []
        for image, mask in zip(images, masks):
            framed, details = frame_curved_slice(image, mask, 100, bounds)
            spacings.append(framed.GetSpacing())
            visible = np.argwhere(sitk.GetArrayViewFromImage(framed)[0] > 0)
            lower, upper = visible.min(axis=0), visible.max(axis=0)
            np.testing.assert_allclose((lower + upper) / 2, [31.5, 31.5], atol=1)
            self.assertTrue(np.all(lower > 0))
            self.assertTrue(np.all(upper < 63))
            self.assertEqual(details["center_source"], "slice_target")
            self.assertEqual(details["slice_target_size_mm"], [10, 8])
        self.assertEqual(*spacings)


if __name__ == "__main__":
    unittest.main()
