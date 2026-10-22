"""仅裁上下完全无源支持的行，完整保留已有像素并将有效视野居中。"""

import copy
import unittest
from unittest import mock

import numpy as np
import SimpleITK as sitk

from src.reconstruction.framing import TargetFramer
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.vertical_support import crop_vertical_padding


def _point(vector):
    return dict(zip("xyz", map(float, vector)))


def _slice(center, row=(1, 0, 0), col=(0, 1, 0), normal=(0, 0, 1), index=1):
    return {
        "index": index,
        "center": _point(center),
        "row_dir": _point(row),
        "col_dir": _point(col),
        "normal": _point(normal),
    }


def _plan(slices, width, height, spacing=(1, 1), fill=0, thickness=1):
    return {
        "plan_id": "vertical_source_support",
        "slice_part": "颈椎",
        "plane": "矢状位",
        "slice_type": "MPR",
        "geometry_type": "plane_stack",
        "source_labels": ["target"],
        "fill_percent": fill,
        "slice_thickness": thickness,
        "slice_spacing": 3,
        "output_spacing_mm": {"x": spacing[0], "y": spacing[1], "z": 0},
        "extent_mm": {"width": width, "height": height},
        "slices": slices,
    }


def _source(size=(101, 81, 121), spacing=(1, 1, 1), direction=None, ramp=False):
    """正数源像素与 -1024 补黑可分辨；真实空气另以相同几何测试。"""
    array = np.ones(tuple(reversed(size)), np.float32)
    if ramp:
        z, y, x = np.indices(array.shape)
        array = (100 + z * 25 + y * 0.3 + x * 0.1).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    direction = np.eye(3) if direction is None else np.asarray(direction)
    image.SetDirection(tuple(direction.ravel()))
    image.SetOrigin(tuple(-direction @ ((np.asarray(size) - 1) * spacing / 2)))
    return image


class VerticalFramingSupportTest(unittest.TestCase):
    def assert_crop_preserves_supported_pixels(self, source, plan, item):
        original_plan, original_item = copy.deepcopy(plan), copy.deepcopy(item)
        sampled = PlaneStackResampler().sample_slice(source, plan, item)
        array = sitk.GetArrayFromImage(sampled)[0]
        # 部分厚层的平均值可能小于 0，不能将正 HU 当作源支持的条件。
        occupied = np.flatnonzero(np.any(array > -1024 + 1e-3, axis=1))

        cropped, updated = crop_vertical_padding(source, plan, item, sampled)

        self.assertEqual(plan, original_plan)
        self.assertEqual(item, original_item)
        self.assertEqual(cropped.GetSpacing(), sampled.GetSpacing())
        self.assertEqual(cropped.GetDirection(), sampled.GetDirection())
        for field in ("index", "row_dir", "col_dir", "normal"):
            self.assertEqual(updated[field], item[field])
        if not len(occupied):
            details = updated["framing"]["vertical_source_support"]
            np.testing.assert_array_equal(sitk.GetArrayFromImage(cropped)[0], array)
            self.assertEqual(cropped.GetSize(), sampled.GetSize())
            self.assertEqual(cropped.GetOrigin(), sampled.GetOrigin())
            self.assertEqual(updated["center"], item["center"])
            self.assertEqual(details["status"], "no_source_support")
            return cropped, updated

        first, last = int(occupied[0]), int(occupied[-1])
        expected = array[first:last + 1]
        actual = sitk.GetArrayFromImage(cropped)[0]
        # ROI 必须逐像素无损：没有再采样，没有为保持目标中心而裁掉另一端有效行。
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(np.any(actual > -1024 + 1e-3, axis=1)))
        if first == 0 and last == len(array) - 1:
            self.assertEqual(cropped.GetSize(), sampled.GetSize())
            self.assertEqual(cropped.GetOrigin(), sampled.GetOrigin())
            self.assertEqual(updated["center"], item["center"])
            return cropped, updated
        details = updated["framing"]["vertical_source_support"]
        self.assertEqual(details["mode"], "source_support_crop")
        self.assertEqual(details["crop_top_rows"], first)
        self.assertEqual(details["crop_bottom_rows"], len(array) - 1 - last)
        self.assertEqual(details["original_rows"], len(array))
        cols, rows, _ = cropped.GetSize()
        spacing_x, spacing_y, _ = cropped.GetSpacing()
        self.assertEqual(updated["framing"]["output_size_xy"], [cols, rows])
        self.assertEqual(updated["framing"]["center"], updated["center"])
        np.testing.assert_allclose(
            [updated["framing"]["extent_mm"][key] for key in ("width", "height")],
            ((cols - 1) * spacing_x, (rows - 1) * spacing_y), atol=1e-9,
        )
        np.testing.assert_allclose(
            [updated["framing"]["output_spacing_mm"][key] for key in "xy"],
            (spacing_x, spacing_y), atol=1e-9,
        )
        expected_center = sampled.TransformContinuousIndexToPhysicalPoint(
            ((cols - 1) / 2, (first + last) / 2, 0)
        )
        actual_center = cropped.TransformContinuousIndexToPhysicalPoint(
            ((cols - 1) / 2, (rows - 1) / 2, 0)
        )
        np.testing.assert_allclose(actual_center, expected_center, atol=1e-8)
        np.testing.assert_allclose(list(updated["center"].values()), expected_center, atol=1e-8)
        np.testing.assert_allclose(
            cropped.GetOrigin(), sampled.TransformIndexToPhysicalPoint((0, first, 0)), atol=1e-8,
        )
        shift = np.asarray(expected_center) - list(item["center"].values())
        for field in ("row_dir", "normal"):
            self.assertAlmostEqual(np.dot(shift, list(item[field].values())), 0, places=8)
        return cropped, updated

    def test_fill_zero_group_centering_removes_only_bottom_padding(self):
        image = _source(ramp=True)
        labels = np.zeros((121, 81, 101), np.uint8)
        labels[10:31, 35:46, 48:53] = 1
        segmentation = sitk.GetImageFromArray(labels)
        segmentation.CopyInformation(image)
        item = _slice((0, 0, 0), row=(0, 1, 0), col=(0, 0, -1), normal=(-1, 0, 0))
        plan = _plan([item], width=120, height=120)
        framed_plan, framed_slice = TargetFramer(
            segmentation, {"target": {"label_ids": [1]}}, plan
        ).plane_geometry(plan, item)
        np.testing.assert_allclose(list(framed_slice["center"].values()), (0, 0, -40))

        cropped, updated = self.assert_crop_preserves_supported_pixels(
            image, framed_plan, framed_slice
        )

        details = updated["framing"]["vertical_source_support"]
        self.assertEqual(details["crop_top_rows"], 0)
        self.assertGreater(details["crop_bottom_rows"], 0)
        self.assertNotEqual(updated["center"], framed_slice["center"])
        array = sitk.GetArrayFromImage(cropped)[0]
        self.assertTrue(np.all(array[:, 0] == -1024), "左右仍允许保留源范围外留白")
        self.assertTrue(np.all(array[:, -1] == -1024))

    def test_all_planes_and_fill_values_with_rotated_reversed_anisotropic_source(self):
        yaw, tilt = np.deg2rad([27, 19])
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]) @ np.array([
            [1, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt), np.cos(tilt)],
        ])
        sizes, spacing = np.array([61, 81, 101]), np.array([0.7, 1.0, 1.3])
        for direction in (np.eye(3), rotation @ np.diag([-1, 1, -1])):
            image = _source(tuple(map(int, sizes)), tuple(spacing), direction)
            for row_axis, col_axis, normal_axis, plane in (
                (0, 1, 2, "横状位"), (0, 2, 1, "冠状位"), (1, 2, 0, "矢状位")
            ):
                for fill in (0, 80, 100):
                    with self.subTest(plane=plane, fill=fill, direction=direction):
                        index = (sizes - 1) / 2
                        index[col_axis] = (sizes[col_axis] - 1) * 0.75
                        center = image.TransformContinuousIndexToPhysicalPoint(tuple(index))
                        item = _slice(
                            center, direction[:, row_axis], direction[:, col_axis],
                            direction[:, normal_axis],
                        )
                        plan = _plan(
                            [item], width=(sizes[row_axis] + 39) * spacing[row_axis],
                            height=(sizes[col_axis] - 1) * spacing[col_axis] * 1.4,
                            spacing=spacing[[row_axis, col_axis]], fill=fill, thickness=2,
                        )
                        plan["plane"] = plane

                        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

                        original_rows = round(plan["extent_mm"]["height"] / spacing[col_axis]) + 1
                        self.assertLess(cropped.GetSize()[1], original_rows)
                        self.assertAlmostEqual(
                            updated["framing"]["extent_mm"]["width"], plan["extent_mm"]["width"]
                        )

    def test_oblique_thick_slab_respects_actual_canvas_columns(self):
        image = _source(size=(65, 65, 65), ramp=True)
        angle, tilt = np.deg2rad([45, 20])
        row = np.array([np.cos(angle), np.sin(angle), 0])
        col = np.array([-np.sin(angle), np.cos(angle), 0]) * np.cos(tilt)
        col[2] = np.sin(tilt)
        normal = np.cross(row, col)
        center = image.TransformContinuousIndexToPhysicalPoint((5.0, 35.0, 32.0))
        item = _slice(center, row, col, normal)
        plan = _plan([item], width=10, height=160, fill=100, thickness=3)

        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

        self.assertLess(cropped.GetSize()[1], 161)
        self.assertEqual(cropped.GetSize()[0], 11)
        self.assertGreater(updated["framing"]["vertical_source_support"]["crop_top_rows"], 0)
        self.assertGreater(updated["framing"]["vertical_source_support"]["crop_bottom_rows"], 0)

    def test_stack_keeps_different_effective_heights_without_padding_or_rescaling(self):
        image = _source(ramp=True)
        slices = [
            _slice((x, 0, z), row=(0, 1, 0), col=(0, 0, -1), normal=(-1, 0, 0), index=i)
            for i, (x, z) in enumerate(((-5, -20), (0, -40), (5, -30)), 1)
        ]
        plan = _plan(slices, width=120, height=120, fill=100, thickness=3)
        sizes = []
        for item in slices:
            cropped, _ = self.assert_crop_preserves_supported_pixels(image, plan, item)
            sizes.append(cropped.GetSize())

        self.assertEqual({size[1] for size in sizes}, {81, 91, 101})
        self.assertEqual({size[0] for size in sizes}, {121})
        self.assertEqual(plan["slice_thickness"], 3)
        self.assertEqual(plan["slice_spacing"], 3)

    def test_supported_extent_is_unchanged_even_when_sides_are_outside(self):
        image = _source()
        item = _slice((0, 0, 0))
        plan = _plan([item], width=160, height=30)

        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

        self.assertEqual(cropped.GetSize(), (161, 31, 1))
        self.assertEqual(updated["center"], item["center"])
        array = sitk.GetArrayFromImage(cropped)[0]
        self.assertTrue(np.all(array[:, 0] == -1024))
        self.assertTrue(np.all(array[:, -1] == -1024))

    def test_completely_unsupported_slice_retains_original_geometry_and_status(self):
        image = _source()
        item = _slice((0, 0, 500))
        plan = _plan([item], width=100, height=200, thickness=3)

        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

        self.assertEqual(cropped.GetSize(), (101, 201, 1))
        self.assertEqual(updated["framing"]["vertical_source_support"]["status"], "no_source_support")

    def test_half_voxel_lower_boundary_is_supported_but_upper_boundary_is_outside(self):
        image = _source(size=(5, 5, 5))
        item = _slice((0, 0, 0))
        plan = _plan([item], width=4, height=7)
        sampled = PlaneStackResampler().sample_slice(image, plan, item)
        array = sitk.GetArrayFromImage(sampled)[0]
        # y=-2.5 对应源连续索引 -0.5；y=2.5 对应索引 size-0.5。
        self.assertTrue(np.all(array[1] == 1))
        self.assertTrue(np.all(array[6] == -1024))

        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

        self.assertEqual(cropped.GetSize(), (5, 5, 1))
        details = updated["framing"]["vertical_source_support"]
        self.assertEqual(details["crop_top_rows"], 1)
        self.assertEqual(details["crop_bottom_rows"], 2)

    def test_half_voxel_verification_does_not_allocate_source_mask_or_resample_slab(self):
        # 常见轴对齐、Fill=0 也会恰落半体素边界；仅两行需要插值器复核。
        image = _source(size=(97, 81, 65))
        item = _slice((0, 0, 0))
        plan = _plan([item], width=96, height=85)
        sampled = PlaneStackResampler().sample_slice(image, plan, item)
        image_constructor = sitk.Image
        resampler_constructor = sitk.ResampleImageFilter
        constructed_sizes, verification_sizes = [], []

        def tracked_image(*args, **kwargs):
            result = image_constructor(*args, **kwargs)
            constructed_sizes.append(result.GetSize())
            return result

        class RecordingResampler:
            def __init__(self):
                self.filter = resampler_constructor()

            def __getattr__(self, name):
                return getattr(self.filter, name)

            def Execute(self, source):
                # 直接用原始有限 HU 复核；不能再创建与源体积同尺寸的掩膜。
                self_test.assertIs(source, image)
                result = self.filter.Execute(source)
                verification_sizes.append(result.GetSize())
                return result

        self_test = self
        with (
            mock.patch.object(sitk, "Image", side_effect=tracked_image),
            mock.patch.object(sitk, "ResampleImageFilter", side_effect=RecordingResampler),
            mock.patch.object(
                PlaneStackResampler, "sample_slice",
                side_effect=AssertionError("裁边复核不能重新采样完整切片或厚层"),
            ),
        ):
            cropped, updated = crop_vertical_padding(image, plan, item, sampled)

        self.assertNotIn(image.GetSize(), constructed_sizes)
        self.assertTrue(verification_sizes, "该用例必须覆盖半体素边界复核路径")
        self.assertTrue(all(
            size[0] <= sampled.GetSize()[0] and size[1] <= sampled.GetSize()[1] and size[2] == 1
            for size in verification_sizes
        ))
        self.assertEqual(cropped.GetSize(), (97, 81, 1))
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(cropped), sitk.GetArrayFromImage(sampled)[:, 2:83],
        )
        details = updated["framing"]["vertical_source_support"]
        self.assertEqual(details["crop_top_rows"], 2)
        self.assertEqual(details["crop_bottom_rows"], 3)

    def test_unambiguous_source_bounds_need_no_additional_sampling(self):
        image = _source(size=(97, 81, 65))
        item = _slice((0, 0, 0))
        # 输出行中心落在整数源索引，不触碰上下半体素边界。
        plan = _plan([item], width=96, height=84)
        sampled = PlaneStackResampler().sample_slice(image, plan, item)

        with (
            mock.patch.object(
                sitk, "ResampleImageFilter",
                side_effect=AssertionError("明确源支持边界必须直接使用几何计算结果"),
            ),
            mock.patch.object(
                PlaneStackResampler, "sample_slice",
                side_effect=AssertionError("明确源支持边界不需要重新采样切片"),
            ),
        ):
            cropped, updated = crop_vertical_padding(image, plan, item, sampled)

        self.assertEqual(cropped.GetSize(), (97, 81, 1))
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(cropped), sitk.GetArrayFromImage(sampled)[:, 2:83],
        )
        details = updated["framing"]["vertical_source_support"]
        self.assertEqual(details["crop_top_rows"], 2)
        self.assertEqual(details["crop_bottom_rows"], 2)

    def test_partial_thick_slab_keeps_support_when_middle_plane_is_outside_source(self):
        image = _source(size=(61, 61, 61))
        item = _slice((0, 25, 31))
        plan = _plan([item], width=60, height=120, thickness=4)
        sampled = PlaneStackResampler().sample_slice(image, plan, item)
        array = sitk.GetArrayFromImage(sampled)[0]
        self.assertLess(array.max(), 0)
        self.assertGreater(array.max(), -1024)

        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

        self.assertEqual(cropped.GetSize()[1], 61)
        self.assertEqual(updated["center"]["z"], 31)

    def test_nearly_axis_aligned_slab_uses_actual_source_layer_normal_offsets(self):
        image = _source(size=(100, 7, 7)) * 100
        image.SetOrigin((0, 0, 0))
        angle = 0.001
        item = _slice(
            (99.5005, 3, 3), row=(0, -1, 0),
            col=(np.cos(angle), 0, -np.sin(angle)),
            normal=(np.sin(angle), 0, np.cos(angle)),
        )
        plan = _plan([item], width=4, height=4, thickness=1)
        sampled = PlaneStackResampler().sample_slice(image, plan, item)
        array = sitk.GetArrayFromImage(sampled)[0]
        np.testing.assert_array_equal(
            np.any(array > -1024 + 1e-3, axis=1), [True, True, False, False, False]
        )

        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

        self.assertEqual(cropped.GetSize(), (5, 2, 1))
        self.assertTrue(np.any(sitk.GetArrayFromImage(cropped)[0, -1] > -1024 + 1e-3))
        self.assertEqual(updated["framing"]["vertical_source_support"]["crop_bottom_rows"], 3)

    def test_rotated_half_voxel_rounding_preserves_every_actual_supported_row(self):
        direction = np.array([
            [-0.09427049925515951, -0.9150882496048981, 0.3920797985171215],
            [-0.23630666541940332, -0.3620102793587254, -0.9017248568809556],
            [0.9670947383005167, -0.17765712221484764, -0.1821145630579652],
        ])
        image = _source(size=(10, 7, 5), spacing=(0.7, 1.1, 1.4), direction=direction) * 100
        image.SetOrigin((-34.15143757958418, 83.35406925363432, 52.55299461339595))
        center = image.TransformContinuousIndexToPhysicalPoint((4.5, 3.0, 2.0))
        item = _slice(center, direction[:, 0], direction[:, 1], direction[:, 2])
        plan = _plan([item], width=6.3, height=9.9, spacing=(0.7, 1.1), thickness=0.5)
        sampled = PlaneStackResampler().sample_slice(image, plan, item)
        array = sitk.GetArrayFromImage(sampled)[0]
        np.testing.assert_array_equal(
            np.any(array > -1024 + 1e-3, axis=1),
            [False, True, True, True, True, True, True, True, True, False],
        )

        cropped, updated = self.assert_crop_preserves_supported_pixels(image, plan, item)

        self.assertEqual(cropped.GetSize(), (10, 8, 1))
        details = updated["framing"]["vertical_source_support"]
        self.assertEqual(details["crop_top_rows"], 1)
        self.assertEqual(details["crop_bottom_rows"], 1)

    def test_real_air_rows_inside_source_volume_are_not_removed(self):
        support = _source()
        image = sitk.GetImageFromArray(np.full((121, 81, 101), -1024, np.float32))
        image.CopyInformation(support)
        item = _slice((0, 25, 0))
        plan = _plan([item], width=100, height=120)
        sampled = PlaneStackResampler().sample_slice(image, plan, item)
        supported_sampled = PlaneStackResampler().sample_slice(support, plan, item)
        supported_array = sitk.GetArrayFromImage(supported_sampled)[0]
        occupied = np.flatnonzero(np.any(supported_array > -1024 + 1e-3, axis=1))

        cropped, updated = crop_vertical_padding(image, plan, item, sampled)

        self.assertEqual(cropped.GetSize()[1], int(occupied[-1] - occupied[0] + 1))
        self.assertGreater(cropped.GetSize()[1], 1)
        self.assertTrue(np.all(sitk.GetArrayFromImage(cropped) == -1024))
        self.assertEqual(updated["framing"]["vertical_source_support"]["crop_top_rows"], int(occupied[0]))
        self.assertEqual(updated["framing"]["vertical_source_support"]["crop_bottom_rows"], 120 - int(occupied[-1]))

    def test_real_air_on_half_voxel_boundary_is_preserved(self):
        item = _slice((0, 0, 0))
        plan = _plan([item], width=6, height=7)
        for pixel_type in (sitk.sitkFloat32, sitk.sitkInt16):
            with self.subTest(source_pixel_type=pixel_type):
                image = sitk.Cast(_source(size=(7, 5, 3)) * -1024, pixel_type)
                sampled = PlaneStackResampler().sample_slice(image, plan, item)

                cropped, updated = crop_vertical_padding(image, plan, item, sampled)

                self.assertEqual(cropped.GetSize(), (7, 5, 1))
                self.assertTrue(np.all(sitk.GetArrayFromImage(cropped) == -1024))
                details = updated["framing"]["vertical_source_support"]
                self.assertEqual(details["crop_top_rows"], 1)
                self.assertEqual(details["crop_bottom_rows"], 2)

    def test_nonfinite_hu_at_half_voxel_boundary_is_not_mistaken_for_padding(self):
        support = _source(size=(7, 5, 3))
        item = _slice((0, 0, 0))
        plan = _plan([item], width=6, height=7)
        supported_sampled = PlaneStackResampler().sample_slice(support, plan, item)
        support_array = sitk.GetArrayFromImage(supported_sampled)[0]
        occupied = np.flatnonzero(np.any(support_array > -1024 + 1e-3, axis=1))
        first, last = int(occupied[0]), int(occupied[-1])
        self.assertEqual((first, last), (1, 5))

        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(source_hu=value):
                image = sitk.GetImageFromArray(np.full((3, 5, 7), value, np.float32))
                image.CopyInformation(support)
                sampled = PlaneStackResampler().sample_slice(image, plan, item)

                cropped, updated = crop_vertical_padding(image, plan, item, sampled)

                self.assertEqual(cropped.GetSize(), (7, last - first + 1, 1))
                np.testing.assert_array_equal(
                    sitk.GetArrayFromImage(cropped),
                    sitk.GetArrayFromImage(sampled)[:, first:last + 1],
                )
                details = updated["framing"]["vertical_source_support"]
                self.assertEqual(details["crop_top_rows"], first)
                self.assertEqual(details["crop_bottom_rows"], 7 - last)


if __name__ == "__main__":
    unittest.main()
