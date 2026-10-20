"""统一占比、逐层目标居中的物理几何与 DICOM 端到端契约。"""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import Request, PlaneConfig, WindowPreset
from src.models.enums import PlaneTypeEnum
from src.reconstruction.framing import TargetFramer, frame_curved_slice, curved_group_bounds
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.curved_stack import CurvedStackResampler
from src.reconstruction.dicom import write_ct_dicom
from src.stages.stage_reconstruction import ReconstructionStage
from src.stages.stage_reporting import ReportingStage


class SliceFramingTest(unittest.TestCase):
    def setUp(self):
        labels = np.zeros((9, 60, 100), np.uint8)
        labels[2, 15:25, 10:40] = 1  # 15mm × 10mm
        labels[5, 30:50, 60:70] = 1  # 5mm × 20mm
        labels[:, :10, 85:] = 2  # 干扰标签不得影响取景
        self.seg = sitk.GetImageFromArray(labels)
        self.seg.SetSpacing((0.5, 1.0, 2.0))
        self.raw = sitk.GetImageFromArray(np.where(labels == 1, 500, -1024).astype(np.float32))
        self.raw.CopyInformation(self.seg)
        self.coords = {"target": {"label_ids": [1]}, "other": {"label_ids": [2]}}
        self.plan = {
            "plan_id": "test", "slice_part": "腰椎", "slice_type": "MPR",
            "plane": "横状位", "geometry_type": "plane_stack", "group_name": "whole_range",
            "source_labels": ["target"], "window_presets": ["骨窗", "软组织窗"],
            "slice_thickness": 1.0, "slice_spacing": 6.0, "fill_percent": 80,
            "extent_mm": {"width": 49.5, "height": 59.0},
            "output_spacing_mm": {"x": 0.5, "y": 1.0, "z": 0.0},
            "slices": [{
                "index": i + 1, "center": {"x": 24.75, "y": 29.5, "z": z},
                "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            } for i, z in enumerate((4.0, 10.0))],
        }
        self.plan["range_start"] = self.plan["slices"][0]["center"]
        self.plan["range_end"] = self.plan["slices"][-1]["center"]
        self.plan["path_points"] = [self.plan["range_start"], self.plan["range_end"]]

    def test_slice_targets_are_centered_with_common_group_scale_without_distortion(self):
        framer = TargetFramer(self.seg, self.coords, self.plan)
        original = copy.deepcopy(self.plan)
        actual_spacing = []
        for item, xy in zip(self.plan["slices"], ((12.25, 19.5), (32.25, 39.5))):
            center = [*xy, item["center"]["z"]]
            target_size = [30, 35]
            plan, framed = framer.plane_geometry(self.plan, item)
            image = PlaneStackResampler().sample_slice(self.raw, plan, framed)
            self.assertEqual(image.GetSize(), (100, 60, 1))
            np.testing.assert_allclose(image.TransformContinuousIndexToPhysicalPoint((49.5, 29.5, 0)), center)
            spacing = np.array(image.GetSpacing()[:2])
            self.assertAlmostEqual(spacing[0] / spacing[1], 0.5)
            occupancy = np.array(target_size) / (np.array([100, 60]) * spacing)
            self.assertAlmostEqual(max(occupancy), 0.8)
            self.assertTrue(np.all(occupancy <= 0.8 + 1e-12))
            # 独立检查实际重采样像素，不能只验证取景元数据自身。
            visible = np.argwhere(sitk.GetArrayViewFromImage(image)[0] > 0)
            lower, upper = visible.min(axis=0), visible.max(axis=0)
            np.testing.assert_allclose((lower + upper) / 2, [29.5, 49.5], atol=0.5)
            self.assertTrue(np.all(lower > 0))
            self.assertTrue(np.all(upper < [59, 99]))
            self.assertEqual(framed["framing"]["center_source"], "slice_target")
            self.assertEqual(framed["center"]["z"], item["center"]["z"])
            actual_spacing.append(tuple(spacing))
        self.assertEqual(*actual_spacing)
        self.assertEqual(self.plan, original)

    def test_omitted_percent_only_centers(self):
        self.plan.pop("fill_percent")
        plan, item = TargetFramer(self.seg, self.coords, self.plan).plane_geometry(self.plan, self.plan["slices"][0])
        self.assertEqual(plan["output_spacing_mm"], self.plan["output_spacing_mm"])
        self.assertEqual(plan["extent_mm"], self.plan["extent_mm"])
        self.assertEqual(item["center"], {"x": 19.75, "y": 32.0, "z": 4.0})
        self.assertEqual(item["framing"]["center_source"], "group_target")

    def test_hundred_percent_touches_one_edge_pair(self):
        self.plan["fill_percent"] = 100
        plan, item = TargetFramer(self.seg, self.coords, self.plan).plane_geometry(self.plan, self.plan["slices"][0])
        size = np.array([100, 60])
        spacing = np.array([plan["output_spacing_mm"][a] for a in "xy"])
        self.assertAlmostEqual(max(np.array([30, 35]) / (size * spacing)), 1)

    def test_zero_percent_matches_omitted_for_mpr_and_interspace(self):
        for slice_type in ("MPR", "椎间隙"):
            with self.subTest(slice_type=slice_type):
                omitted = {k: v for k, v in self.plan.items() if k != "fill_percent"}
                omitted["slice_type"] = slice_type
                zero = {**omitted, "fill_percent": 0}
                framer = TargetFramer(self.seg, self.coords, zero)
                for item in self.plan["slices"]:
                    baseline, baseline_slice = framer.plane_geometry(omitted, item)
                    actual, actual_slice = framer.plane_geometry(zero, item)
                    self.assertEqual(actual["extent_mm"], baseline["extent_mm"])
                    self.assertEqual(actual["output_spacing_mm"], self.plan["output_spacing_mm"])
                    self.assertEqual(actual_slice["center"], baseline_slice["center"])
                    self.assertEqual(actual_slice["framing"]["scale"], 1.0)

    def test_zero_percent_matches_none_for_curved_group(self):
        masks = [sitk.Cast(self.seg[:, :, z:z+1] == 1, sitk.sitkUInt8) for z in (2, 5, 8)]
        bounds = curved_group_bounds(masks)
        for z, mask in zip((2, 5, 8), masks):
            source = self.raw[:, :, z:z+1]
            baseline, _ = frame_curved_slice(source, mask, None, bounds)
            actual, details = frame_curved_slice(source, mask, 0, bounds)
            self.assertEqual(actual.GetSpacing(), source.GetSpacing())
            self.assertEqual(actual.GetOrigin(), baseline.GetOrigin())
            self.assertEqual(actual.GetSize(), source.GetSize())
            self.assertEqual(details["scale"], 1.0)
            np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(baseline))

    def test_small_percent_can_shrink_target(self):
        self.plan["fill_percent"] = 10
        plan, item = TargetFramer(self.seg, self.coords, self.plan).plane_geometry(self.plan, self.plan["slices"][0])
        self.assertLess(item["framing"]["scale"], 1)
        self.assertGreater(plan["output_spacing_mm"]["x"], 0.5)

    def test_unrepresentable_fill_fails_before_resampling(self):
        framer = TargetFramer(self.seg, self.coords, self.plan)
        image = self.raw[:, :, 2:3]
        mask = sitk.Cast(self.seg[:, :, 2:3] == 1, sitk.sitkUInt8)
        for percent in (float.fromhex("0x0.0000000000001p-1022"), 1e-306):
            with self.subTest(percent=percent):
                self.plan["fill_percent"] = percent
                with self.assertRaisesRegex(ValueError, "物理视野"):
                    framer.plane_geometry(self.plan, self.plan["slices"][0])
                with self.assertRaisesRegex(ValueError, "物理视野"):
                    frame_curved_slice(image, mask, percent)

    def test_curved_target_mask_selects_only_plan_labels(self):
        plan = {**self.plan, "geometry_type": "curved_stack"}
        framer = TargetFramer(self.seg, self.coords, plan)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(framer.mask), sitk.GetArrayFromImage(self.seg) == 1,
        )
        self.assertEqual(framer.mask.GetSpacing(), self.seg.GetSpacing())
        self.assertEqual(framer.mask.GetDirection(), self.seg.GetDirection())
        self.assertEqual(framer.mask.GetOrigin(), self.seg.GetOrigin())

    def test_empty_slab_uses_the_same_group_geometry(self):
        item = copy.deepcopy(self.plan["slices"][0])
        item["center"]["z"] = 100
        _, framed = TargetFramer(self.seg, self.coords, self.plan).plane_geometry(self.plan, item)
        self.assertEqual(framed["framing"]["source"], "group_target")
        self.assertEqual(framed["framing"]["center_source"], "group_fallback")
        self.assertEqual(framed["center"], {"x": 19.75, "y": 32.0, "z": 100.0})

    def test_oblique_framing_preserves_cut_plane_and_voxel_geometry(self):
        angle = np.deg2rad(25)
        axes = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        self.seg.SetDirection(tuple(axes.ravel()))
        self.seg.SetOrigin((10, 20, 30))
        item = self.plan["slices"][0]
        center = axes @ np.array([24.75, 29.5, 4]) + self.seg.GetOrigin()
        for key, vector in zip(("row_dir", "col_dir", "normal", "center"), (*axes.T, center)):
            item[key] = dict(zip("xyz", vector))
        _, framed = TargetFramer(self.seg, self.coords, self.plan).plane_geometry(self.plan, item)
        final = np.array(list(framed["center"].values()))
        np.testing.assert_allclose(final, axes @ [12.25, 19.5, 4] + self.seg.GetOrigin())
        self.assertAlmostEqual(np.dot(final - center, axes[:, 2]), 0)

    def test_missing_target_is_not_silently_treated_as_background(self):
        self.plan["source_labels"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "label_ids"):
            TargetFramer(self.seg, self.coords, self.plan)

    def test_curved_display_frame_has_same_fill_semantics(self):
        image = self.raw[:, :, 2:3]
        mask = sitk.Cast(self.seg[:, :, 2:3] == 1, sitk.sitkUInt8)
        framed, details = frame_curved_slice(image, mask, 80)
        self.assertEqual(framed.GetSize(), image.GetSize())
        np.testing.assert_allclose(framed.TransformContinuousIndexToPhysicalPoint((49.5, 29.5, 0)), [12.25, 19.5, 4])
        occupancy = np.array([15, 10]) / (np.array([100, 60]) * framed.GetSpacing()[:2])
        self.assertAlmostEqual(max(occupancy), 0.8)
        centered, _ = frame_curved_slice(image, mask, None)
        self.assertEqual(centered.GetSpacing(), image.GetSpacing())

    def test_curved_empty_mask_requires_explicit_group_bounds(self):
        image = self.raw[:, :, 0:1]
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
        with self.assertRaisesRegex(ValueError, "全组"):
            frame_curved_slice(image, mask, 80)
        _, details = frame_curved_slice(image, mask, 80, (np.array([9.5, 14.5]), np.array([39.5, 24.5])))
        self.assertEqual(details["source"], "group_target")
        self.assertEqual(details["center_source"], "group_fallback")

    def test_tiny_end_slice_keeps_group_scale_and_centers_its_own_target(self):
        labels = sitk.GetArrayFromImage(self.seg)
        labels[8, 20, 20] = 1
        segmentation = sitk.GetImageFromArray(labels)
        segmentation.CopyInformation(self.seg)
        framer = TargetFramer(segmentation, self.coords, self.plan)
        first_plan, first = framer.plane_geometry(self.plan, self.plan["slices"][0])
        tip = copy.deepcopy(self.plan["slices"][0])
        tip["center"] = {"x": 5, "y": 10, "z": 16}
        tip_plan, tip = framer.plane_geometry(self.plan, tip)
        self.assertEqual(first_plan["output_spacing_mm"], tip_plan["output_spacing_mm"])
        self.assertEqual(first["framing"]["scale"], tip["framing"]["scale"])
        self.assertEqual(tip["center"], {"x": 10.0, "y": 20.0, "z": 16.0})
        self.assertEqual(tip["framing"]["slice_target_size_mm"], [0.5, 1.0])

    def test_interspace_groups_only_use_their_own_labels(self):
        other_plan = {**self.plan, "source_labels": ["other"], "slice_type": "椎间隙"}
        _, first = TargetFramer(self.seg, self.coords, self.plan).plane_geometry(self.plan, self.plan["slices"][0])
        _, other = TargetFramer(self.seg, self.coords, other_plan).plane_geometry(other_plan, other_plan["slices"][0])
        self.assertNotEqual(first["framing"]["target_size_mm"], other["framing"]["target_size_mm"])
        self.assertNotEqual(first["center"], other["center"])

    def test_curved_group_shares_scale_but_centers_each_nonempty_slice(self):
        masks = [sitk.Cast(self.seg[:, :, z:z+1] == 1, sitk.sitkUInt8) for z in (2, 5, 8)]
        bounds = curved_group_bounds(masks)
        spacings, centers = [], []
        for z, mask in zip((2, 5, 8), masks):
            image, details = frame_curved_slice(self.raw[:, :, z:z+1], mask, 80, bounds)
            spacings.append(image.GetSpacing())
            centers.append([details["center"][a] for a in "xy"])
            self.assertEqual(details["target_size_mm"], [30, 35])
        self.assertEqual(spacings[0], spacings[1])
        self.assertEqual(spacings[1], spacings[2])
        np.testing.assert_allclose(centers, [[12.25, 19.5], [32.25, 39.5], [19.75, 32.0]])

    def test_curved_mask_sampling_uses_nearest_neighbor_and_slab_union(self):
        labels = sitk.GetImageFromArray(np.array([[[0]], [[1]], [[0]]], np.uint8))
        sampler = CurvedStackResampler()
        sampled = sampler._sample_curved_surface_slab(
            labels, sitk.GetArrayViewFromImage(labels), np.array([[0, 0, 1.0]]),
            np.array([[1, 0, 0.0]]), np.array([[0, 0, 1.0]]), np.array([0.0]),
            [-1, 0, 1, 5], is_mask=True,
        )
        self.assertEqual(float(sampled[0, 0]), 1.0)

    def test_dicom_round_trip_keeps_common_spacing_and_actual_slice_planes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sitk.WriteImage(self.raw, str(root / "raw.nii.gz"))
            sitk.WriteImage(self.seg, str(root / "seg.nii.gz"))
            request = Request("/tmp/dicom", str(root), plane_configs=[
                PlaneConfig(PlaneTypeEnum.横状位, 1, 6, ["骨窗", "软组织窗"], fill_percent=80)
            ])
            context = ProcessingContext(request)
            context.set_result("localization", {
                "raw_for_reconstruction_path": str(root / "raw.nii.gz"),
                "seg_for_reconstruction_path": str(root / "seg.nii.gz"), "coords": self.coords,
            })
            context.set_result("planning", {"plans": [self.plan]})
            ReconstructionStage(config={"parallel": True}).execute(context)
            result = context.get_result("reconstruction")
            for index, expected_center in enumerate(([12.25, 19.5, 4], [32.25, 39.5, 10]), 1):
                bone = pydicom.dcmread(root / "横状位" / "骨窗" / f"{index}.dcm")
                soft = pydicom.dcmread(root / "横状位" / "软组织窗" / f"{index}.dcm")
                for field in ("ImagePositionPatient", "PixelSpacing", "ImageOrientationPatient", "Rows", "Columns"):
                    self.assertEqual(getattr(bone, field), getattr(soft, field))
                center = np.asarray(bone.ImagePositionPatient, float) + np.array([
                    (bone.Columns - 1) * float(bone.PixelSpacing[1]) / 2,
                    (bone.Rows - 1) * float(bone.PixelSpacing[0]) / 2, 0,
                ])
                np.testing.assert_allclose(center, expected_center, atol=1e-7)
                self.assertEqual(float(bone.SliceThickness), 1)
                self.assertEqual(float(bone.SpacingBetweenSlices), 6)
                expected = result["outputs"][0]["framing"][index - 1]["output_spacing_mm"]
                np.testing.assert_allclose(list(map(float, bone.PixelSpacing)), [expected["y"], expected["x"]])
            directory = root / "横状位" / "骨窗"
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), str(bone.SeriesInstanceUID))
            positions = np.array([pydicom.dcmread(f).ImagePositionPatient for f in files], float)
            directions = np.asarray(bone.ImageOrientationPatient, float).reshape(2, 3)
            normal = np.cross(*directions)
            np.testing.assert_allclose(np.diff(positions, axis=0) @ normal, [6], atol=1e-7)
            np.testing.assert_allclose(np.diff(positions, axis=0), [[20, 20, 6]], atol=1e-7)
            baseline_root = root / "baseline"
            ReconstructionStage()._execute_plan(self.raw, self.plan, baseline_root, "1.2.3", {})
            baseline = pydicom.dcmread(baseline_root / "横状位" / "骨窗" / "Localizer.dcm")
            localizer = pydicom.dcmread(directory / "Localizer.dcm")
            for field in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
                self.assertEqual(getattr(localizer, field), getattr(baseline, field))
            # Fill 只影响正式切片，定位图背景与全幅定位线都必须保持一致。
            baseline_lines = np.flatnonzero(np.any(baseline.pixel_array > 55000, axis=1))
            actual_lines = np.flatnonzero(np.any(localizer.pixel_array > 55000, axis=1))
            np.testing.assert_array_equal(actual_lines, baseline_lines)
            background_rows = np.ones(localizer.Rows, dtype=bool)
            background_rows[baseline_lines] = False
            np.testing.assert_array_equal(
                localizer.pixel_array[background_rows], baseline.pixel_array[background_rows],
            )
            np.testing.assert_array_equal(localizer.pixel_array, baseline.pixel_array)
            ReportingStage().execute(context)

    def test_framed_dicom_slice_location_uses_actual_image_plane(self):
        """CPR 展开方向与规划参考平面不同时，位置标签也必须采用成片方向。"""
        angle = np.deg2rad(30)
        axes = np.array([
            [1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
        image = sitk.Image([8, 6, 1], sitk.sitkUInt16)
        image.SetDirection(tuple(axes.ravel()))
        image.SetOrigin((10, 20, 30))
        image.SetSpacing((0.5, 1.0, 1.0))
        planned_slice = {
            **self.plan["slices"][0],
            "framing": {"output_spacing_mm": {"x": 0.5, "y": 1.0, "z": 0.0}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slice.dcm"
            write_ct_dicom(
                image, path, self.plan, planned_slice,
                WindowPreset("骨窗", 2500, 400), "1.2.3", "1.2.4", 1,
            )
            dataset = pydicom.dcmread(path)
        np.testing.assert_allclose(
            np.array(dataset.ImageOrientationPatient, float), axes[:, :2].T.ravel(),
            atol=1e-8,
        )
        self.assertAlmostEqual(
            float(dataset.SliceLocation), float(np.dot(image.GetOrigin(), axes[:, 2])),
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
