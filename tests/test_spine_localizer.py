"""脊柱曲面定位背景的采样与真实切面交线契约。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pydicom
import SimpleITK as sitk

from src.models import WindowPreset
from src.models.reconstruction import ExtentMM, PlannedSlice, Point3D, SlicePlan
from src.reconstruction.spine_localizer import build_spine_localizer_surface
from src.stages.stage_reconstruction import ReconstructionStage


class SpineLocalizerTest(unittest.TestCase):
    """只验证定位背景，避免通过修改正式切片计划改善显示。"""

    @staticmethod
    def _image(array=None, *, transformed=False):
        if array is None:
            z, y, x = np.indices((13, 33, 31), dtype=float)
            array = x + 10.0 * y + 100.0 * z
        image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        if transformed:
            image.SetOrigin((-11.0, 23.0, 89.0))
            image.SetSpacing((1.7, 2.3, 3.4))
            image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0))
        return image

    @staticmethod
    def _physical(image, index):
        return Point3D(*image.TransformContinuousIndexToPhysicalPoint(tuple(index)))

    def _plan(self, image, *, part="颈椎", slice_type="CPR", indices=None):
        if indices is None:
            indices = [(15.0, 5.0, 2.0), (15.0, 17.0, 6.0), (15.0, 9.0, 10.0)]
        points = [self._physical(image, point) for point in indices]
        return SlicePlan(
            plan_id=f"{part}_{slice_type}_矢状位",
            slice_part=part,
            slice_type=slice_type,
            plane="矢状位",
            group_name=slice_type,
            geometry_type="curved_stack" if slice_type == "CPR" else "plane_stack",
            range_start=points[0],
            range_end=points[-1],
            path_points=[points[0], points[-1]],
            slices=[PlannedSlice(0, points[1], Point3D(1.0, 0.0, 0.0),
                                 Point3D(0.0, 1.0, 0.0), Point3D(0.0, 0.0, 1.0))],
            extent_mm=ExtentMM(32.0, 12.0),
            output_spacing_mm=Point3D(1.0, 1.0, 1.0),
            slice_thickness=1.0,
            slice_spacing=1.0,
            window_presets=["骨窗"],
            source_labels=["vertebrae_C2", "vertebrae_C3", "vertebrae_C4"],
            curve_points=points if slice_type == "CPR" else None,
            localizer_reference=self._physical(image, (15.0, 16.0, 6.0)),
            localizer_plane="冠状位",
        ).to_dict()

    def _annotation(self, image, index, *, prism=True):
        point = self._physical(image, index).to_dict()
        return {
            "found": True,
            "centroid_physical": point,
            "prism_found": prism,
            "prism_centroid_physical": point if prism else None,
        }

    @staticmethod
    def _anatomy(annotations):
        return {"spine_geometry": {"filtered_annotation_centroids": annotations}}

    def test_following_background_keeps_signals_at_different_anterior_depths(self):
        """不同前后深度的椎体信号应同时出现，固定冠状薄层会漏掉它们。"""

        array = np.zeros((13, 33, 31), dtype=np.float32)
        for z, y in ((2, 5), (6, 17), (10, 9)):
            array[z, y, 8:23] = 1000.0 + z
        image = self._image(array)
        surface = build_spine_localizer_surface(image, self._plan(image))

        self.assertIsNotNone(surface)
        sampled_image = surface.sample(image)
        self.assertEqual(sampled_image.GetSize(), (31, 1, 13))
        sampled = sitk.GetArrayFromImage(sampled_image)
        self.assertTrue(np.issubdtype(sampled.dtype, np.floating))
        for z in (2, 6, 10):
            self.assertAlmostEqual(float(sampled[z, 0, 15]), 1000.0 + z)
            self.assertEqual(float(array[z, 16, 15]), 0.0)

    def test_fractional_depth_is_interpolated_without_integer_layer_jumps(self):
        image = self._image()
        plan = self._plan(image, indices=[(15, 5.25, 2), (15, 9.25, 6),
                                         (15, 13.25, 10)])
        surface = build_spine_localizer_surface(image, plan)

        self.assertIsNotNone(surface)
        np.testing.assert_allclose(surface.depth_indices[2:11], np.arange(5.25, 14.25))
        sampled = sitk.GetArrayFromImage(surface.sample(image))
        self.assertAlmostEqual(float(sampled[6, 0, 7]), 699.5, places=4)

    def test_ct_integer_samples_are_interpolated_without_overflow(self):
        array = np.zeros((13, 33, 31), dtype=np.int16)
        array[:, 5, :] = -30000
        array[:, 6, :] = 30000
        image = sitk.GetImageFromArray(array)
        plan = self._plan(image, indices=[(15, 5.5, 2), (15, 5.5, 6), (15, 5.5, 10)])
        surface = build_spine_localizer_surface(image, plan)

        sampled = sitk.GetArrayFromImage(surface.sample(image))
        np.testing.assert_allclose(sampled, 0.0, atol=1e-6)

    def test_physical_path_honors_origin_spacing_rotation_and_reversed_head_foot(self):
        image = self._image(transformed=True)
        indices = [(15, 13.25, 10), (15, 9.25, 6), (15, 5.25, 2)]
        plan = self._plan(image, indices=indices)
        surface = build_spine_localizer_surface(image, plan)

        self.assertIsNotNone(surface)
        np.testing.assert_allclose(surface.depth_indices[2:11], np.arange(5.25, 14.25),
                                   atol=1e-6)
        sampled = sitk.GetArrayFromImage(surface.sample(image))
        self.assertAlmostEqual(float(sampled[6, 0, 7]), 699.5, places=4)

    def test_mpr_uses_valid_filtered_bodies_only_within_plan_labels(self):
        """C1及其他计划外标签不能拉偏当前主体背景，坏四棱柱可用过滤主体。"""

        image = self._image()
        plan = self._plan(image, slice_type="MPR")
        plan["source_labels"] = ["vertebrae_C2", "vertebrae_C3", "vertebrae_C4"]
        annotations = {
            "vertebrae_C1": self._annotation(image, (15, 31, 6)),
            "vertebrae_C2": self._annotation(image, (15, 6, 2)),
            "vertebrae_C3": self._annotation(image, (15, 12, 10), prism=False),
            "vertebrae_C4": {"found": False, "prism_found": False,
                              "centroid_physical": {"x": 15, "y": 30, "z": 6}},
            "vertebrae_T1": self._annotation(image, (15, 30, 6)),
        }
        surface = build_spine_localizer_surface(
            image, plan, anatomical_analysis=self._anatomy(annotations))

        self.assertIsNotNone(surface)
        self.assertAlmostEqual(float(surface.depth_indices[2]), 6.0)
        self.assertAlmostEqual(float(surface.depth_indices[6]), 9.0)
        self.assertAlmostEqual(float(surface.depth_indices[10]), 12.0)

    def test_mpr_prefers_prism_physical_centroid_and_anatomical_stage_geometry(self):
        image = self._image(transformed=True)
        plan = self._plan(image, slice_type="MPR")
        annotations = {
            "vertebrae_C2": self._annotation(image, (15, 6, 2)),
            "vertebrae_C3": self._annotation(image, (15, 12, 10)),
        }
        for annotation in annotations.values():
            annotation["centroid_physical"] = self._physical(image, (15, 30, 6)).to_dict()
        localization = self._anatomy({
            "vertebrae_C2": self._annotation(image, (15, 28, 2)),
            "vertebrae_C3": self._annotation(image, (15, 28, 10)),
        })
        surface = build_spine_localizer_surface(
            image, plan, localization, self._anatomy(annotations))

        self.assertIsNotNone(surface)
        self.assertAlmostEqual(float(surface.depth_indices[6]), 9.0)

    def test_mpr_geometry_source_uses_resolved_target_labels_instead_of_all_annotations(self):
        """生产 MPR 的来源可为整体坐标系，实际范围由当前定位目标限定。"""

        image = self._image()
        annotations = {
            "vertebrae_C1": {"found": False, "prism_found": False},
            "vertebrae_C2": self._annotation(image, (15, 6, 2)),
            "vertebrae_C3": self._annotation(image, (15, 12, 10)),
            "vertebrae_T1": self._annotation(image, (15, 30, 6)),
        }
        localization = {"label_targets": {
            "vertebrae_C1": {"label_ids": [1]},
            "vertebrae_C2": {"label_ids": [2]},
            "vertebrae_C3": {"label_ids": [3]},
        }}
        for source in ("anatomical_geometry.global_frame", "correction_scope_bbox"):
            with self.subTest(source=source):
                plan = self._plan(image, slice_type="MPR")
                plan["source_labels"] = [source]
                surface = build_spine_localizer_surface(
                    image, plan, localization, self._anatomy(annotations))
                self.assertIsNotNone(surface)
                self.assertAlmostEqual(float(surface.depth_indices[6]), 9.0)

    def test_lumbar_s1_target_can_read_existing_sacrum_body_geometry(self):
        image = self._image()
        plan = self._plan(image, part="腰椎", slice_type="MPR")
        plan["source_labels"] = ["vertebrae_L5", "vertebrae_S1"]
        surface = build_spine_localizer_surface(
            image, plan, anatomical_analysis=self._anatomy({
                "vertebrae_L5": self._annotation(image, (15, 6, 2)),
                "sacrum": self._annotation(image, (15, 12, 10)),
            }))

        self.assertIsNotNone(surface)
        self.assertAlmostEqual(float(surface.depth_indices[6]), 9.0)

    def test_three_spine_parts_share_mpr_and_cpr_localizer_support(self):
        image = self._image()
        for part, labels in (("颈椎", ["vertebrae_C2", "vertebrae_C3"]),
                             ("胸椎", ["vertebrae_T1", "vertebrae_T2"]),
                             ("腰椎", ["vertebrae_L1", "vertebrae_L2"])):
            for slice_type in ("MPR", "CPR"):
                with self.subTest(part=part, slice_type=slice_type):
                    plan = self._plan(image, part=part, slice_type=slice_type)
                    plan["source_labels"] = labels
                    anatomy = self._anatomy({
                        labels[0]: self._annotation(image, (15, 5, 2)),
                        labels[1]: self._annotation(image, (15, 9, 10)),
                    })
                    surface = build_spine_localizer_surface(
                        image, plan, anatomical_analysis=anatomy)
                    self.assertIsNotNone(surface)

    def test_oblique_mpr_lines_follow_surface_and_finite_output_rectangle(self):
        """有前后法向分量时必须画曲面真实交线，并裁到正式切片宽高。"""

        image = self._image()
        plan = self._plan(image, indices=[(15, 4, 0), (15, 16, 6), (15, 4, 12)])
        surface = build_spine_localizer_surface(image, plan)
        normal = np.array([1.0, 0.5, 0.0]) / np.sqrt(1.25)
        row = np.array([-0.5, 1.0, 0.0]) / np.sqrt(1.25)
        center = np.array([8.0, 16.0, 6.0])
        planned_slice = PlannedSlice(0, Point3D(*center), Point3D(*normal),
                                     Point3D(*row), Point3D(0.0, 0.0, 1.0))
        extent = ExtentMM(12.0, 8.0)
        lines = surface.plane_intersections(image, planned_slice.to_dict(), extent.to_dict())

        self.assertTrue(lines)
        for line in lines:
            self.assertEqual(line.shape[1], 3)
            self.assertGreaterEqual(len(line), 2)
        indices = np.concatenate(lines)
        self.assertTrue(np.isfinite(indices).all())
        physical = np.array([image.TransformContinuousIndexToPhysicalPoint(tuple(p))
                             for p in indices])
        relative = physical - center
        np.testing.assert_allclose(relative @ normal, 0.0, atol=1e-6)
        self.assertLessEqual(float(np.max(np.abs(relative @ row))), extent.width / 2 + 1e-6)
        self.assertLessEqual(float(np.max(np.abs(relative[:, 2]))), extent.height / 2 + 1e-6)
        expected_y = np.where(indices[:, 2] <= 6.0,
                              4.0 + 2.0 * indices[:, 2], 28.0 - 2.0 * indices[:, 2])
        np.testing.assert_allclose(indices[:, 1], expected_y, atol=1e-6)
        self.assertGreater(float(np.ptp(indices[:, 0])), 1.0)
        # 中间椎体深度高于首尾；交线保留弯折，不能复用单层背景上的直线。
        self.assertGreater(float(np.max(indices[:, 1]) -
                                 (indices[0, 1] + indices[-1, 1]) / 2), 1.0)

    def test_plane_without_surface_intersection_has_no_localizer_lines(self):
        image = self._image()
        surface = build_spine_localizer_surface(image, self._plan(image))
        planned_slice = PlannedSlice(0, Point3D(100.0, 0.0, 6.0),
                                     Point3D(1.0, 0.0, 0.0),
                                     Point3D(0.0, 1.0, 0.0), Point3D(0.0, 0.0, 1.0))
        self.assertEqual(surface.plane_intersections(
            image, planned_slice.to_dict(), ExtentMM(10, 10).to_dict()), [])

    def test_oblique_intersections_honor_image_affine_in_patient_space(self):
        image = self._image(transformed=True)
        plan = self._plan(image, indices=[(15, 4, 0), (15, 16, 6), (15, 4, 12)])
        surface = build_spine_localizer_surface(image, plan)
        direction = np.asarray(image.GetDirection()).reshape(3, 3)
        normal = (direction[:, 0] + 0.5 * direction[:, 1]) / np.sqrt(1.25)
        row = (-0.5 * direction[:, 0] + direction[:, 1]) / np.sqrt(1.25)
        col = direction[:, 2]
        center = np.asarray(image.TransformContinuousIndexToPhysicalPoint((8.0, 16.0, 6.0)))
        planned_slice = PlannedSlice(0, Point3D(*center), Point3D(*normal),
                                     Point3D(*row), Point3D(*col))
        extent = ExtentMM(30.0, 30.0)
        lines = surface.plane_intersections(image, planned_slice.to_dict(), extent.to_dict())

        self.assertTrue(lines)
        indices = np.concatenate(lines)
        physical = np.asarray([image.TransformContinuousIndexToPhysicalPoint(tuple(point))
                               for point in indices])
        relative = physical - center
        np.testing.assert_allclose(relative @ normal, 0.0, atol=1e-6)
        self.assertLessEqual(float(np.max(np.abs(relative @ row))), 15.0 + 1e-6)
        self.assertLessEqual(float(np.max(np.abs(relative @ col))), 15.0 + 1e-6)
        self.assertGreater(float(np.ptp(indices[:, 0])), 1.0)

    def test_transverse_intersection_is_finite_and_not_duplicated_at_surface_seam(self):
        """平面法向与左右轴垂直时仍有交线，共用曲面边界不能重复画线。"""

        image = self._image()
        surface = build_spine_localizer_surface(image, self._plan(image))
        planned_slice = PlannedSlice(0, Point3D(15.0, 17.0, 6.0),
                                     Point3D(0.0, 0.0, 1.0),
                                     Point3D(1.0, 0.0, 0.0), Point3D(0.0, 1.0, 0.0))
        lines = surface.plane_intersections(
            image, planned_slice.to_dict(), ExtentMM(10, 4).to_dict())

        self.assertEqual(len(lines), 1)
        np.testing.assert_allclose(lines[0][:, 1:], [[17, 6], [17, 6]], atol=1e-6)
        np.testing.assert_allclose(sorted(lines[0][:, 0]), [10, 20], atol=1e-6)

    def test_missing_or_degenerate_geometry_falls_back_to_existing_localizer(self):
        image = self._image()
        plan = self._plan(image, slice_type="MPR")
        full_centroids_only = {"coords": {
            "vertebrae_C2": {"found": True, "centroid": {"x": 15, "y": 5, "z": 2}},
            "vertebrae_C3": {"found": True, "centroid": {"x": 15, "y": 9, "z": 10}},
        }}
        self.assertIsNone(build_spine_localizer_surface(image, plan))
        self.assertIsNone(build_spine_localizer_surface(image, plan, full_centroids_only))
        for annotations in (
            {"vertebrae_C2": self._annotation(image, (15, 5, 2))},
            {"vertebrae_C2": self._annotation(image, (15, 5, 2)),
             "vertebrae_C3": self._annotation(image, (15, 9, 2))},
            {"vertebrae_C2": {"prism_found": True, "prism_centroid_physical": {"x": float("nan"), "y": 5, "z": 2}},
             "vertebrae_C3": self._annotation(image, (15, 9, 10))},
        ):
            with self.subTest(annotations=annotations):
                self.assertIsNone(build_spine_localizer_surface(
                    image, plan, anatomical_analysis=self._anatomy(annotations)))
        cpr_plan = self._plan(image, indices=[(15, 5, 2), (15, 9, 2), (15, 12, 2)])
        self.assertIsNone(build_spine_localizer_surface(image, cpr_plan))

    def test_other_parts_planes_and_slice_types_keep_existing_localizer(self):
        image = self._image()
        for field, value in (("slice_part", "头部"), ("plane", "冠状位"),
                             ("localizer_plane", "矢状位"), ("slice_type", "MIP")):
            with self.subTest(field=field, value=value):
                plan = self._plan(image)
                plan[field] = value
                self.assertIsNone(build_spine_localizer_surface(image, plan))

    def test_short_spine_label_names_remain_compatible(self):
        image = self._image()
        plan = self._plan(image, slice_type="MPR")
        plan["source_labels"] = ["C2", "C3"]
        surface = build_spine_localizer_surface(
            image, plan, anatomical_analysis=self._anatomy({
                "vertebrae_C2": self._annotation(image, (15, 6, 2)),
                "vertebrae_C3": self._annotation(image, (15, 12, 10)),
            }))

        self.assertIsNotNone(surface)
        self.assertAlmostEqual(float(surface.depth_indices[6]), 9.0)

    def test_mpr_renderer_uses_spine_background_for_all_spine_parts(self):
        """真实整切来源标识经过 renderer 后仍能显示不同深度的各椎体。"""

        array = np.zeros((13, 33, 31), dtype=np.float32)
        for z, y in ((2, 5), (6, 17), (10, 9)):
            array[z, y, 8:23] = 1000.0 + z
        image = self._image(array)
        for part, labels in (("颈椎", ["vertebrae_C2", "vertebrae_C3", "vertebrae_C4"]),
                             ("胸椎", ["vertebrae_T1", "vertebrae_T6", "vertebrae_T12"]),
                             ("腰椎", ["vertebrae_T12", "vertebrae_L3", "vertebrae_S1"]),
                             ("胸腰椎", ["vertebrae_C7", "vertebrae_T6", "vertebrae_S1"]),
                             ("全脊柱", ["vertebrae_C1", "vertebrae_T6", "vertebrae_S1"])):
            with self.subTest(part=part):
                plan = self._plan(image, part=part, slice_type="MPR")
                plan["source_labels"] = ["anatomical_geometry.global_frame"]
                localization = {"label_targets": {label: {"label_ids": [index]}
                                                  for index, label in enumerate(labels, 1)}}
                anatomy = self._anatomy({label: self._annotation(image, (15, y, z))
                                        for label, (z, y) in zip(labels, ((2, 5), (6, 17), (10, 9)))})
                plan_before = copy.deepcopy(plan)
                render = ReconstructionStage._build_plan_localizer_render(
                    image, plan, localization_result=localization,
                    anatomical_analysis=anatomy)

                self.assertIsNotNone(render)
                background = sitk.GetArrayFromImage(render.background_image)
                for z in (2, 6, 10):
                    # 正向 z 的原图在显示中头脚翻转；背景与定位线必须一致。
                    self.assertAlmostEqual(float(background[12 - z, 15]), 1000.0 + z)
                    self.assertEqual(float(array[z, 16, 15]), 0.0)
                self.assertEqual(render.background_kind, "spine_following_coronal")
                self.assertFalse(render.background_image.HasMetaDataKey("0028|135a"))
                self.assertIn("AP sampling depth varies", render.background_image.GetMetaData("0008|2111"))
                self.assertGreater(float(np.max(render.line_alpha)), 0.5)
                self.assertEqual(plan, plan_before)

    def test_cpr_renderer_preserves_offset_projection_and_antialias_coverage(self):
        """替换背景不改变 CPR offset 线位置及亚像素覆盖率。"""

        image = self._image()
        plan = self._plan(image)
        plan["offset_axis"] = {"x": 1.0, "y": 0.0, "z": 0.0}
        plan["offsets_mm"] = [-3.25, 0.0, 3.25]
        plan["curve_spacing_mm"] = 0.5
        render = ReconstructionStage._build_plan_localizer_render(image, plan)
        fixed_render = ReconstructionStage._build_curved_plan_localizer_render(image, plan)

        self.assertIsNotNone(render)
        np.testing.assert_array_equal(render.line_alpha, fixed_render.line_alpha)
        self.assertGreater(float(np.max(render.line_alpha[:, 12])), 0.5)
        self.assertGreater(float(np.max(render.line_alpha[:, 15])), 0.5)
        self.assertGreater(float(np.max(render.line_alpha[:, 18])), 0.5)
        self.assertGreater(np.count_nonzero((render.line_alpha > 0.0) &
                                           (render.line_alpha < 1.0)), 10)
        self.assertEqual(render.background_kind, "spine_following_coronal")
        self.assertEqual(fixed_render.background_kind, "orthogonal_plane")
        self.assertFalse(render.background_image.HasMetaDataKey("0028|135a"))
        self.assertFalse(np.array_equal(sitk.GetArrayFromImage(render.background_image),
                                        sitk.GetArrayFromImage(fixed_render.background_image)))

    def test_mpr_renderer_draws_actual_oblique_curve_instead_of_fixed_layer_line(self):
        image = self._image()
        plan = self._plan(image, slice_type="MPR")
        plan["extent_mm"] = {"width": 12.0, "height": 8.0}
        scale = np.sqrt(1.25)
        plan["slices"] = [PlannedSlice(
            0, Point3D(8, 16, 6), Point3D(1.0 / scale, 0.5 / scale, 0),
            Point3D(-0.5 / scale, 1.0 / scale, 0), Point3D(0, 0, 1)).to_dict()]
        anatomy = self._anatomy({
            "vertebrae_C2": self._annotation(image, (15, 4, 0)),
            "vertebrae_C3": self._annotation(image, (15, 16, 6)),
            "vertebrae_C4": self._annotation(image, (15, 4, 12)),
        })
        render = ReconstructionStage._build_plan_localizer_render(
            image, plan, anatomical_analysis=anatomy)
        alpha = render.line_alpha

        self.assertGreater(float(alpha[12 - 6, 8]), 0.5)
        self.assertGreater(float(alpha[12 - 4, 10]), 0.5)
        self.assertLess(float(alpha[12 - 4, 8]), 1e-6)
        # The Localizer line is clipped by the fixed background grid, so the
        # same oblique curve remains visible at both background edges.
        self.assertGreater(float(np.max(alpha[12 - 2])), 0.5)
        self.assertGreater(float(np.max(alpha[12 - 10])), 0.5)

    def test_mpr_renderer_without_filtered_geometry_preserves_fixed_layer_fallback(self):
        image = self._image()
        plan = self._plan(image, slice_type="MPR")
        render = ReconstructionStage._build_plan_localizer_render(image, plan)

        expected = sitk.GetArrayFromImage(image)[::-1, 16, :]
        np.testing.assert_array_equal(sitk.GetArrayFromImage(render.background_image), expected)
        self.assertEqual(render.background_kind, "orthogonal_plane")
        self.assertFalse(render.background_image.HasMetaDataKey("0028|135a"))

    def test_windowed_dicom_preserves_surface_derivation_and_reference_grid(self):
        """正式窗位写出后保留派生说明与参考网格，不使用顶层来源空间标签。"""

        image = self._image(transformed=True)
        plan = self._plan(image, slice_type="MPR")
        anatomy = self._anatomy({
            "vertebrae_C2": self._annotation(image, (15, 5, 2)),
            "vertebrae_C3": self._annotation(image, (15, 17, 6)),
            "vertebrae_C4": self._annotation(image, (15, 9, 10)),
        })
        window = WindowPreset("骨窗", 1500.0, 400.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            for following in (True, False):
                with self.subTest(following=following):
                    render = ReconstructionStage._build_plan_localizer_render(
                        image, plan, anatomical_analysis=anatomy if following else None)
                    frozen = ReconstructionStage._localizer_geometry_metadata(
                        render.background_image)
                    expected_position = [float(value) for value in frozen["0020|0032"].split("\\")]
                    expected_orientation = [float(value) for value in frozen["0020|0037"].split("\\")]
                    output_dir = Path(temp_dir) / str(following)
                    ReconstructionStage._write_window_localizer(
                        localizer_render=render,
                        geometry_metadata=frozen,
                        window=window,
                        output_dir=output_dir,
                        series_uid="1.2.826.0.1.3680043.2.1125.801",
                        study_uid="1.2.826.0.1.3680043.2.1125.802",
                        source_metadata={},
                        plan=plan,
                    )
                    dataset = pydicom.dcmread(output_dir / "Localizer.dcm")

                    np.testing.assert_allclose(dataset.ImagePositionPatient,
                                               expected_position, atol=1e-9, rtol=0)
                    np.testing.assert_allclose(dataset.ImageOrientationPatient,
                                               expected_orientation, atol=1e-9, rtol=0)
                    self.assertEqual(dataset.pixel_array.shape, render.line_alpha.shape)
                    self.assertNotIn("SpatialLocationsPreserved", dataset)
                    if following:
                        self.assertEqual(render.background_kind, "spine_following_coronal")
                        self.assertIn("Spine-following", dataset.DerivationDescription)
                    else:
                        self.assertEqual(render.background_kind, "orthogonal_plane")

    def test_execute_plan_localizer_spans_surface_independent_of_final_sampled_rectangles(self):
        """正式切片仍按各自FOV采样，定位图交线跨整幅固定曲面网格。"""

        image = self._image()
        plan = self._plan(image, slice_type="MPR")
        plan["slices"] = [PlannedSlice(
            index, Point3D(x, 25, 10), Point3D(1, 0, 0),
            Point3D(0, 1, 0), Point3D(0, 0, 1)).to_dict()
            for index, x in ((1, 8), (2, 20))]
        plan_before = copy.deepcopy(plan)
        anatomy = self._anatomy({
            "vertebrae_C2": self._annotation(image, (15, 5, 2)),
            "vertebrae_C3": self._annotation(image, (15, 17, 6)),
            "vertebrae_C4": self._annotation(image, (15, 9, 10)),
        })

        class FinalRectangleFramer:
            """提供两个明确的最终取景矩形，正式采样继续使用真实重采样器。"""

            def plane_geometry(self, current_plan, planned_slice):
                index = planned_slice["index"]
                extent = {"width": 8.0, "height": 4.0 if index == 1 else 2.0}
                center = {"x": planned_slice["center"]["x"], "y": 10.0,
                          "z": 4.0 if index == 1 else 8.0}
                framing = {"target_size_mm": [8.0, extent["height"]],
                           "context_label_ids": [], "scale": 1.0,
                           "center": center, "extent_mm": extent}
                return ({**current_plan, "extent_mm": extent},
                        {**planned_slice, "center": center, "framing": framing})

        stage = ReconstructionStage({"parallel": False})
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            stage, "_write_plan_windows",
        ) as write_windows:
            result = stage._execute_plan(
                image=image, plan=plan, output_root=Path(temp_dir),
                study_uid="1.2.826.0.1.3680043.2.1125.803", source_metadata={},
                target_framer=FinalRectangleFramer(), anatomical_analysis=anatomy)
            written = write_windows.call_args.kwargs

        sampled = written["sampled_slices"]
        self.assertEqual([sampled_image.GetSize() for _, sampled_image in sampled],
                         [(9, 5, 1), (9, 3, 1)])
        alpha = written["localizer_render"].line_alpha
        self.assertTrue(np.all(alpha[:, 8] > 0.5))
        self.assertTrue(np.all(alpha[:, 20] > 0.5))
        self.assertEqual(float(alpha[:, 7].max()), 0.0)
        self.assertEqual(float(alpha[:, 21].max()), 0.0)
        self.assertEqual(result["localizer_background"], "spine_following_coronal")
        self.assertEqual(plan, plan_before)

    def test_background_operations_do_not_modify_image_or_formal_slice_plan(self):
        image = self._image(transformed=True)
        plan = self._plan(image)
        plan_before = copy.deepcopy(plan)
        pixels_before = sitk.GetArrayFromImage(image)
        metadata_before = (image.GetSize(), image.GetOrigin(), image.GetSpacing(),
                           image.GetDirection(), image.GetPixelID())

        surface = build_spine_localizer_surface(image, plan)
        surface.sample(image)
        surface.plane_intersections(image, plan["slices"][0], plan["extent_mm"])

        self.assertEqual(plan, plan_before)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), pixels_before)
        self.assertEqual((image.GetSize(), image.GetOrigin(), image.GetSpacing(),
                          image.GetDirection(), image.GetPixelID()), metadata_before)


if __name__ == "__main__":
    unittest.main()
