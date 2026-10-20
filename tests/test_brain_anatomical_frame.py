"""验证颅脑左右对称面、关闭形态俯仰的兼容路径及完整体素支撑覆盖。"""

import unittest

import numpy as np
import SimpleITK as sitk

from src.anatomy.brain import estimate_brain_frame_with_diagnostics
from tests.test_symmetry_plane import _rotation_yz


class BrainAnatomicalFrameTest(unittest.TestCase):
    def _make_target(self):
        """在各向异性、斜向物理图像中生成已知姿态的颅脑样椭球壳。"""

        shape = (62, 94, 100)
        image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
        image.SetSpacing((1.0, 1.2, 1.6))
        image.SetOrigin((45.0, -80.0, 12.0))
        reference = _rotation_yz(12.0, -20.0)
        image.SetDirection(tuple(reference.ravel()))
        pose = _rotation_yz(-9.0, 6.0)
        indices_xyz = np.indices(shape).reshape(3, -1).T[:, ::-1]
        local = indices_xyz * np.asarray(image.GetSpacing())
        center = (np.array(shape[::-1]) - 1.0) * np.asarray(image.GetSpacing()) / 2
        anatomical = (local - center) @ pose
        radius = np.sum((anatomical / [30.0, 42.0, 45.0]) ** 2, axis=1)
        mask = ((radius <= 1.0) & (radius >= 0.78)).reshape(shape)
        return image, mask, reference, pose

    def test_recovers_side_tilt_and_bounds_cover_every_target_voxel(self):
        image, mask, reference, pose = self._make_target()
        result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image, pitch_enabled=False)
        frame = result.global_frame
        axes = np.column_stack([
            list(frame.left_right_axis.to_dict().values()),
            list(frame.anterior_posterior_axis.to_dict().values()),
            list(frame.superior_inferior_axis.to_dict().values()),
        ])
        self.assertTrue(frame.applied, diagnostics)
        self.assertGreater(float(np.dot(axes[:, 0], reference @ pose[:, 0])), np.cos(np.radians(1.0)))
        expected_ap = reference[:, 1] - float(np.dot(reference[:, 1], axes[:, 0])) * axes[:, 0]
        expected_ap /= np.linalg.norm(expected_ap)
        np.testing.assert_allclose(axes[:, 1], expected_ap, atol=1e-12)
        indices_xyz = np.argwhere(mask)[:, ::-1]
        points = np.asarray(image.GetOrigin()) + (indices_xyz * np.asarray(image.GetSpacing())) @ reference.T
        projection = points @ axes
        support = 0.5 * np.sum(np.abs(reference.T @ axes) * np.asarray(image.GetSpacing())[:, None], axis=0)
        for index, bound in enumerate((frame.bounds.left_right, frame.bounds.anterior_posterior, frame.bounds.superior_inferior)):
            self.assertAlmostEqual(bound.minimum, float(np.min(projection[:, index]) - support[index]))
            self.assertAlmostEqual(bound.maximum, float(np.max(projection[:, index]) + support[index]))
        np.testing.assert_allclose(list(frame.origin.to_dict().values()), np.mean(points, axis=0), atol=1e-10)
        self.assertEqual(diagnostics["target_point_count"], int(np.count_nonzero(mask)))
        self.assertLessEqual(diagnostics["symmetry"]["sample_count"], 6000)
        self.assertEqual(diagnostics["pitch"]["status"], "disabled")
        self.assertEqual(diagnostics["pitch_policy"], "project_input_ap_onto_symmetry_plane")

    def test_reflected_storage_keeps_physical_points_and_patient_head_direction(self):
        """反向存储不能改物理点，最终头侧必须沿患者+Z而非存储叉积。"""

        image, mask, oblique, pose = self._make_target()
        for flips in ((-1.0, 1.0, 1.0), (1.0, 1.0, -1.0)):
            with self.subTest(flips=flips):
                raw_direction = oblique @ np.diag(flips)
                image.SetDirection(tuple(raw_direction.ravel()))
                result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image, pitch_enabled=False)
                frame = result.global_frame
                axes = np.column_stack([
                    list(frame.left_right_axis.to_dict().values()),
                    list(frame.anterior_posterior_axis.to_dict().values()),
                    list(frame.superior_inferior_axis.to_dict().values()),
                ])
                self.assertTrue(frame.applied, diagnostics)
                self.assertGreater(abs(float(np.dot(axes[:, 0], raw_direction @ pose[:, 0]))), np.cos(np.radians(1.0)))
                self.assertGreater(axes[0, 0], 0.0)
                self.assertGreater(axes[1, 1], 0.0)
                self.assertGreater(axes[2, 2], 0.0)
                np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-10)
                self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0)
                indices_xyz = np.argwhere(mask)[:, ::-1]
                points = np.asarray(image.GetOrigin()) + (indices_xyz * np.asarray(image.GetSpacing())) @ raw_direction.T
                np.testing.assert_allclose(list(frame.origin.to_dict().values()), np.mean(points, axis=0), atol=1e-10)
                projection = points @ axes
                support = 0.5 * np.sum(np.abs(raw_direction.T @ axes) * np.asarray(image.GetSpacing())[:, None], axis=0)
                for index, bound in enumerate((frame.bounds.left_right, frame.bounds.anterior_posterior, frame.bounds.superior_inferior)):
                    self.assertAlmostEqual(bound.minimum, float(np.min(projection[:, index]) - support[index]))
                    self.assertAlmostEqual(bound.maximum, float(np.max(projection[:, index]) + support[index]))

    def test_unreliable_target_keeps_dicom_axes_and_explains_why(self):
        mask = np.zeros((8, 12, 14), dtype=bool)
        mask[4, 4:8, 5:9] = True
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        result, diagnostics = estimate_brain_frame_with_diagnostics(mask, image)
        frame = result.global_frame
        self.assertFalse(frame.applied)
        self.assertTrue(frame.warnings)
        np.testing.assert_array_equal(list(frame.left_right_axis.to_dict().values()), [1.0, 0.0, 0.0])
        self.assertEqual(diagnostics["symmetry"]["status"], "rejected")

    def test_empty_or_mismatched_target_fails(self):
        image = sitk.Image(12, 14, 16, sitk.sitkUInt8)
        for mask in (np.zeros((16, 14, 12), dtype=bool), np.ones((14, 12), dtype=bool)):
            with self.assertRaises(ValueError):
                estimate_brain_frame_with_diagnostics(mask, image)


if __name__ == "__main__":
    unittest.main()
