import unittest

import numpy as np
import SimpleITK as sitk


class LungAnatomicalFrameTest(unittest.TestCase):
    """验证肺主体过滤、左右姿态证据和完整覆盖范围。"""

    @staticmethod
    def _make_sloped_lung(slope: float = 0.2) -> tuple[np.ndarray, sitk.Image]:
        """生成具有可控肺顶斜率的三维整肺测试 Mask。"""

        array = np.zeros((64, 32, 64), dtype=np.uint8)
        for x in range(8, 56):
            # 图像方向为单位矩阵，物理 +Z 指向头侧，因此较大的 z 才是肺顶。
            superior = int(round(38 + slope * (x - 8)))
            array[8:superior + 1, 6:26, x] = 1
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.5, 2.0))
        image.SetOrigin((10.0, 20.0, 30.0))
        return array.astype(bool), image

    def test_sloped_lung_keeps_dicom_axes_but_uses_lung_center_and_bounds(self):
        """肺形态倾斜不应改变切片方向，但肺主体仍应决定中线和范围。"""

        from src.anatomy.lung import estimate_lung_frame_with_diagnostics

        mask, image = self._make_sloped_lung(0.2)
        result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)
        frame = result.global_frame

        self.assertEqual(frame.left_right_axis.to_dict(), {"x": 1.0, "y": 0.0, "z": 0.0})
        self.assertEqual(frame.anterior_posterior_axis.to_dict(), {"x": 0.0, "y": 1.0, "z": 0.0})
        self.assertEqual(frame.superior_inferior_axis.to_dict(), {"x": 0.0, "y": 0.0, "z": 1.0})
        self.assertFalse(frame.applied)
        self.assertEqual(diagnostics["angle_source"], "bilateral_reflection_symmetry")
        self.assertIn("fallback_reason", diagnostics)
        indices_zyx = np.argwhere(mask)
        expected_index_xyz = np.mean(indices_zyx[:, [2, 1, 0]], axis=0)
        expected_center = image.TransformContinuousIndexToPhysicalPoint(
            tuple(float(value) for value in expected_index_xyz)
        )
        self.assertAlmostEqual(frame.origin.x, expected_center[0], places=5)
        self.assertAlmostEqual(frame.origin.y, expected_center[1], places=5)
        self.assertAlmostEqual(frame.origin.z, expected_center[2], places=5)
        self.assertGreater(
            frame.bounds.superior_inferior.maximum,
            frame.bounds.superior_inferior.minimum,
        )

    def test_bilateral_nonspherical_lungs_recover_yaw_roll_and_cover_full_target(self):
        """各向异性斜向输入中，双侧非球形主体恢复联合姿态并完整覆盖全部体素。"""

        from src.anatomy.lung import estimate_lung_frame_with_diagnostics

        def rotation_yz(y_degrees, z_degrees):
            y, z = np.radians([y_degrees, z_degrees])
            ry = np.array([
                [np.cos(y), 0.0, np.sin(y)],
                [0.0, 1.0, 0.0],
                [-np.sin(y), 0.0, np.cos(y)],
            ])
            rz = np.array([
                [np.cos(z), -np.sin(z), 0.0],
                [np.sin(z), np.cos(z), 0.0],
                [0.0, 0.0, 1.0],
            ])
            return rz @ ry

        shape = (80, 100, 128)
        image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
        image.SetSpacing((1.2, 1.1, 1.5))
        image.SetOrigin((40.0, -90.0, 15.0))
        reference = rotation_yz(13.0, 27.0)
        image.SetDirection(tuple(reference.ravel()))
        pose = rotation_yz(-9.0, -8.0)
        spacing = np.asarray(image.GetSpacing())
        grid_xyz = np.indices(shape).reshape(3, -1).T[:, ::-1]
        local = grid_xyz * spacing
        center = (np.array(shape[::-1]) - 1.0) * spacing / 2.0
        anatomical = (local - center) @ pose
        flat_mask = np.zeros(len(anatomical), dtype=bool)
        # 两个独立椭球具备共同左右对称面；AP/SI 轴长不同，不能用球形歧义过关。
        for side_x in (-34.0, 34.0):
            flat_mask |= np.sum(
                ((anatomical - [side_x, 0.0, 0.0]) / [21.0, 31.0, 43.0]) ** 2,
                axis=1,
            ) <= 1.0
        mask = flat_mask.reshape(shape)

        result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)
        frame = result.global_frame
        axes = np.column_stack([
            list(frame.left_right_axis.to_dict().values()),
            list(frame.anterior_posterior_axis.to_dict().values()),
            list(frame.superior_inferior_axis.to_dict().values()),
        ])
        self.assertEqual(diagnostics["retained_component_count"], 2)
        self.assertTrue(frame.applied, diagnostics)
        self.assertTrue(diagnostics["symmetry"]["accepted"])
        expected_lr = reference @ pose[:, 0]
        self.assertGreater(float(np.dot(axes[:, 0], expected_lr)), np.cos(np.radians(0.8)))
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0)
        expected_ap = reference[:, 1] - float(np.dot(reference[:, 1], axes[:, 0])) * axes[:, 0]
        expected_ap /= np.linalg.norm(expected_ap)
        np.testing.assert_allclose(axes[:, 1], expected_ap, atol=1e-10)
        points = np.asarray(image.GetOrigin()) + local[flat_mask] @ reference.T
        projection = points @ axes
        for index, bound in enumerate((
            frame.bounds.left_right,
            frame.bounds.anterior_posterior,
            frame.bounds.superior_inferior,
        )):
            self.assertAlmostEqual(bound.minimum, float(np.min(projection[:, index])))
            self.assertAlmostEqual(bound.maximum, float(np.max(projection[:, index])))
        np.testing.assert_allclose(
            list(frame.origin.to_dict().values()), np.mean(points, axis=0), atol=1e-10,
        )
        self.assertEqual(diagnostics["point_count"], int(np.count_nonzero(mask)))

    def test_nearly_spherical_target_with_tiny_second_blob_has_no_bilateral_evidence(self):
        """单个近球形主体加小误分割块不能伪造双肺姿态证据。"""

        from src.anatomy.lung import estimate_lung_frame_with_diagnostics

        shape = (48, 48, 48)
        z, y, x = np.indices(shape)
        mask = ((x - 23.0) / 12.0) ** 2 + ((y - 24.0) / 12.5) ** 2 + ((z - 22.0) / 12.0) ** 2 <= 1.0
        mask[42:44, 42:44, 42:44] = True
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)

        self.assertEqual(diagnostics["retained_component_count"], 1)
        self.assertEqual(diagnostics["discarded_component_count"], 1)
        self.assertFalse(result.global_frame.applied)
        self.assertTrue(result.global_frame.warnings)
        self.assertIn("fallback_reason", diagnostics)
        self.assertNotIn("symmetry", diagnostics)
        self.assertEqual(result.global_frame.left_right_axis.to_dict(), {"x": 1.0, "y": 0.0, "z": 0.0})
        self.assertEqual(diagnostics["correction_degrees"], 0.0)

    def test_remote_false_positive_component_does_not_expand_lung_range(self):
        """远离双肺的小误分割块不应把横状位规划范围扩展到全身。"""

        from src.anatomy.lung import estimate_lung_frame_with_diagnostics

        mask, image = self._make_sloped_lung(0.2)
        baseline, _ = estimate_lung_frame_with_diagnostics(mask, image)
        contaminated = mask.copy()
        contaminated[60:64, 10:22, 20:44] = True

        actual, diagnostics = estimate_lung_frame_with_diagnostics(contaminated, image)

        expected = baseline.global_frame
        frame = actual.global_frame
        self.assertAlmostEqual(
            frame.bounds.superior_inferior.minimum,
            expected.bounds.superior_inferior.minimum,
            places=5,
        )
        self.assertAlmostEqual(
            frame.bounds.superior_inferior.maximum,
            expected.bounds.superior_inferior.maximum,
            places=5,
        )
        self.assertGreater(diagnostics["discarded_component_count"], 0)
        self.assertGreater(diagnostics["discarded_point_count"], 0)

    def test_degenerate_mask_falls_back_to_dicom_axes_and_keeps_center(self):
        """目标点不足时应保留中心并安全回退到 DICOM 标准轴。"""

        from src.anatomy.lung import estimate_lung_frame_with_diagnostics

        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[3, 4, 5] = True
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        image.SetSpacing((2.0, 3.0, 4.0))
        image.SetOrigin((10.0, 20.0, 30.0))

        result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)
        frame = result.global_frame

        self.assertFalse(frame.applied)
        self.assertEqual(frame.method, "dicom_axes_fallback")
        self.assertEqual(frame.left_right_axis.to_dict(), {"x": 1.0, "y": 0.0, "z": 0.0})
        self.assertEqual(frame.anterior_posterior_axis.to_dict(), {"x": 0.0, "y": 1.0, "z": 0.0})
        self.assertEqual(frame.superior_inferior_axis.to_dict(), {"x": 0.0, "y": 0.0, "z": 1.0})
        self.assertEqual(frame.origin.to_dict(), {"x": 20.0, "y": 32.0, "z": 42.0})
        self.assertTrue(frame.warnings)
        self.assertIn("fallback_reason", diagnostics)

    def test_storage_flips_preserve_patient_axis_signs(self):
        """输入存储轴翻转后，输出轴仍应保持患者LPS朝向。"""

        from src.anatomy.lung import estimate_lung_frame_with_diagnostics

        mask, image = self._make_sloped_lung(0.0)
        image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

        result, _ = estimate_lung_frame_with_diagnostics(mask, image)
        frame = result.global_frame

        self.assertGreater(frame.left_right_axis.x, 0.0)
        self.assertGreater(frame.anterior_posterior_axis.y, 0.0)
        self.assertGreater(frame.superior_inferior_axis.z, 0.0)

if __name__ == "__main__":
    unittest.main()
