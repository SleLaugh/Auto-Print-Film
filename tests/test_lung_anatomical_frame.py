import unittest

import numpy as np
import SimpleITK as sitk


class LungAnatomicalFrameTest(unittest.TestCase):
    """验证整肺主体过滤、中心范围计算与 DICOM 标准方向。"""

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
        self.assertEqual(diagnostics["angle_source"], "disabled")
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

    def test_direction_signs_follow_flipped_dicom_axes(self):
        """输入方向轴翻转后，输出轴符号应与 DICOM 方向保持一致。"""

        from src.anatomy.lung import estimate_lung_frame_with_diagnostics

        mask, image = self._make_sloped_lung(0.0)
        image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

        result, _ = estimate_lung_frame_with_diagnostics(mask, image)
        frame = result.global_frame

        self.assertLess(frame.left_right_axis.x, 0.0)
        self.assertLess(frame.anterior_posterior_axis.y, 0.0)
        self.assertGreater(frame.superior_inferior_axis.z, 0.0)

if __name__ == "__main__":
    unittest.main()
