import unittest

import numpy as np
import SimpleITK as sitk

from src.anatomy.hip import estimate_hip_frame_with_diagnostics


class HipAnatomicalFrameTest(unittest.TestCase):
    """验证髋骨外侧顶部连线生成的横状位解剖坐标。"""

    def test_outer_superior_caps_define_tilt_and_ignore_isolated_spike(self):
        """左右外侧顶部不等高时应倾斜，远端单点不能带偏起始面。"""

        shape = (40, 24, 48)
        left_mask = np.zeros(shape, dtype=bool)
        right_mask = np.zeros(shape, dtype=bool)
        # z/y/x：在本图 DICOM SI=+z，患者左侧主体更高，右侧略低。
        left_mask[10:34, 7:17, 7:18] = True
        right_mask[8:30, 7:17, 30:41] = True
        # 孤立误分割点比主体顶部更高，必须被最大连通域过滤掉。
        right_mask[38, 3, 44] = True
        image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))

        result, diagnostics = estimate_hip_frame_with_diagnostics(
            left_mask=left_mask,
            right_mask=right_mask,
            image=image,
        )

        frame = result.global_frame
        lr = np.asarray(list(frame.left_right_axis.to_dict().values()))
        ap = np.asarray(list(frame.anterior_posterior_axis.to_dict().values()))
        si = np.asarray(list(frame.superior_inferior_axis.to_dict().values()))
        self.assertEqual(frame.method, "hip_outer_superior_caps")
        self.assertTrue(frame.applied)
        self.assertGreater(abs(float(lr[2])), 0.05)
        self.assertTrue(np.allclose(np.cross(lr, ap), si))
        self.assertAlmostEqual(float(np.dot(lr, ap)), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(si)), 1.0, places=6)
        self.assertLess(diagnostics["right_landmark"]["z"], 31.0)
        self.assertGreater(diagnostics["left_landmark"]["z"], diagnostics["right_landmark"]["z"])
        landmark_plane = np.mean([
            np.dot(list(diagnostics["left_landmark"].values()), si),
            np.dot(list(diagnostics["right_landmark"].values()), si),
        ])
        self.assertAlmostEqual(
            frame.bounds.superior_inferior.maximum,
            landmark_plane,
            places=6,
        )
        self.assertEqual(diagnostics["right_discarded_point_count"], 1)

    def test_missing_one_side_is_rejected(self):
        """缺少任一侧髋骨时不能生成看似正常但实际错误的切片角度。"""

        left_mask = np.ones((8, 8, 8), dtype=bool)
        right_mask = np.zeros_like(left_mask)
        image = sitk.GetImageFromArray(np.zeros_like(left_mask, dtype=np.uint8))

        with self.assertRaisesRegex(ValueError, "右侧髋骨"):
            estimate_hip_frame_with_diagnostics(left_mask, right_mask, image)

    def test_direction_signs_follow_flipped_dicom_axes(self):
        """存储方向翻转时输出三轴仍应与该 DICOM 物理方向保持同向。"""

        shape = (24, 16, 32)
        left_mask = np.zeros(shape, dtype=bool)
        right_mask = np.zeros(shape, dtype=bool)
        left_mask[5:20, 4:12, 4:12] = True
        right_mask[7:22, 4:12, 20:28] = True
        image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
        image.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))

        result, _ = estimate_hip_frame_with_diagnostics(
            left_mask, right_mask, image
        )

        frame = result.global_frame
        direction = np.asarray(image.GetDirection()).reshape(3, 3)
        self.assertGreater(np.dot(list(frame.left_right_axis.to_dict().values()), direction[:, 0]), 0.0)
        self.assertGreater(np.dot(list(frame.anterior_posterior_axis.to_dict().values()), direction[:, 1]), 0.0)
        self.assertGreater(np.dot(list(frame.superior_inferior_axis.to_dict().values()), direction[:, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
