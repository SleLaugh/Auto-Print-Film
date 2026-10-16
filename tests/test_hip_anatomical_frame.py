import unittest

import numpy as np
import SimpleITK as sitk

from src.anatomy.hip import estimate_hip_frame_with_diagnostics


class HipAnatomicalFrameTest(unittest.TestCase):
    """验证双髋骨姿态坐标、顶部诊断和完整主体覆盖。"""

    def test_symmetry_defines_frame_and_bounds_cover_all_retained_points(self):
        """双侧主体决定方向，AB 仅供诊断，全部三轴范围都覆盖完整主体。"""

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
        # 盒子具有不同高度，不给它们虚构一个临床真实姿态。本例验证顶部
        # 诊断、主体过滤和投影覆盖；已知旋转的回归由 test_hip_pose_symmetry 验证。
        self.assertIn(frame.method, {"hip_reflection_symmetry", "hip_dicom_axes"})
        self.assertEqual(frame.applied, diagnostics["symmetry"]["accepted"])
        self.assertTrue(np.allclose(np.cross(lr, ap), si))
        self.assertAlmostEqual(float(np.dot(lr, ap)), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(si)), 1.0, places=6)
        self.assertLess(diagnostics["right_landmark"]["z"], 31.0)
        self.assertGreater(diagnostics["left_landmark"]["z"], diagnostics["right_landmark"]["z"])

        self.assertEqual(diagnostics["ab_role"], "diagnostic_only")
        self.assertIn(diagnostics["symmetry"]["status"], {"accepted", "unchanged"})
        # 顶部仍保留在诊断中，但不再要求输出方向等于顶部连线；两侧高度或
        # 形态略有差异时，单个顶部标志点不能替代整块骨盆的对称姿态。
        left_landmark = np.asarray([
            diagnostics["left_landmark"][axis] for axis in ("x", "y", "z")
        ])
        right_landmark = np.asarray([
            diagnostics["right_landmark"][axis] for axis in ("x", "y", "z")
        ])
        # 当前测试图为空间单位阵，z/y/x 索引换成 x/y/z 后就是物理坐标。
        # 已知远端孤立点不属于最大连通主体，不能进入最终覆盖范围。
        retained_mask = np.logical_or(left_mask, right_mask)
        retained_mask[38, 3, 44] = False
        retained_points = np.argwhere(retained_mask)[:, [2, 1, 0]].astype(float)
        retained_projections = retained_points @ si
        self.assertAlmostEqual(
            frame.bounds.superior_inferior.minimum,
            float(np.min(retained_projections)),
            places=6,
        )
        self.assertAlmostEqual(
            frame.bounds.superior_inferior.maximum,
            float(np.max(retained_projections)),
            places=6,
        )
        self.assertTrue(np.all(
            retained_projections >= frame.bounds.superior_inferior.minimum - 1e-6
        ))
        self.assertTrue(np.all(
            retained_projections <= frame.bounds.superior_inferior.maximum + 1e-6
        ))
        for axis, bounds in (
            (lr, frame.bounds.left_right),
            (ap, frame.bounds.anterior_posterior),
        ):
            projection = retained_points @ axis
            self.assertAlmostEqual(bounds.minimum, float(np.min(projection)), places=6)
            self.assertAlmostEqual(bounds.maximum, float(np.max(projection)), places=6)

        ab_plane_projection = float(np.mean([
            np.dot(left_landmark, si),
            np.dot(right_landmark, si),
        ]))
        self.assertGreater(
            frame.bounds.superior_inferior.maximum,
            ab_plane_projection,
        )
        self.assertAlmostEqual(
            diagnostics["ab_plane_projection_mm"],
            ab_plane_projection,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["coverage_superior_projection_mm"],
            frame.bounds.superior_inferior.maximum,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["coverage_inferior_projection_mm"],
            frame.bounds.superior_inferior.minimum,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["superior_shift_from_ab_mm"],
            frame.bounds.superior_inferior.maximum - ab_plane_projection,
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

    def test_storage_flips_preserve_patient_axis_signs(self):
        """存储方向翻转不能把患者头侧或左右参考方向反转。"""

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
        self.assertGreater(frame.left_right_axis.x, 0.0)
        self.assertGreater(frame.anterior_posterior_axis.y, 0.0)
        self.assertGreater(frame.superior_inferior_axis.z, 0.0)


if __name__ == "__main__":
    unittest.main()
