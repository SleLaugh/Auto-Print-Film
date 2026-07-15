import unittest

import numpy as np
import SimpleITK as sitk


class SpineAnatomicalFrameTest(unittest.TestCase):
    """验证首尾椎体质心确定的三维脊柱坐标系。"""

    @staticmethod
    def _make_masks() -> tuple[dict[str, np.ndarray], sitk.Image]:
        """生成同时具有左右、前后和头脚倾斜的三个椎体 Mask。"""

        shape = (48, 40, 48)
        masks = {name: np.zeros(shape, dtype=bool) for name in ("C1", "C4", "C7")}
        masks["C1"][36:40, 8:12, 10:14] = True
        masks["C4"][22:26, 16:20, 20:24] = True
        masks["C7"][8:12, 24:28, 30:34] = True
        image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
        image.SetSpacing((1.0, 1.5, 2.0))
        image.SetOrigin((10.0, 20.0, 30.0))
        return masks, image

    def test_endpoint_centroids_build_complete_orthonormal_frame_and_bounds(self):
        """首尾质心连线应保留 XYZ 分量，全部 Mask 应决定中心和范围。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        masks, image = self._make_masks()
        result, diagnostics = estimate_spine_frame_with_diagnostics(
            masks,
            start_label="C1",
            end_label="C7",
            image=image,
        )
        frame = result.global_frame

        c1 = np.asarray(diagnostics["start_center"], dtype=float)
        c7 = np.asarray(diagnostics["end_center"], dtype=float)
        expected = c7 - c1
        expected /= np.linalg.norm(expected)
        dicom_si = np.asarray(image.GetDirection()).reshape(3, 3)[:, 2]
        if float(np.dot(expected, dicom_si)) < 0.0:
            expected = -expected
        actual = np.asarray(list(frame.superior_inferior_axis.to_dict().values()))
        self.assertTrue(np.allclose(actual, expected))
        axes = [
            np.asarray(list(frame.left_right_axis.to_dict().values())),
            np.asarray(list(frame.anterior_posterior_axis.to_dict().values())),
            actual,
        ]
        for axis in axes:
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot(axes[0], axes[1])), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(axes[0], axes[2])), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(axes[1], axes[2])), 0.0, places=6)
        all_indices = np.concatenate([np.argwhere(mask) for mask in masks.values()])
        expected_index = np.mean(all_indices[:, [2, 1, 0]], axis=0)
        expected_center = image.TransformContinuousIndexToPhysicalPoint(
            tuple(float(value) for value in expected_index)
        )
        self.assertTrue(np.allclose(
            list(frame.origin.to_dict().values()),
            expected_center,
        ))
        self.assertEqual(frame.method, "spine_endpoint_centroids")
        self.assertTrue(frame.applied)
        self.assertEqual(diagnostics["used_labels"], ["C1", "C4", "C7"])
        self.assertEqual(diagnostics["missing_labels"], [])

    def test_missing_middle_label_warns_but_keeps_endpoint_direction(self):
        """首尾存在时允许中间椎体缺失，并在诊断和 frame 警告中记录。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        masks, image = self._make_masks()
        masks["C4"][:] = False

        result, diagnostics = estimate_spine_frame_with_diagnostics(
            masks,
            start_label="C1",
            end_label="C7",
            image=image,
        )

        self.assertEqual(diagnostics["missing_labels"], ["C4"])
        self.assertEqual(diagnostics["used_labels"], ["C1", "C7"])
        self.assertTrue(result.global_frame.warnings)

    def test_missing_endpoint_or_coincident_centroids_are_rejected(self):
        """首尾标签缺失或质心重合时不能静默回退到默认方向。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        masks, image = self._make_masks()
        masks["C1"][:] = False
        with self.assertRaisesRegex(ValueError, "开始椎体 C1"):
            estimate_spine_frame_with_diagnostics(masks, "C1", "C7", image)

        masks, image = self._make_masks()
        masks["C7"] = masks["C1"].copy()
        with self.assertRaisesRegex(ValueError, "质心距离"):
            estimate_spine_frame_with_diagnostics(masks, "C1", "C7", image)

    def test_axis_signs_follow_flipped_dicom_patient_directions(self):
        """输入方向翻转后脊柱三轴仍应与 DICOM 患者方向保持稳定符号。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        masks, image = self._make_masks()
        image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

        result, _ = estimate_spine_frame_with_diagnostics(masks, "C1", "C7", image)
        frame = result.global_frame

        dicom = np.asarray(image.GetDirection()).reshape(3, 3)
        lr = np.asarray(list(frame.left_right_axis.to_dict().values()))
        ap = np.asarray(list(frame.anterior_posterior_axis.to_dict().values()))
        si = np.asarray(list(frame.superior_inferior_axis.to_dict().values()))
        self.assertGreater(float(np.dot(lr, dicom[:, 0])), 0.0)
        self.assertGreater(float(np.dot(ap, dicom[:, 1])), 0.0)
        self.assertGreater(float(np.dot(si, dicom[:, 2])), 0.0)


if __name__ == "__main__":
    unittest.main()
