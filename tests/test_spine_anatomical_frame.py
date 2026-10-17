import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk


class SpineAnatomicalFrameTest(unittest.TestCase):
    """验证保留前后倾角的脊柱对称坐标、质量回退和完整覆盖。"""

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

    def test_endpoints_are_diagnostic_and_missing_symmetry_keeps_reference_axes(self):
        """无可靠左右证据时，斜向首尾连线不能改变输入前后倾角。"""

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
        raw_endpoint_axis = c7 - c1
        raw_endpoint_axis /= np.linalg.norm(raw_endpoint_axis)
        np.testing.assert_allclose(diagnostics["raw_spine_axis"], raw_endpoint_axis)
        self.assertEqual(diagnostics["endpoint_axis_role"], "diagnostic_only")
        actual = np.asarray(list(frame.superior_inferior_axis.to_dict().values()))
        np.testing.assert_allclose(actual, [0.0, 0.0, 1.0])
        self.assertGreater(abs(float(raw_endpoint_axis[1])), 0.1)
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
        self.assertEqual(frame.method, "spine_dicom_axes")
        self.assertFalse(frame.applied)
        self.assertEqual(frame.confidence, 0.0)
        self.assertTrue(any("左右对称证据不足" in message for message in frame.warnings))
        self.assertEqual(diagnostics["used_labels"], ["C1", "C4", "C7"])
        self.assertEqual(diagnostics["missing_labels"], [])

    def test_missing_middle_label_warns_and_preserves_pitch(self):
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

    def test_storage_flips_preserve_patient_axis_signs(self):
        """存储轴反向后脊柱三轴仍应朝患者左、后、头侧。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        masks, image = self._make_masks()
        image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

        result, _ = estimate_spine_frame_with_diagnostics(masks, "C1", "C7", image)
        frame = result.global_frame

        lr = np.asarray(list(frame.left_right_axis.to_dict().values()))
        ap = np.asarray(list(frame.anterior_posterior_axis.to_dict().values()))
        si = np.asarray(list(frame.superior_inferior_axis.to_dict().values()))
        self.assertGreater(lr[0], 0.0)
        self.assertGreater(ap[1], 0.0)
        self.assertGreater(si[2], 0.0)

    def test_indices_path_matches_masks_without_dense_mask_or_argwhere(self):
        """复用稀疏坐标时不读取 Mask，所有几何与原入口完全一致。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        masks, image = self._make_masks()
        masks["C4"][1, 1, 1] = True
        theta = np.deg2rad(23.0)
        image.SetDirection((np.cos(theta), -np.sin(theta), 0.,
                            np.sin(theta), np.cos(theta), 0., 0., 0., 1.))
        indices = {name: np.argwhere(mask) for name, mask in masks.items()}
        annotations = {name: {
            "symmetry_left_right_axis": {"x": 0.9, "y": 0.4, "z": 0.1},
            "symmetry_diagnostics": {"status": "accepted"},
        } for name in masks}
        expected, expected_diagnostics = estimate_spine_frame_with_diagnostics(
            masks, "C1", "C7", image, annotations,
        )
        with patch("numpy.argwhere", side_effect=AssertionError("不能重复提取体素坐标")):
            result, diagnostics = estimate_spine_frame_with_diagnostics(
                {"must_not_read": object()}, "C1", "C7", image, annotations,
                label_indices_zyx=indices,
            )
        self.assertEqual(result.to_dict(), expected.to_dict())
        self.assertEqual(diagnostics, expected_diagnostics)
        all_indices = np.concatenate(list(indices.values()))[:, ::-1]
        points = (all_indices * image.GetSpacing()) @ np.array(image.GetDirection()).reshape(3, 3).T
        points += image.GetOrigin()
        frame = result.global_frame
        for name, axis in (
            ("left_right", frame.left_right_axis),
            ("anterior_posterior", frame.anterior_posterior_axis),
            ("superior_inferior", frame.superior_inferior_axis),
        ):
            projections = points @ np.array(list(axis.to_dict().values()))
            bounds = getattr(frame.bounds, name)
            self.assertAlmostEqual(bounds.minimum, projections.min())
            self.assertAlmostEqual(bounds.maximum, projections.max())

    def test_indices_empty_labels_and_invalid_coordinates_are_checked(self):
        """空标签仍进入缺失诊断；形状、非有限数和越界不能静默参与几何。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        masks, image = self._make_masks()
        indices = {name: np.argwhere(mask) for name, mask in masks.items()}
        indices["C4"] = np.empty((0, 3), dtype=int)
        _, diagnostics = estimate_spine_frame_with_diagnostics(
            None, "C1", "C7", image, label_indices_zyx=indices,
        )
        self.assertEqual(diagnostics["missing_labels"], ["C4"])
        for invalid in (np.empty((0,)), np.ones((2, 2)), [[1, 2, float("nan")]],
                        [[1, 2, float("inf")]], [[-1, 2, 3]], [[1, 2, 48]],
                        [["bad", "coordinates", "here"]]):
            with self.subTest(indices=invalid):
                with self.assertRaisesRegex(ValueError, "椎体 C4"):
                    estimate_spine_frame_with_diagnostics(
                        None, "C1", "C7", image,
                        label_indices_zyx={**indices, "C4": invalid},
                    )
        with self.assertRaisesRegex(ValueError, "开始椎体 C1"):
            estimate_spine_frame_with_diagnostics(
                masks, "C1", "C7", image, label_indices_zyx={},
            )

    def test_single_storage_axis_flip_keeps_right_handed_frame(self):
        """单轴反向存储不能倒置患者头脚或破坏右手关系。"""

        from src.anatomy.spine import estimate_spine_frame_with_diagnostics

        for diagonal in ((-1., 1., 1.), (1., -1., 1.), (1., 1., -1.)):
            with self.subTest(diagonal=diagonal):
                masks, image = self._make_masks()
                direction = np.diag(diagonal)
                image.SetDirection(tuple(direction.ravel()))
                result, _ = estimate_spine_frame_with_diagnostics(
                    None, "C1", "C7", image,
                    label_indices_zyx={name: np.argwhere(mask) for name, mask in masks.items()},
                )
                frame = result.global_frame
                axes = np.column_stack([list(axis.to_dict().values()) for axis in (
                    frame.left_right_axis, frame.anterior_posterior_axis,
                    frame.superior_inferior_axis,
                )])
                np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-6)
                self.assertAlmostEqual(np.linalg.det(axes), 1.0)
                self.assertGreater(axes[0, 0], 0.0)
                self.assertGreater(axes[1, 1], 0.0)
                self.assertGreater(axes[2, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
