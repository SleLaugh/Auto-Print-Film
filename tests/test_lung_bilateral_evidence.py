"""两个连通域必须提供跨左右主体的镜像证据，不能仅凭个数接受姿态。"""

import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.anatomy.lung import estimate_lung_frame_with_diagnostics


class LungBilateralEvidenceTest(unittest.TestCase):
    def test_two_vertically_separated_same_side_lobes_are_not_bilateral_lungs(self):
        """同侧上下两块各自具有 12° 对称面，也不能伪装成可靠双肺。"""

        shape = (100, 90, 120)
        z, y, x = np.indices(shape)
        theta = np.radians(12.0)
        local_x = np.cos(theta) * (x - 80.0) + np.sin(theta) * (y - 45.0)
        local_y = -np.sin(theta) * (x - 80.0) + np.cos(theta) * (y - 45.0)
        mask = np.zeros(shape, dtype=bool)
        for center_z in (23.0, 77.0):
            mask |= (
                (local_x / 14.0) ** 2
                + (local_y / 23.0) ** 2
                + ((z - center_z) / 16.0) ** 2
                <= 1.0
            )
        image = sitk.GetImageFromArray(mask.astype(np.uint8))

        result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)

        self.assertEqual(diagnostics["retained_component_count"], 2)
        self.assertEqual(diagnostics["symmetry"]["source_status"], "accepted")
        self.assertEqual(diagnostics["symmetry"]["status"], "rejected")
        self.assertFalse(result.global_frame.applied)
        self.assertEqual(result.global_frame.confidence, 0.0)
        self.assertFalse(diagnostics["bilateral_evidence"]["reliable"])
        self.assertIn("分离不足", diagnostics["fallback_reason"])
        self.assertTrue(result.global_frame.warnings)
        np.testing.assert_allclose(
            list(result.global_frame.left_right_axis.to_dict().values()),
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(diagnostics["correction_degrees"], 0.0)

    @staticmethod
    def _make_bilateral_pair(*, yaw=12.0, roll=7.0, unequal_sizes=True):
        """构造中心分居左右的两肺，允许两侧自然大小不同。"""

        shape = (110, 100, 140)
        xyz = np.indices(shape).reshape(3, -1).T[:, ::-1].astype(float)
        yaw, roll = np.radians([yaw, roll])
        rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        ry = np.array([
            [np.cos(roll), 0.0, -np.sin(roll)],
            [0.0, 1.0, 0.0],
            [np.sin(roll), 0.0, np.cos(roll)],
        ])
        pose = rz @ ry
        local = (xyz - [70.0, 50.0, 55.0]) @ pose
        mask = np.zeros(len(local), dtype=bool)
        for center, radii in (
            ([-34.0, 0.0, 0.0], [21.0, 25.0, 36.0]),
            ([34.0, 0.0, 0.0], [18.0, 24.0, 34.0] if unequal_sizes else [21.0, 25.0, 36.0]),
        ):
            mask |= np.sum(((local - center) / radii) ** 2, axis=1) <= 1.0
        mask = mask.reshape(shape)
        return mask, sitk.GetImageFromArray(mask.astype(np.uint8)), pose[:, 0]

    def test_true_bilateral_lungs_with_unequal_volumes_keep_reliable_pose(self):
        """双肺可大小不同，不能用强制等体积替代跨侧对应检查。"""

        mask, image, expected_lr = self._make_bilateral_pair()
        result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)

        sizes = diagnostics["retained_component_sizes"]
        self.assertLess(min(sizes) / max(sizes), 0.85)
        self.assertTrue(result.global_frame.applied, diagnostics)
        evidence = diagnostics["bilateral_evidence"]
        self.assertTrue(evidence["reliable"])
        self.assertGreater(min(evidence["cross_component_match_fractions"]), 0.95)
        actual_lr = np.asarray(list(result.global_frame.left_right_axis.to_dict().values()))
        self.assertGreater(float(actual_lr @ expected_lr), np.cos(np.radians(2.0)))

    def test_aligned_bilateral_pair_passes_evidence_without_spurious_rotation(self):
        """无需旋转的正位双肺仍须通过跨侧证据，不应产生拒绝警告。"""

        mask, image, _ = self._make_bilateral_pair(yaw=0.0, roll=0.0, unequal_sizes=False)
        result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)

        self.assertTrue(diagnostics["bilateral_evidence"]["reliable"])
        self.assertEqual(diagnostics["symmetry"]["status"], "unchanged")
        self.assertFalse(result.global_frame.applied)
        self.assertEqual(result.global_frame.warnings, [])

    def test_centroids_on_opposite_sides_still_need_cross_component_matches(self):
        """左右质心虽分开，前后不对应的两块也不能靠各自低残差取得接受。"""

        shape = (80, 200, 140)
        z, y, x = np.indices(shape)
        mask = np.zeros(shape, dtype=bool)
        for cx, cy in ((35.0, 35.0), (105.0, 165.0)):
            mask |= ((x - cx) / 14.0) ** 2 + ((y - cy) / 23.0) ** 2 + ((z - 40.0) / 16.0) ** 2 <= 1.0
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        optimistic_fit = {
            "accepted": True,
            "status": "accepted",
            "angle_degrees": 0.0,
            "confidence": 0.9,
            "warnings": [],
            "reason": "点云自反射残差较低",
        }
        # 隔离第二级证据：即使优化器错误接受，逐侧身份检查也应阻止自匹配。
        with patch(
            "src.anatomy.lung.estimate_reflection_symmetry_plane",
            return_value=(np.array([1.0, 0.0, 0.0]), np.array([70.0, 100.0, 40.0]), optimistic_fit),
        ):
            result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)

        evidence = diagnostics["bilateral_evidence"]
        self.assertGreater(evidence["centroid_lr_separation_mm"], evidence["required_lr_separation_mm"])
        self.assertLess(np.prod(evidence["centroid_signed_distances_mm"]), 0.0)
        self.assertLess(min(evidence["cross_component_match_fractions"]), 0.1)
        self.assertFalse(result.global_frame.applied)
        self.assertIn("匹配自身", diagnostics["fallback_reason"])
        self.assertEqual(diagnostics["symmetry"]["status"], "rejected")

    def test_candidate_plane_must_separate_both_component_centroids(self):
        """候选面偏向一侧主体时，两个分离质心也不能被当成分居左右。"""

        mask, image, _ = self._make_bilateral_pair(yaw=0.0, roll=0.0)
        optimistic_fit = {
            "accepted": True,
            "status": "accepted",
            "angle_degrees": 0.0,
            "confidence": 0.9,
            "warnings": [],
        }
        with patch(
            "src.anatomy.lung.estimate_reflection_symmetry_plane",
            return_value=(np.array([1.0, 0.0, 0.0]), np.array([20.0, 50.0, 55.0]), optimistic_fit),
        ):
            result, diagnostics = estimate_lung_frame_with_diagnostics(mask, image)

        evidence = diagnostics["bilateral_evidence"]
        self.assertGreater(evidence["centroid_lr_separation_mm"], evidence["required_lr_separation_mm"])
        self.assertGreater(np.prod(evidence["centroid_signed_distances_mm"]), 0.0)
        self.assertFalse(result.global_frame.applied)
        self.assertIn("分居", diagnostics["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
