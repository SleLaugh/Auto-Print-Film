"""验证镜像面在物理空间恢复姿态及有证据的回退。"""

import unittest
import warnings

import numpy as np

from src.anatomy.symmetry import (
    build_symmetry_frame_axes,
    estimate_reflection_symmetry_plane,
)


def _rotation_yz(y_degrees, z_degrees):
    """构造已知左右侧倾和横断旋转，不依赖优化器自身的表达式。"""

    y, z = np.radians([y_degrees, z_degrees])
    ry = np.array([[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0.0], [np.sin(z), np.cos(z), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry


def _symmetric_surface(count=2500):
    """生成精确成对但轴长不同的三维表面，避免抽样本身破坏左右对称。"""

    rng = np.random.default_rng(12)
    half = rng.normal(size=(count, 3))
    half[:, 0] = np.abs(half[:, 0])
    half /= np.linalg.norm(half, axis=1)[:, None]
    half *= np.array([40.0, 65.0, 85.0])
    mirror = half.copy()
    mirror[:, 0] *= -1
    return np.concatenate((half, mirror))


class ReflectionSymmetryPlaneTest(unittest.TestCase):
    def test_recovers_joint_rotation_in_oblique_reference(self):
        """斜向存储坐标及非零物理原点下恢复同一个姿态，不假定 LPS=体素轴。"""

        reference = _rotation_yz(17.0, -23.0)
        pose = _rotation_yz(-9.0, 6.0)
        offset = np.array([120.0, -81.0, 41.0])
        points = _symmetric_surface() @ pose.T @ reference.T + offset
        normal, center, diagnostics = estimate_reflection_symmetry_plane(points, reference)

        self.assertTrue(diagnostics["accepted"], diagnostics)
        expected = reference @ pose[:, 0]
        self.assertGreater(float(np.dot(normal, expected)), np.cos(np.radians(0.8)))
        self.assertLess(abs(float(np.dot(center - offset, normal))), 0.8)
        self.assertGreater(diagnostics["improvement_fraction"], 0.5)
        self.assertEqual(diagnostics["sample_count"], len(points))

    def test_identity_is_reliably_unchanged(self):
        """已正位时保留原方向，不能用抽样误差制造额外旋转。"""

        normal, _, diagnostics = estimate_reflection_symmetry_plane(_symmetric_surface(), np.eye(3))
        np.testing.assert_allclose(normal, [1.0, 0.0, 0.0])
        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(diagnostics["status"], "unchanged")
        self.assertEqual(diagnostics["warnings"], [])

    def test_sampling_is_capped_and_deterministic(self):
        """输入数量增加时匹配负担固定，相同输入重复执行得到相同结果。"""

        points = _symmetric_surface(4500) @ _rotation_yz(-8.0, 5.0).T
        first = estimate_reflection_symmetry_plane(points, np.eye(3), max_points=1600)
        second = estimate_reflection_symmetry_plane(points, np.eye(3), max_points=1600)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[2]["sample_count"], 1600)
        self.assertTrue(first[2]["accepted"], first[2])

    def test_search_boundary_is_not_a_success(self):
        """实际倾角超出许可范围时拒绝贴着边界的最优值。"""

        points = _symmetric_surface() @ _rotation_yz(-18.0, 0.0).T
        normal, _, diagnostics = estimate_reflection_symmetry_plane(points, np.eye(3), max_angle_degrees=7.0)
        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(diagnostics["status"], "rejected")
        np.testing.assert_array_equal(normal, [1.0, 0.0, 0.0])
        self.assertTrue(diagnostics["warnings"])

    def test_tiny_and_planar_shapes_retain_reference(self):
        """点数过少或目标只有一个平面时不能声称确定了三维姿态。"""

        for points in (np.ones((6, 3)), np.column_stack((np.arange(100), np.arange(100) ** 2, np.zeros(100)))):
            with self.subTest(count=len(points)):
                normal, _, diagnostics = estimate_reflection_symmetry_plane(points, np.eye(3))
                self.assertEqual(diagnostics["status"], "rejected")
                np.testing.assert_array_equal(normal, [1.0, 0.0, 0.0])

    def test_focal_outliers_do_not_control_the_plane(self):
        """少量非对称附件不应压过主体成对表面。"""

        rng = np.random.default_rng(8)
        points = np.concatenate((_symmetric_surface(), rng.uniform([-150, -100, -100], [150, 100, 100], size=(200, 3))))
        pose = _rotation_yz(-9.0, 6.0)
        normal, _, diagnostics = estimate_reflection_symmetry_plane(points @ pose.T, np.eye(3))
        self.assertTrue(diagnostics["accepted"], diagnostics)
        self.assertGreater(float(np.dot(normal, pose[:, 0])), np.cos(np.radians(0.5)))

    def test_one_sided_or_spherical_shape_does_not_claim_a_pose(self):
        """缺失一半目标和近球形目标均不能靠最优值宣称成功。"""

        surface = _symmetric_surface()
        shapes = (surface[surface[:, 0] > 0.0], surface / [40.0, 65.0, 85.0] * 50.0)
        for points in shapes:
            with self.subTest(count=len(points)):
                normal, _, diagnostics = estimate_reflection_symmetry_plane(points, np.eye(3))
                self.assertEqual(diagnostics["status"], "rejected")
                self.assertTrue(diagnostics["warnings"])
                np.testing.assert_array_equal(normal, [1.0, 0.0, 0.0])

    def test_axially_symmetric_sampling_phase_cannot_create_a_pose(self):
        """多密度、不同点序的圆柱没有唯一左右面，不能因离散相位而旋转。"""

        for count in (17, 23, 47, 71, 257, 1001):
            phi = np.arange(count) * 2.0 * np.pi / count + np.radians(13.0)
            points = np.array([
                [30.0 * np.cos(angle), 30.0 * np.sin(angle), z]
                for z in np.linspace(-60.0, 60.0, 12)
                for angle in phi
            ])
            shuffled = points[np.random.default_rng(25).permutation(len(points))]
            for ordering, sample in (("original", points), ("reversed", points[::-1]), ("shuffled", shuffled)):
                with self.subTest(count=count, ordering=ordering):
                    normal, _, diagnostic = estimate_reflection_symmetry_plane(sample, np.eye(3))
                    self.assertFalse(diagnostic["accepted"], diagnostic)
                    self.assertEqual(diagnostic["status"], "rejected")
                    self.assertEqual(diagnostic["direction_uniqueness"]["status"], "ambiguous")
                    self.assertTrue(diagnostic["warnings"])
                    np.testing.assert_array_equal(normal, [1.0, 0.0, 0.0])

    def test_unique_lr_is_usable_when_only_ap_si_eigenvalues_repeat(self):
        """AP/SI 相同仅使俯仰不唯一，不能因此丢弃独立可靠的左右法向。"""

        surface = _symmetric_surface() / [40.0, 65.0, 85.0] * [40.0, 70.0, 70.0]
        for roll, yaw in ((-9.0, 6.0), (13.0, -17.0), (0.0, 20.0)):
            with self.subTest(roll=roll, yaw=yaw):
                rotation = _rotation_yz(roll, yaw)
                normal, _, diagnostic = estimate_reflection_symmetry_plane(surface @ rotation.T, np.eye(3))
                self.assertTrue(diagnostic["accepted"], diagnostic)
                self.assertGreater(float(normal @ rotation[:, 0]), np.cos(np.radians(0.5)))
                self.assertEqual(diagnostic["direction_uniqueness"]["status"], "passed")

    def test_repeated_lr_spectrum_uses_actual_contour_to_disambiguate(self):
        """协方差重根不等于缺少语义：明确轮廓在搜索范围内只有一个镜像面。"""

        phi = np.arange(180) * 2.0 * np.pi / 180.0
        radius = 30.0 * (1.0 + 0.15 * np.sin(3.0 * phi))
        points = np.array([
            [r * np.cos(angle), r * np.sin(angle), z]
            for z in np.linspace(-60.0, 60.0, 12)
            for r, angle in zip(radius, phi)
        ])
        rotation = _rotation_yz(-4.0, 13.0)
        normal, _, diagnostic = estimate_reflection_symmetry_plane(points @ rotation.T, np.eye(3))
        self.assertTrue(diagnostic["accepted"], diagnostic)
        self.assertGreater(float(normal @ rotation[:, 0]), np.cos(np.radians(0.5)))
        self.assertTrue(diagnostic["direction_uniqueness"]["lr_degenerate_subspaces"])
        self.assertEqual(diagnostic["direction_uniqueness"]["status"], "passed")

    def test_multiple_equivalent_contour_planes_are_not_a_unique_pose(self):
        """存在轮廓信息但许可角度内有多个同样好的面时仍须拒绝。"""

        phi = np.arange(180) * 2.0 * np.pi / 180.0
        radius = 30.0 * (1.0 + 0.15 * np.sin(5.0 * phi))
        points = np.array([
            [r * np.cos(angle), r * np.sin(angle), z]
            for z in np.linspace(-60.0, 60.0, 12)
            for r, angle in zip(radius, phi)
        ])
        rotation = _rotation_yz(-4.0, 13.0)
        normal, _, diagnostic = estimate_reflection_symmetry_plane(points @ rotation.T, np.eye(3))
        self.assertEqual(diagnostic["status"], "rejected")
        self.assertFalse(diagnostic["accepted"])
        self.assertEqual(diagnostic["direction_uniqueness"]["status"], "ambiguous")
        self.assertGreater(diagnostic["direction_uniqueness"]["separation_degrees"], 30.0)
        self.assertTrue(diagnostic["warnings"])
        np.testing.assert_array_equal(normal, [1.0, 0.0, 0.0])

    def test_angular_domain_boundary_refinement_has_no_numerical_warning(self):
        """近重根盒状骶骨的最佳替代角靠近禁区边界，细化不能运算无穷大。"""

        surface = np.ones((14, 18, 18), dtype=bool)
        surface[1:-1, 1:-1, 1:-1] = False
        points = np.argwhere(surface)[:, ::-1] * [0.8, 0.8, 1.2]
        # 提升警告为异常，验证真实边界路径；不屏蔽 SciPy 的数值诊断。
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            normal, _, diagnostic = estimate_reflection_symmetry_plane(points, np.eye(3))
        self.assertEqual(diagnostic["status"], "unchanged")
        self.assertEqual(diagnostic["direction_uniqueness"]["status"], "passed")
        self.assertFalse(diagnostic["accepted"])
        np.testing.assert_array_equal(normal, [1.0, 0.0, 0.0])

    def test_invalid_physical_inputs_are_rejected(self):
        """无效物理坐标和不正交参考系必须明确失败。"""

        for points, axes in ((np.array([[np.nan, 0.0, 0.0]]), np.eye(3)), (np.ones((100, 3)), np.ones((3, 3)))):
            with self.assertRaises(ValueError):
                estimate_reflection_symmetry_plane(points, axes)

    def test_frame_projects_input_ap_without_extra_pitch(self):
        """对称面只提供 LR，另外两轴遵循明确的原 AP 投影约束。"""

        reference = _rotation_yz(7.0, 14.0)
        normal = reference @ _rotation_yz(-9.0, 6.0)[:, 0]
        axes = build_symmetry_frame_axes(normal, reference)
        expected_ap = reference[:, 1] - np.dot(reference[:, 1], normal) * normal
        expected_ap /= np.linalg.norm(expected_ap)
        np.testing.assert_allclose(axes[:, 1], expected_ap)
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0)


if __name__ == "__main__":
    unittest.main()
