"""双侧髋骨应以物理镜像对称性回正，而非以顶部点或体素 PCA 定角。"""

import math
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.anatomy.hip import estimate_hip_frame_with_diagnostics


def _axes_from_yaw_roll(yaw_degrees: float, roll_degrees: float) -> np.ndarray:
    """构造左右轴偏转及保留输入前后基准的右手三轴。"""

    yaw, roll = np.deg2rad([yaw_degrees, roll_degrees])
    lr = np.asarray([
        math.cos(roll) * math.cos(yaw),
        math.cos(roll) * math.sin(yaw),
        math.sin(roll),
    ])
    ap = np.asarray([0.0, 1.0, 0.0])
    ap -= float(ap @ lr) * lr
    ap /= np.linalg.norm(ap)
    return np.column_stack((lr, ap, np.cross(lr, ap)))


def _make_pair(
    yaw_degrees: float,
    roll_degrees: float,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, sitk.Image, np.ndarray]:
    """栅格化有前后非对称形态、但左右严格镜像的一对合成骨盆主体。"""

    shape = (86, 80, 96)
    xyz = np.indices(shape, dtype=float)[::-1].transpose(1, 2, 3, 0)
    centered = (xyz - (np.asarray(shape[::-1]) - 1.0) / 2.0) * spacing
    local_axes = _axes_from_yaw_roll(yaw_degrees, roll_degrees)
    local = centered @ local_axes
    x, y, z = (local[..., axis] for axis in range(3))
    abs_x = np.abs(x)
    body = (
        ((abs_x - 21.0) / 8.0) ** 2
        + ((y - 2.0) / 13.0) ** 2
        + (z / 20.0) ** 2
        <= 1.0
    )
    wing = (
        ((abs_x - 25.0) / 9.0) ** 2
        + ((y + 6.0) / 8.0) ** 2
        + ((z - 16.0) / 10.0) ** 2
        <= 1.0
    )
    stem = (
        ((abs_x - 14.0) / 6.0) ** 2
        + ((y - 10.0) / 7.0) ** 2
        + ((z + 14.0) / 12.0) ** 2
        <= 1.0
    )
    mask = body | wing | stem
    left, right = mask & (x > 0.0), mask & (x < 0.0)
    image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
    image.SetSpacing(spacing)
    image.SetOrigin((103.0, -207.0, 61.0))
    if direction is None:
        direction = np.eye(3)
    image.SetDirection(direction.ravel().tolist())
    return left, right, image, direction @ local_axes


def _frame_axes(frame) -> np.ndarray:
    """读取按 LR/AP/SI 排列的三列物理方向。"""

    return np.asarray([
        [point.x, point.y, point.z]
        for point in (
            frame.left_right_axis,
            frame.anterior_posterior_axis,
            frame.superior_inferior_axis,
        )
    ]).T


class HipPoseSymmetryTest(unittest.TestCase):
    def assert_direction_close(self, actual, expected, degrees=1.0):
        """比较带方向的单位轴，避免反号也被误当作回正成功。"""

        angle = math.degrees(math.acos(float(np.clip(actual @ expected, -1.0, 1.0))))
        self.assertLess(angle, degrees)

    def test_recovers_combined_yaw_and_roll_from_bilateral_shape(self):
        """小于旧 8° 阈值的 yaw 也必须与横滚一起被双侧对称性恢复。"""

        left, right, image, expected_axes = _make_pair(-5.0, 6.0)
        result, diagnostics = estimate_hip_frame_with_diagnostics(left, right, image)
        frame = result.global_frame
        axes = _frame_axes(frame)

        self.assertTrue(frame.applied, diagnostics["symmetry"])
        self.assertEqual(frame.method, "hip_reflection_symmetry")
        self.assert_direction_close(axes[:, 0], expected_axes[:, 0])
        self.assert_direction_close(axes[:, 1], expected_axes[:, 1])
        self.assert_direction_close(axes[:, 2], expected_axes[:, 2])
        self.assertGreater(abs(float(axes[1, 0])), 0.05)
        self.assertGreater(abs(float(axes[2, 0])), 0.07)
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-6)
        self.assertGreater(np.linalg.det(axes), 0.999999)
        self.assertEqual(diagnostics["ab_role"], "diagnostic_only")
        self.assertEqual(
            diagnostics["pitch_source"],
            "projected_input_anterior_posterior_axis",
        )

    def test_oblique_anisotropic_image_recovers_physical_axes_and_full_bounds(self):
        """体素间距和存储方向改变时仍恢复物理姿态，并完整覆盖所有主体点。"""

        direction = _axes_from_yaw_roll(24.0, -17.0)
        left, right, image, expected_axes = _make_pair(
            -12.0,
            8.0,
            spacing=(1.0, 1.3, 1.1),
            direction=direction,
        )
        result, diagnostics = estimate_hip_frame_with_diagnostics(left, right, image)
        axes = _frame_axes(result.global_frame)
        self.assertTrue(result.global_frame.applied, diagnostics["symmetry"])
        for axis in range(3):
            self.assert_direction_close(axes[:, axis], expected_axes[:, axis], degrees=1.5)

        indices = np.argwhere(left | right)[:, ::-1].astype(float)
        points = (
            np.asarray(image.GetOrigin())
            + (indices * np.asarray(image.GetSpacing())) @ direction.T
        )
        for axis, bounds in zip(
            axes.T,
            (
                result.global_frame.bounds.left_right,
                result.global_frame.bounds.anterior_posterior,
                result.global_frame.bounds.superior_inferior,
            ),
        ):
            projection = points @ axis
            self.assertAlmostEqual(bounds.minimum, float(projection.min()), places=6)
            self.assertAlmostEqual(bounds.maximum, float(projection.max()), places=6)

    def test_aligned_pair_does_not_acquire_a_spurious_rotation(self):
        """原本正位的左右对称主体应维持输入轴。"""

        left, right, image, _ = _make_pair(0.0, 0.0)
        result, _ = estimate_hip_frame_with_diagnostics(left, right, image)
        axes = _frame_axes(result.global_frame)
        for axis in range(3):
            self.assert_direction_close(axes[:, axis], np.eye(3)[:, axis])

    def test_reflected_storage_keeps_physical_lr_and_right_handed_frame(self):
        """单轴反向存储不应让拟合失败，物理位置也不能随参考SI翻转。"""

        direction = np.diag((-1.0, 1.0, 1.0))
        left, right, image, expected = _make_pair(-5.0, 6.0, direction=direction)
        result, diagnostics = estimate_hip_frame_with_diagnostics(left, right, image)
        axes = _frame_axes(result.global_frame)
        self.assertTrue(result.global_frame.applied, diagnostics)
        self.assert_direction_close(axes[:, 0], -expected[:, 0])
        self.assertGreater(axes[0, 0], 0.0)
        self.assertGreater(axes[2, 2], 0.0)
        self.assertGreater(np.linalg.det(axes), 0.999999)
        indices = np.argwhere(left | right)[:, ::-1]
        expected_center = np.asarray(image.GetOrigin()) + indices.mean(axis=0) @ direction.T
        np.testing.assert_allclose(
            list(result.global_frame.origin.to_dict().values()), expected_center,
        )

    def test_rejected_fit_warns_and_keeps_input_axes_instead_of_using_ab(self):
        """拟合被质量检查拒绝时不能又偷偷用髂顶线制造一个回正角。"""

        left, right, image, _ = _make_pair(-5.0, 6.0)
        rejected = {
            "accepted": False,
            "status": "rejected",
            "reason": "镜像残差过大",
            "warnings": ["镜像残差过大"],
        }
        with patch(
            "src.anatomy.hip.estimate_reflection_symmetry_plane",
            return_value=(np.asarray([0.99, 0.0, 0.1]), np.zeros(3), rejected),
        ):
            result, diagnostics = estimate_hip_frame_with_diagnostics(left, right, image)

        frame = result.global_frame
        self.assertFalse(frame.applied)
        self.assertEqual(frame.confidence, 0.0)
        self.assertEqual(frame.method, "hip_dicom_axes")
        self.assertTrue(any("镜像残差过大" in warning for warning in frame.warnings))
        np.testing.assert_allclose(_frame_axes(frame), np.eye(3), atol=1e-8)
        self.assertNotAlmostEqual(
            diagnostics["left_landmark"]["z"],
            diagnostics["right_landmark"]["z"],
        )

    def test_overlapping_sides_are_reported_as_unusable_pose_geometry(self):
        """错误地把同一个主体当作两侧，必须明确回退而不是声称对称拟合成功。"""

        left, _, image, _ = _make_pair(0.0, 0.0)
        result, diagnostics = estimate_hip_frame_with_diagnostics(left, left.copy(), image)
        self.assertFalse(result.global_frame.applied)
        self.assertEqual(diagnostics["symmetry"]["status"], "rejected")
        self.assertTrue(any("重叠" in warning for warning in result.global_frame.warnings))
        np.testing.assert_allclose(_frame_axes(result.global_frame), np.eye(3), atol=1e-8)

    def test_severe_unilateral_loss_does_not_define_a_false_symmetry_plane(self):
        """一侧仅残留少量骨块时，不应让全点云的其他对称平面驱动姿态。"""

        left, right, image, _ = _make_pair(0.0, 0.0)
        right[:] = False
        right[35:38, 35:38, 22:25] = True
        result, diagnostics = estimate_hip_frame_with_diagnostics(left, right, image)
        self.assertFalse(result.global_frame.applied)
        self.assertEqual(diagnostics["symmetry"]["status"], "rejected")
        self.assertTrue(any("体积差异" in warning for warning in result.global_frame.warnings))


if __name__ == "__main__":
    unittest.main()
