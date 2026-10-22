"""密实核心恢复必须有原标签证据，端面拟合仅使用受支撑的中央区域。"""

import unittest

import numpy as np

from src.stages.spine_body_core import (
    find_dense_body_core,
    fit_body_endplates_from_core,
    recover_body_from_dense_core,
)


class SpineBodyCoreTest(unittest.TestCase):
    @staticmethod
    def _partial_body():
        raw = np.zeros((44, 72, 64), dtype=bool)
        raw[10:34, 24:48, 20:44] = True
        raw[6:38, 12:16, 20:44] = True
        raw[18:26, 16:24, 30:34] = True
        reference = np.zeros_like(raw)
        reference[10:34, 24:32, 20:44] = True
        return raw, reference

    def test_recovers_missing_body_without_growing_through_thin_anterior_plate(self):
        raw, reference = self._partial_body()
        diagnostics = {}
        recovered = recover_body_from_dense_core(
            raw, reference, (1.0, 1.0, 1.0), diagnostics=diagnostics,
        )

        self.assertIsNotNone(recovered)
        self.assertTrue(np.all(recovered <= raw))
        self.assertTrue(np.all(recovered[10:34, 24:48, 20:44]))
        self.assertFalse(np.any(recovered[:, :16]))
        self.assertEqual(diagnostics["method"], "density_core_recovery")
        self.assertLess(diagnostics["core_reference_coverage"], 0.9)

    def test_reference_containing_the_dense_body_is_unchanged(self):
        raw, _ = self._partial_body()
        reference = np.zeros_like(raw)
        reference[10:34, 24:48, 20:44] = True

        self.assertIsNone(recover_body_from_dense_core(raw, reference, (1, 1, 1)))

    def test_anisotropic_sampling_uses_physical_kernel(self):
        raw, reference = self._partial_body()
        evidence = find_dense_body_core(raw, reference, (1, 1, 1))
        sampled = find_dense_body_core(
            np.repeat(raw, 2, axis=0), np.repeat(reference, 2, axis=0), (0.5, 1, 1),
        )

        self.assertIsNotNone(evidence)
        self.assertIsNotNone(sampled)
        self.assertAlmostEqual(sampled.kernel_zyx[0] * 0.5, evidence.kernel_zyx[0], delta=1.0)
        self.assertEqual(sampled.kernel_zyx[1:], evidence.kernel_zyx[1:])
        self.assertLess(abs(sampled.reference_coverage - evidence.reference_coverage), 0.1)

    def test_invalid_spacing_and_empty_evidence_do_not_recover(self):
        raw, reference = self._partial_body()
        for spacing in ((0, 1, 1), (float("nan"), 1, 1), (1, 1)):
            with self.subTest(spacing=spacing):
                self.assertIsNone(recover_body_from_dense_core(raw, reference, spacing))
        self.assertIsNone(recover_body_from_dense_core(raw, np.zeros_like(raw), (1, 1, 1)))

    def test_short_dense_fragment_cannot_replace_tall_body(self):
        raw = np.zeros((44, 60, 64), bool)
        raw[20:22, 24:48, 20:44] = True
        reference = np.zeros_like(raw)
        reference[4:40, 8:10, 20:44] = True
        raw |= reference

        self.assertIsNone(recover_body_from_dense_core(raw, reference, (1, 1, 1)))

    def test_tilted_body_already_covering_its_core_is_not_refitted(self):
        raw = np.zeros((50, 60, 64), bool)
        for y in range(14, 42):
            start = 8 + (y - 14) // 3
            raw[start:start + 20, y, 20:44] = True

        self.assertIsNone(recover_body_from_dense_core(raw, raw.copy(), (1, 1, 1)))

    def test_endplate_fit_respects_anisotropic_voxel_spacing_and_outward_sign(self):
        mask = np.zeros((48, 48, 48), bool)
        for y in range(10, 30):
            mask[y:y + 12, y, 12:32] = True
        faces = fit_body_endplates_from_core(
            np.argwhere(mask), (2.0, 1.0, 1.0),
            {"min": [18, 14, 18], "max": [28, 25, 25]},
        )

        self.assertIsNotNone(faces)
        self.assertEqual([face["side"] for face in faces], ["min", "max"])
        np.testing.assert_allclose(faces[0]["normal_zyx"], np.array([-1, 2, 0]) / np.sqrt(5))
        np.testing.assert_allclose(faces[1]["normal_zyx"], np.array([1, -2, 0]) / np.sqrt(5))
        self.assertEqual(faces[0]["median_residual_mm"], 0.0)
        self.assertTrue(all(point[0] == point[1] for point in faces[0]["endpoints_zyx"]))

    def test_disconnected_boundary_outliers_cannot_define_an_endplate(self):
        mask = np.zeros((100, 40, 30), bool)
        for y in range(10, 30):
            start = 5 if y % 2 else 60
            mask[start:start + 5, y, 5:25] = True
        self.assertIsNone(fit_body_endplates_from_core(
            np.argwhere(mask), (1, 1, 1), {"min": [5, 10, 8], "max": [64, 29, 21]},
        ))

    def test_endplate_uses_original_label_instead_of_artificial_recovery_cut(self):
        source = np.zeros((60, 48, 48), bool)
        for y in range(10, 30):
            source[y:y + 16, y, 12:32] = True
        recovered = source.copy()
        recovered[:24] = False
        recovered[35:] = False

        faces = fit_body_endplates_from_core(
            np.argwhere(recovered), (1, 1, 1),
            {"min": [25, 14, 18], "max": [32, 25, 25]},
            source_indices_zyx=np.argwhere(source),
        )

        self.assertIsNotNone(faces)
        np.testing.assert_allclose(faces[0]["normal_zyx"], [-1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        np.testing.assert_allclose(faces[1]["normal_zyx"], [1 / np.sqrt(2), -1 / np.sqrt(2), 0])
        self.assertEqual(faces[0]["source"], "original_label")

    def test_lr_consensus_rejects_jagged_boundary_from_larger_damaged_side(self):
        body = np.zeros((100, 60, 60), bool)
        source = np.zeros_like(body)
        for y in range(10, 40):
            start = 30 + y // 2
            body[start:start + 18, y, 10:50] = True
            source[start:start + 18, y, 10:50] = True
            false_start = max(0, start - ((y - 25) ** 2 // 8 + 3))
            source[false_start:start, y, 10:35] = True
        faces = fit_body_endplates_from_core(
            np.argwhere(body), (1, 1, 1),
            {"min": [43, 12, 23], "max": [56, 37, 36]},
            source_indices_zyx=np.argwhere(source),
        )

        self.assertIsNotNone(faces)
        expected = np.array([-1, 0.5, 0]) / np.sqrt(1.25)
        np.testing.assert_allclose(faces[0]["normal_zyx"], expected, atol=0.02)
        self.assertGreaterEqual(faces[0]["lr_profile_count"], 3)
        self.assertGreaterEqual(faces[0]["lr_profile_extent_voxels"][0], 35)

    def test_changing_lr_coverage_cannot_rotate_parallel_endplates(self):
        body = np.zeros((64, 48, 64), bool)
        body[10:40, 10:32, 10:54] = True
        source = np.zeros_like(body)
        source[10:35, 10:29, 10:32] = True
        source[20:45, 13:32, 32:54] = True
        faces = fit_body_endplates_from_core(
            np.argwhere(body), (1, 1, 1),
            {"min": [22, 10, 23], "max": [29, 31, 40]},
            source_indices_zyx=np.argwhere(source),
        )

        self.assertIsNotNone(faces)
        np.testing.assert_allclose(faces[0]["normal_zyx"], [-1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(faces[1]["normal_zyx"], [1, 0, 0], atol=1e-12)

    def test_ct_density_drop_rejects_flat_label_boundary_still_inside_bone(self):
        source = np.zeros((80, 60, 64), bool)
        raw = np.zeros(source.shape, dtype=np.float32)
        for y in range(10, 40):
            top = 35 + y // 2
            raw[25:top, y, 10:54] = 400.0
            source[25:40, y, 10:39] = True
            source[25:top, y, 39:54] = True
        bounds = {"min": [28, 12, 23], "max": [34, 36, 40]}
        indices = np.argwhere(source)
        without_ct = fit_body_endplates_from_core(
            indices, (1, 1, 1), bounds, source_indices_zyx=indices,
        )
        faces = fit_body_endplates_from_core(
            indices, (1, 1, 1), bounds, source_indices_zyx=indices, raw_array=raw,
        )

        self.assertIsNotNone(without_ct)
        np.testing.assert_allclose(without_ct[1]["normal_zyx"], [1, 0, 0])
        self.assertIsNotNone(faces)
        expected = np.array([1, -0.5, 0]) / np.sqrt(1.25)
        np.testing.assert_allclose(faces[1]["normal_zyx"], expected, atol=0.02)
        self.assertEqual(faces[1]["ct_validation"], "density_drop")
        self.assertGreaterEqual(faces[1]["ct_min_positive_fraction"], 0.8)
        self.assertGreaterEqual(faces[1]["ct_min_normalized_drop"], 0.2)

    def test_ct_without_an_actual_boundary_does_not_validate_a_label_plane(self):
        mask = np.zeros((60, 48, 48), bool)
        mask[15:40, 10:32, 10:38] = True
        indices = np.argwhere(mask)
        self.assertIsNone(fit_body_endplates_from_core(
            indices, (1, 1, 1), {"min": [22, 14, 19], "max": [30, 28, 29]},
            source_indices_zyx=indices, raw_array=np.full(mask.shape, 400.0),
        ))


if __name__ == "__main__":
    unittest.main()
