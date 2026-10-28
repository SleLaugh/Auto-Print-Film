"""Regression coverage for LR-supported recovery of flattened body projections."""

import unittest

import numpy as np

from src.stages.stage_localization import LocalizationStage


class PrismLRProjectionRecoveryTest(unittest.TestCase):
    @staticmethod
    def _tilted_bodies(*, shift=4, shifted_layers=1, shear=0):
        """Tilt only a thin LR edge while keeping a gap in every LR section."""

        return tuple(
            np.asarray([
                (z + shear * y + (shift if x < shifted_layers else 0), y, x)
                for x in range(20)
                for z in range(start, start + 10)
                for y in range(10)
            ], dtype=int)
            for start in (0, 12)
        )

    @staticmethod
    def _regions(left, right):
        """Deliberately overlap the regularized envelopes, not the body voxels."""

        points = np.concatenate((left, right))
        minimum = points.min(axis=0)
        shape = points.max(axis=0) - minimum + 1
        return {
            label: (minimum.copy(), np.ones(tuple(shape), dtype=bool))
            for label in ("vertebrae_C2", "vertebrae_C3")
        }

    def _assert_projection_recovery(self, left, right, spacing):
        # The old paths only know unique YZ positions. Their failure proves the
        # test exercises the new fallback without mocking either implementation.
        self.assertIsNone(LocalizationStage._sagittal_body_separating_plane(
            left[:, :2], right[:, :2], spacing,
        ))
        plane = LocalizationStage._sagittal_body_separating_plane(left, right, spacing)
        self.assertIsNotNone(plane)
        normal, offset = plane
        physical_left = left[:, :2] * spacing[:2]
        physical_right = right[:, :2] * spacing[:2]
        left_projection = physical_left @ normal
        right_projection = physical_right @ normal
        left_wrong = np.count_nonzero(left_projection > offset) / len(left)
        right_wrong = np.count_nonzero(right_projection < offset) / len(right)
        self.assertLessEqual(left_wrong, 0.05)
        self.assertLessEqual(right_wrong, 0.05)
        unique_left = np.unique(left[:, :2], axis=0) * spacing[:2]
        unique_right = np.unique(right[:, :2], axis=0) * spacing[:2]
        self.assertGreater(max(
            np.count_nonzero(unique_left @ normal > offset) / len(unique_left),
            np.count_nonzero(unique_right @ normal < offset) / len(unique_right),
        ), 0.05)
        maximum_depth = 2.0 * float(np.abs(normal) @ spacing[:2])
        self.assertLessEqual(float(left_projection.max() - offset), maximum_depth + 1e-6)
        self.assertLessEqual(float(offset - right_projection.min()), maximum_depth + 1e-6)
        for x in np.intersect1d(left[:, 2], right[:, 2]):
            self.assertLess(
                float(left_projection[left[:, 2] == x].max()),
                float(right_projection[right[:, 2] == x].min()),
            )
        return plane

    def _assert_bodies_preserved_without_overlap(self, resolved, left, right):
        occupied = []
        for label, body in (("vertebrae_C2", left), ("vertebrae_C3", right)):
            self.assertIsNotNone(resolved[label])
            minimum, mask = resolved[label]
            self.assertTrue(np.all(mask[tuple((body - minimum).T)]))
            occupied.append(set(map(tuple, np.argwhere(mask) + minimum)))
        self.assertFalse(occupied[0] & occupied[1])

    def test_recovers_shallow_lr_tilt_that_unique_projection_rejects(self):
        left, right = self._tilted_bodies()
        normal, offset = self._assert_projection_recovery(left, right, np.ones(3))
        np.testing.assert_array_equal(normal, [1.0, 0.0])
        self.assertEqual(offset, 12.0)

    def test_recovery_uses_physical_spacing_for_oblique_separator(self):
        left, right = self._tilted_bodies(shear=1)
        for spacing in (np.asarray([2.0, 0.65, 1.2]), np.asarray([0.6, 2.0, 1.2])):
            with self.subTest(spacing=spacing):
                normal, _ = self._assert_projection_recovery(left, right, spacing)
                expected = np.asarray([1.0 / spacing[0], -1.0 / spacing[1]])
                expected /= np.linalg.norm(expected)
                np.testing.assert_allclose(normal, expected, atol=1e-12)

    def test_recovered_prisms_preserve_bodies_and_reverse_order_exactly(self):
        left, right = self._tilted_bodies()
        regions = self._regions(left, right)
        protected = {"vertebrae_C2": left, "vertebrae_C3": right}
        resolved = LocalizationStage._resolve_non_overlapping_prism_regions(
            regions, protected, np.asarray([1.5, 0.65, 0.65]),
        )
        reverse = LocalizationStage._resolve_non_overlapping_prism_regions(
            dict(reversed(regions.items())), dict(reversed(protected.items())),
            np.asarray([1.5, 0.65, 0.65]),
        )
        self._assert_bodies_preserved_without_overlap(resolved, left, right)
        for label in regions:
            np.testing.assert_array_equal(resolved[label][0], reverse[label][0])
            np.testing.assert_array_equal(resolved[label][1], reverse[label][1])
            self.assertTrue(np.all(regions[label][1]))

    def test_balanced_weighted_scores_keep_identical_plane_when_swapping_bodies(self):
        # Equal wrong-side voxel counts must tie exactly in both directions.
        # Subtracting a floating fraction from 1 used to choose z=8 vs z=9.
        left = np.asarray([
            (z + x, y, x) for z in range(7) for y in range(3) for x in range(4)
        ])
        right = left + [8, 0, 0]
        normal, offset = self._assert_projection_recovery(left, right, np.ones(3))
        reverse_normal, reverse_offset = LocalizationStage._sagittal_body_separating_plane(
            right, left,
        )
        np.testing.assert_array_equal(normal, -reverse_normal)
        self.assertEqual(offset, -reverse_offset)
        regions = self._regions(left, right)
        protected = {"vertebrae_C2": left, "vertebrae_C3": right}
        forward = LocalizationStage._resolve_non_overlapping_prism_regions(regions, protected)
        reverse = LocalizationStage._resolve_non_overlapping_prism_regions(
            dict(reversed(regions.items())), protected,
        )
        self._assert_bodies_preserved_without_overlap(forward, left, right)
        for label in regions:
            np.testing.assert_array_equal(forward[label][1], reverse[label][1])

    def test_rejects_thin_deep_tail_despite_low_voxel_error(self):
        left, right = self._tilted_bodies(shift=12)
        # A weighted-only criterion would accept this deep tail at z=12.
        self.assertLessEqual(np.count_nonzero(left[:, 0] > 12) / len(left), 0.05)
        self.assertEqual(np.count_nonzero(right[:, 0] < 12), 0)
        self.assertGreater(left[:, 0].max() - 12, 2)
        self.assertIsNone(LocalizationStage._sagittal_body_separating_plane(left, right))

    def test_rejects_interwoven_common_lr_layer_even_with_shallow_low_error(self):
        left_set = {
            (z, y, x) for z in range(10) for y in range(10) for x in range(20)
        }
        right_set = {
            (z, y, x) for z in range(12, 22) for y in range(10) for x in range(20)
        }
        # In one LR layer the upper body has two shallow outer protrusions,
        # surrounding a lower-body protrusion. No YZ line separates that layer.
        for z in range(10, 14):
            for y in (*range(4), *range(6, 10)):
                left_set.add((z, y, 0))
                right_set.discard((z, y, 0))
        right_set.update((z, y, 0) for z in (10, 11) for y in (4, 5))
        left, right = np.asarray(sorted(left_set)), np.asarray(sorted(right_set))
        self.assertFalse(left_set & right_set)
        self.assertLess(np.count_nonzero(left[:, 0] > 12) / len(left), 0.05)
        self.assertLess(np.count_nonzero(right[:, 0] < 12) / len(right), 0.05)
        self.assertLessEqual(left[:, 0].max() - 12, 2)
        self.assertLessEqual(12 - right[:, 0].min(), 2)
        self.assertIsNone(LocalizationStage._sagittal_body_separating_plane(left, right))

    def test_rejects_disjoint_lr_support(self):
        left, right = self._tilted_bodies()
        right[:, 2] += 30
        self.assertIsNone(LocalizationStage._sagittal_body_separating_plane(left, right))

    def test_requires_half_of_each_body_lr_support(self):
        left, right = self._tilted_bodies()
        extra_right = np.asarray([
            (z, y, x) for x in range(20, 41) for z in range(12, 22) for y in range(10)
        ])
        right = np.concatenate((right, extra_right))
        # All left layers are shared, but fewer than half of right layers are.
        self.assertIsNone(LocalizationStage._sagittal_body_separating_plane(left, right))

    def test_accepts_exactly_half_shared_lr_support(self):
        left, right = self._tilted_bodies()
        right[:, 2] += 10
        self._assert_projection_recovery(left, right, np.ones(3))

    def test_rejects_single_common_lr_layer_even_at_half_coverage(self):
        left = np.asarray([
            (z + shift, y, x) for x, shift in ((0, 0), (1, 4))
            for z in range(10) for y in range(10)
        ])
        right = np.asarray([
            (z + shift, y, x) for x, shift in ((2, 0), (1, 4))
            for z in range(12, 22) for y in range(10)
        ])
        self.assertEqual(len(np.intersect1d(left[:, 2], right[:, 2])), 1)
        self.assertIsNone(LocalizationStage._sagittal_body_separating_plane(left, right))

    def test_true_body_voxel_collision_still_invalidates_both_prisms(self):
        left, right = self._tilted_bodies()
        right = np.concatenate((right, left[:1]))
        resolved = LocalizationStage._resolve_non_overlapping_prism_regions(
            self._regions(left, right),
            {"vertebrae_C2": left, "vertebrae_C3": right},
        )
        self.assertIsNone(resolved["vertebrae_C2"])
        self.assertIsNone(resolved["vertebrae_C3"])

    def test_strong_projected_crossing_still_invalidates_both_prisms(self):
        left, right = self._tilted_bodies(shift=12, shifted_layers=10)
        self.assertFalse(set(map(tuple, left)) & set(map(tuple, right)))
        self.assertIsNone(LocalizationStage._sagittal_body_separating_plane(left, right))
        resolved = LocalizationStage._resolve_non_overlapping_prism_regions(
            self._regions(left, right),
            {"vertebrae_C2": left, "vertebrae_C3": right},
        )
        self.assertIsNone(resolved["vertebrae_C2"])
        self.assertIsNone(resolved["vertebrae_C3"])

    def test_original_strict_separator_is_unchanged(self):
        left, right = self._tilted_bodies(shift=0)
        for spacing in (np.ones(3), np.asarray([2.0, 0.65, 1.2])):
            with self.subTest(spacing=spacing):
                normal, offset = LocalizationStage._sagittal_body_separating_plane(
                    left, right, spacing,
                )
                np.testing.assert_array_equal(normal, [1.0, 0.0])
                self.assertEqual(offset, 10.5 * spacing[0])

    def test_original_five_percent_fallback_is_unchanged_without_lr_evidence(self):
        left = np.asarray([
            (z, y, 0) for z in range(10) for y in range(10)
        ] + [(13, 5, 0)])
        right = np.asarray([
            (z, y, 1) for z in range(12, 22) for y in range(10)
        ] + [(8, 5, 1)])
        normal, offset = LocalizationStage._sagittal_body_separating_plane(left, right)
        np.testing.assert_array_equal(normal, [1.0, 0.0])
        self.assertEqual(offset, 10.5)
        # This legacy success has disjoint LR support and >2-voxel tails; the
        # new fallback guards must not retroactively change its result.
        self.assertGreater(left[:, 0].max() - offset, 2)
        self.assertGreater(offset - right[:, 0].min(), 2)


if __name__ == "__main__":
    unittest.main()
