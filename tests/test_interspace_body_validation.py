"""The interspace-only conflict check must preserve credible existing bodies."""

import unittest
from unittest.mock import patch

import numpy as np

from src.stages.interspace_body import validate_interspace_body


class InterspaceBodyValidationTest(unittest.TestCase):
    def _conflict(self):
        raw = np.zeros((60, 110, 90), dtype=bool)
        front = np.zeros_like(raw)
        rear = np.zeros_like(raw)
        front[8:32, 10:24, 20:60] = True
        rear[40:48, 50:90, 28:40] = True
        raw |= front | rear
        return raw, front, rear

    def test_stable_anterior_core_replaces_long_short_posterior_appendage(self):
        raw, front, rear = self._conflict()
        original_raw, original_body = raw.copy(), rear.copy()
        diagnostics = {}
        replacement = validate_interspace_body(
            raw, rear, (1.0, 1.0, 1.0), vertebra_name="vertebrae_C7",
            diagnostics=diagnostics,
        )
        np.testing.assert_array_equal(replacement, front)
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(rear, original_body)
        self.assertTrue(diagnostics["replaced"])
        self.assertTrue(diagnostics["expensive_validation"])

    def test_normal_front_body_does_not_repeat_support_extraction(self):
        raw, front, _ = self._conflict()
        diagnostics = {}
        with patch("src.stages.stage_localization.LocalizationStage._vertebral_body_support_candidates") as extract:
            replacement = validate_interspace_body(
                raw, front, (1.0, 1.0, 1.0), vertebra_name="L4",
                diagnostics=diagnostics,
            )
        self.assertIsNone(replacement)
        extract.assert_not_called()
        self.assertFalse(diagnostics["expensive_validation"])

    def test_conflict_validation_avoids_structured_row_sorting(self):
        raw, front, rear = self._conflict()
        unique = np.unique

        def stable_unique(values, *args, **kwargs):
            self.assertIsNone(kwargs.get("axis"), "不能重新引入二维结构化排序路径")
            return unique(values, *args, **kwargs)

        with patch("src.stages.interspace_body.np.unique", side_effect=stable_unique):
            replacement = validate_interspace_body(
                raw, rear, (1.0, 1.0, 1.0), vertebra_name="C7",
            )
        np.testing.assert_array_equal(replacement, front)

    def test_physical_evidence_is_stable_under_axis_resampling(self):
        for factors in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)):
            with self.subTest(factors=factors):
                raw, front, rear = self._conflict()
                for axis, factor in enumerate(factors):
                    raw = np.repeat(raw, factor, axis=axis)
                    front = np.repeat(front, factor, axis=axis)
                    rear = np.repeat(rear, factor, axis=axis)
                replacement = validate_interspace_body(
                    raw, rear, tuple(1.0 / factor for factor in factors),
                    vertebra_name="L3",
                )
                np.testing.assert_array_equal(replacement, front)

    def test_c2_and_sacrum_keep_specialized_body(self):
        raw, _, rear = self._conflict()
        for name in ("C1", "vertebrae_C2", "S1", "sacrum"):
            with self.subTest(name=name):
                diagnostics = {}
                self.assertIsNone(validate_interspace_body(
                    raw, rear, (1.0, 1.0, 1.0), vertebra_name=name,
                    diagnostics=diagnostics,
                ))
                self.assertFalse(diagnostics["expensive_validation"])

    def test_tall_large_thin_front_sheet_cannot_displace_short_true_body(self):
        raw = np.zeros((50, 90, 100), dtype=bool)
        raw[2:42, 5:13, 20:80] = True
        body = np.zeros_like(raw)
        body[16:22, 25:40, 35:65] = True
        raw |= body
        self.assertIsNone(validate_interspace_body(
            raw, body, (1.0, 1.0, 1.0), vertebra_name="C5",
        ))

    def test_volume_and_height_alone_cannot_make_a_thin_sheet_a_body(self):
        raw, front, rear = self._conflict()
        front[:] = False
        front[2:39, 10:18, 5:85] = True
        raw = front | rear
        diagnostics = {}
        self.assertIsNone(validate_interspace_body(
            raw, rear, (1.0, 1.0, 1.0), vertebra_name="C7",
            diagnostics=diagnostics,
        ))
        self.assertTrue(diagnostics["expensive_validation"])
        self.assertEqual(diagnostics["reason"], "no_verified_anterior_core")

    def test_fewer_slices_do_not_mean_a_physically_thin_body(self):
        raw, _, rear = self._conflict()
        self.assertIsNone(validate_interspace_body(
            raw, rear, (3.0, 1.0, 1.0), vertebra_name="C7",
        ))

    def test_ambiguous_front_bodies_do_not_choose_by_volume(self):
        raw = np.zeros((65, 150, 90), dtype=bool)
        raw[8:32, 10:24, 20:60] = True
        raw[8:32, 35:49, 20:60] = True
        body = np.zeros_like(raw)
        body[40:48, 80:120, 28:40] = True
        raw |= body
        diagnostics = {}
        self.assertIsNone(validate_interspace_body(
            raw, body, (1.0, 1.0, 1.0), vertebra_name="C7",
            diagnostics=diagnostics,
        ))
        self.assertEqual(diagnostics["reason"], "ambiguous_anterior_cores")

    def test_invalid_or_empty_input_is_non_destructive(self):
        raw, front, _ = self._conflict()
        for body, spacing in ((np.zeros_like(raw), (1, 1, 1)),
                              (front, (1, 0, 1)), (front[:1], (1, 1, 1))):
            with self.subTest(spacing=spacing, shape=body.shape):
                self.assertIsNone(validate_interspace_body(
                    raw, body, spacing, vertebra_name="L1",
                ))


if __name__ == "__main__":
    unittest.main()
