"""普通椎体主体应由中央三维支撑决定，避免先验删除和附件抢占。"""

import unittest

import numpy as np

from src.models import LabelTarget
from src.stages.stage_localization import LocalizationStage


class VertebralBodyCandidateSelectionTest(unittest.TestCase):
    def _body(self, array, label_name):
        return LocalizationStage._vertebral_body_candidate_mask(
            array, LabelTarget(label_name, [5]),
        )

    def test_small_anterior_body_survives_larger_posterior_component(self):
        for label_name in ("vertebrae_C4", "vertebrae_T8", "vertebrae_L3"):
            with self.subTest(label_name=label_name):
                array = np.zeros((40, 70, 70), dtype=np.uint8)
                array[10:25, 10:25, 20:40] = 5
                array[8:30, 35:60, 8:56] = 5

                body = self._body(array, label_name)

                self.assertEqual(int(body[10:25, 10:25, 20:40].sum()), 4500)
                self.assertFalse(np.any(body[:, 35:60, :]))

    def test_branched_posterior_width_does_not_remove_complete_body(self):
        for label_name in ("vertebrae_C6", "vertebrae_T10", "vertebrae_L4"):
            with self.subTest(label_name=label_name):
                array = np.zeros((24, 48, 84), dtype=np.uint8)
                array[4:20, 8:24, 30:54] = 5
                array[6:18, 24:32, 39:45] = 5
                array[6:18, 30:35, 3:9] = 5
                array[6:18, 30:35, 75:81] = 5
                array[6:18, 32:34, 3:81] = 5

                body = self._body(array, label_name)

                self.assertTrue(np.all(body[4:20, 8:24, 30:54]))
                self.assertFalse(np.any(body[:, 30:35, :]))

    def test_small_anterior_protrusion_does_not_replace_complete_body(self):
        for label_name in ("vertebrae_C5", "vertebrae_T9", "vertebrae_L2"):
            with self.subTest(label_name=label_name):
                array = np.zeros((24, 48, 84), dtype=np.uint8)
                array[4:20, 16:32, 30:54] = 5
                array[4:20, 6:12, 32:52] = 5
                array[4:20, 12:16, 41:43] = 5

                body = self._body(array, label_name)

                self.assertTrue(np.all(body[4:20, 16:32, 30:54]))
                self.assertFalse(np.any(body[:, :16, :]))

    def test_skipping_anterior_protrusion_preserves_body_only_terminal_slices(self):
        array = np.zeros((24, 48, 84), dtype=np.uint8)
        array[2:22, 16:32, 30:54] = 5
        array[4:20, 6:12, 32:52] = 5
        array[4:20, 12:16, 41:43] = 5

        body = self._body(array, "vertebrae_L4")

        self.assertTrue(np.all(body[2:22, 16:32, 30:54]))
        self.assertFalse(np.any(body[:, :16, :]))

    def test_tiny_anterior_rows_do_not_remove_supported_body_slices(self):
        array = np.zeros((24, 48, 84), dtype=np.uint8)
        array[2:22, 16:32, 30:54] = 5
        for start, stop in ((4, 8), (16, 20)):
            array[start:stop, 2:3, 30:54] = 5
            array[start:stop, 3:16, 41:43] = 5

        body = self._body(array, "vertebrae_T9")

        self.assertTrue(np.all(body[2:22, 16:32, 30:54]))
        self.assertFalse(np.any(body[:, :16, :]))

    def test_anterior_lateral_fragment_does_not_replace_central_body(self):
        array = np.zeros((12, 40, 80), dtype=np.uint8)
        array[2:10, 15:30, 30:50] = 5
        array[2:10, 14:19, 3:15] = 5

        body = self._body(array, "vertebrae_L3")

        self.assertEqual(int(body.sum()), 2400)
        self.assertFalse(np.any(body[:, :, :15]))

    def test_volume_filter_precedes_si_filter_without_empty_selection(self):
        candidates = np.zeros((120, 60, 80), dtype=bool)
        candidates[0:2, 5:55, :] = True
        candidates[10:110, 15:25, 30:39] = True

        selected = LocalizationStage._select_vertebral_body_support(candidates)

        self.assertGreater(int(selected.sum()), 0)
        self.assertTrue(np.all(selected <= candidates))

    def test_tall_thin_fragment_does_not_set_required_body_height(self):
        array = np.zeros((36, 40, 80), dtype=np.uint8)
        array[12:20, 15:30, 30:50] = 5
        array[2:32, 5:8, 38:42] = 5

        body = self._body(array, "vertebrae_T8")

        self.assertEqual(int(body.sum()), 2400)
        self.assertFalse(np.any(body[:, :8]))

    def test_short_sparse_remnant_cannot_define_a_body_prism(self):
        for height, expected_count in ((8, 0), (24, 4800)):
            with self.subTest(height=height):
                array = np.zeros((36, 70, 90), dtype=np.uint8)
                array[12:12 + height, 10:20, 30:50] = 5
                array[:32, 30:60, :10] = 5
                array[:32, 30:60, 70:80] = 5
                array[15, 19:31, 38:42] = 5
                array[15, 30, :80] = 5

                body = self._body(array, "vertebrae_L3")

                self.assertEqual(int(body.sum()), expected_count)
                self.assertFalse(np.any(body[:, 30:60]))

    def test_local_bbox_and_padded_input_select_identical_body(self):
        array = np.zeros((24, 48, 84), dtype=np.uint8)
        array[4:20, 16:32, 30:54] = 5
        array[4:20, 6:12, 32:52] = 5
        array[4:20, 12:16, 41:43] = 5
        minimum = np.argwhere(array).min(axis=0)
        maximum = np.argwhere(array).max(axis=0)
        region = tuple(slice(int(low), int(high) + 1) for low, high in zip(minimum, maximum))

        full = self._body(array, "vertebrae_T8")
        local = self._body(array[region], "vertebrae_T8")

        np.testing.assert_array_equal(full[region], local)
        self.assertEqual(int(full.sum()), int(local.sum()))

    def test_absent_label_returns_none(self):
        self.assertIsNone(self._body(np.zeros((4, 6, 8), dtype=np.uint8), "vertebrae_C4"))


if __name__ == "__main__":
    unittest.main()
