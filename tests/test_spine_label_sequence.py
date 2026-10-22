"""Sequence recovery requires CT support and preserves original foreground."""

import unittest

import numpy as np

from src.stages.spine_label_sequence import repair_lumbar_body_sequence


class LumbarBodySequenceTests(unittest.TestCase):
    def _case(self, *, false_slot=False, genuine_gap=False):
        # Deliberately unordered, nonconsecutive IDs exercise semantic resolution.
        labels = {"vertebrae_T11": [41], "vertebrae_L2": [13],
                  "vertebrae_S1": [2], "vertebrae_L5": [7],
                  "vertebrae_L4": [11], "vertebrae_L3": [17],
                  "vertebrae_L1": [23], "vertebrae_T12": [31]}
        seg = np.zeros((132, 44, 40), dtype=np.uint8)
        raw = np.full(seg.shape, 20, dtype=np.float32)
        names = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11"]
        center = 10
        for name in names:
            label = labels["vertebrae_" + name][0]
            if name == "L3" and false_slot:
                seg[46, 8:22, 10:20] = label  # cap shares LR projection with L4
                seg[50:58, 34:38, 17:23] = label  # posterior element
                if genuine_gap:
                    center += 16
                continue
            seg[center-5:center+6, 8:22, 10:30] = label
            raw[center-5:center+6, 8:22, 10:30] = 200
            seg[center-4:center+5, 34:38, 17:23] = label
            center += 16
        # A nonvertebral foreground label must never be reassigned.
        seg[40:43, 3:5, 3:5] = 99
        return seg, raw, labels

    @staticmethod
    def _extract(array, target):
        mask = np.isin(array, target.label_ids)
        # Model the anterior body extractor without importing the pipeline.
        # Each local crop starts at anterior y=8; posterior elements are 26 mm away.
        occupied = np.argwhere(mask)
        if not len(occupied):
            return None
        mask[:, int(occupied[:, 1].min()) + 16:, :] = False
        points = np.argwhere(mask)
        if len(points) == 0 or np.ptp(points[:, 0]) < 3:
            return None
        return mask

    def _repair(self, seg, raw, labels, **kwargs):
        return repair_lumbar_body_sequence(
            seg, raw, {"spacing_xyz": [1, 1, 1]}, labels,
            {"vertebrae_T12": {}}, kwargs.get("extractor", self._extract),
        )

    def test_normal_sequence_returns_before_loading_ct(self):
        seg, _, labels = self._case()
        def unexpected_loader():
            self.fail("normal sequence must not load CT")
        result, diagnostics = self._repair(seg, unexpected_loader, labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "unchanged")

    def test_false_slot_restores_anterior_sequence_and_keeps_posterior(self):
        seg, raw, labels = self._case(false_slot=True)
        original = seg.copy()
        result, diagnostics = self._repair(seg, lambda: raw, labels)
        self.assertIsNotNone(result, diagnostics)
        self.assertEqual(diagnostics["status"], "repaired")
        np.testing.assert_array_equal(seg, original)
        np.testing.assert_array_equal(result != 0, seg != 0)
        np.testing.assert_array_equal(result[:, 32:, :], seg[:, 32:, :])
        np.testing.assert_array_equal(result == 99, seg == 99)
        self.assertEqual(result[58, 15, 20], labels["vertebrae_L3"][0])
        self.assertEqual(result[74, 15, 20], labels["vertebrae_L2"][0])
        self.assertEqual(result[90, 15, 20], labels["vertebrae_L1"][0])
        self.assertEqual(result[106, 15, 20], labels["vertebrae_T12"][0])
        self.assertEqual(result[46, 15, 15], labels["vertebrae_L4"][0])

    def test_true_missing_body_with_two_body_steps_is_not_compressed(self):
        seg, raw, labels = self._case(false_slot=True, genuine_gap=True)
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "skipped")

    def test_no_ct_disc_valley_rejects_reassociation(self):
        seg, raw, labels = self._case(false_slot=True)
        raw[:] = 200
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["ct_evidence"]["valley_count"], 0)

    def test_mismatching_ct_shape_is_skipped(self):
        seg, raw, labels = self._case(false_slot=True)
        result, diagnostics = self._repair(seg, raw[:-1], labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "skipped")
        self.assertIn("形状", diagnostics["reason"])

    def test_loader_geometry_error_is_skipped(self):
        seg, _, labels = self._case(false_slot=True)
        def invalid_geometry():
            raise ValueError("原图与分割物理坐标不匹配")
        result, diagnostics = self._repair(seg, invalid_geometry, labels)
        self.assertIsNone(result)
        self.assertIn("物理坐标不匹配", diagnostics["reason"])

    def test_missing_sacral_anchor_prevents_automatic_repair(self):
        seg, raw, labels = self._case(false_slot=True)
        seg[seg == labels["vertebrae_S1"][0]] = 0
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "skipped")

    def test_shared_label_ids_are_ambiguous(self):
        seg, raw, labels = self._case(false_slot=True)
        labels["vertebrae_L1"] = labels["vertebrae_L2"]
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "skipped")

    def test_target_id_must_fit_original_segmentation_dtype(self):
        seg, raw, labels = self._case(false_slot=True)
        labels["vertebrae_L3"] = [300]
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result)
        self.assertIn("数据类型范围", diagnostics["reason"])

    def test_empty_volume_is_skipped(self):
        _, raw, labels = self._case(false_slot=True)
        result, diagnostics = self._repair(np.zeros((0, 2, 2), dtype=np.uint8), raw, labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "skipped")

    def test_invalid_image_spacing_is_skipped(self):
        seg, raw, labels = self._case(false_slot=True)
        result, diagnostics = repair_lumbar_body_sequence(
            seg, raw, {}, labels, body_extractor=self._extract,
        )
        self.assertIsNone(result)
        self.assertIn("间距", diagnostics["reason"])

    def test_uint64_and_unrelated_sparse_id_do_not_change_normal_sequence(self):
        seg, raw, labels = self._case()
        seg = seg.astype(np.uint64)
        seg[0, 0, 0] = 2**40
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "unchanged")

    def test_projected_cap_without_real_three_dimensional_contact_cannot_reindex(self):
        seg, raw, labels = self._case(false_slot=True)
        # A channel in the predicted label leaves the cap laterally detached.
        # Its YZ projection still overlaps L4 and CT still has one disc valley.
        # Actual 3D evidence must veto relabeling the upstream body sequence.
        seg[:, :, 16:24] = 0
        slot_id = labels["vertebrae_L3"][0]
        seg[46, 8:22, :] = np.where(seg[46, 8:22, :] == slot_id, 0, seg[46, 8:22, :])
        seg[46, 8:22, 17:23] = slot_id
        original = seg.copy()
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result, diagnostics)
        self.assertEqual(diagnostics["ct_evidence"]["valley_count"], 1)
        self.assertEqual(diagnostics["status"], "skipped")
        self.assertIn("三维", diagnostics["reason"])
        self.assertTrue(diagnostics["cap_components"])
        self.assertFalse(any(component["status"] == "reassociated"
                             for component in diagnostics["cap_components"]))
        np.testing.assert_array_equal(seg, original)

    def test_superior_body_must_exist_to_restore_requested_t12(self):
        seg, raw, labels = self._case(false_slot=True)
        seg[seg == labels["vertebrae_T11"][0]] = 0
        result, diagnostics = self._repair(seg, raw, labels)
        self.assertIsNone(result)
        self.assertIn("上方独立前体不足", diagnostics["reason"])


if __name__ == "__main__":
    unittest.main()
