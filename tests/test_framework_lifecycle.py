import unittest

from src.core.errors import CapabilityError
from src.core.lifecycle import StageRecord, StageStatus


class FrameworkLifecycleTest(unittest.TestCase):
    def test_framework_error_serializes_stable_fields(self):
        error = CapabilityError(
            "不支持的能力",
            details={"body_part": "颈椎"},
        )

        self.assertEqual(error.to_dict(), {
            "code": "CAPABILITY_ERROR",
            "message": "不支持的能力",
            "stage": None,
            "retryable": False,
            "details": {"body_part": "颈椎"},
        })

    def test_stage_record_tracks_terminal_state(self):
        record = StageRecord("preprocessing")

        record.start()
        record.succeed("preprocessing_v1")

        self.assertIs(record.status, StageStatus.SUCCEEDED)
        self.assertIsNotNone(record.duration_seconds)
        self.assertEqual(record.result_schema_version, "preprocessing_v1")

    def test_stage_record_rejects_invalid_transition(self):
        record = StageRecord("preprocessing")

        with self.assertRaisesRegex(ValueError, "pending.*succeeded"):
            record.succeed("preprocessing_v1")


if __name__ == "__main__":
    unittest.main()
