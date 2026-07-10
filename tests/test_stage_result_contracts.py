import unittest

from src.core.context import ProcessingContext
from src.core.errors import StageInputError
from src.models import Request
from src.models.stage_results import (
    PreprocessingResult,
    SegmentationResult,
)


class StageResultContractsTest(unittest.TestCase):
    def setUp(self):
        self.context = ProcessingContext(
            Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")
        )

    def test_typed_result_is_available_through_new_and_legacy_interfaces(self):
        result = PreprocessingResult(nifti_path="/tmp/raw.nii.gz")

        self.context.set_stage_result(result)

        self.assertIs(
            self.context.require_stage_result(
                "preprocessing", PreprocessingResult
            ),
            result,
        )
        self.assertEqual(
            self.context.get_result("preprocessing")["nifti_path"],
            "/tmp/raw.nii.gz",
        )

    def test_require_stage_result_rejects_wrong_type(self):
        self.context.set_stage_result(
            PreprocessingResult(nifti_path="/tmp/raw.nii.gz")
        )

        with self.assertRaises(StageInputError):
            self.context.require_stage_result(
                "preprocessing", SegmentationResult
            )

    def test_result_serialization_has_common_contract_fields(self):
        result = SegmentationResult(
            seg_path="/tmp/seg.nii.gz",
            input_nifti="/tmp/raw.nii.gz",
            model={"model_id": 292},
            warnings=["模型告警"],
        )

        data = result.to_dict()

        self.assertEqual(data["schema_version"], "segmentation_v1")
        self.assertEqual(data["stage_name"], "segmentation")
        self.assertEqual(data["warnings"], ["模型告警"])
        self.assertEqual(data["model"]["model_id"], 292)


if __name__ == "__main__":
    unittest.main()
