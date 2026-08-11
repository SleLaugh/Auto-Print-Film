import unittest

from src.core.context import ProcessingContext
from src.core.errors import StageInputError
from src.models import Request
from src.models.stage_results import (
    AnatomicalAnalysisStageResult,
    PlanningStageResult,
    PreprocessingResult,
    SegmentationResult,
    SpatialCorrectionStageResult,
)


class StageResultContractsTest(unittest.TestCase):
    def setUp(self):
        self.context = ProcessingContext(
            Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")
        )

    def test_typed_and_serialized_result_views_are_both_available(self):
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

    def test_split_stage_results_expose_payload_fields_at_result_top_level(self):
        """拆分后的业务字段应通过 ProcessingContext 顶层读取。"""

        cases = [
            (
                SpatialCorrectionStageResult(
                    payload={"raw_for_reconstruction_path": "/tmp/corrected.nii.gz"}
                ),
                "spatial_correction",
                "spatial_correction_v1",
                "raw_for_reconstruction_path",
            ),
            (
                AnatomicalAnalysisStageResult(
                    payload={"spine_geometry": {"method": "body_prism"}}
                ),
                "anatomical_analysis",
                "anatomical_analysis_v1",
                "spine_geometry",
            ),
            (
                PlanningStageResult(payload={"plans": [{"plan_id": "one"}]}),
                "planning",
                "planning_v1",
                "plans",
            ),
        ]

        for result, stage_name, schema_version, payload_key in cases:
            with self.subTest(stage_name=stage_name):
                self.context.set_stage_result(result)
                serialized = result.to_dict()
                self.assertEqual(serialized["stage_name"], stage_name)
                self.assertEqual(serialized["schema_version"], schema_version)
                self.assertIn(payload_key, serialized)
                self.assertIn(payload_key, self.context.get_result(stage_name))


if __name__ == "__main__":
    unittest.main()
