import unittest
from unittest.mock import patch

from src.core.context import ProcessingContext
from src.models import PlanningStageResult, Request
from src.planning.base import PlanningResult
from src.stages.stage_planning import PlanningStage


class _RecordingPlanner:
    planner_name = "RecordingPlanner"

    def __init__(self):
        self.localization_result = None

    def build_plans(self, request, localization_result):
        self.localization_result = localization_result
        return PlanningResult(planner_name=self.planner_name)


class StageSplitContractTest(unittest.TestCase):
    def test_planning_requires_anatomical_analysis(self):
        """八阶段主链不能跳过解剖分析直接进入规划。"""

        context = ProcessingContext(
            Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")
        )
        context.set_result("localization", {"coords": {}})

        with self.assertRaisesRegex(ValueError, "anatomical_analysis"):
            PlanningStage().execute(context)

    def test_planning_merges_localization_and_anatomical_analysis(self):
        """规划阶段应向部位 planner 提供一份完整的展平输入。"""

        context = ProcessingContext(
            Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")
        )
        context.set_result("localization", {
            "coords": {"vertebrae_L1": {"found": True}},
        })
        context.set_result("anatomical_analysis", {
            "spine_geometry": {"method": "body_prism"},
        })
        planner = _RecordingPlanner()

        with patch(
            "src.stages.stage_planning.select_planner",
            return_value=planner,
        ), patch(
            "src.stages.stage_planning.is_debug_file_enabled",
            return_value=False,
        ):
            PlanningStage().execute(context)

        self.assertEqual(
            planner.localization_result["spine_geometry"]["method"],
            "body_prism",
        )
        self.assertIn("plans", context.get_result("planning"))
        self.assertIsInstance(
            context.require_stage_result("planning", PlanningStageResult),
            PlanningStageResult,
        )


if __name__ == "__main__":
    unittest.main()
