"""成功交付仍应说明因椎间几何不足而跳过的区域。"""

import unittest

from src.core.context import ProcessingContext
from src.models import Request
from src.stages.stage_reporting import ReportingStage


class InterspaceQualityReportingTest(unittest.TestCase):
    def _context(self, warnings=None):
        request = Request.from_dict({
            "dicom_dir": "/tmp/dicom",
            "output_dir": "/tmp/output",
            "ScanThickness": 1.5,
            "plane_configs": [{
                "Plane": "横状位",
                "SliceType": "椎间隙",
                "SliceThickness": 3,
                "SliceSpacing": 3,
                "WindowPresets": ["骨窗"],
            }],
        })
        context = ProcessingContext(request)
        if warnings is not None:
            context.set_result("planning", {"warnings": warnings})
        context.set_result("reconstruction", {"outputs": [{
            "plane": "横状位",
            "slice_type": "椎间隙",
            "group_name": "L1_L2",
            "slice_thickness": 3,
            "slice_spacing": 3,
            "slice_count": 4,
            "window_outputs": {"骨窗": "/tmp/output/横状位/骨窗/L1_L2"},
        }]})
        return context

    def test_normal_and_ordinary_warnings_keep_success_message_empty(self):
        for warnings in (None, [], ["脊柱区间 L1_L2 缺少可靠左右对称轴，保留原椎间方向"]):
            with self.subTest(warnings=warnings):
                report = ReportingStage()._build_delivery_report(self._context(warnings))
                self.assertTrue(report["Success"])
                self.assertEqual(report["Message"], "")

    def test_skipped_interspaces_are_deduplicated_without_changing_report_contract(self):
        first = "跳过脊柱区间 L2_L3: 主体几何不足，无法形成可靠椎间切面"
        second = "跳过脊柱区间 L3_L4: 主体几何不足，无法形成可靠椎间切面"
        context = self._context([
            first,
            "脊柱区间 L1_L2 缺少可靠左右对称轴，保留原椎间方向",
            second,
            first,
        ])

        report = ReportingStage()._build_delivery_report(context)

        self.assertTrue(report["Success"])
        self.assertEqual(report["Message"], first + "；" + second)
        self.assertEqual(set(report), {
            "Success", "Message", "SlicePart", "StudyType", "ScanThickness", "PlaneList",
        })
        regions = report["PlaneList"][0]["WindowPresetList"][0]["RegionList"]
        self.assertEqual([region["Region"] for region in regions], ["L1_L2"])
        self.assertEqual(regions[0]["ActualSliceImageCount"], 4)


if __name__ == "__main__":
    unittest.main()
