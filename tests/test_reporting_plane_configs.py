import json
import tempfile
import unittest
from pathlib import Path

from src.core.context import ProcessingContext
from src.models import Request
from src.stages.stage_reporting import ReportingStage


class ReportingPlaneConfigsTest(unittest.TestCase):
    def test_report_request_contains_plane_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = Request.from_dict({
                "dicom_dir": "/tmp/dicom",
                "output_dir": temp_dir,
                "plane_configs": [
                    {
                        "plane": "冠状位",
                        "slice_thickness": 5,
                        "slice_spacing": 10,
                        "window_presets": ["骨窗", "纵隔窗"],
                    }
                ],
            })
            context = ProcessingContext(request)

            report_path = ReportingStage()._generate_report(context)

            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertEqual(report["request"]["plane_configs"], [
                {
                    "plane": "冠状位",
                    "slice_thickness": 5.0,
                    "slice_spacing": 10.0,
                    "window_presets": ["骨窗", "纵隔窗"],
                }
            ])


if __name__ == "__main__":
    unittest.main()
