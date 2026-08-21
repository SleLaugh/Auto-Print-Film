import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "batch_test_plan.sh"


class BatchTestPlanScriptTest(unittest.TestCase):
    def test_help_documents_jobs_option(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--jobs N", result.stdout)

    def test_each_case_log_is_written_beside_report(self):
        script_text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('log_dir="$output_dir/log"', script_text)
        self.assertIn('log_file="$log_dir/BatchLog.txt"', script_text)
        self.assertNotIn("_batch_logs", script_text)

    def test_jobs_must_be_positive_integer(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--jobs", "0", "--dry-run"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--jobs 必须是正整数", result.stderr)

    def test_dry_run_requests_all_three_planes(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--plane_configs", result.stdout)
        self.assertIn("横状位", result.stdout)
        self.assertIn("矢状位", result.stdout)
        self.assertIn("冠状位", result.stdout)
        self.assertNotIn("line_extent_mode", result.stdout)
        self.assertIn("SliceType", result.stdout)
        self.assertIn("MPR", result.stdout)
        self.assertIn("CPR", result.stdout)
        self.assertIn("椎间隙", result.stdout)
        self.assertNotIn("full_plane", result.stdout)

    def test_slice_type_configs_cover_batch_parts_and_planes(self):
        script_text = SCRIPT.read_text(encoding="utf-8")
        start = script_text.index("SLICE_TYPE_CONFIGS=(")
        end = script_text.index("\n)", start)
        entries = [
            line.strip().strip('"')
            for line in script_text[start:end].splitlines()[1:]
            if line.strip()
        ]
        mapping = {
            (study_part, plane): slice_type
            for study_part, plane, slice_type in (
                entry.split("|") for entry in entries
            )
        }

        self.assertEqual(
            set(mapping),
            {
                (study_part, plane)
                for study_part in ("颈椎", "胸椎", "腰椎", "肺部", "髋骨")
                for plane in ("横状位", "矢状位", "冠状位")
            },
        )
        self.assertEqual(mapping[("腰椎", "横状位")], "椎间隙")
        self.assertEqual(mapping[("腰椎", "冠状位")], "CPR")
        self.assertEqual(mapping[("颈椎", "冠状位")], "CPR")
        self.assertEqual(mapping[("胸椎", "冠状位")], "CPR")
        self.assertEqual(mapping[("肺部", "冠状位")], "MPR")
        self.assertEqual(mapping[("髋骨", "冠状位")], "MPR")

    def test_chinese_extent_mode_parses_under_c_locale_with_utf8_mode(self):
        """精简 Linux 的 C locale 下，显式 UTF-8 模式仍应正确解析中文请求。"""

        payload = json.dumps({
            "Plane": "横状位",
            "SliceThickness": 5,
            "SliceSpacing": 5,
            "WindowPresets": ["骨窗"],
            "line_extent_mode": "整平面",
        }, ensure_ascii=False)
        environment = dict(os.environ)
        environment.update({
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONUTF8": "1",
        })
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys; "
                    "from src.models import PlaneConfig; "
                    "print(PlaneConfig.from_dict(json.loads(sys.argv[1]))"
                    ".line_extent_mode.value)"
                ),
                payload,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "整平面")
        self.assertIn(
            "export PYTHONUTF8=1",
            SCRIPT.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
