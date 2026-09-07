import json
import os
import shlex
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

    def test_cervical_dry_run_keeps_both_axial_configs(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        command_line = next(
            line.removeprefix("命令:")
            for line in result.stdout.splitlines()
            if line.startswith("命令:") and "--slice_part 颈椎" in line
        )
        command = shlex.split(command_line)
        payload = json.loads(command[command.index("--plane_configs") + 1])

        self.assertEqual(
            [
                (
                    config["Plane"],
                    config["SliceType"],
                    config["SliceThickness"],
                    config["SliceSpacing"],
                )
                for config in payload
            ],
            [
                ("横状位", "MPR", 5.0, 5.0),
                ("横状位", "椎间隙", 3.0, 3.0),
                ("矢状位", "MPR", 5.0, 5.0),
                ("冠状位", "CPR", 5.0, 5.0),
            ],
        )

    def test_slice_type_configs_cover_batch_parts_and_planes(self):
        script_text = SCRIPT.read_text(encoding="utf-8")
        start = script_text.index("SLICE_TYPE_CONFIGS=(")
        end = script_text.index("\n)", start)
        entries = [
            line.strip().strip('"')
            for line in script_text[start:end].splitlines()[1:]
            if line.strip()
        ]
        mapping = {}
        for entry in entries:
            slice_part, plane, slice_type, thickness, spacing = entry.split("|")
            mapping.setdefault((slice_part, plane), []).append(
                (slice_type, float(thickness), float(spacing))
            )

        self.assertEqual(
            set(mapping),
            {
                (slice_part, plane)
                for slice_part in ("颈椎", "胸椎", "腰椎", "肺部", "髋骨")
                for plane in ("横状位", "矢状位", "冠状位")
            },
        )
        self.assertEqual(
            mapping[("颈椎", "横状位")],
            [("MPR", 5.0, 5.0), ("椎间隙", 3.0, 3.0)],
        )
        self.assertEqual(mapping[("颈椎", "冠状位")], [("CPR", 5.0, 5.0)])
        self.assertEqual(
            mapping[("腰椎", "横状位")],
            [("MPR", 5.0, 5.0), ("椎间隙", 5.0, 5.0)],
        )
        self.assertEqual(mapping[("腰椎", "冠状位")], [("CPR", 5.0, 5.0)])
        self.assertEqual(mapping[("胸椎", "冠状位")], [("CPR", 5.0, 5.0)])
        self.assertEqual(mapping[("肺部", "冠状位")], [("MPR", 5.0, 5.0)])
        self.assertEqual(mapping[("髋骨", "冠状位")], [("MPR", 5.0, 5.0)])

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
