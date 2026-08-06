import ast
import json
import subprocess
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
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--jobs N", result.stdout)

    def test_jobs_must_be_positive_integer(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--jobs", "0", "--dry-run"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
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
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--plane_configs", result.stdout)
        self.assertIn("横状位", result.stdout)
        self.assertIn("矢状位", result.stdout)
        self.assertIn("冠状位", result.stdout)

    def test_plane_configs_value_is_valid_json(self):
        assignment = next(
            line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
            if line.startswith("PLANE_CONFIGS=")
        )
        value = ast.literal_eval(assignment.split("=", 1)[1])

        configs = json.loads(value)

        self.assertEqual(
            [config["plane"] for config in configs],
            ["横状位", "矢状位", "冠状位"],
        )
        for config in configs:
            self.assertEqual(config["slice_thickness"], 5)
            self.assertEqual(config["slice_spacing"], 5)
            self.assertEqual(
                config["window_presets"],
                ["骨窗", "肺窗", "纵隔窗", "软组织窗"],
            )


if __name__ == "__main__":
    unittest.main()
