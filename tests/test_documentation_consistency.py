import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote

from src.planning.spine_cpr import (
    SPINE_CPR_LABELS,
    configured_spine_cpr_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
SUPERPOWERS_ROOT = DOCS_ROOT / "superpowers"


class DocumentationConsistencyTests(unittest.TestCase):
    """保证当前文档入口与运行时代码表达同一套业务规则。"""

    def test_configured_spine_cpr_ranges_match_runtime_contract(self):
        """胸腰椎按配置全量使用，只有颈椎从 CPR 中排除 C1。"""

        config_path = PROJECT_ROOT / "src/config/localization_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        aliases = config["label_aliases"]

        thoracic_range = config["targets"]["胸椎"]["whole_range"]
        thoracic_labels = configured_spine_cpr_labels(
            "胸椎",
            {
                "resolved_cut_config": {
                    "start_labels": [aliases[thoracic_range["start"]]],
                    "end_labels": [aliases[thoracic_range["end"]]],
                }
            },
            SPINE_CPR_LABELS["胸椎"],
        )
        self.assertEqual(
            thoracic_labels,
            (
                ["vertebrae_C7"]
                + [f"vertebrae_T{index}" for index in range(1, 13)]
                + ["vertebrae_L1"]
            ),
        )

        cervical_range = config["targets"]["颈椎"]["whole_range"]
        cervical_labels = configured_spine_cpr_labels(
            "颈椎",
            {
                "resolved_cut_config": {
                    "start_labels": [aliases[cervical_range["start"]]],
                    "end_labels": [aliases[cervical_range["end"]]],
                }
            },
            SPINE_CPR_LABELS["颈椎"],
        )
        self.assertEqual(
            cervical_labels,
            [f"vertebrae_C{index}" for index in range(2, 8)],
        )

    def test_current_rule_documents_do_not_restore_known_stale_claims(self):
        """当前规则文档不能重新写入已被代码替代的固定范围或状位结论。"""

        forbidden_statements = (
            "当前冠状位仍为直线 MPR，曲线 CPR 尚未实现",
            "胸椎任务的输出目标段只使用 T1 到 T12",
            "C7/L1 不进入 CPR 曲线",
            "胸椎：T1 到 T12。",
            "矢状位和冠状位生成 CPR",
            "不再执行额外 Hermite 平滑",
            "debug_artifacts_enabled=true",
        )
        current_documents = [PROJECT_ROOT / "README.md"]
        current_documents.extend(
            path
            for path in SUPERPOWERS_ROOT.rglob("*.md")
            if path.is_file()
            and "archive" not in path.relative_to(SUPERPOWERS_ROOT).parts
            and "plans" not in path.relative_to(SUPERPOWERS_ROOT).parts
            and path.name
            not in {
                "2026-07-29-文档与代码一致性审阅设计.md",
                "2026-07-29-全量文档审阅记录.md",
            }
        )
        for path in current_documents:
            text = path.read_text(encoding="utf-8")
            for statement in forbidden_statements:
                self.assertNotIn(
                    statement,
                    text,
                    msg=f"{path.relative_to(PROJECT_ROOT)} 仍包含过时结论: {statement}",
                )

    def test_current_spine_summaries_declare_runtime_contract(self):
        """定位和重建入口必须直接写出配置范围、C1 例外和实际 CPR 状位。"""

        localization_readme = (
            SUPERPOWERS_ROOT / "modules/localization/README.md"
        ).read_text(encoding="utf-8")
        reconstruction_readme = (
            SUPERPOWERS_ROOT / "modules/reconstruction/README.md"
        ).read_text(encoding="utf-8")

        self.assertIn("C1 是唯一固定排除项", localization_readme)
        self.assertIn("胸椎 C7/L1 等其他配置首尾标签正常参与", localization_readme)
        self.assertIn("C7、T1–T12、L1", reconstruction_readme)
        self.assertIn("脊柱冠状位 CPR", reconstruction_readme)
        self.assertIn("脊柱矢状位保持既有直线 MPR", reconstruction_readme)

    def test_current_plan_and_spec_directories_only_contain_current_work(self):
        """完成计划必须归档，旧 specs 路径不能继续保存具体规范。"""

        plans_dir = SUPERPOWERS_ROOT / "plans"
        for path in plans_dir.glob("*.md"):
            if path.name == "README.md":
                continue
            text = path.read_text(encoding="utf-8")
            checkboxes = re.findall(r"^- \[([ xX])\]", text, flags=re.MULTILINE)
            if checkboxes:
                self.assertTrue(
                    any(value == " " for value in checkboxes),
                    msg=f"已完成计划仍位于 plans/: {path.name}",
                )

        specs = [
            path.name
            for path in (SUPERPOWERS_ROOT / "specs").glob("*.md")
            if path.name != "README.md"
        ]
        self.assertEqual(specs, [], msg=f"具体规范仍位于旧 specs/ 路径: {specs}")

    def test_module_documents_do_not_contain_implementation_plans(self):
        """模块目录只保留当前设计，带执行清单的计划必须进入归档。"""

        misplaced = []
        for path in (SUPERPOWERS_ROOT / "modules").rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            if "实施计划" in text.splitlines()[0] or re.search(
                r"^- \[[ xX]\]",
                text,
                flags=re.MULTILINE,
            ):
                misplaced.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(misplaced, [])

    def test_all_markdown_relative_links_exist(self):
        """全部当前和历史文档的相对 Markdown 链接都必须解析到真实目标。"""

        link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        missing_links = []
        for path in [PROJECT_ROOT / "README.md", *DOCS_ROOT.rglob("*.md")]:
            text = path.read_text(encoding="utf-8")
            for raw_target in link_pattern.findall(text):
                target = raw_target.strip().strip("<>")
                if (
                    not target
                    or target.startswith(("#", "http://", "https://", "mailto:"))
                ):
                    continue
                target = unquote(target.split("#", 1)[0])
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    missing_links.append(
                        f"{path.relative_to(PROJECT_ROOT)} -> {target}"
                    )
        self.assertEqual(missing_links, [])


if __name__ == "__main__":
    unittest.main()
