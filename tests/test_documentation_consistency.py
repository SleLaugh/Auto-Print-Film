import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote

from src.config import localization_profiles_for
from src.planning.spine_cpr import configured_spine_cpr_labels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
SUPERPOWERS_ROOT = DOCS_ROOT / "superpowers"


class DocumentationConsistencyTests(unittest.TestCase):
    """保证当前文档入口与运行时代码表达同一套业务规则。"""

    def test_current_docs_use_slice_part_protocol_names(self):
        """当前文档不得重新使用已移除的检查部位协议名称。"""

        current_documents = [PROJECT_ROOT / "README.md"]
        current_documents.extend(
            path
            for path in SUPERPOWERS_ROOT.rglob("*.md")
            if path.is_file()
            and "archive" not in path.relative_to(SUPERPOWERS_ROOT).parts
        )
        removed_names = (
            re.compile(r"(?<![A-Za-z0-9_])StudyPart(?![A-Za-z0-9_])"),
            re.compile(r"--study_part(?![A-Za-z0-9_])"),
            re.compile(r"(?<![A-Za-z0-9_])study_part_config(?![A-Za-z0-9_])"),
            re.compile(r"(?<![A-Za-z0-9_])study_parts(?![A-Za-z0-9_])"),
        )
        for path in current_documents:
            text = path.read_text(encoding="utf-8")
            for pattern in removed_names:
                self.assertIsNone(
                    pattern.search(text),
                    msg=(
                        f"{path.relative_to(PROJECT_ROOT)} 仍包含已移除名称: "
                        f"{pattern.pattern}"
                    ),
                )

    def test_configured_spine_cpr_ranges_match_runtime_contract(self):
        """脊柱 CPR 按统一配置展开范围并应用标签排除项。"""

        config_path = PROJECT_ROOT / "src/config/slice_part_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        aliases = config["label_aliases"]

        thoracic_range = localization_profiles_for(config, "胸椎")["whole_range"]
        thoracic_labels = configured_spine_cpr_labels(
            {
                "resolved_cut_configs": {
                    "whole_range": {
                        "start_labels": [aliases[thoracic_range["start"]]],
                        "end_labels": [aliases[thoracic_range["end"]]],
                    },
                },
                "spine_cpr_config": {"excluded_labels": []},
            },
        )
        self.assertEqual(
            thoracic_labels,
            (
                ["vertebrae_C7"]
                + [f"vertebrae_T{index}" for index in range(1, 13)]
                + ["vertebrae_L1"]
            ),
        )

        cervical_range = localization_profiles_for(config, "颈椎")["whole_range"]
        excluded = config["slice_parts"]["颈椎"]["localization"]["spine_cpr"][
            "excluded_labels"
        ]
        cervical_labels = configured_spine_cpr_labels(
            {
                "resolved_cut_configs": {
                    "whole_range": {
                        "start_labels": [aliases[cervical_range["start"]]],
                        "end_labels": [aliases[cervical_range["end"]]],
                    },
                },
                "spine_cpr_config": {
                    "excluded_labels": [aliases[name] for name in excluded],
                },
            },
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
            "CPR 曲面厚层融合尚未实现",
            "当前实现应观测到多个写入器重叠",
            "CPR 规划是否降级由第六阶段决定",
            "任一计划失败时不应发布该计划的半成品正式 DICOM",
            "胸椎横状位按临床规则保持",
            "使用投影到切片平面的 DICOM 头脚轴",
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

    def test_pose_documentation_covers_all_parts_and_quality_fallback(self):
        """统一姿态说明必须包含全部部位、保留俯仰和质量回退。"""

        text = (
            SUPERPOWERS_ROOT / "modules/reconstruction"
            / "2026-10-16-统一解剖姿态与三状位回正设计.md"
        ).read_text(encoding="utf-8")
        for part in ("头骨", "髋骨", "肺部", "颈椎", "胸椎", "腰椎"):
            self.assertIn(part, text)
        self.assertIn("保留", text)
        self.assertIn("前后倾角", text)
        self.assertIn("质量", text)
        self.assertIn("回退", text)
        self.assertIn("scipy", text)

    def test_current_spine_summaries_declare_runtime_contract(self):
        """定位和重建入口必须写出配置范围、C1 配置和实际 CPR 状位。"""

        localization_readme = (
            SUPERPOWERS_ROOT / "modules/localization/README.md"
        ).read_text(encoding="utf-8")
        reconstruction_readme = (
            SUPERPOWERS_ROOT / "modules/reconstruction/README.md"
        ).read_text(encoding="utf-8")

        self.assertIn("spine_cpr.excluded_labels", localization_readme)
        self.assertIn("排除项和输出间隙", localization_readme)
        self.assertIn("不在代码中固定", localization_readme)
        self.assertIn("C7、T1–T12、L1", reconstruction_readme)
        self.assertIn("脊柱冠状位 CPR", reconstruction_readme)
        self.assertIn("脊柱矢状位保持既有直线 MPR", reconstruction_readme)

    def test_current_docs_describe_prism_intrusion_fallback(self):
        """相邻棱柱文档必须保留小侵入容差和三维主体冲突边界。"""

        localization_readme = (
            SUPERPOWERS_ROOT / "modules/localization/README.md"
        ).read_text(encoding="utf-8")
        prism_doc = (
            SUPERPOWERS_ROOT
            / "modules"
            / "localization"
            / "2026-08-06-椎体主体外包络四棱柱设计.md"
        ).read_text(encoding="utf-8")

        self.assertIn("各不超过 `5%`", localization_readme)
        self.assertIn("占用同一体素", prism_doc)
        self.assertIn("结果不能随字典遍历", prism_doc)

    def test_current_docs_describe_stage8_hard_output_gate(self):
        """DICOM 文档必须说明第八阶段先校验后发布的失败契约。"""

        dicom_readme = (
            SUPERPOWERS_ROOT / "modules/dicom/README.md"
        ).read_text(encoding="utf-8")
        dicom_design = (
            SUPERPOWERS_ROOT
            / "modules"
            / "dicom"
            / "2026-07-14-DICOM数字命名与定位图设计.md"
        ).read_text(encoding="utf-8")
        architecture_readme = (
            SUPERPOWERS_ROOT / "architecture/README.md"
        ).read_text(encoding="utf-8")

        self.assertIn("最终输出硬门禁", dicom_readme)
        self.assertIn("OUTPUT_VALIDATION_ERROR", dicom_design)
        self.assertIn("校验通过后才在 `report.json`", dicom_design)
        self.assertIn("`Success=false`", dicom_design)
        self.assertIn("`kill -9`", architecture_readme)
        self.assertIn("外部执行服务", architecture_readme)

    def test_current_cpr_docs_describe_thick_slab_support(self):
        """当前 CPR 文档应说明层厚参与 offset 覆盖和厚层平均。"""

        cpr_doc = (
            SUPERPOWERS_ROOT
            / "modules"
            / "reconstruction"
            / "2026-07-22-脊柱冠状位CPR设计.md"
        ).read_text(encoding="utf-8")
        coverage_doc = (
            SUPERPOWERS_ROOT
            / "modules"
            / "reconstruction"
            / "2026-07-30-脊柱CPR层厚层间距覆盖设计.md"
        ).read_text(encoding="utf-8")

        self.assertNotIn("厚层 CPR 作为后续版本", cpr_doc)
        self.assertIn("层厚参与覆盖判定", coverage_doc)
        self.assertIn("算术平均", coverage_doc)
        self.assertIn("必须与实际 CPR 重建一致", cpr_doc)
        self.assertIn("复用实际 CPR 的曲线", coverage_doc)
        self.assertIn("`offset_mm`", coverage_doc)
        self.assertNotIn("`position_mm`", coverage_doc)

    def test_superseded_localization_design_is_archived(self):
        """已替代的回正 no-op 设计不能继续作为模块当前规则。"""

        legacy_design = (
            SUPERPOWERS_ROOT
            / "modules"
            / "localization"
            / "2026-07-08-第三阶段目标横状位校正设计.md"
        )
        self.assertFalse(legacy_design.exists())

    def test_current_docs_describe_debug_artifact_pruning(self):
        """当前文档应说明定位与规划调试 NIfTI 的保留边界。"""

        localization_doc = (
            SUPERPOWERS_ROOT
            / "modules"
            / "localization"
            / "2026-07-29-椎体四棱柱质心设计.md"
        ).read_text(encoding="utf-8")
        cpr_doc = (
            SUPERPOWERS_ROOT
            / "modules"
            / "reconstruction"
            / "2026-07-22-脊柱冠状位CPR设计.md"
        ).read_text(encoding="utf-8")
        viewer_readme = (
            SUPERPOWERS_ROOT / "modules/viewer/README.md"
        ).read_text(encoding="utf-8")

        self.assertIn("默认不再写出 `定位框.nii.gz` 和 `目标掩膜.nii.gz`", localization_doc)
        for filename in (
            "标签掩膜.nii.gz",
            "椎体主体掩膜.nii.gz",
            "椎体四棱柱掩膜.nii.gz",
        ):
            self.assertIn(filename, localization_doc)
        self.assertIn("第六阶段不再写出任何规划 NIfTI 调试总览", cpr_doc)
        self.assertIn("第六阶段不生成 NIfTI 调试总览", viewer_readme)

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

    def test_archive_replacement_index_covers_all_plans_and_old_specs(self):
        """每份历史计划和旧规范都应在替代索引中有独立入口。"""

        archive = SUPERPOWERS_ROOT / "archive"
        index = (archive / "2026-10-08-历史文档替代索引.md").read_text(
            encoding="utf-8"
        )
        for folder in ("plans", "specs_old"):
            for path in (archive / folder).glob("*.md"):
                if path.name == "README.md":
                    continue
                self.assertIn(f"]({folder}/{path.name})", index, msg=str(path))

    def test_archived_documents_open_with_history_and_current_design_links(self):
        """搜索直接打开历史文件时，开头必须提示历史属性及当前替代规则。"""

        archive = SUPERPOWERS_ROOT / "archive"
        for folder in ("plans", "specs_old"):
            for path in (archive / folder).glob("*.md"):
                if path.name == "README.md":
                    continue
                intro = "\n".join(path.read_text(encoding="utf-8").splitlines()[:12])
                self.assertIn("历史", intro, msg=str(path))
                self.assertRegex(
                    intro,
                    r"\]\(\.\./\.\./(?:architecture|modules)/[^)]+\.md\)",
                    msg=str(path),
                )

    def test_fill_percent_design_distinguishes_group_framing_from_display(self):
        """占比规范必须说明分组、0 值和屏幕显示的区别。"""

        design = (
            SUPERPOWERS_ROOT
            / "modules/reconstruction/2026-10-08-整组居中与FillPercent取景设计.md"
        ).read_text(encoding="utf-8")
        for statement in (
            "省略、null、0 均仅按整组目标范围居中",
            "占比的参照物是输出图像，不是屏幕",
            "不是只挑面积最大的一层",
            "定位图不应用取景",
            "不能承诺与原 CT 精确三维联动",
        ):
            self.assertIn(statement, design)

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
