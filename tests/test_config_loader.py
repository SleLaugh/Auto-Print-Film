import unittest
import json
import tempfile
from copy import deepcopy
from pathlib import Path

from src.config import (
    load_app_config,
    load_slice_part_config,
    validate_slice_part_config,
)
from src.core.pipeline import Pipeline
from src.stages.stage_localization import LocalizationStage
from src.stages.stage_segmentation import SegmentationStage


class ConfigLoaderTest(unittest.TestCase):
    def test_explicit_config_overrides_environment_and_file(self):
        config = load_app_config(
            explicit={"nnunet": {"results": "/explicit/results"}},
            environ={"nnUNet_results": "/env/results"},
            defaults={"nnunet": {"results": "/file/results"}},
        )

        self.assertEqual(config.nnunet.results, "/explicit/results")

    def test_empty_repository_path_keeps_environment_value(self):
        config = load_app_config(
            environ={"nnUNet_results": "/env/results"},
            defaults={"nnunet": {"results": ""}},
        )

        self.assertEqual(config.nnunet.results, "/env/results")

    def test_explicit_empty_defaults_do_not_read_repository_files(self):
        """显式空默认配置必须保持为空，不得回退读取仓库文件。"""

        config = load_app_config(environ={}, defaults={})

        self.assertEqual(config.slice_part_config, {})

    def test_removed_study_part_config_is_rejected(self):
        """旧切割部位配置入口不得被任何配置来源静默忽略。"""

        cases = (
            ({"study_part_config": {}}, {"nnunet": {}}),
            ({}, {"study_part_config": {}}),
        )
        for explicit, defaults in cases:
            with self.subTest(explicit=explicit, defaults=defaults):
                with self.assertRaisesRegex(ValueError, "study_part_config"):
                    load_app_config(
                        explicit=explicit,
                        environ={},
                        defaults=defaults,
                    )

    def test_nnunet_runtime_keys_are_normalized(self):
        config = load_app_config(
            environ={},
            defaults={
                "_configuration": "3d_lowres",
                "_fold": "2",
                "_continue": False,
                "nnUNet_raw": "/models/raw",
            },
        )

        self.assertEqual(config.nnunet.configuration, "3d_lowres")
        self.assertEqual(config.nnunet.fold, "2")
        self.assertFalse(config.nnunet.continue_mode)
        self.assertEqual(config.nnunet.raw, "/models/raw")

    def test_string_false_is_parsed_as_false(self):
        """环境变量形式的字符串 false 不能被 Python bool 规则误判为 True。"""
        config = load_app_config(
            environ={"_continue": "false"},
            defaults={},
        )

        self.assertFalse(config.nnunet.continue_mode)

    def test_invalid_boolean_text_is_rejected(self):
        """拼写错误的布尔配置应在启动时失败，而不是静默启用。"""
        with self.assertRaisesRegex(ValueError, "布尔值"):
            load_app_config(
                environ={},
                defaults={"_continue": "no"},
            )

    def test_logging_and_pipeline_boolean_text_is_normalized(self):
        """日志和流水线配置同样支持 JSON/调用方常见的 true/false 文本。"""
        config = load_app_config(
            environ={},
            defaults={
                "logging": {
                    "console_enabled": "false",
                    "error_log_file_enabled": "false",
                },
                "pipeline": {
                    "run_preflight": "false",
                },
            },
        )

        self.assertFalse(config.logging.console_enabled)
        self.assertFalse(config.logging.error_log_file_enabled)
        self.assertFalse(config.pipeline.run_preflight)

    def test_stages_accept_the_single_loaded_nnunet_configuration(self):
        stage_config = {
            "_configuration": "3d_lowres",
            "_fold": "3",
            "_continue": False,
            "nnUNet_raw": "/central/raw",
            "nnUNet_preprocessed": "/central/preprocessed",
            "nnUNet_results": "/central/results",
        }

        segmentation = SegmentationStage(stage_config)
        localization = LocalizationStage(stage_config)

        self.assertEqual(segmentation._configuration, "3d_lowres")
        self.assertEqual(segmentation._fold, "3")
        self.assertFalse(segmentation._continue)
        self.assertEqual(
            segmentation._nnunet_env_vars["nnUNet_results"],
            "/central/results",
        )
        self.assertEqual(
            localization._nnunet_paths["nnUNet_results"],
            "/central/results",
        )

    def test_default_pipeline_injects_loaded_config_into_model_stages(self):
        config = load_app_config(
            environ={},
            defaults={
                "nnUNet_results": "/central/results",
            },
        )

        pipeline = Pipeline(config=config)
        segmentation = next(
            stage for stage in pipeline.stages
            if stage.name == "segmentation"
        )

        self.assertEqual(
            segmentation._nnunet_env_vars["nnUNet_results"],
            "/central/results",
        )

    def test_default_loader_combines_nnunet_and_logger_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nnunet_path = root / "nnuNet_config.json"
            nnunet_path.write_text(
                json.dumps({"_fold": "4"}),
                encoding="utf-8",
            )
            (root / "logger_config.json").write_text(
                json.dumps({
                    "debug_file_enabled": True,
                    "debug_log_file_enabled": False,
                }),
                encoding="utf-8",
            )
            (root / "slice_part_config.json").write_text(
                Path("src/config/slice_part_config.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )

            config = load_app_config(
                environ={},
                default_path=nnunet_path,
            )

            self.assertEqual(config.nnunet.fold, "4")
            self.assertTrue(config.logging.debug_file_enabled)
            self.assertFalse(config.logging.debug_log_file_enabled)
            self.assertIn("腰椎", config.slice_part_config["slice_parts"])

    def test_slice_part_config_rejects_unknown_fields_and_references(self):
        base_config = load_slice_part_config()

        cases = []
        old_root = deepcopy(base_config)
        old_root["study_parts"] = old_root.pop("slice_parts")
        cases.append((old_root, "未知字段"))

        unknown_field = deepcopy(base_config)
        unknown_field["slice_parts"]["肺部"]["slice_rules"]["横状位"]["MPR"][
            "slice_range_extension"
        ] = {}
        cases.append((unknown_field, "未知字段"))

        missing_localizer_plane = deepcopy(base_config)
        del missing_localizer_plane["slice_parts"]["肺部"]["slice_rules"][
            "横状位"
        ]["MPR"]["localizer_plane"]
        cases.append((missing_localizer_plane, "localizer_plane"))

        unknown_localizer_plane = deepcopy(base_config)
        unknown_localizer_plane["slice_parts"]["肺部"]["slice_rules"][
            "横状位"
        ]["MPR"]["localizer_plane"] = "斜位"
        cases.append((unknown_localizer_plane, "localizer_plane"))

        same_localizer_plane = deepcopy(base_config)
        same_localizer_plane["slice_parts"]["肺部"]["slice_rules"][
            "横状位"
        ]["MPR"]["localizer_plane"] = "横状位"
        cases.append((same_localizer_plane, "不能与重建状位相同"))

        unknown_label = deepcopy(base_config)
        unknown_label["slice_parts"]["腰椎"]["localization"]["regions"]["default"][
            "start_label"
        ] = "L0"
        cases.append((unknown_label, "未在 label_aliases 中定义"))

        duplicate_alias = deepcopy(base_config)
        duplicate_alias["label_aliases"]["duplicate_l1"] = "vertebrae_L1"
        cases.append((duplicate_alias, "重复引用标签"))

        unused_region = deepcopy(base_config)
        unused_region["slice_parts"]["颈椎"]["localization"]["regions"]["C1_C2"] = {
            "type": "interspace",
            "start_label": "C1",
            "end_label": "C2",
        }
        cases.append((unused_region, "未使用区域"))

        missing_full_label = deepcopy(base_config)
        del missing_full_label["slice_parts"]["胸椎"]["localization"]["spine_cpr"][
            "full_label"
        ]
        cases.append((missing_full_label, "缺少 full_label"))

        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_slice_part_config(config)

    def test_removed_study_part_config_files_do_not_exist(self):
        """旧切割部位配置文件和模块不再保留。"""

        self.assertFalse(Path("src/config/study_part_config.json").exists())
        self.assertFalse(Path("src/config/study_parts.py").exists())


if __name__ == "__main__":
    unittest.main()
