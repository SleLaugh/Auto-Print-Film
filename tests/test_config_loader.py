import unittest
import json
import tempfile
from copy import deepcopy
from pathlib import Path

from src.config import (
    load_app_config,
    load_slice_part_config,
    NnUNetConfig,
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

    def test_inference_options_preserve_defaults_for_existing_config(self):
        config = load_app_config(environ={}, defaults={"_fold": "0"})

        self.assertFalse(config.nnunet.disable_tta)
        self.assertEqual(config.nnunet.step_size, 0.5)
        self.assertFalse(config.summary()["nnunet"]["disable_tta"])
        self.assertEqual(config.summary()["nnunet"]["step_size"], 0.5)
        for name in (
            "num_processes_preprocessing",
            "num_processes_segmentation_export",
        ):
            self.assertEqual(getattr(config.nnunet, name), 1)
            self.assertEqual(config.summary()["nnunet"][name], 1)

    def test_worker_counts_follow_each_configuration_priority(self):
        for name in (
            "num_processes_preprocessing",
            "num_processes_segmentation_export",
        ):
            sources = {
                "explicit": {"nnunet": {name: 2}, name: 3},
                "environ": {name: "4"},
                "defaults": {"nnunet": {name: 5}, name: 6},
            }
            for expected in (2, 3, 4, 5, 6, 1):
                with self.subTest(name=name, expected=expected):
                    config = load_app_config(**sources)
                    self.assertEqual(getattr(config.nnunet, name), expected)
                    self.assertEqual(config.summary()["nnunet"][name], expected)
                if expected == 2:
                    sources["explicit"].pop("nnunet")
                elif expected == 3:
                    sources["explicit"].pop(name)
                elif expected == 4:
                    sources["environ"].pop(name)
                elif expected == 5:
                    sources["defaults"].pop("nnunet")
                elif expected == 6:
                    sources["defaults"].pop(name)

    def test_worker_counts_accept_integer_text(self):
        for value in ("2", " 3 ", "+4", "005"):
            with self.subTest(value=value):
                config = load_app_config(
                    explicit={"nnunet": {"num_processes_preprocessing": value}},
                    environ={"num_processes_segmentation_export": value},
                    defaults={},
                )
                self.assertEqual(config.nnunet.num_processes_preprocessing, int(value))
                self.assertEqual(config.nnunet.num_processes_segmentation_export, int(value))

    def test_worker_counts_reject_nonpositive_and_noninteger_values(self):
        for name in (
            "num_processes_preprocessing",
            "num_processes_segmentation_export",
        ):
            for value in (
                True, False, 0, -1, 1.0, 1.5, float("nan"), float("inf"),
                "0", "-1", "1.0", "1.5", "1e2", "true", "nan", "1_000", " ", [], {},
            ):
                for defaults in ({name: value}, {"nnunet": {name: value}}):
                    with self.subTest(name=name, value=value, defaults=defaults):
                        with self.assertRaisesRegex(ValueError, name):
                            load_app_config(environ={}, defaults=defaults)
            for value in (True, False, 0, -1, 1.0, 1.5, "1", None):
                with self.subTest(name=name, typed_value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        NnUNetConfig(**{name: value})

    def test_inference_options_follow_configuration_precedence(self):
        config = load_app_config(
            explicit={"nnunet": {"disable_tta": False, "step_size": 0.75}},
            environ={"disable_tta": "true", "step_size": "1"},
            defaults={"disable_tta": True, "step_size": 0.25},
        )

        self.assertFalse(config.nnunet.disable_tta)
        self.assertEqual(config.nnunet.step_size, 0.75)

    def test_inference_options_accept_environment_text(self):
        config = load_app_config(
            environ={"disable_tta": "false", "step_size": "0.75"},
            defaults={"disable_tta": True, "step_size": 0.5},
        )

        self.assertFalse(config.nnunet.disable_tta)
        self.assertEqual(config.nnunet.step_size, 0.75)

    def test_invalid_inference_options_fail_during_config_loading(self):
        cases = (
            ("disable_tta", "no"),
            ("disable_tta", 1),
            ("step_size", 0),
            ("step_size", -0.5),
            ("step_size", 1.1),
            ("step_size", True),
            ("step_size", "nan"),
            ("step_size", "inf"),
            ("step_size", "fast"),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    load_app_config(environ={}, defaults={key: value})

    def test_typed_inference_options_also_reject_invalid_values(self):
        for kwargs in (
            {"disable_tta": "false"},
            {"step_size": True},
            {"step_size": 0},
            {"step_size": 10 ** 1000},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    NnUNetConfig(**kwargs)

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
                "disable_tta": True,
                "step_size": 0.75,
                "num_processes_preprocessing": 2,
                "num_processes_segmentation_export": 4,
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
        self.assertTrue(segmentation._disable_tta)
        self.assertEqual(segmentation._step_size, 0.75)
        self.assertEqual(segmentation._num_processes_preprocessing, 2)
        self.assertEqual(segmentation._num_processes_segmentation_export, 4)

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
