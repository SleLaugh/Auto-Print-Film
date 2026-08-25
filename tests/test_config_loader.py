import unittest
import json
import tempfile
from copy import deepcopy
from pathlib import Path

from src.config import (
    load_app_config,
    load_study_part_config,
    validate_study_part_config,
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
            (root / "study_part_config.json").write_text(
                Path("src/config/study_part_config.json").read_text(
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
            self.assertIn("腰椎", config.study_part_config["study_parts"])

    def test_study_part_config_rejects_unknown_fields_and_references(self):
        base_config = load_study_part_config()

        cases = []
        unknown_field = deepcopy(base_config)
        unknown_field["study_parts"]["肺部"]["slice_rules"]["横状位"]["MPR"][
            "slice_range_extension"
        ] = {}
        cases.append((unknown_field, "未知字段"))

        unknown_label = deepcopy(base_config)
        unknown_label["study_parts"]["腰椎"]["localization"]["regions"]["default"][
            "start_label"
        ] = "L0"
        cases.append((unknown_label, "未在 label_aliases 中定义"))

        duplicate_alias = deepcopy(base_config)
        duplicate_alias["label_aliases"]["duplicate_l1"] = "vertebrae_L1"
        cases.append((duplicate_alias, "重复引用标签"))

        unused_region = deepcopy(base_config)
        unused_region["study_parts"]["颈椎"]["localization"]["regions"]["C1_C2"] = {
            "type": "interspace",
            "start_label": "C1",
            "end_label": "C2",
        }
        cases.append((unused_region, "未使用区域"))

        missing_full_label = deepcopy(base_config)
        del missing_full_label["study_parts"]["胸椎"]["localization"]["spine_cpr"][
            "full_label"
        ]
        cases.append((missing_full_label, "缺少 full_label"))

        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_study_part_config(config)


if __name__ == "__main__":
    unittest.main()
