import unittest
import json
import tempfile
from pathlib import Path

from src.config import load_app_config
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

    def test_legacy_nnunet_keys_are_normalized(self):
        config = load_app_config(
            environ={},
            defaults={
                "_configuration": "3d_lowres",
                "_fold": "2",
                "_continue": False,
                "nnUNet_raw": "/models/raw",
                "_model_rules": [{"body_part": "腰椎"}],
            },
        )

        self.assertEqual(config.nnunet.configuration, "3d_lowres")
        self.assertEqual(config.nnunet.fold, "2")
        self.assertFalse(config.nnunet.continue_mode)
        self.assertEqual(config.nnunet.raw, "/models/raw")
        self.assertEqual(config.nnunet.model_rules[0]["body_part"], "腰椎")

    def test_stages_accept_the_single_loaded_nnunet_configuration(self):
        stage_config = {
            "_configuration": "3d_lowres",
            "_fold": "3",
            "_continue": False,
            "_model_rules": [{"body_part": "腰椎"}],
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
                "_model_rules": [{"body_part": "腰椎"}],
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
                json.dumps({"debug_log_file_enabled": True}),
                encoding="utf-8",
            )

            config = load_app_config(
                environ={},
                default_path=nnunet_path,
            )

            self.assertEqual(config.nnunet.fold, "4")
            self.assertTrue(config.logging.debug_log_file_enabled)


if __name__ == "__main__":
    unittest.main()
