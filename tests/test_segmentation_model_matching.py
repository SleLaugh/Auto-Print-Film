import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.context import ProcessingContext
from src.models import Request
from src.stages.stage_segmentation import SegmentationStage


PART_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "config"
    / "slice_part_config.json"
)


class SegmentationModelMatchingTest(unittest.TestCase):
    """验证扫描层厚按配置闭区间选择 nnUNet 模型。"""

    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(PART_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.stage = SegmentationStage({"slice_part_config": cls.config})

    def test_lumbar_boundaries_select_expected_models(self):
        cases = (
            (0.0, 292),
            (1.5, 292),
            (1.6, 297),
            (4.5, 297),
            (4.6, 298),
            (6.0, 298),
        )

        for thickness, expected_model in cases:
            with self.subTest(thickness=thickness):
                model = self.stage._match_model("腰椎", "CT", thickness)
                self.assertEqual(model["model_id"], expected_model)
                self.assertGreaterEqual(
                    thickness,
                    model["min_scan_thickness"],
                )
                self.assertLessEqual(
                    thickness,
                    model["max_scan_thickness"],
                )

    def test_unconfigured_thickness_gap_fails_explicitly(self):
        for thickness in (1.55, 4.55, 6.1):
            with self.subTest(thickness=thickness):
                with self.assertRaisesRegex(ValueError, "未找到匹配"):
                    self.stage._match_model("腰椎", "CT", thickness)

    def test_lung_only_supports_existing_thin_slice_model(self):
        self.assertEqual(
            self.stage._match_model("肺部", "CT", 1.5)["model_id"],
            315,
        )
        with self.assertRaisesRegex(ValueError, "未找到匹配"):
            self.stage._match_model("肺部", "CT", 1.6)

    def test_config_no_longer_contains_scan_thickness_categories(self):
        for part in self.config["slice_parts"].values():
            for mappings in part["model_rules"].values():
                for mapper in mappings:
                    self.assertNotIn("scan_thickness", mapper)
                    self.assertIn("scan_thickness_range_mm", mapper)

    def test_overlapping_ranges_fail_instead_of_picking_first(self):
        mappings = [
            {
                "scan_thickness_range_mm": {"min": 0, "max": 2},
                "model_id": 1,
            },
            {
                "scan_thickness_range_mm": {"min": 1, "max": 3},
                "model_id": 2,
            },
        ]

        for thickness in (0.5, 1.5):
            with self.subTest(thickness=thickness):
                with self.assertRaisesRegex(ValueError, "区间重叠"):
                    SegmentationStage._match_model_mapping(mappings, thickness)

    def test_removed_category_mapper_is_rejected(self):
        mappings = [{"scan_thickness": "default", "model_id": 292}]

        with self.assertRaisesRegex(ValueError, "scan_thickness_range_mm"):
            SegmentationStage._match_model_mapping(mappings, 1.0)

    def test_invalid_request_thickness_is_rejected(self):
        for value in (-1, float("nan"), float("inf"), True, "default"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ScanThickness"):
                    self.stage._normalize_scan_thickness(value)

    def test_existing_segmentation_cache_is_reused_without_running_inference(self):
        """-c/模型ID/segmentation 下已有分割时跳过 nnUNet。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nifti_path = root / "nifti" / "raw_0000.nii.gz"
            nifti_path.parent.mkdir()
            nifti_path.touch()
            cached_path = root / "cache" / "292" / "segmentation" / "seg.nii.gz"
            cached_path.parent.mkdir(parents=True)
            cached_path.touch()
            request = Request(
                dicom_dir=str(root / "dicom"),
                output_dir=str(root / "output"),
                scan_thickness=1.0,
                cache_dir=str(root / "cache"),
            )
            context = ProcessingContext(request)
            context.set_result("preprocessing", {"nifti_path": str(nifti_path)})

            with patch.object(
                self.stage,
                "_run_nnunet_inference",
                side_effect=AssertionError("已有分割时不应再次推理"),
            ):
                self.stage.execute(context)

            result = context.get_result("segmentation")
            self.assertEqual(result["seg_path"], str(cached_path))
            self.assertEqual(request.output_dir, str(root / "output"))

    def test_missing_segmentation_cache_writes_under_model_directory(self):
        """-c 未命中时隔离推理输出，完成后发布到模型缓存目录。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nifti_path = root / "nifti" / "raw_0000.nii.gz"
            nifti_path.parent.mkdir()
            nifti_path.touch()
            request = Request(
                dicom_dir=str(root / "dicom"),
                output_dir=str(root / "output"),
                scan_thickness=1.0,
                cache_dir=str(root / "cache"),
            )
            context = ProcessingContext(request)
            context.set_result("preprocessing", {"nifti_path": str(nifti_path)})

            def write_fake_segmentation(**kwargs):
                kwargs["output_path"].write_bytes(b"complete segmentation")

            with patch.object(
                self.stage,
                "_run_nnunet_inference",
                side_effect=write_fake_segmentation,
            ) as run_inference:
                self.stage.execute(context)

            result = context.get_result("segmentation")
            expected = root / "cache" / "292" / "segmentation" / "seg.nii.gz"
            self.assertEqual(result["seg_path"], str(expected))
            self.assertEqual(expected.read_bytes(), b"complete segmentation")
            temporary_output = run_inference.call_args.kwargs["output_path"]
            self.assertNotEqual(temporary_output, expected)
            self.assertFalse(temporary_output.exists())
            self.assertEqual(
                json.loads((expected.parent / "model.json").read_text()),
                {"model_id": "292"},
            )

    def test_output_directory_remains_the_original_output_root(self):
        """第三阶段输出根目录仍是 -o，不会切换到模型目录。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nifti_path = root / "nifti" / "raw_0000.nii.gz"
            nifti_path.parent.mkdir()
            nifti_path.touch()
            old_file = root / "output" / "report.json"
            old_file.parent.mkdir(parents=True)
            old_file.write_text("old", encoding="utf-8")
            request = Request(
                dicom_dir=str(root / "dicom"),
                output_dir=str(root / "output"),
                scan_thickness=1.0,
                cache_dir=str(root / "cache"),
            )
            context = ProcessingContext(request)
            context.set_result("preprocessing", {"nifti_path": str(nifti_path)})

            with patch.object(
                self.stage,
                "_run_nnunet_inference",
                side_effect=lambda **kwargs: kwargs["output_path"].touch(),
            ):
                self.stage.execute(context)

            self.assertTrue(old_file.exists())
            self.assertEqual(request.output_dir, str(root / "output"))

    def test_without_segmentation_cache_keeps_original_output_behavior(self):
        """未传 -c 时仍将第二阶段结果写入 -o/segmentation。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nifti_path = root / "nifti" / "raw_0000.nii.gz"
            nifti_path.parent.mkdir()
            nifti_path.touch()
            request = Request(
                dicom_dir=str(root / "dicom"),
                output_dir=str(root / "output"),
                scan_thickness=1.0,
            )
            context = ProcessingContext(request)
            context.set_result("preprocessing", {"nifti_path": str(nifti_path)})

            with patch.object(
                self.stage,
                "_run_nnunet_inference",
                side_effect=lambda **kwargs: kwargs["output_path"].touch(),
            ):
                self.stage.execute(context)

            expected = root / "output" / "segmentation" / "seg.nii.gz"
            self.assertEqual(context.get_result("segmentation")["seg_path"], str(expected))


if __name__ == "__main__":
    unittest.main()
