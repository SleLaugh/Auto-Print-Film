import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.core.context import ProcessingContext
from src.models import (
    BodyPartEnum,
    CutModeEnum,
    PlaneConfig,
    PlaneTypeEnum,
    Request,
    WindowPresetEnum,
)
from src.reconstruction.plane_stack import PlaneStackResampler
from src.reconstruction.windowing import apply_window_to_image
from src.stages.stage_reconstruction import ReconstructionStage


class Stage5ReconstructionTest(unittest.TestCase):
    def test_windowing_bakes_hu_into_fixed_uint16_grayscale(self):
        image = sitk.GetImageFromArray(
            np.asarray([[[-850, 400, 1650]]], dtype=np.int16),
        )

        output = apply_window_to_image(image, "骨窗")

        self.assertEqual(output.GetPixelID(), sitk.sitkUInt16)
        self.assertEqual(
            sitk.GetArrayFromImage(output).tolist(),
            [[[0, 32768, 65535]]],
        )

    def test_interval_plan_keeps_group_directory(self):
        plan = {"cut_mode": "interspace", "group_name": "L1-L2"}

        self.assertEqual(ReconstructionStage._plan_group_name(plan), "L1-L2")

    def test_plane_stack_resampler_uses_slice_center_and_in_plane_extent(self):
        image = self._make_ramp_image(size=(8, 8, 8))
        plan = {
            "extent_mm": {"width": 4.0, "height": 4.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 1.0,
        }
        planned_slice = {
            "center": {"x": 2.0, "y": 2.0, "z": 2.0},
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
            "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
        }

        output = PlaneStackResampler().sample_slice(image, plan, planned_slice)

        self.assertEqual(output.GetSize(), (5, 5, 1))
        self.assertEqual(output.GetSpacing(), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(
            float(sitk.GetArrayFromImage(output)[0, 0, 0]),
            200.0,
        )

    def test_axis_aligned_slab_weights_partial_source_layers_by_overlap(self):
        image = sitk.GetImageFromArray(
            np.asarray([[[0]], [[10]], [[20]], [[30]], [[40]], [[50]]], dtype=np.int16),
        )
        image.SetSpacing((1.0, 1.0, 0.65))
        plan = {
            "extent_mm": {"width": 0.0, "height": 0.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 3.0,
        }
        planned_slice = {
            "center": {"x": 0.0, "y": 0.0, "z": 2.0},
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
            "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
        }

        output = PlaneStackResampler().sample_slice(image, plan, planned_slice)

        # slab 为 0.5..3.5mm：0.475*10 + 0.65*20 + 0.65*30 +
        # 0.65*40 + 0.575*50，再除以 3mm。
        self.assertAlmostEqual(
            float(sitk.GetArrayFromImage(output)[0, 0, 0]),
            92.0 / 3.0,
            delta=1e-3,
        )

    def test_reconstruction_consumes_mapping_plans_and_reports_physical_slice_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                window_presets=["骨窗"],
            )

            ReconstructionStage(config={"parallel": False}).execute(context)

            result = context.get_result("reconstruction")
            self.assertEqual(result["plan_count"], 1)
            self.assertEqual(result["slice_count"], 2)
            self.assertEqual(result["image_count"], 2)
            self.assertEqual(result["planes"]["横状位"]["slice_count"], 2)

    def test_reconstruction_rejects_unsupported_geometry_type_with_plan_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                geometry_type="curved_stack",
            )

            with self.assertRaisesRegex(ValueError, r"plan_id=.*geometry_type"):
                ReconstructionStage(config={"parallel": False}).execute(context)

    def test_slab_thickness_averages_source_planes_without_adding_slices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=1,
                window_presets=["骨窗"],
            )
            raw_path = Path(
                context.get_result("localization")["raw_for_reconstruction_path"]
            )
            raw_image = sitk.GetImageFromArray(
                np.asarray([[[0]], [[10]], [[20]]], dtype=np.int16),
            )
            raw_image.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(raw_image, str(raw_path))

            plan = context.get_result("mapping")["plans"][0]
            plan["slice_thickness"] = 3.0
            plan["slice_spacing"] = 5.0
            plan["extent_mm"] = {"width": 0.0, "height": 0.0}
            plan["slices"][0]["center"] = {"x": 0.0, "y": 0.0, "z": 1.0}

            ReconstructionStage(config={"parallel": False}).execute(context)

            result = context.get_result("reconstruction")
            self.assertEqual(result["slice_count"], 1)
            dcm_path = next(Path(temp_dir).rglob("*.dcm"))
            output = sitk.ReadImage(str(dcm_path))
            expected_gray = round((10.0 + 850.0) / 2500.0 * 65535.0)
            self.assertEqual(
                int(sitk.GetArrayFromImage(output)[0, 0, 0]),
                expected_gray,
            )

    def test_reconstruction_writes_one_series_per_window_preset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                window_presets=["骨窗", "脑窗"],
            )

            ReconstructionStage(config={"parallel": False}).execute(context)

            root = Path(context.request.output_dir)
            self.assertEqual(
                len(list((root / "横状位" / "骨窗").rglob("*.dcm"))),
                2,
            )
            self.assertEqual(
                len(list((root / "横状位" / "脑窗").rglob("*.dcm"))),
                2,
            )
            self.assertEqual(
                len(list((root / "横状位" / "骨窗").glob("切片_*.dcm"))),
                2,
            )
            self.assertFalse((root / "横状位" / "骨窗" / "整切").exists())
            bone_paths = sorted((root / "横状位" / "骨窗").glob("切片_*.dcm"))
            bone_images = [sitk.ReadImage(str(path)) for path in bone_paths]
            self.assertEqual(
                len({image.GetMetaData("0020|000e") for image in bone_images}),
                1,
            )
            self.assertIn("_1_2", bone_images[0].GetMetaData("0008|103e"))
            self.assertIn("_2_2", bone_images[1].GetMetaData("0008|103e"))
            self.assertEqual(
                context.get_result("reconstruction")["image_count"],
                4,
            )

    def test_written_dicom_contains_slice_geometry_and_window_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=1,
                window_presets=["骨窗"],
            )

            ReconstructionStage(config={"parallel": False}).execute(context)

            dcm_path = next(Path(context.request.output_dir).rglob("*.dcm"))
            image = sitk.ReadImage(str(dcm_path))
            self.assertEqual(image.GetMetaData("0008|0060"), "CT")
            self.assertEqual(float(image.GetMetaData("0018|0050")), 1.0)
            self.assertEqual(image.GetPixelID(), sitk.sitkUInt16)
            self.assertEqual(image.GetMetaData("0028|1050").strip(), "32768")
            self.assertEqual(image.GetMetaData("0028|1051").strip(), "65535")
            self.assertIn("骨窗", image.GetMetaData("0008|103e"))
            self.assertIn("\\", image.GetMetaData("0020|0037"))

    def test_derived_dicom_inherits_source_study_and_patient_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            source_dir.mkdir()
            source_path = source_dir / "source.dcm"
            source = sitk.GetImageFromArray(np.asarray([[10]], dtype=np.int16))
            source.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
            source.SetMetaData("0008|0018", "1.2.826.0.1.3680043.2.1125.100")
            source.SetMetaData("0008|0060", "CT")
            source.SetMetaData("0008|0020", "20260710")
            source.SetMetaData("0008|0030", "121314")
            source.SetMetaData("0008|0050", "ACC-001")
            source.SetMetaData("0010|0010", "TEST^PATIENT")
            source.SetMetaData("0010|0020", "PID-001")
            source.SetMetaData("0010|0030", "19800101")
            source.SetMetaData("0010|0040", "F")
            source.SetMetaData("0020|000d", "1.2.826.0.1.3680043.2.1125.200")
            source.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125.300")
            source.SetMetaData("0020|0010", "STUDY-001")
            source.SetMetaData("0020|0052", "1.2.826.0.1.3680043.2.1125.400")
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(str(source_path))
            writer.Execute(source)

            context = self._make_context_with_one_plan(
                temp_dir,
                slice_count=2,
                dicom_dir=str(source_dir),
            )
            ReconstructionStage(config={"parallel": False}).execute(context)

            output_paths = sorted(Path(context.request.output_dir).rglob("*.dcm"))
            first = sitk.ReadImage(str(output_paths[0]))
            second = sitk.ReadImage(str(output_paths[1]))
            self.assertEqual(first.GetMetaData("0010|0010").strip(), "TEST^PATIENT")
            self.assertEqual(first.GetMetaData("0010|0020").strip(), "PID-001")
            self.assertEqual(first.GetMetaData("0010|0040").strip(), "F")
            self.assertEqual(first.GetMetaData("0008|0020").strip(), "20260710")
            self.assertEqual(first.GetMetaData("0008|0050").strip(), "ACC-001")
            self.assertEqual(
                first.GetMetaData("0020|000d").strip(),
                "1.2.826.0.1.3680043.2.1125.200",
            )
            self.assertEqual(
                first.GetMetaData("0020|0052").strip(),
                "1.2.826.0.1.3680043.2.1125.400",
            )
            self.assertNotEqual(
                first.GetMetaData("0020|000e"),
                "1.2.826.0.1.3680043.2.1125.300",
            )
            self.assertEqual(
                first.GetMetaData("0020|000e"),
                second.GetMetaData("0020|000e"),
            )
            self.assertNotEqual(
                first.GetMetaData("0008|0018"),
                second.GetMetaData("0008|0018"),
            )

    def test_derived_dicom_decodes_gb18030_surrogate_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._make_context_with_one_plan(temp_dir, slice_count=1)
            context.set_result("preprocessing", {
                "source_dicom_metadata": {
                    "0008|0005": "GB18030",
                    "0008|1030": (
                        "11_\udcd0\udcd8ȫ\udcb8\udcb9ƽɨ"
                        "\udca3\udca8\udcb3\udcc9\udcc8ˣ\udca9"
                    ),
                    "0020|000d": "1.2.826.0.1.3680043.2.1125.500",
                },
            })

            ReconstructionStage(config={"parallel": False}).execute(context)

            dcm_path = next(Path(context.request.output_dir).rglob("*.dcm"))
            image = sitk.ReadImage(str(dcm_path))
            description = image.GetMetaData("0008|1030")
            self.assertNotRegex(description, r"[\ud800-\udfff]")
            self.assertEqual(description.strip(), "11_胸全腹平扫（成人）")

    @staticmethod
    def _make_context_with_one_plan(
        temp_dir: str,
        *,
        slice_count: int = 2,
        window_presets: list[str] | None = None,
        geometry_type: str = "plane_stack",
        dicom_dir: str = "/tmp/dicom",
    ) -> ProcessingContext:
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "raw.nii.gz"
        raw_image = sitk.GetImageFromArray(
            np.zeros((8, 8, 8), dtype=np.int16),
        )
        raw_image.SetSpacing((1.0, 1.0, 1.0))
        raw_image.SetOrigin((0.0, 0.0, 0.0))
        raw_image.SetDirection((
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ))
        sitk.WriteImage(raw_image, str(raw_path))

        centers = [2.0 + 2.0 * index for index in range(slice_count)]
        plan = {
            "plan_id": "测试_整切_横状位",
            "body_part": "腰椎",
            "cut_mode": "whole_range",
            "plane": "横状位",
            "group_name": "whole_range",
            "geometry_type": geometry_type,
            "range_start": {"x": 2.0, "y": 2.0, "z": 2.0},
            "range_end": {"x": 2.0, "y": 2.0, "z": centers[-1]},
            "path_points": [
                {"x": 2.0, "y": 2.0, "z": 2.0},
                {"x": 2.0, "y": 2.0, "z": centers[-1]},
            ],
            "slices": [
                {
                    "index": index + 1,
                    "center": {"x": 2.0, "y": 2.0, "z": center},
                    "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "row_dir": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "col_dir": {"x": 0.0, "y": 1.0, "z": 0.0},
                }
                for index, center in enumerate(centers)
            ],
            "extent_mm": {"width": 4.0, "height": 4.0},
            "output_spacing_mm": {"x": 1.0, "y": 1.0, "z": 0.0},
            "slice_thickness": 1.0,
            "slice_spacing": 2.0,
            "window_presets": window_presets or ["骨窗"],
            "source_labels": ["correction_scope_bbox"],
        }

        request = Request(
            dicom_dir=dicom_dir,
            output_dir=str(output_dir),
            body_part=BodyPartEnum.腰椎,
            cut_mode=CutModeEnum.整切,
            plane_configs=[
                PlaneConfig(
                    plane=PlaneTypeEnum.横状位,
                    slice_thickness=1.0,
                    slice_spacing=2.0,
                    window_presets=[WindowPresetEnum.骨窗],
                ),
            ],
        )
        context = ProcessingContext(request)
        context.set_result("localization", {
            "raw_for_reconstruction_path": str(raw_path),
        })
        context.set_result("mapping", {"plans": [plan]})
        return context

    @staticmethod
    def _make_ramp_image(size: tuple[int, int, int]) -> sitk.Image:
        size_x, size_y, size_z = size
        array = np.zeros((size_z, size_y, size_x), dtype=np.int16)
        for z in range(size_z):
            for y in range(size_y):
                for x in range(size_x):
                    array[z, y, x] = x + 10 * y + 100 * z
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection((
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ))
        return image


if __name__ == "__main__":
    unittest.main()
