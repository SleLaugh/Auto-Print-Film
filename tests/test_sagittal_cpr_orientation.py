"""验证矢状位 CPR 的解剖显示方向贯穿存储翻转、取景和 DICOM 写出。"""

import copy
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_windowing
import SimpleITK as sitk

from src.models import WindowPreset
from src.models.enums import PlaneTypeEnum, SliceType
from src.reconstruction.curved_stack import CurvedStackResampler
from src.reconstruction.dicom import new_uid, write_ct_dicom
from src.reconstruction.framing import curved_group_bounds, frame_curved_slice
from tests import test_sagittal_cpr_coverage as coverage_fixtures


class SagittalCprOrientationTest(unittest.TestCase):
    """相同物理解剖采用不同体素存储方向时，椎体始终在左、棘突在右。"""

    FIXTURES = coverage_fixtures.SagittalCprCoverageTest
    GEOMETRIES = ((False, False), (True, False), (True, True))

    @staticmethod
    def _flip_storage(localization, image, flips):
        """只反转体素存储，同步索引标注；不改变解剖结构的物理位置。"""

        localization = copy.deepcopy(localization)
        size = image.GetSize()
        flipped = sitk.Flip(image, list(flips))
        bboxes = [localization["correction_scope_bbox"]]
        for annotation in localization["coords"].values():
            centroid = annotation["centroid"]
            bboxes.append(annotation["bbox"])
            for axis, length, should_flip in zip("xyz", size, flips):
                if should_flip:
                    centroid[axis] = length - 1 - centroid[axis]
        for bbox in bboxes:
            for axis, length, should_flip in zip("xyz", size, flips):
                if should_flip:
                    low, high = bbox[f"{axis}_min"], bbox[f"{axis}_max"]
                    bbox[f"{axis}_min"] = length - 1 - high
                    bbox[f"{axis}_max"] = length - 1 - low
        localization["image"].update({
            "size_xyz": list(flipped.GetSize()),
            "spacing_xyz": list(flipped.GetSpacing()),
            "origin_xyz": list(flipped.GetOrigin()),
            "direction": list(flipped.GetDirection()),
        })
        return localization, flipped

    def _assert_posterior_on_right(self, pixels):
        body_columns = np.nonzero((pixels >= 90.0) & (pixels <= 110.0))[1]
        posterior_columns = np.nonzero(pixels >= 850.0)[1]
        self.assertGreater(body_columns.size, 0, "成片应包含椎体标记")
        self.assertGreater(posterior_columns.size, 0, "成片应包含棘突标记")
        self.assertGreater(
            posterior_columns.mean(), body_columns.mean(),
            "椎体应在左，后方棘突应在右",
        )

    def _assert_patient_orientation(self, output):
        axes = np.asarray(output.GetDirection()).reshape(3, 3)
        # SimpleITK 物理坐标为 LPS，固定患者参考不随源体素轴翻转。
        self.assertGreater(axes[1, 0], 0.0, "画面向右应指向患者后方")
        self.assertLess(axes[2, 1], 0.0, "画面向下应指向患者足侧")
        self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0, places=6)

    def test_storage_flips_keep_posterior_right_and_superior_up(self):
        for slice_part, labels in self.FIXTURES.CASES:
            for oblique in (False, True):
                localization, source, _ = self.FIXTURES._asymmetric_spine(
                    slice_part, labels, oblique=oblique,
                )
                for flips in itertools.product((False, True), repeat=3):
                    with self.subTest(
                        slice_part=slice_part.value, oblique=oblique, flips=flips,
                    ):
                        reflected, image = self._flip_storage(
                            localization, source, flips,
                        )
                        plan = self.FIXTURES._plan(
                            slice_part, reflected,
                            plane=PlaneTypeEnum.矢状位, slice_type=SliceType.CPR,
                        )
                        layer = min(plan["slices"], key=lambda item: abs(item["offset_mm"]))
                        output = CurvedStackResampler().sample_slice(image, plan, layer)
                        self._assert_posterior_on_right(sitk.GetArrayFromImage(output)[0])
                        self._assert_patient_orientation(output)

    def test_framed_dicom_keeps_anatomy_geometry_and_windowed_pixels(self):
        window = WindowPreset("骨窗", 2500.0, 400.0)
        with tempfile.TemporaryDirectory() as directory:
            for slice_part, labels in self.FIXTURES.CASES:
                for oblique, curved in self.GEOMETRIES:
                    localization, image, target = self.FIXTURES._asymmetric_spine(
                        slice_part, labels, oblique=oblique, curved=curved,
                    )
                    for fill_percent in (0, 80):
                        with self.subTest(
                            slice_part=slice_part.value, oblique=oblique,
                            curved=curved, fill_percent=fill_percent,
                        ):
                            plan = self.FIXTURES._plan(
                                slice_part, localization,
                                plane=PlaneTypeEnum.矢状位, slice_type=SliceType.CPR,
                                fill_percent=fill_percent,
                            )
                            sampler = CurvedStackResampler()
                            masks = [
                                sampler.sample_slice(target, plan, layer, is_mask=True)
                                for layer in plan["slices"]
                            ]
                            bounds = curved_group_bounds(masks)
                            center_index = min(
                                range(len(plan["slices"])),
                                key=lambda index: abs(plan["slices"][index]["offset_mm"]),
                            )
                            layer = plan["slices"][center_index]
                            sampled = sampler.sample_slice(image, plan, layer)
                            output, framing = frame_curved_slice(
                                sampled, masks[center_index], fill_percent, bounds,
                            )
                            source_hu = sitk.GetArrayFromImage(output)[0]
                            self._assert_posterior_on_right(source_hu)
                            self._assert_patient_orientation(output)
                            path = Path(directory) / "sagittal.dcm"
                            write_ct_dicom(
                                image=output, output_path=path, plan=plan,
                                planned_slice={**layer, "framing": framing},
                                window=window, series_uid=new_uid(), study_uid=new_uid(),
                                total_slices=len(plan["slices"]),
                            )
                            dataset = pydicom.dcmread(path)
                            axes = np.asarray(output.GetDirection()).reshape(3, 3)
                            np.testing.assert_allclose(
                                dataset.ImageOrientationPatient,
                                np.concatenate((axes[:, 0], axes[:, 1])),
                                rtol=0.0, atol=1e-8,
                            )
                            np.testing.assert_allclose(
                                dataset.ImagePositionPatient, output.GetOrigin(),
                                rtol=0.0, atol=1e-8,
                            )
                            np.testing.assert_allclose(
                                dataset.PixelSpacing, output.GetSpacing()[1::-1],
                                rtol=0.0, atol=1e-8,
                            )
                            # 独立按 LINEAR_EXACT 定义核对存储灰度及真实查看器显示。
                            expected_display = np.clip(
                                (source_hu.astype(np.float64) - window.window_center)
                                / window.window_width + 0.5,
                                0.0, 1.0,
                            )
                            np.testing.assert_array_equal(
                                dataset.pixel_array,
                                np.rint(expected_display * 65535.0).astype(np.uint16),
                            )
                            modality = apply_modality_lut(dataset.pixel_array, dataset)
                            self._assert_posterior_on_right(modality)
                            viewed = apply_windowing(modality, dataset)
                            normalized = (
                                viewed - float(dataset.RescaleIntercept)
                            ) / (65535.0 * float(dataset.RescaleSlope))
                            np.testing.assert_allclose(
                                normalized, expected_display, rtol=0.0,
                                atol=0.5 / 65535.0 + 1e-7,
                            )

    def test_coronal_sampling_is_unchanged_by_planned_row_reference(self):
        for slice_part, labels in self.FIXTURES.CASES:
            for oblique, curved in self.GEOMETRIES:
                with self.subTest(
                    slice_part=slice_part.value, oblique=oblique, curved=curved,
                ):
                    localization, image, _ = self.FIXTURES._asymmetric_spine(
                        slice_part, labels, oblique=oblique, curved=curved,
                    )
                    plan = self.FIXTURES._plan(
                        slice_part, localization,
                        plane=PlaneTypeEnum.冠状位, slice_type=SliceType.CPR,
                    )
                    layer = min(plan["slices"], key=lambda item: abs(item["offset_mm"]))
                    without_reference = {
                        key: value for key, value in layer.items() if key != "row_dir"
                    }
                    sampler = CurvedStackResampler()
                    actual = sampler.sample_slice(image, plan, layer)
                    original = sampler.sample_slice(image, plan, without_reference)
                    np.testing.assert_array_equal(
                        sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(original),
                    )
                    self.assertEqual(actual.GetOrigin(), original.GetOrigin())
                    self.assertEqual(actual.GetDirection(), original.GetDirection())


if __name__ == "__main__":
    unittest.main()
