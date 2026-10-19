"""从正式规划到成片取景验证矢状位 CPR 的完整前后覆盖。"""

import math
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.models import PlaneConfig, Request
from src.models.enums import PlaneTypeEnum, SlicePartEnum, SliceType
from src.planning.base import select_planner
from src.reconstruction.curved_stack import CurvedStackResampler
from src.reconstruction.framing import curved_group_bounds, frame_curved_slice
from tests import test_stage6_slice_planning as planning_fixtures


class SagittalCprCoverageTest(unittest.TestCase):
    """完整标签比主体中心线更偏后时，后方骨性结构仍应进入成片。"""

    CASES = (
        (SlicePartEnum.颈椎, ("C2", "C3", "C4")),
        (SlicePartEnum.胸椎, ("T1", "T2", "T3")),
        (SlicePartEnum.腰椎, ("L1", "L2", "L3")),
    )

    @staticmethod
    def _asymmetric_spine(slice_part, labels, *, oblique=False, curved=False):
        """保留真实配置契约，用独立后方标记区分完整标签和主体中心线。"""

        result = planning_fixtures.Stage6SlicePlanningTest._spine_localization_result(
            labels, slice_part.value,
        )
        hu = np.zeros((40, 120, 120), dtype=np.float32)
        target = np.zeros_like(hu, dtype=np.uint8)
        for index, coord in enumerate(result["coords"].values()):
            z = 26 - index * 8
            coord["centroid"] = {
                "x": 60.0,
                "y": 54.0 if curved and index == 1 else 50.0,
                "z": float(z),
            }
            coord["bbox"] = {
                "x_min": 55, "x_max": 65,
                "y_min": 45, "y_max": 95,
                "z_min": z - 4, "z_max": z + 4,
            }
            target[z - 4:z + 5, 45:96, 55:66] = 1
            hu[z - 4:z + 5, 45:96, 55:66] = 100.0
            hu[z - 4:z + 5, 80:96, 55:66] = 900.0

        result["correction_scope_bbox"] = {
            "x_min": 55, "x_max": 65,
            "y_min": 45, "y_max": 95,
            "z_min": 6, "z_max": 30,
        }
        if oblique:
            angle = math.radians(25.0)
            result["image"]["direction"] = [
                math.cos(angle), -math.sin(angle), 0.0,
                math.sin(angle), math.cos(angle), 0.0,
                0.0, 0.0, 1.0,
            ]
            result["image"]["origin_xyz"] = [11.0, -7.0, 4.0]

        images = [sitk.GetImageFromArray(hu), sitk.GetImageFromArray(target)]
        for image in images:
            image.SetSpacing(tuple(result["image"]["spacing_xyz"]))
            image.SetOrigin(tuple(result["image"]["origin_xyz"]))
            image.SetDirection(tuple(result["image"]["direction"]))
        return result, *images

    @staticmethod
    def _plan(slice_part, localization, *, plane, slice_type, fill_percent=0):
        request = Request(
            dicom_dir="/tmp/dicom",
            output_dir="/tmp/output",
            slice_part=slice_part,
            plane_configs=[PlaneConfig(
                plane, 1.0, 5.0, ["骨窗"],
                slice_type=slice_type,
                fill_percent=fill_percent,
            )],
        )
        return select_planner(slice_part.value, slice_type.value).build_plans(
            request, localization,
        ).plans[0].to_dict()

    def test_posterior_structure_survives_sampling_and_group_framing(self):
        """在三个部位、斜扫描和弯曲中心线上验证实际 HU 成片覆盖。"""

        for slice_part, labels in self.CASES:
            for oblique, curved in ((False, False), (True, False), (True, True)):
                localization, image, target = self._asymmetric_spine(
                    slice_part, labels, oblique=oblique, curved=curved,
                )
                for fill_percent in (0, 80):
                    with self.subTest(
                        slice_part=slice_part.value,
                        oblique=oblique,
                        curved=curved,
                        fill_percent=fill_percent,
                    ):
                        plan = self._plan(
                            slice_part, localization,
                            plane=PlaneTypeEnum.矢状位,
                            slice_type=SliceType.CPR,
                            fill_percent=fill_percent,
                        )
                        resampler = CurvedStackResampler()
                        sampled = [
                            resampler.sample_slice(image, plan, layer)
                            for layer in plan["slices"]
                        ]
                        masks = [
                            resampler.sample_slice(target, plan, layer, is_mask=True)
                            for layer in plan["slices"]
                        ]
                        bounds = curved_group_bounds(masks)
                        center_index = min(
                            range(len(plan["slices"])),
                            key=lambda index: abs(plan["slices"][index]["offset_mm"]),
                        )
                        raw = sampled[center_index]
                        framed, _ = frame_curved_slice(
                            raw, masks[center_index], fill_percent, bounds,
                        )
                        for output in (raw, framed):
                            pixels = sitk.GetArrayFromImage(output)[0]
                            self.assertGreaterEqual(float(pixels.max()), 899.0)
                            # At least 13 mm of the 16 mm posterior band must
                            # survive; detecting one bright edge pixel is not
                            # enough to establish complete posterior coverage.
                            posterior_width = np.count_nonzero(
                                pixels >= 850.0, axis=1,
                            ).max() * output.GetSpacing()[0]
                            self.assertGreaterEqual(posterior_width, 13.0)

    def test_new_width_calculation_is_confined_to_sagittal_cpr(self):
        """冠状位 CPR 和矢状位 MPR 不应进入新增的曲面宽度计算。"""

        for slice_part, labels in self.CASES:
            with self.subTest(slice_part=slice_part.value):
                localization, _, _ = self._asymmetric_spine(slice_part, labels)
                with patch(
                    "src.planning.spine_cpr._resolve_centered_cpr_width_extent_mm",
                    side_effect=AssertionError("Only sagittal CPR needs centered coverage"),
                ):
                    coronal_cpr = self._plan(
                        slice_part, localization,
                        plane=PlaneTypeEnum.冠状位, slice_type=SliceType.CPR,
                    )
                    sagittal_mpr = self._plan(
                        slice_part, localization,
                        plane=PlaneTypeEnum.矢状位, slice_type=SliceType.MPR,
                    )
                self.assertEqual(coronal_cpr["geometry_type"], "curved_stack")
                self.assertEqual(coronal_cpr["extent_mm"]["width"], 10.0)
                self.assertEqual(sagittal_mpr["geometry_type"], "plane_stack")


if __name__ == "__main__":
    unittest.main()
