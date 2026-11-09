import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig, NnUNetConfig
from src.core.capabilities import CapabilityRegistry
from src.core.errors import CapabilityError, DependencyError, RequestValidationError
from src.core.preflight import PreflightValidator
from src.models import PlaneConfig, Request
from src.models.enums import (
    SlicePartEnum,
    PlaneTypeEnum,
    SliceType,
)
from src.planning.slice_policy import localization_profile_for_slice_type


class SlicePolicyTest(unittest.TestCase):
    """验证 SliceType 只决定显式切法及其定位范围。"""

    def test_slice_types_resolve_localization_profiles(self):
        """MPR/CPR 共用整体范围，椎间隙使用相邻椎体范围。"""

        self.assertEqual(localization_profile_for_slice_type("MPR"), "whole_range")
        self.assertEqual(localization_profile_for_slice_type("CPR"), "whole_range")
        self.assertEqual(localization_profile_for_slice_type("椎间隙"), "interspace")


def make_request(slice_part, slice_type, plane, input_dir="/tmp/dicom", output_dir="/tmp/output"):
    """为测试准备 `make_request` 所需的数据或记录行为。"""

    return Request(
        dicom_dir=input_dir,
        output_dir=output_dir,
        slice_part=SlicePartEnum(slice_part),
        plane_configs=[PlaneConfig(
            PlaneTypeEnum(plane),
            1.0,
            1.0,
            ["骨窗"],
            slice_type=SliceType(slice_type),
        )],
    )


class PreflightTest(unittest.TestCase):
    """汇总 `PreflightTest` 相关测试场景。"""

    def test_current_capabilities_register_three_planes_for_mpr(self):
        """所有已支持部位都应对外声明三个基础 MPR 状位。"""

        combinations = set(CapabilityRegistry.current().combinations())
        for slice_part in ("颈椎", "胸椎", "腰椎", "胸腰椎", "全脊柱", "肺部", "颅脑", "髋骨"):
            for plane in ("横状位", "矢状位", "冠状位"):
                with self.subTest(slice_part=slice_part, plane=plane):
                    self.assertIn(
                        (slice_part, "MPR", plane),
                        combinations,
                    )

    def test_registry_accepts_lumbar_axial_mpr(self):
        """腰椎横状位默认基础切法为 MPR。"""

        CapabilityRegistry.current().validate(
            make_request("腰椎", "MPR", "横状位")
        )

    def test_registry_accepts_lumbar_axial_interspace(self):
        """腰椎横状位额外支持椎间隙切片。"""

        CapabilityRegistry.current().validate(
            make_request("腰椎", "椎间隙", "横状位")
        )

    def test_registry_accepts_cervical_axial_interspace(self):
        """颈椎横状位支持从 C2_C3 开始的椎间隙切片。"""

        CapabilityRegistry.current().validate(
            make_request("颈椎", "椎间隙", "横状位")
        )

    def test_registry_rejects_cervical_interspace_for_non_axial_planes(self):
        """颈椎椎间隙不能用于矢状位或冠状位。"""

        for plane in ("矢状位", "冠状位"):
            with self.subTest(plane=plane):
                with self.assertRaises(CapabilityError):
                    CapabilityRegistry.current().validate(
                        make_request("颈椎", "椎间隙", plane)
                    )

    def test_registry_rejects_cpr_for_lumbar_axial(self):
        """CPR 只允许用于已注册的脊柱冠状位或矢状位。"""

        request = make_request("腰椎", "CPR", "横状位")

        with self.assertRaises(CapabilityError):
            CapabilityRegistry.current().validate(request)

    def test_registry_accepts_sagittal_cpr_for_all_spine_parts(self):
        """各脊柱部位均注册矢状位 CPR 能力。"""

        registry = CapabilityRegistry.current()
        for slice_part in ("颈椎", "胸椎", "腰椎", "胸腰椎", "全脊柱"):
            with self.subTest(slice_part=slice_part):
                registry.validate(make_request(slice_part, "CPR", "矢状位"))

    def test_registry_accepts_coronal_cpr_for_extended_spine_parts(self):
        """胸腰椎和全脊柱均允许冠状位 CPR。"""

        registry = CapabilityRegistry.current()
        for slice_part in ("胸腰椎", "全脊柱"):
            with self.subTest(slice_part=slice_part):
                registry.validate(make_request(slice_part, "CPR", "冠状位"))

    def test_registry_rejects_unsupported_cuts_for_extended_spine_parts(self):
        """胸腰椎和全脊柱的横状位仅支持 MPR，并且不提供椎间隙切法。"""

        registry = CapabilityRegistry.current()
        unsupported = (
            ("CPR", "横状位"),
            ("椎间隙", "横状位"),
            ("椎间隙", "矢状位"),
            ("椎间隙", "冠状位"),
        )
        for slice_part in ("胸腰椎", "全脊柱"):
            for slice_type, plane in unsupported:
                with self.subTest(slice_part=slice_part, slice_type=slice_type, plane=plane):
                    with self.assertRaises(CapabilityError):
                        registry.validate(make_request(slice_part, slice_type, plane))

    def test_preflight_failure_does_not_create_output_directory(self):
        """验证 `test_preflight_failure_does_not_create_output_directory` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "dicom"
            input_dir.mkdir()
            output_dir = root / "not-created"
            request = make_request(
                "腰椎",
                "MPR",
                "横状位",
                str(input_dir),
                str(output_dir),
            )
            validator = PreflightValidator(executable_finder=lambda _: None)

            with self.assertRaises(DependencyError):
                validator.validate(request, AppConfig())

            self.assertFalse(output_dir.exists())

    def test_preflight_accepts_only_the_trained_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "dicom"
            input_dir.mkdir()
            results_dir = root / "results"
            results_dir.mkdir()
            output_dir = root / "output"
            config = AppConfig(nnunet=NnUNetConfig(results=str(results_dir)))

            PreflightValidator(executable_finder=lambda _: "/bin/true").validate(
                make_request("腰椎", "MPR", "横状位", str(input_dir), str(output_dir)),
                config,
            )

            self.assertFalse(output_dir.exists())

    def test_preflight_rejects_missing_or_invalid_trained_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "dicom"
            input_dir.mkdir()
            regular_file = root / "checkpoint.pth"
            regular_file.touch()
            output_dir = root / "output"
            request = make_request("腰椎", "MPR", "横状位", str(input_dir), str(output_dir))
            validator = PreflightValidator(executable_finder=lambda _: "/bin/true")

            for results in ("", str(root / "missing"), str(regular_file)):
                with self.subTest(results=results), self.assertRaisesRegex(DependencyError, "nnUNet_results"):
                    validator.validate(request, AppConfig(nnunet=NnUNetConfig(results=results)))

            self.assertFalse(output_dir.exists())

    def test_preflight_rejects_output_containing_input(self):
        """验证 `test_preflight_rejects_output_containing_input` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            input_dir = output_dir / "dicom"
            input_dir.mkdir(parents=True)
            request = make_request(
                "腰椎", "MPR", "横状位",
                str(input_dir), str(output_dir),
            )

            with self.assertRaisesRegex(RequestValidationError, "输入目录"):
                PreflightValidator(
                    executable_finder=lambda _: "/bin/true",
                    check_model_directories=False,
                ).validate(request, AppConfig(nnunet=NnUNetConfig()))

    def test_preflight_rejects_output_inside_input(self):
        """验证 `test_preflight_rejects_output_inside_input` 场景符合预期。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "dicom"
            input_dir.mkdir()
            output_dir = input_dir / "generated"
            request = make_request(
                "腰椎", "MPR", "横状位",
                str(input_dir), str(output_dir),
            )

            with self.assertRaisesRegex(RequestValidationError, "输出目录"):
                PreflightValidator(
                    executable_finder=lambda _: "/bin/true",
                    check_model_directories=False,
                ).validate(request, AppConfig())

    def test_preflight_rejects_cache_inside_output(self):
        """缓存放在输出目录内会被 -o 清理，必须提前拒绝。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "dicom"
            input_dir.mkdir()
            output_dir = root / "output"
            request = make_request(
                "腰椎", "MPR", "横状位",
                str(input_dir), str(output_dir),
            )
            request.cache_dir = str(output_dir / "cache")

            with self.assertRaisesRegex(RequestValidationError, "缓存目录"):
                PreflightValidator(
                    executable_finder=lambda _: "/bin/true",
                    check_model_directories=False,
                ).validate(request, AppConfig())

if __name__ == "__main__":
    unittest.main()
