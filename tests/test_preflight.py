import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig, NnUNetConfig
from src.core.capabilities import CapabilityRegistry
from src.core.errors import CapabilityError, DependencyError, RequestValidationError
from src.core.preflight import PreflightValidator
from src.models import PlaneConfig, Request
from src.models.enums import (
    BodyPartEnum,
    CutModeEnum,
    PlaneTypeEnum,
    WindowPresetEnum,
)


def make_request(body_part, cut_mode, plane, input_dir="/tmp/dicom", output_dir="/tmp/output"):
    return Request(
        dicom_dir=input_dir,
        output_dir=output_dir,
        body_part=BodyPartEnum(body_part),
        cut_mode=CutModeEnum(cut_mode),
        plane_configs=[PlaneConfig(
            PlaneTypeEnum(plane),
            1.0,
            1.0,
            [WindowPresetEnum.骨窗],
        )],
    )


class PreflightTest(unittest.TestCase):
    def test_registry_accepts_current_lumbar_axial_whole_range(self):
        CapabilityRegistry.current().validate(
            make_request("腰椎", "whole_range", "横状位")
        )

    def test_registry_rejects_unimplemented_cervical_interspace(self):
        with self.assertRaises(CapabilityError) as captured:
            CapabilityRegistry.current().validate(
                make_request("颈椎", "interspace", "横状位")
            )

        self.assertEqual(captured.exception.details["body_part"], "颈椎")

    def test_registry_allows_future_planes_for_a_supported_workflow(self):
        # 状位由阶段四逐步实现；预检不能阻断同一部位/切法下的后续扩展参数。
        request = make_request("腰椎", "whole_range", "矢状位")

        CapabilityRegistry.current().validate(request)

    def test_preflight_failure_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "dicom"
            input_dir.mkdir()
            output_dir = root / "not-created"
            request = make_request(
                "腰椎",
                "whole_range",
                "横状位",
                str(input_dir),
                str(output_dir),
            )
            validator = PreflightValidator(executable_finder=lambda _: None)

            with self.assertRaises(DependencyError):
                validator.validate(request, AppConfig())

            self.assertFalse(output_dir.exists())

    def test_preflight_rejects_output_containing_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            input_dir = output_dir / "dicom"
            input_dir.mkdir(parents=True)
            request = make_request(
                "腰椎", "whole_range", "横状位",
                str(input_dir), str(output_dir),
            )

            with self.assertRaisesRegex(RequestValidationError, "输入目录"):
                PreflightValidator(
                    executable_finder=lambda _: "/bin/true",
                    check_model_directories=False,
                ).validate(request, AppConfig(nnunet=NnUNetConfig()))

    def test_preflight_rejects_output_inside_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "dicom"
            input_dir.mkdir()
            output_dir = input_dir / "generated"
            request = make_request(
                "腰椎", "whole_range", "横状位",
                str(input_dir), str(output_dir),
            )

            with self.assertRaisesRegex(RequestValidationError, "输出目录"):
                PreflightValidator(
                    executable_finder=lambda _: "/bin/true",
                    check_model_directories=False,
                ).validate(request, AppConfig())


if __name__ == "__main__":
    unittest.main()
