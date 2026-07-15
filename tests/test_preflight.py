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
)
from src.planning.cut_policy import cut_mode_for_plane


class CutPolicyTest(unittest.TestCase):
    """汇总 `CutPolicyTest` 相关测试场景。"""

    def test_only_lumbar_axial_uses_interspace(self):
        """验证 `test_only_lumbar_axial_uses_interspace` 场景符合预期。"""

        self.assertEqual(cut_mode_for_plane("腰椎", "横状位"), "interspace")
        self.assertEqual(cut_mode_for_plane("腰椎", "矢状位"), "whole_range")
        self.assertEqual(cut_mode_for_plane("颈椎", "横状位"), "whole_range")


def make_request(body_part, cut_mode, plane, input_dir="/tmp/dicom", output_dir="/tmp/output"):
    """为测试准备 `make_request` 所需的数据或记录行为。"""

    return Request(
        dicom_dir=input_dir,
        output_dir=output_dir,
        body_part=BodyPartEnum(body_part),
        plane_configs=[PlaneConfig(
            PlaneTypeEnum(plane),
            1.0,
            1.0,
            ["骨窗"],
        )],
    )


class PreflightTest(unittest.TestCase):
    """汇总 `PreflightTest` 相关测试场景。"""

    def test_current_capabilities_register_three_planes_for_whole_range(self):
        """所有已支持整切的部位都应对外声明三个默认直线状位。"""

        combinations = set(CapabilityRegistry.current().combinations())
        for body_part in ("颈椎", "胸椎", "腰椎", "肺部", "头骨", "髋骨"):
            for plane in ("横状位", "矢状位", "冠状位"):
                with self.subTest(body_part=body_part, plane=plane):
                    self.assertIn(
                        (body_part, "whole_range", plane),
                        combinations,
                    )

    def test_registry_accepts_current_lumbar_axial_whole_range(self):
        """验证 `test_registry_accepts_current_lumbar_axial_whole_range` 场景符合预期。"""

        CapabilityRegistry.current().validate(
            make_request("腰椎", "whole_range", "横状位")
        )

    def test_registry_derives_cervical_axial_as_whole_range(self):
        """验证 `test_registry_derives_cervical_axial_as_whole_range` 场景符合预期。"""

        CapabilityRegistry.current().validate(
            make_request("颈椎", "interspace", "横状位")
        )

    def test_registry_allows_future_planes_for_a_supported_workflow(self):
        # 状位由阶段四逐步实现；预检不能阻断同一部位/切法下的后续扩展参数。
        """验证 `test_registry_allows_future_planes_for_a_supported_workflow` 场景符合预期。"""

        request = make_request("腰椎", "whole_range", "矢状位")

        CapabilityRegistry.current().validate(request)

    def test_preflight_failure_does_not_create_output_directory(self):
        """验证 `test_preflight_failure_does_not_create_output_directory` 场景符合预期。"""

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
        """验证 `test_preflight_rejects_output_containing_input` 场景符合预期。"""

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
        """验证 `test_preflight_rejects_output_inside_input` 场景符合预期。"""

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
