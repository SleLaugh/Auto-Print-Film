import sys
import unittest

import main
from src.models.enums import PlaneTypeEnum, WindowPresetEnum


class MainPlaneConfigsTest(unittest.TestCase):
    def test_parse_plane_configs_json_string(self):
        configs = main.parse_plane_configs(
            '[{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗","肺窗"]}]'
        )

        self.assertIs(configs[0].plane, PlaneTypeEnum.冠状位)
        self.assertEqual(configs[0].slice_thickness, 5.0)
        self.assertEqual(configs[0].slice_spacing, 10.0)
        self.assertEqual(configs[0].window_presets, [
            WindowPresetEnum.骨窗,
            WindowPresetEnum.肺窗,
        ])

    def test_parse_plane_configs_none_uses_request_default(self):
        self.assertIsNone(main.parse_plane_configs(None))

    def test_parse_plane_configs_rejects_non_array_json(self):
        with self.assertRaisesRegex(ValueError, "plane_configs 必须是 JSON 数组"):
            main.parse_plane_configs('{"plane":"冠状位"}')

    def test_parse_arguments_accepts_plane_configs_alias(self):
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "-i",
                "/tmp/dicom",
                "-o",
                "/tmp/output",
                "--plane-configs",
                '[{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗"]}]',
            ]

            args = main.parse_arguments()

            self.assertEqual(
                args.plane_configs,
                '[{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗"]}]',
            )
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
