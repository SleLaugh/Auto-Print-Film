import unittest

import src.models as models
from src.models import WindowPresetEnum


class WindowPresetsTest(unittest.TestCase):
    EXPECTED_PRESETS = {
        "脑窗": (80, 40),
        "骨窗": (2500, 400),
        "肺窗": (1600, -600),
        "纵隔窗": (350, 40),
        "软组织窗": (400, 50),
        "腹部窗": (350, 50),
        "肝窗": (150, 60),
        "肾窗": (350, 40),
        "胰腺窗": (300, 40),
        "骨盆窗": (400, 40),
        "脊柱窗": (2000, 300),
    }

    def test_all_presets_have_expected_width_and_center(self):
        # 每个中文枚举都必须存在唯一的内部数值定义，避免新增名称后漏配参数。
        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        self.assertEqual(
            {preset.value for preset in WindowPresetEnum},
            set(self.EXPECTED_PRESETS),
        )
        for name, (width, center) in self.EXPECTED_PRESETS.items():
            with self.subTest(name=name):
                preset = get_window_preset(WindowPresetEnum(name))
                self.assertEqual(preset.name, name)
                self.assertEqual(preset.window_width, width)
                self.assertEqual(preset.window_center, center)

    def test_query_accepts_chinese_name(self):
        # 简单调用可以直接使用中文名称，查询结果仍来自同一份内部预设表。
        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        preset = get_window_preset("脑窗")

        self.assertEqual(preset.name, "脑窗")
        self.assertEqual(preset.window_width, 80)
        self.assertEqual(preset.window_center, 40)

    def test_query_rejects_english_name(self):
        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        with self.assertRaisesRegex(ValueError, "不支持的窗宽窗位预设"):
            get_window_preset("Brain")

    def test_query_rejects_unknown_chinese_name(self):
        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        with self.assertRaisesRegex(ValueError, "不支持的窗宽窗位预设"):
            get_window_preset("血管窗")


if __name__ == "__main__":
    unittest.main()
