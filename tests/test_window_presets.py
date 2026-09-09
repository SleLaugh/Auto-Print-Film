import json
import tempfile
import unittest
from pathlib import Path

import src.models as models
from src.models.window import load_window_presets


class WindowPresetsTest(unittest.TestCase):
    """汇总 `WindowPresetsTest` 相关测试场景。"""

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
        # 默认配置文件是未被本次请求覆盖时的名称和值来源。
        """验证 `test_all_presets_have_expected_width_and_center` 场景符合预期。"""

        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        self.assertEqual(set(load_window_presets()), set(self.EXPECTED_PRESETS))
        for name, (width, center) in self.EXPECTED_PRESETS.items():
            with self.subTest(name=name):
                preset = get_window_preset(name)
                self.assertEqual(preset.name, name)
                self.assertEqual(preset.window_width, width)
                self.assertEqual(preset.window_center, center)

    def test_custom_presets_override_defaults_and_fallback_by_name(self):
        """本次列表按名称覆盖默认值，未覆盖名称继续使用默认值。"""

        custom = models.parse_window_presets([
            {"Name": "骨窗", "windowWidth": 50, "windowLevel": 60},
            {"Name": "骨窗1", "windowWidth": 80, "windowLevel": 70},
        ])

        overridden = models.get_window_preset("骨窗", custom)
        fallback = models.get_window_preset("脑窗", custom)
        self.assertEqual(
            (overridden.window_width, overridden.window_center),
            (50, 60),
        )
        self.assertEqual(
            (fallback.window_width, fallback.window_center),
            self.EXPECTED_PRESETS["脑窗"],
        )

    def test_custom_presets_unknown_name_still_fails(self):
        """自定义列表和默认配置都没有的名称必须报错。"""

        custom = models.parse_window_presets([
            {"Name": "骨窗1", "windowWidth": 80, "windowLevel": 70},
        ])
        with self.assertRaisesRegex(ValueError, "不支持的窗宽窗位预设"):
            models.get_window_preset("骨窗2", custom)

    def test_query_accepts_chinese_name(self):
        # 简单调用可以直接使用中文名称，查询结果仍来自同一份内部预设表。
        """验证 `test_query_accepts_chinese_name` 场景符合预期。"""

        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        preset = get_window_preset("脑窗")

        self.assertEqual(preset.name, "脑窗")
        self.assertEqual(preset.window_width, 80)
        self.assertEqual(preset.window_center, 40)

    def test_query_rejects_english_name(self):
        """验证 `test_query_rejects_english_name` 场景符合预期。"""

        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        with self.assertRaisesRegex(ValueError, "不支持的窗宽窗位预设"):
            get_window_preset("Brain")

    def test_query_rejects_unknown_chinese_name(self):
        """验证 `test_query_rejects_unknown_chinese_name` 场景符合预期。"""

        get_window_preset = getattr(models, "get_window_preset", None)
        self.assertIsNotNone(get_window_preset)

        with self.assertRaisesRegex(ValueError, "不支持的窗宽窗位预设"):
            get_window_preset("血管窗")

    def test_loader_accepts_new_name_without_python_enum(self):
        """验证 `test_loader_accepts_new_name_without_python_enum` 场景符合预期。"""

        presets = self._load_temp_config([
            {"name": "血管窗", "window_width": 700, "window_center": 100},
        ])

        self.assertEqual(presets["血管窗"].window_width, 700)
        self.assertEqual(presets["血管窗"].window_center, 100)

    def test_loader_rejects_duplicate_names(self):
        """验证 `test_loader_rejects_duplicate_names` 场景符合预期。"""

        with self.assertRaisesRegex(ValueError, "重复"):
            self._load_temp_config([
                {"name": "骨窗", "window_width": 1000, "window_center": 100},
                {"name": "骨窗", "window_width": 2000, "window_center": 200},
            ])

    def test_loader_rejects_non_positive_width(self):
        """验证 `test_loader_rejects_non_positive_width` 场景符合预期。"""

        with self.assertRaisesRegex(ValueError, "window_width"):
            self._load_temp_config([
                {"name": "骨窗", "window_width": 0, "window_center": 100},
            ])

    @staticmethod
    def _load_temp_config(entries):
        """为测试准备 `_load_temp_config` 所需的数据或记录行为。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "window_presets.json"
            path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
            return load_window_presets(path)


if __name__ == "__main__":
    unittest.main()
