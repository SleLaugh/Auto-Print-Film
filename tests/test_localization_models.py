import unittest

from src.models import LocalizationResult, NiftiImageInfo


class LocalizationModelsTest(unittest.TestCase):
    """验证阶段三定位实体的序列化和调用兼容性。"""

    def test_localization_result_keeps_legacy_positional_constructor(self):
        """新增全部切法配置后，旧位置参数构造仍应保持原字段顺序。"""

        resolved_cut_config = {"start": "lung", "end": "lung"}
        result = LocalizationResult(
            "肺部",
            "whole_range",
            "/tmp/seg.nii.gz",
            {"model_id": 1},
            "/tmp/dataset.json",
            NiftiImageInfo(
                size_xyz=[2, 2, 2],
                spacing_xyz=[1.0, 1.0, 1.0],
                origin_xyz=[0.0, 0.0, 0.0],
                direction=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            {"start": "lung", "end": "lung"},
            resolved_cut_config,
            {},
            {},
        )

        self.assertEqual(result.resolved_cut_config, resolved_cut_config)
        self.assertEqual(result.cut_configs, {})
        self.assertEqual(result.to_dict()["cut_configs"], {})


if __name__ == "__main__":
    unittest.main()
