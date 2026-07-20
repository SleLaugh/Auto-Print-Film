import unittest

from src.models import LocalizationResult, NiftiImageInfo


class LocalizationModelsTest(unittest.TestCase):
    """验证阶段三定位实体的序列化和字段契约。"""

    def test_localization_result_requires_all_cut_configs(self):
        """最终定位实体必须显式传入当前部位的全部切法配置。"""

        with self.assertRaisesRegex(TypeError, "cut_configs"):
            LocalizationResult(
                body_part="肺部",
                cut_mode="whole_range",
                seg_path="/tmp/seg.nii.gz",
                model={"model_id": 1},
                dataset_json_path="/tmp/dataset.json",
                image=NiftiImageInfo(
                    size_xyz=[2, 2, 2],
                    spacing_xyz=[1.0, 1.0, 1.0],
                    origin_xyz=[0.0, 0.0, 0.0],
                    direction=[
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0,
                    ],
                ),
                cut_config={"start": "lung", "end": "lung"},
                resolved_cut_config={"start": "lung", "end": "lung"},
                label_targets={},
                coords={},
            )


if __name__ == "__main__":
    unittest.main()
