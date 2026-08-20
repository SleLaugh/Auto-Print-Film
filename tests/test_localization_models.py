import unittest

from src.models import LocalizationResult, NiftiImageInfo


class LocalizationModelsTest(unittest.TestCase):
    """验证第三至第五阶段调试汇总实体的序列化和字段契约。"""

    def test_localization_result_requires_all_cut_configs(self):
        """最终定位实体必须显式传入当前部位的全部定位范围配置。"""

        with self.assertRaisesRegex(TypeError, "cut_configs"):
            LocalizationResult(
                body_part="肺部",
                localization_profile="whole_range",
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

    def test_localization_result_serializes_spine_geometry_separately(self):
        """第五阶段标注质心应独立序列化，不覆盖第四阶段全量 coords。"""

        result = LocalizationResult(
            body_part="腰椎",
            localization_profile="whole_range",
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
            cut_config={"start": "L1", "end": "S1"},
            cut_configs={"whole_range": {"start": "L1", "end": "S1"}},
            resolved_cut_config={"start": "L1", "end": "S1"},
            resolved_cut_configs={
                "whole_range": {"start": "L1", "end": "S1"},
            },
            label_targets={},
            coords={},
            debug_vertebral_body_prism_mask_overlay_path=(
                "/tmp/椎体四棱柱掩膜.nii.gz"
            ),
            spine_geometry={
                "filtered_annotation": {
                    "method": "vertebral_body_candidate_v1",
                },
                "filtered_annotation_centroids": {
                    "vertebrae_L1": {"found": True},
                },
            },
        )

        data = result.to_dict()

        self.assertIn("spine_geometry", data)
        self.assertEqual(
            data["spine_geometry"]["filtered_annotation_centroids"]
            ["vertebrae_L1"]["found"],
            True,
        )
        self.assertEqual(data["coords"], {})
        self.assertEqual(
            data["debug_vertebral_body_prism_mask_overlay_path"],
            "/tmp/椎体四棱柱掩膜.nii.gz",
        )


if __name__ == "__main__":
    unittest.main()
