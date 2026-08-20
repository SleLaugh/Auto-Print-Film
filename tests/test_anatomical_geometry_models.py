import math
import unittest

from src.models import Point3D


class AnatomicalGeometryModelsTest(unittest.TestCase):
    """验证第三、四步之间统一解剖几何实体的校验和序列化契约。"""

    def _make_frame(self):
        """构造一个标准正交坐标系，供各实体测试复用。"""

        from src.models import AnatomicalBounds, AnatomicalFrame, AxisBounds

        return AnatomicalFrame(
            origin=Point3D(10.0, 20.0, 30.0),
            left_right_axis=Point3D(1.0, 0.0, 0.0),
            anterior_posterior_axis=Point3D(0.0, 1.0, 0.0),
            superior_inferior_axis=Point3D(0.0, 0.0, 1.0),
            bounds=AnatomicalBounds(
                left_right=AxisBounds(1.0, 5.0),
                anterior_posterior=AxisBounds(2.0, 8.0),
                superior_inferior=AxisBounds(3.0, 11.0),
            ),
            method="test_frame",
            confidence=0.9,
            applied=True,
            warnings=["测试警告"],
        )

    def test_result_serializes_global_and_local_frames(self):
        """全局及局部坐标系应同时写入阶段结果。"""

        from src.models import AnatomicalGeometryResult

        frame = self._make_frame()
        result = AnatomicalGeometryResult(
            global_frame=frame,
            local_frames={"L1_L2": frame},
        )

        data = result.to_dict()

        self.assertEqual(data["global_frame"]["origin"], {"x": 10.0, "y": 20.0, "z": 30.0})
        self.assertEqual(data["global_frame"]["bounds"]["superior_inferior"], {
            "minimum": 3.0,
            "maximum": 11.0,
        })
        self.assertEqual(data["global_frame"]["method"], "test_frame")
        self.assertIn("L1_L2", data["local_frames"])

    def test_result_round_trips_from_dict(self):
        """实体经过字典序列化和反序列化后不应丢失字段。"""

        from src.models import AnatomicalGeometryResult

        expected = AnatomicalGeometryResult(
            global_frame=self._make_frame(),
            local_frames={},
        )

        actual = AnatomicalGeometryResult.from_dict(expected.to_dict())

        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_axis_bounds_reject_nonfinite_or_reversed_values(self):
        """轴范围应拒绝非有限数值和颠倒的边界。"""

        from src.models import AxisBounds

        with self.assertRaisesRegex(ValueError, "有限数值"):
            AxisBounds(math.nan, 1.0)
        with self.assertRaisesRegex(ValueError, "minimum.*maximum"):
            AxisBounds(2.0, 1.0)

    def test_frame_rejects_invalid_axes_confidence_and_method(self):
        """坐标系应拒绝非单位轴、非正交轴及非法元数据。"""

        from src.models import AnatomicalFrame

        base = self._make_frame().to_dict()

        base["left_right_axis"] = {"x": 2.0, "y": 0.0, "z": 0.0}
        with self.assertRaisesRegex(ValueError, "单位向量"):
            AnatomicalFrame.from_dict(base)

        base = self._make_frame().to_dict()
        base["anterior_posterior_axis"] = {"x": 1.0, "y": 0.0, "z": 0.0}
        with self.assertRaisesRegex(ValueError, "两两正交"):
            AnatomicalFrame.from_dict(base)

        base = self._make_frame().to_dict()
        base["confidence"] = 1.1
        with self.assertRaisesRegex(ValueError, "confidence"):
            AnatomicalFrame.from_dict(base)

        base = self._make_frame().to_dict()
        base["method"] = ""
        with self.assertRaisesRegex(ValueError, "method"):
            AnatomicalFrame.from_dict(base)


if __name__ == "__main__":
    unittest.main()
