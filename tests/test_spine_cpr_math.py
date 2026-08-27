"""CPR 基础向量模块的独立契约测试。"""

import unittest

from src.models import Point3D
from src.planning.spine_cpr_math import add, cross, dot, midpoint, scale


class SpineCprMathTest(unittest.TestCase):
    """验证 CPR 数学辅助函数与原规划实现保持一致。"""

    def test_vector_operations(self):
        left = Point3D(1.0, 2.0, 3.0)
        right = Point3D(4.0, 5.0, 6.0)

        self.assertEqual(add(left, right), Point3D(5.0, 7.0, 9.0))
        self.assertEqual(scale(left, 2.0), Point3D(2.0, 4.0, 6.0))
        self.assertEqual(dot(left, right), 32.0)
        self.assertEqual(cross(left, right), Point3D(-3.0, 6.0, -3.0))

    def test_midpoint(self):
        points = [Point3D(0.0, 0.0, 0.0), Point3D(2.0, 4.0, 6.0)]

        self.assertEqual(midpoint(points), Point3D(1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
