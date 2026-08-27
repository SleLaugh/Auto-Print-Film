"""脊柱定位使用的几何基础工具。

该 mixin 只提供与医学流程无关的二维轮廓、包围盒和点面判断，
由定位阶段组合使用，保留原有私有方法名以兼容现有调用。
"""

import math

import numpy as np


class LocalizationSpineGeometryMixin:
    """定位阶段的脊柱几何基础工具。"""

    @staticmethod
    def _bbox_from_indices(indices_zyx: np.ndarray) -> dict[str, int]:
        """把 zyx 体素索引转成统一 bbox 字典。"""

        minimum = indices_zyx.min(axis=0)
        maximum = indices_zyx.max(axis=0)
        return {
            "z_min": int(minimum[0]),
            "z_max": int(maximum[0]),
            "y_min": int(minimum[1]),
            "y_max": int(maximum[1]),
            "x_min": int(minimum[2]),
            "x_max": int(maximum[2]),
        }


    @staticmethod
    def _bbox_from_region(
        minimum_zyx: np.ndarray,
        local_mask: np.ndarray,
    ) -> dict[str, int] | None:
        """把局部 mask 区域转成全局 bbox 字典。"""

        local_indices = np.argwhere(local_mask)
        if len(local_indices) == 0:
            return None
        return LocalizationSpineGeometryMixin._bbox_from_indices(local_indices + minimum_zyx)


    @staticmethod
    def _points_inside_convex_polygon(
        points: np.ndarray,
        polygon: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        """判断二维点是否位于有序凸多边形内部，并包含边界。"""

        edges = np.roll(polygon, -1, axis=0) - polygon
        deltas = points[:, None, :] - polygon[None, :, :]
        crosses = (
            edges[None, :, 0] * deltas[:, :, 1]
            - edges[None, :, 1] * deltas[:, :, 0]
        )
        edge_lengths = np.linalg.norm(edges, axis=1)
        if np.any(edge_lengths <= 1e-12):
            return np.zeros(len(points), dtype=bool)
        signed_distances = crosses / edge_lengths[None, :]
        if LocalizationSpineGeometryMixin._signed_area_2d(polygon) > 0.0:
            return np.all(signed_distances >= -tolerance, axis=1)
        return np.all(signed_distances <= tolerance, axis=1)


    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
        """返回二维点集的凸包顶点，按轮廓顺序排列。"""

        if len(points) < 4:
            return points
        ordered = sorted((float(point[0]), float(point[1])) for point in points)
        if len(ordered) < 4:
            return np.asarray(ordered, dtype=float)

        def cross(
            origin: tuple[float, float],
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> float:
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        lower: list[tuple[float, float]] = []
        for point in ordered:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper: list[tuple[float, float]] = []
        for point in reversed(ordered):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        hull = lower[:-1] + upper[:-1]
        return np.asarray(hull, dtype=float)


    @staticmethod
    def _simplify_hull_to_quadrilateral(hull: np.ndarray) -> np.ndarray | None:
        """把轮廓凸包压缩成四个主角点。"""

        if len(hull) < 4:
            return None
        vertices = [np.asarray(point, dtype=float) for point in hull]
        while len(vertices) > 4:
            areas = []
            for index, point in enumerate(vertices):
                previous = vertices[index - 1]
                next_point = vertices[(index + 1) % len(vertices)]
                previous_delta = previous - point
                next_delta = next_point - point
                area = abs(
                    previous_delta[0] * next_delta[1]
                    - previous_delta[1] * next_delta[0]
                ) / 2.0
                areas.append(float(area))
            remove_index = int(np.argmin(areas))
            vertices.pop(remove_index)
        quad = np.asarray(vertices, dtype=float)
        if abs(LocalizationSpineGeometryMixin._signed_area_2d(quad)) <= 1e-6:
            return None
        return quad


    @staticmethod
    def _expand_quadrilateral_to_contain_points(
        quadrilateral: np.ndarray,
        points: np.ndarray,
        tolerance: float = 1e-6,
    ) -> np.ndarray | None:
        """沿四条外法向扩展四边形，保证投影点全部位于其内部。"""

        quad = np.asarray(quadrilateral, dtype=float)
        points = np.asarray(points, dtype=float)
        if quad.shape != (4, 2) or points.ndim != 2 or points.shape[1] != 2:
            return None
        signed_area = LocalizationSpineGeometryMixin._signed_area_2d(quad)
        if abs(signed_area) <= 1e-6 or len(points) == 0:
            return None

        orientation = 1.0 if signed_area > 0.0 else -1.0
        shifted_lines: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(4):
            start = quad[index]
            end = quad[(index + 1) % 4]
            edge = end - start
            edge_length = float(np.linalg.norm(edge))
            if not math.isfinite(edge_length) or edge_length <= 1e-9:
                return None
            signed_distances = orientation * (
                edge[0] * (points[:, 1] - start[1])
                - edge[1] * (points[:, 0] - start[0])
            ) / edge_length
            outward_distance = max(
                0.0,
                float(-np.min(signed_distances)) + float(tolerance),
            )
            outward_normal = np.asarray(
                [orientation * edge[1], -orientation * edge[0]],
                dtype=float,
            ) / edge_length
            shifted_lines.append((
                start + outward_normal * outward_distance,
                edge,
            ))

        expanded_vertices: list[np.ndarray] = []
        for index in range(4):
            previous_start, previous_direction = shifted_lines[index - 1]
            current_start, current_direction = shifted_lines[index]
            denominator = (
                previous_direction[0] * current_direction[1]
                - previous_direction[1] * current_direction[0]
            )
            if abs(float(denominator)) <= 1e-9:
                return None
            delta = current_start - previous_start
            parameter = (
                delta[0] * current_direction[1]
                - delta[1] * current_direction[0]
            ) / denominator
            expanded_vertices.append(
                previous_start + previous_direction * parameter,
            )

        expanded = np.asarray(expanded_vertices, dtype=float)
        if np.all(
            LocalizationSpineGeometryMixin._points_inside_convex_polygon(
                points,
                expanded,
                tolerance=max(float(tolerance), 1e-6),
            )
        ):
            return expanded
        return None


    @staticmethod
    def _oriented_enclosing_quadrilateral(
        points: np.ndarray,
        reference_quadrilateral: np.ndarray,
    ) -> np.ndarray | None:
        """沿主体主边方向生成紧包络，保留椎体自身倾斜角度。"""

        points = np.asarray(points, dtype=float)
        reference = np.asarray(reference_quadrilateral, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) == 0
            or reference.shape != (4, 2)
        ):
            return None

        edges = np.roll(reference, -1, axis=0) - reference
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1e-9):
            return None
        candidate_indices = sorted(
            range(len(edges)),
            key=lambda index: float(lengths[index]),
            reverse=True,
        )
        first_index = candidate_indices[0]
        for index in candidate_indices:
            edge = edges[index]
            if (
                abs(float(edge[0])) > 1e-6
                and abs(float(edge[1])) > 1e-6
                and float(lengths[index]) >= float(lengths[first_index]) * 0.5
            ):
                first_index = index
                break
        axis = edges[first_index] / float(lengths[first_index])
        transverse_axis = np.asarray([-axis[1], axis[0]], dtype=float)

        projections = np.column_stack((
            points @ axis,
            points @ transverse_axis,
        ))
        minimum = np.min(projections, axis=0)
        maximum = np.max(projections, axis=0)
        if np.any(maximum - minimum <= 1e-9):
            return None
        return np.asarray([
            axis * minimum[0] + transverse_axis * minimum[1],
            axis * maximum[0] + transverse_axis * minimum[1],
            axis * maximum[0] + transverse_axis * maximum[1],
            axis * minimum[0] + transverse_axis * maximum[1],
        ], dtype=float)


    @staticmethod
    def _signed_area_2d(points: np.ndarray) -> float:
        """计算二维多边形有向面积。"""

        if len(points) < 3:
            return 0.0
        x_values = points[:, 0]
        y_values = points[:, 1]
        return float(
            0.5 * np.sum(
                x_values * np.roll(y_values, -1)
                - y_values * np.roll(x_values, -1)
            )
        )

