"""由颅骨左右对称面建立保留原前后倾角的三维重建坐标。"""

from typing import Any

import numpy as np
import SimpleITK as sitk

from src.anatomy.symmetry import (
    build_symmetry_frame_axes,
    estimate_reflection_symmetry_plane,
)
from src.anatomy.orientation import patient_reference_axes
from src.models import (
    AnatomicalBounds,
    AnatomicalFrame,
    AnatomicalGeometryResult,
    AxisBounds,
    Point3D,
)


def _point(vector: np.ndarray) -> Point3D:
    """将物理位置或单位方向转换成阶段契约实体。"""

    return Point3D(*(float(value) for value in vector))


def estimate_skull_frame_with_diagnostics(
    target_mask: np.ndarray,
    image: sitk.Image,
) -> tuple[AnatomicalGeometryResult, dict[str, Any]]:
    """纠正头部左右旋转、侧倾，匹配不足时保留输入坐标并报告原因。

    颅骨边界用于估计中矢状对称面；原 AP 方向投影到该平面后构造 AP/SI，
    不额外估计或调整对称面内的前后倾角。目标整体质心及三轴范围均从全部
    目标体素计算，不能因拟合点数上限而裁掉末端或面部。
    """

    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 3 or mask.shape != tuple(reversed(image.GetSize())):
        raise ValueError("头骨 Mask 尺寸与三维输入图像不一致")
    if not np.any(mask):
        raise ValueError("头骨 Mask 为空")
    image_direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    # 参考轴按患者头脚方向定符号，物理点仍由未改写的存储方向换算。
    reference_axes = patient_reference_axes(image_direction)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    # 只将表面点送入有限点云优化；内部体素过密时会掩盖轮廓不匹配。
    contour_image = sitk.BinaryContour(sitk.GetImageFromArray(mask.astype(np.uint8)))
    contour = sitk.GetArrayFromImage(contour_image) != 0
    indices_xyz = np.argwhere(contour)[:, ::-1].astype(float)
    boundary_points = origin + (indices_xyz * spacing) @ image_direction.T
    lr, plane_center, symmetry = estimate_reflection_symmetry_plane(boundary_points, reference_axes)
    final_axes = build_symmetry_frame_axes(lr, reference_axes)
    ap, si = final_axes[:, 1], final_axes[:, 2]

    # 分层计算全部目标的投影，避免为全尺寸头部额外常驻数百万行坐标。
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    sum_points = np.zeros(3, dtype=float)
    count = 0
    for z_index, section in enumerate(mask):
        yx = np.argwhere(section)
        if not len(yx):
            continue
        xyz = np.column_stack((yx[:, 1], yx[:, 0], np.full(len(yx), z_index)))
        points = origin + (xyz * spacing) @ image_direction.T
        projection = points @ final_axes
        lower = np.minimum(lower, np.min(projection, axis=0))
        upper = np.maximum(upper, np.max(projection, axis=0))
        sum_points += np.sum(points, axis=0)
        count += len(points)
    frame = AnatomicalFrame(
        origin=_point(sum_points / count),
        left_right_axis=_point(lr),
        anterior_posterior_axis=_point(ap),
        superior_inferior_axis=_point(si),
        bounds=AnatomicalBounds(*(AxisBounds(float(lower[i]), float(upper[i])) for i in range(3))),
        method="skull_reflection_symmetry",
        confidence=float(symmetry["confidence"]),
        applied=bool(symmetry["accepted"]),
        warnings=list(symmetry["warnings"]),
    )
    diagnostics: dict[str, Any] = {
        "symmetry": symmetry,
        "target_point_count": int(count),
        "boundary_point_count": int(len(boundary_points)),
        "plane_center_lps": plane_center.tolist(),
        "pitch_policy": "project_input_ap_onto_symmetry_plane",
    }
    return AnatomicalGeometryResult(global_frame=frame), diagnostics
