"""按源 CT 的真实采样支持收紧上下画布，保留左右取景余量。"""

import numpy as np
import SimpleITK as sitk

from src.reconstruction.plane_stack import PlaneStackResampler


def _vector(value: dict) -> np.ndarray:
    return np.asarray([value[axis] for axis in "xyz"], dtype=float)


def _matrix(plan: dict) -> tuple[int, int]:
    return tuple(
        max(1, int(round(float(plan["extent_mm"][extent]) / float(plan["output_spacing_mm"][axis]))) + 1)
        for extent, axis in (("width", "x"), ("height", "y"))
    )


def _sampling_offsets(image: sitk.Image, plan: dict, center, normal) -> np.ndarray:
    """复用 HU 采样的厚层定义，包括轴对齐时的真实层重叠。"""

    thickness = float(plan.get("slice_thickness", 1.0))
    aligned = PlaneStackResampler._aligned_normal_axis(image, tuple(normal))
    if aligned is None:
        return np.asarray(PlaneStackResampler._slab_offsets(image, thickness))
    axis, sign = aligned
    spacing = float(image.GetSpacing()[axis])
    # 先筛选可能与厚层相交的源层，再以与 HU 采样相同的逐点运算复核。
    # 近轴对齐时 normal 仍可能含其他轴分量，不能只减单个连续索引；
    # 恰好接触厚层边界的层也须用同一浮点运算判断是否有正重叠。
    offset_start = float((np.asarray(image.GetOrigin()) - center) @ normal)
    offsets = offset_start + np.arange(image.GetSize()[axis]) * spacing * sign
    candidates = np.flatnonzero(np.abs(offsets) <= (spacing + thickness) / 2 + 1e-7)
    selected = []
    for index in candidates:
        source_index = [0, 0, 0]
        source_index[axis] = int(index)
        source_center = image.TransformIndexToPhysicalPoint(source_index)
        offset = sum((source_center[a] - center[a]) * normal[a] for a in range(3))
        overlap = min(offset + spacing / 2, thickness / 2) - max(
            offset - spacing / 2, -thickness / 2,
        )
        if overlap > 0:
            selected.append(offset)
    return np.asarray(selected)


def _verify_boundary_rows(image: sitk.Image, plan: dict, item: dict, indices) -> np.ndarray:
    """复用源 CT 与原二维网格，只复核不确定行的源采样支持。

    不创建整卷掩膜，也不重新计算 HU 厚层平均。保留完整二维 reference，
    防止缩成单行并移动原点后改变 ITK 扫描线的浮点运算次序。NaN/Inf 双
    越界哨兵可区分无数据与源图自身的 NaN、Inf 或真实空气。
    """

    sampler = PlaneStackResampler
    columns, rows = _matrix(plan)
    center = sampler._point(item["center"])
    row, col, normal = (
        sampler._normalize(sampler._point(item[key]))
        for key in ("row_dir", "col_dir", "normal")
    )
    spacing_x, spacing_y = (float(plan["output_spacing_mm"][axis]) for axis in "xy")
    first = sampler._sub(
        sampler._sub(center, sampler._scale(row, (columns - 1) * spacing_x / 2)),
        sampler._scale(col, (rows - 1) * spacing_y / 2),
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([columns, rows, 1])
    resampler.SetOutputSpacing((spacing_x, spacing_y, 1.0))
    resampler.SetOutputDirection(sampler._direction(row, col, normal))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    supported = np.zeros(len(indices), dtype=bool)
    for offset in _sampling_offsets(image, plan, center, normal):
        resampler.SetOutputOrigin(tuple(first[a] + normal[a] * offset for a in range(3)))
        resampler.SetDefaultPixelValue(float("nan"))
        sampled = resampler.Execute(image)
        present = np.any(~np.isnan(sitk.GetArrayViewFromImage(sampled)[0, indices]), axis=1)
        # NaN 可能来自源像素，也可能是越界哨兵；更换哨兵后只有真正越界
        # 才变为 +Inf。只对尚未确认的行复核，不能按 HU 数值裁掉真实空气。
        missing = ~(supported | present)
        if np.any(missing):
            resampler.SetDefaultPixelValue(float("inf"))
            sampled = resampler.Execute(image)
            present[missing] = np.any(
                ~np.isposinf(sitk.GetArrayViewFromImage(sampled)[0, indices[missing]]), axis=1,
            )
        supported |= present
        if np.all(supported):
            break
    return supported


def supported_rows(image: sitk.Image, plan: dict, item: dict) -> np.ndarray:
    """每行是否存在实际输出列和厚层采样点位于源体积内。

    沿输出列的源连续索引是一次函数，六个体积边界给出列坐标区间；
    该区间内存在整数列才算有效。常规路径只计算几何；半体素边界有
    舍入歧义时，在原二维网格上复核不确定行，不创建额外整卷掩膜。
    判定不按 HU 阈值，不会把真实空气、窗位黑色或斜切角部误裁。
    """

    columns, rows = _matrix(plan)
    center = _vector(item["center"])
    row, col, normal = (
        _vector(item[key]) / np.linalg.norm(_vector(item[key]))
        for key in ("row_dir", "col_dir", "normal")
    )
    offsets = _sampling_offsets(image, plan, center, normal)
    if not len(offsets):
        return np.zeros(rows, dtype=bool)
    spacing_x, spacing_y = (float(plan["output_spacing_mm"][axis]) for axis in "xy")
    physical_to_index = np.linalg.inv(
        np.asarray(image.GetDirection()).reshape(3, 3) * image.GetSpacing()
    )
    first = center - row * (columns - 1) * spacing_x / 2 - col * (rows - 1) * spacing_y / 2
    first_index = physical_to_index @ (first - image.GetOrigin())
    column_step = physical_to_index @ (row * spacing_x)
    row_step = physical_to_index @ (col * spacing_y)
    normal_step = physical_to_index @ normal
    starts = (
        first_index
        + np.arange(rows)[None, :, None] * row_step
        + offsets[:, None, None] * normal_step
    )

    def bounded_rows(margin):
        lower = np.zeros(starts.shape[:2])
        upper = np.full(starts.shape[:2], columns - 1.0)
        possible = np.ones(starts.shape[:2], dtype=bool)
        # ITK 连续索引支持 [-0.5, size-0.5)，下端包含、上端不包含。
        for axis, count in enumerate(image.GetSize()):
            minimum, maximum = -0.5 + margin, count - 0.5 - margin
            step = column_step[axis]
            if abs(step) < 1e-12:
                possible &= (starts[..., axis] >= minimum) & (starts[..., axis] < maximum)
                continue
            first_column = (minimum - starts[..., axis]) / step
            last_column = (maximum - starts[..., axis]) / step
            if step > 0:
                lower = np.maximum(lower, np.ceil(first_column))
                upper = np.minimum(upper, np.ceil(last_column) - 1)
            else:
                lower = np.maximum(lower, np.floor(last_column) + 1)
                upper = np.minimum(upper, np.floor(first_column))
        return np.any(possible & (lower <= upper), axis=0)

    strict, loose = bounded_rows(1e-7), bounded_rows(-1e-7)
    uncertain = np.flatnonzero(strict != loose)
    if len(uncertain):
        strict[uncertain] = _verify_boundary_rows(image, plan, item, uncertain)
    return strict


def crop_vertical_padding(
    source: sitk.Image, plan: dict, item: dict, sampled: sitk.Image,
) -> tuple[sitk.Image, dict]:
    """只裁首尾完全没有源采样支持的行，按实际内容重定画布中心。

    逐张取实际行数，不用全组最小高度裁掉其他层的有效内容，也不补框。
    ROI 保留原始 HU、像素间距、列数和网格相位；只改变上下范围及面内
    显示中心，原切面、法向位置、厚层定义和定位图均不受影响。
    """

    occupied = np.flatnonzero(supported_rows(source, plan, item))
    columns, rows, _ = sampled.GetSize()
    if not len(occupied):
        # 扩展范围内的整张空层没有可恢复内容，不能伪造一张有效图像。
        details = {"mode": "source_support_crop", "status": "no_source_support"}
        return sampled, {
            **item,
            "framing": {**item.get("framing", {}), "vertical_source_support": details},
        }
    first, last = int(occupied[0]), int(occupied[-1])
    bottom = rows - 1 - last
    if first == 0 and bottom == 0:
        return sampled, item
    height = last - first + 1
    result = sitk.RegionOfInterest(sampled, [columns, height, 1], [0, first, 0])
    center = dict(zip("xyz", map(float, result.TransformContinuousIndexToPhysicalPoint(
        ((columns - 1) / 2.0, (height - 1) / 2.0, 0.0),
    ))))
    spacing_x, spacing_y, _ = result.GetSpacing()
    previous = item.get("framing", {})
    framing = {
        **previous,
        "center": center,
        "extent_mm": {"width": (columns - 1) * spacing_x, "height": (height - 1) * spacing_y},
        "output_size_xy": [columns, height],
        "output_spacing_mm": {"x": spacing_x, "y": spacing_y, "z": 0.0},
        "vertical_source_support": {
            "mode": "source_support_crop",
            "crop_top_rows": first,
            "crop_bottom_rows": bottom,
            "original_rows": rows,
            "original_center": dict(item["center"]),
        },
    }
    if "center_source" in previous:
        framing["target_center_source"] = previous["center_source"]
        framing["center_source"] = "source_supported_image"
    return result, {**item, "center": center, "framing": framing}
