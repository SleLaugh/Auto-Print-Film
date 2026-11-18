"""按当前切片规则选择已求解区域，不再从回正标签猜测切片范围。"""

from typing import Any


def mpr_region_for_plane(localization_result: dict, plane: Any) -> tuple[str, dict | None]:
    """返回当前状位引用的区域；新契约缺少求解结果时明确报错。"""

    plane_name = str(getattr(plane, "value", plane))
    configured = localization_result.get("resolved_cut_configs", {}).get("whole_range", {})
    regions = localization_result.get("anatomical_regions")
    if regions is None:
        if configured.get("regions"):
            raise ValueError("普通 MPR 缺少第五阶段的 anatomical_regions")
        # 独立几何调用可以直接传入一个 AnatomicalFrame，不要求完整流水线。
        return "", None
    names = localization_result.get("slice_rules", {}).get(plane_name, {}).get("MPR", {}).get("regions")
    if not isinstance(names, list) or len(names) != 1:
        raise ValueError(f"{plane_name}.MPR 必须明确引用一个区域")
    name = names[0]
    if name not in regions:
        raise ValueError(f"{plane_name}.MPR 引用的区域尚未求解: {name}")
    return name, regions[name]
