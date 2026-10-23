"""统一切割部位配置的加载、校验和只读查询。"""

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from src.runtime import config_path


_PLANES = {"横状位", "冠状位", "矢状位"}
_SLICE_TYPES = {"MPR", "CPR", "椎间隙"}
_REGION_TYPES = {"whole_range", "interspace"}


def default_slice_part_config_path() -> Path:
    """返回仓库内统一切割部位配置的标准路径。"""

    return config_path("slice_part_config.json")


def load_slice_part_config(path: Path | None = None) -> Dict[str, Any]:
    """读取并校验统一切割部位配置。"""

    config_path = path or default_slice_part_config_path()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    validate_slice_part_config(config)
    return config


def validate_slice_part_config(config: Mapping[str, Any]) -> None:
    """在任务启动前检查模型、定位区域和切法规则的一致性。"""

    if not isinstance(config, Mapping):
        raise ValueError("slice_part_config 必须是对象")
    _reject_unknown_fields(
        config,
        {"schema_version", "label_aliases", "slice_parts"},
        "slice_part_config",
    )
    if config.get("schema_version") != 1:
        raise ValueError("slice_part_config.schema_version 当前只支持 1")

    aliases = config.get("label_aliases")
    if not isinstance(aliases, Mapping):
        raise ValueError("slice_part_config.label_aliases 必须是对象")
    alias_owners: Dict[str, str] = {}
    for name, value in aliases.items():
        if not isinstance(name, str) or not name:
            raise ValueError("label_aliases 的名称必须是非空字符串")
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"label_aliases.{name} 必须是非空字符串或字符串数组")
        if len(values) != len(set(values)):
            raise ValueError(f"label_aliases.{name} 不能包含重复标签")
        for label_name in values:
            owner = alias_owners.get(label_name)
            if owner is not None:
                raise ValueError(
                    f"label_aliases.{name} 与 {owner} 重复引用标签: "
                    f"{label_name}"
                )
            alias_owners[label_name] = name

    parts = config.get("slice_parts")
    if not isinstance(parts, Mapping) or not parts:
        raise ValueError("slice_part_config.slice_parts 必须是非空对象")
    for slice_part, part_config in parts.items():
        _validate_part_config(slice_part, part_config, aliases)


def _validate_part_config(
    slice_part: str,
    part_config: Any,
    aliases: Mapping[str, Any],
) -> None:
    """校验单个切割部位下的全部配置。"""

    if not isinstance(slice_part, str) or not slice_part:
        raise ValueError("切割部位名称必须是非空字符串")
    if not isinstance(part_config, Mapping):
        raise ValueError(f"切割部位配置必须是对象: {slice_part}")
    _reject_unknown_fields(
        part_config,
        {"enabled", "model_rules", "localization", "slice_rules"},
        slice_part,
    )
    if not isinstance(part_config.get("enabled"), bool):
        raise ValueError(f"{slice_part}.enabled 必须是布尔值")

    model_rules = part_config.get("model_rules")
    if not isinstance(model_rules, Mapping):
        raise ValueError(f"{slice_part}.model_rules 必须是对象")
    for study_type, mappings in model_rules.items():
        _validate_model_mappings(slice_part, study_type, mappings)

    localization = part_config.get("localization")
    if not isinstance(localization, Mapping):
        raise ValueError(f"{slice_part}.localization 必须是对象")
    _reject_unknown_fields(
        localization,
        {"regions", "spine_cpr"},
        f"{slice_part}.localization",
    )
    regions = localization.get("regions")
    if not isinstance(regions, Mapping):
        raise ValueError(f"{slice_part}.localization.regions 必须是对象")
    for region_name, region in regions.items():
        _validate_region(slice_part, region_name, region, aliases)
    _validate_spine_cpr_config(
        slice_part,
        localization.get("spine_cpr"),
        aliases,
    )

    slice_rules = part_config.get("slice_rules")
    if not isinstance(slice_rules, Mapping):
        raise ValueError(f"{slice_part}.slice_rules 必须是对象")
    for plane, type_rules in slice_rules.items():
        if plane not in _PLANES:
            raise ValueError(f"{slice_part}.slice_rules 包含未知状位: {plane}")
        if not isinstance(type_rules, Mapping) or not type_rules:
            raise ValueError(f"{slice_part}.{plane} 的切法配置必须是非空对象")
        for slice_type, rule in type_rules.items():
            _validate_slice_rule(
                slice_part,
                plane,
                slice_type,
                rule,
                regions,
            )

    referenced_regions = {
        region_name
        for type_rules in slice_rules.values()
        for rule in type_rules.values()
        for region_name in rule["regions"]
    }
    unused_regions = set(regions) - referenced_regions
    if unused_regions:
        raise ValueError(
            f"{slice_part}.localization.regions 包含未使用区域: "
            f"{sorted(unused_regions)}"
        )
    supports_cpr = any("CPR" in type_rules for type_rules in slice_rules.values())
    has_cpr_config = "spine_cpr" in localization
    if supports_cpr and not has_cpr_config:
        raise ValueError(f"{slice_part} 支持 CPR，但缺少 localization.spine_cpr")
    if has_cpr_config and not supports_cpr:
        raise ValueError(f"{slice_part} 配置了 spine_cpr，但未配置 CPR 切法")

    if part_config["enabled"]:
        if not model_rules:
            raise ValueError(f"已启用切割部位缺少模型规则: {slice_part}")
        if not regions or not slice_rules:
            raise ValueError(f"已启用切割部位缺少定位区域或切法规则: {slice_part}")


def _validate_model_mappings(
    slice_part: str,
    study_type: Any,
    mappings: Any,
) -> None:
    """校验一种检查类型下的扫描厚度模型表。"""

    if not isinstance(study_type, str) or not study_type:
        raise ValueError(f"{slice_part}.model_rules 的检查类型必须是非空字符串")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError(f"{slice_part}.{study_type} 的模型规则必须是非空数组")

    parsed_ranges = []
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{slice_part}.{study_type} 的模型规则项必须是对象")
        _reject_unknown_fields(
            mapping,
            {"scan_thickness_range_mm", "model_id", "configuration", "fold"},
            f"{slice_part}.{study_type}.model_rules",
        )
        thickness_range = mapping.get("scan_thickness_range_mm")
        if not isinstance(thickness_range, Mapping):
            raise ValueError(
                f"{slice_part}.{study_type} 模型规则缺少 scan_thickness_range_mm"
            )
        _reject_unknown_fields(
            thickness_range,
            {"min", "max"},
            f"{slice_part}.{study_type}.scan_thickness_range_mm",
        )
        minimum = _non_negative_finite_number(
            thickness_range.get("min"),
            f"{slice_part}.{study_type}.scan_thickness_range_mm.min",
        )
        maximum = _non_negative_finite_number(
            thickness_range.get("max"),
            f"{slice_part}.{study_type}.scan_thickness_range_mm.max",
        )
        if minimum > maximum:
            raise ValueError(f"{slice_part}.{study_type} 扫描厚度下限不能大于上限")
        model_id = mapping.get("model_id")
        if isinstance(model_id, bool) or not isinstance(model_id, int) or model_id <= 0:
            raise ValueError(f"{slice_part}.{study_type}.model_id 必须是正整数")
        parsed_ranges.append((minimum, maximum))

    for index, current in enumerate(parsed_ranges):
        for previous in parsed_ranges[:index]:
            if max(current[0], previous[0]) <= min(current[1], previous[1]):
                raise ValueError(
                    f"{slice_part}.{study_type} 扫描厚度区间重叠: "
                    f"{previous} 与 {current}"
                )


def _validate_region(
    slice_part: str,
    region_name: Any,
    region: Any,
    aliases: Mapping[str, Any],
) -> None:
    """校验一个可输出区域的定位方式和端点标签。"""

    if not isinstance(region_name, str) or not region_name:
        raise ValueError(f"{slice_part} 的区域名称必须是非空字符串")
    if not isinstance(region, Mapping):
        raise ValueError(f"{slice_part}.{region_name} 区域必须是对象")
    _reject_unknown_fields(
        region,
        {"type", "start_label", "end_label"},
        f"{slice_part}.{region_name}",
    )
    if region.get("type") not in _REGION_TYPES:
        raise ValueError(f"{slice_part}.{region_name} 包含未知区域类型")
    for field_name in ("start_label", "end_label"):
        if not isinstance(region.get(field_name), str) or not region[field_name]:
            raise ValueError(f"{slice_part}.{region_name}.{field_name} 必须是非空字符串")
        if region[field_name] not in aliases:
            raise ValueError(
                f"{slice_part}.{region_name}.{field_name} 未在 label_aliases 中定义: "
                f"{region[field_name]}"
            )


def _validate_spine_cpr_config(
    slice_part: str,
    config: Any,
    aliases: Mapping[str, Any],
) -> None:
    """校验脊柱 CPR 范围模式和几何排除标签。"""

    if config is None:
        return
    if not isinstance(config, Mapping):
        raise ValueError(f"{slice_part}.localization.spine_cpr 必须是对象")
    unknown_fields = set(config) - {"full_label", "excluded_labels"}
    if unknown_fields:
        raise ValueError(
            f"{slice_part}.localization.spine_cpr 包含未知字段: "
            f"{sorted(unknown_fields)}"
        )
    if "full_label" not in config:
        raise ValueError(f"{slice_part}.localization.spine_cpr 缺少 full_label")
    if not isinstance(config["full_label"], bool):
        raise ValueError(f"{slice_part}.localization.spine_cpr.full_label 必须是布尔值")
    excluded_labels = config.get("excluded_labels", [])
    if not isinstance(excluded_labels, list) or any(
        not isinstance(label, str) or not label
        for label in excluded_labels
    ):
        raise ValueError(
            f"{slice_part}.localization.spine_cpr.excluded_labels 必须是字符串数组"
        )
    if len(excluded_labels) != len(set(excluded_labels)):
        raise ValueError(
            f"{slice_part}.localization.spine_cpr.excluded_labels 不能重复"
        )
    unknown_labels = set(excluded_labels) - set(aliases)
    if unknown_labels:
        raise ValueError(
            f"{slice_part}.localization.spine_cpr.excluded_labels 包含未知标签: "
            f"{sorted(unknown_labels)}"
        )


def _validate_slice_rule(
    slice_part: str,
    plane: str,
    slice_type: Any,
    rule: Any,
    regions: Mapping[str, Any],
) -> None:
    """校验一个状位与切法组合引用的区域和扩展距离。"""

    prefix = f"{slice_part}.{plane}.{slice_type}"
    if slice_type not in _SLICE_TYPES:
        raise ValueError(f"{prefix} 包含未知切法")
    if not isinstance(rule, Mapping):
        raise ValueError(f"{prefix} 切法规则必须是对象")
    _reject_unknown_fields(
        rule,
        {"regions", "range_extension", "localizer_plane", "framing_policy"},
        prefix,
    )
    if "framing_policy" in rule:
        if slice_type != "MPR":
            raise ValueError(f"{prefix}.framing_policy 仅支持 MPR")
        framing = rule["framing_policy"]
        framing_prefix = f"{prefix}.framing_policy"
        if not isinstance(framing, Mapping):
            raise ValueError(f"{framing_prefix} 必须是对象")
        _reject_unknown_fields(framing, {"basis", "margin_mm"}, framing_prefix)
        if framing.get("basis") != "selected_region":
            raise ValueError(f"{framing_prefix}.basis 仅支持 selected_region")
        _non_negative_finite_number(
            framing.get("margin_mm"), f"{framing_prefix}.margin_mm",
        )
    localizer_plane = rule.get("localizer_plane")
    if not isinstance(localizer_plane, str) or localizer_plane not in _PLANES:
        raise ValueError(
            f"{prefix}.localizer_plane 必须是横状位、冠状位或矢状位"
        )
    if localizer_plane == plane:
        raise ValueError(f"{prefix}.localizer_plane 不能与重建状位相同")
    region_names = rule.get("regions")
    if not isinstance(region_names, list) or not region_names:
        raise ValueError(f"{prefix}.regions 必须是非空数组")
    if any(not isinstance(region_name, str) or not region_name for region_name in region_names):
        raise ValueError(f"{prefix}.regions 必须只包含非空字符串")
    if len(region_names) != len(set(region_names)):
        raise ValueError(f"{prefix}.regions 不能包含重复区域")

    expected_region_type = "interspace" if slice_type == "椎间隙" else "whole_range"
    for region_name in region_names:
        if region_name not in regions:
            raise ValueError(f"{prefix} 引用了不存在的区域: {region_name}")
        if regions[region_name]["type"] != expected_region_type:
            raise ValueError(
                f"{prefix} 的区域 {region_name} 必须是 {expected_region_type}"
            )
    if expected_region_type == "whole_range" and len(region_names) != 1:
        raise ValueError(f"{prefix} 当前只支持一个整体区域")

    extension = rule.get("range_extension")
    if not isinstance(extension, Mapping):
        raise ValueError(f"{prefix}.range_extension 必须是对象")
    unknown_fields = set(extension) - {"start_mm", "end_mm"}
    if unknown_fields:
        raise ValueError(f"{prefix}.range_extension 仅支持 start_mm/end_mm")
    for endpoint in ("start_mm", "end_mm"):
        _non_negative_finite_number(extension.get(endpoint), f"{prefix}.{endpoint}")


def _non_negative_finite_number(value: Any, field_name: str) -> float:
    """解析配置中的非负有限数值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是非负有限数字")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} 必须是非负有限数字") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field_name} 必须是非负有限数字")
    return parsed


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed_fields: set[str],
    field_name: str,
) -> None:
    """拒绝配置拼写错误和已废弃字段，避免静默忽略。"""

    unknown_fields = set(value) - allowed_fields
    if unknown_fields:
        raise ValueError(
            f"{field_name} 包含未知字段: "
            f"{sorted(str(field) for field in unknown_fields)}"
        )


def get_slice_part(
    config: Mapping[str, Any],
    slice_part: str,
    *,
    require_enabled: bool = True,
) -> Mapping[str, Any]:
    """读取单个切割部位配置，并按需拒绝未启用项。"""

    parts = config.get("slice_parts", {})
    part_config = parts.get(slice_part) if isinstance(parts, Mapping) else None
    if not isinstance(part_config, Mapping):
        raise ValueError(f"未找到切割部位配置: {slice_part}")
    if require_enabled and part_config.get("enabled") is not True:
        raise ValueError(f"切割部位尚未启用: {slice_part}")
    return part_config


def model_mappings_for(
    config: Mapping[str, Any],
    slice_part: str,
    study_type: str,
) -> Sequence[Mapping[str, Any]]:
    """返回切割部位和检查类型对应的扫描厚度模型表。"""

    part_config = get_slice_part(config, slice_part)
    model_rules = part_config.get("model_rules", {})
    mappings = model_rules.get(study_type) if isinstance(model_rules, Mapping) else None
    if not isinstance(mappings, list) or not mappings:
        raise ValueError(
            f"未找到模型规则: slice_part={slice_part}, study_type={study_type}"
        )
    return mappings


def slice_rules_for(
    config: Mapping[str, Any],
    slice_part: str,
) -> Dict[str, Any]:
    """返回当前切割部位的切法规则副本，供阶段结果安全传递。"""

    part_config = get_slice_part(config, slice_part)
    return deepcopy(dict(part_config["slice_rules"]))


def localization_for(
    config: Mapping[str, Any],
    slice_part: str,
) -> Mapping[str, Any]:
    """返回当前切割部位的定位配置。"""

    part_config = get_slice_part(config, slice_part)
    return part_config["localization"]


def localization_profiles_for(
    config: Mapping[str, Any],
    slice_part: str,
) -> Dict[str, Any]:
    """把区域配置整理为现有定位阶段消费的整体范围和椎间范围。"""

    part_config = get_slice_part(config, slice_part)
    localization = part_config["localization"]
    regions = localization["regions"]
    referenced_names = []
    for type_rules in part_config["slice_rules"].values():
        for rule in type_rules.values():
            for region_name in rule["regions"]:
                if region_name not in referenced_names:
                    referenced_names.append(region_name)

    whole_ranges = []
    interspaces = []
    for region_name in referenced_names:
        region = regions[region_name]
        item = {
            "region": region_name,
            "start": region["start_label"],
            "end": region["end_label"],
        }
        if region["type"] == "whole_range":
            whole_ranges.append(item)
        else:
            interspaces.append(item)

    profiles: Dict[str, Any] = {}
    if whole_ranges:
        unique_whole_ranges = {
            (item["region"], item["start"], item["end"])
            for item in whole_ranges
        }
        if len(unique_whole_ranges) != 1:
            raise ValueError(f"{slice_part} 当前只支持一个整体定位区域")
        profiles["whole_range"] = whole_ranges[0]
    if interspaces:
        profiles["interspace"] = interspaces
    return profiles


def capability_entries(
    config: Mapping[str, Any],
) -> Iterable[Tuple[str, str, str, str]]:
    """按配置顺序返回检查类型、切割部位、切法和状位组合。"""

    for slice_part, part_config in config.get("slice_parts", {}).items():
        if part_config.get("enabled") is not True:
            continue
        for study_type in part_config.get("model_rules", {}):
            for plane, type_rules in part_config.get("slice_rules", {}).items():
                for slice_type in type_rules:
                    yield study_type, slice_part, slice_type, plane
