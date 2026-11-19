"""以用户提供的线上配置为独立基准，防止格式迁移丢失既有范围与毫米余量。

基准保留原始字节，不从当前运行配置反向生成。来源及核对记录见
docs/superpowers/archive/reviews/2026-11-19-线上配置迁移核对记录.md。
旧格式仅用于测试；发布包仍须配套使用新结构的运行配置。
"""

import json
from pathlib import Path

import pytest

from src.config import load_slice_part_config, slice_rules_for


_PRODUCTION = json.loads(
    (Path(__file__).parent / "fixtures" / "production_slice_part_config_20261110.json")
    .read_text(encoding="utf-8")
)
_SPINE_LABELS = [
    *(f"C{index}" for index in range(1, 8)),
    *(f"T{index}" for index in range(1, 13)),
    *(f"L{index}" for index in range(1, 6)),
    "S1",
]
_PRODUCTION_RULES = [
    (part_name, plane, slice_type)
    for part_name, part in _PRODUCTION["slice_parts"].items()
    for plane, rules in part["slice_rules"].items()
    for slice_type in rules
]


@pytest.fixture(scope="module")
def migrated_config():
    return load_slice_part_config()


def _legacy_target_labels(region):
    """旧首尾椎体区间包含中间全部椎体，双侧器官保留两端，同一器官去重。"""
    start, end = region["start_label"], region["end_label"]
    if start in _SPINE_LABELS and end in _SPINE_LABELS:
        return _SPINE_LABELS[_SPINE_LABELS.index(start):_SPINE_LABELS.index(end) + 1]
    return list(dict.fromkeys((start, end)))


def _assert_mpr_region_preserved(legacy, migrated):
    labels = _legacy_target_labels(legacy)
    assert migrated["type"] == "whole_range"
    assert migrated["target_labels"] == labels
    assert migrated.get("transverse_source", "labels") == "labels"
    assert migrated["bounds"] == {
        side: {"labels": labels, "edge": side, "reduce": reduction, "offset_mm": 0.0}
        for side, reduction in (("upper", "max"), ("lower", "min"))
    }


def test_production_aliases_are_preserved(migrated_config):
    for name, labels in _PRODUCTION["label_aliases"].items():
        assert migrated_config["label_aliases"][name] == labels


@pytest.mark.parametrize("part_name", _PRODUCTION["slice_parts"])
def test_production_parts_keep_models_pose_and_capabilities(migrated_config, part_name):
    legacy = _PRODUCTION["slice_parts"][part_name]
    migrated = migrated_config["slice_parts"][part_name]
    assert migrated["enabled"] == legacy["enabled"]
    assert migrated["model_rules"] == legacy["model_rules"]
    assert set(migrated["slice_rules"]) == set(legacy["slice_rules"])
    for plane in legacy["slice_rules"]:
        assert set(migrated["slice_rules"][plane]) == set(legacy["slice_rules"][plane])
    old_localization, new_localization = legacy["localization"], migrated["localization"]
    assert new_localization.get("spine_cpr") == old_localization.get("spine_cpr")
    if not legacy["enabled"]:
        assert migrated == legacy
        return

    default = old_localization["regions"]["default"]
    labels = _legacy_target_labels(default)
    pose = new_localization["pose"]
    if labels[0] in _SPINE_LABELS:
        assert {**pose, "fit_excluded_labels": pose.get("fit_excluded_labels", [])} == {
            "method": "spine",
            "labels": labels,
            "roles": {"start": default["start_label"], "end": default["end_label"]},
            "fit_excluded_labels": old_localization.get("spine_cpr", {}).get("excluded_labels", []),
        }
    elif labels == ["hip_left", "hip_right"]:
        assert pose == {"method": "hip", "roles": {"left": "hip_left", "right": "hip_right"}}
    else:
        assert pose == {"method": labels[0], "labels": labels}


@pytest.mark.parametrize("part_name,plane,slice_type", _PRODUCTION_RULES)
def test_production_rules_keep_margins_framing_and_region_semantics(
    migrated_config, part_name, plane, slice_type,
):
    legacy_part = _PRODUCTION["slice_parts"][part_name]
    migrated_part = migrated_config["slice_parts"][part_name]
    legacy_rule = legacy_part["slice_rules"][plane][slice_type]
    migrated_rule = slice_rules_for(migrated_config, part_name)[plane][slice_type]
    # CPR 区域引用因职责拆分可以改名；毫米值、定位图及取景策略必须原样保留。
    assert {key: value for key, value in migrated_rule.items() if key != "regions"} == {
        key: value for key, value in legacy_rule.items() if key != "regions"
    }
    assert len(migrated_rule["regions"]) == len(legacy_rule["regions"])
    for old_name, new_name in zip(legacy_rule["regions"], migrated_rule["regions"]):
        legacy_region = legacy_part["localization"]["regions"][old_name]
        migrated_region = migrated_part["localization"]["regions"][new_name]
        if slice_type == "MPR":
            _assert_mpr_region_preserved(legacy_region, migrated_region)
        elif slice_type == "CPR":
            assert migrated_region == {**legacy_region, "type": "spine_range"}
        else:
            assert new_name == old_name
            assert migrated_region == legacy_region
