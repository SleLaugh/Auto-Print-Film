from copy import deepcopy

import pytest

from src.config import (
    load_slice_part_config,
    resolve_face_extension,
    slice_rules_for,
    validate_slice_part_config,
)


_FACE_FIELDS = (
    "top_mm",
    "bottom_mm",
    "front_mm",
    "back_mm",
    "left_mm",
    "right_mm",
)


def _config_with_rule(rule):
    config = load_slice_part_config()
    config = deepcopy(config)
    config["slice_parts"]["腰椎"]["slice_rules"]["横状位"]["MPR"] = rule
    return config


def _base_rule(config):
    rule = deepcopy(
        config["slice_parts"]["腰椎"]["slice_rules"]["横状位"]["MPR"]
    )
    rule.pop("face_extension_mm", None)
    return rule


def test_face_extension_is_optional_and_normalized_by_rule_parser():
    config = load_slice_part_config()
    rule = _base_rule(config)
    rule["face_extension_mm"] = {
        field: index + 1 for index, field in enumerate(_FACE_FIELDS)
    }
    config = _config_with_rule(rule)

    validate_slice_part_config(config)
    parsed = slice_rules_for(config, "腰椎")
    assert resolve_face_extension(parsed["横状位"]["MPR"]) == {
        field: float(index + 1) for index, field in enumerate(_FACE_FIELDS)
    }


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda extension: extension.pop("right_mm"), "缺少字段"),
        (lambda extension: extension.__setitem__("inside_mm", 1), "仅支持"),
        (lambda extension: extension.__setitem__("left_mm", -1), "非负有限数字"),
        (lambda extension: extension.__setitem__("front_mm", True), "非负有限数字"),
    ],
)
def test_face_extension_rejects_incomplete_or_invalid_values(mutate, expected):
    config = load_slice_part_config()
    rule = _base_rule(config)
    rule["face_extension_mm"] = {field: 0.0 for field in _FACE_FIELDS}
    mutate(rule["face_extension_mm"])

    with pytest.raises(ValueError, match=expected):
        validate_slice_part_config(_config_with_rule(rule))


def test_rule_requires_face_extension():
    config = load_slice_part_config()
    rule = _base_rule(config)
    with pytest.raises(ValueError, match="必须配置 face_extension_mm"):
        validate_slice_part_config(_config_with_rule(rule))
