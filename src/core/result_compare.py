"""交付结果报告的结构化对比工具。

对比时把报告中的绝对输出路径转换为相对路径，避免不同运行目录导致
无意义的差异；其余字段按 JSON 结构逐项比较。
"""

import json
from pathlib import Path
from typing import Any


REPORT_FILE_NAME = "report.json"


def load_delivery_report(result_dir: str | Path) -> dict[str, Any]:
    """读取结果目录中的交付报告。"""

    report_path = Path(result_dir).expanduser() / REPORT_FILE_NAME
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError(f"交付报告必须是 JSON 对象: {report_path}")
    return report


def normalize_delivery_report(
    report: dict[str, Any],
    result_dir: str | Path,
) -> dict[str, Any]:
    """复制并规范化交付报告中的输出路径。"""

    normalized = json.loads(json.dumps(report, ensure_ascii=False))
    root = Path(result_dir).expanduser().resolve()
    for plane in normalized.get("PlaneList", []):
        for window in plane.get("WindowPresetList", []):
            for region in window.get("RegionList", []):
                path_value = region.get("Path")
                if not path_value:
                    continue
                path = Path(path_value)
                if path.is_absolute():
                    try:
                        path_value = path.resolve().relative_to(root).as_posix()
                    except ValueError:
                        path_value = path.as_posix()
                region["Path"] = path_value
    return normalized


def compare_result_reports(
    left_dir: str | Path,
    right_dir: str | Path,
) -> list[dict[str, Any]]:
    """比较两个结果目录的交付报告，返回结构化差异列表。"""

    left = normalize_delivery_report(
        load_delivery_report(left_dir),
        left_dir,
    )
    right = normalize_delivery_report(
        load_delivery_report(right_dir),
        right_dir,
    )
    differences: list[dict[str, Any]] = []
    _collect_differences(left, right, "", differences)
    return differences


def _collect_differences(
    left: Any,
    right: Any,
    path: str,
    differences: list[dict[str, Any]],
) -> None:
    """递归收集报告差异。"""

    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}" if path else key
            if key not in left:
                differences.append({"path": child_path, "left": None, "right": right[key]})
            elif key not in right:
                differences.append({"path": child_path, "left": left[key], "right": None})
            else:
                _collect_differences(left[key], right[key], child_path, differences)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"path": path, "left": left, "right": right})
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _collect_differences(left_item, right_item, f"{path}[{index}]", differences)
        return
    if left != right:
        differences.append({"path": path, "left": left, "right": right})
