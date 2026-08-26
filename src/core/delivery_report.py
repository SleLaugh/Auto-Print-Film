"""稳定交付 ``report.json`` 的构造与原子写入。"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


REPORT_FILE_NAME = "report.json"


def build_status_report(
    *,
    success: bool,
    message: str,
    request: Any = None,
    slice_part: Any = None,
    study_type: Any = None,
    scan_thickness: Any = None,
    plane_list: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """构造成功和失败路径共用的固定顶层报告结构。"""

    if request is not None:
        request_data = request.to_dict()
        slice_part = request_data.get("SlicePart")
        study_type = request_data.get("StudyType")
        scan_thickness = request_data.get("ScanThickness")

    return {
        "Success": bool(success),
        "Message": str(message or ""),
        "SlicePart": _enum_value(slice_part),
        "StudyType": _enum_value(study_type),
        "ScanThickness": _finite_float_or_none(scan_thickness),
        "PlaneList": list(plane_list or []),
    }


def write_report(output_dir: str | Path, report_data: Dict[str, Any]) -> Path:
    """原子写入报告，避免外部轮询时读到不完整 JSON。"""

    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / REPORT_FILE_NAME
    temporary_path = root / f".{REPORT_FILE_NAME}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(report_data, file, ensure_ascii=False, indent=2)
            file.flush()
        temporary_path.replace(report_path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                # 清理临时文件失败不能覆盖真正的报告写入异常。
                pass
    return report_path


def write_failure_report(
    output_dir: str | Path,
    message: str,
    *,
    request: Any = None,
    slice_part: Any = None,
    study_type: Any = None,
    scan_thickness: Any = None,
) -> Path:
    """写入不包含未完成影像结果的失败报告。"""

    return write_report(
        output_dir,
        build_status_report(
            success=False,
            message=message,
            request=request,
            slice_part=slice_part,
            study_type=study_type,
            scan_thickness=scan_thickness,
        ),
    )


def _enum_value(value: Any) -> Any:
    """把枚举转换成外部协议值，同时保留普通字符串和空值。"""

    return getattr(value, "value", value)


def _finite_float_or_none(value: Any) -> Optional[float]:
    """失败发生在参数解析期时，用空值表示不可用的扫描厚度。"""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
