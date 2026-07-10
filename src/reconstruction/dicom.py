"""使用 SimpleITK/GDCM 写出重建后的 CT DICOM。"""

from pathlib import Path
import uuid
from typing import Any, Dict

import SimpleITK as sitk

from src.models import WindowPreset


_UID_ROOT = "1.2.826.0.1.3680043.2.1125"

# 可从原始序列继承的患者/检查级标签。序列 UID、实例 UID、实例编号以及
# 像素/几何标签不在这里，它们必须由派生序列重新生成或按新切片写入。
_INHERITED_TAGS = (
    "0008|0005",  # Specific Character Set
    "0008|0020",  # Study Date
    "0008|0022",  # Acquisition Date
    "0008|0030",  # Study Time
    "0008|0032",  # Acquisition Time
    "0008|0050",  # Accession Number
    "0008|0080",  # Institution Name
    "0008|0081",  # Institution Address
    "0008|0090",  # Referring Physician Name
    "0008|1030",  # Study Description
    "0008|1070",  # Operators' Name
    "0010|0010",  # Patient Name
    "0010|0020",  # Patient ID
    "0010|0030",  # Patient Birth Date
    "0010|0040",  # Patient Sex
    "0010|1000",  # Other Patient IDs
    "0010|1010",  # Patient Age
    "0018|0015",  # Body Part Examined
    "0018|1030",  # Protocol Name
    "0020|0010",  # Study ID
    "0020|0052",  # Frame of Reference UID
)

_DICOM_CHARSET_CODECS = {
    "GB18030": "gb18030",
    "GBK": "gbk",
    "ISO 2022 IR 58": "gb2312",
    "ISO_IR 192": "utf-8",
    "ISO_IR 100": "latin-1",
    "ISO_IR 6": "ascii",
}


def read_source_dicom_metadata(source: str | Path) -> Dict[str, str]:
    """读取原始序列的患者/检查级标签，供派生 DICOM 继承。"""

    source_path = Path(source)
    if source_path.is_file():
        candidate = source_path
    elif source_path.is_dir():
        candidate = _first_dicom_file(source_path)
    else:
        return {}
    if candidate is None:
        return {}

    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(str(candidate))
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        return {}

    metadata: Dict[str, str] = {}
    for tag in _INHERITED_TAGS + ("0020|000d",):
        if reader.HasMetaDataKey(tag):
            value = reader.GetMetaData(tag).strip()
            if value:
                metadata[tag] = value
    return metadata


def _first_dicom_file(directory: Path) -> Path | None:
    """按 SimpleITK 的序列排序规则优先选择一个原始 DICOM 文件。"""

    try:
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory))
    except RuntimeError:
        names = ()
    if names:
        return Path(names[0])

    # 某些输入目录只有一个文件且 GDCM 未识别为序列，递归兜底查找。
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            return path
    return None


def new_uid() -> str:
    """生成长度合法的 DICOM UID。"""

    suffix = uuid.uuid4().int % (10 ** 30)
    return f"{_UID_ROOT}.{suffix}"


def write_ct_dicom(
    image: sitk.Image,
    output_path: Path,
    plan: Dict[str, Any],
    planned_slice: Dict[str, Any],
    window: WindowPreset,
    series_uid: str,
    study_uid: str,
    total_slices: int,
    source_metadata: Dict[str, str] | None = None,
) -> None:
    """把一张已经固定窗宽窗位的 2D/单层 3D 图像写成 CT DICOM。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_pixel = image.GetOrigin()
    image_2d = _extract_2d_image(image)
    output_spacing = plan["output_spacing_mm"]
    image_2d.SetSpacing((
        float(output_spacing["x"]),
        float(output_spacing["y"]),
    ))

    inherited_metadata = source_metadata or {}
    specific_character_set = str(inherited_metadata.get("0008|0005", ""))
    for tag, value in inherited_metadata.items():
        if tag in _INHERITED_TAGS:
            image_2d.SetMetaData(
                tag,
                _normalize_metadata_value(value, specific_character_set),
            )

    plan_id = str(plan.get("plan_id", "plan"))
    plane = str(plan.get("plane", ""))
    group_name = str(plan.get("group_name", ""))
    if group_name == "whole_range":
        group_name = "整切"
    elif group_name == "interspace":
        group_name = "区间切"
    index = int(planned_slice["index"])
    center = planned_slice["center"]
    normal = _normalize(planned_slice["normal"])
    row_dir = _normalize(planned_slice["row_dir"])
    col_dir = _normalize(planned_slice["col_dir"])

    image_2d.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
    image_2d.SetMetaData("0008|0018", new_uid())
    image_2d.SetMetaData("0008|0008", "DERIVED\\SECONDARY")
    image_2d.SetMetaData("0008|0060", "CT")
    image_2d.SetMetaData(
        "0008|103e",
        f"{plane}-{window.name}-W{window.window_width}-C{window.window_center}-"
        f"{group_name}_{index}_{total_slices}",
    )
    image_2d.SetMetaData("0020|000d", study_uid)
    image_2d.SetMetaData("0020|000e", series_uid)
    image_2d.SetMetaData("0020|0013", str(index))
    image_2d.SetMetaData("0020|0032", _format_point(first_pixel))
    image_2d.SetMetaData(
        "0020|0037",
        "\\".join(_format_number(value) for value in (*row_dir, *col_dir)),
    )
    image_2d.SetMetaData("0018|0050", _format_number(plan["slice_thickness"]))
    image_2d.SetMetaData("0018|0088", _format_number(plan["slice_spacing"]))
    image_2d.SetMetaData(
        "0028|0030",
        f"{_format_number(output_spacing['y'])}\\{_format_number(output_spacing['x'])}",
    )
    # 像素已经按所选预设烘焙成固定灰度，查看器不应再次套用原始 HU 窗口。
    image_2d.SetMetaData("0028|1050", "32768")
    image_2d.SetMetaData("0028|1051", "65535")
    image_2d.SetMetaData("0028|0002", "1")
    image_2d.SetMetaData("0028|0004", "MONOCHROME2")
    image_2d.SetMetaData("0028|0100", "16")
    image_2d.SetMetaData("0028|0101", "16")
    image_2d.SetMetaData("0028|0102", "15")
    image_2d.SetMetaData("0028|0103", "0")
    image_2d.SetMetaData("0020|1041", _format_number(_dot(center, normal)))
    image_2d.SetMetaData("0020|4000", plan_id)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(output_path))
    writer.KeepOriginalImageUIDOn()
    writer.Execute(image_2d)


def _extract_2d_image(image: sitk.Image) -> sitk.Image:
    if image.GetDimension() == 2:
        return sitk.Image(image)
    size = image.GetSize()
    if size[2] != 1:
        raise ValueError("plane_stack 输出图像必须只有一个 z 层")
    return sitk.Extract(image, [size[0], size[1], 0], [0, 0, 0])


def _normalize_metadata_value(value: Any, specific_character_set: str) -> str:
    """把含 surrogateescape 字节的 DICOM 文本恢复成合法 Unicode。"""

    text = str(value)
    if not any(0xD800 <= ord(character) <= 0xDFFF for character in text):
        return text

    # GDCM/SWIG 有时先按 UTF-8+surrogateescape 暴露 GB18030 原始字节。
    # 先无损恢复原始字节，再依据 (0008,0005) 指定的字符集重新解码。
    raw_value = text.encode("utf-8", errors="surrogateescape")
    charset = specific_character_set.strip().upper()
    codec = _DICOM_CHARSET_CODECS.get(charset, "utf-8")
    try:
        return raw_value.decode(codec)
    except (LookupError, UnicodeDecodeError):
        return raw_value.decode(codec, errors="replace")


def _normalize(value: Dict[str, float]) -> tuple[float, float, float]:
    vector = (float(value["x"]), float(value["y"]), float(value["z"]))
    length = sum(component * component for component in vector) ** 0.5
    if length <= 0:
        raise ValueError("DICOM 方向向量长度不能为 0")
    return tuple(component / length for component in vector)


def _dot(point: Dict[str, float], vector: tuple[float, float, float]) -> float:
    return sum(float(point[axis]) * vector[index] for index, axis in enumerate(("x", "y", "z")))


def _format_point(point: Any) -> str:
    return "\\".join(_format_number(value) for value in point)


def _format_number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.8f}".rstrip("0").rstrip(".")
