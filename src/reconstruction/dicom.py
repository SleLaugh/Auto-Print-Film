"""使用 SimpleITK/GDCM 写出重建后的 CT DICOM。"""

import threading
import uuid
from pathlib import Path
from typing import Any, Dict

import pydicom
import SimpleITK as sitk

from src.models import WindowPreset


_UID_ROOT = "1.2.826.0.1.3680043.2.1125"
_DERIVED_CHARACTER_SET = "ISO_IR 192"

# SimpleITK 的 DICOM 写出由 GDCM 原生库完成。不同窗位可以并行计算灰度，
# 但同一进程内并发创建和执行 GDCM 写入器可能触发原生内存重复释放，因此
# CT 成片与 Localizer 必须共用这一把进程内锁。
_GDCM_WRITE_LOCK = threading.Lock()

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


def _execute_gdcm_write(image: sitk.Image, output_path: Path) -> None:
    """串行执行一次 GDCM 写盘，并在释放锁前销毁原生写入器。"""

    with _GDCM_WRITE_LOCK:
        writer = sitk.ImageFileWriter()
        try:
            writer.SetFileName(str(output_path))
            writer.KeepOriginalImageUIDOn()
            writer.Execute(image)
        finally:
            # SWIG 包装对象的析构也属于 GDCM 生命周期；显式在锁内释放，避免
            # 下一个窗位开始写入时上一个原生写入器仍在销毁。
            del writer


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
        if tag in _INHERITED_TAGS and tag != "0008|0005":
            image_2d.SetMetaData(
                tag,
                _normalize_metadata_value(value, specific_character_set),
            )
    # GDCM 接收的是 Python Unicode 文本，派生文件统一声明 UTF-8，避免继承
    # GB18030 声明后实际仍写出 UTF-8 字节，导致标准 DICOM 读取器解码乱码。
    image_2d.SetMetaData("0008|0005", _DERIVED_CHARACTER_SET)

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

    gdcm_temporary_path = _temporary_dicom_path(output_path, "gdcm")
    try:
        # GDCM、pydicom 标签校正全部在同目录临时文件完成，只有所有步骤成功
        # 后才原子替换正式文件，异常时不会遗留层厚/层间距错误的半成品。
        _execute_gdcm_write(image_2d, gdcm_temporary_path)
        _restore_planned_slice_geometry_tags(
            gdcm_temporary_path,
            slice_thickness=plan["slice_thickness"],
            slice_spacing=plan["slice_spacing"],
        )
        gdcm_temporary_path.replace(output_path)
    finally:
        if gdcm_temporary_path.exists():
            gdcm_temporary_path.unlink()


def _restore_planned_slice_geometry_tags(
    output_path: Path,
    *,
    slice_thickness: Any,
    slice_spacing: Any,
) -> None:
    """
    在 GDCM 写出后恢复计划指定的层厚和层间距标签。

    SimpleITK/GDCM 写二维 DICOM 时可能根据二维图像默认几何覆盖已经设置的
    ``SliceThickness`` 和 ``SpacingBetweenSlices``。这里仅校正两个数值标签，
    再通过同目录临时文件原子替换待发布结果；患者、检查及 UTF-8 文本保持原样。
    """

    dataset = pydicom.dcmread(str(output_path))
    dataset.SliceThickness = _format_number(slice_thickness)
    dataset.SpacingBetweenSlices = _format_number(slice_spacing)
    temporary_path = _temporary_dicom_path(output_path, "pydicom")
    try:
        dataset.save_as(str(temporary_path), enforce_file_format=True)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _temporary_dicom_path(output_path: Path, stage: str) -> Path:
    """为一次 DICOM 写出步骤生成同目录、不可见且不重名的临时路径。"""

    return output_path.with_name(
        f".{output_path.stem}.{stage}.{uuid.uuid4().hex}.tmp.dcm"
    )


def write_localizer_dicom(
    image: sitk.Image,
    output_path: Path,
    window: WindowPreset,
    series_uid: str,
    study_uid: str,
    source_metadata: Dict[str, str] | None = None,
) -> None:
    """写出不参与正式切片编号的定位图 DICOM。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_metadata = {
        tag: image.GetMetaData(tag)
        for tag in ("0020|0032", "0020|0037")
        if image.HasMetaDataKey(tag)
    }
    image_2d = _extract_singleton_axis(image)
    inherited_metadata = source_metadata or {}
    specific_character_set = str(inherited_metadata.get("0008|0005", ""))
    for tag, value in inherited_metadata.items():
        if tag in _INHERITED_TAGS and tag != "0008|0005":
            image_2d.SetMetaData(
                tag,
                _normalize_metadata_value(value, specific_character_set),
            )
    image_2d.SetMetaData("0008|0005", _DERIVED_CHARACTER_SET)

    image_2d.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")
    image_2d.SetMetaData("0008|0018", new_uid())
    image_2d.SetMetaData("0008|0008", "DERIVED\\SECONDARY\\LOCALIZER")
    image_2d.SetMetaData("0008|0060", "CT")
    image_2d.SetMetaData("0008|103e", f"Localizer-{window.name}")
    image_2d.SetMetaData("0020|000d", study_uid)
    image_2d.SetMetaData("0020|000e", series_uid)
    image_2d.SetMetaData("0020|0013", "1")
    image_2d.SetMetaData("0028|1050", "32768")
    image_2d.SetMetaData("0028|1051", "65535")
    image_2d.SetMetaData("0028|0002", "1")
    image_2d.SetMetaData("0028|0004", "MONOCHROME2")
    image_2d.SetMetaData("0028|0100", "16")
    image_2d.SetMetaData("0028|0101", "16")
    image_2d.SetMetaData("0028|0102", "15")
    image_2d.SetMetaData("0028|0103", "0")
    for tag, value in geometry_metadata.items():
        image_2d.SetMetaData(tag, value)

    _execute_gdcm_write(image_2d, output_path)


def _extract_2d_image(image: sitk.Image) -> sitk.Image:
    """把平面重建得到的单层三维图像压缩为可写 DICOM 的二维图像。"""

    if image.GetDimension() == 2:
        return sitk.Image(image)
    size = image.GetSize()
    if size[2] != 1:
        raise ValueError("plane_stack 输出图像必须只有一个 z 层")
    return sitk.Extract(image, [size[0], size[1], 0], [0, 0, 0])


def _extract_singleton_axis(image: sitk.Image) -> sitk.Image:
    """从投影后的三维单层图像中移除唯一长度为 1 的轴。"""

    if image.GetDimension() == 2:
        return sitk.Image(image)
    size = list(image.GetSize())
    singleton_axes = [axis for axis, length in enumerate(size) if length == 1]
    if len(singleton_axes) != 1:
        raise ValueError("定位图投影必须且只能保留一个单体素轴")
    size[singleton_axes[0]] = 0
    return sitk.Extract(image, size, [0, 0, 0])


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
    """读取 x/y/z 方向字典并归一化为 DICOM 使用的单位向量。"""

    vector = (float(value["x"]), float(value["y"]), float(value["z"]))
    length = sum(component * component for component in vector) ** 0.5
    if length <= 0:
        raise ValueError("DICOM 方向向量长度不能为 0")
    return tuple(component / length for component in vector)


def _dot(point: Dict[str, float], vector: tuple[float, float, float]) -> float:
    """计算物理点在指定单位方向上的标量投影。"""

    return sum(float(point[axis]) * vector[index] for index, axis in enumerate(("x", "y", "z")))


def _format_point(point: Any) -> str:
    """把坐标序列格式化为 DICOM 多值字段使用的反斜杠分隔文本。"""

    return "\\".join(_format_number(value) for value in point)


def _format_number(value: Any) -> str:
    """以紧凑且稳定的形式格式化 DICOM 小数字符串值。"""

    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.8f}".rstrip("0").rstrip(".")
