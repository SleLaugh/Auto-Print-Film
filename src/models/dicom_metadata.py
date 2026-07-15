#!/src/models/dicom_metadata.py
# -*- coding: utf-8 -*-

"""
DICOM 元数据缓存实体

该实体对应预处理阶段输出的 metadata/dicom_metadata.json。JSON 本身
不支持注释，所以字段含义写在实体字段旁边；写出 JSON 时通过 to_dict。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DicomMetadata:
    """预处理阶段缓存的 DICOM/NIfTI 空间与切片映射信息。"""
    raw_nifti_name: str  # 生成的原始 NIfTI 文件名，nnUNet 单通道输入需以 _0000 结尾
    size_xyz: list[int]  # 图像体素尺寸，SimpleITK 顺序为 x/y/z
    spacing_xyz: list[float]  # 体素物理间距，单位通常为 mm，SimpleITK 顺序为 x/y/z
    origin_xyz: list[float]  # 图像原点物理坐标，SimpleITK 顺序为 x/y/z
    direction: list[float]  # 图像方向余弦矩阵，SimpleITK 以一维数组保存
    slice_count: int  # 原始 DICOM 序列切片数量
    slice_thickness: Optional[float]  # Z 方向层厚/层间距，当前取 spacing_xyz[2]
    dicom_files: list[str]  # SimpleITK 排序后的原始 DICOM 文件路径列表
    z_index_to_dicom_file: Dict[str, str]  # NIfTI 数组 Z 索引到原始 DICOM 文件的映射

    def to_dict(self) -> Dict[str, Any]:
        """序列化为预处理阶段写入 dicom_metadata.json 的字典。"""

        return {
            'raw_nifti_name': self.raw_nifti_name,
            'size_xyz': self.size_xyz,
            'spacing_xyz': self.spacing_xyz,
            'origin_xyz': self.origin_xyz,
            'direction': self.direction,
            'slice_count': self.slice_count,
            'slice_thickness': self.slice_thickness,
            'dicom_files': self.dicom_files,
            'z_index_to_dicom_file': self.z_index_to_dicom_file,
        }
