#!/src/models/localization.py
# -*- coding: utf-8 -*-

"""
定位结果实体

这些实体用于描述 localization_result.json 的结构。JSON 本身不支持注释，
所以字段说明写在实体和字段旁边；生成 JSON 时统一通过 to_dict 输出。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VoxelPoint:
    """体素索引坐标点，坐标顺序按医学数组习惯记录为 z/y/x。"""
    z: float  # Z 轴索引，SimpleITK 转 numpy 后数组的第 0 维
    y: float  # Y 轴索引，SimpleITK 转 numpy 后数组的第 1 维
    x: float  # X 轴索引，SimpleITK 转 numpy 后数组的第 2 维

    def to_dict(self) -> Dict[str, float]:
        return {
            'z': self.z,
            'y': self.y,
            'x': self.x,
        }


@dataclass
class VoxelBBox:
    """目标 label 在 3D 体素数组中的最小外接包围盒。"""
    z_min: int  # 目标体素出现的最小 Z 轴索引
    z_max: int  # 目标体素出现的最大 Z 轴索引
    y_min: int  # 目标体素出现的最小 Y 轴索引
    y_max: int  # 目标体素出现的最大 Y 轴索引
    x_min: int  # 目标体素出现的最小 X 轴索引
    x_max: int  # 目标体素出现的最大 X 轴索引

    def to_dict(self) -> Dict[str, int]:
        return {
            'z_min': self.z_min,
            'z_max': self.z_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'x_min': self.x_min,
            'x_max': self.x_max,
        }


@dataclass
class LabelTarget:
    """需要定位的模型 label 定义。"""
    label_name: str  # dataset.json 中的 label 名称，如 skull、vertebrae_L1
    label_ids: list[int]  # dataset.json 中对应的 label id，普通 label 通常只有一个

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label_name': self.label_name,
            'label_ids': self.label_ids,
        }


@dataclass
class LabelCoordinate:
    """单个目标 label 的定位结果。"""
    label_name: str  # dataset.json 中的 label 名称
    label_ids: list[int]  # 该目标对应的 label id 列表
    voxel_count: int  # seg.nii.gz 中属于该目标的体素数量
    found: bool  # 是否在 seg.nii.gz 中找到该目标体素
    centroid: Optional[VoxelPoint] = None  # 目标体素坐标均值，未找到时为 None
    bbox: Optional[VoxelBBox] = None  # 目标体素包围盒，未找到时为 None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label_name': self.label_name,
            'label_ids': self.label_ids,
            'voxel_count': self.voxel_count,
            'found': self.found,
            'centroid': self.centroid.to_dict() if self.centroid else None,
            'bbox': self.bbox.to_dict() if self.bbox else None,
        }


@dataclass
class NiftiImageInfo:
    """NIfTI 图像空间信息，用于解释体素坐标和后续转物理坐标。"""
    size_xyz: list[int]  # 图像体素尺寸，SimpleITK 顺序为 x/y/z
    spacing_xyz: list[float]  # 每个体素的物理间距，单位通常为 mm，顺序为 x/y/z
    origin_xyz: list[float]  # 图像原点物理坐标，顺序为 x/y/z
    direction: list[float]  # SimpleITK 方向余弦矩阵，按一维数组保存
    array_order: str = "zyx"  # numpy 数组坐标顺序，SimpleITK GetArrayFromImage 固定为 z/y/x

    def to_dict(self) -> Dict[str, Any]:
        return {
            'size_xyz': self.size_xyz,
            'spacing_xyz': self.spacing_xyz,
            'origin_xyz': self.origin_xyz,
            'direction': self.direction,
            'array_order': self.array_order,
        }


@dataclass
class LocalizationResult:
    """定位阶段最终输出实体，对应 metadata/localization_result.json。"""
    body_part: str  # 请求处理的部位，如 头骨、腰椎、肺部
    cut_mode: str  # 切割方式，如 whole_range 或 interspace
    seg_path: str  # 当前用于定位的分割文件路径
    model: Dict[str, Any]  # 分割阶段匹配到的模型信息
    dataset_json_path: str  # 当前模型对应的 dataset.json 路径
    image: NiftiImageInfo  # 分割图像空间信息
    cut_config: Dict[str, Any]  # 原始切割配置，来自 localization_config.json
    resolved_cut_config: Dict[str, Any]  # 已解析 alias 后的切割配置
    label_targets: Dict[str, LabelTarget]  # 本次需要定位的 label 目标
    coords: Dict[str, LabelCoordinate]  # 每个目标 label 的 bbox/centroid 定位结果
    is_skewed: bool = False  # 是否检测到歪斜；当前阶段先保留字段
    skew_params: Dict[str, Any] = field(default_factory=dict)  # 歪斜检测/回正参数

    def to_dict(self) -> Dict[str, Any]:
        return {
            'body_part': self.body_part,
            'cut_mode': self.cut_mode,
            'seg_path': self.seg_path,
            'model': self.model,
            'dataset_json_path': self.dataset_json_path,
            'image': self.image.to_dict(),
            'cut_config': self.cut_config,
            'resolved_cut_config': self.resolved_cut_config,
            'label_targets': {
                name: target.to_dict()
                for name, target in self.label_targets.items()
            },
            'coords': {
                name: coord.to_dict()
                for name, coord in self.coords.items()
            },
            'is_skewed': self.is_skewed,
            'skew_params': self.skew_params,
        }
