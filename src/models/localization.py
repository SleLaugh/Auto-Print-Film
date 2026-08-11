#!/src/models/localization.py
# -*- coding: utf-8 -*-

"""
定位结果实体

这些实体用于描述 localization_result.json 的结构。JSON 本身不支持注释，
所以字段说明写在实体和字段旁边；生成 JSON 时统一通过 to_dict 输出。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .anatomy import AnatomicalGeometryResult


@dataclass
class VoxelPoint:
    """体素索引坐标点，坐标顺序按医学数组习惯记录为 z/y/x。"""
    z: float  # Z 轴索引，SimpleITK 转 numpy 后数组的第 0 维
    y: float  # Y 轴索引，SimpleITK 转 numpy 后数组的第 1 维
    x: float  # X 轴索引，SimpleITK 转 numpy 后数组的第 2 维

    def to_dict(self) -> Dict[str, float]:
        """按 z/y/x 数组索引顺序序列化体素坐标。"""

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
        """序列化目标在三维数组中的闭区间包围盒。"""

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
        """序列化目标标签名称及其对应的模型标签编号。"""

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
        """序列化单个标签的查找状态、质心和包围盒。"""

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
        """序列化解释体素坐标所需的 NIfTI 空间信息。"""

        return {
            'size_xyz': self.size_xyz,
            'spacing_xyz': self.spacing_xyz,
            'origin_xyz': self.origin_xyz,
            'direction': self.direction,
            'array_order': self.array_order,
        }


@dataclass
class LocalizationResult:
    """第三至第五阶段调试汇总实体，对应 metadata/localization_result.json。"""
    body_part: str  # 请求处理的部位，如 头骨、腰椎、肺部
    cut_mode: str  # 切割方式，如 whole_range 或 interspace
    seg_path: str  # 当前用于定位的分割文件路径
    model: Dict[str, Any]  # 分割阶段匹配到的模型信息
    dataset_json_path: str  # 当前模型对应的 dataset.json 路径
    image: NiftiImageInfo  # 分割图像空间信息
    cut_config: Dict[str, Any]  # 原始切割配置，来自 localization_config.json
    cut_configs: Dict[str, Dict[str, Any]]  # 当前部位按 whole_range/interspace 保存的全部原始配置
    resolved_cut_config: Dict[str, Any]  # 已解析 alias 后的切割配置
    label_targets: Dict[str, LabelTarget]  # 本次需要定位的 label 目标
    coords: Dict[str, LabelCoordinate]  # 每个目标 label 的 bbox/centroid 定位结果
    raw_for_reconstruction_path: str = ""  # 第三阶段选定的工作 raw；候选可用时指向回正 raw
    seg_for_reconstruction_path: str = ""  # 与 raw 同步的分割图；候选可用时指向回正 seg
    correction_applied: bool = False  # 是否已把回正候选切换为本轮定位几何和重建输入
    correction_angle_degrees: float = 0.0  # 横状位 XY 候选角；拒绝或禁用时为 0.0
    correction_scope_bbox: Optional[VoxelBBox] = None  # 本次定位目标的整体 bbox，用作候选回正中心参考
    correction_params: Dict[str, Any] = field(default_factory=dict)  # 回正候选状态、阈值、角度和拒绝原因等诊断参数
    corrected_raw_path: str = ""  # 兼容旧字段；未回正时等于 raw_for_reconstruction_path
    corrected_seg_path: str = ""  # 兼容旧字段；未回正时等于 seg_for_reconstruction_path
    candidate_raw_path: str = ""  # 横状位 XY 候选回正 raw；成功提升时也作为主流程 raw
    candidate_seg_path: str = ""  # 横状位 XY 候选回正 seg；成功提升时也作为主流程 seg
    debug_bbox_overlay_path: str = ""  # 研发期 ITK-SNAP bbox 边框 overlay；后续稳定后可删除
    debug_target_mask_overlay_path: str = ""  # 研发期 ITK-SNAP 真实目标 mask overlay；用于避免把外接盒投影视为定位偏移
    debug_label_mask_overlay_path: str = ""  # 研发期 ITK-SNAP 分 label mask overlay；保留各椎体原始 label id
    debug_vertebral_body_mask_overlay_path: str = ""  # 研发期椎体主体候选 mask overlay；用于校验是否能从完整椎体 label 中排除后方附件
    debug_vertebral_body_prism_mask_overlay_path: str = ""  # 研发期椎体四棱柱 mask overlay；用于核对规则化主体与 CPR 质心
    debug_xy_correction_crosshair_overlay_path: str = ""  # 横状位 XY 候选回正方向线 overlay
    is_skewed: bool = False  # 兼容旧字段；当前语义等同于 correction_applied，不再表示已可靠检测到歪斜
    skew_params: Dict[str, Any] = field(default_factory=dict)  # 兼容旧字段；语义等同于 correction_params
    anatomical_geometry: Optional[AnatomicalGeometryResult] = None  # 第五阶段统一解剖坐标；未适配部位为空
    spine_geometry: Optional[Dict[str, Any]] = None  # 仅脊柱生成；标注质心供 CPR，原始 coords/bbox 仍代表全量标注
    spine_cpr_config: Dict[str, Any] = field(default_factory=dict)  # 脊柱 CPR 业务配置，如 offset 范围来源

    def to_dict(self) -> Dict[str, Any]:
        """序列化第三至第五阶段的调试汇总结果。"""

        data = {
            'body_part': self.body_part,
            'cut_mode': self.cut_mode,
            'seg_path': self.seg_path,
            'model': self.model,
            'dataset_json_path': self.dataset_json_path,
            'image': self.image.to_dict(),
            'cut_config': self.cut_config,
            'cut_configs': self.cut_configs,
            'resolved_cut_config': self.resolved_cut_config,
            'label_targets': {
                name: target.to_dict()
                for name, target in self.label_targets.items()
            },
            'coords': {
                name: coord.to_dict()
                for name, coord in self.coords.items()
            },
            'raw_for_reconstruction_path': self.raw_for_reconstruction_path,
            'seg_for_reconstruction_path': self.seg_for_reconstruction_path,
            'correction_applied': self.correction_applied,
            'correction_angle_degrees': self.correction_angle_degrees,
            'correction_scope_bbox': (
                self.correction_scope_bbox.to_dict()
                if self.correction_scope_bbox else None
            ),
            'correction_params': self.correction_params,
            'corrected_raw_path': self.corrected_raw_path,
            'corrected_seg_path': self.corrected_seg_path,
            'candidate_raw_path': self.candidate_raw_path,
            'candidate_seg_path': self.candidate_seg_path,
            'debug_bbox_overlay_path': self.debug_bbox_overlay_path,
            'debug_target_mask_overlay_path': self.debug_target_mask_overlay_path,
            'debug_label_mask_overlay_path': self.debug_label_mask_overlay_path,
            'debug_vertebral_body_mask_overlay_path': self.debug_vertebral_body_mask_overlay_path,
            'debug_vertebral_body_prism_mask_overlay_path': self.debug_vertebral_body_prism_mask_overlay_path,
            'debug_xy_correction_crosshair_overlay_path': self.debug_xy_correction_crosshair_overlay_path,
            'is_skewed': self.is_skewed,
            'skew_params': self.skew_params,
        }
        if self.anatomical_geometry is not None:
            data['anatomical_geometry'] = self.anatomical_geometry.to_dict()
        if self.spine_geometry is not None:
            data['spine_geometry'] = self.spine_geometry
        if self.spine_cpr_config:
            data['spine_cpr_config'] = self.spine_cpr_config
        return data
