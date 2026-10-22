#!/src/stages/stage_preprocessing.py
# -*- coding: utf-8 -*-

"""
第一阶段：预处理
DICOM → NIfTI 转换（单序列处理）
"""

import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import SimpleITK as sitk

from src.core.base_stage import BaseStage
from src.core.cache_lock import cache_lock
from src.core.cache_workspace import cache_workspace, cleanup_cache_workspaces
from src.core.context import ProcessingContext
from src.models import DicomMetadata, PreprocessingResult
from src.reconstruction.dicom import read_source_dicom_metadata
from src.utils import DebugLog, ErrorLog, is_debug_file_enabled

# 设置警告过滤
# 忽略警告信息（DICOM读取时可能产生大量无关警告）
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


class PreprocessingStage(BaseStage):
    """
    预处理阶段 - DICOM转NIfTI（单序列处理）
    
    特点：
    1. 每次处理一个DICOM序列 - 专注单一任务，避免复杂性
    2. 支持高并发调用 - 各走各的路径，互不干扰
    3. 输出路径由调用方指定 - 灵活控制输出位置
    4. 自动处理3D格式 - 确保输出为标准的3D医学图像
    
    工作流程：
        输入DICOM目录 → 读取序列 → 转换为3D格式 → 输出NIfTI文件
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化预处理阶段
        
        Args:
            config: 配置字典，支持以下选项：
                - debug: bool，是否开启调试模式（默认False）
                        开启后会输出详细错误信息和堆栈跟踪
        """
        super().__init__(name="preprocessing", config=config)
        self.debug = config.get('debug', False) if config else False
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        执行DICOM到NIfTI的转换（主入口方法）
        
        流程说明：
        1. 从context中获取输入输出路径
        2. 创建输出目录
        3. 读取DICOM序列
        4. 转换为3D格式
        5. 保存为NIfTI文件
        6. 将结果写回context
        
        Args:
            context: 处理上下文，必须包含：
                - request.dicom_dir: DICOM序列所在目录
                - request.output_dir: 输出根目录
            
        Returns:
            ProcessingContext: 包含处理结果的上下文
                结果中会包含：
                - nifti_path: str，生成的NIfTI文件路径
                - error: str（可选），错误信息（如果失败）
        
        Example:
            >>> context.request.dicom_dir = '/data/patient1/CT'
            >>> context.request.output_dir = '/output'
            >>> result = stage.execute(context)
            >>> print(result.get_result('preprocessing')['nifti_path'])
            '/output/nifti/CT.nii.gz'读取dicom序列
        """
        
        request = context.request
        
        input_dir = Path(request.dicom_dir) # 原始路径
        
        output_root = Path(request.cache_dir or request.output_dir)
        output_file = output_root / 'nifti' / 'raw_0000.nii.gz' # nnUNet 输入必须以 _0000 结尾
        metadata_file = output_root / 'metadata' / 'dicom_metadata.json'

        DebugLog("=" * 60)
        DebugLog("【第一阶段】预处理 - DICOM转NIfTI")
        DebugLog(f"  输入DICOM: {input_dir}")
        DebugLog(f"  输出NIfTI: {output_file}")
        
        
        try:
            result = self._get_or_create_cache(
                input_dir, output_root, output_file, metadata_file
            )
            context.set_stage_result(result)
            DebugLog("  第一阶段完成")
            DebugLog("=" * 60)
        except Exception as e:
            ErrorLog("预处理失败", e)
            raise
        
        return context

    def _get_or_create_cache(
        self, input_dir: Path, output_root: Path, output_file: Path, metadata_file: Path
    ) -> PreprocessingResult:
        """缓存只读复用；缺失时跨进程单写，最后原子发布完整的 NIfTI。"""

        cleanup_cache_workspaces(output_root / ".work", "preprocessing-")
        if output_file.is_file():
            return self._reuse_cache(input_dir, output_file, metadata_file)

        with cache_lock(output_root / ".locks" / "preprocessing.lock"):
            # 其他进程可能已经在等待期间完成了转换。
            if output_file.is_file():
                return self._reuse_cache(input_dir, output_file, metadata_file)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            work_root = output_root / ".work"
            with cache_workspace(work_root, "preprocessing-") as work:
                return self._create_cache(
                    input_dir, output_file, metadata_file, work.path
                )

    @staticmethod
    def _reuse_cache(
        input_dir: Path, output_file: Path, metadata_file: Path
    ) -> PreprocessingResult:
        source_metadata = read_source_dicom_metadata(input_dir)
        metadata_path = str(metadata_file) if metadata_file.is_file() else ""
        DebugLog(f"  已找到现成NIfTI，跳过转换: {output_file}")
        return PreprocessingResult(
            nifti_path=str(output_file),
            metadata_path=metadata_path,
            source_dicom_metadata=source_metadata,
        )

    def _create_cache(
        self, input_dir: Path, output_file: Path, metadata_file: Path, work_dir: Path
    ) -> PreprocessingResult:
        """所有生成步骤成功后才使最终 NIfTI 对其他任务可见。"""

        read_result = self._read_dicom_series(str(input_dir)) # 读取dicom序列
        image = read_result['image'] if read_result else None
        if image is None:
            raise ValueError('读取DICOM失败：未找到有效的DICOM序列')

        image = self._ensure_3d(image) # 确保图像为3D格式
        temporary_nifti = work_dir / output_file.name
        sitk.WriteImage(image, str(temporary_nifti))
        source_metadata = read_source_dicom_metadata(
            read_result['series_file_names'][0]
        )
        metadata_path = ""
        if is_debug_file_enabled():
            metadata = self._build_metadata(image, read_result['series_file_names'])
            temporary_metadata = work_dir / metadata_file.name
            with temporary_metadata.open('w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            temporary_metadata.replace(metadata_file)
            metadata_path = str(metadata_file)

        # raw 沿用旧缓存兼容规则作为完成标记，在图像和元数据完成后发布。
        temporary_nifti.replace(output_file)
        return PreprocessingResult(
            nifti_path=str(output_file),
            metadata_path=metadata_path,
            source_dicom_metadata=source_metadata,
        )

    def _read_dicom_series(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        读取DICOM序列
        
        使用SimpleITK的ImageSeriesReader自动识别并读取DICOM序列。
        该方法能够处理：
        - 单个DICOM文件
        - 同一目录下的多个DICOM文件（组成一个序列）
        - 包含私有标签的DICOM文件
        
        Args:
            filepath: 包含DICOM文件的目录路径
            
        Returns:
            Optional[sitk.Image]: 
                - 成功：返回读取的3D/4D图像对象
                - 失败：返回None
            
        技术细节：
            1. ImageSeriesReader: SimpleITK的专用DICOM读取器
            2. MetaDataDictionaryArrayUpdateOn(): 保留元数据信息
            3. LoadPrivateTagsOn(): 读取私有标签（某些厂商特有信息）
            4. GetGDCMSeriesFileNames(): 自动识别同一序列的所有文件
        """
        
        try:
            reader = sitk.ImageSeriesReader() # 创建DICOM序列读取器 
            reader.MetaDataDictionaryArrayUpdateOn() # 启用元数据字典数组更新（保留每个切片的元数据）
            reader.LoadPrivateTagsOn() # 加载私有标签（某些DICOM包含厂商特有的私有信息）
            
            series_file_names = reader.GetGDCMSeriesFileNames(filepath) # 自动识别目录下的所有DICOM文件并排序
            if not series_file_names: # 没有找到DICOM文件
                return None 
            
            reader.SetFileNames(series_file_names) # 设置文件列表并读取
            image = reader.Execute()
            return {
                'image': image,
                'series_file_names': list(series_file_names)
            }
            
        except Exception as e:
            ErrorLog(f" ⚠️  读取DICOM序列一场: {str(e)}")
            return None

    def _build_metadata(self, image: sitk.Image, series_file_names: list[str]) -> DicomMetadata:
        """
        保存后续 DICOM 回写和 Z 轴映射需要的基础元数据。
        """
        spacing = image.GetSpacing()
        return DicomMetadata(
            raw_nifti_name='raw_0000.nii.gz',
            size_xyz=list(image.GetSize()),
            spacing_xyz=list(spacing),
            origin_xyz=list(image.GetOrigin()),
            direction=list(image.GetDirection()),
            slice_count=len(series_file_names),
            slice_thickness=spacing[2] if len(spacing) >= 3 else None,
            dicom_files=series_file_names,
            z_index_to_dicom_file={
                str(index): filename
                for index, filename in enumerate(series_file_names)
            }
        )
    
    def _ensure_3d(self, image: sitk.Image) -> sitk.Image:
        """
        确保图像是3D格式
        
        医学图像常见格式：
        - 3D: (深度, 高度, 宽度) - 标准的体素数据
        - 4D: (时间, 深度, 高度, 宽度) - 动态图像
        - 4D: (通道, 深度, 高度, 宽度) - 多通道图像
        - 2D: (高度, 宽度) - 单张切片
        
        该方法会将各种格式统一转换为3D：
        - 2D → 3D (添加深度维度，深度=1)
        - 4D → 3D (去除单维度或取第一个通道/时间点)
        
        Args:
            image: 输入的SimpleITK图像对象
            
        Returns:
            sitk.Image: 3D格式的SimpleITK图像对象
            
        处理策略：
            1. 如果是2D: 转换为3D（添加深度维度）
            2. 如果是4D: 尝试去除单维度，或取第一个通道
            3. 如果已经是3D: 直接返回
            
        Example:
            >>> # 4D图像 (时间, 深度, 高度, 宽度) = (5, 128, 128, 128)
            >>> image_3d = stage._ensure_3d(image_4d)
            >>> # 输出: (128, 128, 128) - 取第一个时间点
        """
                
        image_array = sitk.GetArrayFromImage(image) # 获取图像数组， SimpleITK的数组顺序为 (z, y, x) 即 (深度, 高度, 宽度)
        

        if len(image_array.shape) > 3: # 维度大于3（通常是4D）
            image_array = image_array.squeeze() # 去除所有大小为1的维度   例如：(1, 128, 128, 128) → (128, 128, 128)
            
            if len(image_array.shape) > 3: # 如果去除单维度后还是4D，取第一个通道/时间点  例如：(5, 128, 128, 128) → (128, 128, 128)
                image_array = image_array[..., 0] # Python的...表示所有维度，[..., 0]表示取最后一个维度的第一个元素   例如：(128, 128, 128, 5) → (128, 128, 128) 取第5维的第一个
            
            if len(image_array.shape) >= 2:
                width, height = image_array.shape[0], image_array.shape[1] # 重塑为3D格式 (深度, 高度, 宽度)
                image_array = np.reshape(image_array, (width, height, -1)) # -1表示自动计算该维度大小
            
            image = sitk.GetImageFromArray(image_array) # 转换回SimpleITK图像
        
        # 如果是2D，GetImageFromArray会自动添加深度维度
        # 所以不需要额外处理
        return image
    
