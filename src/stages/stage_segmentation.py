#!/src/stages/stage_segmentation.py
# -*- coding: utf-8 -*-

"""
第二阶段：推理分割
模型匹配与推理
"""

from pathlib import Path
from typing import Dict, Any
import json
import math
import subprocess
import shutil
import os

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.config import (
    load_slice_part_config,
    model_mappings_for,
    validate_slice_part_config,
)
from src.models import SegmentationResult
from src.utils import DebugLog, ErrorLog


class SegmentationStage(BaseStage):
    """推理分割阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化分割阶段并加载统一切割部位配置及 nnUNet 运行配置。"""

        super().__init__(name="segmentation", config=config)
        self._load_config()
    
    def _load_config(self):
        """
        加载配置文件
        优先加载外部JSON配置文件，如果不存在则使用默认配置
        """
        # 默认配置
        project_root = Path(__file__).resolve().parents[2]
        default_config = {
            "_configuration": "3d_fullres",  # 模型配置：使用哪种网络架构配置
            "_fold": "0",  # 训练折数（fold）：使用哪个折的模型权重
            "_continue": True,  # 继续模式：如果输出文件已存在则跳过，避免重复预测
            # nnUNet环境变量。为空时沿用系统环境变量；相对路径按项目根目录解析。
            "nnUNet_raw": "",
            "nnUNet_preprocessed": "",
            "nnUNet_results": ""
        }
        
        if self.config:
            # Pipeline 已通过统一 ConfigLoader 形成生效配置，不再重复读取 JSON。
            default_config.update(self.config)
        else:
            # 未通过 Pipeline 注入配置时，直接构造 Stage 仍从标准路径加载配置。
            config_paths = [
                Path.cwd() / "config" / "nnuNet_config.json",
                Path.cwd() / "nnuNet_config.json",
                Path(__file__).parent.parent / "config" / "nnuNet_config.json",
            ]
            config_loaded = False
            for config_path in config_paths:
                if not config_path.exists():
                    continue
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        default_config.update(json.load(f))
                    config_loaded = True
                    print(f"[推理与分割] 加载配置文件: {config_path}")
                    break
                except Exception as e:
                    print(f"[警告] 加载配置文件失败: {config_path}, 错误: {e}")
            if not config_loaded:
                print("[推理与分割] 未找到外部配置文件，使用默认配置")
        
        # 应用配置
        self._configuration = default_config.get('_configuration', "3d_fullres")
        self._fold = str(default_config.get('_fold', "0"))
        self._continue = default_config.get('_continue', True)
        self._slice_part_config = (
            default_config.get('slice_part_config')
            or load_slice_part_config()
        )
        validate_slice_part_config(self._slice_part_config)
        
        # 设置nnUNet环境变量
        self._nnunet_env_vars = {
            'nnUNet_raw': self._resolve_config_path(default_config.get('nnUNet_raw', ''), project_root),
            'nnUNet_preprocessed': self._resolve_config_path(default_config.get('nnUNet_preprocessed', ''), project_root),
            'nnUNet_results': self._resolve_config_path(default_config.get('nnUNet_results', ''), project_root)
        }

    @staticmethod
    def _resolve_config_path(value: str, project_root: Path) -> str:
        """
        解析配置中的 nnUNet 路径
        
        规则：
            1. 配置为空字符串时，不覆盖系统环境变量
            2. 支持 ~ 和环境变量展开，例如 ~/model 或 $HOME/model
            3. 相对路径会按项目根目录解析，方便跨电脑迁移项目
        """
        if not value:
            return ""
        expanded = os.path.expanduser(os.path.expandvars(str(value)))
        path = Path(expanded)
        if not path.is_absolute():
            path = project_root / path
        return str(path)
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """匹配请求对应的 nnUNet 模型，执行推理并登记标准化分割输出。"""

        request = context.request
        prev_result = context.get_result('preprocessing')
        
        if not prev_result:
            raise ValueError("前置阶段 'preprocessing' 结果不存在")
        
        nifti_path = prev_result.get('nifti_path')
        if not nifti_path:
            raise ValueError("NIfTI文件路径不存在")
        
        DebugLog("=" * 60)
        DebugLog("【第二阶段】推理分割 - 模型匹配与推理")
        DebugLog(f"  输入NIfTI: {nifti_path}")
        slice_part = self._enum_value(request.slice_part)
        study_type = self._enum_value(request.study_type)
        scan_thickness = self._normalize_scan_thickness(request.scan_thickness)

        DebugLog(f"  切割部位: {slice_part}")
        DebugLog(f"  检查类型: {study_type}")
        DebugLog(f"  扫描厚度: {scan_thickness:.3f}mm")
        
        # 构建输出路径：统一整理为 seg.nii.gz，方便后续阶段读取
        input_path = Path(nifti_path)
        output_dir = Path(request.output_dir) / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "seg.nii.gz"
        
        # [1] match_model: 匹配模型（获取对应的 nnUNet 模型ID）
        DebugLog("  [1] match_model: 匹配模型...")
        model_info = self._match_model(
            slice_part=slice_part,
            study_type=study_type,
            scan_thickness=scan_thickness
        )
        dataset_id = model_info['model_id']
        configuration = model_info.get('configuration', self._configuration)
        fold = model_info.get('fold', self._fold)
        DebugLog(
            f"    切割部位: {slice_part}, 检查类型: {study_type}, "
            f"扫描厚度: {scan_thickness:.3f}mm "
            f"-> 模型ID: {dataset_id}"
        )
        
        # [2] run_nnunet_inference: 执行推理
        DebugLog("  [2] run_nnunet_inference: 执行nnUNet推理...")
        self._run_nnunet_inference(
            input_dir=input_path.parent,
            input_path=input_path,
            output_path=output_path,
            dataset_id=dataset_id,
            configuration=configuration,
            fold=fold,
            continue_mode=self._continue
        )
        
        result = {
            'seg_path': str(output_path),
            'input_nifti': nifti_path,
            'model': model_info
        }
        
        context.set_stage_result(SegmentationResult(**result))
        
        DebugLog(f"  分割结果保存至: {output_path}")
        DebugLog("  第二阶段完成")
        DebugLog("=" * 60)
        
        return context
    
    def _match_model(
        self,
        slice_part: str,
        study_type: str,
        scan_thickness: float,
    ) -> Dict[str, Any]:
        """
        根据切割部位、检查类型和扫描厚度区间匹配对应模型。

        模型表从统一切割部位配置读取，再按扫描厚度闭区间匹配 model_id。
        
        Args:
            slice_part: 切割部位名称（如 "腰椎"、"头骨"、"肺部"）
            study_type: 检查类型（CT、MR）
            scan_thickness: 当前原始扫描层厚，单位 mm
            
        Returns:
            dict: 模型信息
        """
        normalized_study_type = (study_type or "").strip()
        normalized_scan_thickness = self._normalize_scan_thickness(scan_thickness)
        mappings = model_mappings_for(
            self._slice_part_config,
            slice_part,
            normalized_study_type,
        )
        model_mapping = self._match_model_mapping(
            mappings,
            normalized_scan_thickness,
            slice_part=slice_part,
            study_type=normalized_study_type,
        )
        thickness_range = model_mapping['scan_thickness_range_mm']
        model_id = model_mapping['model_id']
        return {
            'model_id': model_id,
            'dataset_id': model_id,
            'scan_thickness': normalized_scan_thickness,
            'min_scan_thickness': float(thickness_range['min']),
            'max_scan_thickness': float(thickness_range['max']),
            'configuration': model_mapping.get(
                'configuration',
                self._configuration,
            ),
            'fold': str(model_mapping.get('fold', self._fold)),
            'matched_by': 'slice_part_config',
        }

    @staticmethod
    def _match_model_mapping(
        mappings,
        scan_thickness: float,
        *,
        slice_part: str = "",
        study_type: str = "",
    ) -> Dict[str, Any]:
        """
        在单条模型规则中按闭区间匹配扫描厚度对应的模型。

        模型配置示例：
            [
                {
                    "scan_thickness_range_mm": {"min": 0.0, "max": 1.5},
                    "model_id": 292
                }
            ]

        上下限均包含在匹配范围内。配置重叠会导致选模不确定，因此直接报错。
        """
        if not mappings:
            raise ValueError("模型规则不能为空")

        parsed_mappings = []
        for item in mappings:
            if 'model_id' not in item:
                raise ValueError(f"模型规则缺少 model_id: {item}")
            thickness_range = item.get('scan_thickness_range_mm')
            if not isinstance(thickness_range, dict):
                raise ValueError(
                    f"模型规则缺少 scan_thickness_range_mm: {item}"
                )
            minimum = SegmentationStage._parse_model_bound(
                thickness_range,
                'min',
            )
            maximum = SegmentationStage._parse_model_bound(
                thickness_range,
                'max',
            )
            if minimum > maximum:
                raise ValueError(
                    "模型规则扫描厚度下限不能大于上限: "
                    f"{item}"
                )
            parsed_mappings.append({
                **item,
                'scan_thickness_range_mm': {
                    'min': minimum,
                    'max': maximum,
                },
            })

        for index, current in enumerate(parsed_mappings):
            for previous in parsed_mappings[:index]:
                if max(
                    current['scan_thickness_range_mm']['min'],
                    previous['scan_thickness_range_mm']['min'],
                ) <= min(
                    current['scan_thickness_range_mm']['max'],
                    previous['scan_thickness_range_mm']['max'],
                ):
                    raise ValueError(
                        "模型规则扫描厚度区间重叠: "
                        f"{previous} 与 {current}"
                    )

        matches = [
            item
            for item in parsed_mappings
            if item['scan_thickness_range_mm']['min']
            <= scan_thickness
            <= item['scan_thickness_range_mm']['max']
        ]
        if matches:
            return matches[0]

        raise ValueError(
            "未找到匹配的扫描厚度模型: "
            f"slice_part={slice_part}, study_type={study_type}, "
            f"ScanThickness={scan_thickness}mm"
        )

    @staticmethod
    def _parse_model_bound(item: Dict[str, Any], field_name: str) -> float:
        """读取并校验模型映射中的扫描厚度边界。"""

        if field_name not in item or isinstance(item[field_name], bool):
            raise ValueError(f"模型规则缺少有效 {field_name}: {item}")
        try:
            value = float(item[field_name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"模型规则的 {field_name} 必须是非负有限数值: {item}"
            ) from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"模型规则的 {field_name} 必须是非负有限数值: {item}"
            )
        return value

    @staticmethod
    def _normalize_scan_thickness(scan_thickness: Any) -> float:
        """把请求中的当前扫描层厚规范为非负有限毫米数值。"""

        if isinstance(scan_thickness, bool):
            raise ValueError("ScanThickness 必须是非负有限数值")
        try:
            value = float(scan_thickness)
        except (TypeError, ValueError) as error:
            raise ValueError("ScanThickness 必须是非负有限数值") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError("ScanThickness 必须是非负有限数值")
        return value

    @staticmethod
    def _enum_value(value: Any) -> str:
        """
        兼容枚举和普通字符串。

        CLI 会构造枚举类型；单元测试或外部调用有时可能直接传字符串。
        统一转换后，后续匹配逻辑只处理字符串，代码会简单一些。
        """
        return str(getattr(value, 'value', value))
    
    def _run_nnunet_inference(self, 
                              input_dir: Path, 
                              input_path: Path,
                              output_path: Path, 
                              dataset_id: str, 
                              configuration: str, 
                              fold: str, 
                              continue_mode: bool):
        """
        执行nnUNet推理命令
        
        Args:
            input_dir: 输入目录（包含NIfTI文件）
            input_path: 输入 NIfTI 文件路径，文件名应为 {case_id}_0000.nii.gz
            output_path: 统一整理后的输出文件路径
            dataset_id: 数据集ID
            configuration: 模型配置（如 3d_fullres）
            fold: 训练折数
            continue_mode: 是否启用继续模式
        """
        # 构建命令
        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_path.parent),  # 输出目录
            "-d", str(dataset_id),
            "-c", configuration,
            "-f", fold
        ]
        
        # 如果启用continue模式，添加--continue参数
        if continue_mode:
            cmd.append("--continue")
        
        # 设置环境变量（合并当前环境变量和nnUNet环境变量）
        env = os.environ.copy()
        for key, value in self._nnunet_env_vars.items():
            if value:  # 如果值不为空
                env[key] = value
                DebugLog(f"    设置环境变量: {key}={value}")
        
        DebugLog(f"    执行命令: {' '.join(cmd)}")
        
        try:
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env  # 传入自定义环境变量
            )
            
            # 检查执行结果
            if result.returncode != 0:
                DebugLog("    ❌ nnUNet推理失败!")
                DebugLog(f"    错误输出: {result.stderr}")
                raise RuntimeError(f"nnUNet推理失败: {result.stderr}")
            
            DebugLog("    ✅ nnUNet推理完成")
            if result.stdout:
                DebugLog(f"    输出信息: {result.stdout}")
            
            case_id = self._get_case_id(input_path)
            expected_output = output_path.parent / f"{case_id}.nii.gz"
            self._normalize_output_file(expected_output, output_path)
            
        except subprocess.CalledProcessError as e:
            ErrorLog("    ❌ nnUNet命令执行失败", e)
            raise
        except Exception as e:
            ErrorLog("    ❌ nnUNet推理过程中发生错误", e)
            raise

    @staticmethod
    def _get_case_id(input_path: Path) -> str:
        """
        从 nnUNet 输入文件名中还原 case_id。

        nnUNet 输入要求单通道文件名形如 {case_id}_0000.nii.gz；
        推理输出则通常是 {case_id}.nii.gz。比如 raw_0000.nii.gz
        对应输出 raw.nii.gz。
        """
        filename = input_path.name
        if filename.endswith(".nii.gz"):
            filename = filename[:-7]
        else:
            filename = input_path.stem
        if filename.endswith("_0000"):
            filename = filename[:-5]
        return filename

    @staticmethod
    def _normalize_output_file(expected_output: Path, output_path: Path) -> None:
        """
        将 nnUNet 输出整理成固定文件名。

        当前后续阶段统一读取 segmentation/seg.nii.gz。nnUNet 原始输出
        通常是 {case_id}.nii.gz，所以这里先找预期输出；如果不同版本或
        不同参数导致文件名不一致，再在输出目录中寻找唯一的 nii.gz 结果。
        """
        if expected_output.exists():
            if expected_output != output_path:
                if output_path.exists():
                    output_path.unlink()
                shutil.move(str(expected_output), str(output_path))
            return

        possible_outputs = [
            path for path in output_path.parent.glob("*.nii.gz")
            if path != output_path
        ]
        if possible_outputs:
            found_file = possible_outputs[0]
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(found_file), str(output_path))
            return

        if output_path.exists():
            return

        raise FileNotFoundError(f"未找到nnUNet输出文件: {output_path.parent}")
