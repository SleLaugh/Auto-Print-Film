#!/src/stages/stage_segmentation.py
# -*- coding: utf-8 -*-

"""
阶段二：推理分割
模型匹配与推理
"""

from pathlib import Path
from typing import Dict, Any
import json
import subprocess
import shutil
import os

from src.core.base_stage import BaseStage
from src.core.context import ProcessingContext
from src.utils import DebugLog, ErrorLog


class SegmentationStage(BaseStage):
    """推理分割阶段"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="segmentation", config=config)
        self._load_config()
    
    def _load_config(self):
        """
        加载配置文件
        优先加载外部JSON配置文件，如果不存在则使用默认配置
        """
        # 默认配置
        default_config = {
            "_configuration": "3d_fullres",  # 模型配置：使用哪种网络架构配置
            "_fold": "0",  # 训练折数（fold）：使用哪个折的模型权重
            "_continue": True,  # 继续模式：如果输出文件已存在则跳过，避免重复预测
            "_dataset_id": 292,  # 数据集ID
            # nnUNet环境变量
            "nnUNet_raw": "/home/peng/03_data/00_nnUnet/model/nnUNet_raw",
            "nnUNet_preprocessed": "/home/peng/03_data/00_nnUnet/model/nnUNet_preprocessed",
            "nnUNet_results": "/home/peng/03_data/00_nnUnet/model/nnUNet_results"
        }
        
        # 尝试加载外部配置文件
        config_paths = [
            Path.cwd() / "config" / "nnuNet_config.json",
            Path.cwd() / "nnuNet_config.json",
            Path(__file__).parent.parent / "config" / "nnuNet_config.json",
        ]
        
        config_loaded = False
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    default_config.update(config)
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
        self._dataset_id_mapping = default_config.get('_dataset_id_mapping', {})
        
        # 设置nnUNet环境变量
        self._nnunet_env_vars = {
            'nnUNet_raw': default_config.get('nnUNet_raw', ''),
            'nnUNet_preprocessed': default_config.get('nnUNet_preprocessed', ''),
            'nnUNet_results': default_config.get('nnUNet_results', '')
        }
    
    def execute(self, context: ProcessingContext) -> ProcessingContext:
        request = context.request
        prev_result = context.get_result('preprocessing')
        
        if not prev_result:
            raise ValueError("前置阶段 'preprocessing' 结果不存在")
        
        nifti_path = prev_result.get('nifti_path')
        if not nifti_path:
            raise ValueError("NIfTI文件路径不存在")
        
        DebugLog("【阶段二】推理分割 - 模型匹配与推理")
        DebugLog(f"  输入NIfTI: {nifti_path}")
        DebugLog(f"  部位类型: {request.body_part.value}")
        
        # 构建输出路径：request.output_dir / segmentation / {输入文件名}.nii.gz
        input_path = Path(nifti_path)
        input_filename = input_path.stem  # 不带扩展名的文件名
        output_dir = Path(request.output_dir) / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_filename}.nii.gz"
        
        # [1] match_model: 匹配模型（获取对应的数据集ID）
        DebugLog("  [1] match_model: 匹配模型...")
        dataset_id, model_name = self._match_model(request.body_part.value)
        DebugLog(f"    部位: {request.body_part.value} -> 数据集ID: {dataset_id}, 模型: {model_name}")
        
        # [2] run_nnunet_inference: 执行推理
        DebugLog("  [2] run_nnunet_inference: 执行nnUNet推理...")
        self._run_nnunet_inference(
            input_dir=input_path.parent,
            output_path=output_path,
            dataset_id=dataset_id,
            configuration=self._configuration,
            fold=self._fold,
            continue_mode=self._continue
        )
        
        result = {
            'seg_path': str(output_path),
            'input_nifti': nifti_path
        }
        
        context.set_result(self.name, result)
        
        DebugLog(f"  ✅ 分割结果保存至: {output_path}")
        DebugLog("=" * 60)
        
        return context
    
    def _match_model(self, body_part: str) -> tuple:
        """
        根据身体部位匹配对应的数据集ID和模型名称
        
        Args:
            body_part: 身体部位名称（如 "腰椎"）
            
        Returns:
            tuple: (dataset_id, model_name)
        """
        # 从映射中获取数据集ID，如果不存在则使用默认值292
        dataset_id = self._dataset_id_mapping.get(body_part, 292)
        
        # 构建模型名称
        model_name = f"nnUNet_{body_part}_model"
        
        return dataset_id, model_name
    
    def _run_nnunet_inference(self, 
                              input_dir: Path, 
                              output_path: Path, 
                              dataset_id: str, 
                              configuration: str, 
                              fold: str, 
                              continue_mode: bool):
        """
        执行nnUNet推理命令
        
        Args:
            input_dir: 输入目录（包含NIfTI文件）
            output_path: 输出文件路径
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
            
            # 检查输出文件是否存在
            expected_output = output_path.parent / f"{output_path.stem}.nii.gz"
            if not expected_output.exists():
                # 如果输出文件名称与预期不符，尝试查找
                possible_outputs = list(output_path.parent.glob("*.nii.gz"))
                if possible_outputs:
                    # 重命名为预期名称
                    found_file = possible_outputs[0]
                    DebugLog(f"    找到输出文件: {found_file.name}, 重命名为: {output_path.name}")
                    shutil.move(str(found_file), str(output_path))
                else:
                    raise FileNotFoundError(f"未找到nnUNet输出文件: {output_path.parent}")
            else:
                # 如果输出文件存在但名称不同，重命名
                if expected_output != output_path:
                    shutil.move(str(expected_output), str(output_path))
                    DebugLog(f"    重命名输出文件: {expected_output.name} -> {output_path.name}")
            
        except subprocess.CalledProcessError as e:
            ErrorLog("    ❌ nnUNet命令执行失败", e)
            raise
        except Exception as e:
            ErrorLog("    ❌ nnUNet推理过程中发生错误", e)
            raise