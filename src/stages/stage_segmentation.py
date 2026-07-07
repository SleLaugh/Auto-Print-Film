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
        project_root = Path(__file__).resolve().parents[2]
        default_config = {
            "_configuration": "3d_fullres",  # 模型配置：使用哪种网络架构配置
            "_fold": "0",  # 训练折数（fold）：使用哪个折的模型权重
            "_continue": True,  # 继续模式：如果输出文件已存在则跳过，避免重复预测
            "_model_rules": [],
            # nnUNet环境变量。为空时沿用系统环境变量；相对路径按项目根目录解析。
            "nnUNet_raw": "",
            "nnUNet_preprocessed": "",
            "nnUNet_results": ""
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
        self._model_rules = default_config.get('_model_rules', [])
        
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
        request = context.request
        prev_result = context.get_result('preprocessing')
        
        if not prev_result:
            raise ValueError("前置阶段 'preprocessing' 结果不存在")
        
        nifti_path = prev_result.get('nifti_path')
        if not nifti_path:
            raise ValueError("NIfTI文件路径不存在")
        
        DebugLog("【阶段二】推理分割 - 模型匹配与推理")
        DebugLog(f"  输入NIfTI: {nifti_path}")
        body_part = self._enum_value(request.body_part)
        check_type = self._enum_value(request.check_type)
        scan_thickness = self._enum_value(getattr(request, 'scan_thickness', 'default'))

        DebugLog(f"  部位类型: {body_part}")
        DebugLog(f"  检查类型: {check_type}")
        DebugLog(f"  扫描厚度: {scan_thickness}")
        
        # 构建输出路径：统一整理为 seg.nii.gz，方便后续阶段读取
        input_path = Path(nifti_path)
        output_dir = Path(request.output_dir) / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "seg.nii.gz"
        
        # [1] match_model: 匹配模型（获取对应的 nnUNet 模型ID）
        DebugLog("  [1] match_model: 匹配模型...")
        model_info = self._match_model(
            body_part=body_part,
            check_type=check_type,
            scan_thickness=scan_thickness
        )
        dataset_id = model_info['model_id']
        configuration = model_info.get('configuration', self._configuration)
        fold = model_info.get('fold', self._fold)
        DebugLog(
            f"    部位: {body_part}, 检查类型: {check_type}, "
            f"扫描厚度: {scan_thickness} "
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
        
        context.set_result(self.name, result)
        
        DebugLog(f"  ✅ 分割结果保存至: {output_path}")
        DebugLog("=" * 60)
        
        return context
    
    def _match_model(
        self,
        body_part: str,
        check_type: str = "",
        scan_thickness: str = "default"
    ) -> Dict[str, Any]:
        """
        根据身体部位、检查类型和扫描厚度档位匹配对应模型。

        配置结构分两层：
            1. _model_rules: 匹配 body_part + check_type
            2. model_mapper: 在命中的 rule 内匹配 scan_thickness -> model_id
        
        Args:
            body_part: 身体部位名称（如 "腰椎"）
            check_type: 检查类型（CT、MR）
            scan_thickness: 扫描厚度档位（default、3mm、6mm）
            
        Returns:
            dict: 模型信息
        """
        normalized_check_type = (check_type or "").strip().lower()
        normalized_scan_thickness = self._normalize_scan_thickness(scan_thickness)

        for rule in self._model_rules:
            # 第一层：部位和检查类型命中，才进入该 rule 的扫描厚度映射。
            if rule.get('body_part') and rule['body_part'] != body_part:
                continue

            rule_check_type = str(rule.get('check_type', '')).strip().lower()
            if rule_check_type and rule_check_type != normalized_check_type:
                continue

            model_id = self._match_model_mapper(rule, normalized_scan_thickness)
            return {
                'model_id': model_id,
                'dataset_id': model_id,
                'scan_thickness': normalized_scan_thickness,
                'configuration': rule.get('configuration', self._configuration),
                'fold': str(rule.get('fold', self._fold)),
                'matched_by': 'model_rules'
            }

        raise ValueError(
            "未找到匹配的模型规则: "
            f"body_part={body_part}, check_type={check_type}, scan_thickness={normalized_scan_thickness}"
        )

    @staticmethod
    def _match_model_mapper(rule: Dict[str, Any], scan_thickness: str) -> Any:
        """
        在单条模型规则中匹配扫描厚度对应的模型 ID。

        model_mapper 示例：
            [
                {"scan_thickness": "default", "model_id": 292},
                {"scan_thickness": "3mm", "model_id": 292},
                {"scan_thickness": "6mm", "model_id": 292}
            ]

        这里的 scan_thickness 对应 Request.scan_thickness，
        model_id 对应 nnUNet 的 -d。
        """
        mapper = rule.get('model_mapper', [])
        if not mapper:
            raise ValueError(f"模型规则缺少 model_mapper: {rule}")

        for item in mapper:
            mapper_scan_thickness = str(item.get('scan_thickness', 'default')).strip().lower()
            if mapper_scan_thickness != scan_thickness:
                continue
            if 'model_id' not in item:
                raise ValueError(f"model_mapper 缺少 model_id: {item}")
            return item['model_id']

        raise ValueError(
            "未找到匹配的扫描厚度模型: "
            f"body_part={rule.get('body_part')}, check_type={rule.get('check_type')}, "
            f"scan_thickness={scan_thickness}"
        )

    @staticmethod
    def _normalize_scan_thickness(scan_thickness: str) -> str:
        """
        归一化扫描厚度档位。

        配置和命令行当前只允许 default、3mm、6mm。这里保留一个兜底，
        防止外部 API 直接构造 Request 时传入大小写或空值导致匹配异常。
        """
        value = str(scan_thickness or "default").strip().lower()
        if value in {"default", "3mm", "6mm"}:
            return value
        return "default"

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
