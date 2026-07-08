#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ruff: noqa: E402

"""
SpineMPRProcessor - 主入口
"""

import sys
import argparse
import json
from pathlib import Path

# ===== 添加项目根目录到Python路径（必须在import之前）=====
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ===== 导入（在 sys.path 操作之后）=====
from src.core import Pipeline
from src.models import Request, BodyPartEnum, CheckTypeEnum, ScanThicknessEnum, CutModeEnum, PlaneConfig
from src.utils import RunLog, ErrorLog, DebugLog  


def parse_arguments():
    """
    解析命令行参数函数
    作用：让用户通过命令行指定输入输出路径和部位类型
    返回值：解析后的参数对象
    """
    # 创建参数解析器，description是脚本功能的简短描述
    parser = argparse.ArgumentParser(description="SpineMPRProcessor")
    
    # 添加 -i 或 --input 参数
    # required=True 表示这个参数必须提供，否则程序报错
    # help 是参数说明，运行时用 -h 会显示
    parser.add_argument(
        '-i', '--input', 
        type=str,           # 参数类型是字符串
        required=True,      # 必填参数
        help='输入DICOM目录'  # 参数说明
    )
    
    # 添加 -o 或 --output 参数
    # 也是必填，指定处理结果的保存位置
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True, 
        help='输出目录'
    )
    
    # 添加 --body_part 参数
    # default='腰椎' 表示如果不指定，默认处理腰椎
    # choices 限制了只能从列表中选一个值，输入其他会报错
    body_part_choices = [e.value for e in BodyPartEnum]
    parser.add_argument(
        '--body_part',
        '--body-part',
        '--target_part',
        '--target-part',
        dest='body_part',
        type=str,
        default='腰椎',
        choices=body_part_choices,
        help=f'部位，可选值: {", ".join(body_part_choices)}'
    )

    parser.add_argument(
        '--check_type',
        '--check-type',
        dest='check_type',
        type=str,
        default=CheckTypeEnum.CT.value,
        choices=[e.value for e in CheckTypeEnum],
        help='检查类型'
    )

    parser.add_argument(
        '--scan_thickness',
        '--scan-thickness',
        dest='scan_thickness',
        type=str,
        default=ScanThicknessEnum.默认.value,
        choices=[e.value for e in ScanThicknessEnum],
        help='扫描厚度档位，用于匹配对应模型'
    )

    parser.add_argument(
        '--cut_mode',
        '--cut-mode',
        dest='cut_mode',
        type=str,
        default=CutModeEnum.整切.value,
        choices=[e.value for e in CutModeEnum],
        help='切割方式：whole_range 表示整切，interspace 表示椎间切'
    )

    parser.add_argument(
        '--plane_configs',
        '--plane-configs',
        dest='plane_configs',
        type=str,
        default=None,
        help='三平面重建参数 JSON 数组，例如: [{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗"]}]'
    )
    
    # 解析命令行输入的参数
    # 比如运行：python script.py -i /path/to/dicom -o /path/to/output --body_part 颈椎
    # 就会把参数解析成对象返回
    return parser.parse_args()


def parse_plane_configs(value):
    """
    解析命令行传入的三平面配置 JSON。

    返回 None 表示调用方没有显式传参，此时 Request 使用默认三平面骨窗配置。
    """
    if value is None:
        return None

    try:
        raw_configs = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"plane_configs 不是合法 JSON: {e}") from e

    if not isinstance(raw_configs, list):
        raise ValueError("plane_configs 必须是 JSON 数组")

    return [
        PlaneConfig.from_dict(item)
        for item in raw_configs
    ]


def main():
    RunLog("SpineMPRProcessor v1.0.0 启动")
    
    try:
        args = parse_arguments()

        plane_configs = parse_plane_configs(args.plane_configs)
        request_kwargs = {
            'dicom_dir': args.input,
            'output_dir': args.output,
            'body_part': BodyPartEnum(args.body_part),
            'check_type': CheckTypeEnum(args.check_type),
            'scan_thickness': ScanThicknessEnum(args.scan_thickness),
            'cut_mode': CutModeEnum(args.cut_mode),
        }
        if plane_configs is not None:
            request_kwargs['plane_configs'] = plane_configs

        request = Request(**request_kwargs)

        is_valid, error_message = request.validate()
        if not is_valid:
            raise ValueError(error_message)
        
        DebugLog(f"输入路径: {request.dicom_dir}")
        DebugLog(f"输出路径: {request.output_dir}")
        DebugLog(f"部位: {request.body_part.value}")
        DebugLog(f"检查类型: {request.check_type.value}")
        DebugLog(f"扫描厚度: {request.scan_thickness.value}")
        DebugLog(f"切割方式: {request.cut_mode.value}")
        DebugLog(f"三平面参数: {request.to_dict()['plane_configs']}")
        
        pipeline = Pipeline()
        response = pipeline.run(request)
        
        if response.success:
            RunLog(f"✅ 成功，总耗时: {response.total_time:.2f}s")
            DebugLog(f"输出目录: {response.output_dir}")
            return 0
        else:
            RunLog("❌ 处理失败")
            for error in response.errors:
                DebugLog(f"  错误: [{error['stage']}] {error['message']}")
            return 1
            
    except Exception as e:
        ErrorLog("程序执行异常", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
