#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ruff: noqa: E402

"""
SpineMPRProcessor - 主入口
"""

import sys
import argparse
import faulthandler
import json
from pathlib import Path

# ===== 添加项目根目录到Python路径（必须在import之前）=====
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ===== 导入（在 sys.path 操作之后）=====
from src.core import Pipeline
from src.core.delivery_report import write_failure_report
from src.config import load_app_config
from src.models import (
    Request,
    SlicePartEnum,
    StudyTypeEnum,
    PlaneConfig,
    parse_window_presets,
)
from src.utils import RunLog, ErrorLog, DebugLog, configure_logging


class ArgumentParseError(SystemExit):
    """保留 argparse 退出语义，同时让失败报告取得具体错误。"""

    def __init__(self, message):
        super().__init__(2)
        self.message = str(message)


class ReportArgumentParser(argparse.ArgumentParser):
    """把 argparse 的错误文本附加到可捕获异常。"""

    def error(self, message):
        self.print_usage(sys.stderr)
        self._print_message(f"{self.prog}: error: {message}\n", sys.stderr)
        raise ArgumentParseError(message)


def parse_arguments():
    """
    解析命令行参数函数
    作用：让用户通过命令行指定输入输出路径和切割部位
    返回值：解析后的参数对象
    """
    # 创建参数解析器，description是脚本功能的简短描述
    parser = ReportArgumentParser(description="SpineMPRProcessor")
    
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
    
    # 添加 --slice_part 参数
    # default='腰椎' 表示如果不指定，默认处理腰椎
    # choices 限制了只能从列表中选一个值，输入其他会报错
    slice_part_choices = [e.value for e in SlicePartEnum]
    parser.add_argument(
        '--slice_part',
        dest='slice_part',
        type=str,
        default='腰椎',
        choices=slice_part_choices,
        help=f'切割部位，可选值: {", ".join(slice_part_choices)}'
    )

    parser.add_argument(
        '--study_type',
        dest='study_type',
        type=str,
        default=StudyTypeEnum.CT.value,
        choices=[e.value for e in StudyTypeEnum],
        help='检查类型'
    )

    parser.add_argument(
        '--scan_thickness',
        dest='scan_thickness',
        type=float,
        required=True,
        help='当前原始扫描层厚，单位 mm，用于按区间匹配模型'
    )

    parser.add_argument(
        '-c', '--cache_dir',
        dest='cache_dir',
        type=str,
        default=None,
        help='缓存根目录；第一阶段写入 nifti，第二阶段按模型 ID 保存',
    )

    parser.add_argument(
        '--plane_configs',
        dest='plane_configs',
        type=str,
        default=None,
        # 这里先采用 JSON 字符串，避免为一个小参数组同时维护命令行文件入口。
        help='三平面重建参数 JSON 数组，层厚、层间距、数量至少传两项，值为 0 表示未配置，例如: [{"Plane":"冠状位","SliceThickness":5,"SliceImageCount":20,"WindowPresets":["骨窗"],"SliceType":"MPR"}]'
    )

    parser.add_argument(
        '--window_presets',
        dest='window_presets',
        type=str,
        default=None,
        help='本次任务窗宽窗位 JSON 数组，例如: [{"Name":"骨窗","windowWidth":50,"windowLevel":60}]',
    )

    # 解析命令行输入的参数
    # 比如运行：python script.py -i /path/to/dicom -o /path/to/output --slice_part 颈椎
    # 就会把参数解析成对象返回
    return parser.parse_args()


def parse_plane_configs(value):
    """
    解析命令行传入的三平面配置 JSON。

    返回 None 表示调用方没有显式传参，此时 Request 使用默认横状位骨窗配置。
    """
    if value is None:
        return None

    try:
        raw_configs = json.loads(value)
    except json.JSONDecodeError as e:
        # 保留原始 JSONDecodeError 作为 cause，调试时能看到具体列号。
        raise ValueError(f"plane_configs 不是合法 JSON: {e}") from e

    if not isinstance(raw_configs, list):
        raise ValueError("plane_configs 必须是 JSON 数组")

    # 入口只负责 JSON 解析；单项结构和枚举合法性统一交给 PlaneConfig.from_dict。
    return [
        PlaneConfig.from_dict(item)
        for item in raw_configs
    ]


def parse_window_presets_argument(value):
    """解析命令行窗宽窗位覆盖列表。"""

    if value is None:
        return []
    try:
        raw_presets = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"window_presets 不是合法 JSON: {e}") from e
    return parse_window_presets(raw_presets)


def main():
    # SimpleITK/GDCM 等原生扩展触发 SIGABRT 时，至少保留 Python 调用栈。
    faulthandler.enable(all_threads=True)
    output_dir = _cli_value(sys.argv[1:], "-o", "--output")
    slice_part = _cli_value(sys.argv[1:], "--slice_part") or "腰椎"
    study_type = _cli_value(sys.argv[1:], "--study_type") or StudyTypeEnum.CT.value
    scan_thickness = _cli_value(sys.argv[1:], "--scan_thickness")
    request = None
    try:
        args = parse_arguments()
        output_dir = args.output
        slice_part = args.slice_part
        study_type = args.study_type
        scan_thickness = args.scan_thickness

        app_config = load_app_config()
        configure_logging(app_config.logging, output_dir=args.output)
        RunLog("SpineMPRProcessor v1.0.0 启动")

        plane_configs = parse_plane_configs(args.plane_configs)
        window_presets = parse_window_presets_argument(args.window_presets)
        # 先组装基础参数，再按需放入 plane_configs。这样未传该参数时仍使用
        # Request 的默认横状位骨窗配置，保持老命令兼容。
        request_kwargs = {
            'dicom_dir': args.input,
            'output_dir': args.output,
            'slice_part': SlicePartEnum(args.slice_part),
            'study_type': StudyTypeEnum(args.study_type),
            'scan_thickness': args.scan_thickness,
        }
        if plane_configs is not None:
            request_kwargs['plane_configs'] = plane_configs
        if window_presets:
            request_kwargs['window_presets'] = window_presets
        if args.cache_dir is not None:
            request_kwargs['cache_dir'] = args.cache_dir
        request = Request(**request_kwargs)

        is_valid, error_message = request.validate()
        if not is_valid:
            raise ValueError(error_message)
        
        DebugLog(f"输入路径: {request.dicom_dir}")
        DebugLog(f"输出路径: {request.output_dir}")
        DebugLog(f"切割部位: {request.slice_part.value}")
        DebugLog(f"检查类型: {request.study_type.value}")
        DebugLog(f"扫描厚度: {request.scan_thickness}mm")
        if request.cache_dir:
            DebugLog(f"缓存根目录: {request.cache_dir}")
        DebugLog(f"三平面参数: {request.to_dict()['plane_configs']}")
        
        pipeline = Pipeline(config=app_config)
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
            
    except ArgumentParseError as e:
        if output_dir:
            _try_write_failure_report(
                output_dir,
                e.message,
                slice_part=slice_part,
                study_type=study_type,
                scan_thickness=scan_thickness,
            )
        return int(e.code)
    except KeyboardInterrupt as e:
        if output_dir:
            _try_write_failure_report(
                output_dir,
                "任务被中断",
                request=request,
                slice_part=slice_part,
                study_type=study_type,
                scan_thickness=scan_thickness,
            )
        ErrorLog("程序执行被中断", e)
        return 130
    except SystemExit as e:
        if e.code and output_dir:
            _try_write_failure_report(
                output_dir,
                f"程序主动退出，退出码: {e.code}",
                request=request,
                slice_part=slice_part,
                study_type=study_type,
                scan_thickness=scan_thickness,
            )
        return int(e.code or 0)
    except Exception as e:
        if output_dir:
            _try_write_failure_report(
                output_dir,
                str(e) or type(e).__name__,
                request=request,
                slice_part=slice_part,
                study_type=study_type,
                scan_thickness=scan_thickness,
            )
        ErrorLog("程序执行异常", e)
        return 1


def _cli_value(arguments, *names):
    """在 argparse 失败前尽早提取报告目录和请求信息。"""

    for index, argument in enumerate(arguments):
        if argument in names and index + 1 < len(arguments):
            return arguments[index + 1]
        for name in names:
            prefix = f"{name}="
            if argument.startswith(prefix):
                return argument[len(prefix):]
    return None


def _try_write_failure_report(
    output_dir,
    message,
    *,
    request=None,
    slice_part=None,
    study_type=None,
    scan_thickness=None,
):
    """尽力写入失败报告，不让报告异常覆盖真正的处理错误。"""

    try:
        write_failure_report(
            output_dir,
            message,
            request=request,
            slice_part=slice_part,
            study_type=study_type,
            scan_thickness=scan_thickness,
        )
    except Exception as report_error:
        ErrorLog("写入失败 report.json 时发生异常", report_error)


if __name__ == "__main__":
    sys.exit(main())
