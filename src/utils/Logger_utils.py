#!/src/utils/Logger_utils.py
# -*- coding: utf-8 -*-

"""
日志工具模块
提供统一的日志管理功能，支持按日期和小时分类存储
支持运行日志、调试日志和错误日志三种类型

日志写入规则：
- RunLog: 写 RunLog 文件（根据 run_log_file_enabled 控制）
          + 如果 debug_log_file_enabled=True，额外写 DebugLog 文件
- DebugLog: 只写 DebugLog 文件（根据 debug_log_file_enabled 控制）
- ErrorLog: 写 ErrorLog 文件（根据 error_log_file_enabled 控制）
            + 如果 debug_log_file_enabled=True，额外写 DebugLog 文件
- 控制台统一输出（根据 console_enabled 控制）
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import sys
import json


class LoggerManager:
    """
    日志管理器类（单例模式）
    负责创建和管理日志记录器，支持按日期和小时分类存储
    """
    
    # 单例模式
    _instance = None
    
    # 日志类型常量
    LOG_TYPE_RUN = "RunLog"
    LOG_TYPE_DEBUG = "DebugLog"
    LOG_TYPE_ERROR = "ErrorLog"
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日志管理器"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._base_dir = Path.cwd() / "Log"
            self._file_loggers = {}      # 文件日志记录器
            self._console_logger = None  # 控制台日志记录器（单例）
            
            # ===== 加载配置文件 =====
            self._load_config()
            # =====================
    
    def _load_config(self):
        """
        加载配置文件
        优先加载外部JSON配置文件，如果不存在则使用默认配置
        """
        # 默认配置
        default_config = {
            # 控制台统一输出
            'console_enabled': True,
            
            # 文件写入开关（各自独立控制）
            'run_log_file_enabled': True,
            'debug_log_file_enabled': True,
            'error_log_file_enabled': True,
            
            'log_encoding': 'utf-8',
            'base_dir': 'Log',
        }
        
        # 尝试加载外部配置文件
        config_paths = [
            Path.cwd() / "config" / "logger_config.json",
            Path.cwd() / "logger_config.json",
            Path(__file__).parent.parent / "config" / "logger_config.json",
        ]
        
        config_loaded = False
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    default_config.update(config)
                    config_loaded = True
                    print(f"[日志] 加载配置文件: {config_path}")
                    break
                except Exception as e:
                    print(f"[警告] 加载配置文件失败: {config_path}, 错误: {e}")
        
        if not config_loaded:
            print("[日志] 未找到外部配置文件，使用默认配置")
        
        # 应用配置
        self.console_enabled = default_config.get('console_enabled', True)
        self.run_log_file_enabled = default_config.get('run_log_file_enabled', True)
        self.debug_log_file_enabled = default_config.get('debug_log_file_enabled', True)
        self.error_log_file_enabled = default_config.get('error_log_file_enabled', True)
        self.log_encoding = default_config.get('log_encoding', 'utf-8')
        self.base_dir = Path(default_config.get('base_dir', 'Log'))
    
        
    def run_log(self, message: str):
        """写入运行日志"""
        self._write_log(self.LOG_TYPE_RUN, message)
    
    def debug_log(self, message: str, exception: Optional[Exception] = None):
        """写入调试日志"""
        self._write_log(self.LOG_TYPE_DEBUG, message, exception)
    
    def error_log(self, message: str, exception: Optional[Exception] = None):
        """写入错误日志"""
        self._write_log(self.LOG_TYPE_ERROR, message, exception)
    
    def _write_log(self, log_type: str, message: str, exception: Optional[Exception] = None):
        """
        写入日志的通用方法
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 构建日志内容
        log_lines = [f"[{timestamp}]"]
        log_lines.append(message)
        
        if exception is not None:
            error_msg = str(exception)
            if error_msg:
                log_lines.append(f"异常信息: {error_msg}")
            else:
                log_lines.append(f"异常类型: {type(exception).__name__}")
        
        log_lines.append("")
        log_content = "\n".join(log_lines)
        
        # ===== 1. 控制台输出（统一输出一次） =====
        if self.console_enabled:
            console_logger = self._get_console_logger()
            console_logger.info(log_content)
        
        # ===== 2. 文件写入（根据规则） =====
        self._write_to_files(log_type, log_content)
    
    def _get_console_logger(self) -> logging.Logger:
        """获取控制台日志记录器（统一输出到控制台）"""
        if self._console_logger is not None:
            return self._console_logger
        
        logger_name = "ConsoleLogger"
        logger = logging.getLogger(logger_name)
        
        if logger.handlers:
            self._console_logger = logger
            return logger
        
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # 不传播到根logger，避免重复
        
        # 只添加控制台处理器
        if self.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(console_handler)
        
        self._console_logger = logger
        return logger
    
    def _write_to_files(self, log_type: str, log_content: str):
        """
        根据日志类型写入对应的文件
        
        规则：
        - RunLog: 写 RunLog + (如果debug开启则写 DebugLog)
        - DebugLog: 只写 DebugLog
        - ErrorLog: 写 ErrorLog + (如果debug开启则写 DebugLog)
        """
        # 确定要写入的文件类型列表
        file_types_to_write = []
        
        if log_type == self.LOG_TYPE_RUN:
            # RunLog: 写 RunLog
            if self._is_file_enabled(self.LOG_TYPE_RUN):
                file_types_to_write.append(self.LOG_TYPE_RUN)
            # 如果debug开启，额外写 DebugLog
            if self._is_file_enabled(self.LOG_TYPE_DEBUG):
                file_types_to_write.append(self.LOG_TYPE_DEBUG)
                
        elif log_type == self.LOG_TYPE_DEBUG:
            # DebugLog: 只写 DebugLog
            if self._is_file_enabled(self.LOG_TYPE_DEBUG):
                file_types_to_write.append(self.LOG_TYPE_DEBUG)
                
        elif log_type == self.LOG_TYPE_ERROR:
            # ErrorLog: 写 ErrorLog
            if self._is_file_enabled(self.LOG_TYPE_ERROR):
                file_types_to_write.append(self.LOG_TYPE_ERROR)
            # 如果debug开启，额外写 DebugLog
            if self._is_file_enabled(self.LOG_TYPE_DEBUG):
                file_types_to_write.append(self.LOG_TYPE_DEBUG)
        
        # 写入对应的文件
        for file_type in file_types_to_write:
            file_logger = self._get_file_logger(file_type)
            file_logger.info(log_content)
    
    def _get_file_logger(self, log_type: str) -> logging.Logger:
        """获取文件日志记录器（只写文件，不输出控制台）"""
        logger_name = f"FileLogger_{log_type}"
        
        if logger_name in self._file_loggers:
            return self._file_loggers[logger_name]
        
        logger = logging.getLogger(logger_name)
        
        if logger.handlers:
            self._file_loggers[logger_name] = logger
            return logger
        
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # 不传播到根logger，避免重复
        
        # 只添加文件处理器
        log_file = self._get_log_file_path(log_type)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_file,
            encoding=self.log_encoding
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        
        self._file_loggers[logger_name] = logger
        return logger

    def _get_log_file_path(self, log_type: str) -> Path:
        """根据日志类型获取日志文件路径"""
        now = datetime.now()
        date_dir = now.strftime("%Y%m%d")
        log_filename = now.strftime("%Y%m%d%H") + ".txt"
        log_path = self.base_dir / log_type / date_dir / log_filename
        return log_path
    
    def _is_file_enabled(self, log_type: str) -> bool:
        """
        检查指定日志类型的文件写入是否启用
        
        Args:
            log_type: 日志类型
        
        Returns:
            bool: 是否启用文件写入
        """
        if log_type == self.LOG_TYPE_RUN:
            return self.run_log_file_enabled
        elif log_type == self.LOG_TYPE_DEBUG:
            return self.debug_log_file_enabled
        elif log_type == self.LOG_TYPE_ERROR:
            return self.error_log_file_enabled
        return False
  

# 全局日志实例
_logger_manager = LoggerManager()


def RunLog(message: str):
    """写入运行日志"""
    _logger_manager.run_log(message)


def DebugLog(message: str, exception: Optional[Exception] = None):
    """写入调试日志"""
    _logger_manager.debug_log(message, exception)


def ErrorLog(message: str, exception: Optional[Exception] = None):
    """写入错误日志"""
    _logger_manager.error_log(message, exception)


def is_debug_log_file_enabled() -> bool:
    """返回是否保留调试文件产物。"""

    return bool(_logger_manager.debug_log_file_enabled)


__all__ = ['RunLog', 'DebugLog', 'ErrorLog', 'is_debug_log_file_enabled']
