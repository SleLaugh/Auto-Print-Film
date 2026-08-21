#!/src/core/__init__.py
# -*- coding: utf-8 -*-

"""
核心框架包
"""

from .base_stage import BaseStage
from .context import ProcessingContext
from .pipeline import Pipeline
from .response import Response
from .errors import FrameworkError
from .lifecycle import JobStatus, StageStatus, StageRecord
from .delivery_report import (
    build_status_report,
    write_failure_report,
    write_report,
)

__all__ = [
    'BaseStage',
    'ProcessingContext',
    'Pipeline',
    'Response',
    'FrameworkError',
    'JobStatus',
    'StageStatus',
    'StageRecord',
    'build_status_report',
    'write_failure_report',
    'write_report',
]
