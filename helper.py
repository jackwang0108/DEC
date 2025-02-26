"""
helper 定义了工具函数, 用于辅助其他模块的功能实现

    @Time    : 2025/02/26
    @Author  : JackWang
    @File    : helper.py
    @IDE     : VsCode
"""

# Standard Library
import sys
from pathlib import Path

# Third-Party Library
from loguru import logger

# Torch Library

# My Library


def get_logger(log_file: Path, with_time: bool = True):
    global logger

    logger.remove()
    logger.add(
        log_file,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ {{message}}",
    )
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ <level>{{message}}</level>",
    )

    return logger
