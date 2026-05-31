"""
统一日志配置

每个模块按以下约定获取 logger：
    from core.utils.logger import get_logger
    logger = get_logger(__name__)

日志输出：
  - Console: INFO 级别及以上，带颜色
  - File (logs/app.log): DEBUG 级别及以上，包含完整上下文

环境变量：
  LOG_LEVEL: 全局日志级别，默认 INFO
  LOG_DIR:   日志目录，默认项目根目录下的 logs/
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(os.getenv("LOG_DIR", Path(__file__).parent.parent.parent / "logs"))


class ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式化"""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        # 简化模块名：core.agents.base -> agents.base
        module = record.name
        if module.startswith("core."):
            module = module[5:]
        if module.startswith("hk_law."):
            module = module[7:]

        record.levelname_colored = f"{color}{record.levelname:8s}{reset}"
        record.module_short = module
        return super().format(record)


def _setup_logging() -> None:
    """初始化根 logger 配置"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # 清除已有 handler，避免重复
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console handler —— 带颜色，INFO+
    console_fmt = ColoredFormatter(
        fmt="%(levelname_colored)s | %(module_short)-20s | %(message)s",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler —— 完整格式，DEBUG+
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "app.log"
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """获取指定模块的 logger"""
    return logging.getLogger(name)


# 首次导入时自动配置
_setup_logging()
