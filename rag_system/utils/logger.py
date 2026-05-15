"""
utils/logger.py
───────────────
Centralized logging with color output for development.
All modules import get_logger() from here.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.error("Something broke", exc_info=True)
"""

import logging
import sys
from config.settings import settings

try:
    import colorlog
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with colored output (if colorlog is installed).
    Calling multiple times with the same name returns the same logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)

    if _HAS_COLOR:
        formatter = colorlog.ColoredFormatter(
            fmt=(
                "%(log_color)s%(asctime)s%(reset)s "
                "%(cyan)s[%(name)s]%(reset)s "
                "%(log_color)s%(levelname)-8s%(reset)s "
                "%(message)s"
            ),
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG":    "white",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Don't double-log to root logger
    return logger
