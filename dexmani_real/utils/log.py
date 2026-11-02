from __future__ import annotations

import logging
import sys

CONSOLE_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)


def configure_logging() -> None:
    logger = logging.getLogger("dexmani_real")
    if logger.handlers:
        return
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(CONSOLE_FORMATTER)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def error_detail(stage: str, error: BaseException) -> dict[str, str]:
    """结果只存简短原因；原始 traceback 由实际错误边界记录。"""
    return dict(stage=stage, exception_type=type(error).__name__, message=str(error))
