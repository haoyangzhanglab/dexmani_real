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


def error_detail(stage, error):
    message = str(error)
    for cleanup_stage, detail in getattr(error, "cleanup_errors", ()):
        detail_message = error_detail(cleanup_stage, detail)["message"]
        message += f"; {cleanup_stage}: {type(detail).__name__}: {detail_message}"
    return dict(stage=stage, exception_type=type(error).__name__, message=message)
