from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
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
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        log_dir = Path(os.environ.get("DEXMANI_LOG_DIR", str(Path.home() / ".dexmani" / "logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (f"dexmani_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(FORMATTER)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Cannot create file log: %s", exc)
