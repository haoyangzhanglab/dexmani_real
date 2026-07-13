"""Central logging."""

from __future__ import annotations

__all__ = ["get_logger"]

import logging
import os
import sys
import time
from pathlib import Path

_loggers: dict[str, logging.Logger] = {}

_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Shared file handler (created once per process). None if disabled or creation
# failed — logging then falls back to stdout only.
_file_handler: logging.FileHandler | None = None
_file_handler_init = False


def _get_file_handler() -> logging.FileHandler | None:
    """Create (once) a shared file handler for on-disk session logs.

    Directory from $DEXMANI_LOG_DIR (default ~/.dexmani/logs/), file name is
    date-stamped. Fail-safe: any error → return None (stdout logging unaffected).
    """
    global _file_handler, _file_handler_init
    if _file_handler_init:
        return _file_handler
    _file_handler_init = True
    try:
        log_dir = Path(os.environ.get("DEXMANI_LOG_DIR", str(Path.home() / ".dexmani" / "logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"dexmani_{time.strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setFormatter(_FORMATTER)
        _file_handler = handler
    except OSError:
        _file_handler = None  # read-only FS etc. — keep stdout only
    return _file_handler


def get_logger(name: str) -> logging.Logger:
    if name not in _loggers:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_FORMATTER)
            logger.addHandler(handler)
            file_handler = _get_file_handler()
            if file_handler is not None:
                logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
        _loggers[name] = logger
    return _loggers[name]
