from __future__ import annotations

import ctypes
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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


@dataclass
class CapturedProcessOutput:
    text: str = ""


def flush_process_stdout() -> None:
    sys.stdout.flush()
    ctypes.CDLL(None).fflush(None)


@contextmanager
def capture_native_stdout() -> Iterator[CapturedProcessOutput]:
    """仅在初始化阶段捕获进程级 Python/C 输出，退出后读取 text。"""
    captured = CapturedProcessOutput()
    stdout_fd = sys.stdout.fileno()
    with tempfile.TemporaryFile(mode="w+b") as sink:
        flush_process_stdout()
        saved_fd = os.dup(stdout_fd)
        try:
            os.dup2(sink.fileno(), stdout_fd)
            yield captured
        finally:
            try:
                flush_process_stdout()
            finally:
                try:
                    os.dup2(saved_fd, stdout_fd)
                finally:
                    os.close(saved_fd)
                sink.seek(0)
                captured.text = sink.read().decode("utf-8", errors="replace").strip()


def extract_native_diagnostics(text: str, *, ignore: tuple[str, ...] = ()) -> tuple[str, ...]:
    markers = (
        "error",
        "fail",
        "exception",
        "traceback",
        "permission",
        "unknown",
        "unknow",
        "compare_time",
    )
    return tuple(
        line
        for line in (value.strip() for value in text.splitlines())
        if line
        and not any(token in line for token in ignore)
        and any(marker in line.lower() for marker in markers)
    )
