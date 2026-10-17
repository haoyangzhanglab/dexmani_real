from __future__ import annotations

__all__ = [
    "CapturedProcessOutput",
    "capture_native_stdout",
    "extract_native_diagnostics",
    "configure_logging",
    "ThrottledWarner",
]

import ctypes
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
_CONSOLE_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)


def configure_logging() -> None:
    """Configure console and date/PID file logging once, only at an explicit entry."""
    logger = logging.getLogger("dexmani_real")
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_CONSOLE_FORMATTER)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        log_dir = Path(os.environ.get("DEXMANI_LOG_DIR", str(Path.home() / ".dexmani" / "logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (f"dexmani_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log")
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setFormatter(_FORMATTER)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    except OSError:
        # Console logging remains available when the log directory cannot be written.
        pass


@dataclass
class CapturedProcessOutput:
    text: str = ""


def _flush_process_stdout() -> None:
    """Flush Python and C stdio before changing/restoring file descriptor 1."""
    try:
        sys.stdout.flush()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError):
        pass


@contextmanager
def capture_native_stdout() -> Iterator[CapturedProcessOutput]:
    """Capture process-wide Python/C/C++ stdout during initialization only; text is available after exit."""
    captured = CapturedProcessOutput()
    try:
        stdout_fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield captured
        return

    saved_fd: int | None = None
    sink = None
    try:
        saved_fd = os.dup(stdout_fd)
        sink = tempfile.TemporaryFile(mode="w+b")
        _flush_process_stdout()
        os.dup2(sink.fileno(), stdout_fd)
    except OSError:
        if sink is not None:
            sink.close()
        if saved_fd is not None:
            os.close(saved_fd)
        yield captured
        return

    try:
        yield captured
    finally:
        _flush_process_stdout()
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)
        sink.seek(0)
        captured.text = sink.read().decode("utf-8", errors="replace").strip()
        sink.close()


def extract_native_diagnostics(text: str, *, ignore: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return suspicious vendor-output lines while dropping known chatter."""
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


class ThrottledWarner:
    """Throttle each solver warning to once every five seconds."""

    def __init__(self, *, logger: logging.Logger) -> None:
        self._last_ns = 0
        self._logger = logger

    def __call__(self, msg: str, *args: Any, **kwargs: Any) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_ns < 5_000_000_000:
            return
        self._last_ns = now_ns
        self._logger.warning(msg, *args, **kwargs)
