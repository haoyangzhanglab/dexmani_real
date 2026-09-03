"""Central logging."""

from __future__ import annotations

__all__ = [
    "CapturedProcessOutput",
    "capture_native_stdout",
    "extract_native_diagnostics",
    "get_logger",
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

# Shared file handler (created once per process). None if disabled or creation
# failed — logging then falls back to stdout only.
_file_handler: logging.FileHandler | None = None
_file_handler_init = False


def _get_file_handler() -> logging.FileHandler | None:
    """Create (once) a shared file handler for on-disk session logs.

    Directory from $DEXMANI_LOG_DIR (default ~/.dexmani/logs/), file name is
    date/PID-stamped. Fail-safe: any error → return None (stdout unaffected).
    """
    global _file_handler, _file_handler_init
    if _file_handler_init:
        return _file_handler
    _file_handler_init = True
    try:
        log_dir = Path(
            os.environ.get("DEXMANI_LOG_DIR", str(Path.home() / ".dexmani" / "logs"))
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            f"dexmani_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
        )
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setFormatter(_FORMATTER)
        handler.setLevel(logging.DEBUG)
        _file_handler = handler
    except OSError:
        _file_handler = None  # read-only FS etc. — keep stdout only
    return _file_handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_FORMATTER)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        file_handler = _get_file_handler()
        if file_handler is not None:
            logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


# ThrottledWarner uses the project logger format for consistent diagnostics.
_logger = get_logger(__name__)


@dataclass
class CapturedProcessOutput:
    """Text emitted to process stdout inside ``capture_native_stdout``."""

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
    """Capture Python/C/C++ stdout for one bounded vendor-SDK operation.

    The redirection is process-wide, so callers must use it only during device
    initialization, never inside a live control loop.  The captured text is
    made available after the context exits so successful SDK chatter can be
    discarded while a failed operation can still replay its diagnostics.
    """
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

    assert saved_fd is not None and sink is not None
    try:
        yield captured
    finally:
        _flush_process_stdout()
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)
        sink.seek(0)
        captured.text = sink.read().decode("utf-8", errors="replace").strip()
        sink.close()


def extract_native_diagnostics(
    text: str, *, ignore: tuple[str, ...] = ()
) -> tuple[str, ...]:
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
    """Callable that forwards to ``logger.warning`` at most once per *interval_s*.

    Used in hot-path loops to avoid log spam from per-tick conditions
    (torn reads, stale state, producer mismatch).  Default interval: 5.0 s.
    """

    def __init__(
        self, interval_s: float = 5.0, logger: logging.Logger | None = None
    ) -> None:
        self._interval_ns = int(interval_s * 1e9)
        self._last_ns = 0
        self._logger = logger or _logger

    def __call__(self, msg: str, *args: Any, **kwargs: Any) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_ns < self._interval_ns:
            return
        self._last_ns = now_ns
        self._logger.warning(msg, *args, **kwargs)
