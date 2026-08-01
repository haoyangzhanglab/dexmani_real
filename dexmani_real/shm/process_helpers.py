"""Shared helpers for forked child processes (arm/hand).

Extracted from arm_process.py and hand_process.py — both children share the
same SIGINT/SIGTERM handler pattern and ring-close cleanup pattern.
"""

from __future__ import annotations

import signal
import threading
from typing import Any


def install_sigint_handler() -> threading.Event:
    """Install SIGINT/SIGTERM handlers that set a ``threading.Event``.

    Returns the event so the caller can check ``event.is_set()`` to detect
    a shutdown request.  A ``threading.Event`` (RLock-based) is signal-handler
    safe — the mp semaphore behind ``multiprocessing.Event`` is NOT reentrant
    when the handler interrupts a blocked ``is_set()``/``wait()`` on the same
    thread.

    Safe to call from a forked child process.  ``(ValueError, OSError)`` is
    caught and suppressed — ``signal.signal`` fails when not in the main
    thread, which should never happen for the fork child.
    """
    sigint_received = threading.Event()

    try:
        signal.signal(signal.SIGINT, lambda *_: sigint_received.set())
        signal.signal(signal.SIGTERM, lambda *_: sigint_received.set())
    except (ValueError, OSError):
        pass  # not in the main thread

    return sigint_received


def close_rings(*rings: Any) -> None:
    """Safely close ring buffer handles, ignoring errors.

    Each ring is closed exactly once.  ``None`` entries are silently skipped.
    Exceptions during close are caught — a failed close in a ``finally`` block
    must not mask the original exception.
    """
    for ring in rings:
        if ring is None:
            continue
        try:
            ring.close()
        except Exception:
            pass
