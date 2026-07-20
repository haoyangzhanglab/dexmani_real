"""Thin request/response RPC over two SeqlockRingBuffers (cmd + result).

Commands are correlated with results via the cmd ring's sequence number: the
client writes a request and polls the result ring for a record whose
``cmd_seq`` matches the sequence returned by ``write()``. The server
dispatches only commands newer than the last one it served, and converts
handler exceptions into ``ok=0`` results instead of crashing the control child.

No hardware imports — safe to import from both main and child processes.

Ref: docs/arm-hand-process-isolation-plan.md §4.3 (arm_cmd / arm_cmd_result).

Usage:
    # Main process (client)
    client = RpcClient(cmd_ring, result_ring, timeout_s=10.0)
    result = client.call(cmd_frame)        # e.g. ARM_CMD_DTYPE -> ARM_CMD_RESULT_DTYPE

    # Control child (server), once per tick
    server = RpcServer(cmd_ring, result_ring, handler=dispatch_macro)
    while running:
        server.handle_pending()            # non-blocking
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:  # robot_ring may not exist yet; typing-only to stay importable
    from dexmani_real.shm.robot_ring import SeqlockRingBuffer

logger = get_logger(__name__)


class RpcTimeoutError(Exception):
    """Raised when the result ring does not acknowledge a command in time."""


class RpcClient:
    """Issues commands on the cmd ring and blocks until the matching result.

    Correlation is by sequence number: ``call()`` writes the request, then
    polls ``result_ring.read_latest()`` until ``result["cmd_seq"]`` equals the
    sequence returned by ``cmd_ring.write()``.
    """

    def __init__(
        self,
        cmd_ring: "SeqlockRingBuffer",
        result_ring: "SeqlockRingBuffer",
        timeout_s: float = 5.0,
        poll_s: float = 0.005,
    ) -> None:
        self._cmd_ring = cmd_ring
        self._result_ring = result_ring
        self._timeout_s = timeout_s
        self._poll_s = poll_s

    def call(self, request: np.ndarray) -> np.ndarray:
        """Write ``request`` to the cmd ring and block until its result arrives.

        Returns a copy of the matching result record (read_latest already
        returns copies). Raises RpcTimeoutError on a time.monotonic deadline
        expiry; the message includes the request's ``cmd`` code and timeout.
        """
        seq = self._cmd_ring.write(request)
        deadline = time.monotonic() + self._timeout_s
        while True:
            frame = self._result_ring.read_latest()
            if frame is not None:
                data, _ts_ns, _seq = frame
                if int(data["cmd_seq"][0]) == seq:
                    return data
            if time.monotonic() >= deadline:
                raise RpcTimeoutError(
                    f"RPC cmd={self._cmd_code(request)} timed out after "
                    f"{self._timeout_s:.3f}s (no result for cmd seq={seq})"
                )
            time.sleep(self._poll_s)

    @staticmethod
    def _cmd_code(request: np.ndarray) -> int:
        """Best-effort extraction of the ``cmd`` field for error messages."""
        try:
            return int(np.ravel(request["cmd"])[0])
        except Exception:
            return -1


class RpcServer:
    """Dispatches pending commands to a handler and publishes results.

    The handler receives ``(request_record, seq)`` and returns a result record
    with ``cmd_seq`` left zero; the server stamps ``cmd_seq=seq`` before
    writing it to the result ring. Each command sequence is dispatched at most
    once (seqs newer than ``_last_served`` only). Handler exceptions produce
    an ``ok=0`` result (``sdk_ret=-1`` when the result dtype has that field)
    — the server never crashes.
    """

    def __init__(
        self,
        cmd_ring: "SeqlockRingBuffer",
        result_ring: "SeqlockRingBuffer",
        handler: Callable[[np.ndarray, int], np.ndarray],
    ) -> None:
        self._cmd_ring = cmd_ring
        self._result_ring = result_ring
        self._handler = handler
        self._last_served = 0

    def handle_pending(self) -> bool:
        """Dispatch the latest command if newer than the last served one.

        Non-blocking. Returns True if a command was dispatched and its result
        written (or an error result was written on handler failure).
        """
        if self._cmd_ring.latest_sequence <= self._last_served:
            return False
        frame = self._cmd_ring.read_latest()
        if frame is None:
            return False
        request, _ts_ns, seq = frame
        if seq <= self._last_served:
            return False

        try:
            result = self._handler(request, seq)
            result = np.asarray(result, dtype=self._result_ring.dtype)
            if result.ndim == 0:
                result = result.reshape(1)
            result["cmd_seq"] = seq
        except Exception as exc:
            logger.warning("RpcServer handler failed for seq=%d: %s", seq, exc)
            result = np.zeros(1, dtype=self._result_ring.dtype)
            names = result.dtype.names
            if names is not None and "sdk_ret" in names:
                result["sdk_ret"] = -1
            result["cmd_seq"] = seq

        try:
            self._result_ring.write(result)
        except Exception as exc:  # ring failure — client will time out
            logger.warning("RpcServer failed to publish result for seq=%d: %s", seq, exc)

        self._last_served = seq
        return True
