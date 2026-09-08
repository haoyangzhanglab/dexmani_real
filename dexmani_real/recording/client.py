"""Controller-side recorder client and shared control-plane protocol types.

``RecorderClient`` is the policy-side owner of recording decisions and fixed
sample construction.  This module also holds the control-plane types shared
with the RecorderIO process (``recorder_io_loop`` in ``io_process.py``), which
imports them from here.  This module never imports ``io_process``, keeping the
dependency one-way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from dexmani_real.ipc.schema import (
    RECORD_CONTROL_DTYPE,
    RECORD_OPERATOR_BYTES,
    RECORD_STOP_REASON_BYTES,
    RECORD_TASK_LABEL_BYTES,
)
from dexmani_real.recording.frame import episode_source_values
from dexmani_real.recording.sample import EpisodeAction, EpisodeState
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

RECORDER_STOP_TIMEOUT_S = 60.0
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_START_CANCEL_REASON = "recorder_start_timeout"
_STOP_POLL_INTERVAL_S = 0.01


class RecorderCommand(IntEnum):
    START = 1
    STOP = 2


class RecorderPhase(IntEnum):
    READY = 1
    RECORDING = 2
    FINALIZING = 3
    COMPLETED = 4
    ERROR = 5
    STOPPED = 6


@dataclass(frozen=True)
class RecorderStopResult:
    """Policy-visible outcome of one recorder transaction."""

    done: bool
    phase: RecorderPhase | None = None
    generation: int = 0
    saved: bool = False
    error: str | None = None
    path: str | None = None
    frame_count: int = 0
    reason: str = ""
    min_frames_met: bool = False
    failure_count: int = 0


def bounded_control_text(value: str, *, capacity: int, field: str) -> bytes:
    """Encode a control-plane text field without an unbounded JSON side channel."""
    payload = value.encode("utf-8")
    if len(payload) > capacity:
        raise ValueError(f"RecorderIO {field} exceeds fixed capacity {capacity}")
    return payload


class RecorderClient:
    """Policy-side owner of recording decisions and fixed sample construction."""

    def __init__(self, shared: Any) -> None:
        self.shared = shared
        self._generation = 0
        self._frame_count = 0
        self._recording = False
        self._start_pending = False
        self._stop_requested = False
        self._last_poll_status_sequence = 0
        self._last_stop_result: RecorderStopResult | None = None
        self.episode_path: str | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def start_pending(self) -> bool:
        """Whether a START generation awaits acknowledgement or cancellation."""
        return self._start_pending

    @property
    def stop_pending(self) -> bool:
        return self._stop_requested

    @property
    def camera_writer_error(self) -> str | None:
        status = self._read_status()
        if status is None or int(status["generation"]) != self._generation:
            return None
        return (
            self._text(status, "error")
            if int(status["phase"]) == int(RecorderPhase.ERROR)
            else None
        )

    def _read_status_with_sequence(self) -> tuple[np.void, int] | None:
        result = self.shared.record_status_ring.read_latest()
        return (result[0][0], int(result[2])) if result is not None else None

    def _read_status(self) -> np.void | None:
        result = self._read_status_with_sequence()
        return result[0] if result is not None else None

    @staticmethod
    def _text(status: np.void, field: str) -> str:
        length = int(status[f"{field}_length"])
        return bytes(status[field])[:length].decode("utf-8", errors="replace")

    def _write_control(
        self,
        command: RecorderCommand,
        *,
        save: bool = False,
        task_label: str = "",
        operator: str = "",
        stop_reason: str = "",
    ) -> None:
        frame = np.zeros(1, dtype=RECORD_CONTROL_DTYPE)
        frame["command"][0] = int(command)
        frame["generation"][0] = self._generation
        frame["save"][0] = int(save)
        frame["created_monotonic_ns"][0] = time.monotonic_ns()
        frame["task_label"][0] = bounded_control_text(
            task_label, capacity=RECORD_TASK_LABEL_BYTES, field="task_label"
        )
        frame["operator"][0] = bounded_control_text(
            operator, capacity=RECORD_OPERATOR_BYTES, field="operator"
        )
        frame["stop_reason"][0] = bounded_control_text(
            stop_reason, capacity=RECORD_STOP_REASON_BYTES, field="stop_reason"
        )
        self.shared.record_control_ring.write(frame)

    def start_episode(self, *, task_label: str = "", operator: str = "") -> bool:
        if (
            self._recording
            or self._start_pending
            or self._stop_requested
            or not self.shared.is_ready("recorder")
        ):
            return False
        try:
            bounded_control_text(
                task_label, capacity=RECORD_TASK_LABEL_BYTES, field="task_label"
            )
            bounded_control_text(
                operator, capacity=RECORD_OPERATOR_BYTES, field="operator"
            )
        except ValueError:
            logger.error(
                "RecorderIO start metadata exceeds its fixed control boundary",
                exc_info=True,
            )
            return False
        self._generation += 1
        self.episode_path = None
        self._last_stop_result = None
        self._start_pending = True
        self._write_control(
            RecorderCommand.START, task_label=task_label, operator=operator
        )
        deadline = time.monotonic() + RECORDER_START_TIMEOUT_S
        while time.monotonic() < deadline and self.shared.is_running.value:
            status = self._read_status()
            if status is not None and int(status["generation"]) == self._generation:
                phase = RecorderPhase(int(status["phase"]))
                if phase is RecorderPhase.RECORDING:
                    self.episode_path = self._text(status, "path") or None
                    self._recording = True
                    self._start_pending = False
                    self._stop_requested = False
                    self._frame_count = 0
                    return True
                if phase is RecorderPhase.ERROR:
                    self._start_pending = False
                    return False
            # Starting is bounded but may span more than one supervisor tick.
            # RecorderClient is policy-owned, so keep that owner's heartbeat live.
            self.shared.set_heartbeat("policy", time.monotonic())
            time.sleep(0.005)
        self.cancel_pending_start()
        return False

    def cancel_pending_start(
        self,
        *,
        reason: str = RECORDER_START_CANCEL_REASON,
    ) -> bool:
        """Cancel an unacknowledged START without allowing a late recording.

        The control ring is latest-only.  A recorder that has not yet observed
        START may therefore see only this STOP; RecorderIO recognizes this
        bounded cancel reason and publishes a terminal, unsaved transaction.
        """
        if not self._start_pending:
            return False
        self._write_control(RecorderCommand.STOP, save=False, stop_reason=reason)
        self._start_pending = False
        self._stop_requested = True
        return True

    def add_frame(
        self,
        state: EpisodeState,
        action: EpisodeAction,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None = None,
        signals: dict[str, Any] | None = None,
        arm_qpos_sent: np.ndarray | None = None,
    ) -> bool:
        if not self._recording:
            return False
        latest = int(self.shared.record_sample_ring.latest_sequence)
        consumed = int(self.shared.recorder_consumed_sequence.value)
        if latest - consumed >= self.shared.record_sample_ring.maxlen:
            logger.error("RecorderIO sample ring overflow — aborting episode")
            self.stop_episode(save=False, reason="sample_ring_overflow")
            return False

        dtype = self.shared.record_sample_ring.dtype
        frame = np.zeros(1, dtype=dtype)
        frame["generation"][0] = self._generation
        frame["sample_sequence"][0] = self._frame_count + 1
        frame["timestamp"][0] = state.timestamp
        for name, value in episode_source_values(
            state,
            action,
            vr_frame,
            camera_frame=camera_frame,
            signals=signals,
            arm_qpos_sent=arm_qpos_sent,
        ).items():
            frame[name][0] = value
        if camera_frame is not None:
            frame["camera_present"][0] = 1
            frame["camera_rgb"][0] = camera_frame.get(
                "rgb", np.zeros(frame["camera_rgb"][0].shape, np.uint8)
            )
            frame["camera_depth"][0] = camera_frame.get(
                "depth", np.zeros(frame["camera_depth"][0].shape, np.uint16)
            )
        self.shared.record_sample_ring.write(frame)
        self._frame_count += 1
        return True

    def stop_episode(self, save: bool = True, reason: str = "") -> str | None:
        if self._start_pending:
            # There is no recorder transaction to save yet; use the one
            # protocol reason that RecorderIO accepts before START.
            self.cancel_pending_start()
            return None
        if not self._recording or self._stop_requested:
            return None
        self._write_control(RecorderCommand.STOP, save=save, stop_reason=reason)
        self._recording = False
        self._stop_requested = True
        return None

    def _status_result(self, status: np.void, *, done: bool) -> RecorderStopResult:
        phase = RecorderPhase(int(status["phase"]))
        error = self._text(status, "error") or None
        path = self._text(status, "path") or None
        return RecorderStopResult(
            done=done,
            phase=phase,
            generation=int(status["generation"]),
            saved=phase is RecorderPhase.COMPLETED
            and error is None
            and bool(status["saved"]),
            error=error,
            path=path,
            frame_count=int(status["frame_count"]),
            reason=self._text(status, "reason"),
            min_frames_met=bool(status["min_frames_met"]),
            failure_count=int(status["failure_count"]),
        )

    def poll_stop(self) -> RecorderStopResult:
        """Return each newly published recorder status once, including max-stop."""
        status_result = self._read_status_with_sequence()
        if status_result is None:
            return RecorderStopResult(done=False)
        status, sequence = status_result
        if (
            sequence == self._last_poll_status_sequence
            or int(status["generation"]) != self._generation
        ):
            return RecorderStopResult(done=False)
        self._last_poll_status_sequence = sequence
        phase = RecorderPhase(int(status["phase"]))
        if phase is RecorderPhase.FINALIZING:
            self._recording = False
            self._start_pending = False
            self._stop_requested = True
            return self._status_result(status, done=False)
        if phase not in (RecorderPhase.COMPLETED, RecorderPhase.ERROR):
            return RecorderStopResult(
                done=False, phase=phase, generation=self._generation
            )
        self._recording = False
        self._start_pending = False
        self._stop_requested = False
        result = self._status_result(status, done=True)
        self._last_stop_result = result
        return result

    def join_stop(self, timeout: float | None = None) -> RecorderStopResult:
        """Wait for a terminal status without conflating ERROR with success."""
        if not self._stop_requested:
            return self._last_stop_result or RecorderStopResult(done=True)
        timeout_s = RECORDER_STOP_TIMEOUT_S if timeout is None else float(timeout)
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("recorder stop timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self._read_status()
            if status is not None and int(status["generation"]) == self._generation:
                phase = RecorderPhase(int(status["phase"]))
                if phase in (RecorderPhase.COMPLETED, RecorderPhase.ERROR):
                    self._recording = False
                    self._start_pending = False
                    self._stop_requested = False
                    result = self._status_result(status, done=True)
                    self._last_stop_result = result
                    return result
            self.shared.set_heartbeat("policy", time.monotonic())
            time.sleep(_STOP_POLL_INTERVAL_S)
        return RecorderStopResult(
            done=False,
            phase=RecorderPhase.FINALIZING,
            generation=self._generation,
            reason="finalization_timeout",
        )
