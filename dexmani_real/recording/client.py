"""Controller-owned recording decisions, sample publication and Queue results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Empty, Full
from typing import Any, NoReturn

import numpy as np

from dexmani_real.recording.frame import EpisodeFrame
from dexmani_real.runtime.safety import RunEndReason, SafetyState, _revoke_motion_locked

RECORDER_STOP_TIMEOUT_S = 60.0
RECORDER_START_TIMEOUT_S = 10.0
_STOP_POLL_INTERVAL_S = 0.01


@dataclass
class StartRecording:
    task: str
    operator: str
    start_sequence: int
    episode_name: str | None = None


@dataclass(frozen=True)
class StopRecording:
    save: bool
    reason: str
    through_sequence: int
    technical_status: str = "valid"
    had_pause: bool = False


@dataclass
class RecordingStarted:
    path: str


@dataclass(frozen=True)
class RecordingResult:
    """One recorder outcome shared by the queue and its sole consumer."""

    saved: bool = False
    error: str | None = None
    path: str | None = None
    frame_count: int = 0
    reason: str = ""
    min_frames_met: bool = False


class RecorderClient:
    """Sole result-queue consumer; at most one recording is in flight."""

    def __init__(self, shared: Any) -> None:
        self.shared = shared
        self.technical_status = "valid"
        self.had_pause = False
        self._frame_count = 0
        self._recording = False
        self._stop_requested = False
        self._stop_reason = ""
        self._last_stop_result: RecordingResult | None = None
        self.episode_path: str | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def stop_pending(self) -> bool:
        return self._stop_requested

    def _fail_recording(self, error: str) -> NoReturn:
        """Stop motion when the required recording resource fails."""
        with self.shared.motion_lock:
            self.shared.is_running.value = False
            if int(self.shared.safety_state.value) == int(SafetyState.RUNNING):
                _revoke_motion_locked(
                    self.shared, SafetyState.ARMED, reason=RunEndReason.RECORDING_FAILURE
                )
            self.technical_status = "invalid"
        raise RuntimeError(f"Recording failed: {error}")

    def _fail_transport(self, error: str) -> NoReturn:
        self._recording = False
        self._stop_requested = False
        self._last_stop_result = RecordingResult(
            error=error,
            reason=self._stop_reason,
            path=self.episode_path,
            frame_count=self._frame_count,
        )

        self._fail_recording(error)

    def _send_control(self, message: StartRecording | StopRecording) -> None:
        try:
            self.shared.record_control_q.put_nowait(message)
        except (Full, EOFError, OSError, ValueError) as exc:
            self._fail_transport(f"control queue failed: {exc}")

    def start_episode(
        self,
        *,
        task_label: str = "",
        operator: str = "",
        episode_name: str | None = None,
    ) -> bool:
        """Request one recording; ``episode_name=None`` keeps timestamp naming.

        An explicit ``episode_name`` selects the exact published directory and
        is refused loudly by the recorder when it already exists.
        """
        if self._recording or self._stop_requested:
            return False
        if not self.shared.recorder_ready.is_set():
            self._fail_transport("recorder unavailable during START")
        self.technical_status = "valid"
        self.had_pause = False
        self.episode_path = None
        self._frame_count = 0
        self._last_stop_result = None
        self._stop_reason = ""
        start = StartRecording(
            task_label,
            operator,
            int(self.shared.record_sample_ring.latest_sequence) + 1,
            episode_name,
        )
        self._send_control(start)
        deadline = time.monotonic() + RECORDER_START_TIMEOUT_S
        while time.monotonic() < deadline and self.shared.is_running.value:
            if not self.shared.recorder_ready.is_set():
                self._fail_transport("recorder died during START")
            try:
                result = self.shared.record_result_q.get(timeout=_STOP_POLL_INTERVAL_S)
            except Empty:
                continue
            except (EOFError, OSError, ValueError) as exc:
                self._fail_transport(f"start result queue failed: {exc}")
            if isinstance(result, RecordingStarted):
                self.episode_path = result.path
                self._recording = True
                return True
            if isinstance(result, RecordingResult):
                self._finish(result)
                self._fail_transport("recorder ended before START completed")
            self._fail_transport("unexpected start result")
        self._fail_transport("recorder START timed out or runtime stopped")

    def add_frame(self, sample: EpisodeFrame) -> None:
        if not self._recording:
            return
        if not self.shared.recorder_ready.is_set():
            self._fail_transport("recorder unavailable during sample submission")
        try:
            dtype = self.shared.record_sample_ring.dtype
            frame = np.zeros(1, dtype=dtype)
            frame["timestamp"][0] = sample.timestamp_s
            for name, value in sample.data.items():
                frame[name][0] = value
            if sample.camera_rgb is None or sample.camera_depth is None:
                raise ValueError("recording requires RGB and depth for every row")
            frame["camera_present"][0] = 1
            frame["camera_rgb"][0] = sample.camera_rgb
            frame["camera_depth"][0] = sample.camera_depth
            self.shared.record_sample_ring.write(frame)
            self._frame_count += 1
        except Exception as exc:
            self._fail_recording(f"sample submission failed: {exc}")
        if not self.shared.recorder_ready.is_set():
            self._fail_transport("recorder lost during sample submission")

    def stop_episode(self, save: bool = True, reason: str = "") -> None:
        if not self._recording or self._stop_requested:
            return
        # Stop production before capturing the final committed sequence.
        self._recording = False
        self._stop_requested = True
        self._stop_reason = reason or "manual"
        self._send_control(
            StopRecording(
                save=save,
                reason=self._stop_reason,
                through_sequence=int(self.shared.record_sample_ring.latest_sequence),
                technical_status=self.technical_status,
                had_pause=self.had_pause,
            )
        )

    def _finish(self, result: RecordingResult) -> RecordingResult:
        self._recording = False
        self._stop_requested = False
        self._last_stop_result = result
        if result.error:
            self._fail_recording(result.error)
        return result

    def poll_stop(self) -> RecordingResult | None:
        if self._last_stop_result is not None:
            return self._finish(self._last_stop_result)
        if not self.shared.recorder_ready.is_set():
            self._fail_transport("recorder transport unavailable")
        try:
            event = self.shared.record_result_q.get_nowait()
        except Empty:
            return None
        except (EOFError, OSError, ValueError) as exc:
            self._fail_transport(f"result queue failed: {exc}")
        if not isinstance(event, RecordingResult):
            self._fail_transport("unexpected recording result")
        return self._finish(event)

    def join_stop(self, timeout: float | None = None) -> RecordingResult:
        timeout_s = RECORDER_STOP_TIMEOUT_S if timeout is None else float(timeout)
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("recorder stop timeout must be finite and non-negative")
        if not self._stop_requested:
            return (
                self._finish(self._last_stop_result)
                if self._last_stop_result
                else RecordingResult()
            )
        deadline = time.monotonic() + timeout_s
        while True:
            result = self.poll_stop()
            if result is not None:
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_STOP_POLL_INTERVAL_S, remaining))
        self._fail_transport("recorder result wait timed out")
