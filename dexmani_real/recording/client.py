"""Controller-owned recording decisions, sample publication and Queue results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Empty, Full
from typing import Any

import numpy as np

from dexmani_real.recording.frame import EpisodeFrame
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)
RECORDER_STOP_TIMEOUT_S = 60.0
RECORDER_START_TIMEOUT_S = 10.0
_STOP_POLL_INTERVAL_S = 0.01


@dataclass
class StartRecording:
    task: str
    operator: str
    start_sequence: int
    episode_name: str | None = None


@dataclass
class StopRecording:
    save: bool
    reason: str
    through_sequence: int
    retain_partial: bool = False


@dataclass
class RecordingStarted:
    path: str
    max_frames: int
    max_frames_stop_reason: str


@dataclass
class RecordingFinished:
    saved: bool
    path: str | None
    frame_count: int
    reason: str
    error: str | None = None
    min_frames_met: bool = False


@dataclass
class RecorderStopResult:
    """Local result, consumed once by the controller."""

    done: bool
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
        self._frame_count = 0
        self._recording = False
        self._stop_requested = False
        self._unavailable = False
        self._max_frames = 0
        self._max_frames_stop_reason = "max_frames"
        self._stop_reason = ""
        self._last_stop_result: RecorderStopResult | None = None
        # A terminal verdict is delivered to its sole consumer exactly once.
        self._terminal_result_delivered = False
        self.episode_path: str | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def transport_unavailable(self) -> bool:
        """Whether transport integrity was lost; the workflow owns disposition."""
        return self._unavailable

    @property
    def next_frame_reaches_limit(self) -> bool:
        """Whether a successful append now would trigger max-frames auto-save."""
        return bool(
            self._recording
            and self._max_frames > 0
            and self._frame_count + 1 >= self._max_frames
        )

    @property
    def stop_pending(self) -> bool:
        return self._stop_requested

    @property
    def camera_writer_error(self) -> str | None:
        return self._last_stop_result.error if self._last_stop_result else None

    @property
    def last_error(self) -> str | None:
        """Terminal error text of the most recent recording result, if any."""
        return self._last_stop_result.error if self._last_stop_result else None

    def _fail_transport(self, error: str) -> None:
        # Report local unavailability; the workflow decides its disposition.
        logger.error("RecorderIO unavailable: %s", error)
        self._unavailable = True
        self._recording = False
        self._last_stop_result = RecorderStopResult(
            done=False,
            error=error,
            reason=self._stop_reason,
            path=self.episode_path,
            frame_count=self._frame_count,
        )
        self._terminal_result_delivered = False

    def _send_control(self, message: StartRecording | StopRecording) -> bool:
        try:
            self.shared.record_control_q.put_nowait(message)
            return True
        except (Full, OSError, ValueError) as exc:
            self._fail_transport(f"control queue failed: {exc}")
            return False

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
        if (
            self._recording
            or self._stop_requested
            or self._unavailable
            or self.shared.evidence_failed.value
            or not self.shared.is_ready("recorder")
        ):
            return False
        self.episode_path = None
        self._last_stop_result = None
        self._stop_reason = ""
        self._terminal_result_delivered = False
        start = StartRecording(
            task_label,
            operator,
            int(self.shared.record_sample_ring.latest_sequence) + 1,
            episode_name,
        )
        if not self._send_control(start):
            return False
        deadline = time.monotonic() + RECORDER_START_TIMEOUT_S
        while time.monotonic() < deadline and self.shared.is_running.value:
            self.shared.set_heartbeat("policy", time.monotonic())
            try:
                result = self.shared.record_result_q.get(timeout=_STOP_POLL_INTERVAL_S)
            except Empty:
                continue
            except (EOFError, OSError, ValueError) as exc:
                self._fail_transport(f"start result queue failed: {exc}")
                return False
            if isinstance(result, RecordingStarted):
                self.episode_path = result.path
                self._max_frames = result.max_frames
                self._max_frames_stop_reason = result.max_frames_stop_reason
                self._frame_count = 0
                self._recording = True
                return True
            if isinstance(result, RecordingFinished):
                self._finish(result)
                return False
            self._fail_transport("unexpected start result")
            return False
        # Supervisor owns failed-worker shutdown, not a cancellation protocol.
        self._fail_transport(
            "recorder start acknowledgement timed out or runtime stopped"
        )
        return False

    def add_frame(self, sample: EpisodeFrame) -> bool:
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
        frame["timestamp"][0] = sample.timestamp_s
        for name, value in sample.data.items():
            frame[name][0] = value
        if sample.camera_rgb is not None or sample.camera_depth is not None:
            frame["camera_present"][0] = 1
            if sample.camera_rgb is not None:
                frame["camera_rgb"][0] = sample.camera_rgb
            if sample.camera_depth is not None:
                frame["camera_depth"][0] = sample.camera_depth
        self.shared.record_sample_ring.write(frame)
        self._frame_count += 1
        if self._max_frames and self._frame_count >= self._max_frames:
            self.stop_episode(save=True, reason=self._max_frames_stop_reason)
        return True

    def stop_episode(
        self, save: bool = True, reason: str = "", *, retain_partial: bool = False
    ) -> str | None:
        """Stop once; interrupted unpublished captures may retain closed staging.

        ``retain_partial`` is control intent, never a raw sample field. An
        already issued STOP keeps its original save/retention decision.
        """
        if not self._recording or self._stop_requested:
            return None
        # Revoke production before capturing the final committed sequence.
        self._recording = False
        self._stop_requested = True
        self._stop_reason = reason or "manual"
        through = int(self.shared.record_sample_ring.latest_sequence)
        self._send_control(StopRecording(save, self._stop_reason, through, retain_partial))
        return None

    def _finish(self, event: RecordingFinished) -> RecorderStopResult:
        self._recording = False
        self._stop_requested = False
        result = RecorderStopResult(
            done=True,
            saved=event.saved,
            error=event.error,
            path=event.path,
            frame_count=event.frame_count,
            reason=event.reason,
            min_frames_met=event.min_frames_met,
        )
        self._last_stop_result = result
        self._terminal_result_delivered = True
        return result

    def poll_stop(self) -> RecorderStopResult:
        try:
            event = self.shared.record_result_q.get_nowait()
        except Empty:
            if self._unavailable and self._last_stop_result:
                # A transport-lost client still reports its last known state,
                # but a terminal (done) verdict is delivered exactly once:
                # repeating it would make the sole consumer re-consume the same
                # outcome (double-counted saves, repeated result lines).
                if self._last_stop_result.done and self._terminal_result_delivered:
                    return RecorderStopResult(done=False, reason=self._stop_reason)
                self._terminal_result_delivered = self._last_stop_result.done
                return self._last_stop_result
            return RecorderStopResult(done=False, reason=self._stop_reason)
        except (EOFError, OSError, ValueError) as exc:
            self._fail_transport(f"result queue failed: {exc}")
            return self._last_stop_result
        if not isinstance(event, RecordingFinished):
            self._fail_transport("unexpected recording result")
            return self._last_stop_result
        return self._finish(event)

    def join_stop(self, timeout: float | None = None) -> RecorderStopResult:
        if not self._stop_requested:
            if self._last_stop_result is None:
                return RecorderStopResult(done=True)
            if self._last_stop_result.done and self._terminal_result_delivered:
                # The polling path already consumed this terminal verdict;
                # handing it out again would let the owner re-run its
                # completion bookkeeping for a different trial.
                return RecorderStopResult(done=True)
            self._terminal_result_delivered = self._last_stop_result.done
            return self._last_stop_result
        timeout_s = RECORDER_STOP_TIMEOUT_S if timeout is None else float(timeout)
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("recorder stop timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = self.poll_stop()
            if result.done or result.error:
                return result
            self.shared.set_heartbeat("policy", time.monotonic())
            time.sleep(_STOP_POLL_INTERVAL_S)
        self._fail_transport("recorder finalization timed out")
        return self._last_stop_result
