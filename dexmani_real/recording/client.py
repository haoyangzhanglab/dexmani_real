"""Controller-owned recording decisions, sample publication and Queue results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Empty, Full
from typing import Any

import numpy as np

from dexmani_real.recording.frame import EpisodeFrame
from dexmani_real.robot.commands import RobotCommand, read_command_adoption, CommandAdoption
from dexmani_real.runtime.safety import invalidate_coupled_commands
from dexmani_real.recording.storage.schema import (
    FRAME_PARTIAL_ADOPTION, FRAME_ADOPTION_UNKNOWN, validate_command_row,
)
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


@dataclass(frozen=True)
class StopRecording:
    save: bool
    reason: str
    through_sequence: int
    retain_partial: bool = False
    technical_status: str = "valid"
    deadline_monotonic_ns: int = field(default_factory=lambda: time.monotonic_ns() + int(RECORDER_STOP_TIMEOUT_S * 1e9))


@dataclass
class RecordingStarted:
    path: str
    max_frames: int
    max_frames_stop_reason: str


@dataclass
class RecordingResult:
    """One recorder outcome shared by the queue and its sole consumer."""

    done: bool = True
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
        self.adoption_timeout_s = float(shared.adoption_accounting_timeout_s)
        self.pending_command: tuple[RobotCommand, EpisodeFrame] | None = None
        self._known_adoption = CommandAdoption()
        self.last_command_fields: dict[str, int | bool] = {}
        self.last_command_action: dict[str, Any] = {}
        self.technical_status = "valid"
        self._finish_deadline_ns = 0
        self._frame_count = 0
        self._recording = False
        self._stop_requested = False
        self._unavailable = False
        self._max_frames = 0
        self._max_frames_stop_reason = "max_frames"
        self._stop_reason = ""
        self._last_stop_result: RecordingResult | None = None
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
        self.shared.recorder_transport_failed.value = True
        self._recording = False
        self._last_stop_result = RecordingResult(
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
            or self.shared.recorder_transport_failed.value
            or self.shared.evidence_failed.value
            or not self.shared.is_ready("recorder")
        ):
            return False
        self.technical_status = "valid"
        self.last_command_fields = {}
        self.last_command_action = {}
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
            if self.shared.recorder_transport_failed.value:
                self._fail_transport("recorder died during START")
                return False
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
            if isinstance(result, RecordingResult):
                self._finish(result)
                return False
            self._fail_transport("unexpected start result")
            return False
        # Supervisor owns failed-worker shutdown, not a cancellation protocol.
        self._fail_transport(
            "recorder start acknowledgement timed out or runtime stopped"
        )
        return False

    def stage_command(self, command: RobotCommand, sample: EpisodeFrame) -> None:
        if self.pending_command is not None:
            raise RuntimeError("recording already has a pending command")
        if int(self.shared.pending_record_command_id.value) != command.command_id:
            raise RuntimeError("recorded publication omitted accounting barrier")
        self.pending_command = (command, sample)
        self._known_adoption = CommandAdoption()

    def resolve_command(self, *, finish: bool = False) -> bool:
        """Resolve one owned sample before new commands or recorder STOP.

        Boundary waits are bounded; current-run polling is nonblocking. Sensor
        failures do not erase positive adoption evidence already observed.
        """
        pending = self.pending_command
        if pending is None:
            return True
        command, sample = pending
        if finish:
            with self.shared.motion_lock:
                if int(self.shared.run_id.value) == command.run_id:
                    invalidate_coupled_commands(self.shared)
        while True:
            # Order the accounting decision against a concurrent lifecycle fence.
            with self.shared.motion_lock:
                boundary = int(self.shared.pending_record_revoked_ns.value)
                facts = read_command_adoption(self.shared, command, boundary_ns=boundary)
                old = self._known_adoption
                facts = CommandAdoption(
                    facts.arm_adopted or old.arm_adopted,
                    facts.hand_adopted or old.hand_adopted,
                    facts.arm_monotonic_ns or old.arm_monotonic_ns,
                    facts.hand_monotonic_ns or old.hand_monotonic_ns,
                    facts.arm_known, facts.hand_known,
                )
                self._known_adoption = facts
                fresh = facts.arm_known and facts.hand_known
                expired = boundary and time.monotonic_ns() >= boundary + int(self.adoption_timeout_s * 1e9)
                if not boundary and facts.complete(command):
                    outcome = "joint"
                    break
                if boundary and fresh:
                    outcome = ("joint" if facts.complete(command) else
                               "partial" if facts.arm_adopted or facts.hand_adopted else "none")
                    break
                if expired:
                    outcome = "unknown"
                    break
            if not finish:
                return False
            self.shared.set_heartbeat("policy", time.monotonic())
            time.sleep(0.005)
        if self.shared.error_state.value or self.shared.estop_request.value or self.shared.evidence_failed.value:
            self.technical_status = "invalid"
        self.pending_command = None
        fields = facts.fields(command)
        sample.data.update(fields)
        if outcome in {"partial", "unknown"}:
            sample.data["flag_frame_status"] = (FRAME_PARTIAL_ADOPTION if outcome == "partial"
                                                 else FRAME_ADOPTION_UNKNOWN)
            self.technical_status = "invalid"
            self.shared.session_failed.value = True
        if outcome == "joint":
            self.last_command_fields = fields
            self.last_command_action = {k: sample.data[k].copy() for k in
                ("action_arm_joint_sent", "action_hand_joint", "action_arm_ee")}
        try:
            if outcome != "none" and not self.add_frame(sample):
                self.technical_status = "invalid"
                self.shared.evidence_failed.value = True
                self.shared.session_failed.value = True
                write_recording_failure(self.shared, "command sample submission failed")
        except Exception:
            self.technical_status = "invalid"
            self.shared.evidence_failed.value = True
            self.shared.session_failed.value = True
            # Keep the identity marker for the parent's bounded failure sidecar.
            raise
        else:
            with self.shared.motion_lock:
                if int(self.shared.pending_record_command_id.value) == command.command_id:
                    self.shared.pending_record_command_id.value = 0
                    self.shared.pending_record_revoked_ns.value = 0
        return True

    def add_frame(self, sample: EpisodeFrame) -> bool:
        if not self._recording or self.shared.recorder_transport_failed.value:
            return False
        validate_command_row(sample.data)
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
        if not self._stop_requested and self._max_frames and self._frame_count >= self._max_frames:
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
        # Preserve explicit discard/save intent while appending the final row.
        self._stop_requested = True
        self.resolve_command(finish=True)
        # Revoke production before capturing the final committed sequence.
        self._recording = False
        self._stop_requested = True
        self._stop_reason = reason or "manual"
        through = int(self.shared.record_sample_ring.latest_sequence)
        stop = StopRecording(save, self._stop_reason, through, retain_partial, self.technical_status)
        self._finish_deadline_ns = stop.deadline_monotonic_ns
        self._send_control(stop)
        return None

    def _finish(self, event: RecordingResult) -> RecordingResult:
        self._recording = False
        self._stop_requested = False
        self._finish_deadline_ns = 0
        result = event
        self._last_stop_result = result
        self._terminal_result_delivered = True
        return result

    def poll_stop(self) -> RecordingResult:
        deadline = self._finish_deadline_ns or int(self.shared.recorder_finish_deadline_ns.value)
        if (self.shared.recorder_transport_failed.value
                or (deadline > 0 and time.monotonic_ns() >= deadline
                    and not 0 < self.shared.recorder_completed_ns.value < deadline)):
            if not self._unavailable:
                self._fail_transport("recorder transport unavailable or finalization deadline expired")
            # A dead/terminated process may have corrupted a Queue lock. Never read it again.
            return self._last_stop_result or RecordingResult(done=False, error="recorder unavailable")
        try:
            event = self.shared.record_result_q.get_nowait()
        except Empty:
            if self._unavailable and self._last_stop_result:
                # A transport-lost client still reports its last known state,
                # but a terminal (done) verdict is delivered exactly once:
                # repeating it would make the sole consumer re-consume the same
                # outcome (double-counted saves, repeated result lines).
                if self._last_stop_result.done and self._terminal_result_delivered:
                    return RecordingResult(done=False, reason=self._stop_reason)
                self._terminal_result_delivered = self._last_stop_result.done
                return self._last_stop_result
            return RecordingResult(done=False, reason=self._stop_reason)
        except (EOFError, OSError, ValueError) as exc:
            self._fail_transport(f"result queue failed: {exc}")
            return self._last_stop_result
        if not isinstance(event, RecordingResult) or not event.done:
            self._fail_transport("unexpected recording result")
            return self._last_stop_result
        return self._finish(event)

    def join_stop(self, timeout: float | None = None) -> RecordingResult:
        if not self._stop_requested and not (
            self._recording and (self.shared.recorder_finish_deadline_ns.value
                                 or self.shared.recorder_completed_ns.value)
        ):
            if self._last_stop_result is None:
                return RecordingResult(done=True)
            if self._last_stop_result.done and self._terminal_result_delivered:
                # The polling path already consumed this terminal verdict;
                # handing it out again would let the owner re-run its
                # completion bookkeeping for a different episode.
                return RecordingResult(done=True)
            self._terminal_result_delivered = self._last_stop_result.done
            return self._last_stop_result
        timeout_s = RECORDER_STOP_TIMEOUT_S if timeout is None else float(timeout)
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("recorder stop timeout must be finite and non-negative")
        deadline_ns = min(time.monotonic_ns() + int(timeout_s * 1e9),
                          self._finish_deadline_ns or int(self.shared.recorder_finish_deadline_ns.value)
                          or time.monotonic_ns() + int(RECORDER_STOP_TIMEOUT_S * 1e9))
        # Poll once even when the caller arrives after the finish deadline.
        # Completion permits bounded delivery time, never extra finalization time.
        delivery_deadline_ns = time.monotonic_ns() + int(timeout_s * 1e9)
        while True:
            result = self.poll_stop()
            if result.done or result.error:
                return result
            wait_deadline = (delivery_deadline_ns if self.shared.recorder_completed_ns.value
                             else deadline_ns)
            if time.monotonic_ns() >= wait_deadline:
                break
            self.shared.set_heartbeat("policy", time.monotonic())
            time.sleep(_STOP_POLL_INTERVAL_S)
        self._fail_transport("recorder finalization timed out")
        return self._last_stop_result


def write_recording_failure(shared: Any, reason: str) -> None:
    """Parent fallback when the sole recording owner cannot finish its account.

    This sidecar does not reconstruct an observation or consume recorder queues.
    It invalidates the reserved episode even if a late RecorderIO close succeeds.
    """
    import json
    from pathlib import Path
    from dexmani_real.robot.commands import read_robot_command
    from dexmani_real.utils.atomic_io import atomic_json_dump

    raw_path = shared.record_episode_path.value
    if not raw_path:
        return
    path = Path(raw_path.decode())
    result_path = path.with_name(path.name + ".result.json")
    previous = {}
    if result_path.is_file():
        try:
            previous = json.loads(result_path.read_text())
        except (OSError, ValueError):
            logger.warning("could not read previous recording diagnostic", exc_info=True)
    payload = dict(technical_status="invalid", termination_reason=reason, task_success="unknown")
    if shared.pending_record_command_id.value:
        command = read_robot_command(shared)
        if command is not None and command.command_id == int(shared.pending_record_command_id.value):
            boundary = int(shared.pending_record_revoked_ns.value)
            facts = read_command_adoption(shared, command, boundary_ns=boundary)
            payload.update(facts.fields(command))
            payload["adoption_accounting"] = ("recording_incomplete" if facts.arm_known and facts.hand_known
                                               else "ADOPTION_UNKNOWN")
            payload["revoked_monotonic_ns"] = boundary
            payload["arm_qpos"] = None if command.arm_qpos is None else command.arm_qpos.tolist()
            payload["hand_qpos"] = None if command.hand_qpos is None else command.hand_qpos.tolist()
    # A later shutdown without a pending marker must retain already-known facts.
    if previous.get("command_id") == payload.get("command_id"):
        for name in ("arm", "hand"):
            if previous.get(f"{name}_command_adopted"):
                payload[f"{name}_command_adopted"] = True
                payload[f"{name}_command_adopted_monotonic_ns"] = previous[f"{name}_command_adopted_monotonic_ns"]
    try:
        atomic_json_dump(previous | payload, result_path)
    except OSError:
        shared.session_failed.value = True
        logger.exception("could not persist recording failure; continuing safe shutdown")
