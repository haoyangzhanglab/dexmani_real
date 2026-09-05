"""Safety-gated command scheduling for one preflighted physical replay."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.control.publication import (
    prepare_joint_command,
    publish_command,
    wait_command_accepted,
)
from dexmani_real.control.safety_gate import SafetyGate, planner_action_safety_gate
from dexmani_real.ipc.channels import (
    RuntimeChannels,
    read_arm_state_dict,
    read_hand_state_dict,
)
from dexmani_real.planning import (
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.arm_fk import make_arm_fk
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.replay.capture import ReplayRecorder
from dexmani_real.replay.trajectory import TrajectoryData, replay_start_state
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    begin_motion,
    require_transition,
    revoke_motion,
)
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.feedback import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_STATUS_INTERVAL_FRAMES = 50
_WAIT_POLL_INTERVAL_S = 0.01
_ARM_STREAMING_WAIT_TIMEOUT_S = 2.0
_START_HAND_WARMUP_S = 3.0


class ReplayStatus(str, Enum):
    """Terminal state of one replay attempt."""

    COMPLETED = "completed"
    USER_QUIT = "user_quit"
    REJECTED = "rejected"
    ESTOP = "estop"
    FAULT = "fault"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True)
class ReplayOutcome:
    """Replay result, including any samples captured before it stopped."""

    status: ReplayStatus
    replay_data: dict[str, np.ndarray] | None = None
    reason: str = ""

    @property
    def successful(self) -> bool:
        return self.status in (ReplayStatus.COMPLETED, ReplayStatus.USER_QUIT)


def _max_joint_deviation_deg(
    measured_qpos: np.ndarray, reference_qpos: np.ndarray
) -> tuple[float, int]:
    """Return the largest absolute joint error in degrees and its index."""
    deviation_deg = np.rad2deg(
        np.abs(
            np.asarray(measured_qpos, dtype=np.float64)
            - np.asarray(reference_qpos, dtype=np.float64)
        )
    )
    index = int(np.argmax(deviation_deg))
    return float(deviation_deg[index]), index


def arm_error_requires_stop(error_code: int) -> bool:
    """Any non-zero arm controller error is a stop condition (no worker recovery)."""
    return error_code != 0


def arm_feedback_issue(
    state: dict[str, Any] | None,
    max_age_s: float,
    *,
    now_ns: int | None = None,
) -> str | None:
    """Return why xArm feedback is unusable for replay."""
    if state is None:
        return "arm feedback unavailable"
    try:
        return validate_arm_feedback(
            connected=bool(state.get("connected", False)),
            error_code=int(state.get("error_code", -1)),
            state_valid=bool(state.get("state_valid", False)),
            source_monotonic_ns=int(state.get("source_monotonic_ns", 0)),
            now_monotonic_ns=time.monotonic_ns() if now_ns is None else now_ns,
            max_age_s=max_age_s,
            qpos=np.asarray(state.get("qpos")),
            qvel=np.asarray(state.get("qvel")),
        )
    except (TypeError, ValueError) as exc:
        return f"invalid arm feedback: {exc}"


def hand_feedback_is_healthy(
    state: dict[str, Any] | None,
    max_age_s: float,
    *,
    now_ns: int | None = None,
) -> bool:
    """Validate every hand feedback health field used by physical replay."""
    if state is None:
        return False
    try:
        issue = validate_hand_feedback(
            connected=bool(state.get("connected", False)),
            state_valid=bool(state.get("state_valid", False)),
            source_monotonic_ns=int(state.get("source_monotonic_ns", 0)),
            now_monotonic_ns=time.monotonic_ns() if now_ns is None else now_ns,
            max_age_s=max_age_s,
            qpos=np.asarray(state.get("qpos")),
        )
    except (TypeError, ValueError):
        return False
    return issue is None


class EpisodeReplayer:
    """Replay one preflight-validated command stream through arm and hand workers."""

    START_POSE_TOLERANCE_DEG = 5.0

    def __init__(
        self,
        trajectory: TrajectoryData,
        shared: RuntimeChannels,
        *,
        runtime: ResolvedRuntimeConfig,
        health_check: Callable[[], str | None] | None = None,
    ) -> None:
        self.traj = trajectory
        self.shared = shared
        self.runtime = runtime
        self.replay_hz = trajectory.fps
        if not np.isfinite(self.replay_hz) or self.replay_hz <= 0:
            raise ValueError("replay rate must be finite and positive")

        self._recorded_arm_start = replay_start_state(trajectory)[0]
        self._health_check = health_check
        self._home_planner: XArm7MotionPlanner | None = None
        self._replay_planner: XArm7MotionPlanner | None = None
        self._action_safety_gate: SafetyGate | None = None
        self._start_warmup_gate: SafetyGate | None = None
        self._recorder: ReplayRecorder | None = None
        self._running = False
        self._estopped = False
        self._motion_started = False
        self._status = ReplayStatus.COMPLETED
        self._reason = ""
        self._hand_available = trajectory.has_hand
        self._frame_count = trajectory.num_frames

    def _make_planner(
        self, workspace: np.ndarray, *, table: Any | None
    ) -> XArm7MotionPlanner:
        """Create one planner with an explicit table-collision policy."""
        runtime_policy = self.runtime.policy
        return XArm7MotionPlanner(
            XArm7PlannerConfig(
                urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
                srdf_path=str(XARM7_XHAND_SRDF_PATH),
                base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
                workspace_bounds=workspace,
            ),
            teleop_profile=TeleopProfile(
                max_pose_error_pos_m=float(runtime_policy.ik_max_pose_error_pos_m),
                max_pose_error_rot_rad=float(runtime_policy.ik_max_pose_error_rot_rad),
            ),
            hand_dof=True,
            static_boxes=tuple(self.runtime.environment.static_boxes),
            table=table,
        )

    def setup(self) -> None:
        """Create replay and return-home safety boundaries without device IO."""
        runtime_arm = self.runtime.arm
        runtime_hand = self.runtime.hand
        runtime_policy = self.runtime.policy
        workspace = np.array(
            [
                [runtime_policy.workspace.x_min, runtime_policy.workspace.x_max],
                [runtime_policy.workspace.y_min, runtime_policy.workspace.y_max],
                [runtime_policy.workspace.z_min, runtime_policy.workspace.z_max],
            ],
            dtype=np.float64,
        )
        replay_planner = self._make_planner(workspace, table=None)
        home_planner = self._make_planner(
            workspace,
            table=self.runtime.environment.table,
        )
        self._replay_planner = replay_planner
        self._home_planner = home_planner
        self._action_safety_gate = planner_action_safety_gate(
            planner=replay_planner,
            arm_joint_lower_rad=tuple(runtime_arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime_arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime_hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime_hand.qpos_max_rad),
        )
        self._start_warmup_gate = planner_action_safety_gate(
            planner=replay_planner,
            arm_joint_lower_rad=tuple(runtime_arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime_arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime_hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime_hand.qpos_max_rad),
            collision_check=replay_planner.collision_model.check_transition_collision_free,
        )
        print("Replay safety gates ready")

    def _arm_start_deviation(self, arm_qpos: np.ndarray) -> tuple[float, int]:
        """Compare fresh xArm feedback with the recorded physical start state."""
        nearest_arm_start = wrap_nearest_equivalent(
            self._recorded_arm_start,
            np.asarray(arm_qpos, dtype=np.float64),
            tuple(self.runtime.arm.joint_limit_lower),
            tuple(self.runtime.arm.joint_limit_upper),
        )
        return _max_joint_deviation_deg(arm_qpos, nearest_arm_start)

    @staticmethod
    def _format_arm_start_deviation(max_deg: float, joint_index: int) -> str:
        """Format the xArm start error for the operator."""
        return f"arm={max_deg:.1f}° (joint {joint_index + 1})"

    def _first_replay_transition_issue(
        self, arm_qpos: np.ndarray, hand_qpos: np.ndarray
    ) -> str | None:
        """Validate the live start state against the first replay command."""
        assert self._replay_planner is not None
        assert self.traj.action_hand_joint is not None
        first_arm_cmd = wrap_nearest_equivalent(
            self.traj.action_arm_joint[0],
            np.asarray(arm_qpos, dtype=np.float64),
            tuple(self.runtime.arm.joint_limit_lower),
            tuple(self.runtime.arm.joint_limit_upper),
        )
        first_hand_cmd = self.traj.action_hand_joint[0]
        if not self._replay_planner.is_workspace_segment_safe(arm_qpos, first_arm_cmd):
            return "live start->frame 0 workspace check failed"
        if not self._replay_planner.collision_model.check_transition_collision_free(
            arm_qpos,
            first_arm_cmd,
            hand_qpos,
            first_hand_cmd,
        ):
            return "live start->frame 0 collision check failed"
        return None

    def _confirm_replay_start(
        self,
        arm_qpos: np.ndarray,
        hand_qpos: np.ndarray,
    ) -> bool:
        """Accept the live post-warm-up state only when frame 0 remains safe."""
        issue = self._first_replay_transition_issue(arm_qpos, hand_qpos)
        if issue is not None:
            self._enter_terminal_quiescence()
            reason = f"replay start was rejected: {issue}"
            print(f"Replay rejected: {reason}")
            self._reject(reason)
            return False
        print("XHand warm-up complete")
        return True

    def _reject(self, reason: str) -> None:
        """Stop before replay commands without converting a valid runtime into a fault."""
        self._running = False
        self._status = ReplayStatus.REJECTED
        self._reason = reason

    def _read_start_feedback(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Read fresh arm and hand feedback required by start warm-up."""
        arm_state = read_arm_state_dict(self.shared)
        arm_issue = arm_feedback_issue(
            arm_state,
            float(self.runtime.policy.arm_state_stale_threshold_s),
        )
        if arm_state is None or arm_issue is not None or arm_state["error_code"] != 0:
            self._fault(
                "initial arm feedback is unavailable or unhealthy: "
                f"{arm_issue or 'controller error'}"
            )
            return None
        hand_state = read_hand_state_dict(self.shared)
        if not hand_feedback_is_healthy(
            hand_state,
            float(self.runtime.safety.heartbeat_timeouts["hand"]),
        ):
            self._fault("initial hand feedback is unavailable or unhealthy")
            return None
        assert hand_state is not None
        return arm_state, hand_state

    def _warm_up_hand_at_start(self, keyboard: KeyboardHandler) -> bool:
        """Warm up XHand's first target after reset while holding xArm still."""
        feedback = self._read_start_feedback()
        if feedback is None:
            return False
        arm_state, hand_state = feedback
        arm_max_deg, arm_joint_index = self._arm_start_deviation(arm_state["qpos"])
        if arm_max_deg > self.START_POSE_TOLERANCE_DEG:
            print(
                "\nReplay rejected: xArm is not at the recorded start "
                f"({self._format_arm_start_deviation(arm_max_deg, arm_joint_index)}; "
                f"limit {self.START_POSE_TOLERANCE_DEG:.1f}°)."
            )
            print("Move xArm to the recorded start in a separate supervised procedure.")
            self._reject("xArm is not at the recorded trajectory start")
            return False

        assert self._start_warmup_gate is not None
        assert self.traj.action_hand_joint is not None
        print(
            "Warming up XHand from connection reset toward the frame 0 hand target "
            f"({self._format_arm_start_deviation(arm_max_deg, arm_joint_index)}; "
            "Q=quit  ESC=emergency_stop)"
        )
        prepared = prepare_joint_command(
            self.shared,
            np.asarray(arm_state["qpos"], dtype=np.float64),
            self.traj.action_hand_joint[0],
            gate=self._start_warmup_gate,
            is_hold=True,
            action_validity_s=_START_HAND_WARMUP_S,
            hand_mechanical_lower_rad=np.asarray(
                self.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ),
            hand_mechanical_upper_rad=np.asarray(
                self.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ),
            arm_feedback_max_age_s=float(self.runtime.safety.heartbeat_timeouts["arm"]),
            hand_feedback_max_age_s=float(
                self.runtime.safety.heartbeat_timeouts["hand"]
            ),
        )
        candidate = prepared.candidate
        published = (
            publish_command(self.shared, candidate) if candidate is not None else None
        )
        if published is None or not published.published:
            detail = prepared.reason or (published.reason if published else "")
            reason = f"XHand start warm-up was rejected: {detail}"
            if not prepared.unavailable and not prepared.fatal:
                print(f"Replay rejected: {reason}")
                self._reject(reason)
            else:
                self._fault(reason)
            return False
        self._motion_started = True

        deadline_s = time.monotonic() + _START_HAND_WARMUP_S
        while time.monotonic() < deadline_s:
            if not self._poll_control(keyboard, 0.0):
                return False
            feedback = self._read_start_feedback()
            if feedback is None:
                return False
            arm_state, hand_state = feedback
            time.sleep(_WAIT_POLL_INTERVAL_S)

        arm_max_deg, arm_joint_index = self._arm_start_deviation(arm_state["qpos"])
        if arm_max_deg > self.START_POSE_TOLERANCE_DEG:
            self._enter_terminal_quiescence()
            reason = "xArm left the recorded start during XHand warm-up"
            print(
                "Replay rejected: "
                f"{reason} ({self._format_arm_start_deviation(arm_max_deg, arm_joint_index)}; "
                f"limit {self.START_POSE_TOLERANCE_DEG:.1f}°)"
            )
            self._reject(reason)
            return False
        return self._confirm_replay_start(arm_state["qpos"], hand_state["qpos"])

    def _outcome(self) -> ReplayOutcome:
        replay_data = self._recorder.to_dict() if self._recorder is not None else None
        return ReplayOutcome(self._status, replay_data, self._reason)

    def _fault(self, reason: str, *, estop: bool = False) -> None:
        """Latch a replay fault without clearing it during cleanup."""
        self._running = False
        self._estopped = self._estopped or estop
        self._status = ReplayStatus.ESTOP if estop else ReplayStatus.FAULT
        self._reason = reason
        if self.shared is None:
            return
        if estop:
            self.shared.estop_request.value = True
        self.shared.error_state.value = True
        self.shared.is_running.value = False
        require_transition(self.shared, SafetyState.FAULT)

    def _runtime_issue(self) -> tuple[ReplayStatus, str] | None:
        """Return (status, reason) if shared state signals a fault, estop, or health issue; None otherwise."""
        assert self.shared is not None
        if self.shared.estop_request.value:
            return ReplayStatus.ESTOP, "e-stop requested"
        if self.shared.error_state.value:
            return ReplayStatus.FAULT, "sticky error_state set"
        if int(self.shared.safety_state.value) == int(SafetyState.FAULT):
            return ReplayStatus.FAULT, "safety state is FAULT"
        if not self.shared.is_running.value:
            return ReplayStatus.FAULT, "runtime stop requested unexpectedly"
        if self._health_check is not None:
            issue = self._health_check()
            if issue:
                return ReplayStatus.FAULT, issue
        return None

    def _enter_terminal_quiescence(self) -> None:
        """Invalidate queued replay endpoints and publish nothing further.

        An endpoint already accepted by firmware is not retractable; verified
        shutdown later places the controller in State 4.
        """
        assert self.shared is not None
        if not revoke_motion(self.shared, SafetyState.ARMED):
            self._fault("failed to establish terminal replay command boundary")
            return
        run_generation = int(self.shared.run_generation.value)
        logger.info(
            "replay entered terminal command quiescence (run=%d)",
            run_generation,
        )

    def _poll_control(self, keyboard: KeyboardHandler, timeout_s: float) -> bool:
        """Poll keyboard and runtime health for up to *timeout_s* seconds.

        Returns:
            True if the replay loop should continue, False if a stop/fault/quit
            signal was handled.
        """
        assert self.shared is not None
        signals = set(keyboard.poll(timeout=max(0.0, timeout_s)))
        if ControlSignal.EMERGENCY_STOP in signals:
            print("\nESC: emergency stop")
            self._fault("operator emergency stop", estop=True)
            return False
        issue = self._runtime_issue()
        if issue is not None:
            status, reason = issue
            self._fault(reason, estop=status is ReplayStatus.ESTOP)
            return False
        if ControlSignal.QUIT in signals:
            print("\nQ: stopping command publication")
            self._enter_terminal_quiescence()
            self._running = False
            self._status = ReplayStatus.USER_QUIT
            self._reason = "operator quit after entering command quiescence"
            return False
        return True

    def _wait_until_deadline(
        self, keyboard: KeyboardHandler, deadline_s: float
    ) -> bool:
        """Wait in short slices so controls and worker health remain responsive."""
        while self._running:
            remaining_s = deadline_s - time.perf_counter()
            if remaining_s <= 0:
                return self._poll_control(keyboard, 0.0)
            if not self._poll_control(
                keyboard, min(_WAIT_POLL_INTERVAL_S, remaining_s)
            ):
                return False
        return False

    def _wait_arm_streaming(self, keyboard: KeyboardHandler) -> bool:
        """Block until the arm worker is streaming valid state before the first publish.

        The worker enters servo Mode 6 once at startup and publishes its first
        frame just before signalling READY.  Publishing before the worker
        streams would send endpoints nobody applies, so waiting for a valid,
        fault-free frame makes the consumer ready before the producer starts.

        Returns False (with a fault/quit already latched) if the run is stopped
        or the arm never streams within the bounded window.
        """
        assert self.shared is not None
        deadline = time.perf_counter() + _ARM_STREAMING_WAIT_TIMEOUT_S
        while time.perf_counter() < deadline:
            if not self._poll_control(keyboard, 0.0):
                return False
            arm_state = read_arm_state_dict(self.shared)
            if (
                arm_state is not None
                and bool(arm_state.get("state_valid", False))
                and int(arm_state.get("error_code", -1)) == 0
            ):
                return True
            time.sleep(_WAIT_POLL_INTERVAL_S)
        self._fault(
            "arm worker did not start streaming before the replay start deadline"
        )
        return False

    def run(self) -> ReplayOutcome:
        """Execute the replay loop and report its explicit terminal outcome."""
        frame_count = self._frame_count
        print(f"\nReplay: {frame_count} frames @ {self.replay_hz:.1f} Hz")
        print(f"  Source: {self.traj.episode_path}")
        if self.traj.task_label:
            print(f"  Task:   {self.traj.task_label}")
        print(f"  Hand:   {'ON' if self._hand_available else 'OFF'}")
        print("\nControl: Q=quit  ESC=emergency_stop\n")

        has_hand = self._hand_available
        self._recorder = ReplayRecorder(frame_count, has_hand=has_hand)
        keyboard = KeyboardHandler(
            estop_callback=lambda: setattr(self.shared.estop_request, "value", True)
        )
        keyboard.start()
        self._running = True
        error_count = 0
        max_consecutive_errors = int(self.runtime.policy.max_consecutive_errors)
        period_s = 1.0 / self.replay_hz
        next_deadline_s = time.perf_counter()
        start_time = next_deadline_s
        frame_idx = 0

        try:
            if not self._warm_up_hand_at_start(keyboard):
                return self._outcome()
            if not begin_motion(self.shared):
                self._fault("failed to enter replay motion")
                return self._outcome()
            self._motion_started = True
            if not self._wait_arm_streaming(keyboard):
                return self._outcome()
            while frame_idx < frame_count and self._wait_until_deadline(
                keyboard, next_deadline_s
            ):
                arm_cmd = self.traj.action_arm_joint[frame_idx].copy()
                hand_cmd = None
                if has_hand and self.traj.action_hand_joint is not None:
                    hand_cmd = self.traj.action_hand_joint[frame_idx].copy()
                # Replay only slots whose recording flag indicates a queued command.
                send_this = self.traj.send_mask is None or bool(
                    self.traj.send_mask[frame_idx]
                )
                if send_this and (
                    not np.all(np.isfinite(arm_cmd))
                    or (hand_cmd is not None and not np.all(np.isfinite(hand_cmd)))
                ):
                    self._fault(
                        f"frame {frame_idx} contains a non-finite replay action"
                    )
                    break

                arm_state = read_arm_state_dict(self.shared)
                if (
                    arm_feedback_issue(
                        arm_state,
                        float(self.runtime.policy.arm_state_stale_threshold_s),
                    )
                    is not None
                ):
                    error_count += 1
                    if error_count >= max_consecutive_errors:
                        self._fault("too many consecutive arm state read failures")
                        break
                    next_deadline_s = time.perf_counter() + min(
                        period_s, _WAIT_POLL_INTERVAL_S
                    )
                    continue
                assert arm_state is not None

                eef_pos, eef_rot6d = make_arm_fk().compute(arm_state["qpos"])

                error_code = int(arm_state["error_code"])
                if arm_error_requires_stop(error_code):
                    self._fault(f"fatal arm controller error C{error_code}")
                    break

                error_count = 0
                hand_qpos: np.ndarray | None = None
                if has_hand:
                    hand_state = read_hand_state_dict(self.shared)
                    if not hand_feedback_is_healthy(
                        hand_state,
                        float(self.runtime.safety.heartbeat_timeouts["hand"]),
                    ):
                        self._fault(
                            f"frame {frame_idx}: hand feedback is unavailable or unhealthy"
                        )
                        break
                    assert hand_state is not None
                    hand_qpos = hand_state["qpos"]

                if not send_this:
                    # During recorded quiescence, observe but send nothing.
                    self._recorder.record(
                        frame_idx,
                        arm_state["qpos"],
                        eef_pos,
                        eef_rot6d,
                        arm_cmd,
                        hand_cmd,
                        time.perf_counter(),
                        arm_tracking_error=arm_state["tracking_err"],
                        hand_qpos=hand_qpos,
                    )
                    frame_idx += 1
                    next_deadline_s += period_s
                    now_s = time.perf_counter()
                    if next_deadline_s < now_s:
                        next_deadline_s = now_s + period_s
                    continue

                # 2π-canonicalize the replayed command to the measured arm pose
                # (defense-in-depth; the worker no longer wraps).
                arm_cmd = wrap_nearest_equivalent(
                    arm_cmd,
                    arm_state["qpos"],
                    tuple(self.runtime.arm.joint_limit_lower),
                    tuple(self.runtime.arm.joint_limit_upper),
                )
                is_final_frame = frame_idx == frame_count - 1
                assert self._action_safety_gate is not None
                prepared = prepare_joint_command(
                    self.shared,
                    arm_cmd,
                    hand_cmd,
                    gate=self._action_safety_gate,
                    hand_mechanical_lower_rad=np.asarray(
                        self.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
                    ),
                    hand_mechanical_upper_rad=np.asarray(
                        self.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
                    ),
                    arm_feedback_max_age_s=float(
                        self.runtime.safety.heartbeat_timeouts["arm"]
                    ),
                    hand_feedback_max_age_s=float(
                        self.runtime.safety.heartbeat_timeouts["hand"]
                    ),
                )
                candidate = prepared.candidate
                published = (
                    publish_command(self.shared, candidate)
                    if candidate is not None
                    else None
                )
                accepted = None
                if (
                    is_final_frame
                    and candidate is not None
                    and published is not None
                    and published.published
                    and published.ticket is not None
                ):
                    accepted = wait_command_accepted(
                        self.shared,
                        ticket=published.ticket,
                        action_id=candidate.action_id,
                        wait_for_arm=True,
                        wait_for_hand=candidate.hand_qpos is not None,
                        timeout_s=float(self.runtime.policy.action_apply_timeout_s),
                        arm_feedback_max_age_s=float(
                            self.runtime.safety.heartbeat_timeouts["arm"]
                        ),
                        hand_feedback_max_age_s=float(
                            self.runtime.safety.heartbeat_timeouts["hand"]
                        ),
                    )
                if (
                    published is None
                    or not published.published
                    or candidate is None
                    or (is_final_frame and (accepted is None or not accepted.accepted))
                ):
                    boundary = "publish/acceptance" if is_final_frame else "publish"
                    reason = prepared.reason
                    if not reason and published is not None:
                        reason = published.reason
                    if not reason and accepted is not None:
                        reason = accepted.reason
                    self._fault(
                        f"frame {frame_idx}: joint {boundary} boundary rejected: {reason}"
                    )
                    break
                assert candidate.arm_qpos is not None
                sent_arm_cmd = np.asarray(candidate.arm_qpos, dtype=np.float64)
                if candidate.hand_qpos is not None:
                    hand_cmd = np.asarray(candidate.hand_qpos, dtype=np.float64)

                self._recorder.record(
                    frame_idx,
                    arm_state["qpos"],
                    eef_pos,
                    eef_rot6d,
                    sent_arm_cmd,
                    hand_cmd,
                    time.perf_counter(),
                    arm_sent_cmd=sent_arm_cmd,
                    arm_tracking_error=arm_state["tracking_err"],
                    hand_qpos=hand_qpos,
                )
                frame_idx += 1
                if frame_idx % _STATUS_INTERVAL_FRAMES == 0 or frame_idx == 1:
                    elapsed_s = time.perf_counter() - start_time
                    print(
                        f"[T+{elapsed_s:.1f}s f={frame_idx}/{frame_count}] "
                        f"eef={np.round(eef_pos, 3)}m  err={error_count}",
                        flush=True,
                    )
                next_deadline_s += period_s
                now_s = time.perf_counter()
                if next_deadline_s < now_s:
                    next_deadline_s = now_s + period_s
        except KeyboardInterrupt:
            print("\nInterrupted by user; stopping command publication")
            self._enter_terminal_quiescence()
            self._running = False
            self._status = ReplayStatus.USER_QUIT
            self._reason = "operator interrupt after entering command quiescence"
        except Exception as exc:
            logger.error("Unexpected replay failure", exc_info=True)
            self._fault(f"unexpected replay failure: {exc}")
        finally:
            keyboard.stop()
            if not self._estopped and int(self.shared.safety_state.value) == int(
                SafetyState.RUNNING
            ):
                revoke_motion(self.shared, SafetyState.ARMED)

        if self._recorder.count < frame_count:
            print(f"\nReplay stopped at frame {self._recorder.count}/{frame_count}")
        return self._outcome()

    @property
    def can_offer_home(self) -> bool:
        return self._motion_started and not self._estopped

    @property
    def partial_data(self) -> dict[str, np.ndarray] | None:
        return self._recorder.to_dict() if self._recorder is not None else None

    @property
    def planner(self) -> XArm7MotionPlanner | None:
        return self._home_planner

    def shutdown(self) -> None:
        """Signal processes to stop; the session owns RuntimeChannels cleanup."""
        self.shared.is_running.value = False
