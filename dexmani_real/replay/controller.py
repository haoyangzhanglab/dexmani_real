"""Safety-gated command scheduling for one preflighted physical replay."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.control.publication import publish_joint_targets
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
from dexmani_real.replay.trajectory import TrajectoryData
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    advance_run_generation,
    require_transition,
)
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.feedback import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_STATUS_INTERVAL_FRAMES = 50
_WAIT_POLL_INTERVAL_S = 0.01
_ARM_STREAMING_WAIT_TIMEOUT_S = 2.0


class ReplayStatus(str, Enum):
    """Terminal state of one replay attempt."""

    COMPLETED = "completed"
    USER_QUIT = "user_quit"
    REJECTED = "rejected"
    ESTOP = "estop"
    FAULT = "fault"


@dataclass(frozen=True)
class ReplayOutcome:
    """Replay result, including any samples captured before it stopped."""

    status: ReplayStatus
    replay_data: dict[str, np.ndarray] | None = None
    reason: str = ""

    @property
    def successful(self) -> bool:
        return self.status in (ReplayStatus.COMPLETED, ReplayStatus.USER_QUIT)


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

        self._health_check = health_check
        self._planner: XArm7MotionPlanner | None = None
        self._action_safety_gate: SafetyGate | None = None
        self._recorder: ReplayRecorder | None = None
        self._running = False
        self._estopped = False
        self._motion_started = False
        self._status = ReplayStatus.COMPLETED
        self._reason = ""
        self._hand_available = trajectory.has_hand
        self._frame_count = trajectory.num_frames

    def setup(self) -> None:
        """Create the action safety gate without connecting to hardware."""
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
        self._planner = XArm7MotionPlanner(
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
            table=self.runtime.environment.table,
        )
        self._action_safety_gate = planner_action_safety_gate(
            planner=self._planner,
            arm_joint_lower_rad=tuple(runtime_arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime_arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime_hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime_hand.qpos_max_rad),
        )
        print("Replay safety gate ready")

    def _align_to_start(
        self,
        first_arm_cmd: np.ndarray,
        arm_qpos: np.ndarray,
        first_hand_cmd: np.ndarray | None = None,
        hand_qpos: np.ndarray | None = None,
    ) -> bool:
        """Require measured joints to already be close to the validated start."""
        max_dev = float(np.max(np.rad2deg(np.abs(arm_qpos - first_arm_cmd))))
        if first_hand_cmd is not None:
            if hand_qpos is None:
                print(
                    "Cannot verify the hand start pose from fresh connected feedback."
                )
                return False
            max_dev = max(
                max_dev, float(np.max(np.rad2deg(np.abs(hand_qpos - first_hand_cmd))))
            )
        if np.isfinite(max_dev) and max_dev <= self.START_POSE_TOLERANCE_DEG:
            return True
        print(
            f"\nRobot is {max_dev:.1f}° from the trajectory start (limit {self.START_POSE_TOLERANCE_DEG:.1f}°)."
        )
        print(
            "Move to the validated start pose in a separate supervised procedure, then retry replay."
        )
        return False

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
        run_generation = advance_run_generation(self.shared)
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
        arm_state = read_arm_state_dict(self.shared)
        initial_arm_issue = arm_feedback_issue(
            arm_state,
            float(self.runtime.policy.arm_state_stale_threshold_s),
        )
        if (
            arm_state is None
            or initial_arm_issue is not None
            or arm_state["error_code"] != 0
        ):
            self._fault(
                f"initial arm feedback is unavailable or unhealthy: {initial_arm_issue or 'controller error'}"
            )
            return self._outcome()

        first_arm_cmd = self.traj.action_arm_joint[0].copy()
        first_hand_cmd: np.ndarray | None = None
        initial_hand_qpos: np.ndarray | None = None
        if has_hand and self.traj.action_hand_joint is not None:
            first_hand_cmd = self.traj.action_hand_joint[0].copy()
            hand_state = read_hand_state_dict(self.shared)
            if hand_feedback_is_healthy(
                hand_state,
                float(self.runtime.safety.heartbeat_timeouts["hand"]),
            ):
                assert hand_state is not None
                initial_hand_qpos = np.asarray(hand_state["qpos"], dtype=np.float64)
        if not self._align_to_start(
            first_arm_cmd, arm_state["qpos"], first_hand_cmd, initial_hand_qpos
        ):
            self._status = ReplayStatus.REJECTED
            self._reason = "robot is not at the validated trajectory start"
            return self._outcome()

        shared = self.shared
        keyboard = KeyboardHandler(
            estop_callback=lambda: setattr(shared.estop_request, "value", True)
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
            require_transition(self.shared, SafetyState.RUNNING)
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
                published = publish_joint_targets(
                    self.shared,
                    arm_cmd,
                    hand_cmd,
                    prepare_timeout_s=float(
                        self.runtime.policy.action_prepare_timeout_s
                    ),
                    safety_gate=self._action_safety_gate,
                    wait_applied=is_final_frame,
                    apply_timeout_s=float(self.runtime.policy.action_apply_timeout_s),
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
                if not published.succeeded or published.candidate is None:
                    boundary = "publish/APPLIED" if is_final_frame else "publish"
                    self._fault(
                        f"frame {frame_idx}: joint {boundary} boundary rejected: {published.reason}"
                    )
                    break
                candidate = published.candidate
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
                require_transition(self.shared, SafetyState.ARMED)

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
        return self._planner

    def shutdown(self) -> None:
        """Signal processes to stop; the session owns RuntimeChannels cleanup."""
        self.shared.is_running.value = False
