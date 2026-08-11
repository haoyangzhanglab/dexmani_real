"""Preflight-gated trajectory execution through SharedStorage workers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.ipc.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.planning import Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.policy.action_protocol import (
    ActionSafetyGate,
    ActionSafetyGateConfig,
    advance_policy_epoch,
    planner_action_safety_gate,
    publish_joint_targets,
)
from dexmani_real.replay.data import TrajectoryData
from dexmani_real.replay.metrics import ReplayRecorder
from dexmani_real.robot.safety import SafetyState, require_transition
from dexmani_real.shm.shared_storage import SharedStorage, read_arm_state_dict, read_hand_state_dict
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler, validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_STATUS_INTERVAL_FRAMES = 50
_WAIT_POLL_INTERVAL_S = 0.01


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


def arm_error_requires_stop(error_code: int, recoverable_errors: tuple[int, ...]) -> bool:
    """Only explicitly configured controller errors may continue to worker recovery."""
    return error_code != 0 and error_code not in recoverable_errors


def arm_feedback_issue(
    state: dict[str, Any] | None,
    max_age_s: float,
    *,
    now_ns: int | None = None,
) -> str | None:
    """Return why xArm feedback is unusable for replay, excluding controller error codes."""
    if state is None:
        return "arm feedback unavailable"
    try:
        return validate_arm_feedback(
            connected=bool(state.get("connected", False)),
            state_valid=bool(state.get("state_valid", False)),
            source_monotonic_ns=int(state.get("source_monotonic_ns", 0)),
            now_monotonic_ns=time.monotonic_ns() if now_ns is None else now_ns,
            max_age_s=max_age_s,
            qpos=np.asarray(state.get("qpos")),
            qvel=np.asarray(state.get("qvel")),
            eef_pos=np.asarray(state.get("eef_pos")),
            eef_rot6d=np.asarray(state.get("eef_rot6d")),
        )
    except (TypeError, ValueError) as exc:
        return f"invalid arm feedback: {exc}"


def hand_feedback_is_healthy(
    state: dict[str, Any] | None,
    max_age_s: float,
    *,
    now_ns: int | None = None,
) -> bool:
    """Validate every hand feedback health field used by live replay."""
    if state is None:
        return False
    try:
        issue = validate_hand_feedback(
            connected=bool(state.get("connected", False)),
            error_state=bool(state.get("error_state", True)),
            qpos_stale=bool(state.get("qpos_stale", True)),
            state_valid=bool(state.get("state_valid", False)),
            send_healthy=bool(state.get("send_healthy", False)),
            read_healthy=bool(state.get("read_healthy", False)),
            source_monotonic_ns=int(state.get("source_monotonic_ns", 0)),
            now_monotonic_ns=time.monotonic_ns() if now_ns is None else now_ns,
            max_age_s=max_age_s,
            qpos=np.asarray(state.get("qpos")),
        )
    except (TypeError, ValueError):
        return False
    return issue is None


class TrajectoryReplayer:
    """Replay one preflight-validated command stream through arm and hand workers."""

    START_POSE_TOLERANCE_DEG = 5.0

    def __init__(
        self,
        trajectory: TrajectoryData,
        shared: SharedStorage | None,
        *,
        speed: float = 1.0,
        dry_run: bool = False,
        no_hand: bool = False,
        max_frames: int | None = None,
        runtime: ResolvedRuntimeConfig | None = None,
        health_check: Callable[[], str | None] | None = None,
    ) -> None:
        self.traj = trajectory
        self.shared = shared
        self.speed = speed
        self.dry_run = dry_run
        self.no_hand = no_hand
        self.runtime = runtime
        self.replay_hz = trajectory.fps * speed
        if not np.isfinite(self.replay_hz) or self.replay_hz <= 0:
            raise ValueError("replay rate must be finite and positive")

        self._health_check = health_check
        self._planner: XArm7MotionPlanner | None = None
        self._action_safety_gate: ActionSafetyGate | None = None
        self._recorder: ReplayRecorder | None = None
        self._running = False
        self._estopped = False
        self._live_motion_started = False
        self._status = ReplayStatus.COMPLETED
        self._reason = ""
        self._hand_available = trajectory.has_hand and not no_hand
        self._frame_count = trajectory.num_frames if max_frames is None else min(trajectory.num_frames, max_frames)

    def setup(self) -> None:
        """Create the geometry safety gate without connecting to hardware."""
        if self.dry_run:
            return
        if self.runtime is None or self.shared is None:
            raise RuntimeError("live replay requires runtime configuration and SharedStorage")
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
        model_dir = ASSET_DIR / "robots" / "xhand"
        self._planner = XArm7MotionPlanner(
            XArm7PlannerConfig(
                urdf_path=str(model_dir / "xarm7_xhand_collision.urdf"),
                srdf_path=str(model_dir / "xarm7_xhand.srdf"),
                base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
                workspace_bounds=workspace,
            ),
            teleop_profile=TeleopProfile(
                max_pose_error_pos_m=float(runtime_policy.ik_max_pose_error_pos_m),
                max_pose_error_rot_rad=float(runtime_policy.ik_max_pose_error_rot_rad),
            ),
            hand_dof=True,
            static_boxes=tuple(self.runtime.environment.static_boxes),
        )
        self._action_safety_gate = planner_action_safety_gate(
            ActionSafetyGateConfig(
                arm_joint_lower_rad=tuple(runtime_arm.joint_limit_lower),
                arm_joint_upper_rad=tuple(runtime_arm.joint_limit_upper),
                hand_joint_lower_rad=tuple(runtime_hand.qpos_min_rad),
                hand_joint_upper_rad=tuple(runtime_hand.qpos_max_rad),
                arm_max_velocity_rad_s=float(np.deg2rad(runtime_arm.max_joint_velocity_deg_per_s)),
                hand_max_velocity_rad_s=(
                    float(runtime_hand.max_delta_rad) * runtime_policy.control_hz
                    if runtime_hand.max_delta_rad is not None
                    else float(np.deg2rad(runtime_hand.safety_gate_max_velocity_deg_per_s))
                ),
                require_geometry_checks=True,
            ),
            planner=self._planner,
            table_z_surface_m=float(runtime_arm.table_z_surface_m),
            hand_safety_margin_m=float(runtime_arm.hand_safety_margin_m),
        )
        print("Replay geometry safety gate ready")

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
                print("Cannot verify the hand start pose from fresh connected feedback.")
                return False
            max_dev = max(max_dev, float(np.max(np.rad2deg(np.abs(hand_qpos - first_hand_cmd)))))
        if np.isfinite(max_dev) and max_dev <= self.START_POSE_TOLERANCE_DEG:
            return True
        print(f"\nRobot is {max_dev:.1f}° from the trajectory start (limit {self.START_POSE_TOLERANCE_DEG:.1f}°).")
        print("Move to the validated start pose in a separate supervised procedure, then retry replay.")
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

    def _read_terminal_hold_targets(
        self,
        *,
        newer_than_ns: int,
        timeout_s: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Wait for healthy measured joints published after an epoch change."""
        assert self.shared is not None
        assert self.runtime is not None
        deadline_s = time.monotonic() + timeout_s
        last_issue = "feedback not yet newer than the replay epoch"
        while time.monotonic() < deadline_s:
            runtime_issue = self._runtime_issue()
            if runtime_issue is not None:
                raise RuntimeError(runtime_issue[1])

            arm_state = read_arm_state_dict(self.shared)
            arm_issue = arm_feedback_issue(
                arm_state,
                float(self.runtime.policy.arm_state_stale_threshold_s),
            )
            if arm_issue is None and arm_state is not None:
                if int(arm_state["error_code"]) != 0:
                    arm_issue = f"arm controller error C{int(arm_state['error_code'])}"
                elif int(arm_state["source_monotonic_ns"]) <= newer_than_ns:
                    arm_issue = "arm feedback not yet newer than the replay epoch"

            hand_state: dict[str, Any] | None = None
            hand_issue: str | None = None
            if self._hand_available:
                hand_state = read_hand_state_dict(self.shared)
                if not hand_feedback_is_healthy(
                    hand_state,
                    float(self.runtime.safety.heartbeat_timeouts["hand"]),
                ):
                    hand_issue = "hand feedback is unavailable or unhealthy"
                elif hand_state is not None and int(hand_state["source_monotonic_ns"]) <= newer_than_ns:
                    hand_issue = "hand feedback not yet newer than the replay epoch"

            if arm_issue is None and hand_issue is None and arm_state is not None:
                arm_qpos = np.asarray(arm_state["qpos"], dtype=np.float64).copy()
                hand_qpos = None if hand_state is None else np.asarray(hand_state["qpos"], dtype=np.float64).copy()
                return arm_qpos, hand_qpos

            last_issue = arm_issue or hand_issue or last_issue
            time.sleep(_WAIT_POLL_INTERVAL_S)
        raise TimeoutError(last_issue)

    def _publish_terminal_hold(self) -> str | None:
        """Invalidate pending replay endpoints and confirm a measured hold."""
        assert self.shared is not None
        assert self.runtime is not None
        apply_timeout_s = float(self.runtime.policy.action_apply_timeout_s)
        try:
            epoch = advance_policy_epoch(self.shared)
            epoch_changed_ns = time.monotonic_ns()
            arm_qpos, hand_qpos = self._read_terminal_hold_targets(
                newer_than_ns=epoch_changed_ns,
                timeout_s=apply_timeout_s,
            )
        except (RuntimeError, TimeoutError, ValueError) as exc:
            return str(exc)

        candidate = publish_joint_targets(
            self.shared,
            arm_qpos,
            hand_qpos,
            is_hold=True,
            prepare_timeout_s=float(self.runtime.policy.action_prepare_timeout_s),
            dt_s=1.0 / self.replay_hz,
            safety_gate=self._action_safety_gate,
            wait_applied=True,
            apply_timeout_s=apply_timeout_s,
        )
        if candidate is None:
            return f"measured hold was rejected or not applied after advancing to epoch {epoch}"
        logger.info("replay terminal hold applied (epoch=%d action_id=%d)", epoch, candidate.action_id)
        return None

    def _poll_control(self, keyboard: KeyboardHandler, timeout_s: float) -> bool:
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
            print("\nQ: stopping at a measured hold")
            hold_issue = self._publish_terminal_hold()
            if hold_issue is not None:
                self._fault(
                    f"operator quit could not establish a safe hold: {hold_issue}",
                    estop=bool(self.shared.estop_request.value),
                )
            else:
                self._running = False
                self._status = ReplayStatus.USER_QUIT
                self._reason = "operator quit after measured hold was applied"
            return False
        return True

    def _wait_until_deadline(self, keyboard: KeyboardHandler, deadline_s: float) -> bool:
        """Wait in short slices so controls and worker health remain responsive."""
        while self._running:
            remaining_s = deadline_s - time.perf_counter()
            if remaining_s <= 0:
                return self._poll_control(keyboard, 0.0)
            if not self._poll_control(keyboard, min(_WAIT_POLL_INTERVAL_S, remaining_s)):
                return False
        return False

    def _wait_for_arm_recovery(
        self,
        keyboard: KeyboardHandler,
        recoverable_errors: tuple[int, ...],
    ) -> dict[str, Any] | None:
        """Wait for worker-owned recovery without advancing the replay frame."""
        assert self.shared is not None
        assert self.runtime is not None
        deadline = time.monotonic() + float(self.runtime.policy.action_apply_timeout_s)
        max_age_s = float(self.runtime.policy.arm_state_stale_threshold_s)
        while time.monotonic() < deadline:
            if not self._poll_control(keyboard, _WAIT_POLL_INTERVAL_S):
                return None
            state = read_arm_state_dict(self.shared)
            if arm_feedback_issue(state, max_age_s) is not None:
                continue
            assert state is not None
            error_code = int(state["error_code"])
            if error_code == 0:
                return state
            if arm_error_requires_stop(error_code, recoverable_errors):
                self._fault(f"fatal arm controller error C{error_code}")
                return None

        self._fault("arm recovery timeout")
        return None

    def run(self) -> ReplayOutcome:
        """Execute the replay loop and report its explicit terminal outcome."""
        frame_count = self._frame_count
        print(f"\nReplay: {frame_count} frames @ {self.replay_hz:.1f} Hz (speed={self.speed}x)")
        print(f"  Source: {self.traj.episode_path}")
        if self.traj.task_label:
            print(f"  Task:   {self.traj.task_label}")
        print(f"  Hand:   {'ON' if (self._hand_available and not self.dry_run) else 'OFF'}")
        print(f"  Mode:   {'DRY RUN (no robot)' if self.dry_run else 'LIVE'}")
        print("\nControl: Q=quit  ESC=emergency_stop\n")

        if self.dry_run:
            self.validate_offline()
            return self._outcome()
        if self.shared is None or self.runtime is None:
            raise RuntimeError("live replay requires runtime configuration and SharedStorage")

        has_hand = self._hand_available
        self._recorder = ReplayRecorder(frame_count, has_hand=has_hand)
        arm_state = read_arm_state_dict(self.shared)
        initial_arm_issue = arm_feedback_issue(
            arm_state,
            float(self.runtime.policy.arm_state_stale_threshold_s),
        )
        if arm_state is None or initial_arm_issue is not None or arm_state["error_code"] != 0:
            self._fault(f"initial arm feedback is unavailable or unhealthy: {initial_arm_issue or 'controller error'}")
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
        if not self._align_to_start(first_arm_cmd, arm_state["qpos"], first_hand_cmd, initial_hand_qpos):
            self._status = ReplayStatus.REJECTED
            self._reason = "robot is not at the validated trajectory start"
            return self._outcome()

        shared = self.shared
        keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
        keyboard.start()
        self._running = True
        error_count = 0
        max_consecutive_errors = int(self.runtime.policy.max_consecutive_errors)
        recoverable_errors = tuple(int(code) for code in self.runtime.arm.recoverable_errors)
        period_s = 1.0 / self.replay_hz
        next_deadline_s = time.perf_counter()
        start_time = next_deadline_s
        frame_idx = 0

        try:
            require_transition(self.shared, SafetyState.RUNNING)
            self._live_motion_started = True
            while frame_idx < frame_count and self._wait_until_deadline(keyboard, next_deadline_s):
                arm_cmd = self.traj.action_arm_joint[frame_idx].copy()
                hand_cmd = None
                if has_hand and self.traj.action_hand_joint is not None:
                    hand_cmd = self.traj.action_hand_joint[frame_idx].copy()
                if not np.all(np.isfinite(arm_cmd)) or (hand_cmd is not None and not np.all(np.isfinite(hand_cmd))):
                    self._fault(f"frame {frame_idx} contains a non-finite replay action")
                    break

                arm_state = read_arm_state_dict(self.shared)
                if arm_feedback_issue(arm_state, float(self.runtime.policy.arm_state_stale_threshold_s)) is not None:
                    error_count += 1
                    if error_count >= max_consecutive_errors:
                        self._fault("too many consecutive arm state read failures")
                        break
                    next_deadline_s = time.perf_counter() + min(period_s, _WAIT_POLL_INTERVAL_S)
                    continue
                assert arm_state is not None

                error_code = int(arm_state["error_code"])
                if error_code != 0:
                    if arm_error_requires_stop(error_code, recoverable_errors):
                        self._fault(f"fatal arm controller error C{error_code}")
                        break
                    logger.warning("Frame %d: waiting for worker recovery from C%d", frame_idx, error_code)
                    recovered_state = self._wait_for_arm_recovery(keyboard, recoverable_errors)
                    if recovered_state is None:
                        break
                    arm_state = recovered_state
                    next_deadline_s = time.perf_counter()

                error_count = 0
                hand_qpos: np.ndarray | None = None
                if has_hand:
                    hand_state = read_hand_state_dict(self.shared)
                    if not hand_feedback_is_healthy(
                        hand_state,
                        float(self.runtime.safety.heartbeat_timeouts["hand"]),
                    ):
                        self._fault(f"frame {frame_idx}: hand feedback is unavailable or unhealthy")
                        break
                    assert hand_state is not None
                    hand_qpos = hand_state["qpos"]

                is_final_frame = frame_idx == frame_count - 1
                published = publish_joint_targets(
                    self.shared,
                    arm_cmd,
                    hand_cmd,
                    prepare_timeout_s=float(self.runtime.policy.action_prepare_timeout_s),
                    dt_s=period_s,
                    safety_gate=self._action_safety_gate,
                    wait_applied=is_final_frame,
                    apply_timeout_s=float(self.runtime.policy.action_apply_timeout_s),
                )
                if published is None:
                    boundary = "prepare/commit/APPLIED" if is_final_frame else "prepare/commit"
                    self._fault(f"frame {frame_idx}: joint {boundary} boundary rejected")
                    break
                assert published.arm_qpos is not None
                sent_arm_cmd = np.asarray(published.arm_qpos, dtype=np.float64)
                if published.hand_qpos is not None:
                    hand_cmd = np.asarray(published.hand_qpos, dtype=np.float64)

                self._recorder.record(
                    frame_idx,
                    arm_state["qpos"],
                    arm_state["eef_pos"],
                    arm_state["eef_rot6d"],
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
                        f"eef={np.round(arm_state['eef_pos'], 3)}m  err={error_count}",
                        flush=True,
                    )
                next_deadline_s += period_s
                now_s = time.perf_counter()
                if next_deadline_s < now_s:
                    next_deadline_s = now_s + period_s
        except KeyboardInterrupt:
            print("\nInterrupted by user; stopping at a measured hold")
            hold_issue = self._publish_terminal_hold()
            if hold_issue is not None:
                self._fault(
                    f"operator interrupt could not establish a safe hold: {hold_issue}",
                    estop=bool(self.shared.estop_request.value),
                )
            else:
                self._running = False
                self._status = ReplayStatus.USER_QUIT
                self._reason = "operator interrupt after measured hold was applied"
        except Exception as exc:
            logger.error("Unexpected replay failure", exc_info=True)
            self._fault(f"unexpected replay failure: {exc}")
        finally:
            keyboard.stop()
            if not self._estopped and int(self.shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(self.shared, SafetyState.ARMED)

        if self._recorder.count < frame_count:
            print(f"\nReplay stopped at frame {self._recorder.count}/{frame_count}")
        return self._outcome()

    def validate_offline(self) -> None:
        """Validate trajectory arrays without simulating wall-clock playback."""
        frame_count = self._frame_count
        arm = self.traj.action_arm_joint[:frame_count]
        if arm.shape != (frame_count, *ARM_JOINT_SHAPE) or not np.all(np.isfinite(arm)):
            raise ValueError("arm replay actions must be finite with shape (T, 7)")
        if not self.no_hand and self.traj.action_hand_joint is not None:
            hand = self.traj.action_hand_joint[:frame_count]
            if hand.shape != (frame_count, *HAND_JOINT_SHAPE) or not np.all(np.isfinite(hand)):
                raise ValueError("hand replay actions must be finite with shape (T, 12)")
        print(f"Dry-run complete: validated {frame_count} frames at nominal {self.replay_hz:.1f} Hz")

    @property
    def can_offer_home(self) -> bool:
        return self._live_motion_started and not self._estopped

    @property
    def partial_data(self) -> dict[str, np.ndarray] | None:
        return self._recorder.to_dict() if self._recorder is not None else None

    @property
    def planner(self) -> XArm7MotionPlanner | None:
        return self._planner

    @property
    def action_safety_gate(self) -> ActionSafetyGate | None:
        return self._action_safety_gate

    def shutdown(self) -> None:
        """Signal processes to stop; the session owns SharedStorage cleanup."""
        if not self.dry_run and self.shared is not None:
            self.shared.is_running.value = False
