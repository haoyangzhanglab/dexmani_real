"""Policy-side current-tick publication for an isolated inference worker."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, NoReturn

import numpy as np

from dexmani_real.config.defaults import policy, safety
from dexmani_real.utils.schema import HAND_JOINT_SHAPE
from dexmani_real.policy.safety import (
    SafetyGate,
    advance_run_generation,
    send_command,
)
from dexmani_real.policy.inference_process import decode_candidate
from dexmani_real.policy.observation_sources import SharedObservationSource
from dexmani_real.policy.runtime import ActionCandidate, ObservationSnapshot
from dexmani_real.policy.spec import PolicySpec
from dexmani_real.policy.tensor_block import ObservationTensorBlock
from dexmani_real.teleop.keyboard import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_SNAPSHOT_CACHE_SIZE = 64


class CoordinatorTick(Enum):
    IDLE = "idle"
    SNAPSHOT = "snapshot"
    PUBLISHED = "published"
    HELD = "held"
    REJECTED = "rejected"
    REWARMING = "rewarming"


@dataclass(frozen=True)
class LearnedCoordinatorConfig:
    coordinator_hz: float = field(default_factory=lambda: policy.coordinator_hz)
    prepare_timeout_s: float = field(default_factory=lambda: policy.action_prepare_timeout_s)
    apply_timeout_s: float = field(default_factory=lambda: policy.action_apply_timeout_s)
    candidate_timeout_s: float = field(default_factory=lambda: policy.inference_candidate_timeout_s)
    arm_feedback_max_age_s: float = field(default_factory=lambda: policy.arm_state_stale_threshold_s)
    hand_feedback_max_age_s: float = field(default_factory=lambda: safety.heartbeat_timeouts["hand"])
    hand_enabled: bool = True
    hand_feedback_enabled: bool = True

    def __post_init__(self) -> None:
        timing = (
            self.coordinator_hz,
            self.prepare_timeout_s,
            self.apply_timeout_s,
            self.candidate_timeout_s,
            self.arm_feedback_max_age_s,
            self.hand_feedback_max_age_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError("learned coordinator rates/timeouts must be finite and positive")
        if self.hand_enabled and not self.hand_feedback_enabled:
            raise ValueError("hand actions require live hand feedback")


def _hold_before_quit(shared: Any, coordinator: "LearnedPolicyCoordinator") -> bool:
    """Invalidate pending motion and publish a measured hold before a RUNNING quit."""
    from dexmani_real.robot.safety import SafetyState, transition

    if int(shared.safety_state.value) != int(SafetyState.RUNNING):
        shared.quit_requested.value = True
        return False
    coordinator.hold()
    if not transition(shared, SafetyState.ARMED):
        raise RuntimeError("could not enter ARMED after publishing the quit hold")
    shared.quit_requested.value = True
    return True


def learned_policy_loop(
    shared: Any,
    runtime: Any,
    inference: PolicySpec,
    tensor_block: ObservationTensorBlock,
) -> None:
    """Run the policy-side coordinator and explicit operator gate."""
    import signal

    from dexmani_real import ASSET_DIR
    from dexmani_real.planning.planner import PlanningProfile, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
    from dexmani_real.planning.types import Pose
    from dexmani_real.policy.safety import ActionSafetyGateConfig, planner_action_safety_gate
    from dexmani_real.robot.safety import SafetyState, transition
    from dexmani_real.runtime.status import ComponentPhase, FaultCode
    from dexmani_real.shm.shared_storage import publish_component_status
    from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
    from dexmani_real.utils.rate_manager import RateManager

    stop_requested = False

    def _on_sigterm(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_sigterm)
    keyboard: KeyboardHandler | None = None
    coordinator: LearnedPolicyCoordinator | None = None
    failed = False
    try:
        publish_component_status(shared, "policy", ComponentPhase.LOADING)
        hand_enabled = "hand" in inference.actuators
        hand_feedback_enabled = bool(runtime.policy.hand_enabled)
        if hand_enabled and not hand_feedback_enabled:
            raise ValueError("PolicySpec requires hand but resolved runtime disables it")
        urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
        srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")
        planner = XArm7MotionPlanner(
            XArm7PlannerConfig(
                urdf_path=urdf_path,
                srdf_path=srdf_path,
                base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
                workspace_bounds=np.asarray(
                    [
                        [runtime.policy.workspace.x_min, runtime.policy.workspace.x_max],
                        [runtime.policy.workspace.y_min, runtime.policy.workspace.y_max],
                        [runtime.policy.workspace.z_min, runtime.policy.workspace.z_max],
                    ],
                    dtype=np.float64,
                ),
            ),
            planning_profile=PlanningProfile(),
            teleop_profile=TeleopProfile(),
            hand_dof=True,
            static_boxes=tuple(runtime.environment.static_boxes),
            table=runtime.environment.table,
        )
        gate = planner_action_safety_gate(
            ActionSafetyGateConfig(
                arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
                arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
                hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
                hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            ),
            planner=planner,
        )
        coordinator_config = LearnedCoordinatorConfig(
            coordinator_hz=float(runtime.policy.coordinator_hz),
            prepare_timeout_s=float(runtime.policy.action_prepare_timeout_s),
            apply_timeout_s=float(runtime.policy.action_apply_timeout_s),
            candidate_timeout_s=float(runtime.policy.inference_candidate_timeout_s),
            arm_feedback_max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
            hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            hand_enabled=hand_enabled,
            hand_feedback_enabled=hand_feedback_enabled,
        )
        coordinator = LearnedPolicyCoordinator(
            shared,
            inference,
            tensor_block,
            gate,
            config=coordinator_config,
        )

        ready_events: list[tuple[str, Any]] = [("arm", shared.arm_ready), ("inference", shared.inference_ready)]
        if coordinator.sources.requires_hand or hand_feedback_enabled:
            ready_events.append(("hand", shared.hand_ready))
        if coordinator.sources.requires_camera:
            ready_events.append(("camera", shared.camera_ready))
        if coordinator.sources.requires_vr:
            ready_events.append(("vr", shared.vr_ready))
        readiness_timeouts_s = dict(runtime.safety.readiness_timeouts_s)
        for name, event in ready_events:
            if not event.wait(timeout=float(readiness_timeouts_s[name])):
                raise TimeoutError(f"learned policy startup timed out waiting for {name}")

        publish_component_status(shared, "policy", ComponentPhase.WARMING_UP)
        warmup_deadline = time.monotonic() + float(readiness_timeouts_s["inference"])
        live_output_validated = False
        warmup_limiter = RateManager(coordinator_config.coordinator_hz)
        while shared.is_running.value and time.monotonic() < warmup_deadline:
            shared.set_heartbeat("policy", time.monotonic())
            coordinator.publish_snapshot()
            if coordinator.consume_candidate() is not None:
                live_output_validated = True
            if coordinator.snapshot_ready and live_output_validated and shared.inference_ready.is_set():
                break
            warmup_limiter.wait()
        if not coordinator.snapshot_ready or not live_output_validated:
            raise RuntimeError("live observation history/backend output did not validate during warmup")
        coordinator._current_joints()

        keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
        keyboard.start()
        shared.set_heartbeat("policy", time.monotonic())
        shared.policy_ready.set()
        publish_component_status(shared, "policy", ComponentPhase.READY)
        limiter = RateManager(coordinator_config.coordinator_hz)

        while shared.is_running.value and not stop_requested:
            shared.set_heartbeat("policy", time.monotonic())
            for control in keyboard.poll(timeout=0.0):
                if control is ControlSignal.EMERGENCY_STOP:
                    shared.estop_request.value = True
                    break
                if control is ControlSignal.QUIT:
                    if _hold_before_quit(shared, coordinator):
                        publish_component_status(shared, "policy", ComponentPhase.READY)
                    stop_requested = True
                    break
                if (
                    control is ControlSignal.BEGIN
                    and shared.safety_state.value == int(SafetyState.ARMED)
                    and shared.inference_ready.is_set()
                    and not coordinator.rewarm_pending
                ):
                    coordinator._current_joints()
                    coordinator.begin_new_run()
                    if transition(shared, SafetyState.RUNNING):
                        publish_component_status(shared, "policy", ComponentPhase.RUNNING)
                elif control in {ControlSignal.PAUSE, ControlSignal.STOP, ControlSignal.DISCARD} and (
                    shared.safety_state.value == int(SafetyState.RUNNING)
                ):
                    coordinator.hold()
                    if transition(shared, SafetyState.ARMED):
                        publish_component_status(shared, "policy", ComponentPhase.READY)

            if shared.estop_request.value or stop_requested:
                break
            if shared.safety_state.value == int(SafetyState.RUNNING):
                tick_result = coordinator.tick()
                if tick_result is CoordinatorTick.REWARMING:
                    publish_component_status(shared, "policy", ComponentPhase.WARMING_UP)
            else:
                coordinator.publish_snapshot()
                coordinator.consume_candidate()
                if coordinator.rewarm_pending and shared.inference_ready.is_set():
                    coordinator.complete_rewarm()
                    publish_component_status(shared, "policy", ComponentPhase.READY)
            limiter.wait()
    except Exception:
        failed = True
        logger.error("learned policy coordinator failed", exc_info=True)
        publish_component_status(
            shared,
            "policy",
            ComponentPhase.FAULT,
            fault_code=FaultCode.INFERENCE_FAILED,
            detail="coordinator exception; see process log",
        )
        shared.error_state.value = True
    finally:
        if keyboard is not None:
            keyboard.stop()
        if not failed:
            if shared.estop_request.value:
                publish_component_status(
                    shared,
                    "policy",
                    ComponentPhase.FAULT,
                    fault_code=FaultCode.ESTOP,
                    detail="policy exited after e-stop request",
                )
            elif shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
                publish_component_status(
                    shared,
                    "policy",
                    ComponentPhase.FAULT,
                    fault_code=FaultCode.COMMAND_INVALID,
                    detail="policy exited with sticky fault",
                )
            else:
                publish_component_status(shared, "policy", ComponentPhase.STOPPED)
        logger.info("learned policy coordinator exited")


class LearnedPolicyCoordinator:
    """Own snapshots, candidate IDs, SafetyGate, and publication.

    The inference process never receives this object or ``SharedStorage``.  It
    can only read the tensor block and write its candidate mailbox.
    """

    def __init__(
        self,
        shared: Any,
        inference: PolicySpec,
        tensor_block: ObservationTensorBlock,
        safety_gate: SafetyGate,
        *,
        config: LearnedCoordinatorConfig | None = None,
    ) -> None:
        self.shared = shared
        self.inference = inference
        self.tensor_block = tensor_block
        self.safety_gate = safety_gate
        self.config = config or LearnedCoordinatorConfig()
        self.sources = SharedObservationSource(shared, inference.observation)
        self._snapshots: dict[int, ObservationSnapshot] = {}
        self._last_snapshot_ns = 0
        self._last_candidate_sequence = 0
        self._last_candidate_ns = time.monotonic_ns()
        self._last_camera_generation: int | None = None
        self._hold_after_timeout = False
        self._hold_published = False
        self._last_hand_qpos_cmd: np.ndarray | None = None
        self._rewarm_pending = False
        self._rewarm_triggered = False

    @property
    def rewarm_pending(self) -> bool:
        return self._rewarm_pending

    def complete_rewarm(self) -> None:
        if not self.shared.inference_ready.is_set():
            raise RuntimeError("cannot complete policy re-warm before inference is ready")
        self._rewarm_pending = False

    @property
    def snapshot_ready(self) -> bool:
        if not self._snapshots:
            return False
        snapshot = self._snapshots[max(self._snapshots)]
        return all(bool(np.all(snapshot.valid_history_mask[name])) for name in snapshot.valid_history_mask)

    def publish_snapshot(self, *, anchor_monotonic_ns: int | None = None) -> ObservationSnapshot | None:
        """Publish one causal snapshot, tagging camera resets with a new run."""
        now_ns = time.monotonic_ns() if anchor_monotonic_ns is None else int(anchor_monotonic_ns)
        period_ns = int(round(1e9 / self.inference.observation.control_hz))
        if self._last_snapshot_ns and now_ns < self._last_snapshot_ns + period_ns:
            return None
        snapshot = self.sources.build(anchor_monotonic_ns=now_ns)
        if self._last_camera_generation is None:
            self._last_camera_generation = snapshot.camera_generation
        elif snapshot.camera_generation != self._last_camera_generation:
            from dexmani_real.robot.safety import SafetyState, transition

            self._last_camera_generation = snapshot.camera_generation
            current_safety = SafetyState(int(self.shared.safety_state.value))
            run_generation = self.begin_new_run(now_monotonic_ns=now_ns)
            snapshot = replace(snapshot, run_generation=run_generation)
            self._snapshots[snapshot.observation_id] = snapshot
            if current_safety in (SafetyState.ARMED, SafetyState.RUNNING):
                self.hold(now_monotonic_ns=now_ns, invalidate_run=False)
            if current_safety is SafetyState.RUNNING and not transition(self.shared, SafetyState.ARMED):
                raise RuntimeError("camera restart could not place policy in ARMED")
            self.shared.inference_ready.clear()
            self._rewarm_pending = True
            self._rewarm_triggered = True
        self.tensor_block.write(snapshot)
        self._snapshots[snapshot.observation_id] = snapshot
        while len(self._snapshots) > _SNAPSHOT_CACHE_SIZE:
            del self._snapshots[min(self._snapshots)]
        self._last_snapshot_ns = now_ns
        return snapshot

    def _allocate_action_id(self) -> int:
        with self.shared.arm_command_seq.get_lock():
            action_id = int(self.shared.arm_command_seq.value) + 1
            self.shared.arm_command_seq.value = action_id
        return action_id

    def consume_candidate(self, *, now_monotonic_ns: int | None = None) -> ActionCandidate | None:
        """Return the newest unseen candidate from the active run when fresh."""
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        result = self.shared.inference_candidate_ring.read_latest()
        if result is None:
            return None
        data, _publish_ns, sequence = result
        if sequence <= self._last_candidate_sequence:
            return None
        self._last_candidate_sequence = sequence
        raw = decode_candidate(data)
        snapshot = self._snapshots.get(raw.observation_id)
        if snapshot is None:
            return None
        active_run_generation = int(self.shared.run_generation.value)
        if raw.run_generation != active_run_generation or snapshot.run_generation != active_run_generation:
            return None
        if now_ns - snapshot.anchor_monotonic_ns > int(self.inference.action.deadline_s * 1e9):
            self._hold_after_timeout = True
            return None
        if not self.config.hand_enabled and raw.hand_qpos is not None:
            raise ValueError("backend produced a hand action while the hand capability is disabled")
        candidate = replace(raw, action_id=self._allocate_action_id())
        self._last_candidate_ns = now_ns
        self._hold_after_timeout = False
        self._hold_published = False
        return candidate

    def _current_joints(self) -> tuple[np.ndarray, np.ndarray]:
        arm_result = self.shared.arm_state_ring.read_latest()
        hand_result = self.shared.hand_state_ring.read_latest() if self.config.hand_feedback_enabled else None
        now_ns = time.monotonic_ns()
        if arm_result is None or (self.config.hand_feedback_enabled and hand_result is None):
            self._raise_feedback_fault("fresh feedback is required for the arm and configured hand geometry")
        arm_record = arm_result[0][0]
        arm_issue = validate_arm_feedback(
            connected=bool(arm_record["connected"]),
            state_valid=bool(arm_record["state_valid"]),
            source_monotonic_ns=int(arm_record["source_monotonic_ns"]),
            now_monotonic_ns=now_ns,
            max_age_s=self.config.arm_feedback_max_age_s,
            qpos=np.asarray(arm_record["qpos"]),
            qvel=np.asarray(arm_record["qvel"]),
            eef_pos=np.asarray(arm_record["eef_pos"]),
            eef_rot6d=np.asarray(arm_record["eef_rot6d"]),
        )
        if arm_issue is None and int(arm_record["error_code"]) != 0:
            arm_issue = f"arm controller error C{int(arm_record['error_code'])}"
        if arm_issue is not None:
            self._raise_feedback_fault(arm_issue)
        arm_qpos = np.asarray(arm_record["qpos"], dtype=np.float64).copy()
        if hand_result is None:
            hand_qpos = np.asarray(self.shared.hand_home_qpos_rad, dtype=np.float64).copy()
            if hand_qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(hand_qpos)):
                self._raise_feedback_fault(
                    f"configured hand home geometry must be finite with shape {HAND_JOINT_SHAPE}"
                )
        else:
            hand_record = hand_result[0][0]
            hand_issue = validate_hand_feedback(
                connected=bool(hand_record["connected"]),
                error_state=bool(hand_record["error_state"]),
                state_valid=bool(hand_record["state_valid"]),
                send_healthy=bool(hand_record["send_healthy"]),
                read_healthy=bool(hand_record["read_healthy"]),
                source_monotonic_ns=int(hand_record["source_monotonic_ns"]),
                now_monotonic_ns=now_ns,
                max_age_s=self.config.hand_feedback_max_age_s,
                qpos=np.asarray(hand_record["qpos"]),
            )
            if hand_issue is not None:
                self._raise_feedback_fault(hand_issue)
            hand_qpos = np.asarray(hand_record["qpos"], dtype=np.float64).copy()
        return arm_qpos, hand_qpos

    def _raise_feedback_fault(self, reason: str) -> NoReturn:
        """Invalidate pending actions before surfacing unusable measured geometry."""
        run_generation = advance_run_generation(self.shared)
        self._last_candidate_sequence = int(self.shared.inference_candidate_ring.latest_sequence)
        self._hold_after_timeout = False
        self._hold_published = False
        raise RuntimeError(f"{reason}; invalidated run generation {run_generation}")

    def _publish_candidate(self, candidate: ActionCandidate, *, now_ns: int) -> CoordinatorTick:
        snapshot = self._snapshots.get(candidate.observation_id)
        if snapshot is None:
            return CoordinatorTick.REJECTED
        if now_ns > candidate.valid_until_monotonic_ns:
            return CoordinatorTick.REJECTED
        current_arm, current_hand = self._current_joints()
        result = self.safety_gate.validate(
            candidate,
            current_arm_qpos=current_arm,
            current_hand_qpos=current_hand,
            dt_s=self.inference.action.dt_s,
            run_generation=int(self.shared.run_generation.value),
        )
        if not result.accepted or result.candidate is None:
            logger.warning("learned candidate rejected: %s", result.reason)
            return CoordinatorTick.REJECTED
        if not send_command(self.shared, result.candidate, prepare_timeout_s=self.config.prepare_timeout_s):
            raise TimeoutError("learned action publish failed")
        if result.candidate.hand_qpos is not None:
            self._last_hand_qpos_cmd = np.asarray(result.candidate.hand_qpos, dtype=np.float64).copy()
        return CoordinatorTick.PUBLISHED

    def _publish_coordinated_hold(self, *, now_ns: int) -> None:
        if self._hold_published or not self._snapshots:
            return
        snapshot = self._snapshots[max(self._snapshots)]
        current_arm, _current_hand = self._current_joints()
        action_id = self._allocate_action_id()
        hold = ActionCandidate(
            observation_id=snapshot.observation_id,
            run_generation=int(self.shared.run_generation.value),
            action_id=action_id,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns + int(float(self.shared.action_lead_time_s) * 1e9),
            valid_until_monotonic_ns=(
                now_ns + int((float(self.shared.action_lead_time_s) + self.inference.action.dt_s) * 1e9)
            ),
            arm_qpos=current_arm,
            # The latest-wins hand command remains active. Never republish
            # measured feedback as a target during a coordinated hold.
            hand_qpos=None,
            is_hold=True,
        )
        if self._publish_candidate(hold, now_ns=now_ns) is not CoordinatorTick.PUBLISHED:
            raise RuntimeError("coordinated hold was rejected")
        self._hold_published = True

    def begin_new_run(self, *, now_monotonic_ns: int | None = None) -> int:
        """Drop old proposals before an explicit ARMED→RUNNING transition."""
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        run_generation = advance_run_generation(self.shared)
        self._last_candidate_sequence = int(self.shared.inference_candidate_ring.latest_sequence)
        self._last_candidate_ns = now_ns
        self._hold_after_timeout = False
        self._hold_published = False
        return run_generation

    def hold(self, *, now_monotonic_ns: int | None = None, invalidate_run: bool = True) -> None:
        """Invalidate pending actions, then publish a fresh measured arm-only hold."""
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        if invalidate_run:
            self.begin_new_run(now_monotonic_ns=now_ns)
        else:
            self._last_candidate_sequence = int(self.shared.inference_candidate_ring.latest_sequence)
            self._last_candidate_ns = now_ns
            self._hold_after_timeout = False
            self._hold_published = False
        self._publish_coordinated_hold(now_ns=now_ns)

    def tick(self, *, now_monotonic_ns: int | None = None) -> CoordinatorTick:
        """Run one non-blocking coordinator iteration."""
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        snapshot = self.publish_snapshot(anchor_monotonic_ns=now_ns)
        if self._rewarm_triggered:
            self._rewarm_triggered = False
            return CoordinatorTick.REWARMING
        candidate = self.consume_candidate(now_monotonic_ns=now_ns)
        if candidate is not None:
            result = self._publish_candidate(candidate, now_ns=now_ns)
            if result is CoordinatorTick.REJECTED:
                self.hold(now_monotonic_ns=now_ns)
                self._hold_after_timeout = True
                return CoordinatorTick.HELD
            return result
        if self._hold_after_timeout or now_ns - self._last_candidate_ns > int(self.config.candidate_timeout_s * 1e9):
            if not self._hold_published:
                self.hold(now_monotonic_ns=now_ns)
            self._hold_after_timeout = True
            return CoordinatorTick.HELD
        return CoordinatorTick.SNAPSHOT if snapshot is not None else CoordinatorTick.IDLE
