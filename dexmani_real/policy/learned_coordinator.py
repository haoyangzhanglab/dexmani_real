"""Policy-side scheduler for an isolated learned-policy inference worker."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, NoReturn

import numpy as np

from dexmani_real.config.defaults import policy, safety
from dexmani_real.utils.schema import HAND_JOINT_SHAPE
from dexmani_real.policy.safety import (
    SafetyGate,
    advance_policy_epoch,
    send_command,
)
from dexmani_real.policy.inference_process import decode_candidate
from dexmani_real.policy.observation_sources import SharedObservationSource
from dexmani_real.policy.runtime import ActionCandidate, ActionChunk, ObservationSnapshot
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
        )
        gate = planner_action_safety_gate(
            ActionSafetyGateConfig(
                arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
                arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
                hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
                hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
                arm_max_velocity_rad_s=float(np.deg2rad(runtime.arm.max_joint_velocity_deg_per_s)),
                hand_max_velocity_rad_s=(
                    float(runtime.hand.max_delta_rad) * inference.observation.control_hz
                    if runtime.hand.max_delta_rad is not None
                    else float(np.deg2rad(runtime.hand.safety_gate_max_velocity_deg_per_s))
                ),
                observation_max_age_s=max(modality.max_age_s for modality in inference.observation.modalities),
                require_geometry_checks=True,
            ),
            planner=planner,
            table_z_surface_m=float(runtime.arm.table_z_surface_m),
            hand_safety_margin_m=float(runtime.arm.hand_safety_margin_m),
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
            shared.policy_heartbeat_s.value = time.monotonic()
            coordinator.publish_snapshot()
            if coordinator.consume_candidate_chunk() is not None:
                live_output_validated = True
                coordinator.scheduler.reset()
            if coordinator.snapshot_ready and live_output_validated and shared.inference_ready.is_set():
                break
            warmup_limiter.wait()
        if not coordinator.snapshot_ready or not live_output_validated:
            raise RuntimeError("live observation history/backend output did not validate during warmup")
        coordinator._current_joints()

        keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
        keyboard.start()
        shared.policy_heartbeat_s.value = time.monotonic()
        shared.policy_ready.set()
        publish_component_status(shared, "policy", ComponentPhase.READY)
        limiter = RateManager(coordinator_config.coordinator_hz)

        while shared.is_running.value and not stop_requested:
            shared.policy_heartbeat_s.value = time.monotonic()
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
                    coordinator.begin_new_epoch()
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
                coordinator.consume_candidate_chunk()
                coordinator.scheduler.reset()
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


class JointActionScheduler:
    """Policy-owned chunk overlap, replacement, expiry, and per-tick scheduler."""

    def __init__(self, action_spec: Any) -> None:
        self.action_spec = action_spec
        self._future: list[Any] = []
        self._all_late = False

    def submit(self, chunk: Any, *, now_monotonic_ns: int | None = None) -> None:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        self._future.clear()
        accepted = [
            step
            for step in chunk.steps
            if step.target_monotonic_ns > now_ns and step.valid_until_monotonic_ns >= now_ns
        ]
        self._all_late = not accepted
        self._future.extend(accepted)
        self._future.sort(key=lambda step: (step.target_monotonic_ns, step.action_id))

    def reset(self) -> None:
        self._future.clear()
        self._all_late = False

    @property
    def pending(self) -> tuple[Any, ...]:
        return tuple(self._future)

    @property
    def all_late(self) -> bool:
        return self._all_late

    def pop_ready(self, *, lead_time_s: float, now_monotonic_ns: int | None = None) -> Any | None:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        self._future = [
            s for s in self._future if s.target_monotonic_ns > now_ns and s.valid_until_monotonic_ns >= now_ns
        ]
        if not self._future:
            self._all_late = True
            return None
        ready = [s for s in self._future if s.target_monotonic_ns <= now_ns + int(lead_time_s * 1e9)]
        if not ready:
            return None
        selected = ready[0]
        self._future = [s for s in self._future if s.action_id != selected.action_id]
        return selected

    def make_coordinated_hold(
        self,
        *,
        template: Any,
        arm_qpos: np.ndarray,
        hand_qpos: np.ndarray | None,
        action_id: int,
        now_monotonic_ns: int | None = None,
    ) -> Any:
        from dexmani_real.policy.runtime import ActionCandidate

        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        return ActionCandidate(
            observation_id=template.observation_id,
            session_generation=template.session_generation,
            policy_epoch=template.policy_epoch,
            action_id=int(action_id),
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns + int(2 * self.action_spec.dt_s * 1e9),
            valid_until_monotonic_ns=now_ns + int(self.action_spec.dt_s * 1e9),
            arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
            hand_qpos=None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64),
            chunk_id=template.chunk_id,
            step_index=template.step_index,
            is_hold=True,
        )


class LearnedPolicyCoordinator:
    """Own snapshots, candidate IDs, SafetyGate, scheduling, and publication.

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
        if inference.action.chunk_length > int(shared.inference_candidate_ring.maxlen):
            raise ValueError("ActionSpec chunk exceeds inference candidate ring capacity")
        self.shared = shared
        self.inference = inference
        self.tensor_block = tensor_block
        self.safety_gate = safety_gate
        self.config = config or LearnedCoordinatorConfig()
        self.sources = SharedObservationSource(shared, inference.observation)
        self.scheduler = JointActionScheduler(inference.action)
        self._snapshots: dict[int, ObservationSnapshot] = {}
        self._last_snapshot_ns = 0
        self._last_candidate_sequence = 0
        self._last_candidate_ns = time.monotonic_ns()
        self._last_camera_generation: int | None = None
        self._hold_after_timeout = False
        self._hold_published = False
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
            self._snapshots[snapshot.observation_id] = snapshot
            current_safety = SafetyState(int(self.shared.safety_state.value))
            self.begin_new_epoch(now_monotonic_ns=now_ns)
            if current_safety in (SafetyState.ARMED, SafetyState.RUNNING):
                self.hold(now_monotonic_ns=now_ns, invalidate_epoch=False)
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

    def _allocate_action_ids(self, count: int) -> list[int]:
        with self.shared.arm_command_seq.get_lock():
            first = int(self.shared.arm_command_seq.value) + 1
            self.shared.arm_command_seq.value = first + count - 1
        return list(range(first, first + count))

    def consume_candidate_chunk(self, *, now_monotonic_ns: int | None = None) -> ActionChunk | None:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        history = self.shared.inference_candidate_ring.get_last_k(self.shared.inference_candidate_ring.maxlen)
        unseen = [
            (data, sequence) for data, _publish_ns, sequence in history if sequence > self._last_candidate_sequence
        ]
        if not unseen:
            return None

        groups: dict[tuple[int, int], list[tuple[ActionCandidate, int, int]]] = defaultdict(list)
        for data, sequence in unseen:
            candidate, chunk_length = decode_candidate(data)
            groups[(candidate.observation_id, candidate.chunk_id)].append((candidate, chunk_length, sequence))

        complete: list[tuple[int, list[ActionCandidate]]] = []
        for entries in groups.values():
            lengths = {length for _candidate, length, _sequence in entries}
            if len(lengths) != 1:
                continue
            length = lengths.pop()
            if len(entries) != length:
                continue
            by_step = {candidate.step_index: candidate for candidate, _length, _sequence in entries}
            if set(by_step) != set(range(length)):
                continue
            complete.append(
                (max(sequence for _candidate, _length, sequence in entries), [by_step[i] for i in range(length)])
            )
        if not complete:
            return None

        newest_sequence, raw_steps = max(complete, key=lambda item: item[0])
        snapshot = self._snapshots.get(raw_steps[0].observation_id)
        if snapshot is None:
            self._last_candidate_sequence = newest_sequence
            return None
        if now_ns - snapshot.anchor_monotonic_ns > int(self.inference.action.deadline_s * 1e9):
            self._last_candidate_sequence = newest_sequence
            self._hold_after_timeout = True
            return None

        action_ids = self._allocate_action_ids(len(raw_steps))
        chunk_id = action_ids[0]
        normalized: list[ActionCandidate] = []
        for index, (raw, action_id) in enumerate(zip(raw_steps, action_ids)):
            if not self.config.hand_enabled and raw.hand_qpos is not None:
                raise ValueError("backend produced a hand action while the hand capability is disabled")
            normalized.append(
                replace(
                    raw,
                    session_generation=int(self.shared.session_generation.value),
                    policy_epoch=int(self.shared.policy_epoch.value),
                    action_id=action_id,
                    chunk_id=chunk_id,
                    step_index=index,
                )
            )
        chunk = ActionChunk(chunk_id, tuple(normalized))
        self.scheduler.submit(chunk, now_monotonic_ns=now_ns)
        self._last_candidate_sequence = newest_sequence
        self._last_candidate_ns = now_ns
        self._hold_after_timeout = self.scheduler.all_late
        self._hold_published = False
        return chunk

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
                qpos_stale=bool(hand_record["qpos_stale"]),
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
        epoch = advance_policy_epoch(self.shared)
        self.scheduler.reset()
        self._last_candidate_sequence = int(self.shared.inference_candidate_ring.latest_sequence)
        self._hold_after_timeout = False
        self._hold_published = False
        raise RuntimeError(f"{reason}; invalidated policy epoch {epoch}")

    def _publish_candidate(self, candidate: ActionCandidate, *, now_ns: int) -> CoordinatorTick:
        snapshot = self._snapshots.get(candidate.observation_id)
        if snapshot is None:
            return CoordinatorTick.REJECTED
        current_arm, current_hand = self._current_joints()
        result = self.safety_gate.evaluate(
            candidate,
            snapshot=snapshot,
            current_arm_qpos=current_arm,
            current_hand_qpos=current_hand,
            expected_session_generation=int(self.shared.session_generation.value),
            expected_policy_epoch=int(self.shared.policy_epoch.value),
            now_monotonic_ns=now_ns,
            dt_s=self.inference.action.dt_s,
        )
        if not result.accepted or result.candidate is None:
            logger.warning("learned candidate rejected: %s", result.reason)
            return CoordinatorTick.REJECTED
        if not send_command(self.shared, result.candidate, prepare_timeout_s=self.config.prepare_timeout_s):
            raise TimeoutError("learned action publish failed")
        return CoordinatorTick.PUBLISHED

    def _publish_coordinated_hold(self, *, now_ns: int) -> None:
        if self._hold_published or not self._snapshots:
            return
        snapshot = self._snapshots[max(self._snapshots)]
        current_arm, current_hand = self._current_joints()
        action_id = self._allocate_action_ids(1)[0]
        target_ns = now_ns + int(float(self.shared.action_lead_time_s) * 1e9)
        hold = ActionCandidate(
            observation_id=snapshot.observation_id,
            session_generation=int(self.shared.session_generation.value),
            policy_epoch=int(self.shared.policy_epoch.value),
            action_id=action_id,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=target_ns,
            valid_until_monotonic_ns=target_ns + int(self.inference.action.dt_s * 1e9),
            arm_qpos=current_arm,
            hand_qpos=current_hand if self.config.hand_enabled else None,
            chunk_id=action_id,
            step_index=0,
            is_hold=True,
        )
        if self._publish_candidate(hold, now_ns=now_ns) is not CoordinatorTick.PUBLISHED:
            raise RuntimeError("coordinated hold was rejected")
        self._hold_published = True

    def begin_new_epoch(self, *, now_monotonic_ns: int | None = None) -> int:
        """Drop old proposals before an explicit ARMED→RUNNING transition."""
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        epoch = advance_policy_epoch(self.shared)
        self.scheduler.reset()
        self._last_candidate_sequence = int(self.shared.inference_candidate_ring.latest_sequence)
        self._last_candidate_ns = now_ns
        self._hold_after_timeout = False
        self._hold_published = False
        return epoch

    def hold(self, *, now_monotonic_ns: int | None = None, invalidate_epoch: bool = True) -> None:
        """Invalidate pending actions, then publish a hold from fresh measured joints."""
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        if invalidate_epoch:
            self.begin_new_epoch(now_monotonic_ns=now_ns)
        else:
            self.scheduler.reset()
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
        self.consume_candidate_chunk(now_monotonic_ns=now_ns)
        ready = self.scheduler.pop_ready(lead_time_s=float(self.shared.action_lead_time_s), now_monotonic_ns=now_ns)
        if ready is not None:
            result = self._publish_candidate(ready, now_ns=now_ns)
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
