"""Thin learned-policy scheduler and physical command executor.

Inference publishes flat :class:`Prediction` records.  This module owns the
RUNNING episode, selects at most one due action per control-grid slot, decodes
that action, applies the shared physical safety boundary, and publishes one
coupled arm/hand command.  Hardware SDKs remain in their worker processes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    PUBLISH_REASON_ESTOP,
    PUBLISH_REASON_EXPIRED,
    PUBLISH_REASON_FAULT,
    PUBLISH_REASON_GENERATION,
    PUBLISH_REASON_RUNTIME_STOPPED,
    PUBLISH_REASON_SAFETY_STATE,
    PreparedCommand,
    PublishResult,
    build_action_candidate,
    command_publishability_reason,
    prepare_command,
    publish_command,
)
from dexmani_real.control.safety_gate import SafetyGate
from dexmani_real.deployment.evaluation import (
    EvaluationOutcome,
    RolloutRecordingConfig,
    evaluation_outcome_stop_reason,
    write_rollout_result,
)
from dexmani_real.deployment.metrics import PolicyStats, flush_every
from dexmani_real.deployment.prediction import Prediction
from dexmani_real.deployment.timing import (
    first_future_step_index,
    next_periodic_deadline_ns,
)
from dexmani_real.ipc.causal import (
    read_camera_frame_causal,
    read_causal_structured_frame,
)
from dexmani_real.ipc.channels import (
    RuntimeChannels,
    read_arm_state_dict,
    read_hand_state_dict,
)
from dexmani_real.ipc.schema import (
    MAX_POLICY_ACTION_DIM,
    MAX_PREDICTION_STEPS,
    PREDICTION_DTYPE,
)
from dexmani_real.planning import (
    OnlineIKConfig,
    Pose,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.planning.paths import (
    WORKSPACE_BOUNDS_TOLERANCE_M,
    interpolate_waypoints,
    wrap_nearest_equivalent,
)
from dexmani_real.recording.client import (
    RecorderClient,
    RecorderStopResult,
)
from dexmani_real.recording.sample import (
    EpisodeAction,
    EpisodeState,
    build_episode_state,
)
from dexmani_real.robot.model import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    StopRequest,
    begin_requested_motion,
    read_run_state_snapshot,
    revoke_motion,
)
from dexmani_real.sensor.camera.worker import CameraHealth
from dexmani_real.utils.feedback import (
    FeedbackIssueCode,
    diagnose_arm_feedback,
    diagnose_hand_feedback,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)

_POLICY_WORKSPACE_INTERPOLATION_MAX_STEP_RAD = 0.02
_JOINT_ACTION_DIM = 19
_EE_ACTION_DIM = 21
_RECORDING_TACTILE_MAX_AGE_NS = 250_000_000
_RECORD_FRAME_OK = 0
_RECORD_FRAME_HELD = 1
_RECORD_FRAME_IK_FAIL = 2
_RECORD_FRAME_SAFETY_REJECT = 3


class _RejectKind(Enum):
    """Explicit attribution of a policy-step rejection, never a reason string."""

    IK = auto()
    SAFETY = auto()


@dataclass
class _CommandProgress:
    """Independent latest-wins acceptance watermarks for one generation."""

    generation: int | None = None
    latest_published_action_id: int | None = None
    arm_accepted_action_id: int | None = None
    hand_accepted_action_id: int | None = None
    hand_last_sdk_setpoint_accepted_ns: int | None = None
    arm_last_progress_ns: int | None = None
    hand_last_progress_ns: int | None = None

    def reset(self, generation: int | None) -> None:
        self.generation = generation
        self.latest_published_action_id = None
        self.arm_accepted_action_id = None
        self.hand_accepted_action_id = None
        self.hand_last_sdk_setpoint_accepted_ns = None
        self.arm_last_progress_ns = None
        self.hand_last_progress_ns = None

    def observe(
        self,
        *,
        generation: int,
        arm_action_id: int | None,
        hand_action_id: int | None,
        now_ns: int,
        timeout_ns: int,
        hand_setpoint_accepted_ns: int | None = None,
    ) -> str | None:
        if generation != self.generation:
            return "command progress generation does not match active run"
        if arm_action_id is not None:
            previous_arm_id = self.arm_accepted_action_id
            if arm_action_id < 0:
                return "arm command progress is negative"
            if previous_arm_id is not None and arm_action_id < previous_arm_id:
                return "arm command progress regressed"
            self.arm_accepted_action_id = arm_action_id
            if previous_arm_id is None or arm_action_id > previous_arm_id:
                self.arm_last_progress_ns = now_ns

        if hand_action_id is not None:
            previous_hand_id = self.hand_accepted_action_id
            if hand_action_id < 0:
                return "hand command progress is negative"
            if previous_hand_id is not None and hand_action_id < previous_hand_id:
                return "hand command progress regressed"
            self.hand_accepted_action_id = hand_action_id

        if hand_setpoint_accepted_ns is not None:
            previous_setpoint_ns = self.hand_last_sdk_setpoint_accepted_ns
            if hand_setpoint_accepted_ns < 0:
                return "hand SDK setpoint progress is negative"
            if hand_setpoint_accepted_ns > now_ns:
                return "hand SDK setpoint progress is in the future"
            if (
                previous_setpoint_ns is not None
                and hand_setpoint_accepted_ns < previous_setpoint_ns
            ):
                return "hand SDK setpoint progress regressed"
            self.hand_last_sdk_setpoint_accepted_ns = hand_setpoint_accepted_ns
            if (
                previous_setpoint_ns is not None
                and hand_setpoint_accepted_ns > previous_setpoint_ns
            ):
                self.hand_last_progress_ns = hand_setpoint_accepted_ns

        latest = self.latest_published_action_id
        if latest is None:
            return None
        for worker in ("arm", "hand"):
            accepted = getattr(self, f"{worker}_accepted_action_id")
            progressed_ns = getattr(self, f"{worker}_last_progress_ns")
            if accepted is None or progressed_ns is None:
                continue
            if accepted < latest and now_ns - progressed_ns > timeout_ns:
                return f"{worker} worker command progress timeout"
        return None

    def record_publication(self, action_id: int, published_ns: int) -> None:
        previous_latest = self.latest_published_action_id
        if previous_latest is not None and action_id <= previous_latest:
            raise RuntimeError("published action IDs must increase")
        for worker in ("arm", "hand"):
            accepted = getattr(self, f"{worker}_accepted_action_id")
            if accepted is None:
                raise RuntimeError(f"{worker} command progress baseline is unavailable")
            if previous_latest is None or accepted >= previous_latest:
                setattr(self, f"{worker}_last_progress_ns", published_ns)
        self.latest_published_action_id = action_id


def prediction_from_record(record: np.void) -> Prediction:
    """Deserialize one IPC prediction and validate its timing metadata."""
    if not isinstance(record, np.void) or record.dtype != PREDICTION_DTYPE:
        raise ValueError("prediction record has an invalid IPC schema")
    num_steps = int(record["num_steps"])
    if not 0 < num_steps <= MAX_PREDICTION_STEPS:
        raise ValueError("prediction has an invalid num_steps")
    action_dim = int(record["action_dim"])
    if action_dim not in {_JOINT_ACTION_DIM, _EE_ACTION_DIM}:
        raise ValueError(
            "prediction action_dim must be a supported policy representation "
            f"(<= {MAX_POLICY_ACTION_DIM})"
        )
    return Prediction(
        run_generation=int(record["run_generation"]),
        source_monotonic_ns=int(record["source_monotonic_ns"]),
        logical_step_monotonic_ns=int(record["logical_step_monotonic_ns"]),
        inference_latency_ms=record["inference_latency_ms"],
        observation_age_ms=record["observation_age_ms"],
        observation_skew_ms=record["observation_skew_ms"],
        actions=np.array(
            record["actions"][:num_steps, :action_dim], dtype=np.float64, copy=True
        ),
    )


def read_latest_prediction(shared: RuntimeChannels) -> tuple[Prediction, int] | None:
    """Read the newest prediction plus its latest-wins ring sequence."""
    result = shared.prediction_ring.read_latest()
    if result is None:
        return None
    return prediction_from_record(result[0][0]), int(result[2])


def _command_watchdog_reason(
    *,
    now_ns: int,
    run_started_ns: int,
    last_valid_command_ns: int | None,
    first_command_timeout_ns: int,
    command_silence_timeout_ns: int,
) -> str | None:
    if last_valid_command_ns is None:
        if now_ns - run_started_ns > first_command_timeout_ns:
            return "first command timeout"
        return None
    if now_ns - last_valid_command_ns > command_silence_timeout_ns:
        return "command silence timeout"
    return None


def _rollout_timeout_result(mode: str, reason: str) -> tuple[EvaluationOutcome, str]:
    """Map the three rollout time limits to the run/eval metadata contract."""
    if mode not in {"run", "eval"}:
        raise ValueError("rollout mode must be run or eval")
    events = {
        "run time limit": "timeout",
        "first command timeout": "first_command_timeout",
        "command silence timeout": "command_silence_timeout",
    }
    event = events[reason]
    if mode == "eval":
        return EvaluationOutcome.FAILURE, f"eval:failure:{event}"
    if event == "timeout":
        return EvaluationOutcome.STOPPED, "run:stopped:timeout"
    return EvaluationOutcome.INVALID, f"run:invalid:{event}"


def _advance_control_grid_ns(due_ns: int, terminal_ns: int, step_dt_ns: int) -> int:
    """Advance one slot without accumulating jitter or catching up."""
    lateness_ns = max(0, terminal_ns - due_ns)
    if lateness_ns >= step_dt_ns:
        return terminal_ns + step_dt_ns
    return due_ns + step_dt_ns


def _build_policy_planner(runtime: ExperimentConfig) -> XArm7MotionPlanner:
    """Build kinematics-only policy IK; realtime collision checks stay disabled."""
    return XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=np.asarray(
                runtime.policy.workspace.as_tuple(), dtype=np.float64
            ),
        ),
        teleop_profile=OnlineIKConfig(
            max_pose_error_pos_m=float(runtime.policy.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(runtime.policy.ik_max_pose_error_rot_rad),
            check_self_collision=False,
        ),
        hand_dof=False,
    )


def _build_policy_workspace_check(
    runtime: ExperimentConfig,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Return the reject-only interpolated joint-policy workspace predicate."""
    bounds = np.asarray(runtime.policy.workspace.as_tuple(), dtype=np.float64)
    arm_fk = make_arm_fk()

    def is_workspace_segment_safe(
        start_arm_qpos: np.ndarray, end_arm_qpos: np.ndarray
    ) -> bool:
        path = interpolate_waypoints(
            np.stack([start_arm_qpos, end_arm_qpos]),
            max_step=_POLICY_WORKSPACE_INTERPOLATION_MAX_STEP_RAD,
        )
        for arm_qpos in path:
            eef_position_base, _ = arm_fk.compute(arm_qpos)
            position = np.asarray(eef_position_base, dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("arm FK returned an invalid workspace position")
            if np.any(position < bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M) or np.any(
                position > bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M
            ):
                return False
        return True

    return is_workspace_segment_safe


def _build_policy_safety_gate(runtime: ExperimentConfig) -> SafetyGate:
    return SafetyGate(
        arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
        hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
        workspace_check=_build_policy_workspace_check(runtime),
        max_hand_delta_rad=float(runtime.policy.hand_max_action_jump_rad),
        endpoint_delta_tolerance_rad=float(runtime.policy.endpoint_delta_tolerance_rad),
    )


def decode_policy_action(
    action: np.ndarray,
    policy_spec: Any,
    current_arm_qpos: np.ndarray,
    *,
    previous_arm_command_qpos: np.ndarray | None,
    planner: XArm7MotionPlanner | None,
) -> tuple[np.ndarray | None, np.ndarray, str]:
    """Interpret one already-validated flat action and perform EE IK when needed.

    The inference boundary owns flat shape and finite-value validation.  A
    representational or IK failure rejects this policy step; it does not imply a
    hardware fault.
    """
    reference = (
        current_arm_qpos
        if previous_arm_command_qpos is None
        else previous_arm_command_qpos
    )
    if policy_spec.action_key == "action":
        return np.asarray(action[:7], dtype=np.float64), action[7:19], ""

    hand_qpos = action[9:21]
    if planner is None:
        return None, hand_qpos, "EE planner is unavailable"
    try:
        target = Pose(
            p=action[:3],
            q=rot6d_to_quat_wxyz(action[3:9]),
        )
        result = planner.solve_teleop_ik(target, current_arm_qpos, reference)
    except Exception as exc:
        return None, hand_qpos, f"EE IK failed: {type(exc).__name__}"
    if not result.success or result.qpos is None:
        return None, hand_qpos, result.reason or "EE IK found no usable solution"
    return np.asarray(result.qpos, dtype=np.float64), hand_qpos, ""


def _validate_policy_arm_action(
    target_arm_qpos: np.ndarray,
    reference_arm_qpos: np.ndarray,
    runtime: ExperimentConfig,
) -> tuple[np.ndarray | None, str | None]:
    """Canonicalize then reject, never clip, one learned-policy arm endpoint."""
    lower = np.asarray(runtime.arm.joint_limit_lower, dtype=np.float64)
    upper = np.asarray(runtime.arm.joint_limit_upper, dtype=np.float64)
    canonical = wrap_nearest_equivalent(
        target_arm_qpos,
        reference_arm_qpos,
        runtime.arm.joint_limit_lower,
        runtime.arm.joint_limit_upper,
    )
    if np.any(canonical < lower) or np.any(canonical > upper):
        return None, "arm joint limit violation"

    delta = canonical - reference_arm_qpos
    limit = float(runtime.policy.arm_max_action_jump_rad)
    if np.any(np.abs(delta) > limit):
        return None, "arm action jump exceeds limit"
    return canonical, None


def _read_command_progress(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    *,
    now_ns: int,
) -> tuple[int | None, int | None, int | None, str | None]:
    """Read healthy worker watermarks; stale feedback means no new progress."""
    arm_state = read_arm_state_dict(shared)
    hand_state = read_hand_state_dict(shared)
    arm_action_id: int | None = None
    hand_action_id: int | None = None
    hand_setpoint_accepted_ns: int | None = None
    if arm_state is not None:
        issue = diagnose_arm_feedback(
            connected=bool(arm_state["connected"]),
            error_code=int(arm_state["error_code"]),
            state_valid=bool(arm_state["state_valid"]),
            source_monotonic_ns=int(arm_state["source_monotonic_ns"]),
            now_monotonic_ns=now_ns,
            max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
            qpos=np.asarray(arm_state["qpos"], dtype=np.float64),
            qvel=np.asarray(arm_state["qvel"], dtype=np.float64),
        )
        if issue is None:
            arm_action_id = int(arm_state["last_cmd_seq"])
        elif issue.code is not FeedbackIssueCode.STALE:
            return None, None, None, f"fatal arm feedback: {issue.code.value}"
    if hand_state is not None:
        issue = diagnose_hand_feedback(
            connected=bool(hand_state["connected"]),
            state_valid=bool(hand_state["state_valid"]),
            source_monotonic_ns=int(hand_state["source_monotonic_ns"]),
            now_monotonic_ns=now_ns,
            max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            qpos=np.asarray(hand_state["qpos"], dtype=np.float64),
        )
        if issue is None:
            hand_action_id = int(hand_state["accepted_target_action_id"])
            hand_setpoint_accepted_ns = int(
                hand_state["last_sdk_setpoint_accepted_monotonic_ns"]
            )
        elif issue.code is not FeedbackIssueCode.STALE:
            return None, None, None, f"fatal hand feedback: {issue.code.value}"
    return arm_action_id, hand_action_id, hand_setpoint_accepted_ns, None


def _physical_start_pose_rejection(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    *,
    execute: bool,
) -> str | None:
    """Return why B cannot open a physical epoch, or ``None`` at arm home."""
    if not execute:
        return None
    if not bool(shared.physical_home_completed.value):
        return "physical home sequence has not completed; press H before B"
    arm_state = read_arm_state_dict(shared)
    if arm_state is None:
        return "arm feedback unavailable; press H after feedback is ready"
    issue = diagnose_arm_feedback(
        connected=bool(arm_state["connected"]),
        error_code=int(arm_state["error_code"]),
        state_valid=bool(arm_state["state_valid"]),
        source_monotonic_ns=int(arm_state["source_monotonic_ns"]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
        qpos=np.asarray(arm_state["qpos"], dtype=np.float64),
        qvel=np.asarray(arm_state["qvel"], dtype=np.float64),
    )
    if issue is not None:
        return f"arm feedback unhealthy ({issue.detail}); press H after recovery"
    current = np.asarray(arm_state["qpos"], dtype=np.float64)
    home = np.asarray(runtime.arm.home_qpos, dtype=np.float64)
    max_abs_delta = float(np.max(np.abs(current - home)))
    tolerance = float(runtime.arm.homing.convergence_rad)
    if max_abs_delta <= tolerance:
        return None
    return (
        "arm is not at the training start pose; press H before B: "
        f"max_abs_delta_rad={max_abs_delta:.9f} tolerance_rad={tolerance:.9f}"
    )


def _end_policy_run(
    shared: RuntimeChannels,
    reason: str,
    *,
    stats: PolicyStats,
    aborted: bool,
) -> None:
    """Fence one episode into ARMED without converting policy failure to FAULT."""
    shared.physical_home_completed.value = False
    lifecycle_faulted = bool(
        shared.error_state.value
        or shared.estop_request.value
        or int(shared.safety_state.value) == int(SafetyState.FAULT)
    )
    if not lifecycle_faulted and int(shared.safety_state.value) == int(
        SafetyState.RUNNING
    ):
        if not revoke_motion(shared, SafetyState.ARMED):
            shared.error_state.value = True
            revoke_motion(shared, SafetyState.FAULT)
            aborted = True
            logger.critical("executor: failed to fence episode into ARMED (%s)", reason)
    if aborted:
        stats.flush(prefix="executor metrics")
        logger.warning("executor: policy episode ended: %s", reason)
    else:
        logger.info("executor: policy episode ended: %s", reason)


class PolicyExecutor:
    """One process-local owner of learned-policy scheduling and publication."""

    def __init__(
        self,
        shared: RuntimeChannels,
        runtime: ExperimentConfig,
        policy_spec: Any,
        *,
        execute: bool,
        max_running_s: float | None,
        evaluation_config: RolloutRecordingConfig | None = None,
    ) -> None:
        self.shared = shared
        self.runtime = runtime
        self.policy_spec = policy_spec
        self.execute = execute
        self.max_running_s = max_running_s
        if evaluation_config is not None:
            if not isinstance(evaluation_config, RolloutRecordingConfig):
                raise TypeError("evaluation_config must be a RolloutRecordingConfig")
            if not execute:
                raise ValueError("recorded rollout requires execute=True")
            if self.max_running_s != evaluation_config.max_running_s:
                raise ValueError("rollout timeout must match max_running_s")
        self.recording_config = evaluation_config
        self.recorder = (
            RecorderClient(shared) if evaluation_config is not None else None
        )
        self.last_recorded_action: EpisodeAction | None = None
        self.next_record_ns = 0
        self.control_period_s = 1.0 / float(runtime.policy.control_hz)
        self.step_dt_ns = int(round(self.control_period_s * 1e9))
        self.first_command_timeout_ns = int(
            float(runtime.policy.first_command_timeout_s) * 1e9
        )
        self.command_silence_timeout_ns = int(
            float(runtime.policy.max_command_silence_s) * 1e9
        )
        self.command_progress_timeout_ns = int(
            float(runtime.policy.command_progress_timeout_s) * 1e9
        )
        self.max_running_ns = (
            None if self.max_running_s is None else int(self.max_running_s * 1e9)
        )

        self.gate = _build_policy_safety_gate(runtime)
        self.ee_planner = (
            _build_policy_planner(runtime)
            if policy_spec.action_key == "action_ee"
            else None
        )
        self.stats = PolicyStats()
        self.progress = _CommandProgress()

        self.run_generation: int | None = None
        self.run_started_ns: int | None = None
        self.last_seen_prediction_sequence: int | None = None
        self.active_prediction: Prediction | None = None
        self.step_index = 0
        self.schedule_base_ns: int | None = None
        self.next_command_due_ns: int | None = None
        self.last_publication_ns: int | None = None
        self.last_valid_command_ns: int | None = None
        self.previous_arm_command_qpos: np.ndarray | None = None
        self.previous_hand_command_qpos: np.ndarray | None = None
        self.episode_steps = 0
        self.rollout_started = False
        self.rollout_result = None
        self.last_metrics_flush_ns = time.monotonic_ns()

    def _clear_execution(self, generation: int | None) -> None:
        self.run_generation = generation
        self.last_seen_prediction_sequence = None
        self.active_prediction = None
        self.step_index = 0
        self.schedule_base_ns = None
        self.next_command_due_ns = None
        self.last_publication_ns = None
        self.last_valid_command_ns = None
        self.previous_arm_command_qpos = None
        self.previous_hand_command_qpos = None
        self.last_recorded_action = None
        self.progress.reset(generation)

    def _finish_episode(
        self,
        reason: str,
        *,
        aborted: bool = True,
        stop_reason: str = "executor_boundary",
        outcome: EvaluationOutcome = EvaluationOutcome.INVALID,
        recorder_save: bool = True,
    ) -> None:
        if self.run_started_ns is None:
            return
        metrics = self.stats.snapshot()
        _end_policy_run(
            self.shared,
            reason,
            stats=self.stats,
            aborted=aborted,
        )
        if self.recorder is not None:
            self.rollout_result = dict(
                outcome=outcome,
                stop_reason=stop_reason,
                duration_s=max(0.0, (time.monotonic_ns() - self.run_started_ns) / 1e9),
                metrics=metrics,
            )
            self.recorder.stop_episode(
                save=recorder_save,
                reason=stop_reason,
            )
        self.run_started_ns = None
        self._clear_execution(None)

    def _invalidate_rollout(
        self, reason: str, *, stop_reason: str, recorder_save: bool
    ) -> bool:
        if self.run_started_ns is not None:
            self._finish_episode(
                reason,
                stop_reason=stop_reason,
                recorder_save=recorder_save,
                aborted=True,
            )
        return True

    def _invalid_stop_reason(self, reason: str) -> str:
        """Keep shared fault metadata in the current rollout namespace."""
        mode = self.recording_config.mode if self.recording_config is not None else "run"
        return f"{mode}:invalid:{reason}"

    def _fault(
        self,
        reason: str,
        *,
        stop_reason: str | None = None,
        recorder_save: bool | None = None,
    ) -> None:
        if stop_reason is None:
            stop_reason = self._invalid_stop_reason("hardware_fault")
        self.shared.error_state.value = True
        self.shared.physical_home_completed.value = False
        revoke_motion(self.shared, SafetyState.FAULT)
        if self.run_started_ns is not None:
            self._finish_episode(
                reason,
                stop_reason=stop_reason,
                recorder_save=True if recorder_save is None else recorder_save,
            )
        elif self.recorder is not None and self.recorder.is_recording:
            self.recorder.stop_episode(save=False, reason=stop_reason)
        self.stats.flush(prefix="executor metrics")
        logger.critical("executor: runtime fault: %s", reason)
        self.run_started_ns = None
        self._clear_execution(None)

    @staticmethod
    def _recording_vr_sentinel() -> dict[str, np.ndarray]:
        """Return the existing raw-schema sentinel for a non-VR rollout."""
        return {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }

    def _recorded_hold_action(self, state: EpisodeState) -> EpisodeAction:
        if self.last_recorded_action is not None:
            return self.last_recorded_action
        target_eef_pos, target_eef_rot6d = make_arm_fk().compute(state.arm_qpos)
        return EpisodeAction(
            arm_qpos_cmd=np.asarray(state.arm_qpos, dtype=np.float64),
            hand_qpos_cmd=np.asarray(state.hand_qpos, dtype=np.float64),
            target_eef_pos=np.asarray(target_eef_pos, dtype=np.float64),
            target_eef_rot6d=np.asarray(target_eef_rot6d, dtype=np.float64),
        )

    def _recorded_action_from_command(
        self,
        *,
        arm_qpos: np.ndarray,
        hand_qpos: np.ndarray,
        raw_action: np.ndarray,
    ) -> EpisodeAction:
        if self.policy_spec.action_key == "action_ee":
            target_eef_pos = np.asarray(raw_action[:3], dtype=np.float64)
            target_eef_rot6d = quat_wxyz_to_rot6d(
                rot6d_to_quat_wxyz(np.asarray(raw_action[3:9], dtype=np.float64))
            )
        else:
            target_eef_pos, target_eef_rot6d = make_arm_fk().compute(arm_qpos)
        return EpisodeAction(
            arm_qpos_cmd=np.asarray(arm_qpos, dtype=np.float64),
            hand_qpos_cmd=np.asarray(hand_qpos, dtype=np.float64),
            target_eef_pos=np.asarray(target_eef_pos, dtype=np.float64),
            target_eef_rot6d=np.asarray(target_eef_rot6d, dtype=np.float64),
        )

    def _record_frame(
        self,
        inputs: tuple[EpisodeState, dict, dict],
        action: EpisodeAction,
        *,
        signals: dict[str, Any],
        arm_qpos_sent: np.ndarray | None = None,
    ) -> bool:
        if self.recorder is None or self.run_generation is None:
            return False
        try:
            return self.recorder.add_frame(
                inputs[0],
                action,
                self._recording_vr_sentinel(),
                camera_frame=inputs[1],
                signals={**inputs[2], **signals},
                arm_qpos_sent=arm_qpos_sent,
            )
        except Exception:
            logger.error(
                "executor: rollout sample construction failed", exc_info=True
            )
            return False

    def _poll_recorder(self) -> bool:
        if self.recorder is None:
            return True
        was_stop_pending = self.recorder.stop_pending
        try:
            result = self.recorder.poll_stop()
        except Exception:
            finished = self._invalidate_rollout(
                "RecorderIO status polling failed",
                stop_reason=self._invalid_stop_reason("recorder_fault"),
                recorder_save=False,
            )
            return not finished
        ended = False
        if result.error:
            if (
                self.run_started_ns is None
                and not was_stop_pending
                and result.reason == "start_error"
            ):
                logger.warning(
                    "executor: RecorderIO rejected an eval start: %s",
                    result.error or "unknown error",
                )
                return True
            ended = self._invalidate_rollout(
                f"RecorderIO failed: {result.error or 'unknown error'}",
                stop_reason=self._invalid_stop_reason("recorder_fault"),
                recorder_save=False,
            )
        elif (
            self.recorder.stop_pending or result.done
        ) and self.run_started_ns is not None:
            if result.reason == self._invalid_stop_reason("max_frames"):
                ended = self._invalidate_rollout(
                    "RecorderIO reached its rollout frame capacity",
                    stop_reason=self._invalid_stop_reason("max_frames"),
                    recorder_save=True,
                )
            else:
                ended = self._invalidate_rollout(
                    "RecorderIO finalized unexpectedly: "
                    f"{result.reason or 'unknown reason'}",
                    stop_reason=self._invalid_stop_reason("recorder_fault"),
                    recorder_save=False,
                )
        if result.done:
            self._complete_recording(result)
        return not ended

    def _complete_recording(self, result: RecorderStopResult) -> None:
        """Consume terminal storage status once, after the motion fence."""
        if self.rollout_result is None:
            self.shared.is_recording.value = False
            return
        if result.error or not result.saved:
            self.rollout_result.update(
                outcome=EvaluationOutcome.INVALID, stop_reason="recording_failure"
            )
        try:
            episode_path = result.path or self.recorder.episode_path
            if episode_path is None:
                raise RuntimeError("Recorder returned no episode path")
            path = write_rollout_result(
                self.recording_config,
                episode_path,
                saved=result.saved,
                **self.rollout_result,
            )
            logger.info("rollout result: %s", path)
        except Exception:
            logger.error(
                "rollout INVALID: result metadata could not be saved", exc_info=True
            )
        finally:
            self.shared.is_recording.value = False
            self.rollout_result = None

    def _record_rejection(
        self,
        inputs: tuple[EpisodeState, dict, dict],
        *,
        kind: _RejectKind,
    ) -> bool:
        action = self._recorded_hold_action(inputs[0])
        safety_reject = kind is _RejectKind.SAFETY
        return self._record_frame(
            inputs,
            action,
            signals={
                "action_queued": False,
                "frame_status": (
                    _RECORD_FRAME_SAFETY_REJECT
                    if safety_reject
                    else _RECORD_FRAME_IK_FAIL
                ),
            },
            arm_qpos_sent=action.arm_qpos_cmd,
        )

    def _record_command(
        self,
        inputs: tuple[EpisodeState, dict, dict],
        candidate: ActionCandidate,
        raw_action: np.ndarray,
    ) -> bool:
        assert candidate.arm_qpos is not None
        assert candidate.hand_qpos is not None
        try:
            action = self._recorded_action_from_command(
                arm_qpos=candidate.arm_qpos,
                hand_qpos=candidate.hand_qpos,
                raw_action=raw_action,
            )
        except Exception:
            logger.error(
                "executor: failed to build rollout action record", exc_info=True
            )
            return False
        self.last_recorded_action = action
        return self._record_frame(
            inputs,
            action,
            signals={
                "action_queued": True,
                "frame_status": _RECORD_FRAME_OK,
            },
            arm_qpos_sent=candidate.arm_qpos,
        )

    def _record_rollout_tick(
        self,
        now_ns: int,
        *,
        candidate: ActionCandidate | None = None,
        raw_action: np.ndarray | None = None,
        reject_kind: _RejectKind | None = None,
    ) -> None:
        """Record a completed control decision, or an ordinary idle grid sample.

        Command rows are observed immediately so S/Q cannot lose an already
        published command in a second event buffer. The recorder owns raw-grid
        alignment; no source read here admits or rejects a robot command.
        """
        if self.recorder is None or self.run_started_ns is None:
            return
        if raw_action is None and now_ns < self.next_record_ns:
            return
        self.next_record_ns = next_periodic_deadline_ns(
            self.next_record_ns, self.step_dt_ns, now_ns
        )
        try:
            sources = {}
            for name, ring in (
                ("arm", self.shared.arm_state_ring),
                ("hand", self.shared.hand_state_ring),
                ("tactile", self.shared.hand_tactile_ring),
            ):
                sources[name] = read_causal_structured_frame(
                    ring, source_field="source_monotonic_ns", anchor_monotonic_ns=now_ns
                )
            if sources["arm"] is None or sources["hand"] is None:
                raise RuntimeError("recording arm/hand feedback unavailable")
            arm, _arm_publish_ns, _arm_sequence = sources["arm"]
            hand, _hand_publish_ns, _hand_sequence = sources["hand"]
            camera = read_camera_frame_causal(self.shared, anchor_monotonic_ns=now_ns)
            if camera is None:
                raise RuntimeError("recording camera unavailable")
            camera = dict(camera)
            camera["camera_age_s"] = (now_ns - int(camera["source_monotonic_ns"])) / 1e9
            camera["camera_fresh"] = (
                int(camera["camera_health"]) == int(CameraHealth.OK)
                and camera["camera_age_s"] <= self.runtime.camera.max_frame_age_s
            )
            if not camera["camera_fresh"]:
                raise RuntimeError("recording camera unhealthy or stale")
            tactile = None if sources["tactile"] is None else sources["tactile"][0]
            tactile_fresh = bool(
                tactile is not None
                and tactile["fresh"][0]
                and now_ns - int(tactile["source_monotonic_ns"][0])
                <= _RECORDING_TACTILE_MAX_AGE_NS
            )
            state = build_episode_state(
                arm,
                hand,
                tactile if tactile_fresh else None,
                timestamp_s=now_ns / 1e9,
            )
            signals = {
                "observation_anchor_monotonic_ns": now_ns,
                "observation_valid": False,
                "policy_observation_valid": False,
                "tracking_error": (
                    float(arm["tracking_err"][0])
                    if "tracking_err" in (arm.dtype.names or ())
                    else np.nan
                ),
                "arm_source_monotonic_ns": int(arm["source_monotonic_ns"][0]),
                "hand_source_monotonic_ns": int(hand["source_monotonic_ns"][0]),
                "vr_source_monotonic_ns": 0,
                "camera_source_monotonic_ns": int(camera["source_monotonic_ns"]),
                "tactile_fresh": tactile_fresh,
                "tactile_source_monotonic_ns": (
                    0 if tactile is None else int(tactile["source_monotonic_ns"][0])
                ),
                "tactile_calibrated": (
                    False if tactile is None else bool(tactile["calibrated"][0])
                ),
                "tactile_unit_code": (
                    0 if tactile is None else int(tactile["unit_code"][0])
                ),
            }
            inputs = (state, camera, signals)
            if raw_action is None:
                action = self._recorded_hold_action(state)
                recorded = self._record_frame(
                    inputs,
                    action,
                    signals={
                        "action_queued": False,
                        "frame_status": _RECORD_FRAME_HELD,
                    },
                    arm_qpos_sent=action.arm_qpos_cmd,
                )
            else:
                if candidate is not None:
                    recorded = self._record_command(inputs, candidate, raw_action)
                else:
                    assert reject_kind is not None
                    recorded = self._record_rejection(
                        inputs,
                        kind=reject_kind,
                    )
            if not recorded:
                raise RuntimeError("rollout sample rejected by RecorderIO")
        except Exception as exc:
            logger.error("rollout recording failed", exc_info=True)
            self._invalidate_rollout(
                str(exc),
                stop_reason=self._invalid_stop_reason("recording_failure"),
                recorder_save=False,
            )

    def _start_requested_episode(self) -> None:
        if not bool(self.shared.start_request.value):
            return
        if self.rollout_started:
            with self.shared.motion_lock:
                self.shared.start_request.value = False
            logger.warning("rollout already completed; restart run_policy.py")
            return
        if self.recorder is not None and self.recorder.stop_pending:
            with self.shared.motion_lock:
                self.shared.start_request.value = False
            logger.warning("executor: ignored B while RecorderIO is finalizing")
            return
        rejection = _physical_start_pose_rejection(
            self.shared, self.runtime, execute=self.execute
        )
        if rejection is not None:
            with self.shared.motion_lock:
                self.shared.start_request.value = False
                if self.recorder is not None:
                    self.shared.evaluation_outcome.value = int(
                        EvaluationOutcome.INVALID
                    )
            logger.warning("executor: ignored B: %s", rejection)
            return
        if self.recorder is not None:
            assert self.recording_config is not None
            # Reserve the transaction before the bounded START wait so Q cannot
            # let lifecycle shutdown race an in-flight recorder acknowledgement.
            self.shared.is_recording.value = True
            if not self.recorder.start_episode(
                task_label=self.recording_config.task_label,
                operator=self.recording_config.operator,
            ):
                self.shared.is_recording.value = self.recorder.stop_pending
                with self.shared.motion_lock:
                    self.shared.start_request.value = False
                    self.shared.evaluation_outcome.value = int(
                        EvaluationOutcome.INVALID
                    )
                logger.warning("executor: RecorderIO did not acknowledge eval START")
                return
            self.shared.is_recording.value = True
            rejection = _physical_start_pose_rejection(
                self.shared, self.runtime, execute=self.execute
            )
            if rejection is not None:
                with self.shared.motion_lock:
                    self.shared.start_request.value = False
                    self.shared.evaluation_outcome.value = int(
                        EvaluationOutcome.INVALID
                    )
                self.recorder.stop_episode(
                    save=False,
                    reason=self._invalid_stop_reason("start_recheck_failed"),
                )
                logger.warning("executor: cancelled eval START: %s", rejection)
                return

        if self.recorder is None:
            epoch = begin_requested_motion(self.shared)
        else:
            # B, S/Q, outcome reset, and RUNNING transition share one RLock.
            # This prevents an outcome written by the operator from being
            # clobbered after it has ordered its stop request.
            with self.shared.motion_lock:
                if (
                    not bool(self.shared.start_request.value)
                    or int(self.shared.stop_request.value) != int(StopRequest.NONE)
                    or int(self.shared.safety_state.value) != int(SafetyState.ARMED)
                    or not bool(self.shared.is_running.value)
                    or bool(self.shared.error_state.value)
                    or bool(self.shared.estop_request.value)
                ):
                    epoch = None
                else:
                    self.shared.evaluation_outcome.value = int(EvaluationOutcome.NONE)
                    epoch = begin_requested_motion(self.shared)
        if epoch is None:
            if self.recorder is not None:
                self.recorder.stop_episode(
                    save=False,
                    reason=self._invalid_stop_reason("start_cancelled"),
                )
            return
        if self.execute:
            self.shared.physical_home_completed.value = False
        self.run_started_ns = epoch.started_monotonic_ns
        self.rollout_started = True
        self.stats = PolicyStats()
        self.next_record_ns = epoch.started_monotonic_ns + self.step_dt_ns
        self.episode_steps = 0
        self._clear_execution(epoch.generation)
        logger.info("policy_executor_loop: RUNNING generation=%d", epoch.generation)

    def _handle_run_boundary(self) -> None:
        if not self._poll_recorder():
            return
        if bool(self.shared.quit_requested.value) and not (
            self.shared.error_state.value
            or self.shared.estop_request.value
            or int(self.shared.safety_state.value) == int(SafetyState.FAULT)
        ):
            if self.run_started_ns is not None:
                if self.recorder is not None:
                    self._finish_episode(
                        "operator quit",
                        stop_reason="operator_abort",
                        outcome=(
                            EvaluationOutcome.INVALID
                            if self.recording_config.mode == "eval"
                            else EvaluationOutcome.STOPPED
                        ),
                        recorder_save=True,
                        aborted=False,
                    )
                else:
                    self._finish_episode("operator quit", aborted=False)
            # Supervisor owns global shutdown. Stay alive until it observes Q,
            # otherwise a clean executor exit can be misclassified as worker death.
            return
        if bool(self.shared.error_state.value) or bool(self.shared.estop_request.value):
            revoke_motion(self.shared, SafetyState.FAULT)
            if self.run_started_ns is not None and self.recorder is not None:
                stop_reason = (
                    self._invalid_stop_reason("estop")
                    if bool(self.shared.estop_request.value)
                    else self._invalid_stop_reason("hardware_fault")
                )
                self._finish_episode(
                    (
                        "emergency stop"
                        if bool(self.shared.estop_request.value)
                        else "hardware fault"
                    ),
                    stop_reason=stop_reason,
                    recorder_save=True,
                    aborted=False,
                )
            elif self.recorder is not None and self.recorder.is_recording:
                self._fault(
                    "formal recorder was active before a motion epoch faulted",
                    stop_reason=(
                        self._invalid_stop_reason("estop")
                        if bool(self.shared.estop_request.value)
                        else self._invalid_stop_reason("hardware_fault")
                    ),
                    recorder_save=False,
                )
            else:
                self.shared.physical_home_completed.value = False
                self.run_started_ns = None
                self._clear_execution(None)
            return

        run_snapshot = read_run_state_snapshot(self.shared)
        raw_stop = run_snapshot.stop_request
        if raw_stop not in {int(StopRequest.NONE), int(StopRequest.OPERATOR)}:
            self._fault("invalid stop request code")
            return
        if self.run_started_ns is not None and raw_stop == int(StopRequest.OPERATOR):
            with self.shared.motion_lock:
                self.shared.start_request.value = False
            if self.recorder is not None:
                try:
                    outcome = EvaluationOutcome(
                        int(self.shared.evaluation_outcome.value)
                    )
                except ValueError:
                    self._fault(
                        "invalid rollout outcome wire value",
                        stop_reason=self._invalid_stop_reason("recorder_fault"),
                        recorder_save=False,
                    )
                    return
                if self.recording_config.mode == "run":
                    outcome = EvaluationOutcome.STOPPED
                elif outcome is EvaluationOutcome.NONE:
                    outcome = EvaluationOutcome.INVALID
                stop_reason = evaluation_outcome_stop_reason(outcome)
                self._finish_episode(
                    "operator evaluation stop",
                    stop_reason=stop_reason,
                    outcome=outcome,
                    recorder_save=True,
                    aborted=False,
                )
            else:
                self._finish_episode("operator stop", aborted=False)
            # Q waits for this acknowledgement before letting the supervisor
            # tear down RecorderIO.  At this point _finish_* has already
            # fenced motion and queued its recorder STOP decision.
            with self.shared.motion_lock:
                if int(self.shared.stop_request.value) == int(StopRequest.OPERATOR):
                    self.shared.stop_request.value = int(StopRequest.NONE)
            return

        if run_snapshot.state is not SafetyState.RUNNING:
            if self.run_started_ns is not None:
                # Operator S revokes motion before this process observes its flag.
                if self.recorder is not None:
                    self._finish_episode(
                        "motion revoked outside formal stop request",
                        stop_reason=self._invalid_stop_reason("hardware_fault"),
                        recorder_save=True,
                        aborted=True,
                    )
                return
            self._clear_execution(None)
            if not self.execute:
                with self.shared.motion_lock:
                    if not bool(self.shared.start_request.value) and int(
                        self.shared.stop_request.value
                    ) == int(StopRequest.OPERATOR):
                        self.shared.stop_request.value = int(StopRequest.NONE)
            self._start_requested_episode()

    def _observe_worker_progress(self, now_ns: int) -> bool:
        if not self.execute:
            return True
        try:
            arm_id, hand_id, hand_setpoint_accepted_ns, feedback_fault = (
                _read_command_progress(self.shared, self.runtime, now_ns=now_ns)
            )
        except Exception as exc:
            self._fault(f"command progress feedback failed: {type(exc).__name__}")
            return False
        if feedback_fault is not None:
            self._fault(feedback_fault)
            return False
        assert self.run_generation is not None
        reason = self.progress.observe(
            generation=self.run_generation,
            arm_action_id=arm_id,
            hand_action_id=hand_id,
            hand_setpoint_accepted_ns=hand_setpoint_accepted_ns,
            now_ns=now_ns,
            timeout_ns=self.command_progress_timeout_ns,
        )
        if reason is not None:
            self.stats.command_progress_timeout_count += 1
            self._fault(reason)
            return False
        return True

    def _ingest_latest_prediction(self, now_ns: int) -> bool:
        try:
            latest = read_latest_prediction(self.shared)
        except Exception as exc:
            self._fault(f"invalid prediction IPC record: {exc}")
            return False
        if latest is None:
            return True
        prediction, sequence = latest
        if sequence == self.last_seen_prediction_sequence:
            return True
        self.last_seen_prediction_sequence = sequence
        if prediction.run_generation != self.run_generation:
            return True
        # Include even wholly stale chunks: their timing explains the discard.
        # Old generations and repeated ring reads never add samples.
        self.stats.inference_latency_ms = float(prediction.inference_latency_ms)
        self.stats.observation_age_ms = float(prediction.observation_age_ms)
        self.stats.observation_skew_ms = float(prediction.observation_skew_ms)

        first_index = first_future_step_index(
            prediction.logical_step_monotonic_ns,
            self.step_dt_ns,
            now_ns,
            prediction.num_steps,
        )
        if first_index is None:
            self.stats.stale_prediction_count += 1
            return True
        self.active_prediction = prediction
        self.step_index = first_index
        self.schedule_base_ns = prediction.logical_step_monotonic_ns
        self.stats.skipped_prefix_steps = first_index
        return True

    def _next_due_action(self, now_ns: int) -> tuple[np.ndarray, int, int] | None:
        prediction = self.active_prediction
        if prediction is None or self.schedule_base_ns is None:
            return None

        target_ns = self.schedule_base_ns + self.step_index * self.step_dt_ns
        if now_ns >= target_ns:
            first_index = first_future_step_index(
                self.schedule_base_ns,
                self.step_dt_ns,
                now_ns,
                prediction.num_steps,
            )
            if first_index is None:
                self.active_prediction = None
                self.stats.stale_prediction_count += 1
                return None
            self.step_index = max(self.step_index, first_index)

        target_ns = self.schedule_base_ns + self.step_index * self.step_dt_ns
        # Publish the upcoming target during its preceding control interval.
        # Waiting until target_ns would make every target stale at admission.
        slot_ns = max(1, target_ns - self.step_dt_ns)
        due_ns = (
            slot_ns
            if self.next_command_due_ns is None
            else max(slot_ns, self.next_command_due_ns)
        )
        if due_ns >= target_ns:
            return None
        if now_ns < due_ns:
            return None
        return prediction.actions[self.step_index], target_ns, due_ns

    def _consume_control_slot(self, due_ns: int, terminal_ns: int) -> None:
        lateness_ms = max(0, terminal_ns - due_ns) / 1e6
        self.stats.schedule_lateness_ms = lateness_ms
        self.next_command_due_ns = _advance_control_grid_ns(
            due_ns, terminal_ns, self.step_dt_ns
        )

    def _advance_prediction(self) -> None:
        prediction = self.active_prediction
        if prediction is None:
            raise RuntimeError("cannot advance without an active prediction")
        self.step_index += 1
        if self.step_index >= prediction.num_steps:
            self.active_prediction = None
            self.schedule_base_ns = None
            return
        first_index = first_future_step_index(
            prediction.logical_step_monotonic_ns,
            self.step_dt_ns,
            time.monotonic_ns(),
            prediction.num_steps,
        )
        if first_index is None:
            self.active_prediction = None
        else:
            self.step_index = max(self.step_index, first_index)

    def _commit_terminal_step(self) -> None:
        """Advance after a published or rejected control step."""
        self.episode_steps += 1
        self._advance_prediction()

    def _reject_due_step(self, due_ns: int, reason: str) -> int:
        terminal_ns = time.monotonic_ns()
        self._consume_control_slot(due_ns, terminal_ns)
        logger.warning("executor: rejected policy step: %s", reason)
        self._commit_terminal_step()
        return terminal_ns

    def _decode_due_action(
        self, action: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, _RejectKind | None, str | None]:
        arm_state = read_arm_state_dict(self.shared)
        if arm_state is None:
            return None, None, None
        try:
            issue = diagnose_arm_feedback(
                connected=bool(arm_state["connected"]),
                error_code=int(arm_state["error_code"]),
                state_valid=bool(arm_state["state_valid"]),
                source_monotonic_ns=int(arm_state["source_monotonic_ns"]),
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=float(self.runtime.safety.heartbeat_timeouts["arm"]),
                qpos=np.asarray(arm_state["qpos"], dtype=np.float64),
                qvel=np.asarray(arm_state["qvel"], dtype=np.float64),
            )
        except Exception as exc:
            self._fault(f"malformed arm feedback: {type(exc).__name__}")
            return None, None, None
        if issue is not None:
            if issue.code is FeedbackIssueCode.STALE:
                pass
            else:
                self._fault(f"fatal arm feedback: {issue.code.value}")
            return None, None, None
        if self.policy_spec.action_key == "action_ee" and self.ee_planner is None:
            self._fault("EE policy executor has no IK planner")
            return None, None, None
        arm_qpos, hand_qpos, rejection = decode_policy_action(
            action,
            self.policy_spec,
            np.asarray(arm_state["qpos"], dtype=np.float64),
            previous_arm_command_qpos=self.previous_arm_command_qpos,
            planner=self.ee_planner,
        )
        if arm_qpos is None:
            if self.policy_spec.action_key == "action_ee":
                self.stats.ik_rejection_count += 1
            return (
                None,
                _RejectKind.IK,
                rejection or "EE action has no usable IK solution",
            )
        reference_arm_qpos = (
            np.asarray(arm_state["qpos"], dtype=np.float64)
            if self.previous_arm_command_qpos is None
            else self.previous_arm_command_qpos
        )
        arm_qpos, rejection = _validate_policy_arm_action(
            arm_qpos, reference_arm_qpos, self.runtime
        )
        if arm_qpos is None:
            self.stats.safety_rejection_count += 1
            return None, _RejectKind.SAFETY, rejection
        return (arm_qpos, hand_qpos), None, None

    def _publish_due_action(
        self,
        action: np.ndarray,
        *,
        scheduled_target_ns: int,
        due_ns: int,
    ) -> None:
        decoded, reject_kind, decode_rejection = self._decode_due_action(action)
        if decoded is None:
            if bool(self.shared.error_state.value):
                return
            if decode_rejection is not None:
                terminal_ns = self._reject_due_step(due_ns, decode_rejection)
                self._record_rollout_tick(
                    terminal_ns,
                    raw_action=action,
                    reject_kind=reject_kind,
                )
            return
        arm_qpos, hand_qpos = decoded
        assert self.active_prediction is not None
        candidate = build_action_candidate(
            self.shared,
            arm_qpos,
            hand_qpos,
            run_generation=self.active_prediction.run_generation,
            is_hold=False,
            scheduled_target_monotonic_ns=scheduled_target_ns,
            action_validity_s=float(self.runtime.policy.action_validity_s),
        )
        if candidate is None:
            self._advance_prediction()
            self.stats.stale_prediction_count += 1
            return
        prepared = prepare_command(
            self.shared,
            candidate,
            gate=self.gate,
            arm_feedback_max_age_s=float(self.runtime.safety.heartbeat_timeouts["arm"]),
            hand_feedback_max_age_s=float(
                self.runtime.safety.heartbeat_timeouts["hand"]
            ),
            hand_delta_reference_qpos=self.previous_hand_command_qpos,
            hand_mechanical_lower_rad=np.asarray(
                self.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ),
            hand_mechanical_upper_rad=np.asarray(
                self.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ),
            canonicalize_policy_hand_roundoff=True,
        )
        if not prepared.accepted:
            self._handle_preparation_rejection(
                prepared,
                candidate,
                due_ns,
                raw_action=action,
            )
            return
        published_candidate = prepared.candidate
        assert published_candidate is not None
        # Decode/IK can consume the remaining future window. Never publish an
        # elapsed policy target, even if the transport validity window remains.
        if time.monotonic_ns() >= scheduled_target_ns:
            self._advance_prediction()
            self.stats.stale_prediction_count += 1
            return
        publish_result: PublishResult | None = None
        if self.execute:
            publish_result = publish_command(
                self.shared,
                published_candidate,
                required_safety_state=SafetyState.RUNNING,
                minimum_delivery_window_s=self.control_period_s,
            )
        else:
            reason = command_publishability_reason(
                self.shared,
                published_candidate,
                required_safety_state=SafetyState.RUNNING,
                minimum_delivery_window_s=self.control_period_s,
            )
            if reason:
                publish_result = PublishResult(False, reason=reason)
        if publish_result is not None and not publish_result.published:
            self._handle_publication_rejection(publish_result)
            return

        publication_ns = time.monotonic_ns()
        if self.execute:
            if publish_result is None or publish_result.ticket is None:
                self._fault("physical publication omitted its command ticket")
                return
            publication_ns = int(publish_result.ticket.published_monotonic_ns)
            if publication_ns <= 0:
                self._fault("physical publication omitted its monotonic timestamp")
                return
            self.progress.record_publication(
                published_candidate.action_id, publication_ns
            )
        assert published_candidate.arm_qpos is not None
        assert published_candidate.hand_qpos is not None
        self.previous_arm_command_qpos = published_candidate.arm_qpos.copy()
        self.previous_hand_command_qpos = published_candidate.hand_qpos.copy()
        self._consume_control_slot(due_ns, publication_ns)
        if self.last_publication_ns is not None:
            interval_ms = (publication_ns - self.last_publication_ns) / 1e6
            self.stats.publication_interval_ms = interval_ms
        self.last_publication_ns = publication_ns
        self.last_valid_command_ns = publication_ns
        self._commit_terminal_step()
        self._record_rollout_tick(
            publication_ns,
            candidate=published_candidate,
            raw_action=action,
        )

    def _handle_preparation_rejection(
        self,
        prepared: PreparedCommand,
        candidate: ActionCandidate,
        due_ns: int,
        *,
        raw_action: np.ndarray,
    ) -> None:
        if prepared.unavailable:
            return
        if prepared.fatal:
            self._fault(prepared.reason or "physical safety check failed")
            return
        if prepared.gate_code is not None or candidate.hand_qpos is not None:
            self.stats.safety_rejection_count += 1
        terminal_ns = self._reject_due_step(
            due_ns, prepared.reason or "physical safety rejection"
        )
        self._record_rollout_tick(
            terminal_ns,
            raw_action=raw_action,
            reject_kind=_RejectKind.SAFETY,
        )

    def _handle_publication_rejection(self, result: PublishResult) -> None:
        if result.reason == PUBLISH_REASON_EXPIRED:
            self._advance_prediction()
            self.stats.stale_prediction_count += 1
            return
        if result.reason in {PUBLISH_REASON_ESTOP, PUBLISH_REASON_FAULT}:
            return
        # A concurrent S/generation fence is an ordinary episode boundary.  The
        # next loop observes the operator request before any further command.
        if result.reason in {
            PUBLISH_REASON_GENERATION,
            PUBLISH_REASON_RUNTIME_STOPPED,
        } or (result.reason.startswith(PUBLISH_REASON_SAFETY_STATE)):
            return
        self._fault(
            f"unrecognized publication rejection: {result.reason or 'missing reason'}"
        )

    def _run_active_tick(self, now_ns: int) -> None:
        assert self.run_started_ns is not None
        run_snapshot = read_run_state_snapshot(self.shared)
        if self.run_generation != run_snapshot.generation:
            if (
                run_snapshot.state is SafetyState.ARMED
                and run_snapshot.stop_request == int(StopRequest.OPERATOR)
            ):
                # S revoked this run after the loop boundary check. The next
                # iteration consumes its request and ends the episode normally.
                return
            self._fault("RUNNING generation changed outside an episode boundary")
            return
        if not self._observe_worker_progress(now_ns):
            return
        if (
            self.max_running_ns is not None
            and now_ns - self.run_started_ns >= self.max_running_ns
        ):
            if self.recorder is not None:
                outcome, stop_reason = _rollout_timeout_result(
                    self.recording_config.mode, "run time limit"
                )
                self._finish_episode(
                    "run time limit",
                    stop_reason=stop_reason,
                    outcome=outcome,
                    aborted=False,
                )
            else:
                self._finish_episode("run time limit", aborted=False)
            return
        watchdog_reason = _command_watchdog_reason(
            now_ns=now_ns,
            run_started_ns=self.run_started_ns,
            last_valid_command_ns=self.last_valid_command_ns,
            first_command_timeout_ns=self.first_command_timeout_ns,
            command_silence_timeout_ns=self.command_silence_timeout_ns,
        )
        if watchdog_reason is not None:
            if self.recorder is not None:
                outcome, stop_reason = _rollout_timeout_result(
                    self.recording_config.mode, watchdog_reason
                )
                self._finish_episode(
                    watchdog_reason,
                    stop_reason=stop_reason,
                    outcome=outcome,
                )
            else:
                self._finish_episode(watchdog_reason)
            return
        if self.execute and (
            self.progress.arm_accepted_action_id is None
            or self.progress.hand_accepted_action_id is None
        ):
            return
        if not self._ingest_latest_prediction(now_ns):
            return
        due = self._next_due_action(now_ns)
        if due is None:
            return
        action, scheduled_target_ns, due_ns = due
        self._publish_due_action(
            action,
            scheduled_target_ns=scheduled_target_ns,
            due_ns=due_ns,
        )

    def run(self) -> None:
        """Run the readable poll loop; control-grid timing stays independent."""
        executor_poll_hz = float(self.runtime.policy.executor_poll_hz)
        rate = LoopRate(
            executor_poll_hz,
            label="policy executor",
            busy_wait=False,
        )
        try:
            while self.shared.is_running.value:
                self.shared.set_heartbeat("policy", time.monotonic())
                self._handle_run_boundary()
                run_snapshot = read_run_state_snapshot(self.shared)
                if (
                    self.run_started_ns is not None
                    and run_snapshot.state is SafetyState.RUNNING
                ):
                    self._run_active_tick(time.monotonic_ns())
                    if self.run_started_ns is not None:
                        self._record_rollout_tick(time.monotonic_ns())
                self.last_metrics_flush_ns = flush_every(
                    self.stats,
                    last_ns=self.last_metrics_flush_ns,
                    prefix="executor metrics",
                )
                rate.wait()
        finally:
            # A runtime stop can end the loop before its next boundary poll.
            # Storage finalization remains owned by RecorderIO.
            if self.run_started_ns is not None:
                self._finish_episode("runtime shutdown", stop_reason="runtime_shutdown")
            if self.recorder is not None and (
                self.recorder.stop_pending or self.rollout_result is not None
            ):
                result = self.recorder.join_stop()
                if result.done:
                    self._complete_recording(result)
                else:
                    logger.error("rollout recording finalization timed out")
            self.stats.flush(prefix="executor metrics")


def policy_executor_loop(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    policy_spec: Any,
    execute: bool,
    max_running_s: float | None = None,
    evaluation_config: RolloutRecordingConfig | None = None,
) -> None:
    """Process entry point for one lightweight policy executor."""
    PolicyExecutor(
        shared,
        runtime,
        policy_spec,
        execute=execute,
        max_running_s=max_running_s,
        evaluation_config=evaluation_config,
    ).run()
