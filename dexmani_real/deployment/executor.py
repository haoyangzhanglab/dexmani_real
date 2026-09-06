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
from dexmani_real.deployment.config import PolicyDeploymentConfig
from dexmani_real.deployment.evaluation import (
    EvaluationOutcome,
    PolicyEvaluationConfig,
    evaluation_outcome_stop_reason,
)
from dexmani_real.deployment.metrics import PolicyStats, flush_every
from dexmani_real.deployment.prediction import Prediction
from dexmani_real.deployment.timing import first_future_step_index
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
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.planning.paths import (
    WORKSPACE_BOUNDS_TOLERANCE_M,
    interpolate_waypoints,
    wrap_nearest_equivalent,
)
from dexmani_real.recording.client import RecorderClient, RecorderPhase
from dexmani_real.recording.sample import (
    EpisodeAction,
    EpisodeState,
    build_episode_state,
)
from dexmani_real.robot.model import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
    XHAND_RIGHT_URDF_PATH,
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
_EVALUATION_TACTILE_MAX_AGE_NS = 250_000_000
_EVALUATION_FRAME_OK = 0
_EVALUATION_FRAME_HELD = 1
_EVALUATION_FRAME_IK_FAIL = 2
_EVALUATION_FRAME_SAFETY_REJECT = 3


class _RejectKind(Enum):
    """Explicit attribution of a policy-step rejection, never a reason string."""

    IK = auto()
    SAFETY = auto()


@dataclass(frozen=True)
class _EvaluationFrameInputs:
    """One causal, post-RUNNING recorder input assembled by PolicyExecutor."""

    state: EpisodeState
    camera_frame: dict[str, Any]
    signals: dict[str, Any]


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

    def covers(self, action_id: int) -> bool:
        return bool(
            self.arm_accepted_action_id is not None
            and self.hand_accepted_action_id is not None
            and self.arm_accepted_action_id >= action_id
            and self.hand_accepted_action_id >= action_id
        )


def prediction_from_record(record: np.void) -> Prediction:
    """Deserialize and ownership-copy one exact flat prediction IPC record."""
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
        deployment: PolicyDeploymentConfig,
        *,
        execute: bool,
        max_running_s: float | None,
        evaluation_config: PolicyEvaluationConfig | None = None,
    ) -> None:
        self.shared = shared
        self.runtime = runtime
        self.policy_spec = policy_spec
        self.deployment = deployment
        self.execute = execute
        self.max_running_s = max_running_s
        if evaluation_config is not None:
            if not isinstance(evaluation_config, PolicyEvaluationConfig):
                raise TypeError("evaluation_config must be a PolicyEvaluationConfig")
            if not execute:
                raise ValueError("formal policy evaluation requires execute=True")
            if self.max_running_s != evaluation_config.max_running_s:
                raise ValueError("formal evaluation timeout must match max_running_s")
        self.evaluation_config = evaluation_config
        self.recorder = (
            RecorderClient(shared) if evaluation_config is not None else None
        )
        self.evaluation_hand_fk: HandKinematics | None = None
        self.evaluation_initial_sample_pending = False
        self.evaluation_initial_deadline_ns: int | None = None
        self.sync_mode = deployment.inference_mode == "sync"
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
        self.pending_truncation_action_id: int | None = None
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
        self.pending_truncation_action_id = None
        self.evaluation_initial_sample_pending = False
        self.evaluation_initial_deadline_ns = None
        self.progress.reset(generation)
        if self.sync_mode:
            self.shared.inference_request.clear()

    def _finish_episode(
        self,
        reason: str,
        *,
        aborted: bool = True,
        evaluation_stop_reason: str | None = None,
        recorder_save: bool = True,
    ) -> None:
        _end_policy_run(
            self.shared,
            reason,
            stats=self.stats,
            aborted=aborted,
        )
        if self.recorder is not None:
            self.shared.is_recording.value = False
            self.recorder.stop_episode(
                success=recorder_save,
                reason=(
                    evaluation_stop_reason
                    if evaluation_stop_reason is not None
                    else "eval:invalid:executor_boundary"
                ),
            )
        self.run_started_ns = None
        self._clear_execution(None)

    def _finish_evaluation_episode(
        self,
        reason: str,
        *,
        stop_reason: str,
        recorder_save: bool = True,
        aborted: bool = False,
    ) -> None:
        """Fence a formal trial before asynchronously finalizing its recorder."""
        self._finish_episode(
            reason,
            aborted=aborted,
            evaluation_stop_reason=stop_reason,
            recorder_save=recorder_save,
        )

    def _invalidate_evaluation(
        self,
        reason: str,
        *,
        stop_reason: str,
        recorder_save: bool,
    ) -> None:
        """End a formal trial as INVALID without raising a robot runtime fault.

        Evaluation-only evidence/recording failures must not convert a healthy
        control lifecycle into global FAULT.  The already-committed control
        result stays committed; this only fences the trial to ARMED and queues a
        recorder STOP with an explicit invalid reason.
        """
        self._finish_evaluation_episode(
            reason,
            stop_reason=stop_reason,
            recorder_save=recorder_save,
            aborted=True,
        )

    def _fault(
        self,
        reason: str,
        *,
        evaluation_stop_reason: str = "eval:invalid:hardware_fault",
        recorder_save: bool | None = None,
    ) -> None:
        self.shared.error_state.value = True
        self.shared.physical_home_completed.value = False
        revoke_motion(self.shared, SafetyState.FAULT)
        if self.recorder is not None:
            self.shared.is_recording.value = False
            self.recorder.stop_episode(
                success=(
                    not self.evaluation_initial_sample_pending
                    if recorder_save is None
                    else recorder_save
                ),
                reason=evaluation_stop_reason,
            )
        self.stats.flush(prefix="executor metrics")
        logger.critical("executor: runtime fault: %s", reason)
        self.run_started_ns = None
        self._clear_execution(None)

    @staticmethod
    def _evaluation_vr_sentinel() -> dict[str, np.ndarray]:
        """Return the existing raw-schema sentinel for a non-VR rollout."""
        return {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }

    def _evaluation_stop_reason_for_source_failure(self, reason: str) -> str:
        if reason.startswith("camera"):
            return "eval:invalid:camera_fault"
        return "eval:invalid:hardware_fault"

    def _evaluation_hand_kinematics(self) -> HandKinematics:
        if self.evaluation_hand_fk is None:
            self.evaluation_hand_fk = HandKinematics(
                str(XHAND_RIGHT_URDF_PATH),
                list(self.runtime.hand.fingertip_link_names),
            )
        return self.evaluation_hand_fk

    def _build_evaluation_frame_inputs(
        self,
        anchor_ns: int,
    ) -> tuple[_EvaluationFrameInputs | None, str, bool]:
        """Select a causal post-RUNNING state/camera cut for recorder evidence.

        Returns ``(inputs, reason, fatal)``.  A missing post-epoch source may
        become available on the next executor tick; malformed or unhealthy
        feedback/camera data is an immediate invalid trial condition.
        """
        if self.run_started_ns is None:
            return None, "hardware: no active RUNNING epoch", True
        try:
            arm_result = read_causal_structured_frame(
                self.shared.arm_state_ring,
                source_field="source_monotonic_ns",
                anchor_monotonic_ns=anchor_ns,
            )
            hand_result = read_causal_structured_frame(
                self.shared.hand_state_ring,
                source_field="source_monotonic_ns",
                anchor_monotonic_ns=anchor_ns,
            )
        except Exception as exc:
            return (
                None,
                f"hardware: causal feedback read failed ({type(exc).__name__})",
                True,
            )
        if arm_result is None or hand_result is None:
            return None, "hardware: waiting for causal arm/hand feedback", False

        arm_state, arm_publish_ns, arm_sequence = arm_result
        hand_state, hand_publish_ns, hand_sequence = hand_result
        arm = arm_state[0]
        hand = hand_state[0]
        arm_source_ns = int(arm["source_monotonic_ns"])
        hand_source_ns = int(hand["source_monotonic_ns"])
        if arm_source_ns < self.run_started_ns or hand_source_ns < self.run_started_ns:
            return None, "hardware: waiting for post-RUNNING arm/hand feedback", False
        try:
            arm_issue = diagnose_arm_feedback(
                connected=bool(arm["connected"]),
                error_code=int(arm["error_code"]),
                state_valid=bool(arm["state_valid"]),
                source_monotonic_ns=arm_source_ns,
                now_monotonic_ns=anchor_ns,
                max_age_s=float(self.runtime.safety.heartbeat_timeouts["arm"]),
                qpos=np.asarray(arm["qpos"], dtype=np.float64),
                qvel=np.asarray(arm["qvel"], dtype=np.float64),
            )
            hand_issue = diagnose_hand_feedback(
                connected=bool(hand["connected"]),
                state_valid=bool(hand["state_valid"]),
                source_monotonic_ns=hand_source_ns,
                now_monotonic_ns=anchor_ns,
                max_age_s=float(self.runtime.safety.heartbeat_timeouts["hand"]),
                qpos=np.asarray(hand["qpos"], dtype=np.float64),
            )
        except Exception as exc:
            return None, f"hardware: malformed feedback ({type(exc).__name__})", True
        if arm_issue is not None:
            return (
                None,
                f"hardware: arm feedback unhealthy ({arm_issue.detail})",
                arm_issue.code is not FeedbackIssueCode.STALE,
            )
        if hand_issue is not None:
            return (
                None,
                f"hardware: hand feedback unhealthy ({hand_issue.detail})",
                hand_issue.code is not FeedbackIssueCode.STALE,
            )

        try:
            camera_frame = read_camera_frame_causal(
                self.shared,
                anchor_monotonic_ns=anchor_ns,
            )
        except Exception as exc:
            return None, f"camera: causal read failed ({type(exc).__name__})", True
        if camera_frame is None:
            return None, "camera: waiting for causal RGB-D evidence", False
        camera_source_ns = int(camera_frame.get("source_monotonic_ns", 0))
        if camera_source_ns < self.run_started_ns:
            return None, "camera: waiting for post-RUNNING RGB-D evidence", False
        camera_age_s = (anchor_ns - camera_source_ns) / 1e9
        if camera_age_s < 0.0:
            return None, "camera: source timestamp is in the future", True
        if camera_age_s > float(self.runtime.camera.max_frame_age_s):
            return None, "camera: RGB-D evidence is stale", False
        try:
            camera_health = CameraHealth(int(camera_frame.get("camera_health", -1)))
        except ValueError:
            return None, "camera: health enum is invalid", True
        if camera_health is not CameraHealth.OK:
            return None, f"camera: health={camera_health.name}", True
        if bool(camera_frame.get("clock_reset", False)) or bool(
            camera_frame.get("duplicate", False)
        ):
            return None, "camera: reset or duplicate frame", True
        for field_name in (
            "camera_generation",
            "receive_monotonic_ns",
            "publish_monotonic_ns",
            "wait_return_monotonic_ns",
            "payload_ready_monotonic_ns",
        ):
            if int(camera_frame.get(field_name, 0)) <= 0:
                return None, f"camera: missing {field_name}", True
        valid_depth_ratio = float(camera_frame.get("valid_depth_ratio", np.nan))
        if not np.isfinite(valid_depth_ratio) or not 0.0 <= valid_depth_ratio <= 1.0:
            return None, "camera: valid depth ratio is invalid", True
        for field_name in (
            "backlog_s",
            "delivery_delay_above_floor_s",
            "depth_device_timestamp_s",
            "color_device_timestamp_s",
        ):
            value = float(camera_frame.get(field_name, np.nan))
            if not np.isfinite(value) or value < 0.0:
                return None, f"camera: {field_name} is invalid", True
        camera_frame = dict(camera_frame)
        camera_frame["camera_age_s"] = camera_age_s
        camera_frame["camera_fresh"] = True

        tactile_state: np.ndarray | None = None
        tactile_fresh = False
        tactile_source_ns = 0
        tactile_calibrated = False
        tactile_unit_code = 0
        try:
            tactile_result = read_causal_structured_frame(
                self.shared.hand_tactile_ring,
                source_field="source_monotonic_ns",
                anchor_monotonic_ns=anchor_ns,
            )
        except Exception:
            tactile_result = None
        if tactile_result is not None:
            candidate_tactile, _tactile_publish_ns, _tactile_sequence = tactile_result
            tactile = candidate_tactile[0]
            tactile_source_ns = int(tactile["source_monotonic_ns"])
            tactile_calibrated = bool(tactile["calibrated"])
            tactile_unit_code = int(tactile["unit_code"])
            tactile_fresh = bool(
                bool(tactile["fresh"])
                and 0 < tactile_source_ns <= anchor_ns
                and anchor_ns - tactile_source_ns <= _EVALUATION_TACTILE_MAX_AGE_NS
                and np.all(
                    np.isfinite(np.asarray(tactile["tactile_force"], dtype=np.float64))
                )
            )
            if tactile_fresh:
                tactile_state = candidate_tactile

        try:
            hand_fk = self._evaluation_hand_kinematics()
            if not hand_fk.is_ready():
                return None, "hardware: fingertip kinematics is unavailable", True
            state = build_episode_state(
                arm_state,
                hand_state,
                tactile_state,
                hand_fk=hand_fk,
                handbase_position_eef_m=np.asarray(
                    self.runtime.hand.T_eef_handbase_pos_xyz,
                    dtype=np.float64,
                ),
                handbase_quat_eef_wxyz=np.asarray(
                    self.runtime.hand.T_eef_handbase_quat_wxyz,
                    dtype=np.float64,
                ),
                timestamp_s=anchor_ns / 1e9,
            )
        except Exception as exc:
            return None, f"hardware: state assembly failed ({type(exc).__name__})", True
        for field_name in (
            "arm_qpos",
            "arm_qvel",
            "arm_tau",
            "eef_pos",
            "eef_rot6d",
            "hand_qpos",
            "hand_current",
            "hand_tactile_sum",
            "hand_tactile_force",
            "fingertip_pos",
        ):
            if not np.all(np.isfinite(np.asarray(getattr(state, field_name)))):
                return None, f"hardware: {field_name} is non-finite", True

        source_ns = np.array(
            [arm_source_ns, hand_source_ns, 0, camera_source_ns],
            dtype=np.uint64,
        )
        receive_ns = np.array(
            [
                int(arm_publish_ns),
                int(hand_publish_ns),
                0,
                int(camera_frame["receive_monotonic_ns"]),
            ],
            dtype=np.uint64,
        )
        source_valid = np.array([True, True, False, True], dtype=bool)
        source_ages_s = np.full(4, np.nan, dtype=np.float64)
        source_ages_s[source_valid] = (
            anchor_ns - source_ns[source_valid].astype(np.int64)
        ) / 1e9
        source_skew_s = np.full(4, np.nan, dtype=np.float64)
        newest_source_ns = int(np.max(source_ns[source_valid]))
        source_skew_s[source_valid] = (
            newest_source_ns - source_ns[source_valid].astype(np.int64)
        ) / 1e9
        signals = {
            "observation_id": anchor_ns,
            "observation_anchor_monotonic_ns": anchor_ns,
            "arm_source_sequence": int(arm_sequence),
            "hand_source_sequence": int(hand_sequence),
            "vr_source_sequence": 0,
            "camera_source_sequence": int(camera_frame["ring_sequence"]),
            "arm_source_monotonic_ns": arm_source_ns,
            "hand_source_monotonic_ns": hand_source_ns,
            "vr_source_monotonic_ns": 0,
            "camera_source_monotonic_ns": camera_source_ns,
            "arm_publish_monotonic_ns": int(arm_publish_ns),
            "hand_publish_monotonic_ns": int(hand_publish_ns),
            "vr_publish_monotonic_ns": 0,
            "camera_publish_monotonic_ns": int(camera_frame["publish_monotonic_ns"]),
            "observation_source_receive_monotonic_ns": receive_ns,
            "observation_source_age_s": source_ages_s,
            "observation_source_skew_s": source_skew_s,
            "observation_history_valid_mask": source_valid[:, None],
            # The raw recording observation includes VR provenance.  Policy
            # observation provenance has separate camera-aligned semantics, so
            # state-only evaluation keeps both contracts explicitly invalid.
            "observation_valid": False,
            "observation_skew_s": float(np.nanmax(source_skew_s, initial=0.0)),
            "policy_observation_valid": False,
            "policy_observation_skew_s": np.nan,
            "hand_accepted_target_action_id": int(hand["accepted_target_action_id"]),
            "tactile_fresh": tactile_fresh,
            "tactile_source_monotonic_ns": tactile_source_ns,
            "tactile_calibrated": tactile_calibrated,
            "tactile_unit_code": tactile_unit_code,
            "pointcloud_valid_depth_ratio": valid_depth_ratio,
        }
        return _EvaluationFrameInputs(state, camera_frame, signals), "", False

    def _evaluation_hold_action(self, state: EpisodeState) -> EpisodeAction:
        return EpisodeAction(
            arm_qpos_cmd=np.asarray(state.arm_qpos, dtype=np.float64),
            hand_qpos_cmd=np.asarray(state.hand_qpos, dtype=np.float64),
            target_eef_pos=np.asarray(state.eef_pos, dtype=np.float64),
            target_eef_rot6d=np.asarray(state.eef_rot6d, dtype=np.float64),
        )

    def _evaluation_action_from_command(
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

    def _record_evaluation_frame(
        self,
        inputs: _EvaluationFrameInputs,
        action: EpisodeAction,
        *,
        signals: dict[str, Any],
        diagnostics: dict[str, Any] | None = None,
        arm_qpos_sent: np.ndarray | None = None,
    ) -> bool:
        if self.recorder is None or self.run_generation is None:
            return False
        try:
            return self.recorder.add_frame(
                inputs.state,
                action,
                self._evaluation_vr_sentinel(),
                camera_frame=inputs.camera_frame,
                signals={**inputs.signals, **signals},
                arm_qpos_sent=arm_qpos_sent,
                diagnostics=diagnostics,
                control_run_generation=self.run_generation,
            )
        except Exception:
            logger.error(
                "executor: evaluation sample construction failed", exc_info=True
            )
            return False

    def _record_initial_evaluation_sample(self, now_ns: int) -> bool:
        inputs, reason, fatal = self._build_evaluation_frame_inputs(now_ns)
        if inputs is None:
            deadline_ns = self.evaluation_initial_deadline_ns
            if fatal:
                self._fault(
                    reason,
                    evaluation_stop_reason=self._evaluation_stop_reason_for_source_failure(
                        reason
                    ),
                )
            elif deadline_ns is not None and now_ns >= deadline_ns:
                self._finish_evaluation_episode(
                    "initial evaluation evidence timeout",
                    stop_reason="eval:invalid:initial_evidence_timeout",
                    recorder_save=False,
                    aborted=True,
                )
            return False
        action = self._evaluation_hold_action(inputs.state)
        recorded = self._record_evaluation_frame(
            inputs,
            action,
            signals={
                "action_queued": False,
                "ik_attempted": False,
                "ik_ok": False,
                "retarget_ok": False,
                "held": True,
                "flag_safety_reject": False,
                "frame_status": _EVALUATION_FRAME_HELD,
            },
        )
        if not recorded:
            self._invalidate_evaluation(
                "evaluation initial sample could not enter RecorderIO",
                stop_reason="eval:invalid:recorder_fault",
                recorder_save=False,
            )
            return False
        self.evaluation_initial_sample_pending = False
        self.evaluation_initial_deadline_ns = None
        if self.sync_mode:
            self.shared.inference_request.set()
        return True

    def _poll_evaluation_recorder(self) -> bool:
        if self.recorder is None:
            return True
        was_stop_pending = self.recorder.stop_pending
        try:
            result = self.recorder.poll_stop()
        except Exception:
            self._invalidate_evaluation(
                "RecorderIO status polling failed",
                stop_reason="eval:invalid:recorder_fault",
                recorder_save=False,
            )
            return False
        if result.phase is RecorderPhase.ERROR or (result.done and result.error):
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
            self._invalidate_evaluation(
                f"RecorderIO failed: {result.error or 'unknown error'}",
                stop_reason="eval:invalid:recorder_fault",
                recorder_save=False,
            )
            return False
        if result.phase is RecorderPhase.FINALIZING and self.run_started_ns is not None:
            if result.reason == "max_frames":
                self._finish_evaluation_episode(
                    "RecorderIO reached its formal-eval frame capacity",
                    stop_reason="eval:invalid:max_frames",
                    aborted=True,
                )
            else:
                self._invalidate_evaluation(
                    "RecorderIO finalized unexpectedly: "
                    f"{result.reason or 'unknown reason'}",
                    stop_reason="eval:invalid:recorder_fault",
                    recorder_save=False,
                )
            return False
        return True

    def _evaluation_raw_action_parts(
        self,
        raw_action: np.ndarray,
        *,
        fallback_action: EpisodeAction,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep finite policy raw fields without relabelling their semantics."""
        if self.policy_spec.action_key == "action":
            raw_arm = np.asarray(raw_action[:7], dtype=np.float64)
            raw_hand = np.asarray(raw_action[7:19], dtype=np.float64)
        else:
            raw_arm = np.asarray(fallback_action.arm_qpos_cmd, dtype=np.float64)
            raw_hand = np.asarray(raw_action[9:21], dtype=np.float64)
        if raw_arm.shape != (7,) or not np.all(np.isfinite(raw_arm)):
            raw_arm = np.asarray(fallback_action.arm_qpos_cmd, dtype=np.float64)
        if raw_hand.shape != (12,) or not np.all(np.isfinite(raw_hand)):
            raw_hand = np.asarray(fallback_action.hand_qpos_cmd, dtype=np.float64)
        return raw_arm, raw_hand

    def _record_evaluation_rejection(
        self,
        inputs: _EvaluationFrameInputs,
        raw_action: np.ndarray,
        *,
        ik_attempted: bool,
        ik_ok: bool,
        kind: _RejectKind,
    ) -> bool:
        action = self._evaluation_hold_action(inputs.state)
        raw_arm, raw_hand = self._evaluation_raw_action_parts(
            raw_action,
            fallback_action=action,
        )
        safety_reject = kind is _RejectKind.SAFETY
        return self._record_evaluation_frame(
            inputs,
            action,
            signals={
                "action_queued": False,
                "ik_attempted": ik_attempted,
                "ik_ok": ik_ok,
                "retarget_ok": False,
                "held": True,
                "flag_safety_reject": safety_reject,
                "frame_status": (
                    _EVALUATION_FRAME_SAFETY_REJECT
                    if safety_reject
                    else _EVALUATION_FRAME_IK_FAIL
                ),
                "action_arm_joint_raw": raw_arm,
            },
            diagnostics={"action_hand_joint_raw": raw_hand},
        )

    def _record_evaluation_command(
        self,
        inputs: _EvaluationFrameInputs,
        candidate: ActionCandidate,
        raw_action: np.ndarray,
    ) -> bool:
        assert candidate.arm_qpos is not None
        assert candidate.hand_qpos is not None
        try:
            action = self._evaluation_action_from_command(
                arm_qpos=candidate.arm_qpos,
                hand_qpos=candidate.hand_qpos,
                raw_action=raw_action,
            )
        except Exception:
            logger.error(
                "executor: failed to build evaluation action record", exc_info=True
            )
            return False
        raw_arm, raw_hand = self._evaluation_raw_action_parts(
            raw_action,
            fallback_action=action,
        )
        is_ee_action = self.policy_spec.action_key == "action_ee"
        return self._record_evaluation_frame(
            inputs,
            action,
            signals={
                "action_id": candidate.action_id,
                "action_created_monotonic_ns": candidate.created_monotonic_ns,
                "action_target_monotonic_ns": candidate.target_monotonic_ns,
                "action_valid_until_monotonic_ns": candidate.valid_until_monotonic_ns,
                "action_queued": True,
                "ik_attempted": is_ee_action,
                "ik_ok": is_ee_action,
                "retarget_ok": False,
                "held": False,
                "flag_safety_reject": False,
                "frame_status": _EVALUATION_FRAME_OK,
                "action_arm_joint_raw": raw_arm,
            },
            diagnostics={"action_hand_joint_raw": raw_hand},
            arm_qpos_sent=candidate.arm_qpos,
        )

    def _record_evaluation_command_evidence(
        self,
        raw_action: np.ndarray,
        candidate: ActionCandidate,
    ) -> None:
        """Record a successful command's evidence after control has committed."""
        if self.recorder is None:
            return
        anchor_ns = time.monotonic_ns()
        inputs, source_reason, _source_fatal = self._build_evaluation_frame_inputs(
            anchor_ns
        )
        self.stats.observe_evaluation_state_build_ms(
            (time.monotonic_ns() - anchor_ns) / 1e6
        )
        if inputs is None:
            self._invalidate_evaluation(
                source_reason,
                stop_reason=self._evaluation_stop_reason_for_source_failure(
                    source_reason
                ),
                recorder_save=True,
            )
            return
        record_start_ns = time.monotonic_ns()
        recorded = self._record_evaluation_command(inputs, candidate, raw_action)
        self.stats.observe_evaluation_record_ms(
            (time.monotonic_ns() - record_start_ns) / 1e6
        )
        if not recorded:
            self._invalidate_evaluation(
                "evaluation command sample could not enter RecorderIO",
                stop_reason="eval:invalid:recorder_fault",
                recorder_save=False,
            )

    def _record_evaluation_rejection_evidence(
        self,
        raw_action: np.ndarray,
        *,
        kind: _RejectKind,
        ik_attempted: bool,
        ik_ok: bool,
    ) -> None:
        """Record a rejected step's evidence after its control slot is committed."""
        if self.recorder is None:
            return
        anchor_ns = time.monotonic_ns()
        inputs, source_reason, _source_fatal = self._build_evaluation_frame_inputs(
            anchor_ns
        )
        self.stats.observe_evaluation_state_build_ms(
            (time.monotonic_ns() - anchor_ns) / 1e6
        )
        if inputs is None:
            self._invalidate_evaluation(
                source_reason,
                stop_reason=self._evaluation_stop_reason_for_source_failure(
                    source_reason
                ),
                recorder_save=True,
            )
            return
        record_start_ns = time.monotonic_ns()
        recorded = self._record_evaluation_rejection(
            inputs,
            raw_action,
            ik_attempted=ik_attempted,
            ik_ok=ik_ok,
            kind=kind,
        )
        self.stats.observe_evaluation_record_ms(
            (time.monotonic_ns() - record_start_ns) / 1e6
        )
        if not recorded:
            self._invalidate_evaluation(
                "evaluation rejection sample could not enter RecorderIO",
                stop_reason="eval:invalid:recorder_fault",
                recorder_save=False,
            )

    def _start_requested_episode(self) -> None:
        if not bool(self.shared.start_request.value):
            return
        if self.recorder is not None and (
            self.recorder.start_pending or self.recorder.stop_pending
        ):
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
            assert self.evaluation_config is not None
            if not self.recorder.start_episode(
                task_label=self.evaluation_config.task_label,
                operator=self.evaluation_config.operator,
            ):
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
                self.shared.is_recording.value = False
                self.recorder.stop_episode(
                    success=False,
                    reason="eval:invalid:start_recheck_failed",
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
                self.shared.is_recording.value = False
                self.recorder.stop_episode(
                    success=False,
                    reason="eval:invalid:start_cancelled",
                )
            return
        if self.execute:
            self.shared.physical_home_completed.value = False
        self.run_started_ns = epoch.started_monotonic_ns
        self.episode_steps = 0
        self._clear_execution(epoch.generation)
        if self.recorder is not None:
            self.evaluation_initial_sample_pending = True
            self.evaluation_initial_deadline_ns = (
                epoch.started_monotonic_ns + self.first_command_timeout_ns
            )
        elif self.sync_mode:
            self.shared.inference_request.set()
        logger.info("policy_executor_loop: RUNNING generation=%d", epoch.generation)

    def _handle_run_boundary(self) -> None:
        if not self._poll_evaluation_recorder():
            return
        if bool(self.shared.quit_requested.value):
            if self.run_started_ns is not None:
                if self.recorder is not None:
                    self._finish_evaluation_episode(
                        "operator quit",
                        stop_reason="eval:invalid:operator",
                        recorder_save=not self.evaluation_initial_sample_pending,
                        aborted=False,
                    )
                else:
                    self._finish_episode("operator quit", aborted=False)
            # Supervisor owns global shutdown. Stay alive until it observes Q,
            # otherwise a clean executor exit can be misclassified as worker death.
            return
        if bool(self.shared.error_state.value) or bool(self.shared.estop_request.value):
            if self.run_started_ns is not None and self.recorder is not None:
                stop_reason = (
                    "eval:failure:estop"
                    if bool(self.shared.estop_request.value)
                    else "eval:invalid:hardware_fault"
                )
                self._finish_evaluation_episode(
                    (
                        "emergency stop"
                        if bool(self.shared.estop_request.value)
                        else "hardware fault"
                    ),
                    stop_reason=stop_reason,
                    recorder_save=not self.evaluation_initial_sample_pending,
                    aborted=False,
                )
            elif self.recorder is not None and self.recorder.is_recording:
                self._fault(
                    "formal recorder was active before a motion epoch faulted",
                    evaluation_stop_reason=(
                        "eval:failure:estop"
                        if bool(self.shared.estop_request.value)
                        else "eval:invalid:hardware_fault"
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
                        "invalid formal evaluation outcome wire value",
                        evaluation_stop_reason="eval:invalid:recorder_fault",
                        recorder_save=False,
                    )
                    return
                if outcome is EvaluationOutcome.NONE:
                    stop_reason = "eval:invalid:operator"
                else:
                    stop_reason = evaluation_outcome_stop_reason(outcome)
                self._finish_evaluation_episode(
                    "operator evaluation stop",
                    stop_reason=stop_reason,
                    recorder_save=not self.evaluation_initial_sample_pending,
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
                    self._finish_evaluation_episode(
                        "motion revoked outside formal stop request",
                        stop_reason="eval:invalid:hardware_fault",
                        recorder_save=not self.evaluation_initial_sample_pending,
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
                _read_command_progress(
                    self.shared, self.runtime, now_ns=now_ns
                )
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
        if prediction.actions.shape[1] != int(self.policy_spec.control_action_dim):
            self._fault("prediction action dimension conflicts with PolicySpec")
            return False
        if self.sync_mode:
            if self.active_prediction is not None:
                self._fault("sync inference published before prediction completion")
                return False
            self.active_prediction = prediction
            self.step_index = 0
            self.schedule_base_ns = now_ns
            return True

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
        self.stats.observe_skipped_prefix_steps(first_index)
        return True

    def _next_due_action(self, now_ns: int) -> tuple[np.ndarray, int, int] | None:
        prediction = self.active_prediction
        if prediction is None or self.schedule_base_ns is None:
            return None

        if not self.sync_mode:
            target_ns = self.schedule_base_ns + self.step_index * self.step_dt_ns
            if now_ns >= target_ns + self.step_dt_ns:
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
        due_ns = (
            target_ns
            if self.next_command_due_ns is None
            else max(target_ns, self.next_command_due_ns)
        )
        if not self.sync_mode and due_ns >= target_ns + self.step_dt_ns:
            return None
        if now_ns < due_ns:
            return None
        return prediction.actions[self.step_index], target_ns, due_ns

    def _consume_control_slot(self, due_ns: int, terminal_ns: int) -> None:
        lateness_ms = max(0, terminal_ns - due_ns) / 1e6
        self.stats.observe_schedule_lateness_ms(lateness_ms)
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
            if self.sync_mode:
                self.shared.inference_request.set()
            return
        if not self.sync_mode:
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

    def _record_terminal_step(
        self,
        *,
        successful: bool,
        candidate: ActionCandidate | None,
    ) -> None:
        self.episode_steps += 1
        limit = self.deployment.max_action_steps
        if limit is not None and self.episode_steps >= limit:
            if self.execute and successful:
                assert candidate is not None
                self.pending_truncation_action_id = candidate.action_id
                return
            if self.recorder is not None:
                self._finish_evaluation_episode(
                    "action_step_limit",
                    stop_reason="eval:failure:action_step_limit",
                    aborted=False,
                )
            else:
                self._finish_episode("action_step_limit", aborted=False)
            return
        self._advance_prediction()

    def _reject_due_step(self, due_ns: int, reason: str) -> None:
        terminal_ns = time.monotonic_ns()
        self._consume_control_slot(due_ns, terminal_ns)
        logger.warning("executor: rejected policy step: %s", reason)
        self._record_terminal_step(successful=False, candidate=None)

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
                self._reject_due_step(due_ns, decode_rejection)
                self._record_evaluation_rejection_evidence(
                    action,
                    kind=reject_kind,
                    ik_attempted=self.policy_spec.action_key == "action_ee",
                    ik_ok=False,
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
            self.stats.observe_publication_interval_ms(interval_ms)
        self.last_publication_ns = publication_ns
        self.last_valid_command_ns = publication_ns
        self._record_terminal_step(successful=True, candidate=published_candidate)
        self._record_evaluation_command_evidence(action, published_candidate)

    def _handle_preparation_rejection(
        self,
        prepared: PreparedCommand,
        candidate: ActionCandidate,
        due_ns: int,
        *,
        raw_action: np.ndarray,
    ) -> None:
        if prepared.unavailable:
            if self.recorder is not None:
                self._fault(
                    "evaluation command preparation lost healthy feedback",
                    evaluation_stop_reason="eval:invalid:hardware_fault",
                )
            return
        if prepared.fatal:
            self._fault(prepared.reason or "physical safety check failed")
            return
        if prepared.gate_code is not None or candidate.hand_qpos is not None:
            self.stats.safety_rejection_count += 1
        self._reject_due_step(due_ns, prepared.reason or "physical safety rejection")
        self._record_evaluation_rejection_evidence(
            raw_action,
            kind=_RejectKind.SAFETY,
            ik_attempted=self.policy_spec.action_key == "action_ee",
            ik_ok=self.policy_spec.action_key == "action_ee",
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
        if self.evaluation_initial_sample_pending:
            # Formal evaluation has one wall-clock budget.  Waiting for the
            # first causal evidence row must not let a short configured trial
            # exceed that budget merely because the first-command watchdog is
            # longer.
            if (
                self.max_running_ns is not None
                and now_ns - self.run_started_ns >= self.max_running_ns
            ):
                self._finish_evaluation_episode(
                    "run time limit before initial evidence",
                    stop_reason="eval:failure:timeout",
                    recorder_save=False,
                    aborted=False,
                )
                return
            self._record_initial_evaluation_sample(now_ns)
            return
        if not self._observe_worker_progress(now_ns):
            return
        if (
            self.max_running_ns is not None
            and now_ns - self.run_started_ns >= self.max_running_ns
        ):
            if self.recorder is not None:
                self._finish_evaluation_episode(
                    "run time limit",
                    stop_reason="eval:failure:timeout",
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
                stop_reason = (
                    "eval:failure:first_command_timeout"
                    if watchdog_reason == "first command timeout"
                    else "eval:failure:command_silence_timeout"
                )
                self._finish_evaluation_episode(
                    watchdog_reason,
                    stop_reason=stop_reason,
                )
            else:
                self._finish_episode(watchdog_reason)
            return
        if self.execute and (
            self.progress.arm_accepted_action_id is None
            or self.progress.hand_accepted_action_id is None
        ):
            return
        if self.pending_truncation_action_id is not None:
            if self.progress.covers(self.pending_truncation_action_id):
                if self.recorder is not None:
                    self._finish_evaluation_episode(
                        "action_step_limit",
                        stop_reason="eval:failure:action_step_limit",
                        aborted=False,
                    )
                else:
                    self._finish_episode("action_step_limit", aborted=False)
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
                self.last_metrics_flush_ns = flush_every(
                    self.stats,
                    last_ns=self.last_metrics_flush_ns,
                    prefix="executor metrics",
                )
                rate.wait()
        finally:
            self.stats.flush(prefix="executor metrics")


def policy_executor_loop(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    policy_spec: Any,
    deployment: PolicyDeploymentConfig,
    execute: bool,
    max_running_s: float | None = None,
    evaluation_config: PolicyEvaluationConfig | None = None,
) -> None:
    """Process entry point for one lightweight policy executor."""
    PolicyExecutor(
        shared,
        runtime,
        policy_spec,
        deployment,
        execute=execute,
        max_running_s=max_running_s,
        evaluation_config=evaluation_config,
    ).run()
