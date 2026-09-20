"""Synchronous policy owner: observe, infer, queue, dispatch at the policy rate.

Model/CUDA and scheduling stay in this child. Hardware SDKs remain in their
workers, behind the existing command publication and final SDK fences.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    PUBLISH_REASON_ESTOP,
    PUBLISH_REASON_FAULT,
    PUBLISH_REASON_FIFO_FULL,
    PUBLISH_REASON_GENERATION,
    PUBLISH_REASON_RUNTIME_STOPPED,
    PUBLISH_REASON_SAFETY_STATE,
    CommandFeedbackSnapshot,
    PreparedCommand,
    PublishResult,
    PublishWaitTracker,
    build_action_candidate,
    command_publishability_reason,
    prepare_command,
    publish_command,
    read_command_feedback,
)
from dexmani_real.control.projection import project_arm_command, project_hand_command
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.deployment.config import (
    FingertipAssemblerConfig,
    PolicyRuntimeConfig,
    RolloutRecordingConfig,
)
from dexmani_real.deployment.inference.observation import (
    _build_observation,
    _select_control_grid_reference_ns,
    _to_policy_observation,
    build_fingertip_runtime,
    observation_sources,
    observation_timing_ms,
)
from dexmani_real.deployment.inference.runtime import PolicyRuntime
from dexmani_real.deployment.metrics import PolicyStats, flush_every
from dexmani_real.ipc.causal import (
    read_camera_frame_causal,
    read_causal_structured_frame,
)
from dexmani_real.ipc.channels import (
    RuntimeChannels,
    read_arm_state_dict,
)
from dexmani_real.planning import (
    OnlineIKConfig,
    Pose,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.planning.paths import WORKSPACE_BOUNDS_TOLERANCE_M
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
)
from dexmani_real.utils.log import ThrottledWarner, get_logger

logger = get_logger(__name__)

_RECORD_FRAME_OK = 0
_RECORD_FRAME_HELD = 1
_RECORD_FRAME_IK_FAIL = 2
_RECORD_FRAME_SAFETY_REJECT = 3


class _RejectKind(Enum):
    """Explicit attribution of a policy-step rejection, never a reason string."""

    IK = auto()
    SAFETY = auto()


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
    """Return the reject-only endpoint workspace predicate for the joint policy.

    The per-control-step command delta is already bounded once by the
    projection (soft jump clip) on top of the operational joint limits, so
    the workspace contract for an ordinary joint policy is the necessary
    endpoint check — not a dense interpolated critic. The bounds and their
    edge tolerance are unchanged; EE policies keep the existing IK profile's
    reference/selection rules (no IK algorithm change).
    """
    bounds = np.asarray(runtime.policy.workspace.as_tuple(), dtype=np.float64)
    arm_fk = make_arm_fk()

    def is_workspace_endpoint_safe(
        start_arm_qpos: np.ndarray, end_arm_qpos: np.ndarray
    ) -> bool:
        eef_position_base, _ = arm_fk.compute(end_arm_qpos)
        position = np.asarray(eef_position_base, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("arm FK returned an invalid workspace position")
        return not (
            np.any(position < bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M)
            or np.any(position > bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M)
        )

    return is_workspace_endpoint_safe


def _build_policy_safety_gate(runtime: ExperimentConfig) -> SafetyGate:
    return SafetyGate(
        arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
        hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
        workspace_check=_build_policy_workspace_check(runtime),
        max_hand_delta_rad=None,
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

    The inference boundary owns flat shape and finite-value validation. An
    ordinary IK no-solution is a RECOVERABLE miss: ``(None, hand, reason)``
    lets the caller drop the unpublished chunk suffix and re-observe in the
    same trial. Contract violations (a missing planner, an illegal rotation
    representation, or an internal solver exception) raise instead — they are
    never swallowed into the recoverable-miss path.
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
        raise RuntimeError("EE policy action reached decode without an IK planner")
    target = Pose(
        p=action[:3],
        q=rot6d_to_quat_wxyz(action[3:9]),
    )
    result = planner.solve_teleop_ik(target, current_arm_qpos, reference)
    if not result.success or result.qpos is None:
        return None, hand_qpos, result.reason or "EE IK found no usable solution"
    return np.asarray(result.qpos, dtype=np.float64), hand_qpos, ""


def _project_policy_targets(
    target_arm_qpos: np.ndarray,
    target_hand_qpos: np.ndarray,
    reference_arm_qpos: np.ndarray,
    runtime: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Project finite physical endpoints through the single-owner projection.

    Canonicalization, operational bounds, the soft delta clip, and the
    float64 round-off guard all live in ``control/projection.py`` — the one
    owner shared with the teleop producer. Workers retain final hard-limit
    SDK authority but never re-reject the same soft threshold. A raised
    ``ValueError`` is a projector-invariant/contract violation, not an
    ordinary recoverable IK miss.
    """
    arm = project_arm_command(
        target_arm_qpos,
        reference_arm_qpos,
        joint_lower_rad=runtime.arm.joint_limit_lower,
        joint_upper_rad=runtime.arm.joint_limit_upper,
        max_command_jump_rad=float(runtime.arm.max_servo_command_jump_rad),
    )
    hand = project_hand_command(
        target_hand_qpos,
        qpos_min_rad=runtime.hand.qpos_min_rad,
        qpos_max_rad=runtime.hand.qpos_max_rad,
    )
    return arm, hand


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
            logger.critical("policy: failed to fence episode into ARMED (%s)", reason)
    stats.flush(prefix="policy final metrics", debug=True)
    stats.log_summary()
    if aborted:
        logger.warning("policy: episode ended: %s", reason)
    else:
        logger.info("policy: episode ended: %s", reason)


class PolicyRunner:
    """One process-local owner of learned-policy scheduling and publication."""

    def __init__(
        self,
        shared: RuntimeChannels,
        runtime: ExperimentConfig,
        policy_spec: Any,
        *,
        model_runtime: PolicyRuntime,
        fingertip_runtime: Any,
        execute: bool,
        max_running_s: float | None,
        num_trials: int = 1,
        recording_config: RolloutRecordingConfig | None = None,
    ) -> None:
        self.shared = shared
        self.runtime = runtime
        self.policy_spec = policy_spec
        self.model_runtime = model_runtime
        self.fingertip_runtime = fingertip_runtime
        self.actions: deque[np.ndarray] = deque()
        self.observation_id = 0
        self.execute = execute
        self.max_running_s = max_running_s
        if type(num_trials) is not int or isinstance(num_trials, bool) or num_trials < 1:
            raise ValueError("num_trials must be a positive integer")
        if recording_config is not None:
            if not isinstance(recording_config, RolloutRecordingConfig):
                raise TypeError("recording_config must be a RolloutRecordingConfig")
            if not execute:
                raise ValueError("recorded rollout requires execute=True")
            # The recorder capacity contract is derived from the RUN budget by
            # the lifecycle; recording never owns or re-validates the plan.
            if self.max_running_s is None:
                raise ValueError(
                    "recorded rollout requires an explicit max_running_s run budget"
                )
        self.num_trials = int(num_trials)
        self.recording_config = recording_config
        self.recorder = RecorderClient(shared) if recording_config is not None else None
        self.last_recorded_action: EpisodeAction | None = None
        self.next_record_ns = 0
        self.control_period_s = float(policy_spec.control_dt_s)
        self.poll_period_s = 1.0 / float(runtime.policy.executor_poll_hz)
        self.chunk_sources: dict[str, tuple[int, ...]] = {}
        self.chunk_action_index = 0
        self._decision_recorded = False
        self.step_dt_ns = int(round(self.control_period_s * 1e9))
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
        # Prepared-but-uncommitted queue head during recoverable FIFO
        # backpressure. FULL never rebuilds, re-solves IK, or re-clips this
        # candidate; only a successful commit advances the action index, the
        # continuity reference, and the publication cadence.
        self._pending_dispatch: ActionCandidate | None = None
        self._fifo_wait = PublishWaitTracker("policy")

        self.run_generation: int | None = None
        self.run_started_ns: int | None = None
        self.last_publication_ns: int | None = None
        self.previous_arm_command_qpos: np.ndarray | None = None
        # Transition marker for the visible observation WAIT/RESUME pair.
        self._observation_waiting_since_ns: int | None = None
        # Run-owner trial budget: a trial counts exactly once when a truly
        # begun trial ends; the saved-episode count is independent evidence
        # status, never the termination condition.
        self.completed_trials = 0
        self.saved_episodes = 0
        # Simple evidence-status fields (no state machine): the START channel
        # became unusable, or at least one evidence error was marked. Neither
        # ever terminates RUNNING control early.
        self.recording_unavailable = False
        self.evidence_failed = False
        self.evidence_failure_reason: str | None = None
        self._evidence_logged_this_trial = False
        self._evidence_warn = ThrottledWarner(interval_s=2.0)
        self._recorder_start_wait_ms = 0.0
        self._pending_stop_reason: str | None = None
        self.last_metrics_flush_ns = time.monotonic_ns()
        # Session research statistics (small in-memory accumulators, reported
        # once in the session summary — no monitoring platform):
        # effective publication Hz uses successful commits over total RUNNING
        # wall duration, and latency mean/p95 use real completed predict
        # samples only.
        self.session_publication_count = 0
        self.session_running_ns = 0
        self.session_inference_ms: list[float] = []

    def _clear_execution(self, generation: int | None) -> None:
        self.run_generation = generation
        self.last_publication_ns = None
        self.previous_arm_command_qpos = None
        self.last_recorded_action = None
        self.actions.clear()
        if self._pending_dispatch is not None or self._fifo_wait.waiting:
            # Epoch/trial boundary: an uncommitted kept candidate is discarded
            # here, so the backpressure span ends with its own visible line and
            # the next trial's [WAIT] starts from a clean state.
            self._fifo_wait.note_dropped("epoch_boundary")
        self._pending_dispatch = None
        self._observation_waiting_since_ns = None
        self.chunk_sources.clear()
        self.chunk_action_index = 0
        self.observation_id = 0

    def _invalidate_chunk(self, reason: str) -> None:
        """Discard the unexecuted chunk suffix after a recoverable break.

        A replanning boundary, not an episode boundary: run_generation,
        observation_id, model episode state, and recording state are
        untouched, as is last_publication_ns (already-occurred physical
        history stays the cadence anchor). The continuity reference is KEPT:
        the next chunk's first action still anchors behind the last committed
        command instead of snapping to measured qpos. Only a new epoch
        (``_clear_execution``) rebuilds the initial reference. The next
        queue-empty iteration builds a fresh causal observation and performs
        a fresh blocking inference.
        """
        # One visible line per real chunk discard; the dropped actions never
        # reach the SDK and the next query replaces them.
        logger.warning(
            "[DROP] policy gen=%s q=%s idx=%s remaining=%d reason=%s",
            self.run_generation,
            self.observation_id,
            self.chunk_action_index,
            len(self.actions),
            reason,
        )
        self.actions.clear()
        if self._pending_dispatch is not None:
            self._pending_dispatch = None
            self._fifo_wait.note_dropped(reason)
        self.chunk_sources.clear()
        self.chunk_action_index = 0

    def _finish_episode(
        self,
        reason: str,
        *,
        aborted: bool = True,
        stop_reason: str = "executor_boundary",
        recorder_save: bool = True,
    ) -> None:
        if self.run_started_ns is None:
            return
        _end_policy_run(
            self.shared,
            reason,
            stats=self.stats,
            aborted=aborted,
        )
        # Trial counting belongs to the run owner: a truly begun trial counts
        # exactly once here, independent of whether its evidence saved.
        self.completed_trials += 1
        self._evidence_logged_this_trial = False
        self.session_running_ns += max(
            0, time.monotonic_ns() - int(self.run_started_ns)
        )
        if self.recorder is not None:
            self._pending_stop_reason = stop_reason
            self.recorder.stop_episode(
                save=recorder_save,
                reason=stop_reason,
            )
        logger.info(
            "policy: trial %d/%d ended (%s); saved_episodes=%d",
            self.completed_trials,
            self.num_trials,
            reason,
            self.saved_episodes,
        )
        if self.completed_trials >= self.num_trials:
            logger.info(
                "policy: all %d requested trials complete; requesting shutdown",
                self.num_trials,
            )
            self.shared.quit_requested.value = True
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

    def _fault(
        self,
        reason: str,
        *,
        stop_reason: str | None = None,
        recorder_save: bool | None = None,
    ) -> None:
        if stop_reason is None:
            stop_reason = "hardware_fault"
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
        self.stats.flush(prefix="policy metrics", debug=True)
        logger.critical("policy: runtime fault: %s", reason)
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
        """Build the recorded action row, keeping EE intent distinct from execution.

        For an EE policy, ``target_eef_pos/rot6d`` record the model's raw EE
        INTENT (the pre-IK command); the executed joints live in
        ``arm_qpos_cmd`` / ``action_arm_joint_sent`` (the projected IK result
        actually committed to the FIFO). The intent is never relabeled as the
        executed EEF pose — the executed Cartesian pose is derivable offline
        by FK over the recorded joints. For a joint policy the recorded EE
        columns are exactly FK(projected joints).
        """
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
                "policy: rollout sample construction failed", exc_info=True
            )
            return False

    def _poll_recorder(self) -> bool:
        """Poll recorder results; in-band evidence errors never end control.

        Only the run-budget-derived recording capacity (``max_frames``) still
        ends the trial, through the ordinary fenced episode boundary. Writer,
        polling, unexpected-finalization, and sampling failures change only
        the evidence state; a required model-input sensor failure is handled
        by its own producer worker (stall latch / heartbeat / error_state),
        never through this evidence path.
        """
        if self.recorder is None:
            return True
        was_stop_pending = self.recorder.stop_pending
        try:
            result = self.recorder.poll_stop()
        except Exception as exc:
            self._mark_evidence_failed(
                f"RecorderIO status polling failed: {type(exc).__name__}: {exc}"
            )
            return True
        ended = False
        if result.error:
            if (
                self.run_started_ns is None
                and not was_stop_pending
                and result.reason == "start_error"
            ):
                # A refused start never began motion; the operator may retry
                # B (physical home is still authorized) or quit.
                logger.warning(
                    "policy: RecorderIO rejected a rollout start: %s",
                    result.error or "unknown error",
                )
                return True
            self._mark_evidence_failed(
                f"RecorderIO failed ({result.reason or 'unknown'}): "
                f"{result.error or 'unknown error'}"
            )
        elif (
            self.recorder.stop_pending or result.done
        ) and self.run_started_ns is not None:
            if result.reason == "max_frames":
                ended = self._invalidate_rollout(
                    "RecorderIO reached its rollout frame capacity",
                    stop_reason="max_frames",
                    recorder_save=True,
                )
            else:
                self._mark_evidence_failed(
                    "RecorderIO finalized unexpectedly: "
                    f"{result.reason or 'unknown reason'}"
                )
        if result.done:
            self._complete_recording(result)
        return not ended

    def _request_failed_session_shutdown(self) -> None:
        """Request a failed-session shutdown; this is not a hardware fault.

        Motion is already fenced by the invalidation that precedes this call;
        the supervisor observes session_failed and runs verified shutdown to
        DISARMED rather than SafetyState.FAULT.
        """
        self.shared.session_failed.value = True
        self.shared.quit_requested.value = True

    def _complete_recording(self, result: RecorderStopResult) -> None:
        """Consume terminal storage status once, after the motion fence.

        ``saved=True`` with no error is itself the save witness: RecorderIO
        never upgrades a discard to a save. The saved count is independent
        evidence status — it never gates control, and a failed finalization
        marks evidence state without requesting quit or a motion fault.
        """
        pending_reason = self._pending_stop_reason
        self._pending_stop_reason = None
        self.shared.is_recording.value = False
        if result.error is not None:
            self._mark_evidence_failed(
                f"rollout recording failed ({result.reason or pending_reason or 'unknown'}): "
                f"{result.error}"
            )
        elif result.saved:
            self.saved_episodes += 1
            print(
                f"Trial {self.completed_trials}/{self.num_trials}: episode saved "
                f"({self.saved_episodes} saved)",
                flush=True,
            )
            logger.info(
                "[RECORD] trial=%d reason=%s saved_rows=%d path=%s",
                self.completed_trials,
                result.reason or pending_reason or "stop",
                result.frame_count,
                result.path or "<unknown>",
            )
        else:
            logger.info(
                "policy: rollout recording discarded (%s)",
                result.reason or pending_reason or "unknown",
            )

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
                "policy: failed to build rollout action record", exc_info=True
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
        """Record a completed control decision, or an ordinary idle sample.

        Command rows are observed immediately so S/Q cannot lose an already
        published command in a second event buffer. The controller emits the
        recording sample; RecorderIO preserves that source row without creating
        a second time grid. No source read here admits or rejects a robot command.
        """
        if (
            self.recorder is None
            or self.run_started_ns is None
            or not self.recorder.is_recording
        ):
            # No recorder, no active trial, or this trial's evidence already
            # ended (refused START / recorder-side failure): skipping rows is
            # not a new failure and never re-reads sources for nothing.
            return
        if raw_action is None and (
            self._decision_recorded or now_ns < self.next_record_ns
        ):
            return
        try:
            sources = {}
            for name, ring in (
                ("arm", self.shared.arm_state_ring),
                ("hand", self.shared.hand_state_ring),
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
                # Keep the causal recording anchor: a later clock would change
                # the age of this historical sample. Report admission health
                # separately from age and expose each stage of its latency.
                health_code = int(camera["camera_health"])
                try:
                    health_name = CameraHealth(health_code).name
                except ValueError:
                    health_name = "UNKNOWN"
                source_ns = int(camera["source_monotonic_ns"])
                receive_ns = int(camera["receive_monotonic_ns"])
                publish_ns = int(camera["publish_monotonic_ns"])
                raise RuntimeError(
                    "recording camera unhealthy or stale: "
                    f"health={health_name}({health_code}) "
                    f"age_ms={camera['camera_age_s'] * 1e3:.3f} "
                    f"max_age_ms={self.runtime.camera.max_frame_age_s * 1e3:.3f} "
                    f"source_to_receive_ms={(receive_ns - source_ns) / 1e6:.3f} "
                    f"receive_to_publish_ms={(publish_ns - receive_ns) / 1e6:.3f} "
                    f"publish_to_anchor_ms={(now_ns - publish_ns) / 1e6:.3f} "
                    f"ring_sequence={camera['ring_sequence']} "
                    f"depth_frame={camera['depth_frame_number']} "
                    f"color_frame={camera['color_frame_number']} "
                    f"generation={camera['camera_generation']} "
                    f"source_ns={source_ns} receive_ns={receive_ns} "
                    f"publish_ns={publish_ns} anchor_ns={now_ns}"
                )
            # One causal hand sample carries qpos/current and both tactile
            # payloads with a single source identity; no backward tactile search.
            state = build_episode_state(
                arm,
                hand,
                timestamp_s=now_ns / 1e9,
            )
            signals = {
                "observation_anchor_monotonic_ns": now_ns,
                # This raw row is not the policy runner's assembled model
                # observation; preserve the stored field without claiming that verdict.
                "observation_valid": False,
                "tracking_error": (
                    float(arm["tracking_err"][0])
                    if "tracking_err" in (arm.dtype.names or ())
                    else np.nan
                ),
                "arm_source_monotonic_ns": int(arm["source_monotonic_ns"][0]),
                "hand_source_monotonic_ns": int(hand["source_monotonic_ns"][0]),
                "vr_source_monotonic_ns": 0,
                "camera_source_monotonic_ns": int(camera["source_monotonic_ns"]),
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
                if not (self.recorder.is_recording or self.recorder.stop_pending):
                    # Evidence for this trial already ended on the recorder
                    # side; skipping rows now is not a new failure.
                    return
                raise RuntimeError("rollout sample rejected by RecorderIO")
            self.next_record_ns = now_ns + self.step_dt_ns
            self._decision_recorded = raw_action is not None
        except Exception as exc:
            # A RUNNING-phase sampling/writer evidence error changes only the
            # evidence state; it never ends control or latches a motion fault.
            self._mark_evidence_failed(
                f"rollout recording sample failed: {exc}", exc_info=True
            )

    def _start_requested_episode(self) -> None:
        if not bool(self.shared.start_request.value):
            return
        if self.recorder is not None and self.recorder.stop_pending:
            with self.shared.motion_lock:
                self.shared.start_request.value = False
            logger.warning("policy: ignored B while RecorderIO is finalizing")
            return
        rejection = _physical_start_pose_rejection(
            self.shared, self.runtime, execute=self.execute
        )
        if rejection is not None:
            with self.shared.motion_lock:
                self.shared.start_request.value = False
            logger.warning("policy: ignored B: %s", rejection)
            return
        if self.recorder is not None and not self.recording_unavailable:
            assert self.recording_config is not None
            # One bounded START preparation while non-RUNNING; no async
            # START and no pre-ACK sample buffer. Reserve the transaction
            # before the bounded wait so Q cannot let lifecycle shutdown race
            # an in-flight recorder acknowledgement.
            self.shared.is_recording.value = True
            recorder_start_ns = time.monotonic_ns()
            started = self.recorder.start_episode(
                task_label=self.recording_config.task_label,
                operator=self.recording_config.operator,
                episode_name=f"episode_{self.completed_trials + 1:03d}",
            )
            self._recorder_start_wait_ms = (
                time.monotonic_ns() - recorder_start_ns
            ) / 1e6
            if not started:
                self.shared.is_recording.value = self.recorder.stop_pending
                if self.recorder.transport_unavailable:
                    # A timed-out/corrupted START channel is never reused
                    # while its finalizer may still be unfinished, never
                    # auto-restarted, and no second same-owner writer is
                    # opened: later trials simply run without recording.
                    self.recording_unavailable = True
                    self._mark_evidence_failed(
                        "RecorderIO START channel unusable; this session records "
                        f"no further trials: {self.recorder.last_error or 'unknown error'}"
                    )
                else:
                    # A refused START makes THIS trial's evidence unavailable;
                    # it never rejects a valid B.
                    self._mark_evidence_failed(
                        "RecorderIO refused the recording START; the trial runs "
                        f"without recording: {self.recorder.last_error or 'unknown error'}"
                    )
                # Fall through: after the preparation the lifecycle rechecks
                # B/S/Q, generation, and start state below before RUNNING.
            else:
                self.shared.is_recording.value = True
                rejection = _physical_start_pose_rejection(
                    self.shared, self.runtime, execute=self.execute
                )
                if rejection is not None:
                    with self.shared.motion_lock:
                        self.shared.start_request.value = False
                    self.recorder.stop_episode(
                        save=False,
                        reason="start_recheck_failed",
                    )
                    logger.warning("policy: cancelled rollout START: %s", rejection)
                    return
        elif self.recorder is not None:
            # Recording is unavailable for this session: the trial still runs.
            self.shared.is_recording.value = False

        if self.recorder is None:
            epoch = begin_requested_motion(self.shared)
        else:
            # B, S/Q, and the RUNNING transition share one RLock so a stop
            # ordered by the operator cannot be clobbered by a concurrent
            # start.
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
                    epoch = begin_requested_motion(self.shared)
        if epoch is None:
            if self.recorder is not None:
                self.recorder.stop_episode(
                    save=False,
                    reason="start_cancelled",
                )
            return
        if self.execute:
            self.shared.physical_home_completed.value = False
        self.run_started_ns = epoch.started_monotonic_ns
        self.stats = PolicyStats()
        self.last_metrics_flush_ns = epoch.started_monotonic_ns
        self.next_record_ns = epoch.started_monotonic_ns + self.step_dt_ns
        self._evidence_logged_this_trial = False
        self._clear_execution(epoch.generation)
        self.model_runtime.reset_episode()
        recording_note = (
            "  [evidence unavailable — running without recording]"
            if self.recorder is not None and not self.recorder.is_recording
            else ""
        )
        print(
            f"Trial {self.completed_trials + 1}/{self.num_trials} RUNNING"
            f"{recording_note}",
            flush=True,
        )
        logger.debug("policy_runner_loop: RUNNING generation=%d", epoch.generation)

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
                        stop_reason="quit",
                        recorder_save=True,
                        aborted=False,
                    )
                else:
                    self._finish_episode("operator quit", aborted=False)
            # Supervisor owns global shutdown. Stay alive until it observes Q,
            # otherwise a clean policy exit can be misclassified as worker death.
            return
        if bool(self.shared.error_state.value) or bool(self.shared.estop_request.value):
            revoke_motion(self.shared, SafetyState.FAULT)
            if self.run_started_ns is not None and self.recorder is not None:
                stop_reason = (
                    "estop"
                    if bool(self.shared.estop_request.value)
                    else "hardware_fault"
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
                        "estop"
                        if bool(self.shared.estop_request.value)
                        else "hardware_fault"
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
                self._finish_episode(
                    "operator stop",
                    stop_reason="operator",
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
                self._finish_episode(
                    "motion revoked outside formal stop request",
                    stop_reason="executor_boundary",
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

    def _handle_recoverable_miss(
        self,
        reason: str,
        *,
        reject_kind: _RejectKind,
        raw_action: np.ndarray,
    ) -> None:
        """Drop the unpublished chunk suffix and re-observe in the SAME trial.

        An ordinary IK no-solution or workspace miss is a replanning boundary,
        never a trial boundary: nothing is published for this action, while
        the committed FIFO prefix, the continuity reference, the run
        generation, and the model episode state stay untouched. The rejection
        row is recorded and the chunk drop prints one visible line. There is
        no first-miss terminalization and no consecutive-miss cap — the run
        budget or the operator ends the trial.
        """
        self.stats.count_rejection(reason)
        self._record_rollout_tick(
            time.monotonic_ns(), raw_action=raw_action, reject_kind=reject_kind
        )
        self._invalidate_chunk(reason)

    def _session_failure(self, reason: str, *, log_exc: bool = False) -> None:
        """End the trial on a model/contract failure — not a hardware fault.

        Motion is fenced into ARMED and the session is marked failed so the
        supervisor runs verified non-FAULT shutdown. Internal contract
        violations (decode, rotation representation, projector invariants,
        preparation) are never swallowed as recoverable IK misses and never
        presented as physical faults. The recording prefix up to the failure
        is technically valid evidence and stays saved with its technical stop
        reason; quality selection happens offline.
        """
        self._invalidate_rollout(
            reason, stop_reason="policy_failure", recorder_save=True
        )
        self._request_failed_session_shutdown()
        if log_exc:
            logger.critical("policy: %s", reason, exc_info=True)
        else:
            logger.critical("policy: %s", reason)

    def _mark_evidence_failed(self, reason: str, *, exc_info: bool = False) -> None:
        """Move recording/evidence state to failed without touching control.

        Sampling, writer, result-polling, and finalization errors during
        RUNNING change only this evidence state: they never call the control
        termination path, never request quit, and never latch a motion fault.
        The session RESULT becomes non-zero at the natural end of control.
        The first failure of each trial logs one full visible line; repeats
        are throttled to avoid flooding at control rate.
        """
        self.evidence_failed = True
        if self.evidence_failure_reason is None:
            self.evidence_failure_reason = reason
        if not self._evidence_logged_this_trial:
            self._evidence_logged_this_trial = True
            logger.error(
                "[EVIDENCE] recording failed; control continues: %s",
                reason,
                exc_info=exc_info,
            )
        else:
            self._evidence_warn("[EVIDENCE] recording still failing: %s", reason)

    def _decode_action(
        self, action: np.ndarray, feedback: CommandFeedbackSnapshot
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, _RejectKind | None, str | None]:
        """Decode/IK one action against the caller-selected feedback snapshot.

        Side-effect free with respect to feedback I/O: the caller has already
        read and validated ``feedback`` once for this dispatch.
        """
        if self.policy_spec.action_key == "action_ee" and self.ee_planner is None:
            self._session_failure("EE policy runner has no IK planner")
            return None, None, None
        try:
            arm_qpos, hand_qpos, rejection = decode_policy_action(
                action,
                self.policy_spec,
                feedback.arm_qpos,
                previous_arm_command_qpos=self.previous_arm_command_qpos,
                planner=self.ee_planner,
            )
        except Exception as exc:
            # Contract violation (illegal rotation representation, internal
            # solver exception): a session failure, never a recoverable miss.
            self._session_failure(
                f"policy action decode contract violation: "
                f"{type(exc).__name__}: {exc}",
                log_exc=True,
            )
            return None, None, None
        if arm_qpos is None:
            # Ordinary IK no-solution: recoverable miss in the same trial.
            if self.policy_spec.action_key == "action_ee":
                self.stats.ik_rejection_count += 1
            return (
                None,
                _RejectKind.IK,
                rejection or "EE action has no usable IK solution",
            )
        reference_arm_qpos = (
            feedback.arm_qpos
            if self.previous_arm_command_qpos is None
            else self.previous_arm_command_qpos
        )
        try:
            arm_qpos, hand_qpos = _project_policy_targets(
                arm_qpos, hand_qpos, reference_arm_qpos, self.runtime
            )
        except Exception as exc:
            self._session_failure(
                f"policy target projection invariant violation: "
                f"{type(exc).__name__}: {exc}",
                log_exc=True,
            )
            return None, None, None
        return (arm_qpos, hand_qpos), None, None

    def _prepare_dispatch_candidate(self, action: np.ndarray) -> ActionCandidate | None:
        """Decode, project, and gate the queue head exactly once.

        decode/IK and SafetyGate consume the SAME immutable feedback snapshot
        selected here. Feedback truthfulness and causality are enforced; the
        age-only veto is not — an old-but-real control state stays usable and
        worker liveness is owned by supervision, so a slow predict with
        advancing producers is never a device failure. An unavailable ring
        (no committed sample yet) invalidates the pending chunk and waits.
        """
        feedback, reason, issue = read_command_feedback(
            self.shared,
            require_hand=True,
            arm_max_age_s=None,
            hand_max_age_s=None,
        )
        if feedback is None:
            if issue is None or issue.code is FeedbackIssueCode.STALE:
                self._invalidate_chunk("command_feedback_unavailable")
                return None
            self._fault(f"fatal command feedback: {issue.code.value}: {issue.detail}")
            return None

        decoded, reject_kind, decode_rejection = self._decode_action(action, feedback)
        if decoded is None:
            if decode_rejection is not None:
                assert reject_kind is not None
                self._handle_recoverable_miss(
                    decode_rejection, raw_action=action, reject_kind=reject_kind
                )
            return None
        arm_qpos, hand_qpos = decoded
        candidate = build_action_candidate(
            self.shared,
            arm_qpos,
            hand_qpos,
            run_generation=self.run_generation,
            is_hold=False,
        )
        try:
            prepared = prepare_command(
                self.shared,
                candidate,
                gate=self.gate,
                arm_feedback_max_age_s=None,
                hand_feedback_max_age_s=None,
                feedback_snapshot=feedback,
            )
        except Exception as exc:
            self._session_failure(
                f"policy command preparation contract violation: "
                f"{type(exc).__name__}: {exc}",
                log_exc=True,
            )
            return None
        if not prepared.accepted:
            self._handle_preparation_rejection(prepared, raw_action=action)
            return None
        candidate = prepared.candidate
        assert candidate is not None
        return candidate

    def _dispatch_action(self, action: np.ndarray) -> None:
        """Commit the queue head; a FULL FIFO keeps the identical candidate.

        Preparation happens exactly once per action. A FULL commit result is
        recoverable backpressure: the same immutable prepared candidate is
        retried from the main poll cadence without rebuilding, re-solving IK,
        or re-clipping, and only a successful commit advances the action
        index, the continuity reference, and the actual-publication cadence.
        """
        if self._pending_dispatch is None:
            candidate = self._prepare_dispatch_candidate(action)
            if candidate is None:
                return
            self._pending_dispatch = candidate
        candidate = self._pending_dispatch

        publication_check_ns = time.monotonic_ns()
        if not self._running_generation_is_live() or self._running_time_expired(
            publication_check_ns
        ):
            return
        if self.execute:
            result = publish_command(
                self.shared,
                candidate,
                required_safety_state=SafetyState.RUNNING,
            )
            if not result.published:
                self._handle_publication_rejection(result)
                return
            if result.command is None or result.command.published_monotonic_ns <= 0:
                self._fault("physical publication omitted its command receipt/timestamp")
                return
            publication_ns = int(result.command.published_monotonic_ns)
        else:
            reason = command_publishability_reason(
                self.shared,
                candidate,
                required_safety_state=SafetyState.RUNNING,
            )
            if reason:
                self._handle_publication_rejection(PublishResult(False, reason=reason))
                return
            publication_ns = time.monotonic_ns()

        self._fifo_wait.note_committed()
        self._pending_dispatch = None
        assert candidate.arm_qpos is not None
        self.stats.publication_input_age_ms = max(
            ((publication_ns - times[-1]) / 1e6 for times in self.chunk_sources.values()),
            default=0.0,
        )
        logger.debug(
            "policy publish generation=%s query=%s index=%s planned_ns=%s actual_ns=%s input_age_ms=%s raw=%s arm=%s hand=%s chunk_arm_delta=%s",
            self.run_generation,
            self.observation_id,
            self.chunk_action_index,
            (
                None
                if self.last_publication_ns is None
                else self.last_publication_ns + self.step_dt_ns
            ),
            publication_ns,
            self.stats.publication_input_age_ms,
            action.tolist(),
            candidate.arm_qpos.tolist(),
            candidate.hand_qpos.tolist(),
            (
                (candidate.arm_qpos - self.previous_arm_command_qpos).tolist()
                if self.chunk_action_index == 0
                and self.previous_arm_command_qpos is not None
                else None
            ),
        )
        self.previous_arm_command_qpos = candidate.arm_qpos.copy()
        if self.last_publication_ns is not None:
            self.stats.publication_interval_ms = (
                publication_ns - self.last_publication_ns
            ) / 1e6
        self.last_publication_ns = publication_ns
        self.session_publication_count += 1
        self.actions.popleft()
        self.chunk_action_index += 1
        self._record_rollout_tick(
            publication_ns, candidate=candidate, raw_action=action
        )

    def _handle_preparation_rejection(
        self,
        prepared: PreparedCommand,
        *,
        raw_action: np.ndarray,
    ) -> None:
        if prepared.unavailable:
            return
        if prepared.gate_code is GateRejectCode.WORKSPACE:
            # Ordinary workspace miss: a recoverable replanning boundary in
            # the same trial, with the committed prefix and reference kept.
            self.stats.safety_rejection_count += 1
            self._handle_recoverable_miss(
                prepared.reason or "policy workspace violation",
                raw_action=raw_action,
                reject_kind=_RejectKind.SAFETY,
            )
            return
        # Anything else after the single-owner projection (a joint-limit
        # rejection, a failed collision/workspace check, a fatal contract
        # break) is an internal invariant violation: session failure — never
        # a recoverable miss and never presented as a hardware fault.
        self._session_failure(
            prepared.reason or "post-projection safety invariant failed"
        )

    def _handle_publication_rejection(self, result: PublishResult) -> None:
        if result.reason == PUBLISH_REASON_FIFO_FULL:
            # Recoverable backpressure: keep the identical prepared candidate
            # and retry from the poll cadence. One visible [WAIT] line per
            # continuous full span; STOP/fault/timeout keep priority because
            # the main loop polls them between retries.
            keep_action = (
                self._pending_dispatch.action_id
                if self._pending_dispatch is not None
                else 0
            )
            self._fifo_wait.note_full(result.fifo_depth, keep_action)
            return
        if result.reason in {PUBLISH_REASON_ESTOP, PUBLISH_REASON_FAULT}:
            return
        # A concurrent S/generation fence is an ordinary episode boundary.  The
        # next loop observes the operator request before any further command;
        # the pending candidate belongs to the revoked epoch and is dropped by
        # the boundary handling, never committed.
        if result.reason in {
            PUBLISH_REASON_GENERATION,
            PUBLISH_REASON_RUNTIME_STOPPED,
        } or (result.reason.startswith(PUBLISH_REASON_SAFETY_STATE)):
            return
        self._fault(
            f"unrecognized publication rejection: {result.reason or 'missing reason'}"
        )

    def _running_generation_is_live(self) -> bool:
        snapshot = read_run_state_snapshot(self.shared)
        return (
            snapshot.state is SafetyState.RUNNING
            and snapshot.generation == self.run_generation
            and snapshot.stop_request == int(StopRequest.NONE)
            and bool(self.shared.is_running.value)
            and not bool(self.shared.quit_requested.value)
            and not bool(self.shared.error_state.value)
            and not bool(self.shared.estop_request.value)
        )

    def _running_time_expired(self, now_ns: int) -> bool:
        assert self.run_started_ns is not None
        if (
            self.max_running_ns is not None
            and now_ns - self.run_started_ns >= self.max_running_ns
        ):
            self._finish_episode("run time limit", stop_reason="timeout", aborted=False)
            return True
        return False

    def _next_control_boundary_ns(self) -> int | None:
        """Return the next action-cadence boundary, anchored to actual publication.

        ``None`` means no command has been published yet (episode first chunk):
        the queue may be filled immediately.  Otherwise a command must occupy one
        full ``step_dt_ns`` before the next dispatch or fresh observation query.
        """
        if self.last_publication_ns is None:
            return None
        return self.last_publication_ns + self.step_dt_ns

    def _run_active_tick(self, now_ns: int) -> None:
        assert self.run_started_ns is not None
        if not self._running_generation_is_live():
            # Batch-invalidate this epoch's uncommitted predictions with one
            # visible line; the queued FIFO records are invalidated under the
            # motion lock by the revocation itself.
            if self.actions or self._pending_dispatch is not None:
                logger.warning(
                    "[DROP] policy q=%s uncommitted_actions=%d reason=generation_revoked",
                    self.run_generation,
                    len(self.actions),
                )
                self._fifo_wait.note_dropped("generation_revoked")
            self.actions.clear()
            self._pending_dispatch = None
            self._handle_run_boundary()
            # Also end a RUNNING epoch whose generation changed externally.
            if (
                self.run_started_ns is not None
                and not self._running_generation_is_live()
            ):
                self._finish_episode("motion generation changed")
            return
        if self._running_time_expired(now_ns):
            return

        # Action cadence is anchored solely to the previous physical publication.
        # Neither a chunk[1:] dispatch nor a queue-empty replan may start before
        # that publication has occupied one full control period.
        boundary_ns = self._next_control_boundary_ns()
        if boundary_ns is not None and now_ns < boundary_ns:
            return

        if self.actions:
            self._dispatch_action(self.actions[0])
            return

        # Queue empty and boundary reached: fresh synchronous replan.
        self.observation_id += 1
        observation = _build_observation(
            self.shared,
            self.policy_spec,
            observation_id=self.observation_id,
            run_generation=self.run_generation,
            run_started_ns=self.run_started_ns,
            anchor_ns=time.monotonic_ns(),
            step_dt_ns=self.step_dt_ns,
        )
        if observation is None:
            # Required-history-unavailable is an explicit WAIT for the next
            # poll: one visible transition line (never per-poll spam), and
            # never a DROP — no prepared action was discarded here.
            if self._observation_waiting_since_ns is None:
                self._observation_waiting_since_ns = time.monotonic_ns()
                logger.warning(
                    "[WAIT] policy observation q=%d reason=required_history_unavailable",
                    self.observation_id,
                )
            return
        if self._observation_waiting_since_ns is not None:
            logger.info(
                "[RESUME] policy observation q=%d wait_ms=%.0f",
                self.observation_id,
                (time.monotonic_ns() - self._observation_waiting_since_ns) / 1e6,
            )
            self._observation_waiting_since_ns = None
        self.stats.observation_age_ms, self.stats.observation_skew_ms = (
            observation_timing_ms(observation)
        )
        policy_observation = _to_policy_observation(
            observation,
            self.policy_spec,
            fingertip_runtime=self.fingertip_runtime,
        )
        if not self._running_generation_is_live():
            return
        started_ns = time.monotonic_ns()
        if self._running_time_expired(started_ns):
            return
        try:
            predicted = self.model_runtime.predict(policy_observation)
        except Exception as exc:
            # Model/contract failure (CUDA OOM, forward error, shape/NaN/Inf) is a
            # session failure, not a physical fault: fence, mark failed, and let
            # verified shutdown complete.
            self._session_failure(f"policy inference failed: {exc}", log_exc=True)
            return
        finished_ns = time.monotonic_ns()
        self.stats.inference_latency_ms = (finished_ns - started_ns) / 1e6
        # Real completed predict sample for the session mean/p95 statistics.
        self.session_inference_ms.append(self.stats.inference_latency_ms)
        # Main can revoke motion during blocking inference. Never accept its
        # result before rechecking the episode's original generation/state.
        if not self._running_generation_is_live():
            self._invalidate_chunk("inference_run_boundary")
            return
        if self._running_time_expired(finished_ns) or not self._poll_recorder():
            return
        self.chunk_sources = observation_sources(observation)
        self.chunk_action_index = 0
        logger.debug(
            "policy query generation=%s query=%s anchor_ns=%s references_ns=%s sources_ns=%s inference_start_ns=%s inference_end_ns=%s predicted=%s",
            self.run_generation,
            self.observation_id,
            observation.anchor_monotonic_ns,
            _select_control_grid_reference_ns(
                run_started_ns=self.run_started_ns,
                anchor_ns=observation.anchor_monotonic_ns,
                history_len=int(self.policy_spec.n_obs_steps),
                step_dt_ns=self.step_dt_ns,
            )[0].tolist(),
            self.chunk_sources,
            started_ns,
            finished_ns,
            predicted.tolist(),
        )
        self.actions.extend(predicted)
        # The boundary elapsed before observation/inference began; dispatch the
        # first action immediately without waiting another control period.
        self._dispatch_action(self.actions[0])

    def run(self) -> None:
        """Poll lifecycle while preparing chunks and waiting for publication deadlines."""
        try:
            while self.shared.is_running.value:
                tick_started_ns = time.monotonic_ns()
                self._decision_recorded = False
                self._recorder_start_wait_ms = 0.0
                phase_ms = {"active_control": 0.0, "recording": 0.0, "metrics": 0.0}
                self.shared.set_heartbeat("policy", time.monotonic())
                boundary_started_ns = time.monotonic_ns()
                self._handle_run_boundary()
                phase_ms["boundary"] = (
                    time.monotonic_ns() - boundary_started_ns
                ) / 1e6
                # Nested in boundary time, not an additional phase to sum.
                phase_ms["boundary_recorder_start_wait"] = self._recorder_start_wait_ms
                run_snapshot = read_run_state_snapshot(self.shared)
                if (
                    self.run_started_ns is not None
                    and run_snapshot.state is SafetyState.RUNNING
                ):
                    active_started_ns = time.monotonic_ns()
                    self._run_active_tick(active_started_ns)
                    phase_ms["active_control"] = (
                        time.monotonic_ns() - active_started_ns
                    ) / 1e6
                    if (
                        self.run_started_ns is not None
                        and self._running_generation_is_live()
                    ):
                        record_started_ns = time.monotonic_ns()
                        self._record_rollout_tick(record_started_ns)
                        phase_ms["recording"] = (
                            time.monotonic_ns() - record_started_ns
                        ) / 1e6
                        metrics_started_ns = time.monotonic_ns()
                        self.last_metrics_flush_ns = flush_every(
                            self.stats,
                            last_ns=self.last_metrics_flush_ns,
                            prefix="policy metrics",
                            debug=True,
                        )
                        phase_ms["metrics"] = (
                            time.monotonic_ns() - metrics_started_ns
                        ) / 1e6
                phase_ms["work"] = (time.monotonic_ns() - tick_started_ns) / 1e6
                logger.debug("policy loop phases_ms=%s", phase_ms)
                wait_s = self.poll_period_s
                boundary_ns = self._next_control_boundary_ns()
                if boundary_ns is not None:
                    remaining_s = (boundary_ns - time.monotonic_ns()) / 1e9
                    if remaining_s > 0:
                        wait_s = min(wait_s, remaining_s)
                time.sleep(max(0.0, wait_s))
        finally:
            # A runtime stop can end the loop before its next boundary poll.
            # Storage finalization remains owned by RecorderIO.
            if self.run_started_ns is not None:
                self._finish_episode("runtime shutdown", stop_reason="runtime_shutdown")
            if self.recorder is not None and (
                self.recorder.stop_pending or self._pending_stop_reason is not None
            ):
                result = self.recorder.join_stop()
                if result.done:
                    self._complete_recording(result)
                else:
                    self._mark_evidence_failed(
                        "rollout recording finalization timed out"
                    )
            if (
                self.evidence_failed
                and not bool(self.shared.error_state.value)
                and not bool(self.shared.estop_request.value)
            ):
                # Evidence failure makes the session RESULT non-zero only after
                # control has ended naturally; it never requested an early end
                # of RUNNING and never overrides a physical fault classification.
                self.shared.session_failed.value = True
            logger.info(
                "policy session summary: trials=%d/%d saved_episodes=%d "
                "evidence_failed=%s%s",
                self.completed_trials,
                self.num_trials,
                self.saved_episodes,
                self.evidence_failed,
                (
                    f" first_reason={self.evidence_failure_reason}"
                    if self.evidence_failure_reason
                    else ""
                ),
            )
            self._log_session_statistics()

    def _log_session_statistics(self) -> None:
        """Report the four session research statistics from real accumulators.

        nominal_hz = 1/control_dt; effective publication Hz = successful
        commits over total RUNNING wall duration (not a physical arrival
        rate); inference mean/p95 use the completed predict samples and are
        reported as ``unavailable`` when there are none.
        """
        nominal_hz = (
            1.0 / self.control_period_s if self.control_period_s > 0 else float("nan")
        )
        running_s = self.session_running_ns / 1e9
        if running_s > 0.0:
            effective_hz: str = f"{self.session_publication_count / running_s:.3f}"
        else:
            effective_hz = "unavailable"
        if self.session_inference_ms:
            samples = np.asarray(self.session_inference_ms, dtype=np.float64)
            mean_ms = f"{float(np.mean(samples)):.3f}"
            p95_ms = f"{float(np.percentile(samples, 95.0)):.3f}"
        else:
            mean_ms = "unavailable"
            p95_ms = "unavailable"
        logger.info(
            "policy session stats: nominal_hz=%.3f effective_publication_hz=%s "
            "publications=%d running_wall_s=%.2f inference_ms_mean=%s "
            "inference_ms_p95=%s predict_samples=%d",
            nominal_hz,
            effective_hz,
            self.session_publication_count,
            running_s,
            mean_ms,
            p95_ms,
            len(self.session_inference_ms),
        )


def _load_policy_runtime(config: PolicyRuntimeConfig) -> PolicyRuntime:
    """Load model/CUDA only inside the policy child, through the public API."""
    from dexmani_policy.deployment import load_experiment

    from dexmani_real.deployment.inference.dexmani_policy import DexManiPolicyAdapter

    loaded = load_experiment(
        config.experiment,
        device=config.device,
        seed=config.seed,
        artifact=config.artifact,
        inference_steps=config.inference_steps,
    )
    try:
        return DexManiPolicyAdapter(loaded, config.spec)
    except BaseException:
        loaded.close()
        raise


def policy_runner_loop(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    config: PolicyRuntimeConfig,
    execute: bool,
    max_running_s: float | None = None,
    num_trials: int = 1,
    recording_config: RolloutRecordingConfig | None = None,
    fingertip_config: FingertipAssemblerConfig | None = None,
) -> None:
    """Load and warm up before READY; own the model until verified shutdown."""
    shared.set_heartbeat("policy", time.monotonic())
    model_runtime = _load_policy_runtime(config)
    try:
        fingertip_runtime = build_fingertip_runtime(config.spec, fingertip_config)
        timings_s = model_runtime.warmup(samples=5)
        logger.debug(
            "policy warmup: samples_ms=%s",
            ",".join(
                f"{value * 1e3:.3f}" for value in timings_s
                if np.isfinite(value) and value >= 0
            ),
        )
        runner = PolicyRunner(
            shared,
            runtime,
            config.spec,
            model_runtime=model_runtime,
            fingertip_runtime=fingertip_runtime,
            execute=execute,
            max_running_s=max_running_s,
            num_trials=num_trials,
            recording_config=recording_config,
        )
        shared.set_heartbeat("policy", time.monotonic())
        shared.set_ready("policy")
        runner.run()
    finally:
        try:
            model_runtime.close()
        except Exception:
            logger.warning("policy: runtime.close raised", exc_info=True)
