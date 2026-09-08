"""One causal teleoperation grid tick from observation through publication."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    PUBLISH_REASON_SAFETY_STATE,
    PreparedCommand,
    PublishResult,
    prepare_joint_command,
    publish_command,
)
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.ipc.causal import (
    read_camera_frame_causal,
    read_causal_structured_frame,
    read_hand_tactile_causal,
    read_structured_frame_aligned_to_source,
    read_vr_frame_causal,
    vr_frame_is_fresh,
)
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.ipc.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.planning import Pose, XArm7MotionPlanner
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.pose import (
    normalize_quat_wxyz,
    quat_wxyz_to_rot6d,
    rot6d_to_quat_wxyz,
)
from dexmani_real.recording.client import RecorderClient
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_loop.action_proposal import (
    compute_arm_joint_proposal,
    compute_hand_joint_proposal,
    compute_target_eef_pose,
)
from dexmani_real.teleop.control_loop.camera_freshness import CameraFreshnessTracker
from dexmani_real.teleop.control_loop.hand_control import (
    HandRetargetObservationCache,
    reset_hand_retargeter,
)
from dexmani_real.teleop.control_loop.timing import StageTimer
from dexmani_real.teleop.control_loop.vr_mapping import VRWristMapper
from dexmani_real.teleop.episode_samples import (
    FRAME_IK_FAIL,
    FRAME_OK,
    FRAME_RETARGET_FAIL,
    FRAME_SAFETY_REJECT,
    record_frame,
    record_held,
    stop_recording,
)
from dexmani_real.utils.feedback import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import ThrottledWarner, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _TeleopCommandLimits:
    """Resolved command bounds owned by the teleoperation control loop."""

    arm_joint_lower_rad: np.ndarray
    arm_joint_upper_rad: np.ndarray
    teleop_arm_max_delta_rad_per_tick: np.ndarray | None
    hand_home_qpos_rad: np.ndarray
    hand_command_lower_rad: np.ndarray
    hand_command_upper_rad: np.ndarray
    # Teleop endpoint shaping bound. The hand worker independently reuses this
    # value for SDK-level slew protection.
    hand_max_delta_rad_per_tick: np.ndarray
    workspace_bounds_world_m: np.ndarray

    @classmethod
    def from_config(cls, config: TeleopConfig) -> "_TeleopCommandLimits":
        arm_lower = np.asarray(config.runtime.arm.joint_limit_lower, dtype=np.float64)
        arm_upper = np.asarray(config.runtime.arm.joint_limit_upper, dtype=np.float64)
        configured_delta = config.runtime.policy.teleop_arm_max_delta_rad_per_tick
        max_delta = (
            None
            if configured_delta is None
            else np.broadcast_to(
                np.asarray(configured_delta, dtype=np.float64), arm_lower.shape
            ).copy()
        )
        hand_lower = np.asarray(config.runtime.hand.qpos_min_rad, dtype=np.float64)
        hand_max_delta = np.broadcast_to(
            np.asarray(
                config.runtime.hand.hand_max_delta_rad_per_tick,
                dtype=np.float64,
            ),
            hand_lower.shape,
        ).copy()
        return cls(
            arm_joint_lower_rad=arm_lower.copy(),
            arm_joint_upper_rad=arm_upper.copy(),
            teleop_arm_max_delta_rad_per_tick=max_delta,
            hand_home_qpos_rad=np.deg2rad(
                np.asarray(config.runtime.hand.home_qpos_deg, dtype=np.float64)
            ),
            hand_command_lower_rad=hand_lower.copy(),
            hand_command_upper_rad=np.asarray(
                config.runtime.hand.qpos_max_rad, dtype=np.float64
            ).copy(),
            hand_max_delta_rad_per_tick=hand_max_delta,
            workspace_bounds_world_m=np.asarray(
                config.runtime.policy.workspace.as_tuple(), dtype=np.float64
            ).copy(),
        )


def _advance_arm_feedback_error_count(
    current_count: int,
    issue: str | None,
    *,
    max_consecutive_errors: int,
) -> tuple[int, bool]:
    """Reset on valid feedback; fault at the configured invalid-frame limit."""
    if issue is None:
        return 0, False
    next_count = current_count + 1
    return next_count, next_count >= max_consecutive_errors


def _prepare_and_publish_joint_command(
    shared: RuntimeChannels,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None,
    *,
    gate: SafetyGate,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
    is_hold: bool = False,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
) -> tuple[PreparedCommand, PublishResult | None]:
    """Run the explicit non-blocking teleop safety/publication path."""
    prepared = prepare_joint_command(
        shared,
        arm_qpos,
        hand_qpos,
        gate=gate,
        is_hold=is_hold,
        observation_id=observation_id,
        observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
        arm_feedback_max_age_s=arm_feedback_max_age_s,
        hand_feedback_max_age_s=hand_feedback_max_age_s,
    )
    if not prepared.accepted:
        return prepared, None
    assert prepared.candidate is not None
    return prepared, publish_command(
        shared,
        prepared.candidate,
        required_safety_state=SafetyState.RUNNING,
    )


@dataclass(frozen=True)
class TeleopGridResources:
    """Read-only dependencies used to execute one control-grid observation."""

    planner: XArm7MotionPlanner
    safety_gate: SafetyGate
    recorder: RecorderClient | None
    command_limits: _TeleopCommandLimits
    camera_freshness: CameraFreshnessTracker
    stage_timer: StageTimer
    validation_warn: ThrottledWarner
    arm_feedback_warn: ThrottledWarner
    hand_ramp_total_frames: int
    max_observation_skew_s: float


@dataclass(frozen=True)
class TeleopGridObservation:
    """One validated causal observation ready for command computation."""

    arm_state: np.ndarray
    arm_ring_sequence: int
    arm_qpos_rad: np.ndarray
    vr_frame: dict[str, Any]
    camera_frame: dict[str, Any] | None
    hand_state: np.ndarray | None
    hand_ring_sequence: int
    hand_tactile: np.ndarray | None
    anchor_monotonic_ns: int
    control_run_generation: int
    policy_observation_signals: dict[str, object] | None


@dataclass(frozen=True)
class TeleopActionComputation:
    """Mapped targets and solver result for one grid tick."""

    target_position_world_m: np.ndarray
    target_quat_world_wxyz: np.ndarray
    hand_qpos_rad: np.ndarray
    hand_retarget_succeeded: bool
    ik_qpos_rad: np.ndarray | None
    ik_failure_reason: str


@dataclass(frozen=True)
class TeleopGridTickResult:
    """Session-owned state changes produced by one causal grid tick."""

    keep_running: bool = True
    pause_reason: str | None = None
    pause_released: bool = False
    recording_active: bool = False
    arm_feedback_error_count: int = 0
    hand_disconnected_at_s: float | None = None


class TeleopController:
    """Own only persistent arm/hand mapping and proposal state."""

    def __init__(
        self,
        *,
        planner: XArm7MotionPlanner,
        arm_mapper: VRWristMapper,
        config: TeleopConfig,
        command_limits: _TeleopCommandLimits,
        initial_arm_qpos_rad: np.ndarray,
        initial_hand_qpos_rad: np.ndarray,
        hand_retargeter: Any = None,
    ) -> None:
        self.planner = planner
        self.arm_mapper = arm_mapper
        self.command_limits = command_limits
        self.hand_enabled = bool(config.runtime.policy.hand_enabled)
        self.ema_alpha_position = float(config.runtime.policy.ema.alpha_pos)
        self.ema_alpha_rotation = float(config.runtime.policy.ema.alpha_rot)
        self.hand_retargeter = hand_retargeter
        self.prev_qpos_cmd = np.asarray(initial_arm_qpos_rad, dtype=np.float64).copy()
        self.prev_hand_qpos = np.asarray(initial_hand_qpos_rad, dtype=np.float64).copy()
        self.ema_prev_pos: np.ndarray | None = None
        self.ema_prev_quat: np.ndarray | None = None
        self.hand_ramp_start: np.ndarray | None = None
        self.hand_ramp_step = 0
        self.hand_retarget_cache = HandRetargetObservationCache()
        self.consecutive_ik_hold_frames = 0
        self.ik_hold_started_s = 0.0
        self.last_target_eef_pos = np.full(3, np.nan)
        self.last_target_eef_rot6d = np.full(6, np.nan)
        self.planner.set_hand_qpos(self.prev_hand_qpos)

    def clear_reference(self) -> None:
        """Drop temporal proposal state at a command-silent boundary."""
        self.arm_mapper.clear()
        self.ema_prev_pos = None
        self.ema_prev_quat = None
        self.hand_ramp_start = None
        self.hand_ramp_step = 0
        self.hand_retarget_cache.reset()

    def reset_reference(
        self,
        arm_state: np.ndarray,
        vr_frame: dict[str, Any],
        hand_state: np.ndarray | None,
    ) -> bool:
        """Re-anchor mapping from fresh measured feedback after a pause."""
        try:
            eef_pos, eef_rot6d = make_arm_fk().compute(
                np.asarray(arm_state["qpos"][0], dtype=np.float64)
            )
            wrist_pos = np.asarray(vr_frame["wrist_pos"], dtype=np.float64)
            wrist_quat = np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)
            self.arm_mapper.reset(
                wrist_pos=wrist_pos,
                wrist_quat_wxyz=wrist_quat,
                eef_pos=eef_pos,
                eef_quat_wxyz=rot6d_to_quat_wxyz(eef_rot6d),
            )
        except Exception:
            self.arm_mapper.clear()
            return False
        if not self.arm_mapper.is_ready():
            return False
        hand_anchor: np.ndarray | None = None
        if self.hand_enabled:
            if hand_state is None or not np.all(np.isfinite(hand_state["qpos"][0])):
                self.arm_mapper.clear()
                return False
            hand_anchor = np.asarray(hand_state["qpos"][0], dtype=np.float64).copy()
            self.prev_hand_qpos = hand_anchor
        self.ema_prev_pos = None
        self.ema_prev_quat = None
        self.hand_ramp_start = hand_anchor
        self.hand_ramp_step = 0
        self.hand_retarget_cache.reset()
        reset_hand_retargeter(self.hand_retargeter, hand_anchor)
        return True

    def compute(
        self,
        observation: TeleopGridObservation,
        resources: TeleopGridResources,
    ) -> TeleopActionComputation | None:
        """Map one validated observation and solve its arm/hand proposal."""
        mapped = self.arm_mapper.map(
            observation.vr_frame["wrist_pos"],
            observation.vr_frame["wrist_quat_wxyz"],
        )
        if mapped is None:
            return None
        target = compute_target_eef_pose(
            mapped["pos"],
            mapped["quat_wxyz"],
            previous_position_world_m=self.ema_prev_pos,
            previous_quat_world_wxyz=self.ema_prev_quat,
            workspace_bounds_world_m=self.command_limits.workspace_bounds_world_m,
            ema_alpha_position=self.ema_alpha_position,
            ema_alpha_rotation=self.ema_alpha_rotation,
        )
        if target.smoothing_state_incomplete:
            logger.warning(
                "teleop_loop: previous EEF quaternion is missing — skipping EMA"
            )
        hand = compute_hand_joint_proposal(
            self.hand_retargeter,
            observation.vr_frame,
            self.prev_hand_qpos,
            hand_available=self.hand_enabled,
            retarget_cache=self.hand_retarget_cache,
            ramp_start_qpos_rad=self.hand_ramp_start,
            ramp_step=self.hand_ramp_step,
            ramp_total_frames=resources.hand_ramp_total_frames,
            command_lower_rad=self.command_limits.hand_command_lower_rad,
            command_upper_rad=self.command_limits.hand_command_upper_rad,
            max_delta_rad_per_tick=self.command_limits.hand_max_delta_rad_per_tick,
        )
        self.hand_ramp_start = hand.next_ramp_start_qpos_rad
        self.hand_ramp_step = hand.next_ramp_step
        self.planner.set_hand_qpos(hand.qpos_rad)
        ik_result = self.planner.solve_teleop_ik(
            Pose(p=target.position_world_m, q=target.quat_world_wxyz),
            observation.arm_qpos_rad,
            self.prev_qpos_cmd,
        )
        return TeleopActionComputation(
            target_position_world_m=target.position_world_m,
            target_quat_world_wxyz=target.quat_world_wxyz,
            hand_qpos_rad=hand.qpos_rad,
            hand_retarget_succeeded=hand.retarget_succeeded,
            ik_qpos_rad=ik_result.qpos if ik_result.success else None,
            ik_failure_reason=ik_result.reason,
        )


def feedback_is_newer_than_pause(
    pause_since_ns: int,
    *,
    arm_source_monotonic_ns: int,
    vr_receive_monotonic_ns: int,
    hand_source_monotonic_ns: int | None,
) -> bool:
    """Require every enabled feedback source to strictly postdate a pause."""
    boundary_ns = int(pause_since_ns)
    if boundary_ns <= 0:
        return False
    return bool(
        int(arm_source_monotonic_ns) > boundary_ns
        and int(vr_receive_monotonic_ns) > boundary_ns
        and (
            hand_source_monotonic_ns is None
            or int(hand_source_monotonic_ns) > boundary_ns
        )
    )


def _empty_policy_observation_signals() -> dict[str, object]:
    """Return an explicit invalid policy-observation record."""
    return {
        "policy_observation_arm_qpos": np.full(ARM_JOINT_SHAPE, np.nan),
        "policy_observation_hand_qpos": np.full(HAND_JOINT_SHAPE, np.nan),
        "policy_observation_valid": False,
    }


def _recording_policy_observation_signals(
    shared: RuntimeChannels,
    camera_frame: dict[str, Any] | None,
    *,
    anchor_monotonic_ns: int,
    max_observation_skew_s: float,
) -> dict[str, object]:
    """Pair causal arm/hand feedback with the recorded camera source time.

    Teleoperation itself continues to use the latest feedback at the grid cut.
    This separate record is the observation a point-cloud policy will receive
    at deployment, so recording it prevents an offline train/deploy time shift.
    """
    signals = _empty_policy_observation_signals()
    if camera_frame is None:
        return signals
    reference_ns = int(camera_frame.get("source_monotonic_ns", 0))
    anchor_ns = int(anchor_monotonic_ns)
    arm_result = read_structured_frame_aligned_to_source(
        shared.arm_state_ring,
        source_field="source_monotonic_ns",
        reference_source_monotonic_ns=reference_ns,
        anchor_monotonic_ns=anchor_ns,
    )
    hand_result = read_structured_frame_aligned_to_source(
        shared.hand_state_ring,
        source_field="source_monotonic_ns",
        reference_source_monotonic_ns=reference_ns,
        anchor_monotonic_ns=anchor_ns,
    )
    if arm_result is None or hand_result is None:
        return signals
    arm_state, _arm_publish_ns, _arm_sequence = arm_result
    hand_state, _hand_publish_ns, _hand_sequence = hand_result
    arm_names = arm_state.dtype.names or ()
    hand_names = hand_state.dtype.names or ()
    if (
        "state_valid" not in arm_names
        or "state_valid" not in hand_names
        or "qpos" not in arm_names
        or "qpos" not in hand_names
        or not bool(arm_state["state_valid"][0])
        or not bool(hand_state["state_valid"][0])
        or ("qpos_stale" in hand_names and bool(hand_state["qpos_stale"][0]))
    ):
        return signals
    arm_qpos = np.asarray(arm_state["qpos"][0], dtype=np.float64)
    hand_qpos = np.asarray(hand_state["qpos"][0], dtype=np.float64)
    arm_source_ns = int(arm_state["source_monotonic_ns"][0])
    hand_source_ns = int(hand_state["source_monotonic_ns"][0])
    if (
        arm_qpos.shape != ARM_JOINT_SHAPE
        or hand_qpos.shape != HAND_JOINT_SHAPE
        or not np.all(np.isfinite(arm_qpos))
        or not np.all(np.isfinite(hand_qpos))
        or min(reference_ns, arm_source_ns, hand_source_ns) <= 0
    ):
        return signals
    if (
        reference_ns - min(arm_source_ns, hand_source_ns)
    ) > int(round(float(max_observation_skew_s) * 1e9)):
        return signals
    signals.update(
        {
            "policy_observation_arm_qpos": arm_qpos.copy(),
            "policy_observation_hand_qpos": hand_qpos.copy(),
            "policy_observation_valid": True,
        }
    )
    return signals


def _record_grid_hold(
    controller: TeleopController,
    shared: RuntimeChannels,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    *,
    recording_active: bool,
    action_queued: bool = False,
    frame_status: int | None = None,
) -> None:
    """Record one fallback command with the common causal grid provenance."""
    if not recording_active:
        return
    kwargs: dict[str, Any] = {}
    if frame_status is not None:
        kwargs["frame_status"] = frame_status
    record_held(
        resources.recorder,
        observation.arm_state,
        controller.prev_qpos_cmd,
        controller.prev_hand_qpos,
        observation.vr_frame,
        observation.camera_frame,
        hand_state=observation.hand_state,
        hand_tactile=observation.hand_tactile,
        arm_qpos_sent=controller.prev_qpos_cmd.copy(),
        action_queued=action_queued,
        target_eef_pos=controller.last_target_eef_pos,
        target_eef_rot6d=controller.last_target_eef_rot6d,
        observation_anchor_monotonic_ns=observation.anchor_monotonic_ns,
        shared=shared,
        max_observation_skew_s=resources.max_observation_skew_s,
        policy_observation=observation.policy_observation_signals,
        **kwargs,
    )


def _read_control_grid_observation(
    controller: TeleopController,
    shared: RuntimeChannels,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    *,
    teleop_active: bool,
    recording_active: bool,
    pause_since_ns: int,
    pause_reason: str | None,
    arm_feedback_error_count: int,
    hand_disconnected_at_s: float | None,
    loop_count: int,
    observation_anchor_monotonic_ns: int,
    control_run_generation: int,
) -> tuple[TeleopGridTickResult, TeleopGridObservation | None]:
    """Read and validate one causal sensor cut, remaining silent when unsafe."""
    recorder = resources.recorder
    _camera_freshness = resources.camera_freshness
    stage_timer = resources.stage_timer
    _validate_warn = resources.validation_warn
    _arm_feedback_warn = resources.arm_feedback_warn
    _current_grid_anchor_ns = observation_anchor_monotonic_ns
    arm_result = read_causal_structured_frame(
        shared.arm_state_ring,
        source_field="source_monotonic_ns",
        anchor_monotonic_ns=_current_grid_anchor_ns,
    )
    arm_state = None if arm_result is None else arm_result[0]
    arm_ring_sequence = 0 if arm_result is None else int(arm_result[2])
    if arm_state is None:
        arm_issue = "arm feedback unavailable"
    else:
        arm_issue = validate_arm_feedback(
            connected=bool(arm_state["connected"][0]),
            error_code=int(arm_state["error_code"][0]),
            state_valid=bool(arm_state["state_valid"][0]),
            source_monotonic_ns=int(arm_state["source_monotonic_ns"][0]),
            now_monotonic_ns=time.monotonic_ns(),
            max_age_s=cfg.runtime.policy.arm_state_stale_threshold_s,
            qpos=np.asarray(arm_state["qpos"][0]),
            qvel=np.asarray(arm_state["qvel"][0]),
        )
    arm_feedback_error_count, arm_feedback_fault = _advance_arm_feedback_error_count(
        arm_feedback_error_count,
        arm_issue,
        max_consecutive_errors=cfg.runtime.policy.max_consecutive_errors,
    )
    if arm_issue is not None:
        _arm_feedback_warn(
            "teleop_loop: invalid arm feedback (%d/%d): %s",
            arm_feedback_error_count,
            cfg.runtime.policy.max_consecutive_errors,
            arm_issue,
        )
        if arm_feedback_fault:
            logger.error("teleop_loop: arm feedback fault: %s", arm_issue)
            shared.error_state.value = True
            return (
                TeleopGridTickResult(
                    keep_running=False,
                    recording_active=recording_active,
                    arm_feedback_error_count=arm_feedback_error_count,
                    hand_disconnected_at_s=hand_disconnected_at_s,
                ),
                None,
            )
        return (
            TeleopGridTickResult(
                pause_reason=(
                    "arm_feedback" if teleop_active and pause_reason is None else None
                ),
                recording_active=recording_active,
                arm_feedback_error_count=arm_feedback_error_count,
                hand_disconnected_at_s=hand_disconnected_at_s,
            ),
            None,
        )
    assert arm_state is not None  # validation above proved availability
    arm_qpos = arm_state["qpos"][0].copy()

    vr_frame = read_vr_frame_causal(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
    vr_stale = not vr_frame_is_fresh(
        vr_frame,
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=cfg.runtime.policy.vr_mapping.stale_threshold_s,
    )
    stage_timer.mark("vr")

    # VR control does not consume camera pixels.  Scan/copy the large
    # payload only while the policy-owned recorder requests it.
    cam = (
        read_camera_frame_causal(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
        if recording_active
        else None
    )
    if recording_active:
        cam, _camera_stalled = _camera_freshness.observe(cam)
        if _camera_stalled:
            logger.error(
                "Camera source stale for %.1fs — discarding episode; teleoperation remains RUNNING",
                cfg.runtime.camera.recording_stall_abort_s,
            )
            print("  ⚠ 相机连续失帧超过阈值，当前 episode 已废弃；遥操作继续")
            stop_recording(
                recorder,
                True,
                save=False,
                shared=shared,
                reason="camera_stall",
            )
            recording_active = False
    stage_timer.mark("cam")

    hand_result = read_causal_structured_frame(
        shared.hand_state_ring,
        source_field="source_monotonic_ns",
        anchor_monotonic_ns=_current_grid_anchor_ns,
    )
    hand_state = None if hand_result is None else hand_result[0]
    hand_ring_sequence = 0 if hand_result is None else int(hand_result[2])
    hand_tactile = read_hand_tactile_causal(
        shared, anchor_monotonic_ns=_current_grid_anchor_ns
    )

    if not cfg.runtime.policy.hand_enabled:
        hand_issue = None
    elif hand_state is None:
        hand_issue = "hand feedback unavailable"
    else:
        hand_issue = validate_hand_feedback(
            connected=bool(hand_state["connected"][0]),
            state_valid=bool(hand_state["state_valid"][0]),
            source_monotonic_ns=int(hand_state["source_monotonic_ns"][0]),
            now_monotonic_ns=time.monotonic_ns(),
            max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
            qpos=np.asarray(hand_state["qpos"][0]),
        )
    if cfg.runtime.policy.hand_enabled and hand_issue is not None:
        now_s = time.monotonic()
        if hand_disconnected_at_s is None:
            hand_disconnected_at_s = now_s
            logger.warning("Hand feedback unhealthy — pausing motion: %s", hand_issue)
        unhealthy_duration_s = now_s - hand_disconnected_at_s
        if unhealthy_duration_s >= cfg.runtime.policy.hand_disconnect_timeout_s:
            logger.error(
                "Hand feedback remained unhealthy for %.1fs: %s",
                unhealthy_duration_s,
                hand_issue,
            )
            shared.error_state.value = True
            return (
                TeleopGridTickResult(
                    keep_running=False,
                    recording_active=recording_active,
                    arm_feedback_error_count=arm_feedback_error_count,
                    hand_disconnected_at_s=hand_disconnected_at_s,
                ),
                None,
            )
    elif cfg.runtime.policy.hand_enabled and hand_disconnected_at_s is not None:
        unhealthy_duration_s = time.monotonic() - hand_disconnected_at_s
        hand_disconnected_at_s = None
        logger.info(
            "Hand feedback recovered after %.1fs — waiting for fresh re-anchor",
            unhealthy_duration_s,
        )

    if loop_count % cfg.runtime.policy.status_print_interval == 0:
        _arm_age = (
            (time.monotonic_ns() - int(arm_state["source_monotonic_ns"][0])) * 1e-9
            if arm_state is not None
            else -1.0
        )
        _print_status(
            loop_count,
            arm_state,
            vr_frame,
            teleop_active,
            recording_active,
            arm_feedback_error_count,
            arm_state_age_s=_arm_age,
        )

    requested_pause_reason = None
    if teleop_active and pause_reason is None:
        if vr_stale:
            requested_pause_reason = "vr_stale"
        elif cfg.runtime.policy.hand_enabled and hand_issue is not None:
            requested_pause_reason = "hand_feedback"
    hand_source_ns = (
        int(hand_state["source_monotonic_ns"][0])
        if controller.hand_enabled and hand_state is not None
        else None
    )

    if (
        not teleop_active
        or vr_stale
        or pause_reason is not None
        or requested_pause_reason
    ):
        # Resume only with feedback newer than the pause boundary.
        if (
            teleop_active
            and not vr_stale
            and pause_reason is not None
            and pause_since_ns > 0
            and vr_frame is not None
            and (not controller.hand_enabled or hand_state is not None)
            and (not controller.hand_enabled or hand_issue is None)
            and feedback_is_newer_than_pause(
                pause_since_ns,
                arm_source_monotonic_ns=int(arm_state["source_monotonic_ns"][0]),
                vr_receive_monotonic_ns=int(vr_frame["recv_ts_ns"]),
                hand_source_monotonic_ns=hand_source_ns,
            )
        ):
            if controller.reset_reference(arm_state, vr_frame, hand_state):
                logger.info(
                    "teleop_loop: released %s pause boundary after fresh re-anchor",
                    pause_reason,
                )
                pause_released = True
            else:
                _validate_warn(
                    "teleop_loop: re-anchor inputs invalid — remaining command-silent"
                )
                pause_released = False
        else:
            pause_released = False
        # Track measured position while silent without publishing a hold target.
        controller.prev_qpos_cmd = arm_qpos.copy()
        controller.ema_prev_pos = None
        controller.ema_prev_quat = None
        return (
            TeleopGridTickResult(
                pause_reason=requested_pause_reason,
                pause_released=pause_released,
                recording_active=recording_active,
                arm_feedback_error_count=arm_feedback_error_count,
                hand_disconnected_at_s=hand_disconnected_at_s,
            ),
            None,
        )

    assert vr_frame is not None
    policy_observation_signals = (
        _recording_policy_observation_signals(
            shared,
            cam,
            anchor_monotonic_ns=observation_anchor_monotonic_ns,
            max_observation_skew_s=resources.max_observation_skew_s,
        )
        if recording_active
        else None
    )
    return (
        TeleopGridTickResult(
            recording_active=recording_active,
            arm_feedback_error_count=arm_feedback_error_count,
            hand_disconnected_at_s=hand_disconnected_at_s,
        ),
        TeleopGridObservation(
            arm_state=arm_state,
            arm_ring_sequence=arm_ring_sequence,
            arm_qpos_rad=arm_qpos,
            vr_frame=vr_frame,
            camera_frame=cam,
            hand_state=hand_state,
            hand_ring_sequence=hand_ring_sequence,
            hand_tactile=hand_tactile,
            anchor_monotonic_ns=observation_anchor_monotonic_ns,
            control_run_generation=control_run_generation,
            policy_observation_signals=policy_observation_signals,
        ),
    )


def _publish_arm_safety_hold(
    controller: TeleopController,
    shared: RuntimeChannels,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    *,
    recording_active: bool,
    failure_context: str,
    frame_status: int,
) -> bool:
    """Publish and record an arm-only hold after a rejected proposal."""
    prepared_hold, hold_result = _prepare_and_publish_joint_command(
        shared,
        controller.prev_qpos_cmd.copy(),
        None,
        gate=resources.safety_gate,
        is_hold=True,
        observation_id=int(observation.vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(observation.vr_frame["recv_ts_ns"]),
        arm_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["arm"]),
        hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
    )
    published_hold = prepared_hold.candidate
    if hold_result is None or not hold_result.published or published_hold is None:
        reason = prepared_hold.reason or (hold_result.reason if hold_result else "")
        logger.error(
            "teleop_loop: %s hold publish failed: %s",
            failure_context,
            reason,
        )
        shared.error_state.value = True
        return False
    _record_grid_hold(
        controller,
        shared,
        resources,
        observation,
        recording_active=recording_active,
        action_queued=True,
        frame_status=frame_status,
    )
    return True


def _publish_ik_failure_hold(
    controller: TeleopController,
    shared: RuntimeChannels,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    computation: TeleopActionComputation,
    *,
    recording_active: bool,
) -> bool:
    """Publish a bounded hold while preserving independent safe hand motion."""
    if controller.consecutive_ik_hold_frames == 0:
        controller.ik_hold_started_s = time.monotonic()
        logger.warning(
            "teleop_loop: IK hold started: %s",
            computation.ik_failure_reason or "no feasible solution",
        )
    controller.consecutive_ik_hold_frames += 1

    safe_hand_qpos = computation.hand_qpos_rad if controller.hand_enabled else None
    prepared_command, publish_result = _prepare_and_publish_joint_command(
        shared,
        controller.prev_qpos_cmd.copy(),
        safe_hand_qpos,
        gate=resources.safety_gate,
        is_hold=True,
        observation_id=int(observation.vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(observation.vr_frame["recv_ts_ns"]),
        arm_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["arm"]),
        hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
    )

    published_candidate = prepared_command.candidate
    if (
        publish_result is None
        or not publish_result.published
        or published_candidate is None
    ):
        reason = prepared_command.reason or (
            publish_result.reason if publish_result else ""
        )
        logger.error(
            "teleop_loop: IK-failure hold publish failed: %s",
            reason,
        )
        shared.error_state.value = True
        return False
    if controller.hand_enabled:
        if published_candidate.arm_qpos is not None:
            controller.prev_qpos_cmd = np.asarray(
                published_candidate.arm_qpos, dtype=np.float64
            )
        if published_candidate.hand_qpos is not None:
            controller.prev_hand_qpos = np.asarray(
                published_candidate.hand_qpos, dtype=np.float64
            ).copy()

    _record_grid_hold(
        controller,
        shared,
        resources,
        observation,
        recording_active=recording_active,
        action_queued=True,
        frame_status=FRAME_IK_FAIL,
    )
    return True


def _publish_solved_action(
    controller: TeleopController,
    shared: RuntimeChannels,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    computation: TeleopActionComputation,
    *,
    recording_active: bool,
) -> bool:
    """Validate, publish, and record one successful IK solution."""
    assert computation.ik_qpos_rad is not None
    planner = resources.planner
    gate = resources.safety_gate
    recorder = resources.recorder
    command_limits = resources.command_limits
    stage_timer = resources.stage_timer
    _current_grid_anchor_ns = observation.anchor_monotonic_ns
    arm_state = observation.arm_state
    vr_frame = observation.vr_frame
    cam = observation.camera_frame
    hand_state = observation.hand_state
    hand_tactile = observation.hand_tactile
    target_pos = computation.target_position_world_m
    target_quat = computation.target_quat_world_wxyz
    hand_cmd = computation.hand_qpos_rad
    retarget_ok = computation.hand_retarget_succeeded

    if controller.consecutive_ik_hold_frames:
        logger.info(
            "teleop_loop: IK recovered after %d frames (%.3fs)",
            controller.consecutive_ik_hold_frames,
            time.monotonic() - controller.ik_hold_started_s,
        )
        controller.consecutive_ik_hold_frames = 0
        controller.ik_hold_started_s = 0.0

    arm_proposal = compute_arm_joint_proposal(
        computation.ik_qpos_rad,
        controller.prev_qpos_cmd,
        joint_lower_rad=command_limits.arm_joint_lower_rad,
        joint_upper_rad=command_limits.arm_joint_upper_rad,
        max_delta_rad_per_tick=(command_limits.teleop_arm_max_delta_rad_per_tick),
        compute_qpos_delta=planner.compute_qpos_delta,
    )
    arm_cmd = arm_proposal.qpos_rad

    reject_reason = arm_proposal.validation_issue
    if reject_reason is not None:
        resources.validation_warn(
            "teleop_loop: action rejected — %s",
            reject_reason,
        )
        return _publish_arm_safety_hold(
            controller,
            shared,
            cfg,
            resources,
            observation,
            recording_active=recording_active,
            failure_context="rejected-action",
            frame_status=FRAME_SAFETY_REJECT,
        )

    prepared_command, publish_result = _prepare_and_publish_joint_command(
        shared,
        arm_cmd.copy(),
        hand_cmd.copy() if controller.hand_enabled else None,
        gate=gate,
        observation_id=int(vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(vr_frame["recv_ts_ns"]),
        arm_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["arm"]),
        hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
    )
    published_candidate = prepared_command.candidate
    workspace_rejected = prepared_command.gate_code in (
        GateRejectCode.WORKSPACE,
        GateRejectCode.WORKSPACE_CHECK_FAILED,
    )
    if workspace_rejected:
        resources.validation_warn(
            "teleop_loop: action rejected — %s; publishing hold",
            prepared_command.reason,
        )
        return _publish_arm_safety_hold(
            controller,
            shared,
            cfg,
            resources,
            observation,
            recording_active=recording_active,
            failure_context="workspace-rejection",
            frame_status=FRAME_SAFETY_REJECT,
        )
    if (
        publish_result is None
        or not publish_result.published
        or published_candidate is None
    ):
        # Recoverable holds keep arm and hand in place without latching a fault.
        reason = prepared_command.reason or (
            publish_result.reason if publish_result else ""
        )
        hold_status = prepared_command.unavailable
        if publish_result is not None and publish_result.reason:
            logger.info(
                "teleop_loop: joint publication stopped by runtime gate: %s",
                publish_result.reason,
            )
            if not publish_result.reason.startswith(PUBLISH_REASON_SAFETY_STATE):
                return False
            hold_status = True
        if not hold_status:
            logger.error("teleop_loop: joint publish failed: %s", reason)
            shared.error_state.value = True
            return False
        _record_grid_hold(
            controller,
            shared,
            resources,
            observation,
            recording_active=recording_active,
        )
        return True
    stage_timer.mark("send")

    if published_candidate.arm_qpos is not None:
        arm_cmd = np.asarray(published_candidate.arm_qpos, dtype=np.float64)
    if published_candidate.hand_qpos is not None:
        hand_cmd = np.asarray(published_candidate.hand_qpos, dtype=np.float64)
    controller.prev_qpos_cmd = arm_cmd.copy()
    controller.prev_hand_qpos = hand_cmd.copy()
    controller.ema_prev_pos = target_pos.copy()
    controller.ema_prev_quat = target_quat.copy()

    if recording_active:
        controller.last_target_eef_pos = target_pos.copy()
        controller.last_target_eef_rot6d = quat_wxyz_to_rot6d(
            normalize_quat_wxyz(target_quat)
        )
        if not retarget_ok and controller.hand_enabled:
            _f_status = FRAME_RETARGET_FAIL
        else:
            _f_status = FRAME_OK
        record_frame(
            recorder,
            arm_state,
            hand_state,
            arm_cmd,
            hand_cmd,
            target_pos,
            target_quat,
            vr_frame,
            cam,
            hand_tactile,
            frame_status=_f_status,
            observation_anchor_monotonic_ns=_current_grid_anchor_ns,
            shared=shared,
            max_observation_skew_s=resources.max_observation_skew_s,
            policy_observation=observation.policy_observation_signals,
        )
    stage_timer.mark("rec")

    return True


def run_control_grid_tick(
    controller: TeleopController,
    shared: RuntimeChannels,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    *,
    teleop_active: bool,
    recording_active: bool,
    pause_since_ns: int,
    pause_reason: str | None,
    arm_feedback_error_count: int,
    hand_disconnected_at_s: float | None,
    loop_count: int,
    observation_anchor_monotonic_ns: int,
) -> TeleopGridTickResult:
    """Consume one causal observation and publish at most one action."""
    gate = resources.safety_gate
    control_run_generation = int(shared.run_generation.value)
    tick_result, observation = _read_control_grid_observation(
        controller,
        shared,
        cfg,
        resources,
        teleop_active=teleop_active,
        recording_active=recording_active,
        pause_since_ns=pause_since_ns,
        pause_reason=pause_reason,
        arm_feedback_error_count=arm_feedback_error_count,
        hand_disconnected_at_s=hand_disconnected_at_s,
        loop_count=loop_count,
        observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
        control_run_generation=control_run_generation,
    )
    if observation is None:
        return tick_result
    vr_frame = observation.vr_frame
    computation = controller.compute(observation, resources)
    if computation is None:
        prepared_hold, hold_result = _prepare_and_publish_joint_command(
            shared,
            controller.prev_qpos_cmd.copy(),
            None,
            gate=gate,
            is_hold=True,
            observation_id=int(vr_frame["ring_sequence"]),
            observation_anchor_monotonic_ns=int(vr_frame["recv_ts_ns"]),
            arm_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["arm"]),
            hand_feedback_max_age_s=float(
                cfg.runtime.safety.heartbeat_timeouts["hand"]
            ),
        )
        published_hold = prepared_hold.candidate
        if hold_result is None or not hold_result.published or published_hold is None:
            reason = prepared_hold.reason or (hold_result.reason if hold_result else "")
            logger.error(
                "teleop_loop: mapper hold publish failed: %s",
                reason,
            )
            shared.error_state.value = True
            return TeleopGridTickResult(
                keep_running=False,
                recording_active=tick_result.recording_active,
                arm_feedback_error_count=tick_result.arm_feedback_error_count,
                hand_disconnected_at_s=tick_result.hand_disconnected_at_s,
            )
        _record_grid_hold(
            controller,
            shared,
            resources,
            observation,
            recording_active=tick_result.recording_active,
            action_queued=True,
        )
        return tick_result

    if computation.ik_qpos_rad is None:
        keep_running = _publish_ik_failure_hold(
            controller,
            shared,
            cfg,
            resources,
            observation,
            computation,
            recording_active=tick_result.recording_active,
        )
    else:
        keep_running = _publish_solved_action(
            controller,
            shared,
            cfg,
            resources,
            observation,
            computation,
            recording_active=tick_result.recording_active,
        )
    if keep_running:
        return tick_result
    return TeleopGridTickResult(
        keep_running=False,
        recording_active=tick_result.recording_active,
        arm_feedback_error_count=tick_result.arm_feedback_error_count,
        hand_disconnected_at_s=tick_result.hand_disconnected_at_s,
    )


def _print_status(
    loop_count: int,
    arm_state: np.ndarray | None,
    vr_frame: dict | None,
    teleop_active: bool,
    recording_active: bool,
    error_count: int,
    arm_state_age_s: float = -1.0,
) -> None:
    """Periodic status print."""
    if arm_state is not None:
        try:
            _e, _ = make_arm_fk().compute(
                np.asarray(arm_state["qpos"][0], dtype=np.float64)
            )
            eef_str = f"eef={_e[0]:.3f},{_e[1]:.3f},{_e[2]:.3f}"
        except Exception:
            eef_str = "eef=?,?,?"
    else:
        eef_str = "eef=?,?,?"
    if vr_frame is not None:
        vr_age_ms = (time.monotonic_ns() - vr_frame.get("recv_ts_ns", 0)) / 1e6
        vr_str = f"vr={vr_age_ms:.0f}ms"
    else:
        vr_str = "vr=?ms"
    parts = [
        f"f={loop_count:>5d}",
        eef_str,
        f"T={'1' if teleop_active else '0'}",
        f"R={'1' if recording_active else '0'}",
        vr_str,
        f"err={error_count}",
    ]
    if arm_state_age_s >= 0:
        parts.append(f"arm_age={arm_state_age_s:.2f}s")
    print("  ".join(parts), flush=True)
