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
)
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.ipc.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.planning import Pose, XArm7MotionPlanner
from dexmani_real.planning.arm_fk import make_arm_fk
from dexmani_real.planning.hand_fk import HandKinematics
from dexmani_real.planning.poses import (
    normalize_quat_wxyz,
    quat_wxyz_to_rot6d,
    rot6d_to_quat_wxyz,
)
from dexmani_real.recording.client import RecorderClient
from dexmani_real.teleop.action_proposal import (
    compute_arm_joint_proposal,
    compute_hand_joint_proposal,
    compute_target_eef_pose,
)
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.camera_freshness import CameraFreshnessTracker
from dexmani_real.teleop.config import TeleopCommandLimits, TeleopConfig
from dexmani_real.teleop.episode_samples import (
    FRAME_IK_FAIL,
    FRAME_OK,
    FRAME_RETARGET_FAIL,
    FRAME_SAFETY_REJECT,
    record_frame,
    record_held,
    stop_recording,
)
from dexmani_real.teleop.hand_control import (
    HandRetargetObservationCache,
    reset_hand_retargeter,
)
from dexmani_real.teleop.safety import (
    advance_arm_feedback_error_count,
    arm_feedback_issue,
    hand_feedback_issue,
)
from dexmani_real.teleop.timing import StageTimer
from dexmani_real.utils.log import ThrottledWarner, get_logger

logger = get_logger(__name__)


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
    hand_mechanical_lower_rad: np.ndarray | None = None,
    hand_mechanical_upper_rad: np.ndarray | None = None,
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
        hand_mechanical_lower_rad=hand_mechanical_lower_rad,
        hand_mechanical_upper_rad=hand_mechanical_upper_rad,
        arm_feedback_max_age_s=arm_feedback_max_age_s,
        hand_feedback_max_age_s=hand_feedback_max_age_s,
    )
    if not prepared.accepted:
        return prepared, None
    assert prepared.candidate is not None
    return prepared, publish_command(shared, prepared.candidate)


@dataclass(frozen=True)
class TeleopGridResources:
    """Read-only dependencies used to execute one control-grid observation."""

    planner: XArm7MotionPlanner
    safety_gate: SafetyGate
    recorder: RecorderClient | None
    command_limits: TeleopCommandLimits
    camera_freshness: CameraFreshnessTracker
    stage_timer: StageTimer
    validation_warn: ThrottledWarner
    arm_feedback_warn: ThrottledWarner
    hand_fk: HandKinematics | None
    handbase_position_eef_m: np.ndarray
    handbase_quat_eef_wxyz: np.ndarray
    hand_ramp_total_frames: int


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
    policy_observation_signals: dict[str, object] | None


@dataclass(frozen=True)
class TeleopActionComputation:
    """Mapped targets, solver result, and diagnostics for one grid tick."""

    target_position_world_m: np.ndarray
    target_quat_world_wxyz: np.ndarray
    raw_target_position_world_m: np.ndarray
    raw_target_quat_world_wxyz: np.ndarray
    position_before_workspace_clamp_world_m: np.ndarray
    hand_qpos_rad: np.ndarray
    raw_hand_qpos_rad: np.ndarray
    hand_retarget_succeeded: bool
    hand_validation_issue: str | None
    hand_retarget_time_ms: float
    ik_qpos_rad: np.ndarray | None
    ik_failure_reason: str
    ik_solve_time_ms: float
    policy_map_time_ms: float
    policy_compute_started_s: float


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
        arm_mapper: ArmWristMapper,
        config: TeleopConfig,
        command_limits: TeleopCommandLimits,
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
        compute_started_s = time.perf_counter()
        map_started_s = time.perf_counter()
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
        policy_map_time_ms = (time.perf_counter() - map_started_s) * 1000.0
        resources.stage_timer.mark("map")
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
            mechanical_lower_rad=self.command_limits.hand_mechanical_lower_rad,
            mechanical_upper_rad=self.command_limits.hand_mechanical_upper_rad,
        )
        self.hand_ramp_start = hand.next_ramp_start_qpos_rad
        self.hand_ramp_step = hand.next_ramp_step
        if hand.validation_issue is not None:
            resources.validation_warn(
                "teleop_loop: invalid hand command — holding: %s",
                hand.validation_issue,
            )
        self.planner.set_hand_qpos(hand.qpos_rad)
        ik_started_s = time.perf_counter()
        ik_result = self.planner.solve_teleop_ik(
            Pose(p=target.position_world_m, q=target.quat_world_wxyz),
            observation.arm_qpos_rad,
            self.prev_qpos_cmd,
        )
        ik_solve_time_ms = (time.perf_counter() - ik_started_s) * 1000.0
        resources.stage_timer.mark("ik")
        return TeleopActionComputation(
            target_position_world_m=target.position_world_m,
            target_quat_world_wxyz=target.quat_world_wxyz,
            raw_target_position_world_m=target.raw_position_world_m,
            raw_target_quat_world_wxyz=target.raw_quat_world_wxyz,
            position_before_workspace_clamp_world_m=(
                target.position_before_workspace_clamp_world_m
            ),
            hand_qpos_rad=hand.qpos_rad,
            raw_hand_qpos_rad=hand.raw_qpos_rad,
            hand_retarget_succeeded=hand.retarget_succeeded,
            hand_validation_issue=hand.validation_issue,
            hand_retarget_time_ms=hand.compute_time_ms,
            ik_qpos_rad=ik_result.qpos if ik_result.success else None,
            ik_failure_reason=ik_result.reason,
            ik_solve_time_ms=ik_solve_time_ms,
            policy_map_time_ms=policy_map_time_ms,
            policy_compute_started_s=compute_started_s,
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
        "policy_observation_reference_monotonic_ns": 0,
        "policy_observation_arm_source_sequence": 0,
        "policy_observation_hand_source_sequence": 0,
        "policy_observation_arm_source_monotonic_ns": 0,
        "policy_observation_hand_source_monotonic_ns": 0,
        "policy_observation_arm_publish_monotonic_ns": 0,
        "policy_observation_hand_publish_monotonic_ns": 0,
        "policy_observation_valid": False,
        "policy_observation_skew_s": np.nan,
    }


def _recording_policy_observation_signals(
    shared: RuntimeChannels,
    camera_frame: dict[str, Any] | None,
    *,
    anchor_monotonic_ns: int,
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
    arm_state, arm_publish_ns, arm_sequence = arm_result
    hand_state, hand_publish_ns, hand_sequence = hand_result
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
    signals.update(
        {
            "policy_observation_arm_qpos": arm_qpos.copy(),
            "policy_observation_hand_qpos": hand_qpos.copy(),
            "policy_observation_reference_monotonic_ns": reference_ns,
            "policy_observation_arm_source_sequence": int(arm_sequence),
            "policy_observation_hand_source_sequence": int(hand_sequence),
            "policy_observation_arm_source_monotonic_ns": arm_source_ns,
            "policy_observation_hand_source_monotonic_ns": hand_source_ns,
            "policy_observation_arm_publish_monotonic_ns": int(arm_publish_ns),
            "policy_observation_hand_publish_monotonic_ns": int(hand_publish_ns),
            "policy_observation_valid": True,
            "policy_observation_skew_s": (
                reference_ns - min(arm_source_ns, hand_source_ns)
            )
            / 1e9,
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
    action_candidate: ActionCandidate | None = None,
    frame_status: int | None = None,
    retarget_ok: bool = False,
    diagnostics: dict[str, Any] | None = None,
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
        retarget_ok=retarget_ok,
        arm_qpos_sent=controller.prev_qpos_cmd.copy(),
        diagnostics=diagnostics,
        target_eef_pos=controller.last_target_eef_pos,
        target_eef_rot6d=controller.last_target_eef_rot6d,
        hand_fk=resources.hand_fk,
        T_eef_handbase_pos=resources.handbase_position_eef_m,
        T_eef_handbase_quat_wxyz=resources.handbase_quat_eef_wxyz,
        observation_anchor_monotonic_ns=observation.anchor_monotonic_ns,
        arm_ring_sequence=observation.arm_ring_sequence,
        hand_ring_sequence=observation.hand_ring_sequence,
        shared=shared,
        action_candidate=action_candidate,
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
    arm_issue = arm_feedback_issue(
        arm_state,
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=cfg.runtime.policy.arm_state_stale_threshold_s,
    )
    arm_feedback_error_count, arm_feedback_fault = advance_arm_feedback_error_count(
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
    vr_stale = vr_frame is None or (
        (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0))
        > cfg.runtime.policy.vr_mapping.stale_threshold_s * 1e9
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

    hand_issue = hand_feedback_issue(cfg, hand_state)
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
                vr_receive_monotonic_ns=int(vr_frame["local_recv_ns"]),
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
    retarget_ok: bool,
) -> bool:
    """Publish and record an arm-only hold after a rejected proposal."""
    prepared_hold, hold_result = _prepare_and_publish_joint_command(
        shared,
        controller.prev_qpos_cmd.copy(),
        None,
        gate=resources.safety_gate,
        is_hold=True,
        observation_id=int(observation.vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(observation.vr_frame["local_recv_ns"]),
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
        action_candidate=published_hold,
        frame_status=frame_status,
        retarget_ok=retarget_ok,
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

    safe_hand_qpos = (
        computation.hand_qpos_rad
        if controller.hand_enabled and computation.hand_validation_issue is None
        else None
    )
    prepared_command, publish_result = _prepare_and_publish_joint_command(
        shared,
        controller.prev_qpos_cmd.copy(),
        safe_hand_qpos,
        gate=resources.safety_gate,
        is_hold=True,
        observation_id=int(observation.vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(observation.vr_frame["local_recv_ns"]),
        hand_mechanical_lower_rad=resources.command_limits.hand_mechanical_lower_rad,
        hand_mechanical_upper_rad=resources.command_limits.hand_mechanical_upper_rad,
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

    arm_names = observation.arm_state.dtype.names or ()
    diagnostics = {
        "tracking_error": (
            float(observation.arm_state["tracking_err"][0])
            if "tracking_err" in arm_names
            else 0.0
        ),
        "ik_solve_time_ms": computation.ik_solve_time_ms,
        "target_pos_before_clamp": (
            computation.position_before_workspace_clamp_world_m.copy()
        ),
        "head_quat_wxyz": np.asarray(
            observation.vr_frame.get("head_quat_wxyz", np.full(4, np.nan)),
            dtype=np.float64,
        ),
        "target_eef_pos_raw": computation.raw_target_position_world_m.copy(),
        "target_eef_rot6d_raw": quat_wxyz_to_rot6d(
            normalize_quat_wxyz(computation.raw_target_quat_world_wxyz)
        ),
        "action_hand_joint_raw": computation.raw_hand_qpos_rad.copy(),
        "policy_map_time_ms": computation.policy_map_time_ms,
        "hand_retarget_time_ms": computation.hand_retarget_time_ms,
        "transition_check_time_ms": 0.0,
        "policy_compute_time_ms": (
            time.perf_counter() - computation.policy_compute_started_s
        )
        * 1000.0,
    }
    _record_grid_hold(
        controller,
        shared,
        resources,
        observation,
        recording_active=recording_active,
        action_candidate=published_candidate,
        frame_status=FRAME_IK_FAIL,
        retarget_ok=computation.hand_retarget_succeeded,
        diagnostics=diagnostics,
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
    _hand_fk = resources.hand_fk
    _T_eef_handbase_pos = resources.handbase_position_eef_m
    _T_eef_handbase_quat_wxyz = resources.handbase_quat_eef_wxyz
    _current_grid_anchor_ns = observation.anchor_monotonic_ns
    arm_state = observation.arm_state
    vr_frame = observation.vr_frame
    cam = observation.camera_frame
    hand_state = observation.hand_state
    hand_tactile = observation.hand_tactile
    target_pos = computation.target_position_world_m
    target_quat = computation.target_quat_world_wxyz
    target_pos_raw = computation.raw_target_position_world_m
    target_quat_raw = computation.raw_target_quat_world_wxyz
    target_pos_before_clamp = computation.position_before_workspace_clamp_world_m
    hand_cmd = computation.hand_qpos_rad
    hand_cmd_raw = computation.raw_hand_qpos_rad
    retarget_ok = computation.hand_retarget_succeeded
    hand_cmd_valid = computation.hand_validation_issue is None
    hand_retarget_time_ms = computation.hand_retarget_time_ms
    ik_solve_time_ms = computation.ik_solve_time_ms
    policy_map_time_ms = computation.policy_map_time_ms
    _policy_compute_t0 = computation.policy_compute_started_s

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
        max_delta_rad_per_tick=(command_limits.arm_max_delta_rad_per_tick),
        compute_qpos_delta=planner.compute_qpos_delta,
    )
    arm_cmd = arm_proposal.qpos_rad
    arm_cmd_raw = arm_proposal.raw_qpos_rad

    reject_reason = arm_proposal.validation_issue
    if reject_reason is None and not hand_cmd_valid:
        reject_reason = "hand command validation failed"
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
            retarget_ok=computation.hand_retarget_succeeded,
        )

    prepared_command, publish_result = _prepare_and_publish_joint_command(
        shared,
        arm_cmd.copy(),
        hand_cmd.copy() if controller.hand_enabled else None,
        gate=gate,
        observation_id=int(vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
        hand_mechanical_lower_rad=command_limits.hand_mechanical_lower_rad,
        hand_mechanical_upper_rad=command_limits.hand_mechanical_upper_rad,
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
            retarget_ok=computation.hand_retarget_succeeded,
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
        policy_compute_time_ms = (time.perf_counter() - _policy_compute_t0) * 1000.0
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
            ik_solve_time_ms,
            target_pos_before_clamp,
            hand_tactile,
            retarget_ok=retarget_ok,
            frame_status=_f_status,
            target_eef_pos_raw=target_pos_raw,
            target_eef_rot6d_raw=quat_wxyz_to_rot6d(
                normalize_quat_wxyz(target_quat_raw)
            ),
            action_hand_joint_raw=hand_cmd_raw,
            action_arm_joint_raw=arm_cmd_raw,
            policy_map_time_ms=policy_map_time_ms,
            hand_retarget_time_ms=hand_retarget_time_ms,
            policy_compute_time_ms=policy_compute_time_ms,
            hand_fk=_hand_fk,
            T_eef_handbase_pos=_T_eef_handbase_pos,
            T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
            observation_anchor_monotonic_ns=_current_grid_anchor_ns,
            arm_ring_sequence=observation.arm_ring_sequence,
            hand_ring_sequence=observation.hand_ring_sequence,
            shared=shared,
            action_candidate=published_candidate,
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
            observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
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
            action_candidate=published_hold,
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
        vr_age_ms = (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) / 1e6
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
