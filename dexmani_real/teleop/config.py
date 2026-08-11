"""Configuration for the VR teleoperation experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.defaults import arm, camera, environment, hand, policy, safety


@dataclass
class TeleopConfig:
    """All experiment parameters consumed by the VR teleoperation loop."""

    control_hz: float = field(default_factory=lambda: policy.control_hz)
    coordinator_hz: float = field(default_factory=lambda: policy.coordinator_hz)
    action_prepare_timeout_s: float = field(default_factory=lambda: policy.action_prepare_timeout_s)
    action_apply_timeout_s: float = field(default_factory=lambda: policy.action_apply_timeout_s)
    arm_state_stale_threshold_s: float = field(default_factory=lambda: policy.arm_state_stale_threshold_s)
    quit_save_timeout_s: float = field(default_factory=lambda: policy.quit_save_timeout_s)
    post_teleop_timeout_s: float = field(default_factory=lambda: policy.post_teleop_timeout_s)
    readiness_timeouts_s: dict[str, float] = field(default_factory=lambda: dict(safety.readiness_timeouts_s))

    # Mode 6 firmware parameters (deg — matches CLI convention)
    joint_max_speed_deg_s: float = field(default_factory=lambda: arm.max_joint_velocity_deg_per_s)
    joint_max_acc_deg_s2: float = field(default_factory=lambda: arm.max_joint_acceleration_deg_per_s2)
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)
    arm_home_qpos: tuple[float, ...] = field(default_factory=lambda: arm.home_qpos)
    arm_home_convergence_timeout_s: float = field(default_factory=lambda: arm.homing.convergence_timeout_s)
    arm_home_request_queue_timeout_s: float = field(default_factory=lambda: arm.homing.request_queue_timeout_s)
    arm_home_state_max_age_s: float = field(default_factory=lambda: arm.homing.state_max_age_s)
    arm_home_max_speed_rad_s: float = field(default_factory=lambda: float(np.deg2rad(arm.homing.max_speed_deg_s)))
    arm_home_target_timeout_s: float = field(default_factory=lambda: arm.homing.target_timeout_s)
    arm_home_velocity_convergence_rad_s: float = field(default_factory=lambda: arm.homing.velocity_convergence_rad_s)
    arm_home_result_tolerance_rad: float = field(default_factory=lambda: arm.homing.convergence_rad)
    arm_heartbeat_timeout_s: float = field(default_factory=lambda: safety.heartbeat_timeouts["arm"])
    hand_heartbeat_timeout_s: float = field(default_factory=lambda: safety.heartbeat_timeouts["hand"])
    hand_safety_margin_m: float = field(default_factory=lambda: arm.hand_safety_margin_m)

    vr_pos_scale: float = field(default_factory=lambda: policy.vr_mapping.pos_scale)
    vr_rot_scale: float = field(default_factory=lambda: policy.vr_mapping.rot_scale)
    vr_max_delta_rot_rad: float = field(default_factory=lambda: policy.vr_mapping.max_delta_rot_rad)
    vr_stale_threshold_s: float = field(default_factory=lambda: policy.vr_mapping.stale_threshold_s)
    ik_max_pose_error_pos_m: float = field(default_factory=lambda: policy.ik_max_pose_error_pos_m)
    ik_max_pose_error_rot_rad: float = field(default_factory=lambda: policy.ik_max_pose_error_rot_rad)
    ik_nullspace_step_rate_deg_s: float = field(default_factory=lambda: policy.ik_nullspace_step_rate_deg_s)
    # Workspace bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] (m)
    workspace_bounds: tuple = field(default_factory=lambda: policy.workspace.as_tuple())
    static_collision_boxes: tuple[Any, ...] = field(default_factory=lambda: environment.static_boxes)

    # Contact-stall resync. Table height is context only, never a pose limit.
    contact_stall_enabled: bool = field(default_factory=lambda: policy.contact_stall_enabled)
    contact_stall_table_z_surface_m: float = field(default_factory=lambda: arm.table_z_surface_m)
    contact_stall_table_context_height_m: float = field(
        default_factory=lambda: policy.contact_stall_table_context_height_m
    )
    contact_stall_min_downward_target_m: float = field(
        default_factory=lambda: policy.contact_stall_min_downward_target_m
    )
    contact_stall_tracking_error_rad: float = field(default_factory=lambda: policy.contact_stall_tracking_error_rad)
    contact_stall_max_closing_speed_rad_s: float = field(
        default_factory=lambda: policy.contact_stall_max_closing_speed_rad_s
    )

    # Cartesian EMA smoothing (tuned at 16Hz)
    ema_alpha_pos: float = field(default_factory=lambda: policy.ema.alpha_pos)
    ema_alpha_rot: float = field(default_factory=lambda: policy.ema.alpha_rot)

    max_record_seconds: float = field(default_factory=lambda: policy.max_record_duration_s)
    min_record_seconds: float = field(default_factory=lambda: policy.min_record_duration_s)
    episodes_dir: str = field(default_factory=lambda: policy.episodes_dir)
    recording_enabled: bool = field(default_factory=lambda: policy.recording_enabled)
    task_label: str = ""
    operator: str = ""

    camera_max_frame_age_s: float = field(default_factory=lambda: camera.max_frame_age_s)
    camera_recording_stall_abort_s: float = field(default_factory=lambda: camera.recording_stall_abort_s)
    camera_width: int = field(default_factory=lambda: camera.width)
    camera_height: int = field(default_factory=lambda: camera.height)
    camera_fps: int = field(default_factory=lambda: camera.fps)
    camera_align_mode: str = field(default_factory=lambda: camera.align_mode)

    # Status print interval (in control ticks)
    status_every: int = field(default_factory=lambda: policy.status_print_interval)

    max_consecutive_errors: int = field(default_factory=lambda: policy.max_consecutive_errors)

    hand_enabled: bool = field(default_factory=lambda: policy.hand_enabled)
    hand_retargeting_type: str = field(default_factory=lambda: policy.hand_retargeting_type)
    hand_output_smoothing_alpha: float = field(default_factory=lambda: policy.hand_output_smoothing_alpha)
    hand_ramp_duration_s: float = field(default_factory=lambda: policy.hand_ramp_duration_s)
    begin_motion_gate_timeout_s: float = field(default_factory=lambda: policy.begin_motion_gate_timeout_s)
    hand_disconnect_timeout_s: float = field(default_factory=lambda: policy.hand_disconnect_timeout_s)

    # Hand FK (fingertip positions)
    hand_urdf_path: str = field(default_factory=lambda: str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"))
    fingertip_link_names: tuple[str, ...] = field(default_factory=lambda: hand.fingertip_link_names)
    T_eef_handbase_pos_xyz: tuple[float, float, float] = field(default_factory=lambda: hand.T_eef_handbase_pos_xyz)
    T_eef_handbase_quat_wxyz: tuple[float, float, float, float] = field(
        default_factory=lambda: hand.T_eef_handbase_quat_wxyz
    )

    # Joint-space hard limits — sourced from arm singleton via shared_storage.
    joint_limit_lower: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_lower)
    joint_limit_upper: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_upper)

    hand_home_qpos_deg: tuple[float, ...] = field(default_factory=lambda: hand.home_qpos_deg)
    hand_qpos_lower_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_min_rad)
    hand_qpos_upper_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_max_rad)
    hand_feedback_bound_tolerance_rad: float = field(default_factory=lambda: hand.feedback_bound_tolerance_rad)
    hand_home_settle_timeout_s: float = field(default_factory=lambda: hand.home_settle_timeout_s)
    hand_home_settle_tolerance_rad: float = field(default_factory=lambda: hand.home_settle_tol_rad)
    hand_max_delta_rad: float | None = field(default_factory=lambda: hand.max_delta_rad)
    hand_safety_gate_max_velocity_deg_per_s: float = field(
        default_factory=lambda: hand.safety_gate_max_velocity_deg_per_s
    )
    tag_retargeting_config: Any | None = None

    # VR transform config path (relative to repo root)
    vr_transform_path: str = "dexmani_real/config/vr_transform.json"

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if not np.isfinite(self.coordinator_hz) or self.coordinator_hz < self.control_hz:
            raise ValueError("coordinator_hz must be finite and >= control_hz")
        timing = (
            self.action_prepare_timeout_s,
            self.action_apply_timeout_s,
            self.arm_state_stale_threshold_s,
            self.quit_save_timeout_s,
            self.post_teleop_timeout_s,
            self.hand_home_settle_timeout_s,
            self.arm_home_convergence_timeout_s,
            self.arm_home_request_queue_timeout_s,
            self.arm_home_state_max_age_s,
            self.arm_home_max_speed_rad_s,
            self.arm_home_target_timeout_s,
            self.arm_home_velocity_convergence_rad_s,
            self.arm_home_result_tolerance_rad,
            self.arm_heartbeat_timeout_s,
            self.hand_heartbeat_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError("teleop action, freshness, and operator timeouts must be finite and positive")
        if not np.isfinite(self.hand_home_settle_tolerance_rad) or self.hand_home_settle_tolerance_rad <= 0:
            raise ValueError("hand_home_settle_tolerance_rad must be finite and positive")
        required_subsystems = {"arm", "hand", "camera", "recorder", "vr"}
        if required_subsystems - self.readiness_timeouts_s.keys() or not all(
            np.isfinite(value) and value > 0 for value in self.readiness_timeouts_s.values()
        ):
            raise ValueError("teleop readiness_timeouts_s is incomplete or invalid")
        if not (0.0 <= self.hand_output_smoothing_alpha <= 1.0):
            raise ValueError("hand_output_smoothing_alpha must be in [0, 1]")
        if not np.isfinite(self.hand_ramp_duration_s) or self.hand_ramp_duration_s < 0:
            raise ValueError("hand_ramp_duration_s must be finite and >= 0")
        if (
            not np.isfinite(self.hand_safety_gate_max_velocity_deg_per_s)
            or self.hand_safety_gate_max_velocity_deg_per_s <= 0
        ):
            raise ValueError("hand_safety_gate_max_velocity_deg_per_s must be finite and positive")
        if not np.isfinite(self.begin_motion_gate_timeout_s) or self.begin_motion_gate_timeout_s < 0:
            raise ValueError("begin_motion_gate_timeout_s must be finite and >= 0")
        if (
            not np.isfinite(self.ik_max_pose_error_pos_m)
            or not np.isfinite(self.ik_max_pose_error_rot_rad)
            or not np.isfinite(self.ik_nullspace_step_rate_deg_s)
            or self.ik_max_pose_error_pos_m <= 0
            or self.ik_max_pose_error_rot_rad <= 0
            or self.ik_nullspace_step_rate_deg_s <= 0
        ):
            raise ValueError("teleop IK limits must be finite and positive")
        if self.camera_max_frame_age_s <= 0:
            raise ValueError("camera_max_frame_age_s must be > 0")
        if self.camera_recording_stall_abort_s <= self.camera_max_frame_age_s:
            raise ValueError("camera_recording_stall_abort_s must exceed camera_max_frame_age_s")

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        task_label: str = "",
        operator: str = "",
        hand_urdf_path: str | None = None,
    ) -> "TeleopConfig":
        arm_cfg = getattr(runtime, "arm")
        hand_cfg = getattr(runtime, "hand")
        policy_cfg = getattr(runtime, "policy")
        camera_cfg = getattr(runtime, "camera")
        environment_cfg = getattr(runtime, "environment")
        safety_cfg = getattr(runtime, "safety")
        return cls(
            control_hz=float(policy_cfg.control_hz),
            coordinator_hz=float(policy_cfg.coordinator_hz),
            action_prepare_timeout_s=float(policy_cfg.action_prepare_timeout_s),
            action_apply_timeout_s=float(policy_cfg.action_apply_timeout_s),
            arm_state_stale_threshold_s=float(policy_cfg.arm_state_stale_threshold_s),
            quit_save_timeout_s=float(policy_cfg.quit_save_timeout_s),
            post_teleop_timeout_s=float(policy_cfg.post_teleop_timeout_s),
            readiness_timeouts_s={
                str(name): float(timeout_s) for name, timeout_s in safety_cfg.readiness_timeouts_s.items()
            },
            joint_max_speed_deg_s=float(arm_cfg.max_joint_velocity_deg_per_s),
            joint_max_acc_deg_s2=float(arm_cfg.max_joint_acceleration_deg_per_s2),
            arm_loop_hz=float(arm_cfg.loop_hz),
            arm_home_qpos=tuple(arm_cfg.home_qpos),
            arm_home_convergence_timeout_s=float(arm_cfg.homing.convergence_timeout_s),
            arm_home_request_queue_timeout_s=float(arm_cfg.homing.request_queue_timeout_s),
            arm_home_state_max_age_s=float(arm_cfg.homing.state_max_age_s),
            arm_home_max_speed_rad_s=float(np.deg2rad(arm_cfg.homing.max_speed_deg_s)),
            arm_home_target_timeout_s=float(arm_cfg.homing.target_timeout_s),
            arm_home_velocity_convergence_rad_s=float(arm_cfg.homing.velocity_convergence_rad_s),
            arm_home_result_tolerance_rad=float(arm_cfg.homing.convergence_rad),
            arm_heartbeat_timeout_s=float(safety_cfg.heartbeat_timeouts["arm"]),
            hand_heartbeat_timeout_s=float(safety_cfg.heartbeat_timeouts["hand"]),
            hand_safety_margin_m=float(arm_cfg.hand_safety_margin_m),
            vr_pos_scale=float(policy_cfg.vr_mapping.pos_scale),
            vr_rot_scale=float(policy_cfg.vr_mapping.rot_scale),
            vr_max_delta_rot_rad=float(policy_cfg.vr_mapping.max_delta_rot_rad),
            vr_stale_threshold_s=float(policy_cfg.vr_mapping.stale_threshold_s),
            ik_max_pose_error_pos_m=float(policy_cfg.ik_max_pose_error_pos_m),
            ik_max_pose_error_rot_rad=float(policy_cfg.ik_max_pose_error_rot_rad),
            ik_nullspace_step_rate_deg_s=float(policy_cfg.ik_nullspace_step_rate_deg_s),
            workspace_bounds=(
                (float(policy_cfg.workspace.x_min), float(policy_cfg.workspace.x_max)),
                (float(policy_cfg.workspace.y_min), float(policy_cfg.workspace.y_max)),
                (float(policy_cfg.workspace.z_min), float(policy_cfg.workspace.z_max)),
            ),
            static_collision_boxes=tuple(environment_cfg.static_boxes),
            contact_stall_enabled=bool(policy_cfg.contact_stall_enabled),
            contact_stall_table_z_surface_m=float(arm_cfg.table_z_surface_m),
            contact_stall_table_context_height_m=float(policy_cfg.contact_stall_table_context_height_m),
            contact_stall_min_downward_target_m=float(policy_cfg.contact_stall_min_downward_target_m),
            contact_stall_tracking_error_rad=float(policy_cfg.contact_stall_tracking_error_rad),
            contact_stall_max_closing_speed_rad_s=float(policy_cfg.contact_stall_max_closing_speed_rad_s),
            ema_alpha_pos=float(policy_cfg.ema.alpha_pos),
            ema_alpha_rot=float(policy_cfg.ema.alpha_rot),
            max_record_seconds=float(policy_cfg.max_record_duration_s),
            min_record_seconds=float(policy_cfg.min_record_duration_s),
            episodes_dir=str(policy_cfg.episodes_dir),
            recording_enabled=bool(policy_cfg.recording_enabled),
            task_label=task_label,
            operator=operator,
            camera_max_frame_age_s=float(camera_cfg.max_frame_age_s),
            camera_recording_stall_abort_s=float(camera_cfg.recording_stall_abort_s),
            camera_width=int(camera_cfg.width),
            camera_height=int(camera_cfg.height),
            camera_fps=int(camera_cfg.fps),
            camera_align_mode=str(camera_cfg.align_mode),
            status_every=int(policy_cfg.status_print_interval),
            max_consecutive_errors=int(policy_cfg.max_consecutive_errors),
            hand_enabled=bool(policy_cfg.hand_enabled),
            hand_retargeting_type=str(policy_cfg.hand_retargeting_type),
            hand_output_smoothing_alpha=float(policy_cfg.hand_output_smoothing_alpha),
            hand_ramp_duration_s=float(policy_cfg.hand_ramp_duration_s),
            begin_motion_gate_timeout_s=float(policy_cfg.begin_motion_gate_timeout_s),
            hand_disconnect_timeout_s=float(policy_cfg.hand_disconnect_timeout_s),
            hand_urdf_path=(
                str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf") if hand_urdf_path is None else hand_urdf_path
            ),
            fingertip_link_names=tuple(hand_cfg.fingertip_link_names),
            T_eef_handbase_pos_xyz=tuple(hand_cfg.T_eef_handbase_pos_xyz),
            T_eef_handbase_quat_wxyz=tuple(hand_cfg.T_eef_handbase_quat_wxyz),
            joint_limit_lower=tuple(arm_cfg.joint_limit_lower),
            joint_limit_upper=tuple(arm_cfg.joint_limit_upper),
            hand_home_qpos_deg=tuple(hand_cfg.home_qpos_deg),
            hand_qpos_lower_rad=tuple(hand_cfg.qpos_min_rad),
            hand_qpos_upper_rad=tuple(hand_cfg.qpos_max_rad),
            hand_feedback_bound_tolerance_rad=float(hand_cfg.feedback_bound_tolerance_rad),
            hand_home_settle_timeout_s=float(hand_cfg.home_settle_timeout_s),
            hand_home_settle_tolerance_rad=float(hand_cfg.home_settle_tol_rad),
            hand_max_delta_rad=None if hand_cfg.max_delta_rad is None else float(hand_cfg.max_delta_rad),
            hand_safety_gate_max_velocity_deg_per_s=float(hand_cfg.safety_gate_max_velocity_deg_per_s),
            tag_retargeting_config=getattr(runtime, "tag_retargeting"),
        )
