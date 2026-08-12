"""Small teleoperation view over the canonical typed runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config


@dataclass(frozen=True)
class TeleopConfig:
    """Session-only values plus a reference to the canonical runtime snapshot.

    The previous implementation copied more than sixty runtime fields into a
    second dataclass.  Keeping aliases here preserves the established teleop
    call sites while ensuring every value is read from one immutable source.
    New code should prefer ``config.runtime.<section>.<field>`` directly.
    """

    runtime: ResolvedRuntimeConfig = field(default_factory=resolve_runtime_config)
    task_label: str = ""
    operator: str = ""
    hand_urdf_path: str = field(default_factory=lambda: str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"))
    vr_transform_path: str = "dexmani_real/config/vr_transform.json"

    _ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "control_hz": ("policy", "control_hz"),
        "coordinator_hz": ("policy", "coordinator_hz"),
        "action_prepare_timeout_s": ("policy", "action_prepare_timeout_s"),
        "action_apply_timeout_s": ("policy", "action_apply_timeout_s"),
        "arm_state_stale_threshold_s": ("policy", "arm_state_stale_threshold_s"),
        "quit_save_timeout_s": ("policy", "quit_save_timeout_s"),
        "post_teleop_timeout_s": ("policy", "post_teleop_timeout_s"),
        "joint_max_speed_deg_s": ("arm", "max_joint_velocity_deg_per_s"),
        "joint_max_acc_deg_s2": ("arm", "max_joint_acceleration_deg_per_s2"),
        "arm_loop_hz": ("arm", "loop_hz"),
        "arm_home_qpos": ("arm", "home_qpos"),
        "arm_home_convergence_timeout_s": ("arm", "homing", "convergence_timeout_s"),
        "arm_home_request_queue_timeout_s": (
            "arm",
            "homing",
            "request_queue_timeout_s",
        ),
        "arm_home_state_max_age_s": ("arm", "homing", "state_max_age_s"),
        "arm_home_target_timeout_s": ("arm", "homing", "target_timeout_s"),
        "arm_home_velocity_convergence_rad_s": (
            "arm",
            "homing",
            "velocity_convergence_rad_s",
        ),
        "arm_home_result_tolerance_rad": ("arm", "homing", "convergence_rad"),
        "hand_safety_margin_m": ("arm", "hand_safety_margin_m"),
        "vr_pos_scale": ("policy", "vr_mapping", "pos_scale"),
        "vr_rot_scale": ("policy", "vr_mapping", "rot_scale"),
        "vr_max_delta_rot_rad": ("policy", "vr_mapping", "max_delta_rot_rad"),
        "vr_stale_threshold_s": ("policy", "vr_mapping", "stale_threshold_s"),
        "ik_max_pose_error_pos_m": ("policy", "ik_max_pose_error_pos_m"),
        "ik_max_pose_error_rot_rad": ("policy", "ik_max_pose_error_rot_rad"),
        "ik_nullspace_step_rate_deg_s": ("policy", "ik_nullspace_step_rate_deg_s"),
        "contact_stall_enabled": ("policy", "contact_stall_enabled"),
        "contact_stall_table_z_surface_m": ("arm", "table_z_surface_m"),
        "contact_stall_table_context_height_m": (
            "policy",
            "contact_stall_table_context_height_m",
        ),
        "contact_stall_min_downward_target_m": (
            "policy",
            "contact_stall_min_downward_target_m",
        ),
        "contact_stall_tracking_error_rad": (
            "policy",
            "contact_stall_tracking_error_rad",
        ),
        "contact_stall_max_closing_speed_rad_s": (
            "policy",
            "contact_stall_max_closing_speed_rad_s",
        ),
        "ema_alpha_pos": ("policy", "ema", "alpha_pos"),
        "ema_alpha_rot": ("policy", "ema", "alpha_rot"),
        "max_record_seconds": ("policy", "max_record_duration_s"),
        "min_record_seconds": ("policy", "min_record_duration_s"),
        "episodes_dir": ("policy", "episodes_dir"),
        "recording_enabled": ("policy", "recording_enabled"),
        "status_every": ("policy", "status_print_interval"),
        "max_consecutive_errors": ("policy", "max_consecutive_errors"),
        "hand_enabled": ("policy", "hand_enabled"),
        "hand_retargeting_type": ("policy", "hand_retargeting_type"),
        "hand_output_smoothing_alpha": ("policy", "hand_output_smoothing_alpha"),
        "hand_ramp_duration_s": ("policy", "hand_ramp_duration_s"),
        "begin_motion_gate_timeout_s": ("policy", "begin_motion_gate_timeout_s"),
        "hand_disconnect_timeout_s": ("policy", "hand_disconnect_timeout_s"),
        "camera_max_frame_age_s": ("camera", "max_frame_age_s"),
        "camera_recording_stall_abort_s": ("camera", "recording_stall_abort_s"),
        "camera_width": ("camera", "width"),
        "camera_height": ("camera", "height"),
        "camera_fps": ("camera", "fps"),
        "camera_align_mode": ("camera", "align_mode"),
        "fingertip_link_names": ("hand", "fingertip_link_names"),
        "T_eef_handbase_pos_xyz": ("hand", "T_eef_handbase_pos_xyz"),
        "T_eef_handbase_quat_wxyz": ("hand", "T_eef_handbase_quat_wxyz"),
        "joint_limit_lower": ("arm", "joint_limit_lower"),
        "joint_limit_upper": ("arm", "joint_limit_upper"),
        "hand_home_qpos_deg": ("hand", "home_qpos_deg"),
        "hand_qpos_lower_rad": ("hand", "qpos_min_rad"),
        "hand_qpos_upper_rad": ("hand", "qpos_max_rad"),
        "hand_mechanical_qpos_lower_rad": ("hand", "mechanical_qpos_min_rad"),
        "hand_mechanical_qpos_upper_rad": ("hand", "mechanical_qpos_max_rad"),
        "hand_home_command_ack_timeout_s": ("hand", "home_command_ack_timeout_s"),
        "hand_max_delta_rad": ("hand", "max_delta_rad"),
        "hand_safety_gate_max_velocity_deg_per_s": (
            "hand",
            "safety_gate_max_velocity_deg_per_s",
        ),
        "tag_retargeting_config": ("tag_retargeting",),
        "table_collision": ("environment", "table"),
        "static_collision_boxes": ("environment", "static_boxes"),
    }

    def __post_init__(self) -> None:
        if not self.hand_urdf_path:
            raise ValueError("hand_urdf_path must be non-empty")

    def __getattr__(self, name: str) -> Any:
        path = self._ALIASES.get(name)
        if path is None:
            if name == "readiness_timeouts_s":
                return dict(self.runtime.safety.readiness_timeouts_s)
            if name == "arm_heartbeat_timeout_s":
                return float(self.runtime.safety.heartbeat_timeouts["arm"])
            if name == "hand_heartbeat_timeout_s":
                return float(self.runtime.safety.heartbeat_timeouts["hand"])
            if name == "arm_home_max_speed_rad_s":
                return float(np.deg2rad(self.runtime.arm.homing.max_speed_deg_s))
            if name == "workspace_bounds":
                return self.runtime.policy.workspace.as_tuple()
            raise AttributeError(name)
        value: Any = self.runtime
        for part in path:
            value = getattr(value, part)
        return value

    @classmethod
    def from_runtime(
        cls,
        runtime: ResolvedRuntimeConfig,
        *,
        task_label: str = "",
        operator: str = "",
        hand_urdf_path: str | None = None,
    ) -> "TeleopConfig":
        return cls(
            runtime=runtime,
            task_label=task_label,
            operator=operator,
            hand_urdf_path=(
                str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf") if hand_urdf_path is None else hand_urdf_path
            ),
        )
