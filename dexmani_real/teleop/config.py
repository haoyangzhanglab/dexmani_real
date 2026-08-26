"""Small teleoperation view over the canonical typed runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.robot_spec import XHAND_RIGHT_URDF_PATH


@dataclass(frozen=True)
class TeleopCommandLimits:
    """Resolved command bounds used by the teleoperation process."""

    arm_joint_lower_rad: np.ndarray
    arm_joint_upper_rad: np.ndarray
    arm_max_delta_rad_per_tick: np.ndarray | None
    hand_home_qpos_rad: np.ndarray
    hand_command_lower_rad: np.ndarray
    hand_command_upper_rad: np.ndarray
    # Policy-grid endpoint bound shared with learned-policy deployment.  The
    # hand worker keeps its measured-state limiter as an independent hardware
    # backstop.
    hand_max_delta_rad_per_tick: np.ndarray
    hand_mechanical_lower_rad: np.ndarray
    hand_mechanical_upper_rad: np.ndarray
    workspace_bounds_world_m: np.ndarray

    @classmethod
    def from_config(cls, config: "TeleopConfig") -> "TeleopCommandLimits":
        arm_lower = np.asarray(config.runtime.arm.joint_limit_lower, dtype=np.float64)
        arm_upper = np.asarray(config.runtime.arm.joint_limit_upper, dtype=np.float64)
        configured_delta = config.runtime.policy.arm_max_delta_rad_per_tick
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
            arm_max_delta_rad_per_tick=max_delta,
            hand_home_qpos_rad=np.deg2rad(
                np.asarray(config.runtime.hand.home_qpos_deg, dtype=np.float64)
            ),
            hand_command_lower_rad=hand_lower.copy(),
            hand_command_upper_rad=np.asarray(
                config.runtime.hand.qpos_max_rad, dtype=np.float64
            ).copy(),
            hand_max_delta_rad_per_tick=hand_max_delta,
            hand_mechanical_lower_rad=np.asarray(
                config.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ).copy(),
            hand_mechanical_upper_rad=np.asarray(
                config.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ).copy(),
            workspace_bounds_world_m=np.asarray(
                config.runtime.policy.workspace.as_tuple(), dtype=np.float64
            ).copy(),
        )


@dataclass(frozen=True)
class TeleopConfig:
    """Session-only values plus a reference to the canonical runtime snapshot.

    Every runtime value is read from one immutable source via
    ``config.runtime.<section>.<field>``; this dataclass carries only the
    session-only fields plus that reference.
    """

    runtime: ResolvedRuntimeConfig = field(default_factory=resolve_runtime_config)
    task_label: str = ""
    operator: str = ""
    hand_urdf_path: str = field(default_factory=lambda: str(XHAND_RIGHT_URDF_PATH))
    vr_transform_path: str = "dexmani_real/config/vr_transform.json"

    def __post_init__(self) -> None:
        if not self.hand_urdf_path:
            raise ValueError("hand_urdf_path must be non-empty")

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
                str(XHAND_RIGHT_URDF_PATH) if hand_urdf_path is None else hand_urdf_path
            ),
        )
