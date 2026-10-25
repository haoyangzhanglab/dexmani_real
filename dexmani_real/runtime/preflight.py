"""跨设备的运行前验证；不连接设备或读取标定。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dexmani_real.config import ExperimentConfig


def validate_robot_config(cfg: ExperimentConfig) -> None:
    """Values used by every robot owner, including HOME and hand-disabled geometry."""
    cfg.arm.validate()
    cfg.arm.homing.validate()
    if cfg.hand.enabled:
        cfg.hand.validate()
    else:
        cfg.hand.validate_home_pose()
    cfg.safety.validate()


def validate_control_rate(cfg: ExperimentConfig, control_hz: float) -> None:
    """Validate cadence before connecting; never silently decimate hand targets."""
    if not np.isfinite(control_hz) or not 0 < control_hz <= cfg.arm.servo_hz:
        raise ValueError("target frequency must be positive and no greater than arm.servo_hz")
    if 1 / control_hz >= cfg.arm.feedback_max_age_s:
        raise ValueError(
            "target period must fit arm.feedback_max_age_s (coordinator health budget)"
        )
    if cfg.hand.enabled:
        if control_hz > 83:
            raise ValueError("XHand targets exceed the documented 83 Hz upper bound")
        if 1 / control_hz >= cfg.hand.feedback_max_age_s:
            raise ValueError(
                "target period must fit hand.feedback_max_age_s; configure both explicitly"
            )


def validate_keyboard_workspace(runtime: ExperimentConfig) -> np.ndarray:
    """Validate the selected jog configuration before any device connects."""
    cfg = runtime.keyboard_teleop
    cfg.validate()
    validate_control_rate(runtime, cfg.control_hz)
    workspace = runtime.environment.workspace.as_array()
    if np.any(
        workspace[:, 0] + cfg.workspace_command_margin_m
        >= workspace[:, 1] - cfg.workspace_command_margin_m
    ):
        raise ValueError("keyboard workspace command margin leaves no interior workspace")
    return workspace
