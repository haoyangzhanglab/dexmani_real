"""跨设备的运行前验证；不连接设备或读取标定。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.robot.system import FEEDBACK_PERIOD_NS

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def validate_robot_config(cfg: RealConfig) -> None:
    cfg.arm.validate()
    if cfg.hand.enabled:
        cfg.hand.validate()
    else:
        cfg.hand.validate_home_pose()
    if cfg.hand.enabled and FEEDBACK_PERIOD_NS >= int(cfg.hand.feedback_max_age_s * 1e9):
        raise ValueError("30 Hz feedback service must fit hand.feedback_max_age_s")
    cfg.environment.validate()


def validate_control_rate(cfg: RealConfig, control_hz: float) -> None:
    if not np.isfinite(control_hz) or not 0 < control_hz <= cfg.arm.servo_hz:
        raise ValueError("target frequency must be positive and no greater than arm.servo_hz")
    if cfg.hand.enabled and control_hz > 83:
        raise ValueError("XHand targets exceed the documented 83 Hz upper bound")


def jog_bounds(runtime: RealConfig) -> tuple[np.ndarray, np.ndarray]:
    cfg = runtime.keyboard_teleop
    cfg.validate()
    validate_control_rate(runtime, cfg.control_hz)
    workspace = runtime.environment.workspace.as_array()
    margin = cfg.workspace_command_margin_m
    lower, upper = workspace[:, 0] + margin, workspace[:, 1] - margin
    if np.any(lower >= upper):
        raise ValueError("keyboard workspace command margin leaves no interior workspace")
    return lower, upper
