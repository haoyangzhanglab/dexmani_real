"""跨设备的运行前验证；不连接设备或读取标定。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.robot.system import FEEDBACK_PERIOD_NS
from dexmani_real.utils.timing import sampling_clock_ns

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def validate_robot_config(cfg: RealConfig) -> None:
    cfg.arm.validate()
    if cfg.hand.enabled:
        cfg.hand.validate()
        if FEEDBACK_PERIOD_NS >= int(cfg.hand.feedback_max_age_s * 1e9):
            raise ValueError("30 Hz feedback service must fit hand.feedback_max_age_s")


def validate_control_rate(cfg: RealConfig, control_hz: float) -> None:
    sampling_clock_ns(control_hz)
    if cfg.hand.enabled and control_hz > 83:
        raise ValueError("XHand targets exceed the documented 83 Hz upper bound")


def jog_bounds(runtime: RealConfig) -> tuple[np.ndarray, np.ndarray]:
    cfg = runtime.keyboard_teleop
    cfg.validate()
    sampling_clock_ns(cfg.control_hz)
    workspace = runtime.environment.workspace.as_array()
    margin = cfg.workspace_command_margin_m
    lower, upper = workspace[:, 0] + margin, workspace[:, 1] - margin
    if np.any(lower >= upper):
        raise ValueError("keyboard workspace command margin leaves no interior workspace")
    return lower, upper
