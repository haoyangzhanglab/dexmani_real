"""跨设备的运行前验证；不连接设备或读取标定。"""

from __future__ import annotations

from dataclasses import astuple
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dexmani_real.config import ExperimentConfig


def validate_robot_config(cfg: ExperimentConfig) -> None:
    cfg.arm.validate()
    if not all(np.isfinite(value) and value > 0 for value in astuple(cfg.arm.homing)):
        raise ValueError("all homing parameters must be finite and positive")
    if cfg.hand.enabled:
        cfg.hand.validate()
    else:
        cfg.hand.validate_home_pose()
    if cfg.environment.table.enabled:
        cfg.environment.table.validate()
    for box in cfg.environment.static_boxes:
        box.validate()
    names = [box.name for box in cfg.environment.static_boxes]
    if len(names) != len(set(names)):
        raise ValueError("static collision box names must be unique")


def validate_control_rate(cfg: ExperimentConfig, control_hz: float) -> None:
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
    cfg = runtime.keyboard_teleop
    for name in ("target_max_age_s", "delta_pos_m", "delta_rpy_rad"):
        value = getattr(cfg, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"keyboard {name} must be finite and positive")
    margin = cfg.workspace_command_margin_m
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("workspace_command_margin_m must be finite and non-negative")
    validate_control_rate(runtime, cfg.control_hz)
    workspace = runtime.environment.workspace.as_array()
    if np.any(workspace[:, 0] + margin >= workspace[:, 1] - margin):
        raise ValueError("keyboard workspace command margin leaves no interior workspace")
    return workspace
