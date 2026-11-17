"""跨设备的运行前验证；不连接设备或读取标定。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexmani_real.robot.system import FEEDBACK_PERIOD_NS

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def validate_robot_config(cfg: RealConfig) -> None:
    cfg.arm.validate()
    if cfg.hand.enabled:
        cfg.hand.validate()
        if FEEDBACK_PERIOD_NS >= int(cfg.hand.feedback_max_age_s * 1e9):
            raise ValueError("30 Hz feedback service must fit hand.feedback_max_age_s")


def validate_xhand_target_rate(control_hz: float) -> None:
    if control_hz > 83:
        raise ValueError("XHand targets exceed the documented 83 Hz upper bound")
