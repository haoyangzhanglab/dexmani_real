from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE
from dexmani_real.robot.xarm7.config import (
    SDK_TIMEOUT_S,
    ArmConfig,
)

logger = logging.getLogger(__name__)


CONTROLLER_ERROR_HELP: dict[int, str] = {
    1: "Emergency Stop button pressed — release the E-stop button, then re-enable the robot",
    2: "Emergency IO of the control box triggered — ground the two EI pins, then re-enable the robot",
    3: "Three-state switch E-stop pressed — release the three-state switch, then re-enable the robot",
    10: "servo motor error",
    21: "kinematic error",
    22: "self-collision error",
    23: "joints angle exceed limit",
    24: "speed exceeds limit",
    31: "collision caused abnormal current",
}


@dataclass(frozen=True)
class ControllerStatus:
    state: int
    error_code: int
    warn_code: int


def check_sdk_return_code(code: Any, operation: str) -> None:
    if isinstance(code, bool) or not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RuntimeError(f"{operation} failed with SDK code {code!r}")


def describe_controller_error(code: int) -> str:
    code = int(code)
    if 11 <= code <= 17:
        return f"servo motor {code - 10} error"
    return CONTROLLER_ERROR_HELP.get(code, "controller error")


class XArm7:
    def __init__(self, cfg: ArmConfig) -> None:
        self.cfg = cfg
        self.api: Any = None

    @property
    def is_connected(self) -> bool:
        return self.api is not None and bool(self.api.connected)

    def connect(self) -> None:
        from xarm.wrapper import XArmAPI

        self.api = XArmAPI(self.cfg.ip, is_radian=True)
        self.api.set_timeout(SDK_TIMEOUT_S)
        self.wait_for_axis_report()
        self.check_controller_error()

    def initialize(self, *, check_abort):
        check_abort()
        self.check_controller_error()
        check_abort()
        check_sdk_return_code(self.api.motion_enable(True), "startup motion_enable")
        self.enter_mode(0, check_abort=check_abort)
        check_abort()
        self.apply_config()

    def stop(self) -> None:
        """SDK 接受停止请求不代表物理停稳。"""
        if self.api is None:
            raise RuntimeError("stop has no connected SDK owner")
        # SDK emergency_stop 丢弃返回码，因此直接检查 set_state(4)。
        code = self.api.set_state(4)
        check_sdk_return_code(code, "stop set_state(4)")

    def close(self) -> None:
        if self.api is None:
            return
        self.api.disconnect()
        self.api = None

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        code, states = self.api.get_joint_states(is_radian=True, num=2)
        check_sdk_return_code(code, "get_joint_states")
        if not isinstance(states, (list, tuple)) or len(states) != 2:
            raise RuntimeError("get_joint_states(num=2) must return position and velocity")
        qpos = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
        qvel = np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
        if any(v.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(v)) for v in (qpos, qvel)):
            raise RuntimeError(
                "get_joint_states returned invalid qpos/qvel shape or non-finite values"
            )
        return qpos, qvel

    def servo(self, qpos: np.ndarray) -> None:
        """Mode 1 只发送采样参考，SDK 速度参数不负责轨迹规划。"""
        code = self.api.set_servo_angle_j(angles=qpos.tolist(), is_radian=True)
        check_sdk_return_code(code, "servo")

    def read_status(self):
        code, state = self.api.get_state()
        check_sdk_return_code(code, "get_state")
        code, values = self.api.get_err_warn_code()
        check_sdk_return_code(code, "get_err_warn_code")
        return ControllerStatus(
            state=int(state), error_code=int(values[0]), warn_code=int(values[1])
        )

    @property
    def mode(self):
        """SDK 缓存的模式，不是带来源时间的同步观测。"""
        return int(self.api.mode)

    def move_home_milestone(self, target: np.ndarray) -> None:
        code = self.api.set_servo_angle(
            angle=target,
            is_radian=True,
            speed=self.cfg.homing.max_speed_rad_per_s,
            mvacc=self.cfg.max_joint_acceleration_rad_per_s2,
            wait=False,
            radius=None,
        )
        check_sdk_return_code(code, "home milestone")

    def enter_mode(self, mode, *, check_abort):
        check_abort()
        check_sdk_return_code(self.api.set_mode(mode), f"set_mode({mode})")
        check_abort()
        check_sdk_return_code(self.api.set_state(0), f"set_state(0) after Mode {mode}")

        deadline = time.monotonic() + 1.0
        last: ControllerStatus | None = None
        while time.monotonic() < deadline:
            check_abort()
            last = self.read_status()
            check_abort()
            if (
                bool(self.api.connected)
                and last.error_code == 0
                and int(self.api.mode) == mode
                and int(last.state) in (0, 1, 2)
            ):
                return last.state
            time.sleep(0.03)
        check_abort()
        raise RuntimeError(
            f"controller postcondition failed: expected mode={mode} "
            f"error=0 movable-state, got state={last.state if last else None} "
            f"error={last.error_code if last else None}"
        )

    @property
    def axis(self) -> int:
        return int(self.api.axis)

    def wait_for_axis_report(self) -> None:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.axis > 0:
                break
            time.sleep(0.05)
        if self.axis != ARM_JOINT_SHAPE[0]:
            raise RuntimeError(f"device reports {self.axis} axes, expected {ARM_JOINT_SHAPE[0]}")

    def check_controller_error(self) -> None:
        live = self.read_status()
        if live.error_code != 0:
            raise RuntimeError(
                f"startup controller error C{live.error_code} (warn={live.warn_code}, "
                f"state={live.state}): {describe_controller_error(live.error_code)} — "
                "refusing to clear"
            )
        if live.warn_code != 0:
            logger.warning("xarm7: startup controller warn=%d (diagnostic only)", live.warn_code)

    def apply_config(self) -> None:
        check_sdk_return_code(
            self.api.set_collision_sensitivity(self.cfg.collision_sensitivity),
            "set_collision_sensitivity",
        )
        check_sdk_return_code(
            self.api.set_tcp_load(
                weight=self.cfg.tcp_load_mass_kg,
                center_of_gravity=list(self.cfg.tcp_load_cog_mm),
            ),
            "set_tcp_load",
        )
        check_sdk_return_code(
            self.api.set_joint_maxacc(
                self.cfg.max_joint_acceleration_rad_per_s2,
                is_radian=True,
            ),
            "set_joint_maxacc",
        )
