from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.config.xarm7 import (
    SDK_TIMEOUT_S,
    ArmConfig,
)
from dexmani_real.robot.model import ARM_JOINT_SHAPE

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
    mode: int
    state: int
    error_code: int
    warn_code: int
    report_timestamp_ns: int


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
        self._status: ControllerStatus | None = None

    @property
    def is_connected(self) -> bool:
        return self.api is not None and bool(self.api.connected)

    def connect(self, *, check_abort) -> None:
        check_abort()
        from xarm.wrapper import XArmAPI

        # SDK 的启停提示使用独立 logger；保留控制器告警和错误。
        logging.getLogger("origin.print").setLevel(logging.WARNING)
        self.api = XArmAPI(
            self.cfg.ip, is_radian=True, report_type="rich", max_callback_thread_count=0,
        )
        check_abort()
        self.api.set_timeout(SDK_TIMEOUT_S)
        self.api.register_report_callback(
            self._on_report,
            report_cartesian=False, report_joints=False,
            report_state=True, report_error_code=True, report_warn_code=True,
            report_mtable=False, report_mtbrake=False, report_cmd_num=False,
        )
        self.wait_for_report(check_abort)
        self.check_controller_error()

    def _on_report(self, data):
        # 在 SDK 上报线程内替换完整快照，不查询设备、不等待 IPC。
        # rich 上报的速度字段在此回调之后才更新，关节位置/速度另用一次查询读取。
        received_ns = time.monotonic_ns()
        self._status = ControllerStatus(
            mode=int(self.api.mode), state=int(data["state"]),
            error_code=int(data["error_code"]), warn_code=int(data["warn_code"]),
            report_timestamp_ns=received_ns,
        )

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
            return  # 启动在 SDK 创建前取消，没有设备调用需要收尾。
        # SDK emergency_stop 丢弃返回码，因此直接检查 set_state(4)。
        code = self.api.set_state(4)
        check_sdk_return_code(code, "stop set_state(4)")

    def close(self) -> None:
        if self.api is None:
            return
        try:
            self.api.release_report_callback(self._on_report)
        finally:
            self.api.disconnect()
            self.api = None
            self._status = None

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        code, states = self.api.get_joint_states(is_radian=True, num=2)
        check_sdk_return_code(code, "get_joint_states")
        if not isinstance(states, (list, tuple)) or len(states) != 2:
            raise RuntimeError("get_joint_states(num=2) must return position and velocity")
        qpos = np.asarray(states[0], dtype=np.float64).copy()
        qvel = np.asarray(states[1], dtype=np.float64).copy()
        if any(v.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(v)) for v in (qpos, qvel)):
            raise RuntimeError(
                "get_joint_states returned invalid qpos/qvel shape or non-finite values"
            )
        return qpos, qvel

    def servo(self, qpos: np.ndarray) -> None:
        """Mode 1 只发送采样参考，SDK 速度参数不负责轨迹规划。"""
        code = self.api.set_servo_angle_j(angles=qpos.tolist(), is_radian=True)
        check_sdk_return_code(code, "servo")

    def read_status(self) -> ControllerStatus:
        """只消费上报快照；读取缓存不能延长上报寿命。"""
        status = self._status
        if not self.is_connected:
            raise RuntimeError("xArm disconnected")
        if status is None or not 0 < status.report_timestamp_ns <= time.monotonic_ns() < (
            status.report_timestamp_ns + int(self.cfg.feedback_max_age_s * 1e9)
        ):
            raise RuntimeError("xArm status report unavailable/stale")
        return status

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

        issued_ns = time.monotonic_ns()
        deadline = time.monotonic() + 1.0
        last: ControllerStatus | None = None
        while time.monotonic() < deadline:
            check_abort()
            last = self.read_status()
            check_abort()
            if time.monotonic() >= deadline:
                break
            if last.error_code:
                raise RuntimeError(f"controller error C{last.error_code} while entering Mode {mode}")
            if (
                last.report_timestamp_ns > issued_ns
                and last.mode == mode
                and last.state in (0, 1, 2)
            ):
                return last.state
            time.sleep(0.03)
        check_abort()
        raise RuntimeError(
            f"controller postcondition failed: expected mode={mode} "
            f"error=0 movable-state, got mode={last.mode if last else None} "
            f"state={last.state if last else None} "
            f"error={last.error_code if last else None}"
        )

    @property
    def axis(self) -> int:
        return int(self.api.axis)

    def wait_for_report(self, check_abort) -> None:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            check_abort()
            if self._status is not None and self.axis > 0:
                break
            time.sleep(0.05)
        else:
            check_abort()
            raise TimeoutError("xArm initial status report unavailable")
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
