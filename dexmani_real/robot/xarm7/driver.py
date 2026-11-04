from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.robot.model import ARM_JOINT_SHAPE

# 低频命令允许较慢回复；set_state 内还会查询状态，需给 3 s 停止等待留余量。
SDK_TIMEOUT_S = 1.0
SDK_STREAM_TIMEOUT_S = 0.020

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


def check_sdk_return_code(code: Any, operation: str) -> None:
    if code != 0:
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
        self._motion_configured = False

    def connect(self, *, check_abort) -> None:
        check_abort()
        from xarm.wrapper import XArmAPI

        # SDK 的启停提示使用独立 logger；保留控制器告警和错误。
        logging.getLogger("origin.print").setLevel(logging.WARNING)
        self._motion_configured = False
        self.api = XArmAPI(
            self.cfg.ip, is_radian=True, report_type="real",
        )
        check_abort()
        self.api.set_timeout(SDK_TIMEOUT_S)
        if self.api.axis != ARM_JOINT_SHAPE[0]:
            raise RuntimeError(
                f"device reports {self.api.axis} axes, expected {ARM_JOINT_SHAPE[0]}"
            )

        self.check_controller_error()

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
        self.api.disconnect()
        self.api = None

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        # real 上报没有错误码，且 SDK 的关节读取会将控制器错误标志归一化为成功。
        self.api.set_timeout(SDK_STREAM_TIMEOUT_S)
        try:
            self.check_controller_error()
            code, states = self.api.get_joint_states(is_radian=True, num=2)
        finally:
            self.api.set_timeout(SDK_TIMEOUT_S)
        check_sdk_return_code(code, "get_joint_states")
        qpos, qvel = (np.asarray(value, dtype=np.float64).copy() for value in states)
        if any(v.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(v)) for v in (qpos, qvel)):
            raise RuntimeError(
                "get_joint_states returned invalid qpos/qvel shape or non-finite values"
            )
        return qpos, qvel

    def servo(self, qpos: np.ndarray) -> None:
        """Mode 1 只发送采样参考，SDK 速度参数不负责轨迹规划。"""
        # Owner 仅在准备成功后建立参考，停止或失败时清除；mode 是异步报告缓存。
        if self.api.state not in (0, 1, 2):
            raise RuntimeError("xArm is not in streaming state")
        self.api.set_timeout(SDK_STREAM_TIMEOUT_S)
        try:
            code = self.api.set_servo_angle_j(angles=qpos.tolist(), is_radian=True)
        finally:
            self.api.set_timeout(SDK_TIMEOUT_S)
        check_sdk_return_code(code, "servo")

    def move_home(self, target: np.ndarray) -> None:
        code = self.api.set_servo_angle(
            angle=target,
            is_radian=True,
            speed=self.cfg.homing.max_speed_rad_per_s,
            mvacc=self.cfg.homing.max_acceleration_rad_per_s2,
            wait=False,
            radius=None,
        )
        check_sdk_return_code(code, "home move")

    def prepare_motion(self, mode, *, check_abort):
        check_abort()
        self.check_controller_error()
        check_abort()
        check_sdk_return_code(self.api.motion_enable(True), "motion_enable")
        check_abort()
        # 明确发送模式命令并检查应答；api.mode 不能同步确认模式切换。
        check_sdk_return_code(self.api.set_mode(mode), f"set_mode({mode})")
        check_abort()
        check_sdk_return_code(self.api.set_state(0), f"set_state(0) after Mode {mode}")
        check_abort()
        if not self._motion_configured:
            check_sdk_return_code(
                self.api.set_collision_sensitivity(self.cfg.collision_sensitivity),
                "set_collision_sensitivity",
            )
            check_abort()
            check_sdk_return_code(
                self.api.set_tcp_load(
                    weight=self.cfg.tcp_load_mass_kg,
                    center_of_gravity=list(self.cfg.tcp_load_cog_mm),
                ),
                "set_tcp_load",
            )
        check_abort()
        acceleration = (self.cfg.homing.max_acceleration_rad_per_s2 if mode == 0
                        else self.cfg.max_joint_acceleration_rad_per_s2)
        check_sdk_return_code(self.api.set_joint_maxacc(acceleration, is_radian=True), "set_joint_maxacc")
        check_abort()
        check_sdk_return_code(self.api.set_state(0), "final preparation set_state(0)")
        check_abort()
        # 静态参数只在本 SDK 会话首次完整准备成功后复用。
        self._motion_configured = True

    def check_controller_error(self) -> None:
        code, codes = self.api.get_err_warn_code()
        check_sdk_return_code(code, "get_err_warn_code")
        error, warn = codes
        if error != 0:
            raise RuntimeError(
                f"xArm controller error C{error} (warn={warn}): "
                f"{describe_controller_error(error)}"
            )
