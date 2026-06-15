from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xarm.wrapper import XArmAPI


@dataclass
class XArm7Config:
    ip: str = "192.168.1.111"
    dt: float = 1.0 / 50.0
    init_qpos: np.ndarray = field(
        default_factory=lambda: np.deg2rad([-30, -45, 0, 20, -180, 25, 0])
    )
    qpos_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad([-360, -118, -360, -11, -360, -97, -360])
    )
    qpos_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad([360, 120, 360, 225, 360, 180, 360])
    )
    max_qvel: np.ndarray = field(
        default_factory=lambda: np.deg2rad([90, 90, 90, 90, 120, 120, 150])
    )
    reset_speed: float = np.deg2rad(20)
    reset_acc: float = np.deg2rad(180)
    use_delta_limit: bool = True
    clip_joint_limit: bool = True


class XArm7:
    def __init__(self, config: XArm7Config):
        self.config = config
        self.arm: XArmAPI | None = None

        self.connected_flag = False
        self.error_state = False
        self.last_error_message = ""
        self.last_action_code: int | None = None

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False

    def connect(self) -> bool:
        if self.connected_flag and self.arm is not None:
            return True

        try:
            self.arm = XArmAPI(self.config.ip, is_radian=True)
        except Exception as e:
            self.error_state = True
            self.last_error_message = f"XArmAPI init failed: {e}"
            return False

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self._set_mode(1)

        state = self.get_state()
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = self.config.init_qpos.copy()

        self.last_cmd_time = time.time()
        self.connected_flag = True
        self.error_state = False
        return True

    def disconnect(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()
        self.connected_flag = False

    def is_connected(self) -> bool:
        return self.arm is not None and self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        if self.arm is None:
            return True
        if not self.connected_flag:
            return True
        if self.error_state:
            return True
        if self.arm.error_code != 0:
            return True
        return False

    def clear_error(self) -> bool:
        if self.arm is None:
            return False
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self._set_mode(1)
        self.error_state = False
        self.last_error_message = ""
        return self.arm.error_code == 0

    def stop(self) -> bool:
        if self.arm is None:
            return False
        code = self.arm.set_state(4)
        self.error_state = True
        return code == 0

    def get_state(self, full: bool = False) -> dict[str, Any]:
        code, states = self.arm.get_joint_states(is_radian=True, num=3)
        if code == 0:
            qpos = self._array7(states[0])
            qvel = self._array7(states[1])
            tau = self._array7(states[2])
        else:
            qpos = self._read_qpos()
            qvel = np.full(7, np.nan, dtype=np.float64)
            tau = np.full(7, np.nan, dtype=np.float64)

        state: dict[str, Any] = {
            "qpos": qpos,
            "qvel": qvel,
            "tau": tau,
            "timestamp": time.time(),
        }

        if full:
            state.update({
                "mode": self.arm.mode,
                "state": self.arm.state,
                "connected": self.arm.connected,
                "error_code": self.arm.error_code,
                "warn_code": self.arm.warn_code,
                "cmd_num": self.arm.cmd_num,
                "servo_codes": getattr(self.arm, "servo_codes", None),
                "temperatures": self._array7(getattr(self.arm, "temperatures", None)),
                "currents": self._array7(getattr(self.arm, "currents", None)),
                "voltages": self._array7(getattr(self.arm, "voltages", None)),
                "motor_enable_states": self._array7(getattr(self.arm, "motor_enable_states", None)),
                "motor_brake_states": self._array7(getattr(self.arm, "motor_brake_states", None)),
                "connected_flag": self.connected_flag,
                "error_state": self.error_state,
                "last_error_message": self.last_error_message,
                "last_action_code": self.last_action_code,
                "last_joint_limit_clipped": self.last_joint_limit_clipped,
                "last_delta_limited": self.last_delta_limited,
            })
        return state

    # ------------------------------------------------------------------
    # 动作发送
    # ------------------------------------------------------------------

    def send_action(self, action: np.ndarray) -> bool:
        if self.arm is None:
            self.error_state = True
            self.last_error_message = "arm not connected"
            return False

        target_qpos = np.asarray(action, dtype=np.float64).reshape(7)
        target_qpos = self._limit_joint_range(target_qpos)
        qpos_cmd = self._limit_joint_step(target_qpos)

        if self.arm.mode != 1:
            self._set_mode(1)

        code = self.arm.set_servo_angle_j(angles=qpos_cmd.tolist(), is_radian=True)
        self.last_action_code = code

        if code == 0:
            self.last_qpos_cmd = qpos_cmd.copy()
            self.last_cmd_time = time.time()
            return True

        self.error_state = True
        self.last_error_message = f"set_servo_angle_j failed: code={code}"
        return False

    def reset(self, target: np.ndarray | None = None) -> bool:
        qpos = self.config.init_qpos if target is None else np.asarray(target, dtype=np.float64).reshape(7)
        self._set_mode(0)
        code = self.arm.set_servo_angle(
            angle=qpos.tolist(),
            speed=self.config.reset_speed,
            mvacc=self.config.reset_acc,
            is_radian=True,
            wait=True,
        )
        self._set_mode(1)

        state = self.get_state()
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = qpos.copy()
        self.last_cmd_time = time.time()
        self.last_action_code = code
        return code == 0

    def _set_mode(self, mode: int):
        self.arm.set_mode(mode)
        self.arm.set_state(0)

    def _read_qpos(self) -> np.ndarray:
        code, qpos = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            return np.full(7, np.nan, dtype=np.float64)
        return self._array7(qpos)

    def _limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        if not self.config.clip_joint_limit:
            self.last_joint_limit_clipped = False
            return qpos
        clipped = np.clip(qpos, self.config.qpos_min, self.config.qpos_max)
        self.last_joint_limit_clipped = not np.allclose(qpos, clipped)
        return clipped

    def _limit_joint_step(self, target_qpos: np.ndarray) -> np.ndarray:
        if not self.config.use_delta_limit:
            self.last_delta_limited = False
            return target_qpos

        now = time.time()
        if self.last_qpos_cmd is None:
            state = self.get_state()
            self.last_qpos_cmd = state["qpos"].copy()
            if not np.all(np.isfinite(self.last_qpos_cmd)):
                self.last_qpos_cmd = self.config.init_qpos.copy()
        if self.last_cmd_time is None:
            self.last_cmd_time = now

        dt = max(now - self.last_cmd_time, self.config.dt)
        max_step = self.config.max_qvel * dt
        raw_step = target_qpos - self.last_qpos_cmd
        step = np.clip(raw_step, -max_step, max_step)
        self.last_delta_limited = not np.allclose(raw_step, step)
        return self.last_qpos_cmd + step

    @staticmethod
    def _array7(value) -> np.ndarray:
        if value is None:
            return np.full(7, np.nan, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 7:
            return arr[:7]
        out = np.full(7, np.nan, dtype=np.float64)
        out[: arr.size] = arr
        return out


def _print_state(state: dict[str, Any]):
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, value={np.round(value, 6)}")
        else:
            print(f"{key}: {value}")


def example():
    config = XArm7Config(ip="192.168.1.111")
    robot = XArm7(config)

    if not robot.connect():
        raise RuntimeError(f"Failed to connect XArm7: {robot.last_error_message}")

    try:
        print("=== get_state() ===")
        _print_state(robot.get_state())

        print("\n=== get_state(full=True) ===")
        _print_state(robot.get_state(full=True))

        print("\n=== reset() ===")
        ok = robot.reset()
        print(f"reset ok: {ok}")
        _print_state(robot.get_state())

        print("\n=== hold 3s ===")
        qpos = robot.get_state()["qpos"]
        end_time = time.time() + 3.0
        while time.time() < end_time:
            if not robot.send_action(qpos):
                print(f"send_action failed: {robot.last_error_message}")
                break
            time.sleep(robot.config.dt)
        _print_state(robot.get_state())

    finally:
        robot.disconnect()


if __name__ == "__main__":
    example()
