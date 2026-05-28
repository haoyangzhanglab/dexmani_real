from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xarm.wrapper import XArmAPI

np.set_printoptions(suppress=True, precision=3)

@dataclass
class XArm7Config:
    ip: str = "192.168.1.111"
    dt: float = 1.0 / 50.0
    # init_qpos: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
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
        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None

    def connect(self):
        self.arm = XArmAPI(self.config.ip, is_radian=True)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.set_servo_mode()
        self.last_qpos_cmd = self.get_obs()["qpos"].copy()
        self.last_cmd_time = time.time()
        return self

    def disconnect(self):
        if self.arm is not None:
            self.arm.disconnect()

    def set_servo_mode(self):
        self.arm.set_mode(1)
        self.arm.set_state(0)

    def set_position_mode(self):
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def clear_error(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.set_servo_mode()

    def stop(self):
        return self.arm.set_state(4)

    def reset(self, qpos: np.ndarray | None = None):
        target = self.config.init_qpos if qpos is None else np.asarray(qpos, dtype=np.float64)
        self.set_position_mode()
        code = self.arm.set_servo_angle(
            angle=target.tolist(),
            speed=self.config.reset_speed,
            mvacc=self.config.reset_acc,
            is_radian=True,
            wait=True,
        )
        self.set_servo_mode()
        self.last_qpos_cmd = self.get_obs()["qpos"].copy()
        self.last_cmd_time = time.time()
        return code

    def get_obs(self, full: bool = False) -> dict[str, Any]:
        code, states = self.arm.get_joint_states(is_radian=True, num=3)
        if code == 0:
            qpos = self.array7(states[0])
            qvel = self.array7(states[1])
            tau = self.array7(states[2])
        else:
            qpos = self.read_qpos()
            qvel = self.array7(getattr(self.arm, "realtime_joint_speeds", None))
            tau = self.read_tau()

        obs = {
            "qpos": qpos,
            "qvel": qvel,
            "tau": tau,
            "timestamp": time.time(),
        }

        if full:
            obs.update(
                {
                    "mode": self.arm.mode,
                    "state": self.arm.state,
                    "connected": self.arm.connected,
                    "error_code": self.arm.error_code,
                    "warn_code": self.arm.warn_code,
                    "cmd_num": self.arm.cmd_num,
                    "servo_codes": getattr(self.arm, "servo_codes", None),
                    "temperatures": self.array7(getattr(self.arm, "temperatures", None)),
                    "currents": self.array7(getattr(self.arm, "currents", None)),
                    "voltages": self.array7(getattr(self.arm, "voltages", None)),
                    "motor_enable_states": self.array7(getattr(self.arm, "motor_enable_states", None)),
                    "motor_brake_states": self.array7(getattr(self.arm, "motor_brake_states", None)),
                }
            )
        return obs

    def send_action(self, action) -> dict[str, Any]:
        target_qpos = self.parse_action(action)
        target_qpos = self.limit_joint_range(target_qpos)
        qpos_cmd = self.limit_joint_step(target_qpos)

        if self.arm.mode != 1:
            self.set_servo_mode()

        code = self.arm.set_servo_angle_j(angles=qpos_cmd.tolist(), is_radian=True)
        ok = code == 0
        if ok:
            self.last_qpos_cmd = qpos_cmd.copy()
            self.last_cmd_time = time.time()

        return {
            "ok": ok,
            "code": code,
            "qpos_cmd": qpos_cmd,
            "qpos_target": target_qpos,
        }

    def parse_action(self, action) -> np.ndarray:
        if isinstance(action, dict):
            action = action["qpos"]
        return np.asarray(action, dtype=np.float64).reshape(7)

    def limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        if not self.config.clip_joint_limit:
            return qpos
        return np.clip(qpos, self.config.qpos_min, self.config.qpos_max)

    def limit_joint_step(self, target_qpos: np.ndarray) -> np.ndarray:
        if not self.config.use_delta_limit:
            return target_qpos

        now = time.time()
        if self.last_qpos_cmd is None:
            self.last_qpos_cmd = self.get_obs()["qpos"].copy()
        if self.last_cmd_time is None:
            self.last_cmd_time = now

        dt = max(now - self.last_cmd_time, self.config.dt)
        max_step = self.config.max_qvel * dt
        step = np.clip(target_qpos - self.last_qpos_cmd, -max_step, max_step)
        return self.last_qpos_cmd + step

    def read_qpos(self) -> np.ndarray:
        code, qpos = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            return np.full(7, np.nan, dtype=np.float64)
        return self.array7(qpos)

    def read_tau(self) -> np.ndarray:
        code, tau = self.arm.get_joints_torque()
        if code != 0:
            return np.full(7, np.nan, dtype=np.float64)
        return self.array7(tau)

    def array7(self, value) -> np.ndarray:
        if value is None:
            return np.full(7, np.nan, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 7:
            return arr[:7]
        out = np.full(7, np.nan, dtype=np.float64)
        out[: arr.size] = arr
        return out


def print_obs(obs: dict[str, Any]):
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {np.round(value, 6)}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.111")
    parser.add_argument("--test", default="obs", choices=["obs", "full-obs", "reset", "hold"])
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    robot = XArm7(XArm7Config(ip=args.ip)).connect()
    try:
        if args.test == "obs":
            print_obs(robot.get_obs())
            print(np.rad2deg(robot.get_obs()["qpos"]))
        elif args.test == "full-obs":
            print_obs(robot.get_obs(full=True))
        elif args.test == "reset":
            print("reset code:", robot.reset())
            print_obs(robot.get_obs())
        elif args.test == "hold":
            qpos = robot.get_obs()["qpos"]
            end_time = time.time() + args.seconds
            while time.time() < end_time:
                result = robot.send_action(qpos)
                if not result["ok"]:
                    print(result)
                    break
                time.sleep(robot.config.dt)
            print_obs(robot.get_obs())
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()