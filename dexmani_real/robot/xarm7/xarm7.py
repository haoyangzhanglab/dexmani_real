"""xArm7 7-DOF robot arm hardware driver — simplified for PID process isolation.

Control mode: position servo via set_servo_angle_j (blocking moves only: reset, home).
Velocity PID control is owned by PIDProcess (robot/pid_process.py), which has its
own XArmAPI connection in a separate process.

The xArm7 class is now a thin hardware wrapper — no inner threads, no PID, no velocity mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xarm.wrapper import XArmAPI

from dexmani_real.log import get_logger
from dexmani_real.robot._connection_state import ConnectionStateMixin
from dexmani_real.utils.array_utils import nan_array, safe_resize
from dexmani_real.utils.serialization import from_dict_helper

logger = get_logger(__name__)


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
        default_factory=lambda: np.deg2rad([180, 180, 180, 180, 180, 180, 180])
    )
    reset_speed: float = np.deg2rad(20)
    reset_acc: float = np.deg2rad(180)
    clip_joint_limit: bool = True

    # Collision detection — set to 1 (least sensitive) to prevent false C31 during teleop.
    # PIDProcess manages its own collision params independently.
    collision_sensitivity: int = 1

    # TCP load for correct dynamics torque estimation
    tcp_load_kg: float = 1.2
    tcp_load_cog_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 80.0])


class XArm7(ConnectionStateMixin):
    """Thin xArm7 hardware wrapper — blocking moves only (reset, home).

    Velocity PID control is owned by PIDProcess in a separate process.
    This class only handles: connect, disconnect, reset, stop, get_state,
    and blocking position moves via set_servo_angle_j.
    """

    def __init__(self, config: XArm7Config):
        super().__init__()
        self.config = config
        self.arm: XArmAPI | None = None

        self.last_sdk_error_code: int = 0
        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_joint_limit_clipped = False

    def connect(self) -> bool:
        if self.connected_flag and self.arm is not None:
            return True

        try:
            self.arm = XArmAPI(self.config.ip, is_radian=True)
        except (OSError, ConnectionError, RuntimeError) as e:
            self.error_state = True
            self.last_error_message = f"XArmAPI init failed: {e}"
            return False

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.robot_init()

        state = self.get_state()
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = self.config.init_qpos.copy()
        self.last_cmd_time = time.time()

        if not self.error_state:
            self.connected_flag = True
        else:
            logger.error("connect(): robot_init detected hardware error, aborting")
            return False

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
        self.robot_init()
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
            qvel = nan_array(7)
            tau = nan_array(7)

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
                "cartesian_position": self.get_position(),
                "cartesian_position_aa": self.get_position_aa(),
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
                "last_sdk_error_code": self.last_sdk_error_code,
                "last_joint_limit_clipped": self.last_joint_limit_clipped,
            })
        return state

    def send_action(self, action: np.ndarray) -> bool:
        """Send joint position command for blocking moves (reset).

        Simple position servo — clips joint limits, then set_servo_angle_j.
        For teleop position servo, use PIDProcess.
        """
        if self.arm is None:
            self.error_state = True
            self.last_error_message = "arm not connected"
            return False

        if not self.arm.connected:
            self.error_state = True
            self.last_error_message = "SDK reports arm not connected"
            return False
        if self.arm.error_code != 0:
            self.last_sdk_error_code = self.arm.error_code
            self.error_state = True
            self.last_error_message = f"SDK error code: {self.arm.error_code}"
            return False

        target_qpos = np.asarray(action, dtype=np.float64).reshape(7)
        target_qpos = self._limit_joint_range(target_qpos)

        if self.arm.mode != 1:
            self._set_mode(1)

        code = self.arm.set_servo_angle_j(angles=target_qpos.tolist(), is_radian=True)
        self.last_action_code = code

        if code == 0:
            self.last_qpos_cmd = target_qpos.copy()
            self.last_cmd_time = time.time()
            return True

        try:
            ret, err_warn = self.arm.get_err_warn_code()
            sdk_err = err_warn[0] if len(err_warn) > 0 else -1
        except (RuntimeError, OSError, ValueError, IndexError):
            sdk_err = -1

        self.last_sdk_error_code = int(sdk_err)
        self.error_state = True
        self.last_error_message = f"set_servo_angle_j failed: code={code}, sdk_err={sdk_err}"
        return False

    def reset(self, target: np.ndarray | None = None) -> bool:
        qpos = self.config.init_qpos if target is None else np.asarray(target, dtype=np.float64).reshape(7)
        self._set_mode(0)  # position control mode for blocking move
        code = self.arm.set_servo_angle(
            angle=qpos.tolist(),
            speed=self.config.reset_speed,
            mvacc=self.config.reset_acc,
            is_radian=True,
            wait=True,
        )
        self._set_mode(1)  # back to position servo

        state = self.get_state()
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = qpos.copy()
        self.last_cmd_time = time.time()
        self.last_action_code = code
        return code == 0

    # ── Cartesian pose queries ──

    def get_position(self) -> np.ndarray:
        if self.arm is None:
            return nan_array(6)
        code, pos = self.arm.get_position(is_radian=True)
        if code != 0:
            return nan_array(6)
        return np.asarray(pos, dtype=np.float64)

    def get_position_aa(self) -> np.ndarray:
        if self.arm is None:
            return nan_array(6)
        code, pos = self.arm.get_position_aa(is_radian=True)
        if code != 0:
            return nan_array(6)
        return np.asarray(pos, dtype=np.float64)

    # ── Init ──

    def robot_init(self) -> None:
        """Full initialization sequence for the xArm7 controller."""
        if self.arm is None:
            return

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self._set_mode(1)  # position servo mode
        self._configure_collision_params()

        _, err_warn = self.arm.get_err_warn_code()
        if err_warn[0] != 0:
            self.error_state = True
            self.last_error_message = f"robot_init post-check failed: err_warn={err_warn}"

    def _configure_collision_params(self) -> None:
        if self.arm is None:
            return
        cfg = self.config
        try:
            self.arm.set_tcp_load(cfg.tcp_load_kg, list(cfg.tcp_load_cog_mm))
        except RuntimeError as e:
            self.last_error_message = f"set_tcp_load exception: {e}"
        try:
            self.arm.set_collision_sensitivity(cfg.collision_sensitivity)
        except RuntimeError as e:
            self.last_error_message = f"set_collision_sensitivity exception: {e}"

    def _set_mode(self, mode: int):
        """Transition arm to target control mode via idle intermediate state."""
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.05)
        self.arm.set_mode(mode)
        self.arm.set_state(0)
        time.sleep(0.05)
        self.arm.set_state(0)

    # ── Internal helpers ──

    def _read_qpos(self) -> np.ndarray:
        code, qpos = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            return nan_array(7)
        return self._array7(qpos)

    def _limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        if not self.config.clip_joint_limit:
            self.last_joint_limit_clipped = False
            return qpos
        clipped = np.clip(qpos, self.config.qpos_min, self.config.qpos_max)
        self.last_joint_limit_clipped = not np.allclose(qpos, clipped)
        return clipped

    @staticmethod
    def _array7(value) -> np.ndarray:
        return safe_resize(value, 7)
