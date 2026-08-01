"""xArm7 7-DOF robot arm hardware driver — thin wrapper for blocking moves.

Control mode: position servo via set_servo_angle_j (Mode 1) for blocking moves
(reset, return-to-home waypoints). Teleop arm control uses ArmInnerLoop with
Mode 6 (joint online trajectory planning, set_servo_angle, firmware trajectory
planner). The two use independent XArmAPI connections — XArm7 for blocking moves,
ArmInnerLoop for continuous teleop.

This class is a thin hardware wrapper — no inner threads, no PID, no velocity mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xarm.wrapper import XArmAPI

from dexmani_real.robot.xarm7.error_codes import decode_error
from dexmani_real.utils.array_utils import nan_array, safe_resize
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.serialization import from_dict_helper

logger = get_logger(__name__)

# Joint-limit insets. The firmware reduced range is a hardware backstop; the
# software clip must sit STRICTLY INSIDE it so boundary-pinned commands (e.g.
# EEF dragged past a workspace edge) never trip a reduced-mode fault.
_FIRMWARE_LIMIT_MARGIN_RAD = np.deg2rad(2.0)
_SOFT_LIMIT_MARGIN_RAD = _FIRMWARE_LIMIT_MARGIN_RAD + np.deg2rad(0.5)


def _inset_joint_limits(q_min: np.ndarray, q_max: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Inset joint limits by ``margin``, skipping full-rotation (±360°) joints."""
    q_min = q_min.copy()
    q_max = q_max.copy()
    full_rot = np.deg2rad(360.0) - 0.01
    for j in range(len(q_min)):
        if q_min[j] <= -full_rot and q_max[j] >= full_rot:
            continue  # full-rotation joint — no limit to inset
        q_min[j] += margin
        q_max[j] -= margin
    return q_min, q_max


@dataclass
class XArm7Config:
    ip: str = "192.168.1.111"
    dt: float = 1.0 / 50.0
    # Comfortable home posture — joint-safe, high manipulability, EEF low near desk.
    # J4=13.5deg (24.5deg above flip boundary), J6=74.7deg (mid-range, 105deg from limit).
    # EEF world pos ~ [0.305, 0.0, 0.176] m (with 30deg Z base rotation).
    init_qpos: np.ndarray = field(default_factory=lambda: np.deg2rad([-30.0, -1.9, 0.0, 13.5, -180.0, 74.7, 0.0]))
    qpos_min: np.ndarray = field(default_factory=lambda: np.deg2rad([-360, -118, -360, -11, -360, -97, -360]))
    qpos_max: np.ndarray = field(default_factory=lambda: np.deg2rad([360, 120, 360, 225, 360, 180, 360]))
    reset_speed: float = np.deg2rad(20)
    reset_acc: float = np.deg2rad(180)
    clip_joint_limit: bool = True

    # Collision detection — set to 1 (least sensitive) to prevent false C31 during teleop.
    # ArmInnerLoop manages its own collision params independently.
    collision_sensitivity: int = 1

    # TCP load for correct dynamics torque estimation
    tcp_load_kg: float = 1.2
    tcp_load_cog_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 80.0])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "XArm7Config":
        """Reconstruct from a serialized dict."""
        return cls(**from_dict_helper(cls, d))  # type: ignore[arg-type]


class XArm7:
    """Thin xArm7 hardware wrapper — blocking moves only (reset, home).

    Velocity PID control is owned by ArmInnerLoop in a separate process.
    This class only handles: connect, disconnect, reset, stop, get_state,
    and blocking position moves via set_servo_angle_j.
    """

    def __init__(self, config: XArm7Config):
        self.config = config
        self.arm: Any = None
        self.connected_flag: bool = False
        self.error_state: bool = False
        self.last_error_message: str = ""
        self.last_action_code: int | None = None

        # Software clip limits — strictly inside the firmware reduced range
        # (see _inset_joint_limits) so clipped commands never violate it.
        self.qpos_min_soft, self.qpos_max_soft = _inset_joint_limits(
            config.qpos_min, config.qpos_max, _SOFT_LIMIT_MARGIN_RAD
        )

        self.last_sdk_error_code: int = 0
        self.last_qpos_cmd: np.ndarray | None = None
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
        """Clear latched error state without changing the control mode.

        Unlike :meth:`robot_init`, this does NOT call ``_set_mode(1)``, so it is
        safe to call while an :class:`ArmInnerLoop` is running in velocity-control
        mode (mode 4) or online trajectory planning mode (mode 6).  Switching the
        global arm firmware mode underneath the inner loop would cause its
        ``vc_set_joint_velocity`` / ``set_servo_angle`` calls to fail,
        forcing an unnecessary emergency stop.

        After a C31/C32 collision the arm disables motion; ``motion_enable(True)``
        re-arms it.  Collision sensitivity / TCP load are NOT reset — they persist
        across error clears.
        """
        if self.arm is None or not self.connected_flag:
            return False
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_state(0)
        _, err_warn = self.arm.get_err_warn_code()
        if err_warn[0] != 0:
            self.error_state = True
            self.last_error_message = (
                f"clear_error post-check failed: err_warn={err_warn} ({decode_error(err_warn[0])})"
            )
            return False
        self.error_state = False
        self.last_error_message = ""
        return True

    def stop(self) -> bool:
        if self.arm is None or not self.connected_flag:
            return False
        code = self.arm.set_state(4)
        self.error_state = True
        return code == 0

    def get_state(self) -> dict[str, Any]:
        code, states = self.arm.get_joint_states(is_radian=True, num=3)
        if code == 0:
            qpos = self._array7(states[0])
            qvel = self._array7(states[1])
            tau = self._array7(states[2])
        else:
            qpos = self._read_qpos()
            qvel = nan_array(7)
            tau = nan_array(7)

        return {
            "qpos": qpos,
            "qvel": qvel,
            "tau": tau,
            "timestamp": time.time(),
        }

    def send_action(self, action: np.ndarray) -> bool:
        """Send joint position command for blocking moves (reset).

        Simple position servo — clips joint limits, then set_servo_angle_j.
        For teleop position servo, use ArmInnerLoop.
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
            self.last_error_message = (
                f"SDK error code: {self.last_sdk_error_code} ({decode_error(self.last_sdk_error_code)})"
            )
            return False

        target_qpos = np.asarray(action, dtype=np.float64).reshape(7)
        target_qpos = self._limit_joint_range(target_qpos)

        if self.arm.mode != 1:
            self._set_mode(1)

        code = self.arm.set_servo_angle_j(angles=target_qpos.tolist(), is_radian=True)
        self.last_action_code = code

        if code == 0:
            self.last_qpos_cmd = target_qpos.copy()
            return True

        try:
            ret, err_warn = self.arm.get_err_warn_code()
            sdk_err = err_warn[0] if len(err_warn) > 0 else -1
        except (RuntimeError, OSError, ValueError, IndexError):
            sdk_err = -1

        self.last_sdk_error_code = int(sdk_err)
        self.error_state = True
        self.last_error_message = f"set_servo_angle_j failed: code={code}, sdk_err={sdk_err} ({decode_error(sdk_err)})"
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
        # Mode left at 0 — send_action() will set Mode 1 on first use.

        state = self.get_state()
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = qpos.copy()
        self.last_action_code = code
        return code == 0

    # ── Init ──

    def robot_init(self) -> None:
        """Full initialization sequence for the xArm7 controller.

        Does NOT set a control mode — the mode is set on first use by
        :meth:`send_action` (Mode 1 for blocking moves) or by
        :class:`ArmInnerLoop` (Mode 6 for teleop).
        """
        if self.arm is None:
            return

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self._configure_collision_params()
        self._set_reduced_joint_limits()

        _, err_warn = self.arm.get_err_warn_code()
        if err_warn[0] != 0:
            self.error_state = True
            self.last_error_message = f"robot_init post-check failed: err_warn={err_warn} ({decode_error(err_warn[0])})"

    def _set_reduced_joint_limits(self) -> None:
        """Push firmware-level joint limits as an independent hardware safety gate.

        Shrinks config qpos_min/qpos_max by a small margin for joints with
        physical limits (skipping continuous-rotation joints) and pushes them
        to firmware via the SDK reduced-joint-range API.  This provides a
        hardware-level backstop independent of software-side joint clipping.
        """
        if self.arm is None:
            return

        q_min, q_max = _inset_joint_limits(self.config.qpos_min, self.config.qpos_max, _FIRMWARE_LIMIT_MARGIN_RAD)

        joint_range = np.column_stack([q_min, q_max]).ravel().tolist()
        self.arm.set_reduced_joint_range(joint_range, is_radian=True)
        self.arm.set_reduced_mode(True)
        logger.info(
            "Firmware reduced joint limits applied (margin=%.1f°)",
            float(np.degrees(_FIRMWARE_LIMIT_MARGIN_RAD)),
        )

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
        # Verify mode switch succeeded (mirrors robot_init/clear_error pattern)
        _, err_warn = self.arm.get_err_warn_code()
        if err_warn[0] != 0:
            self.error_state = True
            self.last_error_message = (
                f"_set_mode({mode}) post-check failed: err_warn={err_warn} ({decode_error(err_warn[0])})"
            )

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
        clipped = np.clip(qpos, self.qpos_min_soft, self.qpos_max_soft)
        self.last_joint_limit_clipped = not np.allclose(qpos, clipped)
        return clipped

    @staticmethod
    def _array7(value) -> np.ndarray:
        return safe_resize(value, 7)
