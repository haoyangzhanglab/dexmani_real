"""Headless xArm SDK double — injectable via ``sys.modules`` for arm_loop tests.

``arm_loop`` performs ``from xarm.wrapper import XArmAPI`` inside its body, so the
harness installs a fake ``xarm`` package and ``xarm.wrapper`` module into
``sys.modules`` before running it.  The fake models just enough firmware state
(``state`` / ``mode`` / ``error_code`` and joint feedback) for the worker's
state machine to advance deterministically, with no real controller or network.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import numpy as np

ARM_DOF = 7


class FakeXArmAPI:
    """Deterministic stand-in for ``xarm.wrapper.XArmAPI``.

    Firmware postconditions are modelled exactly as the worker's helpers expect:
    ``set_state(0)`` reports ready State 2, ``set_state(4)`` reports State 4,
    and ``set_state(6)`` reports State 6 (accepted by ``_apply_decelerated_stop``
    as the error-free non-ready state).  ``set_servo_angle`` moves the modelled
    joint position to the target immediately so homing convergence checks settle
    in bounded dwell time.
    """

    last_instance: "FakeXArmAPI | None" = None

    def __init__(self, ip: str = "192.0.2.1", is_radian: bool = True) -> None:
        self.ip = ip
        self.is_radian = bool(is_radian)
        self.state = 4  # stopped
        self.mode = 0
        self.error_code = 0
        self.warn_code = 0
        self.qpos = np.zeros(ARM_DOF, dtype=np.float64)
        self.qvel = np.zeros(ARM_DOF, dtype=np.float64)
        self.tau = np.zeros(ARM_DOF, dtype=np.float64)
        self.device_type = "xarm7"
        self.version = "2.7.1"
        self.sn = "FAKE-SN-0001"
        self.connected = True
        self.servo_calls: list[np.ndarray] = []
        self.state_calls: list[int] = []
        self.mode_calls: list[int] = []
        self.fail_servo_code: int = 0  # non-zero -> set_servo_angle rejects (fault injection)
        self._lock = threading.Lock()
        FakeXArmAPI.last_instance = self

    # -- SDK surface ---------------------------------------------------------
    def clean_error(self) -> int:
        self.error_code = 0
        return 0

    def clean_warn(self) -> int:
        self.warn_code = 0
        return 0

    def motion_enable(self, enable: bool = True) -> int:
        return 0

    def set_mode(self, mode: int) -> int:
        with self._lock:
            self.mode = int(mode)
            self.mode_calls.append(int(mode))
        return 0

    def set_state(self, state: int) -> int:
        with self._lock:
            self._transition_state(int(state))
            self.state_calls.append(int(state))
        return 0

    def get_state(self) -> tuple[int, int]:
        return 0, self.state

    def get_err_warn_code(self) -> tuple[int, list[int]]:
        return 0, [self.error_code, self.warn_code]

    def get_joint_states(
        self, is_radian: bool = True, num: int = 1
    ) -> tuple[int, list[np.ndarray]]:
        with self._lock:
            return 0, [self.qpos.copy(), self.qvel.copy(), self.tau.copy()]

    def set_servo_angle(
        self,
        angle: Any,
        is_radian: bool = True,
        speed: float | None = None,
        mvacc: float | None = None,
        wait: bool = False,
        radius: float | None = None,
    ) -> int:
        if self.fail_servo_code != 0:
            return self.fail_servo_code
        target = np.asarray(angle, dtype=np.float64)[:ARM_DOF]
        with self._lock:
            self.qpos = target.copy()
            self.qvel = np.zeros(ARM_DOF, dtype=np.float64)
            self.servo_calls.append(target.copy())
        return 0

    def set_collision_sensitivity(self, value: int) -> int:
        return 0

    def set_tcp_load(self, weight: float, center_of_gravity: Any) -> int:
        return 0

    def set_joint_maxacc(self, value: float, is_radian: bool = True) -> int:
        return 0

    def emergency_stop(self) -> int:
        self.state = 4
        return 0

    def get_c31_error_info(self) -> tuple[int, list[Any]]:
        return 0, [1, 0.0, 10.0]

    def disconnect(self) -> int:
        self.connected = False
        return 0

    # -- helpers -------------------------------------------------------------
    def _transition_state(self, requested: int) -> None:
        # "ready" request -> firmware reports State 2 once motion-enabled
        self.state = 2 if requested == 0 else requested


def _build_modules() -> tuple[types.ModuleType, types.ModuleType]:
    wrapper = types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = FakeXArmAPI
    pkg = types.ModuleType("xarm")
    pkg.wrapper = wrapper
    return pkg, wrapper


def install_xarm_fake() -> None:
    """Install the fake ``xarm.wrapper.XArmAPI`` into ``sys.modules``."""
    pkg, wrapper = _build_modules()
    sys.modules["xarm"] = pkg
    sys.modules["xarm.wrapper"] = wrapper
    FakeXArmAPI.last_instance = None


def remove_xarm_fake() -> None:
    sys.modules.pop("xarm", None)
    sys.modules.pop("xarm.wrapper", None)
    FakeXArmAPI.last_instance = None
