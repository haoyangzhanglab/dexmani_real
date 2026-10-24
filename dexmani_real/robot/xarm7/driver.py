"""Checked xArm primitives; the worker owns HOME, streaming and stop failures."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dexmani_real.robot.xarm7.config import ArmParams
from dexmani_real.robot.model import ARM_JOINT_SHAPE
from dexmani_real.utils.log import (
    capture_native_stdout,
    extract_native_diagnostics,
)

logger = logging.getLogger(__name__)


# Vendor error titles and recovery actions; startup errors are never cleared implicitly.
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
class LiveStateError:
    """Synchronous controller read: state and the (error, warn) register."""

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


def read_live_state_and_error(arm_api: Any) -> LiveStateError:
    """Synchronously read controller state and the (error, warn) register."""
    code, state = arm_api.get_state()
    check_sdk_return_code(code, "get_state")
    code, values = arm_api.get_err_warn_code()
    check_sdk_return_code(code, "get_err_warn_code")
    return LiveStateError(state=int(state), error_code=int(values[0]), warn_code=int(values[1]))


def wait_controller_ready(
    arm_api: Any,
    *,
    expected_mode: int,
    timeout_s: float,
    check_abort: Callable[[], None],
) -> int:
    """Wait for SDK-reported mode/connection readiness, error==0 and a movable state within the budget."""
    deadline = time.monotonic() + timeout_s
    last: LiveStateError | None = None
    while time.monotonic() < deadline:
        check_abort()
        last = read_live_state_and_error(arm_api)
        check_abort()
        if (
            bool(arm_api.connected)
            and last.error_code == 0
            and int(arm_api.mode) == expected_mode
            and int(last.state) in (0, 1, 2)
        ):
            return last.state
        time.sleep(0.03)
    check_abort()
    raise RuntimeError(
        f"controller postcondition failed: expected mode={expected_mode} "
        f"error=0 movable-state, got state={last.state if last else None} "
        f"error={last.error_code if last else None}"
    )


def enter_mode(arm_api: Any, mode: int, *, check_abort: Callable[[], None]) -> None:
    """Enter a controller mode and wait for a movable state; raise on failure."""
    check_abort()
    check_sdk_return_code(arm_api.set_mode(mode), f"set_mode({mode})")
    check_abort()
    check_sdk_return_code(arm_api.set_state(0), f"set_state(0) after Mode {mode}")
    wait_controller_ready(arm_api, expected_mode=mode, timeout_s=1.0, check_abort=check_abort)


def decode_joint_states(code: Any, states: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate measured radians and rad/s at the driver boundary."""
    check_sdk_return_code(code, "get_joint_states")
    if not isinstance(states, (list, tuple)) or len(states) != 2:
        raise RuntimeError("get_joint_states(num=2) must return position and velocity")
    qpos = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    qvel = np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]].copy()
    if any(v.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(v)) for v in (qpos, qvel)):
        raise RuntimeError("get_joint_states returned invalid qpos/qvel shape or non-finite values")
    return qpos, qvel


class XArm7:
    """Normalize finite vendor operations; the arm child owns HOME and stopping."""

    def __init__(self, cfg: ArmParams) -> None:
        cfg.validate()
        self.cfg = cfg
        self.api: Any = None

    @property
    def is_connected(self) -> bool:
        return self.api is not None and bool(self.api.connected)

    def connect(self) -> None:
        """Connect and inspect; enabling/mode changes belong to explicit initialize/prepare."""
        sdk_connect_output = None
        try:
            with capture_native_stdout() as sdk_connect_output:
                from xarm.wrapper import XArmAPI

                logging.getLogger("origin.print").setLevel(logging.WARNING)
                self.api = XArmAPI(self.cfg.ip, is_radian=True)
        except Exception as exc:
            vendor_detail = sdk_connect_output.text if sdk_connect_output is not None else ""
            raise RuntimeError(
                f"connect failed: {exc}"
                + (f"; vendor output:\n{vendor_detail}" if vendor_detail else "")
            ) from exc
        sdk_diagnostics = extract_native_diagnostics(sdk_connect_output.text)
        if sdk_diagnostics:
            logger.warning("xArm SDK initialization diagnostics:\n%s", "\n".join(sdk_diagnostics))

        self.api.set_timeout(self.cfg.sdk_timeout_s)
        self.wait_for_axis_report()
        self.check_controller_error()

    def initialize(self, *, check_abort):
        """Explicit startup only, after the owner confirmed fresh measured standstill."""
        check_abort()
        self.check_controller_error()
        check_abort()
        check_sdk_return_code(self.api.motion_enable(True), "startup motion_enable")
        self.enter_mode0(check_abort=check_abort)
        check_abort()
        self.apply_config()

    def stop(self) -> int:
        """Request State 4 without confirmation polling; checked SDK acceptance is not physical standstill."""
        if self.api is None:
            raise RuntimeError("stop has no connected SDK owner")
        # The vendor emergency_stop helper discards return codes; all stops use this checked primitive.
        code = self.api.set_state(4)
        check_sdk_return_code(code, "stop set_state(4)")
        return int(code)

    def close(self) -> None:
        """Disconnect and propagate SDK errors to the runtime cleanup owner."""
        if self.api is None:
            return
        self.api.disconnect()
        self.api = None

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read radians and rad/s for control and measured standstill checks."""
        code, states = self.api.get_joint_states(is_radian=True, num=2)
        return decode_joint_states(code, states)

    def servo(self, qpos: np.ndarray) -> int:
        """One Mode-1 reference; speed/acceleration arguments do not plan this path."""
        return self.api.set_servo_angle_j(angles=qpos.tolist(), is_radian=True)

    def read_status(self):
        return read_live_state_and_error(self.api)

    @property
    def mode(self):
        """SDK report cache, not a timestamped joint observation or synchronous query."""
        return int(self.api.mode)

    def move_home_milestone(self, target: np.ndarray) -> None:
        """Submit one prevalidated Mode-0 milestone; arrival belongs to the child owner."""
        code = self.api.set_servo_angle(
            angle=target,
            is_radian=True,
            speed=self.cfg.homing.max_speed_rad_per_s,
            mvacc=self.cfg.max_joint_acceleration_rad_per_s2,
            wait=False,
            radius=None,
        )
        check_sdk_return_code(code, "home milestone")

    def enter_mode0(self, *, check_abort: Callable[[], None]) -> None:
        """Enter Mode 0 (MoveJoint) and wait for a movable state; raise on failure."""
        enter_mode(self.api, 0, check_abort=check_abort)

    def enter_mode1(self, *, check_abort: Callable[[], None]) -> None:
        """Enter servo Mode 1 and wait for a movable state; raise on failure."""
        enter_mode(self.api, 1, check_abort=check_abort)

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
        """Refuse to proceed on a pre-existing controller error; warn is diagnostic."""
        live = read_live_state_and_error(self.api)
        if live.error_code != 0:
            raise RuntimeError(
                f"startup controller error C{live.error_code} (warn={live.warn_code}, "
                f"state={live.state}): {describe_controller_error(live.error_code)} — "
                "refusing to clear"
            )
        if live.warn_code != 0:
            logger.warning("xarm7: startup controller warn=%d (diagnostic only)", live.warn_code)

    def apply_config(self) -> None:
        """Preserve collision/load configuration and the firmware Mode-0 HOME acc cap."""
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
