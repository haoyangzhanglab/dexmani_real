"""Shared offline test doubles for pass-item regression checks.

These are plain recording objects — not a mock framework — that let each check
exercise the real production functions against a stand-in for the vendor SDK
and the fixed NumPy frame dtypes.  Nothing here touches hardware.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.robot.xhand import XHandSample
from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    ARM_STATE_DTYPE,
    HAND_CONTACT_SHAPE,
    HAND_DOF,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)


class FakeArm:
    """Recording stand-in for the xArm SDK surface used by ``arm_loop`` helpers.

    Every SDK call appends ``(name, args, kwargs)`` to :attr:`calls` and returns
    a configurable integer code.  A method can be made to fail (return a
    non-zero code) via :meth:`fail`, or to raise via :meth:`raise_on`, so checks
    can assert fail-closed behaviour without a device.
    """

    def __init__(
        self,
        *,
        state: int = 4,
        mode: int = 6,
        error_code: int = 0,
        connected: bool = True,
        joint_state: np.ndarray | None = None,
    ) -> None:
        self.state = state
        self.mode = mode
        self.error_code = error_code
        self.connected = connected
        self.joint_state = (
            np.zeros(ARM_JOINT_SHAPE, dtype=np.float64)
            if joint_state is None
            else np.asarray(joint_state, dtype=np.float64)
        )
        # Report-cache identity surface used by the arm worker's startup
        # identity validation (safe defaults; checks may override them).
        self.axis = 7
        self.device_type = "xArm7"
        self.sn = "FAKE"
        self.version = "1.18.4"
        self.version_number = (1, 18, 4)
        # Report-cache rated dynamics limits (per-joint lists in radian units,
        # matching a connected xArm7); safe defaults for the connect-time check.
        self.joint_speed_limit = [float(np.pi)] * 7
        self.joint_acc_limit = [20.0] * 7
        self.calls: list[tuple[str, tuple, dict]] = []
        self._codes: dict[str, int] = {}
        self._raise: dict[str, Exception] = {}

    def fail(self, name: str, code: int = 1) -> None:
        """Make ``name`` return a non-zero (failure) code."""
        self._codes[name] = code

    def raise_on(self, name: str, exc: Exception) -> None:
        """Make ``name`` raise ``exc`` instead of returning."""
        self._raise[name] = exc

    def call_order(self) -> list[str]:
        """Return the SDK method names in the order they were invoked."""
        return [name for name, _, _ in self.calls]

    def first_call_index(self, name: str) -> int:
        """Index of the first call to ``name``, or ``-1`` when absent."""
        for index, (call_name, _, _) in enumerate(self.calls):
            if call_name == name:
                return index
        return -1

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def _maybe_raise(self, name: str) -> None:
        if name in self._raise:
            raise self._raise[name]

    def _code(self, name: str) -> int:
        return self._codes.get(name, 0)

    # -- xArm SDK surface ------------------------------------------------
    def clean_error(self) -> int:
        self._record("clean_error", (), {})
        self._maybe_raise("clean_error")
        return self._code("clean_error")

    def clean_warn(self) -> int:
        self._record("clean_warn", (), {})
        self._maybe_raise("clean_warn")
        return self._code("clean_warn")

    def motion_enable(self, enable: bool = True, servo_id: int | None = None) -> int:
        self._record("motion_enable", (enable, servo_id), {})
        self._maybe_raise("motion_enable")
        return self._code("motion_enable")

    def set_mode(self, mode: int) -> int:
        self._record("set_mode", (mode,), {})
        self._maybe_raise("set_mode")
        if self._code("set_mode") == 0:
            self.mode = mode
        return self._code("set_mode")

    def set_state(self, state: int) -> int:
        self._record("set_state", (state,), {})
        self._maybe_raise("set_state")
        if self._code("set_state") != 0:
            return self._code("set_state")
        # Faithfully model the SDK: State 0 is "not ready", State 1 is moving,
        # State 2 is idle, State 4 is stopped.  State 0 does NOT resolve to
        # State 2 — that was a test-only assumption that masked the P0 bug.
        self.state = state
        return 0

    def emergency_stop(self) -> None:
        self._record("emergency_stop", (), {})
        self._maybe_raise("emergency_stop")
        # SDK 1.18.4 returns no integer code; a successful call is a no-op here.

    def get_state(self) -> tuple[int, int]:
        self._record("get_state", (), {})
        self._maybe_raise("get_state")
        return self._code("get_state"), self.state

    def get_err_warn_code(self) -> tuple[int, tuple[int, ...]]:
        self._record("get_err_warn_code", (), {})
        self._maybe_raise("get_err_warn_code")
        return self._code("get_err_warn_code"), (self.error_code, 0)

    def get_joint_states(self, is_radian: bool = True, num: int = 1) -> tuple[int, list]:
        self._record("get_joint_states", (is_radian, num), {})
        self._maybe_raise("get_joint_states")
        return self._code("get_joint_states"), [self.joint_state.copy() for _ in range(max(1, num))]

    def set_servo_angle(
        self,
        *,
        angle: Any,
        is_radian: bool = True,
        speed: float = 0.0,
        mvacc: float = 0.0,
        wait: bool = False,
    ) -> int:
        self._record("set_servo_angle", (angle, is_radian, speed, mvacc, wait), {})
        self._maybe_raise("set_servo_angle")
        return self._code("set_servo_angle")

    def disconnect(self) -> int:
        self._record("disconnect", (), {})
        self._maybe_raise("disconnect")
        return self._code("disconnect")


class FakeHand:
    """Recording stand-in for the XHand driver surface used by ``hand_loop``.

    Models the driver's public contract (not the vendor SDK's internal
    ``control`` object): ``connect`` / ``get_state`` / ``send_action`` /
    ``disconnect`` / ``initialize_tactile`` plus the
    fields ``hand_loop`` reads (``connected_flag``, ``error_state``,
    ``last_qpos_cmd``, ``tactile_calibrated``, ``device_identity``).  Every
    method appends ``(name, args, kwargs)`` to :attr:`calls`; a method can be
    made to return a failure via :meth:`fail` or to raise via :meth:`raise_on`,
    so worker checks can assert fail-closed behaviour without a device.
    """

    def __init__(
        self,
        *,
        connected: bool = False,
        error_state: bool = False,
        tactile_calibrated: bool = False,
        qpos: np.ndarray | None = None,
        last_qpos_cmd: np.ndarray | None = None,
    ) -> None:
        self.connected_flag = connected
        self.error_state = error_state
        self.tactile_calibrated = tactile_calibrated
        self._qpos = (
            np.zeros(HAND_JOINT_SHAPE, dtype=np.float64)
            if qpos is None
            else np.asarray(qpos, dtype=np.float64)
        )
        self.last_qpos_cmd = (
            self._qpos.copy()
            if last_qpos_cmd is None
            else np.asarray(last_qpos_cmd, dtype=np.float64)
        )
        # Device-identity surface read by the hand worker's startup block.
        self.device_identity: dict[str, str] = {
            "backend": "fake",
            "hand_type": "right",
            "sdk_version": "fake",
            "serial_number": "fake",
        }
        self.calls: list[tuple[str, tuple, dict]] = []
        self._codes: dict[str, int] = {}
        self._raise: dict[str, Exception] = {}

    def fail(self, name: str, code: int = 1) -> None:
        """Make ``name`` behave as a failed SDK call (False / non-zero)."""
        self._codes[name] = code

    def raise_on(self, name: str, exc: Exception) -> None:
        """Make ``name`` raise ``exc`` instead of returning."""
        self._raise[name] = exc

    def call_order(self) -> list[str]:
        """Return the driver method names in invocation order."""
        return [name for name, _, _ in self.calls]

    def first_call_index(self, name: str) -> int:
        """Index of the first call to ``name``, or ``-1`` when absent."""
        for index, (call_name, _, _) in enumerate(self.calls):
            if call_name == name:
                return index
        return -1

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def _maybe_raise(self, name: str) -> None:
        if name in self._raise:
            raise self._raise[name]

    # -- XHand driver surface -------------------------------------------
    def connect(self) -> bool:
        self._record("connect", (), {})
        self._maybe_raise("connect")
        if self._codes.get("connect", 0) != 0:
            return False
        self.connected_flag = True
        return True

    def get_state(self, force_update: bool | None = None) -> XHandSample:
        self._record("get_state", (force_update,), {})
        self._maybe_raise("get_state")
        return XHandSample(
            qpos=self._qpos.copy(),
            current=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            tactile_force=np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64),
            tactile_sum=np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64),
            tactile_contact=np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
            commboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            jointboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            tipboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
        )

    def send_action(self, action: np.ndarray) -> bool:
        self._record("send_action", (action,), {})
        self._maybe_raise("send_action")
        if self._codes.get("send_action", 0) != 0:
            return False
        # Faithful to the driver: only an SDK success advances last_qpos_cmd.
        self.last_qpos_cmd = np.asarray(action, dtype=np.float64).copy()
        return True

    def disconnect(self) -> None:
        self._record("disconnect", (), {})
        self._maybe_raise("disconnect")
        self.connected_flag = False

    def initialize_tactile(self) -> bool:
        # Phase 4 splits tactile reset/bias out of connect(); modelled here as
        # the explicit worker-invoked step (see check_tactile_init).
        self._record("initialize_tactile", (), {})
        self._maybe_raise("initialize_tactile")
        if self._codes.get("initialize_tactile", 0) != 0:
            return False
        self.tactile_calibrated = True
        return True


def make_arm_state_frame(
    qpos: np.ndarray,
    *,
    last_cmd_seq: int = 0,
    connected: int = 1,
    state_valid: int = 1,
    error_code: int = 0,
) -> np.ndarray:
    """Build one valid ``ARM_STATE_DTYPE`` frame at the given feedback pose."""
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape != ARM_JOINT_SHAPE:
        raise ValueError(f"arm qpos must have shape {ARM_JOINT_SHAPE}")
    frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
    frame["qpos"][0] = q
    frame["qvel"][0] = np.zeros(ARM_JOINT_SHAPE, dtype=np.float64)
    frame["tau"][0] = np.zeros(ARM_JOINT_SHAPE, dtype=np.float64)
    frame["eef_pos"][0] = np.zeros(3, dtype=np.float64)
    frame["eef_rot6d"][0] = np.zeros(6, dtype=np.float64)
    frame["error_code"][0] = error_code
    frame["connected"][0] = connected
    frame["mode"][0] = 6
    frame["tracking_err"][0] = 0.0
    frame["last_cmd_seq"][0] = last_cmd_seq
    now_ns = time.monotonic_ns()
    frame["source_monotonic_ns"][0] = now_ns
    frame["publish_monotonic_ns"][0] = now_ns
    frame["state_valid"][0] = state_valid
    frame["timestamp"][0] = now_ns / 1e9
    return frame


def make_hand_state_frame(
    qpos: np.ndarray,
    *,
    last_cmd_seq: int = 0,
    connected: int = 1,
    state_valid: int = 1,
    send_healthy: int = 1,
    read_healthy: int = 1,
    error_state: int = 0,
    source_monotonic_ns: int | None = None,
) -> np.ndarray:
    """Build one valid ``HAND_STATE_DTYPE`` frame at the given feedback pose."""
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape != HAND_JOINT_SHAPE:
        raise ValueError(f"hand qpos must have shape {HAND_JOINT_SHAPE}")
    frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
    frame["qpos"][0] = q
    frame["current"][0] = q
    frame["last_cmd_seq"][0] = last_cmd_seq
    frame["last_cmd_qpos"][0] = q
    frame["connected"][0] = connected
    frame["state_valid"][0] = state_valid
    frame["send_healthy"][0] = send_healthy
    frame["read_healthy"][0] = read_healthy
    frame["error_state"][0] = error_state
    frame["qpos_stale"][0] = 0
    now_ns = time.monotonic_ns()
    src_ns = now_ns if source_monotonic_ns is None else source_monotonic_ns
    frame["source_monotonic_ns"][0] = src_ns
    frame["publish_monotonic_ns"][0] = now_ns
    frame["timestamp"][0] = now_ns / 1e9
    return frame
