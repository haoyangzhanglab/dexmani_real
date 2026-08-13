"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

Primary entry point: ``arm_loop(shared)`` — mp.Process target using SharedStorage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Callable

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.defaults import arm, safety
from dexmani_real.planning.kinematics import ArmFK
from dexmani_real.planning.path_utils import wrap_nearest_equivalent
from dexmani_real.policy.safety import worker_validate_arm
from dexmani_real.robot.homing import HOME_SENTINEL, HomeRequest, HomeResult
from dexmani_real.robot.safety import SafetyState
from dexmani_real.robot.types import ArmControlKind
from dexmani_real.runtime.status import ComponentPhase, FaultCode
from dexmani_real.shm.shared_storage import new_frame, publish_component_status
from dexmani_real.utils.log import (
    ThrottledWarner,
    capture_native_stdout,
    extract_native_diagnostics,
    get_logger,
)
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter
from dexmani_real.utils.schema import (
    ARM_COMMAND_DTYPE,
    ARM_CONTROL_DTYPE,
    ARM_JOINT_SHAPE,
    ARM_STATE_DTYPE,
)

logger = get_logger(__name__)


@dataclass
class ArmLoopConfig:
    """Mode 6 joint online trajectory planning configuration."""

    joint_max_speed_rad_per_s: float = field(
        default_factory=lambda: arm.max_joint_velocity_rad_per_s
    )
    joint_max_acc_rad_per_s2: float = field(
        default_factory=lambda: arm.max_joint_acceleration_rad_per_s2
    )
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)

    joint_limit_lower: tuple[float, ...] = field(
        default_factory=lambda: arm.joint_limit_lower
    )
    joint_limit_upper: tuple[float, ...] = field(
        default_factory=lambda: arm.joint_limit_upper
    )

    tracking_error_warn_rad: float = field(
        default_factory=lambda: arm.tracking_error_warn_rad
    )

    arm_ip: str = field(default_factory=lambda: arm.ip)

    home_qpos: tuple[float, ...] = field(default_factory=lambda: arm.home_qpos)

    collision_sensitivity: int = field(
        default_factory=lambda: arm.collision_sensitivity
    )
    recoverable_errors: frozenset[int] = field(
        default_factory=lambda: arm.recoverable_errors
    )
    collision_fault_errors: frozenset[int] = field(
        default_factory=lambda: arm.collision_fault_errors
    )
    max_consecutive_recoveries: int = field(
        default_factory=lambda: safety.max_consecutive_recoveries
    )

    homing_convergence_rad: float = field(
        default_factory=lambda: arm.homing.convergence_rad
    )
    homing_step_interval_s: float = field(
        default_factory=lambda: arm.homing.step_interval_s
    )
    homing_max_speed_rad_per_s: float = field(
        default_factory=lambda: np.deg2rad(arm.homing.max_speed_deg_s)
    )
    homing_target_timeout_s: float = field(
        default_factory=lambda: arm.homing.target_timeout_s
    )
    homing_velocity_convergence_rad_s: float = field(
        default_factory=lambda: arm.homing.velocity_convergence_rad_s
    )
    homing_dwell_s: float = field(default_factory=lambda: arm.homing.dwell_s)

    def __post_init__(self) -> None:
        lower = np.asarray(self.joint_limit_lower, dtype=np.float64)
        upper = np.asarray(self.joint_limit_upper, dtype=np.float64)
        home = np.asarray(self.home_qpos, dtype=np.float64)
        if (
            lower.shape != ARM_JOINT_SHAPE
            or upper.shape != ARM_JOINT_SHAPE
            or home.shape != ARM_JOINT_SHAPE
        ):
            raise ValueError(
                f"arm loop joint limits/home must have shape {ARM_JOINT_SHAPE}"
            )
        if not np.all(np.isfinite(np.concatenate((lower, upper, home)))) or np.any(
            lower > upper
        ):
            raise ValueError("arm loop joint limits/home must be finite and ordered")
        if self.recoverable_errors & self.collision_fault_errors:
            raise ValueError(
                "recoverable and collision-fault error codes must be disjoint"
            )
        if self.recoverable_errors != frozenset({24}) or not frozenset(
            {22, 31}
        ).issubset(self.collision_fault_errors):
            raise ValueError(
                "arm loop requires only C24 recoverable and C22/C31 collision-fatal"
            )
        if self.max_consecutive_recoveries <= 0:
            raise ValueError("max_consecutive_recoveries must be positive")
        timing = (
            self.joint_max_speed_rad_per_s,
            self.joint_max_acc_rad_per_s2,
            self.arm_loop_hz,
            self.tracking_error_warn_rad,
            self.homing_convergence_rad,
            self.homing_step_interval_s,
            self.homing_max_speed_rad_per_s,
            self.homing_target_timeout_s,
            self.homing_velocity_convergence_rad_s,
            self.homing_dwell_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError(
                "arm loop motion/homing parameters must be finite and positive"
            )
        if not self.arm_ip or not (0 <= self.collision_sensitivity <= 5):
            raise ValueError("arm loop IP/collision sensitivity is invalid")

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ArmLoopConfig":
        cfg = runtime.arm
        return cls(
            joint_max_speed_rad_per_s=float(
                np.deg2rad(cfg.max_joint_velocity_deg_per_s)
            ),
            joint_max_acc_rad_per_s2=float(
                np.deg2rad(cfg.max_joint_acceleration_deg_per_s2)
            ),
            arm_loop_hz=float(cfg.loop_hz),
            joint_limit_lower=tuple(cfg.joint_limit_lower),
            joint_limit_upper=tuple(cfg.joint_limit_upper),
            tracking_error_warn_rad=float(cfg.tracking_error_warn_rad),
            arm_ip=str(cfg.ip),
            home_qpos=tuple(cfg.home_qpos),
            collision_sensitivity=int(cfg.collision_sensitivity),
            recoverable_errors=frozenset(int(code) for code in cfg.recoverable_errors),
            collision_fault_errors=frozenset(
                int(code) for code in cfg.collision_fault_errors
            ),
            max_consecutive_recoveries=int(runtime.safety.max_consecutive_recoveries),
            homing_convergence_rad=float(cfg.homing.convergence_rad),
            homing_step_interval_s=float(cfg.homing.step_interval_s),
            homing_max_speed_rad_per_s=float(np.deg2rad(cfg.homing.max_speed_deg_s)),
            homing_target_timeout_s=float(cfg.homing.target_timeout_s),
            homing_velocity_convergence_rad_s=float(
                cfg.homing.velocity_convergence_rad_s
            ),
            homing_dwell_s=float(cfg.homing.dwell_s),
        )


# Controller errors: C24 is recoverable; C22/C31 are immediate collision faults.
def _require_sdk_ok(operation: str, code: Any) -> None:
    """Raise when an xArm setter reports failure without raising."""
    if not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RuntimeError(f"{operation} failed with SDK code {code!r}")


def _recover_c24_measured_hold(
    arm_api: Any, cfg: ArmLoopConfig, *, operation_prefix: str = "C24"
) -> np.ndarray:
    """Clear one C24, read fresh joints, and send exactly one measured hold."""
    _require_sdk_ok(f"{operation_prefix} clean_error", arm_api.clean_error())
    _require_sdk_ok(f"{operation_prefix} clean_warn", arm_api.clean_warn())
    _require_sdk_ok(f"{operation_prefix} set_mode(6)", arm_api.set_mode(6))
    _require_sdk_ok(f"{operation_prefix} set_state(0)", arm_api.set_state(0))
    state_code, measured = arm_api.get_joint_states(is_radian=True, num=1)
    _require_sdk_ok(f"{operation_prefix} fresh get_joint_states", state_code)
    measured_hold = np.asarray(measured[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
    if measured_hold.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(measured_hold)):
        raise RuntimeError(f"{operation_prefix} measured hold is invalid")
    _require_sdk_ok(
        f"{operation_prefix} measured hold",
        arm_api.set_servo_angle(
            angle=measured_hold,
            is_radian=True,
            speed=cfg.joint_max_speed_rad_per_s,
            mvacc=cfg.joint_max_acc_rad_per_s2,
            wait=False,
        ),
    )
    return measured_hold.copy()


def _parse_arm_action_metadata(
    action: Any, received_s: float
) -> tuple[int, float, bool]:
    """Return ``(sequence, created_s, is_hold)`` for a fixed command frame."""
    if (
        isinstance(action, np.ndarray)
        and action.shape == (1,)
        and action.dtype == ARM_COMMAND_DTYPE
    ):
        command_seq = int(action["action_id"][0])
        created_s = int(action["created_monotonic_ns"][0]) / 1e9
        if not np.isfinite(created_s) or created_s <= 0.0 or created_s > received_s:
            created_s = received_s
        return command_seq, created_s, bool(action["is_hold"][0])
    return 0, received_s, False


def _arm_control_request_is_current(
    request: Any,
    *,
    expected_run_generation: int,
    now_monotonic_ns: int,
) -> bool:
    """Validate a fixed-dtype STOP/RESUME request at the worker boundary."""
    if (
        not isinstance(request, np.ndarray)
        or request.shape != (1,)
        or request.dtype != ARM_CONTROL_DTYPE
    ):
        return False
    kind = int(request["kind"][0])
    if kind not in (int(ArmControlKind.DECELERATED_STOP), int(ArmControlKind.RESUME)):
        return False
    integer_fields = tuple(
        int(request[name][0])
        for name in (
            "action_id",
            "run_generation",
            "created_monotonic_ns",
            "valid_until_monotonic_ns",
        )
    )
    if not all(value > 0 for value in integer_fields):
        return False
    _action_id, run_generation, created_ns, valid_until_ns = integer_fields
    return bool(
        run_generation == int(expected_run_generation)
        and created_ns <= int(now_monotonic_ns)
        and valid_until_ns > int(now_monotonic_ns)
        and valid_until_ns > created_ns
    )


def _enter_mode6_ready(arm_api: Any, *, operation_prefix: str) -> None:
    """Enter Mode 6/State 0 and verify the controller reports ready State 2."""
    _require_sdk_ok(f"{operation_prefix} set_mode(6)", arm_api.set_mode(6))
    _require_sdk_ok(f"{operation_prefix} set_state(0)", arm_api.set_state(0))
    _wait_live_status(arm_api, expected_state=2, expected_mode=6)


def _resume_mode6_with_measured_hold(
    arm_api: Any,
    cfg: ArmLoopConfig,
    *,
    operation_prefix: str = "resume",
) -> np.ndarray:
    """Resume Mode 6 behind a fresh, zero-lead worker-measured endpoint."""
    state_code, states = arm_api.get_joint_states(is_radian=True, num=1)
    measured_qpos, _qvel, _tau = _decode_joint_state_feedback(state_code, states)
    _require_sdk_ok(f"{operation_prefix} set_mode(6)", arm_api.set_mode(6))
    _require_sdk_ok(f"{operation_prefix} set_state(0)", arm_api.set_state(0))
    # Publish the fresh anchor immediately after State 0 is accepted. Waiting
    # for State 2 first would leave a window in which firmware could continue
    # toward the endpoint superseded by the preceding State-6 stop.
    _require_sdk_ok(
        f"{operation_prefix} measured hold",
        arm_api.set_servo_angle(
            angle=measured_qpos,
            is_radian=True,
            speed=cfg.joint_max_speed_rad_per_s,
            mvacc=cfg.joint_max_acc_rad_per_s2,
            wait=False,
        ),
    )
    _wait_live_status(arm_api, expected_state=2, expected_mode=6, timeout_s=0.5)
    return measured_qpos.copy()


def _apply_decelerated_stop(
    arm_api: Any, *, operation_prefix: str = "decelerated stop"
) -> int:
    """Request State 6 braking and return the controller's reported stop state.

    ``set_state(6)`` is a command, not a stable feedback postcondition.  xArm
    firmware 2.7.1 reports State 5 once that request is accepted; other SDK /
    firmware combinations may briefly retain State 6.  Both prevent normal
    command execution until State 0; the caller separately verifies measured
    velocity has settled before it re-anchors keyboard motion.
    """
    _require_sdk_ok(f"{operation_prefix} set_state(6)", arm_api.set_state(6))
    return _wait_live_status(
        arm_api,
        expected_state=(5, 6),
        expected_mode=6,
        timeout_s=0.25,
    )


def _take_next_current_arm_action(
    action_q: Any, *, expected_run_generation: int
) -> Any | None:
    """Return one actionable queue item while draining invalidated endpoints."""
    while True:
        try:
            queued_action = action_q.get(timeout=0.0)
        except Empty:
            return None
        now_ns = time.monotonic_ns()
        if isinstance(queued_action, np.ndarray):
            if worker_validate_arm(
                queued_action,
                expected_run_generation=expected_run_generation,
                now_monotonic_ns=now_ns,
            ):
                return queued_action
            logger.info(
                "arm_loop: discarded malformed, stale-generation, or expired command"
            )
            continue
        return queued_action


def _take_latest_arm_control_request(
    control_ring: Any,
    *,
    last_sequence: int,
    expected_run_generation: int,
) -> tuple[np.ndarray | None, int]:
    """Consume at most one new priority control request from a latest-wins ring."""
    result = control_ring.read_latest()
    if result is None:
        return None, int(last_sequence)
    request, _publish_ns, sequence = result
    sequence = int(sequence)
    if sequence <= int(last_sequence):
        return None, int(last_sequence)
    if _arm_control_request_is_current(
        request,
        expected_run_generation=expected_run_generation,
        now_monotonic_ns=time.monotonic_ns(),
    ):
        return request.copy(), sequence
    logger.info(
        "arm_loop: discarded malformed, stale-generation, or expired arm control request"
    )
    return None, sequence


def _decode_joint_state_feedback(
    code: Any, states: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one xArm feedback response at the worker boundary."""
    _require_sdk_ok("get_joint_states", code)
    if not isinstance(states, (list, tuple)) or not states:
        raise RuntimeError("get_joint_states returned no joint state")
    qpos = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
    qvel = (
        np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
        if len(states) > 1
        else np.zeros(ARM_JOINT_SHAPE, dtype=np.float64)
    )
    tau = (
        np.asarray(states[2], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
        if len(states) > 2
        else np.zeros(ARM_JOINT_SHAPE, dtype=np.float64)
    )
    for name, value in (("qpos", qpos), ("qvel", qvel), ("tau", tau)):
        if value.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"get_joint_states returned invalid {name}: shape={value.shape}"
            )
    return qpos, qvel, tau


def _update_state_read_watchdog(counter: RetryCounter, *, succeeded: bool) -> bool:
    """Update the consecutive feedback-failure counter and report escalation."""
    if succeeded:
        counter.reset()
        return False
    counter.inc()
    return counter.triggered


def _read_live_status(arm_api: Any) -> tuple[int, int, int]:
    """Return live ``(state, mode, error)`` or raise on any failed read."""
    if hasattr(arm_api, "get_state"):
        code, state = arm_api.get_state()
        _require_sdk_ok("get_state", code)
    else:
        state = getattr(arm_api, "state")
    mode = int(getattr(arm_api, "mode"))
    code, values = arm_api.get_err_warn_code()
    _require_sdk_ok("get_err_warn_code", code)
    return int(state), mode, int(values[0])


def _wait_live_status(
    arm_api: Any,
    *,
    expected_state: int | tuple[int, ...],
    expected_mode: int | None = None,
    timeout_s: float = 1.0,
) -> int:
    expected_states = (
        (expected_state,) if isinstance(expected_state, int) else tuple(expected_state)
    )
    if not expected_states or any(
        not isinstance(value, int) for value in expected_states
    ):
        raise ValueError(
            "expected_state must contain one or more integer controller states"
        )
    deadline = time.monotonic() + timeout_s
    last: tuple[int, int, int] | None = None
    while time.monotonic() < deadline:
        last = _read_live_status(arm_api)
        state, mode, error = last
        if (
            state in expected_states
            and error == 0
            and (expected_mode is None or mode == expected_mode)
        ):
            return state
        time.sleep(0.03)
    expected_label = (
        expected_states[0] if len(expected_states) == 1 else expected_states
    )
    raise RuntimeError(
        f"controller postcondition failed: expected state={expected_label} mode={expected_mode} error=0, got {last}"
    )


def _latch_collision_fault(shared: Any, arm_api: Any, error_code: int) -> None:
    details: Any = None
    if error_code == 31 and hasattr(arm_api, "get_c31_error_info"):
        try:
            code, info = arm_api.get_c31_error_info()
            if code == 0:
                details = info
        except Exception:
            logger.warning("arm_loop: failed to read C31 diagnostics", exc_info=True)
    if (
        error_code == 31
        and isinstance(details, (list, tuple, np.ndarray))
        and len(details) >= 3
    ):
        try:
            servo_id = int(details[0])
            theoretical_tau = float(details[1])
            actual_tau = float(details[2])
        except (TypeError, ValueError, OverflowError):
            logger.error("arm_loop: collision fault C31 detected; details=%s", details)
        else:
            logger.error(
                "arm_loop: collision fault C31 detected; servo_id=%d "
                "theoretical_tau=%.3f actual_tau=%.3f delta_tau=%.3f",
                servo_id,
                theoretical_tau,
                actual_tau,
                actual_tau - theoretical_tau,
            )
    else:
        logger.error(
            "arm_loop: collision fault C%d detected; details=%s", error_code, details
        )
    shared.error_state.value = True


def _publish_startup_fault_impl(shared: Any, detail: str) -> None:
    publish_component_status(
        shared,
        "arm",
        ComponentPhase.FAULT,
        fault_code=FaultCode.STARTUP_FAILED,
        detail=detail,
    )


def arm_loop(shared, config: ArmLoopConfig | None = None) -> None:
    """Arm process entry point — reads arm_action_q, servos arm via Mode 6.

    mp.Process target communicating exclusively through SharedStorage.
    """
    _tracking_warn = ThrottledWarner(interval_s=5.0)
    _fk_warn = ThrottledWarner(interval_s=5.0)
    _state_read_warn = ThrottledWarner(interval_s=5.0)
    cfg = config or ArmLoopConfig()
    _recovery_counter = RetryCounter(
        max_consecutive=cfg.max_consecutive_recoveries, label="arm_servo"
    )
    _state_error_counter = RetryCounter(
        max_consecutive=cfg.max_consecutive_recoveries, label="arm_state"
    )
    _tracking_err_count = 0
    publish_component_status(shared, "arm", ComponentPhase.LOADING)

    def _publish_startup_fault(detail: str) -> None:
        return _publish_startup_fault_impl(shared, detail)

    # URDF-consistent FK (replaces arm.get_position_aa). xArm firmware uses a
    # different EEF coordinate definition — Pinocchio FK ensures all consumers
    # share a single coordinate system.
    _urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    _arm_fk = ArmFK(_urdf_path)

    sdk_connect_output = None
    try:
        # The vendor SDK prints a multi-line banner directly to stdout and its
        # ``origin.print`` logger announces every ready/not-ready transition.
        # Our worker validates and reports those states itself, so suppress the
        # successful duplicate chatter while retaining SDK WARNING/ERROR logs.
        with capture_native_stdout() as sdk_connect_output:
            from xarm.wrapper import XArmAPI

            logging.getLogger("origin.print").setLevel(logging.WARNING)
            arm = XArmAPI(cfg.arm_ip, is_radian=True)
        sdk_diagnostics = extract_native_diagnostics(sdk_connect_output.text)
        if sdk_diagnostics:
            logger.warning(
                "xArm SDK initialization diagnostics:\n%s", "\n".join(sdk_diagnostics)
            )
    except Exception as e:
        vendor_detail = (
            sdk_connect_output.text if sdk_connect_output is not None else ""
        )
        logger.error(
            "arm_loop: connect failed: %s%s",
            e,
            f"; vendor output:\n{vendor_detail}" if vendor_detail else "",
        )
        _publish_startup_fault("SDK connect failed")
        shared.error_state.value = True
        return

    if hasattr(shared, "arm_device_identity"):
        identity = {
            "device_type": str(getattr(arm, "device_type", "unavailable")),
            "firmware_version": str(getattr(arm, "version", "unavailable")),
            "model": "xArm7",
            "serial_number": str(getattr(arm, "sn", "unavailable")),
        }
        encoded_identity = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        shared.arm_device_identity.value = encoded_identity[:1023].ljust(1024, b"\x00")

    # Connect/configure while the application remains DISARMED. Motion mode is
    # entered only on the ARMED edge below.
    _CONNECT_MAX_RETRIES = 3
    for _attempt in range(_CONNECT_MAX_RETRIES):
        try:
            _require_sdk_ok("startup clean_error", arm.clean_error())
            _require_sdk_ok("startup clean_warn", arm.clean_warn())
            _require_sdk_ok("startup motion_enable", arm.motion_enable(True))
            time.sleep(0.3)
            _require_sdk_ok("startup set_mode(0)", arm.set_mode(0))
            time.sleep(0.1)
            _require_sdk_ok("startup set_state(0)", arm.set_state(0))
            time.sleep(0.3)
            # Live read via get_err_warn_code() — more reliable than the cached
            # .error_code property (background report thread, ~200ms refresh).
            try:
                _rc, _codes = arm.get_err_warn_code()
                _err = _codes[0] if _rc == 0 else 1
            except Exception:
                _err = getattr(arm, "error_code", 0) or 0
            _state = getattr(arm, "state", -1)
            if _err == 0 and _state == 2:
                break
            logger.warning(
                "arm_loop: connect recovery attempt %d/%d: err=%s state=%s",
                _attempt + 1,
                _CONNECT_MAX_RETRIES,
                _err,
                _state,
            )
            time.sleep(0.5)
        except Exception:
            logger.warning(
                "arm_loop: connect recovery attempt %d/%d raised exception",
                _attempt + 1,
                _CONNECT_MAX_RETRIES,
                exc_info=True,
            )
            time.sleep(0.5)
    else:
        logger.error(
            "arm_loop: connect recovery failed after %d attempts", _CONNECT_MAX_RETRIES
        )
        _publish_startup_fault("controller recovery failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return

    # Post-connect configuration. Every setter return code is authoritative.
    try:
        _require_sdk_ok(
            "set_collision_sensitivity",
            arm.set_collision_sensitivity(cfg.collision_sensitivity),
        )
        # TCP load: XHand (1.1 kg). COG in tool-flange frame (link_eef) from
        # URDF weighted-COM of all end-effector links; flange_joint2 corrected
        # 0.043→0.033 m per physical measurement.
        _require_sdk_ok(
            "set_tcp_load",
            arm.set_tcp_load(weight=1.1, center_of_gravity=[16.3, 7.9, 109.5]),
        )
        # Torque-based collision detection (level 1, least-sensitive enabled
        # setting). Keep this firmware backstop enabled during intentional contact.
        _require_sdk_ok(
            "set_joint_maxacc",
            arm.set_joint_maxacc(cfg.joint_max_acc_rad_per_s2, is_radian=True),
        )
    except Exception as e:
        logger.error("arm_loop: post-recovery config failed: %s", e)
        _publish_startup_fault("controller configuration failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return

    # A single failed startup read is not enough to declare the controller dead.
    _STATE_READ_MAX_RETRIES = 10
    for _attempt in range(_STATE_READ_MAX_RETRIES):
        try:
            code, states = arm.get_joint_states(is_radian=True, num=1)
            if code == 0 and len(states) > 0:
                last_qpos = np.asarray(states[0], dtype=np.float64)[
                    : ARM_JOINT_SHAPE[0]
                ].copy()
                break
            logger.warning(
                "arm_loop: initial joint state read attempt %d/%d: code=%d",
                _attempt + 1,
                _STATE_READ_MAX_RETRIES,
                code,
            )
        except Exception:
            logger.warning(
                "arm_loop: initial joint state read attempt %d/%d raised exception",
                _attempt + 1,
                _STATE_READ_MAX_RETRIES,
                exc_info=True,
            )
        time.sleep(0.1)
    else:
        logger.error(
            "arm_loop: cannot read initial joint states after %d attempts",
            _STATE_READ_MAX_RETRIES,
        )
        _publish_startup_fault("initial feedback unavailable")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    last_target = last_qpos.copy()
    # Ready means connected/configured and physically STOPPED, not Mode-6
    # motion-enabled. Confirm state 4 before exposing arm_ready.
    try:
        _require_sdk_ok("startup set_state(4)", arm.set_state(4))
        _wait_live_status(arm, expected_state=4)
    except Exception:
        logger.error(
            "arm_loop: failed to enter confirmed DISARMED state", exc_info=True
        )
        _publish_startup_fault("confirmed state-4 startup stop failed")
        shared.error_state.value = True
        _disconnect_arm(arm)
        return
    last_cmd_seq = 0
    last_cmd_created_s = 0.0
    last_cmd_received_s = 0.0
    last_cmd_applied_s = 0.0
    last_cmd_queue_latency_s = 0.0
    last_cmd_apply_latency_s = 0.0
    last_cmd_sdk_duration_s = 0.0
    last_cmd_is_hold = False
    last_arm_control_sequence = 0
    motion_enabled = False
    controller_decel_stopped = False
    last_safety_state = int(SafetyState.DISARMED)
    last_state_source_ns = time.monotonic_ns()
    last_c24_s = float("-inf")
    terminal_feedback_fault = False

    # Publish initial state BEFORE arm_ready — consumers wait on arm_ready and
    # expect the ring to already contain a valid frame.  Without this, there is
    # a one-tick window where arm_ready is set but arm_state_ring is empty.
    try:
        eef_pos_init, eef_rot6d_init = _arm_fk.compute(last_qpos)
    except Exception:
        eef_pos_init = np.zeros(3, dtype=np.float64)
        eef_rot6d_init = np.zeros(6, dtype=np.float64)
    _frame = new_frame(ARM_STATE_DTYPE)
    _frame["qpos"][0] = last_qpos
    _frame["qvel"][0] = np.zeros(ARM_JOINT_SHAPE, dtype=np.float64)
    _frame["tau"][0] = np.zeros(ARM_JOINT_SHAPE, dtype=np.float64)
    _frame["eef_pos"][0] = eef_pos_init
    _frame["eef_rot6d"][0] = eef_rot6d_init
    _frame["error_code"][0] = 0
    _frame["connected"][0] = 1
    _frame["mode"][0] = getattr(arm, "mode", 6)
    _frame["tracking_err"][0] = 0.0
    _frame["last_cmd_seq"][0] = last_cmd_seq
    _frame["last_cmd_created_s"][0] = last_cmd_created_s
    _frame["last_cmd_received_s"][0] = last_cmd_received_s
    _frame["last_cmd_applied_s"][0] = last_cmd_applied_s
    _frame["last_cmd_queue_latency_s"][0] = last_cmd_queue_latency_s
    _frame["last_cmd_apply_latency_s"][0] = last_cmd_apply_latency_s
    _frame["last_cmd_sdk_duration_s"][0] = last_cmd_sdk_duration_s
    _frame["last_cmd_is_hold"][0] = int(last_cmd_is_hold)
    _frame["source_monotonic_ns"][0] = last_state_source_ns
    _frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    _frame["state_valid"][0] = 1
    _frame["timestamp"][0] = last_state_source_ns / 1e9
    shared.arm_state_ring.write(_frame)

    # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
    # (same pattern as vr_loop).  Main's supervisor checks heartbeats immediately
    # after all ready events.
    shared.arm_heartbeat_s.value = time.monotonic()
    shared.arm_ready.set()
    publish_component_status(shared, "arm", ComponentPhase.READY)
    logger.info(
        "arm_loop: ready and DISARMED (state=4, ip=%s, hz=%.0f)",
        cfg.arm_ip,
        cfg.arm_loop_hz,
    )

    limiter = RateManager(cfg.arm_loop_hz)
    low = np.asarray(cfg.joint_limit_lower, dtype=np.float64)
    high = np.asarray(cfg.joint_limit_upper, dtype=np.float64)

    while shared.is_running.value:
        c24_recovered_this_tick = False
        # Heartbeat — written even when holding position (proves we're alive)
        shared.arm_heartbeat_s.value = time.monotonic()

        if shared.estop_request.value:
            # SDK emergency_stop() is the fastest path to kill motor power.
            # Fall back to set_state(4) in cleanup if the SDK method is
            # unavailable or fails (belt-and-suspenders per Xarm7-).
            try:
                arm.emergency_stop()
            except Exception:
                logger.warning(
                    "arm_loop: emergency_stop call failed; cleanup will enforce state 4",
                    exc_info=True,
                )
            break

        # Safety-state/controller lifecycle edges. DISARMED/FAULT always map to
        # confirmed controller state 4; ARMED enters Mode 6 then state 0 and
        # confirms the live postcondition before accepting commands.
        # When gated (DISARMED or FAULT), skip action read + servo but continue
        # to publish state (for monitoring) and rate-limit normally.
        _safety = shared.safety_state.value
        if _safety in (SafetyState.DISARMED, SafetyState.FAULT):
            if motion_enabled or last_safety_state not in (
                SafetyState.DISARMED,
                SafetyState.FAULT,
            ):
                try:
                    _require_sdk_ok("safety set_state(4)", arm.set_state(4))
                    _wait_live_status(arm, expected_state=4)
                except Exception:
                    logger.error("arm_loop: failed to confirm safe stop", exc_info=True)
                    shared.error_state.value = True
                    break
                motion_enabled = False
                controller_decel_stopped = False
        elif (
            _safety in (SafetyState.ARMED, SafetyState.RUNNING)
            and not motion_enabled
            and not controller_decel_stopped
        ):
            try:
                _enter_mode6_ready(arm, operation_prefix="armed")
            except Exception:
                logger.error(
                    "arm_loop: failed ARMED Mode-6 postcondition", exc_info=True
                )
                shared.error_state.value = True
                try:
                    arm.set_state(4)
                except Exception:
                    logger.error("arm_loop: fallback stop failed", exc_info=True)
                break
            motion_enabled = True
        last_safety_state = int(_safety)
        if (
            _safety in (SafetyState.ARMED, SafetyState.RUNNING)
            and not shared.error_state.value
        ):
            # STOP/RESUME control is latest-wins and has priority over the
            # ordered endpoint queue.  It therefore remains deliverable even
            # when two Mode-6 endpoints already occupy that bounded queue.
            control_request, last_arm_control_sequence = (
                _take_latest_arm_control_request(
                    shared.arm_control_ring,
                    last_sequence=last_arm_control_sequence,
                    expected_run_generation=int(shared.run_generation.value),
                )
            )
            action = None
            if control_request is not None:
                control_kind = ArmControlKind(int(control_request["kind"][0]))
                received_s = time.monotonic()
                sdk_started_s = received_s
                try:
                    if control_kind is ArmControlKind.DECELERATED_STOP:
                        reported_stop_state = (
                            _wait_live_status(
                                arm,
                                expected_state=(5, 6),
                                expected_mode=6,
                                timeout_s=0.25,
                            )
                            if controller_decel_stopped
                            else _apply_decelerated_stop(arm)
                        )
                        last_target = last_qpos.copy()
                        motion_enabled = False
                        controller_decel_stopped = True
                    else:
                        resumed_qpos = _resume_mode6_with_measured_hold(
                            arm,
                            cfg,
                            operation_prefix="explicit resume",
                        )
                        last_qpos = resumed_qpos.copy()
                        last_target = resumed_qpos.copy()
                        motion_enabled = True
                        controller_decel_stopped = False
                except Exception:
                    operation = (
                        "decelerated stop"
                        if control_kind is ArmControlKind.DECELERATED_STOP
                        else "resume"
                    )
                    logger.error(
                        "arm_loop: failed to confirm controller %s postcondition",
                        operation,
                        exc_info=True,
                    )
                    try:
                        _require_sdk_ok(
                            f"{operation} fallback set_state(4)", arm.set_state(4)
                        )
                        _wait_live_status(arm, expected_state=4)
                    except Exception:
                        logger.error(
                            "arm_loop: %s fallback state 4 failed",
                            operation,
                            exc_info=True,
                        )
                    motion_enabled = False
                    controller_decel_stopped = False
                    shared.error_state.value = True
                    break
                applied_s = time.monotonic()
                last_cmd_seq = int(control_request["action_id"][0])
                last_cmd_created_s = (
                    int(control_request["created_monotonic_ns"][0]) / 1e9
                )
                last_cmd_received_s = received_s
                last_cmd_applied_s = applied_s
                last_cmd_queue_latency_s = max(0.0, received_s - last_cmd_created_s)
                last_cmd_apply_latency_s = max(0.0, applied_s - last_cmd_created_s)
                last_cmd_sdk_duration_s = max(0.0, applied_s - sdk_started_s)
                last_cmd_is_hold = True
                _recovery_counter.reset()
                if control_kind is ArmControlKind.DECELERATED_STOP:
                    logger.info(
                        "arm_loop: State-6 decelerated stop accepted "
                        "(reported_state=%d, action_id=%d, apply_latency=%.1fms)",
                        reported_stop_state,
                        last_cmd_seq,
                        1e3 * last_cmd_apply_latency_s,
                    )
                else:
                    logger.info(
                        "arm_loop: explicit Mode-6 resume accepted behind measured hold "
                        "(action_id=%d, apply_latency=%.1fms)",
                        last_cmd_seq,
                        1e3 * last_cmd_apply_latency_s,
                    )
            else:
                action = _take_next_current_arm_action(
                    shared.arm_action_q,
                    expected_run_generation=int(shared.run_generation.value),
                )

            # HOME sentinel — collision-validated path with request ID.
            if (
                isinstance(action, tuple)
                and len(action) == 2
                and action[0] == HOME_SENTINEL
            ):
                _request = action[1]
                if not isinstance(_request, HomeRequest):
                    logger.error("arm_loop: rejecting malformed HOME request")
                    continue
                logger.info(
                    "arm_loop: HOME sentinel — planned homing (%d validated milestones)",
                    len(_request.waypoints),
                )

                def _publish_homing_feedback(
                    qpos: np.ndarray,
                    qvel: np.ndarray,
                    tau: np.ndarray,
                    target: np.ndarray,
                ) -> None:
                    nonlocal last_state_source_ns
                    last_state_source_ns = time.monotonic_ns()
                    try:
                        eef_pos, eef_rot6d = _arm_fk.compute(qpos)
                    except Exception:
                        _fk_warn(
                            "arm_loop: Pinocchio FK failed during homing — publishing zero EEF"
                        )
                        eef_pos = np.zeros(3, dtype=np.float64)
                        eef_rot6d = np.zeros(6, dtype=np.float64)
                    try:
                        error_code = int(getattr(arm, "error_code", 0) or 0)
                    except Exception:
                        error_code = 0
                    _frame["qpos"][0] = qpos
                    _frame["qvel"][0] = qvel
                    _frame["tau"][0] = tau
                    _frame["eef_pos"][0] = eef_pos
                    _frame["eef_rot6d"][0] = eef_rot6d
                    _frame["error_code"][0] = error_code
                    _frame["connected"][0] = 1
                    _frame["mode"][0] = getattr(arm, "mode", 6)
                    _frame["tracking_err"][0] = float(np.max(np.abs(qpos - target)))
                    _frame["source_monotonic_ns"][0] = last_state_source_ns
                    _frame["publish_monotonic_ns"][0] = time.monotonic_ns()
                    _frame["state_valid"][0] = 1
                    _frame["timestamp"][0] = last_state_source_ns / 1e9
                    shared.arm_state_ring.write(_frame)

                _home_started_s = time.monotonic()
                _home_result = _planned_homing(
                    arm,
                    _request,
                    cfg,
                    shared=shared,
                    feedback_callback=_publish_homing_feedback,
                )
                try:
                    shared.arm_home_result_q.put(_home_result, timeout=0.2)
                except Exception:
                    logger.error(
                        "arm_loop: failed to publish HOME result", exc_info=True
                    )
                if _home_result.final_qpos.shape == ARM_JOINT_SHAPE and np.all(
                    np.isfinite(_home_result.final_qpos)
                ):
                    last_qpos = _home_result.final_qpos.copy()
                    last_target = last_qpos.copy()
                if _home_result.success:
                    motion_enabled = True
                    controller_decel_stopped = False
                    logger.info(
                        "arm_loop: HOME complete in %.2fs",
                        time.monotonic() - _home_started_s,
                    )
                elif shared.is_running.value:
                    logger.error("arm_loop: HOME failed — %s", _home_result.reason)
                    shared.error_state.value = True
                continue

            elif action is not None and not isinstance(action, tuple):
                if not worker_validate_arm(
                    action,
                    expected_run_generation=int(shared.run_generation.value),
                    now_monotonic_ns=time.monotonic_ns(),
                ):
                    logger.info(
                        "arm_loop: discarded malformed, stale-generation, or expired command"
                    )
                elif controller_decel_stopped or not motion_enabled:
                    # Only the priority RESUME request may leave State 5/6.
                    # Consuming an endpoint as an implicit resume can block the
                    # worker while producers fill the two-slot queue and can
                    # make firmware continue toward a superseded target.
                    logger.warning(
                        "arm_loop: discarded endpoint while explicit resume is pending"
                    )
                else:
                    target = np.asarray(action["qpos_cmd"][0], dtype=np.float64)
                    try:
                        target = wrap_nearest_equivalent(
                            target,
                            last_qpos,
                            cfg.joint_limit_lower,
                            cfg.joint_limit_upper,
                        )
                    except ValueError:
                        target = last_qpos.copy()
                    else:
                        last_target = target.copy()
                        _sdk_started_s = time.monotonic()
                        try:
                            code = arm.set_servo_angle(
                                angle=target,
                                is_radian=True,
                                speed=cfg.joint_max_speed_rad_per_s,
                                mvacc=cfg.joint_max_acc_rad_per_s2,
                                wait=False,
                            )
                        except Exception:
                            logger.error(
                                "arm_loop: set_servo_angle raised", exc_info=True
                            )
                            shared.error_state.value = True
                            break
                        if code == 0:
                            _recovery_counter.reset()
                            last_cmd_seq, last_cmd_created_s, last_cmd_is_hold = (
                                _parse_arm_action_metadata(action, _sdk_started_s)
                            )
                            last_cmd_received_s = time.monotonic()
                            last_cmd_applied_s = last_cmd_received_s
                            last_cmd_queue_latency_s = max(
                                0.0, last_cmd_received_s - last_cmd_created_s
                            )
                            last_cmd_apply_latency_s = max(
                                0.0, last_cmd_applied_s - last_cmd_created_s
                            )
                            last_cmd_sdk_duration_s = time.monotonic() - _sdk_started_s
                        else:
                            err_code = int(getattr(arm, "error_code", 0) or 0)
                            if err_code in cfg.collision_fault_errors:
                                _latch_collision_fault(shared, arm, err_code)
                                break
                            if err_code == 24:
                                now_s = time.monotonic()
                                if now_s - last_c24_s <= 2.0:
                                    logger.error(
                                        "arm_loop: second C24 inside 2s — latching fault"
                                    )
                                    shared.error_state.value = True
                                    break
                                last_c24_s = now_s
                                try:
                                    last_target = _recover_c24_measured_hold(arm, cfg)
                                    c24_recovered_this_tick = True
                                except Exception:
                                    logger.error(
                                        "arm_loop: C24 measured-hold recovery failed",
                                        exc_info=True,
                                    )
                                    shared.error_state.value = True
                                    break
                            else:
                                logger.error(
                                    "arm_loop: SDK failure code=%d err=%d",
                                    code,
                                    err_code,
                                )
                                shared.error_state.value = True
                                break
            elif action is not None:
                logger.error(
                    "arm_loop: rejecting malformed action queue item %r", action
                )

        arm_connected = True
        state_read_succeeded = False
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
            qpos, qvel, tau = _decode_joint_state_feedback(code, states)
            last_state_source_ns = time.monotonic_ns()
            last_qpos = qpos.copy()
            state_read_succeeded = True
        except Exception:
            logger.warning("arm_loop: get_joint_states failed", exc_info=True)
            qpos, qvel, tau = (
                last_qpos.copy(),
                np.zeros(ARM_JOINT_SHAPE),
                np.zeros(ARM_JOINT_SHAPE),
            )
            arm_connected = False
        state_read_fault = _update_state_read_watchdog(
            _state_error_counter,
            succeeded=state_read_succeeded,
        )

        # Pinocchio URDF-consistent FK (see note above).
        try:
            eef_pos, eef_rot6d = _arm_fk.compute(qpos)
        except Exception:
            _fk_warn("arm_loop: Pinocchio FK failed — publishing zero EEF")
            eef_pos = np.zeros(3, dtype=np.float64)
            eef_rot6d = np.zeros(6, dtype=np.float64)

        if controller_decel_stopped:
            # State 6 chooses the physical terminal point.  Following error to
            # the superseded Mode-6 endpoint is meaningless while braking;
            # continuously re-anchor diagnostics to fresh feedback instead.
            last_target = qpos.copy()
            tracking_err = 0.0
        else:
            tracking_err = float(np.max(np.abs(qpos - last_target)))

        if tracking_err > cfg.tracking_error_warn_rad:
            _tracking_err_count += 1
            if _tracking_err_count >= 3:
                _tracking_warn(
                    "arm_loop: tracking_err=%.3f_rad threshold=%.3f_rad",
                    tracking_err,
                    cfg.tracking_error_warn_rad,
                )
        else:
            _tracking_err_count = 0

        # arm.error_code is an SDK cached property (background report thread
        # ~every 200ms), not a per-access network call.
        try:
            error_code = arm.error_code
        except Exception:
            error_code = 0
            arm_connected = False

        if error_code in cfg.collision_fault_errors:
            _latch_collision_fault(shared, arm, error_code)
            break
        elif error_code in cfg.recoverable_errors and not c24_recovered_this_tick:
            # A C24 observed through the state stream follows the same bounded
            # recovery as a command return: discard the old endpoint, recover
            # Mode 6, read fresh feedback, then issue exactly one measured hold.
            now_s = time.monotonic()
            if now_s - last_c24_s <= 2.0:
                logger.error("arm_loop: repeated C24 inside 2s — latching fault")
                shared.error_state.value = True
                break
            last_c24_s = now_s
            try:
                last_target = _recover_c24_measured_hold(
                    arm, cfg, operation_prefix="state C24"
                )
            except Exception:
                logger.error(
                    "arm_loop: state C24 measured-hold recovery failed", exc_info=True
                )
                shared.error_state.value = True
                break
        elif error_code != 0 and not (c24_recovered_this_tick and error_code == 24):
            shared.error_state.value = True
            break

        # Publish state
        _frame["qpos"][0] = qpos
        _frame["qvel"][0] = qvel
        _frame["tau"][0] = tau
        _frame["eef_pos"][0] = eef_pos
        _frame["eef_rot6d"][0] = eef_rot6d
        _frame["error_code"][0] = int(error_code)
        _frame["connected"][0] = 1 if arm_connected else 0
        _frame["mode"][0] = getattr(arm, "mode", 6)
        _frame["tracking_err"][0] = tracking_err
        _frame["last_cmd_seq"][0] = last_cmd_seq
        _frame["last_cmd_created_s"][0] = last_cmd_created_s
        _frame["last_cmd_received_s"][0] = last_cmd_received_s
        _frame["last_cmd_applied_s"][0] = last_cmd_applied_s
        _frame["last_cmd_queue_latency_s"][0] = last_cmd_queue_latency_s
        _frame["last_cmd_apply_latency_s"][0] = last_cmd_apply_latency_s
        _frame["last_cmd_sdk_duration_s"][0] = last_cmd_sdk_duration_s
        _frame["last_cmd_is_hold"][0] = int(last_cmd_is_hold)
        _frame["source_monotonic_ns"][0] = last_state_source_ns
        _frame["publish_monotonic_ns"][0] = time.monotonic_ns()
        _frame["state_valid"][0] = int(arm_connected)
        _frame["timestamp"][0] = last_state_source_ns / 1e9
        shared.arm_state_ring.write(_frame)

        if state_read_fault:
            terminal_feedback_fault = True
            shared.error_state.value = True
            logger.error(
                "arm_loop: %d consecutive feedback-read failures — latching global fault",
                _state_error_counter.count,
            )
            publish_component_status(
                shared,
                "arm",
                ComponentPhase.FAULT,
                fault_code=FaultCode.DEVICE_IO,
                detail="persistent get_joint_states failure",
            )
            break

        # Rate limit
        limiter.wait()
    # Cleanup
    stopped_cleanly = False
    try:
        _require_sdk_ok("cleanup set_state(4)", arm.set_state(4))
        _wait_live_status(arm, expected_state=4)
        arm.disconnect()
        stopped_cleanly = True
    except Exception:
        logger.warning("arm_loop: cleanup failed", exc_info=True)
        shared.error_state.value = True
    if terminal_feedback_fault:
        publish_component_status(
            shared,
            "arm",
            ComponentPhase.FAULT,
            fault_code=FaultCode.DEVICE_IO,
            detail="persistent get_joint_states failure",
        )
    elif stopped_cleanly:
        publish_component_status(shared, "arm", ComponentPhase.STOPPED)
    else:
        publish_component_status(
            shared,
            "arm",
            ComponentPhase.FAULT,
            fault_code=FaultCode.DEVICE_IO,
            detail="state-4 cleanup failed",
        )
    logger.info("arm_loop: exited")


def _disconnect_arm(arm: Any) -> None:
    """Disconnect arm and report best-effort cleanup failures."""
    try:
        arm.disconnect()
    except Exception:
        logger.warning("arm disconnect failed during cleanup", exc_info=True)


def _result_impl(
    request: HomeRequest,
    success: bool,
    reason: str,
    qpos: np.ndarray,
) -> HomeResult:
    return HomeResult(
        request_id=request.request_id,
        success=success,
        reason=reason,
        final_qpos=np.asarray(qpos, dtype=np.float64).copy(),
        completed_at_s=time.monotonic(),
    )


def _shared_abort_reason_impl(shared: Any) -> str | None:
    if shared is None:
        return None
    if not shared.is_running.value:
        return "shutdown requested"
    if shared.estop_request.value:
        return "e-stop requested"
    if shared.error_state.value:
        return "sticky error_state set during homing"
    if shared.safety_state.value == SafetyState.FAULT:
        return "FAULT during homing"
    return None


def _confirm_home_dwell_impl(
    arm: Any,
    request: HomeRequest,
    cfg: ArmLoopConfig,
    home_qpos: np.ndarray,
    shared: Any,
    current: np.ndarray,
    current_qvel: np.ndarray,
    failure_reason: str,
) -> HomeResult:
    if (
        float(np.max(np.abs(current - home_qpos))) > cfg.homing_convergence_rad
        or float(np.max(np.abs(current_qvel)))
        > cfg.homing_velocity_convergence_rad_s
    ):
        return _result_impl(request, False, failure_reason, current)
    stable_since = time.monotonic()
    while time.monotonic() - stable_since < cfg.homing_dwell_s:
        abort_reason = _shared_abort_reason_impl(shared)
        if abort_reason is not None:
            return _result_impl(request, False, abort_reason, current)
        time.sleep(min(cfg.homing_step_interval_s, cfg.homing_dwell_s))
        code, states = arm.get_joint_states(is_radian=True, num=3)
        if code != 0 or len(states) <= 1:
            return _result_impl(request, 
                False, "state/qvel unavailable during home dwell", current
            )
        current = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
        current_qvel = np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
        if (
            current.shape != ARM_JOINT_SHAPE
            or current_qvel.shape != ARM_JOINT_SHAPE
            or not np.all(np.isfinite(current))
            or not np.all(np.isfinite(current_qvel))
            or float(np.max(np.abs(current - home_qpos)))
            > cfg.homing_convergence_rad
            or float(np.max(np.abs(current_qvel)))
            > cfg.homing_velocity_convergence_rad_s
        ):
            return _result_impl(request, 
                False, "home dwell interrupted by position/velocity", current
            )
    return _result_impl(request, True, "already at canonical home and settled", current)


def _execute_mode0_milestones_impl(
    arm: Any,
    request: HomeRequest,
    cfg: ArmLoopConfig,
    home_qpos: np.ndarray,
    shared: Any,
    feedback_callback: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None,
    execution_targets: np.ndarray,
    current: np.ndarray,
) -> tuple[HomeResult, np.ndarray]:
    _overall_deadline = time.monotonic() + request.execution_timeout_s
    _milestone_tol = min(cfg.homing_convergence_rad, np.deg2rad(0.5))

    for _target_index, _target in enumerate(execution_targets, start=1):
        if shared is not None:
            _abort_reason = _shared_abort_reason_impl(shared)
            if _abort_reason is not None:
                return _result_impl(request, False, _abort_reason, current), current
            shared.arm_heartbeat_s.value = time.monotonic()
        if time.monotonic() >= _overall_deadline:
            return _result_impl(request, 
                False,
                f"overall timeout before milestone {_target_index}/{len(execution_targets)}",
                current,
            ), current

        _segment_start = current.copy()
        _segment_started_s = time.monotonic()
        try:
            _code = arm.set_servo_angle(
                angle=_target,
                is_radian=True,
                speed=cfg.homing_max_speed_rad_per_s,
                mvacc=cfg.joint_max_acc_rad_per_s2,
                wait=False,
                radius=None,
            )
        except Exception:
            logger.warning("_planned_homing: milestone send failed", exc_info=True)
            return _result_impl(request, False, f"milestone {_target_index} send raised", current), current
        if _code != 0:
            return _result_impl(request, 
                False,
                f"milestone {_target_index} rejected (SDK code={_code})",
                current,
            ), current

        _segment_timeout_s = _estimate_homing_segment_timeout_s(
            _segment_start, _target, cfg
        )
        _segment_deadline = min(
            _overall_deadline, _segment_started_s + _segment_timeout_s
        )
        _stable_since_s: float | None = None
        while time.monotonic() < _segment_deadline:
            if shared is not None:
                _abort_reason = _shared_abort_reason_impl(shared)
                if _abort_reason is not None:
                    return _result_impl(request, False, _abort_reason, current), current
                shared.arm_heartbeat_s.value = time.monotonic()
            try:
                _state_code, _states = arm.get_joint_states(is_radian=True, num=3)
            except Exception:
                logger.warning(
                    "_planned_homing: milestone state read raised", exc_info=True
                )
                return _result_impl(request, 
                    False,
                    f"state read raised at milestone {_target_index}",
                    current,
                ), current
            if _state_code != 0 or len(_states) == 0:
                return _result_impl(request, 
                    False,
                    f"state read failed at milestone {_target_index} (code={_state_code})",
                    current,
                ), current
            current = np.asarray(_states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
            if current.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(current)):
                return _result_impl(request, 
                    False, f"invalid state at milestone {_target_index}", current
                ), current
            if len(_states) <= 1:
                return _result_impl(request, 
                    False, f"qvel unavailable at milestone {_target_index}", current
                ), current
            qvel = np.asarray(_states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
            tau = (
                np.asarray(_states[2], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
                if len(_states) > 2
                else np.zeros(ARM_JOINT_SHAPE)
            )
            try:
                _controller_error = int(getattr(arm, "error_code", 0) or 0)
            except Exception:
                _controller_error = 0
            if _controller_error != 0:
                return _result_impl(request, 
                    False,
                    f"controller error C{_controller_error} at milestone {_target_index}",
                    current,
                ), current
            if feedback_callback is not None:
                try:
                    feedback_callback(
                        current.copy(), qvel.copy(), tau.copy(), _target.copy()
                    )
                except Exception:
                    logger.warning(
                        "_planned_homing: feedback publication failed",
                        exc_info=True,
                    )
            if (
                float(np.max(np.abs(current - _target))) <= _milestone_tol
                and float(np.max(np.abs(qvel)))
                <= cfg.homing_velocity_convergence_rad_s
            ):
                if _stable_since_s is None:
                    _stable_since_s = time.monotonic()
                if time.monotonic() - _stable_since_s >= cfg.homing_dwell_s:
                    break
            else:
                _stable_since_s = None
            time.sleep(cfg.homing_step_interval_s)
        else:
            _error = np.abs(current - _target)
            _joint = int(np.argmax(_error))
            _elapsed_s = time.monotonic() - _segment_started_s
            if time.monotonic() >= _overall_deadline:
                _timeout_kind = "overall timeout"
            else:
                _timeout_kind = "convergence timeout"
            return _result_impl(request, 
                False,
                f"{_timeout_kind} at milestone {_target_index}/{len(execution_targets)} "
                f"after {_elapsed_s:.2f}s (J{_joint + 1} error={np.rad2deg(_error[_joint]):.2f}deg)",
                current,
            ), current

    _final_error = float(np.max(np.abs(current - home_qpos)))
    if _final_error > cfg.homing_convergence_rad:
        return _result_impl(request, 
            False, f"final error {np.rad2deg(_final_error):.2f}deg", current
        ), current
    return _result_impl(request, True, "canonical home reached", current), current


def _planned_homing(
    arm: Any,
    request: HomeRequest,
    cfg: ArmLoopConfig | None = None,
    *,
    shared: Any = None,
    feedback_callback: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None
    ) = None,
) -> HomeResult:
    """Execute collision-validated milestones with the firmware joint planner.

    The caller densely checks every joint-space segment for collision, but only
    the sparse segment endpoints cross the process boundary.  Homing temporarily
    enters Mode 0 and uses unblended ``MoveJoint`` commands so the controller,
    rather than this process, owns the point-to-point trajectory.  Normal Mode 6
    teleoperation is restored before returning from healthy paths; E-stop,
    shutdown, and controller-fault paths stop instead.  Completion is based
    only on fresh encoder feedback; no state is fabricated on SDK read failure.
    """
    _cfg = cfg or ArmLoopConfig()

    def _result(success: bool, reason: str, qpos: np.ndarray) -> HomeResult:
        return _result_impl(request, success, reason, qpos)

    def _shared_abort_reason() -> str | None:
        return _shared_abort_reason_impl(shared)

    waypoints = np.asarray(request.waypoints, dtype=np.float64)
    home_qpos = np.asarray(request.final_qpos, dtype=np.float64)
    if (
        not isinstance(request.request_id, (int, np.integer))
        or int(request.request_id) <= 0
    ):
        return _result(False, "invalid request_id", np.full(ARM_JOINT_SHAPE, np.nan))
    if (
        waypoints.ndim != 2
        or waypoints.shape[1:] != ARM_JOINT_SHAPE
        or not np.all(np.isfinite(waypoints))
    ):
        return _result(
            False, "invalid waypoint array", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    if home_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(home_qpos)):
        return _result(False, "invalid final_qpos", np.full(ARM_JOINT_SHAPE, np.nan))
    if (
        not np.isfinite(request.execution_timeout_s)
        or request.execution_timeout_s <= 0.0
    ):
        return _result(
            False, "invalid execution timeout", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    _lower = np.asarray(_cfg.joint_limit_lower, dtype=np.float64)
    _upper = np.asarray(_cfg.joint_limit_upper, dtype=np.float64)
    if len(waypoints) > 0 and not np.all((waypoints >= _lower) & (waypoints <= _upper)):
        return _result(
            False, "waypoint violates joint limits", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    if len(waypoints) > 0 and float(np.max(np.abs(waypoints[-1] - home_qpos))) > 1e-6:
        return _result(
            False,
            "final milestone does not match canonical home",
            np.full(ARM_JOINT_SHAPE, np.nan),
        )

    try:
        code, states = arm.get_joint_states(is_radian=True, num=3)
        if code == 0 and len(states) > 0:
            current = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
            current_qvel = (
                np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
                if len(states) > 1
                else np.full(ARM_JOINT_SHAPE, np.inf)
            )
        else:
            return _result(
                False,
                f"initial state read failed (code={code})",
                np.full(ARM_JOINT_SHAPE, np.nan),
            )
    except Exception:
        logger.warning("_planned_homing: initial state read raised", exc_info=True)
        return _result(
            False, "initial state read raised", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    if current.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(current)):
        return _result(
            False, "initial state is invalid", np.full(ARM_JOINT_SHAPE, np.nan)
        )

    def _confirm_home_dwell(failure_reason: str) -> HomeResult:
        return _confirm_home_dwell_impl(
            arm, request, _cfg, home_qpos, shared, current, current_qvel, failure_reason
        )

    if len(waypoints) == 0:
        return _confirm_home_dwell(
            "empty path while away from stationary canonical home"
        )
    if float(np.max(np.abs(current - waypoints[0]))) > _cfg.homing_convergence_rad:
        return _result(
            False, "current state moved too far from planned path start", current
        )

    _execution_targets = waypoints[1:]
    if len(_execution_targets) == 0:
        return _confirm_home_dwell(
            "single-point path is not at stationary canonical home"
        )
    _preflight_abort = _shared_abort_reason()
    if _preflight_abort is not None:
        return _result(False, _preflight_abort, current)

    def _execute_mode0_milestones() -> HomeResult:
        nonlocal current
        _home_result, current = _execute_mode0_milestones_impl(
            arm, request, _cfg, home_qpos, shared, feedback_callback,
            _execution_targets, current,
        )
        return _home_result

    # Mode 6 is designed for continuously changing online targets and its
    # per-joint velocity profiles need not be synchronous.  A planned homing
    # path instead uses Mode 0 MoveJoint.  Explicitly restore Mode 6 after
    # healthy entry/execution failures so the worker never silently changes
    # semantics; global-stop and controller-fault paths remain stopped.
    _mode_switch_attempted = False
    try:
        logger.info(
            "arm_loop: homing entering Mode 0 MoveJoint (%d motion milestones, speed=%.1fdeg/s)",
            len(_execution_targets),
            np.rad2deg(_cfg.homing_max_speed_rad_per_s),
        )
        _mode_switch_attempted = True
        _require_sdk_ok("set_mode(0)", arm.set_mode(0))
        _require_sdk_ok("set_state(0) after Mode 0", arm.set_state(0))
    except Exception as exc:
        logger.error("_planned_homing: failed to enter Mode 0", exc_info=True)
        _home_result = _result(False, f"Mode 0 entry failed: {exc}", current)
    else:
        _home_result = _execute_mode0_milestones()

    _restore_error: Exception | None = None
    _post_homing_abort = _shared_abort_reason()
    try:
        _controller_error_after_home = int(getattr(arm, "error_code", 0) or 0)
    except Exception:
        _controller_error_after_home = 0
    _restore_mode6 = _post_homing_abort is None and _controller_error_after_home == 0
    if _mode_switch_attempted and _restore_mode6:
        try:
            _require_sdk_ok("restore set_mode(6)", arm.set_mode(6))
            _require_sdk_ok("restore set_state(0)", arm.set_state(0))
        except Exception as exc:
            _restore_error = exc
            logger.error("_planned_homing: failed to restore Mode 6", exc_info=True)
    elif _mode_switch_attempted:
        _stop_reason = (
            _post_homing_abort or f"controller error C{_controller_error_after_home}"
        )
        try:
            _require_sdk_ok("stop after interrupted homing", arm.set_state(4))
        except Exception as exc:
            _restore_error = exc
            logger.error(
                "_planned_homing: failed to stop after interrupted homing",
                exc_info=True,
            )
        if _home_result.success:
            _home_result = _result(
                False, f"homing interrupted after convergence: {_stop_reason}", current
            )
    if _restore_error is not None:
        _operation = "Mode 6 restore" if _restore_mode6 else "safe stop"
        return _result(
            False,
            f"{_home_result.reason}; {_operation} failed: {_restore_error}",
            current,
        )
    if _restore_mode6:
        logger.info("arm_loop: homing restored Mode 6")
    return _home_result


def _estimate_homing_segment_timeout_s(
    start: np.ndarray, target: np.ndarray, cfg: ArmLoopConfig
) -> float:
    """Deadline for one firmware-planned milestone, including settle time."""
    delta_rad = float(np.max(np.abs(np.asarray(target) - np.asarray(start))))
    nominal_s = delta_rad / max(cfg.homing_max_speed_rad_per_s, 1e-6)
    return max(
        cfg.homing_target_timeout_s, 2.0 * nominal_s + cfg.homing_target_timeout_s
    )
