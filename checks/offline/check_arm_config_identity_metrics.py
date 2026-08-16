"""Phase C: config bounds, device identity, and command-latency/mode-drift.

Covers doc §11.4 — the Mode 6 speed/mvacc config is validated in radians
against the SDK command-path clamps; the reported device identity is checked
with an integer-tuple firmware compare and no model guess; command latency is
computed from three monotonic samples with a controllable clock; and the
cached-mode drift window only faults after a bounded wall-clock mismatch.
Runs against the pure helpers, not the SDK.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import make_arm_state_frame

from dexmani_real.config.defaults import ArmParams, SafetyParams, arm, safety
from dexmani_real.robot.arm_loop import (
    _advance_mode_drift,
    _compute_command_latency,
    _decode_joint_state_feedback,
    _parse_arm_action_metadata,
)
from dexmani_real.robot.arm_sdk import (
    ArmLoopConfig,
    validate_command_dynamics_intersection,
    validate_device_identity,
)
from dexmani_real.shm.shared_storage import read_arm_state_dict
from dexmani_real.utils.schema import ARM_COMMAND_DTYPE


# ── C1: speed/mvacc config bounds + dynamics intersection ──


def _test_speed_acc_config_bounds() -> None:
    # Defaults resolve inside the SDK command-path clamps.
    ArmParams()  # 120 deg/s → 2.094 rad/s; 900 deg/s² → 15.7 rad/s²

    def _raises(**kwargs: object) -> None:
        try:
            ArmParams(**kwargs)
        except ValueError:
            return
        raise AssertionError(f"ArmParams({kwargs}) must reject out-of-range dynamics")

    _raises(max_joint_velocity_deg_per_s=200.0)  # 3.49 rad/s > π
    _raises(max_joint_velocity_deg_per_s=0.001)  # 1.7e-5 rad/s < 0.0001
    _raises(max_joint_acceleration_deg_per_s2=2000.0)  # 34.9 rad/s² > 20
    _raises(max_joint_acceleration_deg_per_s2=0.1)  # 1.7e-3 rad/s² < 0.01


def _test_dynamics_intersection() -> None:
    # Within both clamps and the device report → accepted.
    assert (
        validate_command_dynamics_intersection(
            config_speed_rad_per_s=2.0,
            config_acc_rad_per_s2=15.0,
            device_speed_limits=[np.pi] * 7,
            device_acc_limits=[20.0] * 7,
        )
        is None
    )
    # Above the SDK hard clamp → rejected.
    assert (
        validate_command_dynamics_intersection(
            config_speed_rad_per_s=4.0,
            config_acc_rad_per_s2=15.0,
            device_speed_limits=None,
            device_acc_limits=None,
        )
        is not None
    )
    # Below the lower clamp → rejected.
    assert (
        validate_command_dynamics_intersection(
            config_speed_rad_per_s=0.00001,
            config_acc_rad_per_s2=15.0,
            device_speed_limits=None,
            device_acc_limits=None,
        )
        is not None
    )
    # Inside the SDK clamp but above the reported device limit → rejected.
    assert (
        validate_command_dynamics_intersection(
            config_speed_rad_per_s=2.0,
            config_acc_rad_per_s2=15.0,
            device_speed_limits=[1.0] * 7,  # most restrictive joint is 1.0 rad/s
            device_acc_limits=[20.0] * 7,
        )
        is not None
    )
    # A missing device report falls back to the SDK hard clamp alone.
    assert (
        validate_command_dynamics_intersection(
            config_speed_rad_per_s=np.pi,
            config_acc_rad_per_s2=20.0,
            device_speed_limits=None,
            device_acc_limits=None,
        )
        is None
    )


# ── C2: device identity validation ──


def _test_device_identity() -> None:
    def _ok(**kwargs: object) -> None:
        assert validate_device_identity(**kwargs) is None, kwargs

    def _bad(**kwargs: object) -> None:
        assert validate_device_identity(**kwargs) is not None, kwargs

    base = dict(
        axis=7,
        device_type="xArm7",
        serial_number="SN-123",
        firmware=(1, 18, 4),
        expected_axis=7,
        expected_serial=None,
        min_firmware=None,
        device_profile=None,
    )
    _ok(**base)

    _bad(**{**base, "axis": 6})
    _bad(**{**base, "expected_serial": "SN-999"})
    _bad(**{**base, "min_firmware": (1, 19, 0)})  # (1,18,4) < (1,19,0)
    _bad(**{**base, "device_profile": "xArm6"})

    # Integer-tuple compare, not string: "1.18.4" vs "1.9" would order wrongly.
    _ok(**{**base, "firmware": (1, 18, 4), "min_firmware": (1, 18, 4)})  # equal ok
    _bad(**{**base, "firmware": (1, 9, 0), "min_firmware": (1, 18, 0)})
    # A missing device profile performs no type check (no model guess).
    _ok(**{**base, "device_profile": None, "device_type": "xArm7"})


def _field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _test_identity_config_fields() -> None:
    # New identity fields exist on both config layers (defaults + loop config).
    for cls in (ArmParams, ArmLoopConfig):
        names = _field_names(cls)
        for field in ("expected_axis", "device_profile", "serial_number", "min_firmware"):
            assert field in names, f"{cls.__name__}.{field}"

    # The renamed health threshold replaces the old recovery name everywhere it
    # lives (SafetyParams + the loop config that copies it), and the identity
    # fields resolve from the defaults through from_runtime.
    for cls in (SafetyParams, ArmLoopConfig):
        names = _field_names(cls)
        assert "max_consecutive_recoveries" not in names, cls.__name__
        assert "max_consecutive_arm_health_failures" in names, cls.__name__
    assert SafetyParams().max_consecutive_arm_health_failures > 0

    cfg = ArmLoopConfig.from_runtime(
        SimpleNamespace(arm=arm, safety=safety)
    )
    assert cfg.expected_axis == 7
    assert cfg.device_profile is None
    assert cfg.serial_number is None
    assert cfg.min_firmware is None
    assert (
        cfg.max_consecutive_arm_health_failures
        == safety.max_consecutive_arm_health_failures
    )


# ── C3: command latency (controllable clock) ──


def _test_command_latency() -> None:
    queue, apply_, sdk = _compute_command_latency(
        created_s=10.0, received_s=10.5, applied_s=10.6, sdk_started_s=10.4
    )
    assert abs(queue - 0.5) < 1e-12
    assert abs(apply_ - 0.6) < 1e-12
    assert abs(sdk - 0.2) < 1e-12

    # Never negative (a stale created timestamp clamps to zero).
    queue, _, _ = _compute_command_latency(
        created_s=11.0, received_s=10.5, applied_s=10.6, sdk_started_s=10.4
    )
    assert queue == 0.0


def _test_parse_metadata_clamp() -> None:
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    frame["action_id"][0] = 5
    frame["created_monotonic_ns"][0] = int(20.0 * 1e9)  # in the future
    frame["is_hold"][0] = 1
    seq, created, is_hold = _parse_arm_action_metadata(frame, 10.0)
    assert seq == 5
    assert created == 10.0  # clamped to the received timestamp
    assert is_hold is True


# ── C4: mode drift window (controllable clock) ──


def _test_mode_drift() -> None:
    def _adv(*, since: float | None, now: float, healthy: bool = True) -> tuple[float | None, bool]:
        return _advance_mode_drift(
            monitoring=True,
            report_mode=0,
            expected_mode=6,
            feedback_healthy=healthy,
            mismatch_since_s=since,
            now_s=now,
            timeout_s=1.0,
        )

    # Expected mode or not-monitoring resets/stays silent.
    since, fault = _advance_mode_drift(
        monitoring=False, report_mode=0, expected_mode=6,
        feedback_healthy=True, mismatch_since_s=3.0, now_s=10.0, timeout_s=1.0,
    )
    assert since is None and not fault
    since, fault = _advance_mode_drift(
        monitoring=True, report_mode=6, expected_mode=6,
        feedback_healthy=True, mismatch_since_s=3.0, now_s=10.0, timeout_s=1.0,
    )
    assert since is None and not fault

    # First mismatch starts the window without faulting.
    since, fault = _adv(since=None, now=0.0)
    assert since == 0.0 and not fault
    # Persistent but within the window → still no fault.
    since, fault = _adv(since=0.0, now=0.5)
    assert since == 0.0 and not fault
    # Past the window while feedback is healthy → fault.
    since, fault = _adv(since=0.0, now=1.0)
    assert since == 0.0 and fault
    # Past the window but feedback unhealthy → no fault (cache cannot be trusted).
    since, fault = _adv(since=0.0, now=1.0, healthy=False)
    assert since == 0.0 and not fault


# ── C6: read_arm_state_dict exposes mode ──


def _test_arm_state_dict_mode() -> None:
    frame = make_arm_state_frame(np.zeros(7, dtype=np.float64))
    frame["mode"][0] = 6

    class _Ring:
        def read_latest(self) -> tuple[object, int, int]:
            return frame, 0, 0

    shared = SimpleNamespace(arm_state_ring=_Ring())
    state = read_arm_state_dict(shared)  # type: ignore[arg-type]
    assert state is not None
    assert state["mode"] == 6


def _test_joint_state_feedback_decode() -> None:
    qpos = np.arange(7, dtype=np.float64) + 0.1
    qvel = np.arange(7, dtype=np.float64) + 10.1
    effort = np.arange(7, dtype=np.float64) + 20.1
    decoded = _decode_joint_state_feedback(0, [qpos, qvel, effort])
    for actual, expected in zip(decoded, (qpos, qvel, effort)):
        np.testing.assert_array_equal(actual, expected)
        assert actual is not expected, "worker boundary must copy SDK arrays"

    malformed = (
        (1, [qpos, qvel, effort]),
        (0, [qpos, qvel]),
        (0, [qpos[:6], qvel, effort]),
        (0, [qpos, qvel, np.full(7, np.nan)]),
    )
    for code, states in malformed:
        try:
            _decode_joint_state_feedback(code, states)
        except RuntimeError:
            pass
        else:
            raise AssertionError("incomplete/non-finite position, velocity, or effort must fail")


def _test_startup_feedback_source_structural() -> None:
    import dexmani_real.robot.arm_loop as arm_loop_mod

    source = Path(arm_loop_mod.__file__).read_text()
    startup = source[source.index("_STATE_READ_MAX_RETRIES = 10") : source.index("last_target = last_qpos.copy()")]
    assert "get_joint_states(is_radian=True, num=3)" in startup
    assert "_decode_joint_state_feedback" in startup
    initial_publish = source[
        source.index("# Publish initial state BEFORE arm_ready") : source.index('shared.set_ready("arm")')
    ]
    assert '_frame["qvel"][0] = initial_qvel' in initial_publish
    assert '_frame["tau"][0] = initial_effort' in initial_publish


def main() -> int:
    _test_speed_acc_config_bounds()
    _test_dynamics_intersection()
    _test_device_identity()
    _test_identity_config_fields()
    _test_command_latency()
    _test_parse_metadata_clamp()
    _test_mode_drift()
    _test_arm_state_dict_mode()
    _test_joint_state_feedback_decode()
    _test_startup_feedback_source_structural()
    print("check_arm_config_identity_metrics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
