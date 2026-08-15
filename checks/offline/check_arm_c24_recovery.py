"""A1: C24 measured-hold recovery ordering and motion-enable fail-closed.

Asserts the recovery sequence clears controller errors before re-enabling
motion, re-enters Mode 6/State 2 before reading fresh joints, and issues
exactly one measured hold — and that a failed ``motion_enable`` short-circuits
before any servo command.
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeArm

from dexmani_real.robot.arm_loop import _recover_c24_measured_hold
from dexmani_real.robot.arm_sdk import ArmLoopConfig


def _assert_order(calls: list[str], *names: str) -> None:
    order = [(name, calls.index(name) if name in calls else -1) for name in names]
    positions = [pos for _, pos in order]
    if any(pos < 0 for pos in positions):
        raise AssertionError(f"missing expected call among {names}: {order}")
    if positions != sorted(positions):
        raise AssertionError(f"call order violated: {order}")


def main() -> int:
    cfg = ArmLoopConfig()
    qpos = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float64)

    # ── Happy path: correct order, one measured hold ────────────────────
    arm = FakeArm(joint_state=qpos, state=4, mode=6, error_code=0)
    held = _recover_c24_measured_hold(arm, cfg, operation_prefix="C24")
    np.testing.assert_allclose(held, qpos)
    _assert_order(
        arm.call_order(),
        "clean_error",
        "clean_warn",
        "motion_enable",
        "set_mode",
        "set_state",
        "get_joint_states",
        "set_servo_angle",
    )
    # motion_enable must carry the enable flag.
    for name, args, _ in arm.calls:
        if name == "motion_enable":
            assert args[0] is True, f"motion_enable called with {args}"
    # Exactly one measured hold is issued, at the freshly read joints.
    servo_calls = [args for name, args, _ in arm.calls if name == "set_servo_angle"]
    assert len(servo_calls) == 1, f"expected exactly one servo, got {len(servo_calls)}"
    np.testing.assert_allclose(np.asarray(servo_calls[0][0]), qpos)

    # ── Fail-closed: motion_enable != 0 must prevent any servo command ──
    arm2 = FakeArm(joint_state=qpos, state=4, mode=6, error_code=0)
    arm2.fail("motion_enable", 1)
    try:
        _recover_c24_measured_hold(arm2, cfg, operation_prefix="C24")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when motion_enable fails")
    assert "set_servo_angle" not in arm2.call_order(), (
        "servo must not be issued after a failed motion_enable"
    )

    print("check_arm_c24_recovery: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
