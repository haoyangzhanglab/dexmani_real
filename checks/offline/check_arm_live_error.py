"""A2: live controller error reads fail closed.

``_read_live_error_code`` must return the live register on success and raise on
any failed read (never fall back to the cached value).  The connect-recovery
wrapper ``_read_live_error_or_fail`` must map that failure to a non-ready
status (1) instead of declaring the controller ready.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeArm

from dexmani_real.robot.arm_loop import _read_live_error_code, _read_live_error_or_fail


def main() -> int:
    # Success path returns the live register value.
    assert _read_live_error_code(FakeArm(error_code=0)) == 0
    assert _read_live_error_code(FakeArm(error_code=24)) == 24

    # A raising live read must not silently fall back to the cache.
    arm_raise = FakeArm()
    arm_raise.raise_on("get_err_warn_code", RuntimeError("vendor read refused"))
    try:
        _read_live_error_code(arm_raise)
    except RuntimeError:
        pass
    else:
        raise AssertionError("_read_live_error_code must raise on a failed read")

    # A non-zero SDK return code must also raise (via _require_sdk_ok).
    arm_bad_code = FakeArm()
    arm_bad_code.fail("get_err_warn_code", -1)
    try:
        _read_live_error_code(arm_bad_code)
    except RuntimeError:
        pass
    else:
        raise AssertionError("_read_live_error_code must raise on a non-zero code")

    # Connect-recovery wrapper fails closed to 1 on any failed read.
    assert _read_live_error_or_fail(FakeArm(error_code=0)) == 0
    assert _read_live_error_or_fail(FakeArm(error_code=24)) == 24
    assert _read_live_error_or_fail(arm_raise) == 1
    assert _read_live_error_or_fail(arm_bad_code) == 1

    print("check_arm_live_error: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
