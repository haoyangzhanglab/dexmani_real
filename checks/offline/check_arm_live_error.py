"""Live controller error/state reads fail closed.

``_read_live_error_code`` must return the live register on success and raise on
any failed read (never fall back to the cached value).  ``read_live_state_and_error``
must likewise raise if either synchronous read fails — the startup path relies on
this fail-closed behaviour instead of the former connect-recovery wrapper.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeArm

from dexmani_real.robot.arm_sdk import _read_live_error_code, read_live_state_and_error


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

    # read_live_state_and_error returns both registers on success and fails
    # closed (raises) when either synchronous read fails.
    live = read_live_state_and_error(FakeArm(state=4, error_code=24, mode=6))
    assert live.state == 4
    assert live.error_code == 24
    assert live.warn_code == 0

    try:
        read_live_state_and_error(arm_raise)
    except RuntimeError:
        pass
    else:
        raise AssertionError("read_live_state_and_error must raise on a failed read")

    arm_bad_state = FakeArm()
    arm_bad_state.fail("get_state", -1)
    try:
        read_live_state_and_error(arm_bad_state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("read_live_state_and_error must raise on get_state failure")

    print("check_arm_live_error: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
