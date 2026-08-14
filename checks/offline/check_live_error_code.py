"""Offline check: ``_read_live_error_code`` reads live, never falls back.

The live error code is the source of truth for control decisions (setter
failure classification and post-homing Mode 6 restore).  This check proves the
helper returns ``values[0]`` on a clean read and raises when the live getter
fails — it must not silently fall back to the cached ``arm.error_code``.

Run from the repo root:
    conda run -n real_robot python checks/offline/check_live_error_code.py
"""

from __future__ import annotations

from dexmani_real.robot.arm_loop import _read_live_error_code


class FakeArm:
    def __init__(self, code: int, values: list[int]) -> None:
        self._code = code
        self._values = values

    def get_err_warn_code(self):
        return self._code, self._values


def main() -> int:
    assert _read_live_error_code(FakeArm(0, [24, 0])) == 24
    assert _read_live_error_code(FakeArm(0, [31, 0])) == 31
    assert _read_live_error_code(FakeArm(0, [0, 0])) == 0

    # A failed live getter must raise (no cached fallback).
    try:
        _read_live_error_code(FakeArm(1, [0, 0]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on failed get_err_warn_code")

    print("OK: live error read succeeds and fails closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
