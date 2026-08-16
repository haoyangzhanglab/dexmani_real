"""Phase B: single sticky fault path.

Covers doc §11.3 — C22/C24/C31 all enter the same latch→stop path; the sticky
``error_state`` is written before the stop attempt; the worker never calls
``clean_error``; a non-zero setter API code with a zero live controller error
still faults; and the live error read fails closed.  Runs against ``FakeArm``
and a lightweight ``SimpleNamespace`` shared-storage stand-in.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeArm

from dexmani_real.config.defaults import ArmParams
from dexmani_real.robot.arm_loop import latch_arm_fault
from dexmani_real.robot.arm_sdk import ArmLoopConfig, _read_live_error_code


def _shared() -> SimpleNamespace:
    return SimpleNamespace(error_state=SimpleNamespace(value=False))


class _ProbeArm(FakeArm):
    """FakeArm that records whether the sticky fault is set before set_state(4)."""

    def __init__(self, shared: SimpleNamespace, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._shared = shared
        self.sticky_before_stop: bool | None = None

    def set_state(self, state: int) -> int:
        if state == 4:
            self.sticky_before_stop = bool(self._shared.error_state.value)
        return super().set_state(state)


def _test_single_fault_path() -> None:
    # C22/C24/C31 all latch the same sticky fault and never clear an error.
    for error_code in (22, 24, 31):
        sh = _shared()
        arm = _ProbeArm(sh, state=4, mode=6, error_code=error_code)
        latch_arm_fault(sh, arm, "boom", controller_error=error_code)
        assert sh.error_state.value is True, f"C{error_code} must latch"
        assert "clean_error" not in arm.call_order(), "worker must never clean_error"
        assert "set_state" in arm.call_order(), "worker must attempt a stop"


def _test_sticky_before_stop() -> None:
    # The sticky fault is written before stop_controller issues set_state(4).
    sh = _shared()
    arm = _ProbeArm(sh, state=4, mode=6, error_code=24)
    latch_arm_fault(sh, arm, "boom", controller_error=24)
    assert arm.sticky_before_stop is True, "sticky fault must precede the stop"


def _test_sticky_survives_failed_stop() -> None:
    # Even when the stop cannot be confirmed, the sticky fault is not lost.
    sh = _shared()
    arm = _ProbeArm(sh, state=0, mode=6, error_code=24)
    arm.fail("set_state", 1)  # set_state reports failure, state stays 0
    latch_arm_fault(sh, arm, "boom", controller_error=24)
    assert sh.error_state.value is True, "sticky fault must survive a failed stop"


def _test_setter_failure_while_error_zero() -> None:
    # A non-zero setter API code with a zero live controller error still faults.
    sh = _shared()
    arm = _ProbeArm(sh, state=4, mode=6, error_code=0)
    latch_arm_fault(sh, arm, "setter failed", api_code=5, controller_error=0)
    assert sh.error_state.value is True


def _test_live_read_fails_closed() -> None:
    arm = FakeArm(error_code=0)
    arm.fail("get_err_warn_code", -1)
    try:
        _read_live_error_code(arm)
    except RuntimeError:
        pass
    else:
        raise AssertionError("_read_live_error_code must raise on a failed read")


def _test_config_fields_removed() -> None:
    for cls in (ArmParams, ArmLoopConfig):
        assert not hasattr(cls, "recoverable_errors"), cls
        assert not hasattr(cls, "collision_fault_errors"), cls


def main() -> int:
    _test_single_fault_path()
    _test_sticky_before_stop()
    _test_sticky_survives_failed_stop()
    _test_setter_failure_while_error_zero()
    _test_live_read_fails_closed()
    _test_config_fields_removed()
    print("check_arm_fault_path: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
