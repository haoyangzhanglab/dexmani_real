"""A8: arm worker disconnect is exception-safe during best-effort cleanup.

``_disconnect_arm`` is the single teardown helper invoked from the arm loop's
``finally`` block.  A vendor ``disconnect()`` that raises must be swallowed
(and logged), never propagated into the process-exit path.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeArm

from dexmani_real.robot.arm_loop import _disconnect_arm


def main() -> int:
    # Normal disconnect: no exception, exactly one call.
    arm = FakeArm()
    _disconnect_arm(arm)
    assert arm.call_order() == ["disconnect"], arm.call_order()

    # Failing disconnect: swallowed, no exception escapes the helper.
    arm_bad = FakeArm()
    arm_bad.raise_on("disconnect", RuntimeError("vendor disconnect refused"))
    _disconnect_arm(arm_bad)  # must not raise
    assert arm_bad.call_order() == ["disconnect"], arm_bad.call_order()

    print("check_worker_cleanup: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
